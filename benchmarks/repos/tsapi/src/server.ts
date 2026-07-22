/**
 * Process entry point: build dependencies, listen, and shut down cleanly.
 *
 * The HTTP surface is deliberately thin -- reading the body, generating a
 * request id and funnelling errors are the only things that happen here.
 */

import { randomUUID } from "node:crypto";
import { createServer, type IncomingMessage, type ServerResponse, type Server } from "node:http";
import { loadConfig, describeConfig, type AppConfig } from "./config";
import { ProjectRepository, type Pool } from "./db";
import { NotificationHub } from "./notifications";
import { buildRouter } from "./routes";
import { toErrorResponse, type RequestContext, type Router } from "./router";

export const SHUTDOWN_GRACE_MS = 10_000;
export const MAX_BODY_BYTES = 1_048_576;

export class PayloadTooLarge extends Error {
  public readonly status = 413;
  public readonly code = "payload_too_large";

  constructor() {
    super(`request body may not exceed ${MAX_BODY_BYTES} bytes`);
    this.name = "PayloadTooLarge";
  }
}

/**
 * Read the whole request body, refusing anything oversized.
 *
 * The limit is enforced while streaming rather than after buffering, so a
 * hostile client cannot make the process allocate a gigabyte before we say no.
 */
export async function readJsonBody(req: IncomingMessage): Promise<unknown> {
  const chunks: Buffer[] = [];
  let total = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    total += buffer.length;
    if (total > MAX_BODY_BYTES) {
      throw new PayloadTooLarge();
    }
    chunks.push(buffer);
  }
  if (total === 0) {
    return undefined;
  }
  const text = Buffer.concat(chunks).toString("utf8");
  try {
    return JSON.parse(text);
  } catch {
    throw new Error("request body is not valid JSON");
  }
}

/** Turn a Node request into the framework-neutral context the router wants. */
export async function toRequestContext(req: IncomingMessage): Promise<RequestContext> {
  const url = new URL(req.url ?? "/", "http://localhost");
  const query: Record<string, string> = {};
  url.searchParams.forEach((value, key) => {
    query[key] = value;
  });
  return {
    method: req.method ?? "GET",
    path: url.pathname,
    params: {},
    query,
    headers: req.headers as Record<string, string | undefined>,
    body: await readJsonBody(req),
    remoteAddress: req.socket.remoteAddress ?? undefined,
    responseHeaders: {},
    requestId: randomUUID(),
  };
}

/** Adapter that runs one request through the router and writes the response. */
export function createRequestListener(router: Router) {
  return async (req: IncomingMessage, res: ServerResponse): Promise<void> => {
    let requestId = "unknown";
    try {
      const ctx = await toRequestContext(req);
      requestId = ctx.requestId;
      const response = await router.handle(ctx);
      res.writeHead(response.status, {
        ...response.headers,
        "x-request-id": requestId,
      });
      res.end(JSON.stringify(response.body));
    } catch (error) {
      const response = toErrorResponse(error, requestId);
      res.writeHead(response.status, response.headers);
      res.end(JSON.stringify(response.body));
    }
  };
}

export interface RunningServer {
  server: Server;
  hub: NotificationHub;
  close(): Promise<void>;
}

/**
 * Compose everything and start listening.
 *
 * Shutdown stops accepting new sockets, closes the notification hub so
 * subscribers get a proper close frame, then waits a bounded grace period for
 * in-flight requests before forcing exit.
 */
export function start(pool: Pool, config: AppConfig = loadConfig()): RunningServer {
  const projects = new ProjectRepository(pool, config.database);
  const hub = new NotificationHub(config.auth);
  hub.start();
  const router = buildRouter({ config, projects, hub });
  const server = createServer(createRequestListener(router));
  server.listen(config.port, () => {
    // eslint-disable-next-line no-console
    console.log(JSON.stringify({ msg: "listening", ...describeConfig(config) }));
  });

  const close = async (): Promise<void> => {
    hub.stop();
    await new Promise<void>((resolve) => {
      const timer = setTimeout(resolve, SHUTDOWN_GRACE_MS);
      server.close(() => {
        clearTimeout(timer);
        resolve();
      });
    });
    await pool.end();
  };

  process.once("SIGTERM", () => {
    void close();
  });

  return { server, hub, close };
}
