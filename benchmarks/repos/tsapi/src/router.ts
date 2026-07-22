/**
 * A minimal Express-shaped router: path patterns, middleware chains and a
 * single error funnel.
 *
 * Routes are matched by splitting on `/` rather than by regex so that a
 * pathological pattern cannot cause catastrophic backtracking, and so that
 * `:param` segments are captured without a dependency.
 */

export interface Principal {
  id: string;
  roles: string[];
}

export interface RequestContext {
  method: string;
  path: string;
  params: Record<string, string>;
  query: Record<string, string>;
  headers: Record<string, string | undefined>;
  body: unknown;
  remoteAddress?: string;
  principal?: Principal;
  responseHeaders: Record<string, string>;
  requestId: string;
}

export interface ApiResponse {
  status: number;
  body: unknown;
  headers: Record<string, string>;
}

export type NextFunction = () => Promise<void>;
export type Middleware = (ctx: RequestContext, next: NextFunction) => Promise<void>;
export type Handler = (ctx: RequestContext) => Promise<ApiResponse> | ApiResponse;

interface Route {
  method: string;
  segments: string[];
  handler: Handler;
  middleware: Middleware[];
}

export class NotFound extends Error {
  public readonly status = 404;

  constructor(method: string, path: string) {
    super(`no route for ${method} ${path}`);
    this.name = "NotFound";
  }
}

/** Split a path into segments, ignoring leading, trailing and doubled slashes. */
export function splitPath(path: string): string[] {
  return path.split("/").filter((segment) => segment.length > 0);
}

/**
 * Match a request path against a route pattern.
 *
 * Returns the captured `:param` values, or `null` when the pattern does not
 * apply. A trailing `*` segment swallows the remainder, which is what the
 * static asset route relies on.
 */
export function matchSegments(
  pattern: string[],
  actual: string[],
): Record<string, string> | null {
  const params: Record<string, string> = {};
  for (let i = 0; i < pattern.length; i += 1) {
    const expected = pattern[i];
    if (expected === "*") {
      params.wildcard = actual.slice(i).join("/");
      return params;
    }
    if (i >= actual.length) {
      return null;
    }
    if (expected.startsWith(":")) {
      params[expected.slice(1)] = decodeURIComponent(actual[i]);
      continue;
    }
    if (expected !== actual[i]) {
      return null;
    }
  }
  return pattern.length === actual.length ? params : null;
}

export class Router {
  private readonly routes: Route[] = [];
  private readonly globalMiddleware: Middleware[] = [];

  /** Register middleware that runs for every request, in registration order. */
  public use(middleware: Middleware): this {
    this.globalMiddleware.push(middleware);
    return this;
  }

  public get(path: string, handler: Handler, ...middleware: Middleware[]): this {
    return this.register("GET", path, handler, middleware);
  }

  public post(path: string, handler: Handler, ...middleware: Middleware[]): this {
    return this.register("POST", path, handler, middleware);
  }

  public patch(path: string, handler: Handler, ...middleware: Middleware[]): this {
    return this.register("PATCH", path, handler, middleware);
  }

  public delete(path: string, handler: Handler, ...middleware: Middleware[]): this {
    return this.register("DELETE", path, handler, middleware);
  }

  private register(
    method: string,
    path: string,
    handler: Handler,
    middleware: Middleware[],
  ): this {
    this.routes.push({
      method,
      segments: splitPath(path),
      handler,
      middleware,
    });
    return this;
  }

  /** Find the first route whose method and pattern accept this request. */
  public resolve(method: string, path: string): { route: Route; params: Record<string, string> } | null {
    const actual = splitPath(path);
    for (const route of this.routes) {
      if (route.method !== method) {
        continue;
      }
      const params = matchSegments(route.segments, actual);
      if (params) {
        return { route, params };
      }
    }
    return null;
  }

  /**
   * Run the middleware chain and then the handler.
   *
   * The chain is built as a linked series of closures so a middleware that
   * never calls `next()` short-circuits everything after it, exactly like
   * Express. Calling `next()` twice is a programming error and throws.
   */
  public async handle(ctx: RequestContext): Promise<ApiResponse> {
    const resolved = this.resolve(ctx.method, ctx.path);
    if (!resolved) {
      throw new NotFound(ctx.method, ctx.path);
    }
    ctx.params = resolved.params;
    const chain = [...this.globalMiddleware, ...resolved.route.middleware];
    let response: ApiResponse | null = null;
    let index = -1;

    const dispatch = async (position: number): Promise<void> => {
      if (position <= index) {
        throw new Error("next() called more than once in a middleware");
      }
      index = position;
      if (position === chain.length) {
        response = await resolved.route.handler(ctx);
        return;
      }
      await chain[position](ctx, () => dispatch(position + 1));
    };

    await dispatch(0);
    if (!response) {
      throw new Error("handler produced no response");
    }
    return {
      ...response,
      headers: { ...ctx.responseHeaders, ...response.headers },
    };
  }
}

/** Convert any thrown value into the JSON error envelope clients expect. */
export function toErrorResponse(error: unknown, requestId: string): ApiResponse {
  const status =
    typeof error === "object" && error !== null && "status" in error
      ? Number((error as { status: unknown }).status) || 500
      : 500;
  const code =
    typeof error === "object" && error !== null && "code" in error
      ? String((error as { code: unknown }).code)
      : "internal_error";
  const message = error instanceof Error ? error.message : "unexpected failure";
  return {
    status,
    body: {
      error: { code, message, requestId },
    },
    headers: { "content-type": "application/json" },
  };
}

/** Helper used by handlers so they never build the response shape by hand. */
export function json(status: number, body: unknown): ApiResponse {
  return { status, body, headers: { "content-type": "application/json" } };
}
