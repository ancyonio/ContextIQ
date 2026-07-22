/**
 * Route table: wires validation, auth, throttling and persistence together.
 *
 * Handlers here stay thin -- parse, delegate, shape the response. Anything
 * with a policy in it belongs in the module that owns that policy.
 */

import type { AppConfig } from "./config";
import { requireAuth, requireRole, issueTokenPair } from "./auth";
import { rateLimitMiddleware, SlidingWindowLimiter } from "./rateLimiter";
import { ProjectRepository, clampPageSize } from "./db";
import { NotificationHub } from "./notifications";
import { Router, json, type RequestContext, type ApiResponse } from "./router";
import {
  createProjectSchema,
  issuesToResponse,
  type CreateProjectBody,
} from "./validation";

export interface Dependencies {
  config: AppConfig;
  projects: ProjectRepository;
  hub: NotificationHub;
  limiter?: SlidingWindowLimiter;
}

/** Request-scoped logging so every line can be tied back to one call. */
export function requestLogger(): (ctx: RequestContext, next: () => Promise<void>) => Promise<void> {
  return async (ctx, next) => {
    const startedAt = Date.now();
    try {
      await next();
    } finally {
      const elapsed = Date.now() - startedAt;
      // eslint-disable-next-line no-console
      console.log(
        JSON.stringify({
          requestId: ctx.requestId,
          method: ctx.method,
          path: ctx.path,
          principal: ctx.principal?.id ?? "anonymous",
          durationMs: elapsed,
        }),
      );
    }
  };
}

/** POST /auth/token -- exchange credentials for an access/refresh pair. */
export async function handleIssueToken(
  ctx: RequestContext,
  config: AppConfig,
): Promise<ApiResponse> {
  const body = (ctx.body ?? {}) as { userId?: string; roles?: string[] };
  if (!body.userId) {
    return json(422, { error: "validation_failed", issues: [{ path: "userId", code: "required", message: "userId is required" }] });
  }
  const tokens = issueTokenPair(body.userId, body.roles ?? ["member"], config.auth);
  return json(200, {
    accessToken: tokens.accessToken,
    refreshToken: tokens.refreshToken,
    expiresIn: tokens.expiresIn,
    tokenType: "Bearer",
  });
}

/** POST /projects -- validate the payload, insert, then notify watchers. */
export async function handleCreateProject(
  ctx: RequestContext,
  projects: ProjectRepository,
  hub: NotificationHub,
): Promise<ApiResponse> {
  const parsed = createProjectSchema.parse(ctx.body);
  if (!parsed.success) {
    return json(422, issuesToResponse(parsed.issues));
  }
  const input = parsed.data as CreateProjectBody;
  const project = await projects.create(
    {
      name: input.name,
      slug: input.slug,
      ownerId: ctx.principal?.id ?? "unknown",
      seatCount: input.seatCount,
    },
    input.memberEmails,
  );
  hub.broadcast(`user:${project.ownerId}`, "project.created", {
    projectId: project.id,
    slug: project.slug,
  });
  return json(201, project);
}

/** GET /projects -- keyset-paginated list scoped to the caller. */
export async function handleListProjects(
  ctx: RequestContext,
  projects: ProjectRepository,
): Promise<ApiResponse> {
  const pageSize = clampPageSize(Number(ctx.query.pageSize));
  const after = ctx.query.after ?? null;
  const rows = await projects.listForOwner(ctx.principal?.id ?? "", after, pageSize);
  const nextCursor = rows.length === pageSize ? rows[rows.length - 1].id : null;
  return json(200, { data: rows, nextCursor, pageSize });
}

/** GET /projects/:projectId -- single record, 404 when it is not there. */
export async function handleGetProject(
  ctx: RequestContext,
  projects: ProjectRepository,
): Promise<ApiResponse> {
  const project = await projects.findById(ctx.params.projectId);
  return json(200, project);
}

/** DELETE /projects/:projectId -- archive and tell subscribers. */
export async function handleArchiveProject(
  ctx: RequestContext,
  projects: ProjectRepository,
  hub: NotificationHub,
): Promise<ApiResponse> {
  const project = await projects.archive(ctx.params.projectId);
  hub.broadcast(`project:${project.id}`, "project.archived", { projectId: project.id });
  return json(200, project);
}

/** GET /healthz -- probes the database and reports hub occupancy. */
export async function handleHealth(
  projects: ProjectRepository,
  hub: NotificationHub,
): Promise<ApiResponse> {
  const databaseUp = await projects.ping().catch(() => false);
  return json(databaseUp ? 200 : 503, {
    status: databaseUp ? "ok" : "degraded",
    database: databaseUp,
    sockets: hub.stats(),
  });
}

/**
 * Build the router.
 *
 * Ordering matters: logging wraps everything so failures are still recorded,
 * the limiter runs before auth on public routes, and role checks are attached
 * per-route rather than globally.
 */
export function buildRouter(deps: Dependencies): Router {
  const { config, projects, hub } = deps;
  const limiter = deps.limiter ?? new SlidingWindowLimiter(config.rateLimit);
  const router = new Router();

  router.use(requestLogger());
  router.use(rateLimitMiddleware(config.rateLimit, limiter));

  router.post("/auth/token", (ctx) => handleIssueToken(ctx, config));
  router.get("/healthz", () => handleHealth(projects, hub));

  const authenticated = requireAuth(config.auth);
  router.post(
    "/projects",
    (ctx) => handleCreateProject(ctx, projects, hub),
    authenticated,
  );
  router.get("/projects", (ctx) => handleListProjects(ctx, projects), authenticated);
  router.get(
    "/projects/:projectId",
    (ctx) => handleGetProject(ctx, projects),
    authenticated,
  );
  router.delete(
    "/projects/:projectId",
    (ctx) => handleArchiveProject(ctx, projects, hub),
    authenticated,
    requireRole("owner", "admin"),
  );

  return router;
}
