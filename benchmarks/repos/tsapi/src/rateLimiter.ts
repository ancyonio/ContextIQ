/**
 * Per-caller request throttling.
 *
 * The algorithm is a sliding window built from two fixed buckets: the current
 * window's count plus a weighted share of the previous window's. That gives
 * far smoother behaviour at a window boundary than a naive fixed counter,
 * without keeping a timestamp per request the way a log-based limiter does.
 */

import type { RateLimitConfig } from "./config";
import type { RequestContext, Middleware, NextFunction } from "./router";

export const RETRY_AFTER_HEADER = "retry-after";
export const LIMIT_HEADER = "x-ratelimit-limit";
export const REMAINING_HEADER = "x-ratelimit-remaining";

/** Buckets untouched for this long are dropped by the sweeper. */
export const BUCKET_TTL_MS = 10 * 60 * 1000;

export interface Bucket {
  windowStart: number;
  currentCount: number;
  previousCount: number;
}

export interface LimitDecision {
  allowed: boolean;
  remaining: number;
  retryAfterSeconds: number;
  weightedCount: number;
}

export class TooManyRequests extends Error {
  public readonly status = 429;
  public readonly retryAfterSeconds: number;

  constructor(retryAfterSeconds: number) {
    super(`rate limit exceeded, retry in ${retryAfterSeconds}s`);
    this.name = "TooManyRequests";
    this.retryAfterSeconds = retryAfterSeconds;
  }
}

/**
 * Work out who is being limited.
 *
 * An authenticated caller is keyed by principal so that a shared office NAT
 * does not throttle everyone at once; anonymous traffic falls back to the
 * client address, honouring the proxy header only when we actually sit behind
 * a load balancer we trust.
 */
export function resolveClientKey(ctx: RequestContext, trustProxy: boolean): string {
  if (ctx.principal) {
    return `user:${ctx.principal.id}`;
  }
  if (trustProxy) {
    const forwarded = ctx.headers["x-forwarded-for"];
    if (forwarded) {
      return `ip:${forwarded.split(",")[0].trim()}`;
    }
  }
  return `ip:${ctx.remoteAddress ?? "unknown"}`;
}

export class SlidingWindowLimiter {
  private readonly buckets = new Map<string, Bucket>();

  constructor(
    private readonly config: RateLimitConfig,
    private readonly now: () => number = Date.now,
  ) {}

  /**
   * Record a hit for `key` and say whether it may proceed.
   *
   * The previous window is weighted by how far into the current window we
   * are, so the effective count decays smoothly instead of resetting to zero
   * on the tick.
   */
  public hit(key: string): LimitDecision {
    const now = this.now();
    const windowMs = this.config.windowMs;
    const bucket = this.buckets.get(key) ?? {
      windowStart: now,
      currentCount: 0,
      previousCount: 0,
    };
    const elapsed = now - bucket.windowStart;
    if (elapsed >= windowMs * 2) {
      bucket.windowStart = now;
      bucket.previousCount = 0;
      bucket.currentCount = 0;
    } else if (elapsed >= windowMs) {
      bucket.windowStart = bucket.windowStart + windowMs;
      bucket.previousCount = bucket.currentCount;
      bucket.currentCount = 0;
    }
    const progress = (now - bucket.windowStart) / windowMs;
    const weighted = bucket.previousCount * (1 - progress) + bucket.currentCount + 1;
    const allowed = weighted <= this.config.maxRequests;
    if (allowed) {
      bucket.currentCount += 1;
    }
    this.buckets.set(key, bucket);
    const remaining = Math.max(0, Math.floor(this.config.maxRequests - weighted));
    return {
      allowed,
      remaining,
      weightedCount: Math.round(weighted * 100) / 100,
      retryAfterSeconds: allowed
        ? 0
        : Math.max(1, Math.ceil((windowMs - (now - bucket.windowStart)) / 1000)),
    };
  }

  /** Clear one caller's history, used by support tooling and by tests. */
  public reset(key: string): void {
    this.buckets.delete(key);
  }

  /** Drop buckets nobody has touched recently so the map cannot grow forever. */
  public sweep(): number {
    const cutoff = this.now() - BUCKET_TTL_MS;
    let removed = 0;
    for (const [key, bucket] of this.buckets) {
      if (bucket.windowStart < cutoff) {
        this.buckets.delete(key);
        removed += 1;
      }
    }
    return removed;
  }

  public size(): number {
    return this.buckets.size;
  }
}

/**
 * Express-style middleware wrapping {@link SlidingWindowLimiter}.
 *
 * Roles listed in `exemptRoles` bypass the limiter entirely, which is how
 * internal batch jobs avoid throttling themselves.
 */
export function rateLimitMiddleware(
  config: RateLimitConfig,
  limiter = new SlidingWindowLimiter(config),
): Middleware {
  return async (ctx: RequestContext, next: NextFunction) => {
    const roles = ctx.principal?.roles ?? [];
    if (roles.some((role) => config.exemptRoles.includes(role))) {
      await next();
      return;
    }
    const key = resolveClientKey(ctx, config.trustProxy);
    const decision = limiter.hit(key);
    ctx.responseHeaders[LIMIT_HEADER] = String(config.maxRequests);
    ctx.responseHeaders[REMAINING_HEADER] = String(decision.remaining);
    if (!decision.allowed) {
      ctx.responseHeaders[RETRY_AFTER_HEADER] = String(decision.retryAfterSeconds);
      throw new TooManyRequests(decision.retryAfterSeconds);
    }
    await next();
  };
}
