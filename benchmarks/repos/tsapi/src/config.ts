/**
 * Environment-driven configuration for the tsapi service.
 *
 * Everything the process can be tuned with is read exactly once, here, and
 * then passed around as a plain object. No other module is allowed to touch
 * `process.env` -- that keeps the deployable surface easy to audit and makes
 * tests able to spin up several configurations in one worker.
 */

export const DEFAULT_PORT = 8080;
export const ACCESS_TOKEN_TTL_SECONDS = 900;
export const REFRESH_TOKEN_TTL_SECONDS = 60 * 60 * 24 * 14;
export const RATE_LIMIT_WINDOW_MS = 60_000;
export const RATE_LIMIT_MAX_REQUESTS = 120;

export interface DatabaseConfig {
  url: string;
  poolMin: number;
  poolMax: number;
  statementTimeoutMs: number;
  idleTimeoutMs: number;
}

export interface AuthConfig {
  jwtSecret: string;
  issuer: string;
  audience: string;
  accessTokenTtl: number;
  refreshTokenTtl: number;
  clockToleranceSeconds: number;
}

export interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  trustProxy: boolean;
  exemptRoles: string[];
}

export interface AppConfig {
  env: "development" | "staging" | "production";
  port: number;
  database: DatabaseConfig;
  auth: AuthConfig;
  rateLimit: RateLimitConfig;
  websocketPath: string;
  logLevel: string;
}

export class ConfigError extends Error {
  public readonly variable: string;

  constructor(variable: string, message: string) {
    super(`${variable}: ${message}`);
    this.name = "ConfigError";
    this.variable = variable;
  }
}

/** Read a required variable or fail loudly at boot rather than at first use. */
export function requireEnv(env: NodeJS.ProcessEnv, key: string): string {
  const value = env[key];
  if (value === undefined || value.trim() === "") {
    throw new ConfigError(key, "is required but was not set");
  }
  return value;
}

/** Read an optional integer, falling back to a documented default. */
export function intEnv(env: NodeJS.ProcessEnv, key: string, fallback: number): number {
  const raw = env[key];
  if (raw === undefined || raw.trim() === "") {
    return fallback;
  }
  const parsed = Number.parseInt(raw, 10);
  if (Number.isNaN(parsed)) {
    throw new ConfigError(key, `expected an integer, received "${raw}"`);
  }
  return parsed;
}

/** Split a comma separated list, dropping empty entries and whitespace. */
export function listEnv(env: NodeJS.ProcessEnv, key: string): string[] {
  const raw = env[key];
  if (!raw) {
    return [];
  }
  return raw
    .split(",")
    .map((entry) => entry.trim())
    .filter((entry) => entry.length > 0);
}

/**
 * Build the whole application configuration from an environment map.
 *
 * `DATABASE_URL` and `JWT_SECRET` are mandatory. Production additionally
 * refuses to start with a short signing secret, because a weak secret is a
 * silent authentication bypass rather than a crash.
 */
export function loadConfig(env: NodeJS.ProcessEnv = process.env): AppConfig {
  const environment = (env.NODE_ENV ?? "development") as AppConfig["env"];
  const jwtSecret = requireEnv(env, "JWT_SECRET");
  if (environment === "production" && jwtSecret.length < 32) {
    throw new ConfigError("JWT_SECRET", "must be at least 32 characters in production");
  }
  return {
    env: environment,
    port: intEnv(env, "PORT", DEFAULT_PORT),
    database: {
      url: requireEnv(env, "DATABASE_URL"),
      poolMin: intEnv(env, "DB_POOL_MIN", 2),
      poolMax: intEnv(env, "DB_POOL_MAX", 10),
      statementTimeoutMs: intEnv(env, "DB_STATEMENT_TIMEOUT_MS", 5000),
      idleTimeoutMs: intEnv(env, "DB_IDLE_TIMEOUT_MS", 30_000),
    },
    auth: {
      jwtSecret,
      issuer: env.JWT_ISSUER ?? "tsapi",
      audience: env.JWT_AUDIENCE ?? "tsapi-clients",
      accessTokenTtl: intEnv(env, "ACCESS_TOKEN_TTL", ACCESS_TOKEN_TTL_SECONDS),
      refreshTokenTtl: intEnv(env, "REFRESH_TOKEN_TTL", REFRESH_TOKEN_TTL_SECONDS),
      clockToleranceSeconds: intEnv(env, "JWT_CLOCK_TOLERANCE", 30),
    },
    rateLimit: {
      windowMs: intEnv(env, "RATE_LIMIT_WINDOW_MS", RATE_LIMIT_WINDOW_MS),
      maxRequests: intEnv(env, "RATE_LIMIT_MAX", RATE_LIMIT_MAX_REQUESTS),
      trustProxy: env.TRUST_PROXY === "true",
      exemptRoles: listEnv(env, "RATE_LIMIT_EXEMPT_ROLES"),
    },
    websocketPath: env.WEBSOCKET_PATH ?? "/ws",
    logLevel: env.LOG_LEVEL ?? "info",
  };
}

/** Config summary safe to write to logs: secrets are replaced with a marker. */
export function describeConfig(config: AppConfig): Record<string, unknown> {
  return {
    env: config.env,
    port: config.port,
    databaseHost: config.database.url.replace(/\/\/[^@]*@/, "//***@"),
    poolMax: config.database.poolMax,
    rateLimit: `${config.rateLimit.maxRequests}/${config.rateLimit.windowMs}ms`,
    websocketPath: config.websocketPath,
  };
}
