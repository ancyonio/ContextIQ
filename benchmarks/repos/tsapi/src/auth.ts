/**
 * Bearer-token authentication middleware and the token helpers behind it.
 *
 * The service issues short-lived access tokens and long-lived refresh tokens;
 * both are signed with the same HMAC secret but carry a different `typ` claim
 * so a refresh token can never be replayed as an access token.
 */

import { createHmac, timingSafeEqual } from "node:crypto";
import type { AuthConfig } from "./config";
import type { RequestContext, Middleware, NextFunction } from "./router";

export const AUTH_HEADER = "authorization";
export const BEARER_PREFIX = "Bearer ";

export interface TokenClaims {
  sub: string;
  roles: string[];
  iss: string;
  aud: string;
  typ: "access" | "refresh";
  iat: number;
  exp: number;
  jti?: string;
}

export class AuthError extends Error {
  public readonly status: number;
  public readonly code: string;

  constructor(code: string, message: string, status = 401) {
    super(message);
    this.name = "AuthError";
    this.code = code;
    this.status = status;
  }
}

function base64UrlEncode(input: Buffer | string): string {
  const buffer = typeof input === "string" ? Buffer.from(input, "utf8") : input;
  return buffer
    .toString("base64")
    .replace(/\+/g, "-")
    .replace(/\//g, "_")
    .replace(/=+$/, "");
}

function base64UrlDecode(input: string): Buffer {
  const padded = input.replace(/-/g, "+").replace(/_/g, "/");
  const remainder = padded.length % 4;
  return Buffer.from(remainder ? padded + "=".repeat(4 - remainder) : padded, "base64");
}

/**
 * Compare two signatures without leaking how many bytes matched.
 *
 * `timingSafeEqual` throws on mismatched lengths, so the length check happens
 * first and returns the same answer a constant-time compare would.
 */
export function signaturesMatch(expected: string, provided: string): boolean {
  const a = Buffer.from(expected, "utf8");
  const b = Buffer.from(provided, "utf8");
  if (a.length !== b.length) {
    return false;
  }
  return timingSafeEqual(a, b);
}

/** Sign a payload with HMAC-SHA256 and return the compact serialisation. */
export function signToken(claims: TokenClaims, secret: string): string {
  const header = base64UrlEncode(JSON.stringify({ alg: "HS256", typ: "JWT" }));
  const body = base64UrlEncode(JSON.stringify(claims));
  const signature = createHmac("sha256", secret)
    .update(`${header}.${body}`)
    .digest();
  return `${header}.${body}.${base64UrlEncode(signature)}`;
}

/**
 * Verify signature, issuer, audience and expiry, then hand back the claims.
 *
 * A small clock tolerance is allowed because the API and the identity service
 * sit on different hosts and NTP drift of a second or two is normal.
 */
export function verifyToken(token: string, config: AuthConfig): TokenClaims {
  const parts = token.split(".");
  if (parts.length !== 3) {
    throw new AuthError("malformed_token", "token must have three segments");
  }
  const [header, body, signature] = parts;
  const expected = base64UrlEncode(
    createHmac("sha256", config.jwtSecret).update(`${header}.${body}`).digest(),
  );
  if (!signaturesMatch(expected, signature)) {
    throw new AuthError("bad_signature", "token signature does not verify");
  }
  let claims: TokenClaims;
  try {
    claims = JSON.parse(base64UrlDecode(body).toString("utf8")) as TokenClaims;
  } catch {
    throw new AuthError("malformed_token", "token payload is not valid JSON");
  }
  const now = Math.floor(Date.now() / 1000);
  if (claims.exp + config.clockToleranceSeconds < now) {
    throw new AuthError("token_expired", "token has expired");
  }
  if (claims.iss !== config.issuer) {
    throw new AuthError("bad_issuer", `unexpected issuer ${claims.iss}`);
  }
  if (claims.aud !== config.audience) {
    throw new AuthError("bad_audience", `unexpected audience ${claims.aud}`);
  }
  return claims;
}

/** Mint a pair of tokens for a freshly authenticated principal. */
export function issueTokenPair(
  subject: string,
  roles: string[],
  config: AuthConfig,
): { accessToken: string; refreshToken: string; expiresIn: number } {
  const now = Math.floor(Date.now() / 1000);
  const base = { sub: subject, roles, iss: config.issuer, aud: config.audience, iat: now };
  const accessToken = signToken(
    { ...base, typ: "access", exp: now + config.accessTokenTtl },
    config.jwtSecret,
  );
  const refreshToken = signToken(
    { ...base, typ: "refresh", exp: now + config.refreshTokenTtl },
    config.jwtSecret,
  );
  return { accessToken, refreshToken, expiresIn: config.accessTokenTtl };
}

/** Pull the credential out of the `Authorization` header, if there is one. */
export function extractBearer(headers: Record<string, string | undefined>): string | null {
  const raw = headers[AUTH_HEADER] ?? headers[AUTH_HEADER.toUpperCase()];
  if (!raw || !raw.startsWith(BEARER_PREFIX)) {
    return null;
  }
  const token = raw.slice(BEARER_PREFIX.length).trim();
  return token.length > 0 ? token : null;
}

/**
 * Middleware that rejects anonymous traffic and attaches the caller identity.
 *
 * Anything the route later needs to know about who is calling comes from
 * `ctx.principal`; handlers never re-parse the header themselves.
 */
export function requireAuth(config: AuthConfig): Middleware {
  return async (ctx: RequestContext, next: NextFunction) => {
    const token = extractBearer(ctx.headers);
    if (!token) {
      throw new AuthError("missing_token", "an Authorization header is required");
    }
    const claims = verifyToken(token, config);
    if (claims.typ !== "access") {
      throw new AuthError("wrong_token_type", "refresh tokens cannot call the API");
    }
    ctx.principal = { id: claims.sub, roles: claims.roles };
    await next();
  };
}

/** Middleware factory that gates a route behind one of several roles. */
export function requireRole(...allowed: string[]): Middleware {
  return async (ctx: RequestContext, next: NextFunction) => {
    const roles = ctx.principal?.roles ?? [];
    if (!roles.some((role) => allowed.includes(role))) {
      throw new AuthError(
        "forbidden",
        `requires one of: ${allowed.join(", ")}`,
        403,
      );
    }
    await next();
  };
}
