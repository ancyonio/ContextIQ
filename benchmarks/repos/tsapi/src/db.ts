/**
 * Postgres data access for projects and their members.
 *
 * Everything goes through parameterised queries; string interpolation into
 * SQL is forbidden in this file and the review checklist calls it out. The
 * pool is created once and injected, so tests can pass a fake with the same
 * three methods.
 */

import type { DatabaseConfig } from "./config";

export interface QueryResult<T> {
  rows: T[];
  rowCount: number;
}

export interface PoolClient {
  query<T>(text: string, values?: unknown[]): Promise<QueryResult<T>>;
  release(): void;
}

export interface Pool {
  query<T>(text: string, values?: unknown[]): Promise<QueryResult<T>>;
  connect(): Promise<PoolClient>;
  end(): Promise<void>;
}

export interface ProjectRow {
  id: string;
  name: string;
  slug: string;
  owner_id: string;
  seat_count: number;
  archived_at: string | null;
  created_at: string;
}

export interface Project {
  id: string;
  name: string;
  slug: string;
  ownerId: string;
  seatCount: number;
  archived: boolean;
  createdAt: string;
}

export class NotFoundError extends Error {
  public readonly status = 404;

  constructor(entity: string, id: string) {
    super(`${entity} ${id} does not exist`);
    this.name = "NotFoundError";
  }
}

export class DuplicateSlug extends Error {
  public readonly status = 409;
  public readonly code = "duplicate_slug";

  constructor(slug: string) {
    super(`the slug "${slug}" is already taken`);
    this.name = "DuplicateSlug";
  }
}

export const UNIQUE_VIOLATION = "23505";
export const DEFAULT_PAGE_SIZE = 25;
export const MAX_PAGE_SIZE = 100;

/** Map a snake_case database row onto the camelCase shape the API returns. */
export function toProject(row: ProjectRow): Project {
  return {
    id: row.id,
    name: row.name,
    slug: row.slug,
    ownerId: row.owner_id,
    seatCount: Number(row.seat_count),
    archived: row.archived_at !== null,
    createdAt: row.created_at,
  };
}

/**
 * Clamp a client-supplied page size so one caller cannot ask for the table.
 */
export function clampPageSize(requested: number | undefined): number {
  if (!requested || Number.isNaN(requested)) {
    return DEFAULT_PAGE_SIZE;
  }
  return Math.min(Math.max(1, Math.floor(requested)), MAX_PAGE_SIZE);
}

export class ProjectRepository {
  constructor(
    private readonly pool: Pool,
    private readonly config: DatabaseConfig,
  ) {}

  /**
   * Run a callback inside a transaction, committing on success and rolling
   * back on any throw. The client is always returned to the pool.
   */
  public async withTransaction<T>(work: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query("BEGIN");
      await client.query("SET LOCAL statement_timeout = $1", [
        this.config.statementTimeoutMs,
      ]);
      const result = await work(client);
      await client.query("COMMIT");
      return result;
    } catch (error) {
      await client.query("ROLLBACK");
      throw error;
    } finally {
      client.release();
    }
  }

  public async findById(id: string): Promise<Project> {
    const result = await this.pool.query<ProjectRow>(
      "SELECT * FROM projects WHERE id = $1 AND deleted_at IS NULL",
      [id],
    );
    const row = result.rows[0];
    if (!row) {
      throw new NotFoundError("project", id);
    }
    return toProject(row);
  }

  /**
   * Keyset pagination: callers pass the last id they saw rather than an
   * offset, so deep pages stay cheap and rows cannot be skipped when a new
   * project is inserted mid-scroll.
   */
  public async listForOwner(
    ownerId: string,
    afterId: string | null,
    pageSize: number,
  ): Promise<Project[]> {
    const limit = clampPageSize(pageSize);
    const result = afterId
      ? await this.pool.query<ProjectRow>(
          "SELECT * FROM projects WHERE owner_id = $1 AND id > $2 " +
            "AND deleted_at IS NULL ORDER BY id ASC LIMIT $3",
          [ownerId, afterId, limit],
        )
      : await this.pool.query<ProjectRow>(
          "SELECT * FROM projects WHERE owner_id = $1 AND deleted_at IS NULL " +
            "ORDER BY id ASC LIMIT $2",
          [ownerId, limit],
        );
    return result.rows.map(toProject);
  }

  /**
   * Insert a project and its members atomically. A unique index collision on
   * the slug is translated into a domain error rather than leaking the driver
   * error code to the client.
   */
  public async create(
    input: { name: string; slug: string; ownerId: string; seatCount: number },
    memberEmails: string[],
  ): Promise<Project> {
    return this.withTransaction(async (client) => {
      let inserted: QueryResult<ProjectRow>;
      try {
        inserted = await client.query<ProjectRow>(
          "INSERT INTO projects (name, slug, owner_id, seat_count) " +
            "VALUES ($1, $2, $3, $4) RETURNING *",
          [input.name, input.slug, input.ownerId, input.seatCount],
        );
      } catch (error) {
        if ((error as { code?: string }).code === UNIQUE_VIOLATION) {
          throw new DuplicateSlug(input.slug);
        }
        throw error;
      }
      const project = toProject(inserted.rows[0]);
      for (const email of memberEmails) {
        await client.query(
          "INSERT INTO project_members (project_id, email, role) " +
            "VALUES ($1, $2, 'member') ON CONFLICT DO NOTHING",
          [project.id, email],
        );
      }
      return project;
    });
  }

  /** Soft delete so audit history and billing records stay intact. */
  public async archive(id: string): Promise<Project> {
    const result = await this.pool.query<ProjectRow>(
      "UPDATE projects SET archived_at = now() WHERE id = $1 " +
        "AND archived_at IS NULL RETURNING *",
      [id],
    );
    if (result.rowCount === 0) {
      throw new NotFoundError("project", id);
    }
    return toProject(result.rows[0]);
  }

  /** Cheap liveness probe that also proves the pool can hand out a client. */
  public async ping(): Promise<boolean> {
    const result = await this.pool.query<{ ok: number }>("SELECT 1 AS ok");
    return result.rows.length === 1;
  }
}
