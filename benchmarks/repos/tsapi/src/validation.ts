/**
 * A tiny zod-flavoured schema builder used to validate request payloads.
 *
 * The API deliberately does not pull a validation library into the hot path;
 * the shapes it accepts are simple enough that a few hundred lines of
 * composable validators keep the dependency tree, and the cold-start time,
 * small. Every validator returns a discriminated result rather than throwing,
 * so route handlers can turn failures into a 422 without a try/catch.
 */

export interface ValidationIssue {
  path: string;
  code: string;
  message: string;
}

export type ParseResult<T> =
  | { success: true; data: T }
  | { success: false; issues: ValidationIssue[] };

export const MAX_STRING_LENGTH = 4096;
export const EMAIL_PATTERN = /^[^\s@]+@[^\s@]+\.[^\s@]{2,}$/;

export abstract class Schema<T> {
  public abstract parse(value: unknown, path: string): ParseResult<T>;

  /** Wrap this schema so `undefined` and `null` become the supplied default. */
  public withDefault(fallback: T): Schema<T> {
    const inner = this;
    return new (class extends Schema<T> {
      public parse(value: unknown, path: string): ParseResult<T> {
        if (value === undefined || value === null) {
          return { success: true, data: fallback };
        }
        return inner.parse(value, path);
      }
    })();
  }
}

export class StringSchema extends Schema<string> {
  private minLength = 0;
  private maxLength = MAX_STRING_LENGTH;
  private pattern: RegExp | null = null;

  public min(length: number): this {
    this.minLength = length;
    return this;
  }

  public max(length: number): this {
    this.maxLength = length;
    return this;
  }

  public matching(pattern: RegExp): this {
    this.pattern = pattern;
    return this;
  }

  public email(): this {
    return this.matching(EMAIL_PATTERN);
  }

  public parse(value: unknown, path: string): ParseResult<string> {
    if (typeof value !== "string") {
      return fail(path, "invalid_type", "expected a string");
    }
    if (value.length < this.minLength) {
      return fail(path, "too_short", `must be at least ${this.minLength} characters`);
    }
    if (value.length > this.maxLength) {
      return fail(path, "too_long", `must be at most ${this.maxLength} characters`);
    }
    if (this.pattern && !this.pattern.test(value)) {
      return fail(path, "pattern_mismatch", "does not match the expected format");
    }
    return { success: true, data: value };
  }
}

export class NumberSchema extends Schema<number> {
  private minimum = Number.NEGATIVE_INFINITY;
  private maximum = Number.POSITIVE_INFINITY;
  private integerOnly = false;

  public min(value: number): this {
    this.minimum = value;
    return this;
  }

  public max(value: number): this {
    this.maximum = value;
    return this;
  }

  public int(): this {
    this.integerOnly = true;
    return this;
  }

  public parse(value: unknown, path: string): ParseResult<number> {
    const numeric = typeof value === "string" ? Number(value) : value;
    if (typeof numeric !== "number" || Number.isNaN(numeric)) {
      return fail(path, "invalid_type", "expected a number");
    }
    if (this.integerOnly && !Number.isInteger(numeric)) {
      return fail(path, "not_integer", "must be a whole number");
    }
    if (numeric < this.minimum) {
      return fail(path, "too_small", `must be >= ${this.minimum}`);
    }
    if (numeric > this.maximum) {
      return fail(path, "too_large", `must be <= ${this.maximum}`);
    }
    return { success: true, data: numeric };
  }
}

export class ArraySchema<T> extends Schema<T[]> {
  constructor(
    private readonly item: Schema<T>,
    private readonly maxItems = 100,
  ) {
    super();
  }

  public parse(value: unknown, path: string): ParseResult<T[]> {
    if (!Array.isArray(value)) {
      return fail(path, "invalid_type", "expected an array");
    }
    if (value.length > this.maxItems) {
      return fail(path, "too_many_items", `at most ${this.maxItems} entries`);
    }
    const issues: ValidationIssue[] = [];
    const data: T[] = [];
    value.forEach((entry, index) => {
      const result = this.item.parse(entry, `${path}[${index}]`);
      if (result.success) {
        data.push(result.data);
      } else {
        issues.push(...result.issues);
      }
    });
    return issues.length > 0 ? { success: false, issues } : { success: true, data };
  }
}

export class ObjectSchema<T extends Record<string, unknown>> extends Schema<T> {
  constructor(private readonly shape: { [K in keyof T]: Schema<T[K]> }) {
    super();
  }

  /**
   * Validate every field, collecting all issues instead of stopping at the
   * first one -- a client fixing a form wants the whole list in one round trip.
   */
  public parse(value: unknown, path = "$"): ParseResult<T> {
    if (typeof value !== "object" || value === null || Array.isArray(value)) {
      return fail(path, "invalid_type", "expected an object");
    }
    const source = value as Record<string, unknown>;
    const issues: ValidationIssue[] = [];
    const data: Record<string, unknown> = {};
    for (const key of Object.keys(this.shape) as (keyof T & string)[]) {
      const childPath = path === "$" ? key : `${path}.${key}`;
      const result = this.shape[key].parse(source[key], childPath);
      if (result.success) {
        data[key] = result.data;
      } else {
        issues.push(...result.issues);
      }
    }
    return issues.length > 0
      ? { success: false, issues }
      : { success: true, data: data as T };
  }
}

function fail(path: string, code: string, message: string): ParseResult<never> {
  return { success: false, issues: [{ path, code, message }] };
}

export const str = (): StringSchema => new StringSchema();
export const num = (): NumberSchema => new NumberSchema();
export const arrayOf = <T>(item: Schema<T>, maxItems?: number): ArraySchema<T> =>
  new ArraySchema(item, maxItems);
export const object = <T extends Record<string, unknown>>(shape: {
  [K in keyof T]: Schema<T[K]>;
}): ObjectSchema<T> => new ObjectSchema(shape);

export interface CreateProjectBody extends Record<string, unknown> {
  name: string;
  slug: string;
  memberEmails: string[];
  seatCount: number;
}

/** Schema for `POST /projects`, reused by both the route and its tests. */
export const createProjectSchema = object<CreateProjectBody>({
  name: str().min(2).max(120),
  slug: str().min(2).max(64).matching(/^[a-z0-9-]+$/),
  memberEmails: arrayOf(str().email(), 50),
  seatCount: num().int().min(1).max(500).withDefault(1),
});

/** Turn a failed parse into the JSON body the API returns for a 422. */
export function issuesToResponse(issues: ValidationIssue[]): Record<string, unknown> {
  return {
    error: "validation_failed",
    issues: issues.map((issue) => ({
      path: issue.path,
      code: issue.code,
      message: issue.message,
    })),
  };
}
