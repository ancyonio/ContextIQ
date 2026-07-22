/**
 * WebSocket fan-out hub.
 *
 * Clients open a socket, authenticate with the same access token the REST API
 * uses, then subscribe to topics. The hub keeps a topic -> sockets index so a
 * broadcast is O(subscribers) rather than O(all connections), and drops
 * sockets that stop answering heartbeats.
 */

import type { AuthConfig } from "./config";
import { verifyToken, AuthError } from "./auth";

export const HEARTBEAT_INTERVAL_MS = 30_000;
export const MISSED_PONGS_BEFORE_DROP = 2;
export const MAX_TOPICS_PER_SOCKET = 20;

export type SocketState = "connecting" | "open" | "closing" | "closed";

export interface Socket {
  id: string;
  state: SocketState;
  send(payload: string): void;
  close(code: number, reason: string): void;
}

export interface Envelope {
  topic: string;
  event: string;
  payload: unknown;
  emittedAt: string;
}

interface Connection {
  socket: Socket;
  principalId: string;
  roles: string[];
  topics: Set<string>;
  missedPongs: number;
  connectedAt: number;
}

export class SubscriptionError extends Error {
  public readonly code: string;

  constructor(code: string, message: string) {
    super(message);
    this.name = "SubscriptionError";
    this.code = code;
  }
}

/** Build the JSON frame that goes over the wire for one event. */
export function buildEnvelope(topic: string, event: string, payload: unknown): Envelope {
  return { topic, event, payload, emittedAt: new Date().toISOString() };
}

/**
 * Topics are `entity:id` pairs; a caller may only subscribe to a topic whose
 * entity they are allowed to read. Admins are exempt.
 */
export function canSubscribe(topic: string, principalId: string, roles: string[]): boolean {
  if (roles.includes("admin")) {
    return true;
  }
  const [entity, id] = topic.split(":");
  if (!entity || !id) {
    return false;
  }
  if (entity === "user") {
    return id === principalId;
  }
  return entity === "project" || entity === "system";
}

export class NotificationHub {
  private readonly connections = new Map<string, Connection>();
  private readonly topicIndex = new Map<string, Set<string>>();
  private heartbeat: ReturnType<typeof setInterval> | null = null;

  constructor(private readonly authConfig: AuthConfig) {}

  /**
   * Accept a new socket after checking its token.
   *
   * The token is passed as a query parameter because browsers cannot set
   * headers on a WebSocket handshake; it is verified with exactly the same
   * routine the REST middleware uses so the two cannot drift apart.
   */
  public accept(socket: Socket, token: string): Connection {
    const claims = verifyToken(token, this.authConfig);
    if (claims.typ !== "access") {
      throw new AuthError("wrong_token_type", "socket requires an access token");
    }
    const connection: Connection = {
      socket,
      principalId: claims.sub,
      roles: claims.roles,
      topics: new Set<string>(),
      missedPongs: 0,
      connectedAt: Date.now(),
    };
    this.connections.set(socket.id, connection);
    socket.send(JSON.stringify(buildEnvelope("system:hello", "connected", {
      socketId: socket.id,
      heartbeatMs: HEARTBEAT_INTERVAL_MS,
    })));
    return connection;
  }

  /** Add a socket to a topic after an authorisation check. */
  public subscribe(socketId: string, topic: string): void {
    const connection = this.connections.get(socketId);
    if (!connection) {
      throw new SubscriptionError("unknown_socket", `no socket ${socketId}`);
    }
    if (connection.topics.size >= MAX_TOPICS_PER_SOCKET) {
      throw new SubscriptionError(
        "too_many_topics",
        `a socket may hold at most ${MAX_TOPICS_PER_SOCKET} subscriptions`,
      );
    }
    if (!canSubscribe(topic, connection.principalId, connection.roles)) {
      throw new SubscriptionError("forbidden_topic", `not allowed to watch ${topic}`);
    }
    connection.topics.add(topic);
    const subscribers = this.topicIndex.get(topic) ?? new Set<string>();
    subscribers.add(socketId);
    this.topicIndex.set(topic, subscribers);
  }

  public unsubscribe(socketId: string, topic: string): void {
    this.connections.get(socketId)?.topics.delete(topic);
    const subscribers = this.topicIndex.get(topic);
    if (subscribers) {
      subscribers.delete(socketId);
      if (subscribers.size === 0) {
        this.topicIndex.delete(topic);
      }
    }
  }

  /**
   * Push an event to everyone watching `topic`.
   *
   * A send that throws means the peer went away between our last heartbeat
   * and now, so the connection is reaped rather than allowed to accumulate.
   */
  public broadcast(topic: string, event: string, payload: unknown): number {
    const subscribers = this.topicIndex.get(topic);
    if (!subscribers || subscribers.size === 0) {
      return 0;
    }
    const frame = JSON.stringify(buildEnvelope(topic, event, payload));
    let delivered = 0;
    for (const socketId of [...subscribers]) {
      const connection = this.connections.get(socketId);
      if (!connection || connection.socket.state !== "open") {
        this.disconnect(socketId, "stale socket");
        continue;
      }
      try {
        connection.socket.send(frame);
        delivered += 1;
      } catch {
        this.disconnect(socketId, "send failed");
      }
    }
    return delivered;
  }

  /** Remove a socket from every index and close it. */
  public disconnect(socketId: string, reason: string): void {
    const connection = this.connections.get(socketId);
    if (!connection) {
      return;
    }
    for (const topic of connection.topics) {
      this.unsubscribe(socketId, topic);
    }
    this.connections.delete(socketId);
    if (connection.socket.state === "open") {
      connection.socket.close(1001, reason);
    }
  }

  /** Called when a client answers a ping; resets its strike count. */
  public recordPong(socketId: string): void {
    const connection = this.connections.get(socketId);
    if (connection) {
      connection.missedPongs = 0;
    }
  }

  /**
   * Ping every socket and evict the ones that have ignored several pings.
   *
   * TCP will happily hold a half-open connection for hours, so an
   * application-level heartbeat is the only reliable way to notice a client
   * that vanished without a close frame.
   */
  public sweepDeadConnections(): number {
    let dropped = 0;
    for (const [socketId, connection] of [...this.connections]) {
      if (connection.missedPongs >= MISSED_PONGS_BEFORE_DROP) {
        this.disconnect(socketId, "missed heartbeats");
        dropped += 1;
        continue;
      }
      connection.missedPongs += 1;
      try {
        connection.socket.send(JSON.stringify(buildEnvelope("system:ping", "ping", {})));
      } catch {
        this.disconnect(socketId, "ping failed");
        dropped += 1;
      }
    }
    return dropped;
  }

  public start(): void {
    if (this.heartbeat) {
      return;
    }
    this.heartbeat = setInterval(() => this.sweepDeadConnections(), HEARTBEAT_INTERVAL_MS);
  }

  public stop(): void {
    if (this.heartbeat) {
      clearInterval(this.heartbeat);
      this.heartbeat = null;
    }
    for (const socketId of [...this.connections.keys()]) {
      this.disconnect(socketId, "server shutting down");
    }
  }

  public stats(): { sockets: number; topics: number } {
    return { sockets: this.connections.size, topics: this.topicIndex.size };
  }
}
