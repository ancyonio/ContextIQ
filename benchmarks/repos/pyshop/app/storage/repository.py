"""Persistence layer: table metadata plus the order repository.

The mapping is written in the declarative style a SQLAlchemy project uses, but
the session object is injected, so nothing here opens a connection on import.
Query construction lives in this module exclusively -- no SQL strings are
allowed to leak into handlers or workers.
"""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from typing import Any, Iterator

from ..models import Customer, Money, Order, OrderLine, OrderStatus
from ..orders.state_machine import TransitionEvent


ORDERS_TABLE = "orders"
ORDER_LINES_TABLE = "order_lines"
TRANSITIONS_TABLE = "order_transitions"

#: Statuses the fulfilment worker is allowed to pick up.
CLAIMABLE_STATUSES = (OrderStatus.PAID.value, OrderStatus.FULFILLING.value)


class RecordNotFound(LookupError):
    """Raised when a lookup by primary key finds nothing."""


class ConcurrentUpdate(RuntimeError):
    """Raised when an optimistic-locking update matches zero rows."""


@contextmanager
def unit_of_work(session: Any) -> Iterator[Any]:
    """Run a block inside one transaction, rolling back on any exception."""
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise


def row_to_order(row: dict[str, Any], line_rows: list[dict[str, Any]]) -> Order:
    """Rehydrate an :class:`Order` aggregate from flat database rows."""
    customer = None
    if row.get("customer_id"):
        customer = Customer(
            id=row["customer_id"],
            email=row.get("customer_email", ""),
            tier=row.get("customer_tier", "standard"),
            lifetime_orders=int(row.get("customer_lifetime_orders", 0)),
        )
    order = Order(
        id=row["id"],
        customer=customer,
        status=OrderStatus(row["status"]),
        currency=row.get("currency", "USD"),
        discount_code=row.get("discount_code"),
        payment_reference=row.get("payment_reference"),
        idempotency_key=row.get("idempotency_key"),
    )
    for line in line_rows:
        order.lines.append(OrderLine(
            sku=line["sku"],
            quantity=int(line["quantity"]),
            unit_price=Money(Decimal(str(line["unit_price"])), order.currency),
            category=line.get("category", "general"),
        ))
    return order


def order_to_row(order: Order) -> dict[str, Any]:
    """Flatten an aggregate into the column dictionary the driver wants."""
    return {
        "id": order.id,
        "customer_id": order.customer.id if order.customer else None,
        "status": order.status.value,
        "currency": order.currency,
        "discount_code": order.discount_code,
        "payment_reference": order.payment_reference,
        "idempotency_key": order.idempotency_key,
        "item_count": order.item_count(),
        "subtotal": str(order.subtotal().amount),
        "updated_at": (order.updated_at or datetime.now(timezone.utc)).isoformat(),
    }


class OrderRepository:
    """All reads and writes for the order aggregate.

    Methods take and return domain objects; the caller never sees a row. The
    session is expected to expose ``execute``/``commit``/``rollback``, which is
    what both a real SQLAlchemy session and the in-memory test double provide.
    """

    def __init__(self, session: Any):
        self.session = session

    def get(self, order_id: str) -> Order:
        """Load one order with its lines, or raise :class:`RecordNotFound`."""
        rows = self.session.execute(
            f"SELECT * FROM {ORDERS_TABLE} WHERE id = :id", {"id": order_id}
        ).fetchall()
        if not rows:
            raise RecordNotFound(f"no order with id {order_id}")
        lines = self.session.execute(
            f"SELECT * FROM {ORDER_LINES_TABLE} WHERE order_id = :id ORDER BY sku",
            {"id": order_id},
        ).fetchall()
        return row_to_order(dict(rows[0]), [dict(r) for r in lines])

    def find_by_idempotency_key(self, key: str) -> Order | None:
        """Support at-least-once submission from flaky mobile clients."""
        rows = self.session.execute(
            f"SELECT id FROM {ORDERS_TABLE} WHERE idempotency_key = :key",
            {"key": key},
        ).fetchall()
        if not rows:
            return None
        return self.get(dict(rows[0])["id"])

    def save(self, order: Order) -> Order:
        """Upsert the aggregate and replace its line rows."""
        payload = order_to_row(order)
        self.session.execute(
            f"INSERT INTO {ORDERS_TABLE} (id, customer_id, status, currency, "
            "discount_code, payment_reference, idempotency_key, item_count, "
            "subtotal, updated_at) VALUES (:id, :customer_id, :status, "
            ":currency, :discount_code, :payment_reference, :idempotency_key, "
            ":item_count, :subtotal, :updated_at) "
            "ON CONFLICT (id) DO UPDATE SET status = :status, "
            "payment_reference = :payment_reference, updated_at = :updated_at",
            payload,
        )
        self.session.execute(
            f"DELETE FROM {ORDER_LINES_TABLE} WHERE order_id = :id",
            {"id": order.id},
        )
        for line in order.lines:
            self.session.execute(
                f"INSERT INTO {ORDER_LINES_TABLE} (order_id, sku, quantity, "
                "unit_price, category) VALUES (:order_id, :sku, :quantity, "
                ":unit_price, :category)",
                {
                    "order_id": order.id, "sku": line.sku,
                    "quantity": line.quantity,
                    "unit_price": str(line.unit_price.amount),
                    "category": line.category,
                },
            )
        return order

    def claim_batch(self, limit: int, worker_id: str) -> list[Order]:
        """Lock and return a batch of orders for the fulfilment worker.

        ``SKIP LOCKED`` is what lets several worker processes share one queue
        table without blocking each other on the same rows.
        """
        rows = self.session.execute(
            f"UPDATE {ORDERS_TABLE} SET claimed_by = :worker "
            f"WHERE id IN (SELECT id FROM {ORDERS_TABLE} "
            "WHERE status = ANY(:statuses) AND claimed_by IS NULL "
            "ORDER BY updated_at LIMIT :limit FOR UPDATE SKIP LOCKED) "
            "RETURNING id",
            {"worker": worker_id, "statuses": list(CLAIMABLE_STATUSES),
             "limit": limit},
        ).fetchall()
        return [self.get(dict(row)["id"]) for row in rows]

    def release(self, order_id: str) -> None:
        """Drop a worker's claim so another process can pick the order up."""
        self.session.execute(
            f"UPDATE {ORDERS_TABLE} SET claimed_by = NULL WHERE id = :id",
            {"id": order_id},
        )

    def record_transition(self, event: TransitionEvent) -> None:
        """Append one audit row for a status change."""
        self.session.execute(
            f"INSERT INTO {TRANSITIONS_TABLE} (order_id, source, target, "
            "reason, occurred_at) VALUES (:order_id, :from, :to, :reason, "
            ":occurred_at)",
            event.to_dict(),
        )

    def count_by_status(self) -> dict[str, int]:
        """Dashboard query: how many orders sit in each status right now."""
        rows = self.session.execute(
            f"SELECT status, COUNT(*) AS n FROM {ORDERS_TABLE} GROUP BY status"
        ).fetchall()
        return {dict(r)["status"]: int(dict(r)["n"]) for r in rows}
