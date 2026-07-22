"""The order lifecycle, expressed as an explicit transition table.

Every status change in the system goes through :class:`OrderStateMachine`.
Handlers and workers never assign ``order.status`` themselves -- doing so is
how illegal states (a refunded order sliding back into fulfilment) get into
production.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..models import Order, OrderStatus


#: The single source of truth for which moves are legal.
ALLOWED_TRANSITIONS: dict[OrderStatus, frozenset[OrderStatus]] = {
    OrderStatus.DRAFT: frozenset({OrderStatus.PENDING_PAYMENT, OrderStatus.CANCELLED}),
    OrderStatus.PENDING_PAYMENT: frozenset({OrderStatus.PAID, OrderStatus.CANCELLED}),
    OrderStatus.PAID: frozenset({OrderStatus.FULFILLING, OrderStatus.REFUNDED}),
    OrderStatus.FULFILLING: frozenset({OrderStatus.SHIPPED, OrderStatus.REFUNDED}),
    OrderStatus.SHIPPED: frozenset({OrderStatus.REFUNDED}),
    OrderStatus.CANCELLED: frozenset(),
    OrderStatus.REFUNDED: frozenset(),
}

#: Statuses from which no further move is possible.
TERMINAL_STATUSES = frozenset({OrderStatus.CANCELLED, OrderStatus.REFUNDED})


class IllegalTransition(ValueError):
    """Raised when a caller asks for a move the transition table forbids."""

    def __init__(self, order_id: str, source: OrderStatus, target: OrderStatus):
        self.order_id = order_id
        self.source = source
        self.target = target
        super().__init__(
            f"order {order_id}: cannot move from {source.value} to {target.value}"
        )


@dataclass(frozen=True)
class TransitionEvent:
    """An audit record emitted for every accepted transition."""

    order_id: str
    source: OrderStatus
    target: OrderStatus
    reason: str
    occurred_at: datetime

    def to_dict(self) -> dict[str, str]:
        return {
            "order_id": self.order_id,
            "from": self.source.value,
            "to": self.target.value,
            "reason": self.reason,
            "occurred_at": self.occurred_at.isoformat(),
        }


class OrderStateMachine:
    """Validates and applies status changes, recording an audit trail.

    The machine is stateless with respect to any individual order: it takes an
    :class:`~app.models.Order`, checks the move against
    :data:`ALLOWED_TRANSITIONS` and mutates it in place. The emitted events are
    accumulated so the caller can persist them in the same transaction as the
    order row.
    """

    def __init__(self) -> None:
        self.events: list[TransitionEvent] = []

    def can_transition(self, source: OrderStatus, target: OrderStatus) -> bool:
        """Cheap predicate used by handlers to decide whether to even try."""
        return target in ALLOWED_TRANSITIONS.get(source, frozenset())

    def transition(
        self, order: Order, target: OrderStatus, reason: str = ""
    ) -> TransitionEvent:
        """Move ``order`` into ``target`` or raise :class:`IllegalTransition`.

        Re-entering the current status is treated as a no-op rather than an
        error, because at-least-once delivery means the worker will replay
        transitions it has already applied.
        """
        source = order.status
        if source == target:
            event = TransitionEvent(
                order.id, source, target, reason or "idempotent replay",
                datetime.now(timezone.utc),
            )
            self.events.append(event)
            return event
        if not self.can_transition(source, target):
            raise IllegalTransition(order.id, source, target)
        order.status = target
        order.touch()
        event = TransitionEvent(
            order.id, source, target, reason or "unspecified",
            datetime.now(timezone.utc),
        )
        self.events.append(event)
        return event

    def mark_paid(self, order: Order, payment_reference: str) -> TransitionEvent:
        """Record a successful capture and attach the gateway reference."""
        order.payment_reference = payment_reference
        return self.transition(order, OrderStatus.PAID, reason="payment captured")

    def cancel(self, order: Order, reason: str) -> TransitionEvent:
        """Cancel an order that has not yet been paid for."""
        if order.status in TERMINAL_STATUSES:
            raise IllegalTransition(order.id, order.status, OrderStatus.CANCELLED)
        return self.transition(order, OrderStatus.CANCELLED, reason=reason)

    def refund(self, order: Order, reason: str) -> TransitionEvent:
        """Reverse a captured payment for a paid, fulfilling or shipped order."""
        return self.transition(order, OrderStatus.REFUNDED, reason=reason)

    def drain_events(self) -> list[TransitionEvent]:
        """Hand the accumulated audit records to the caller and reset."""
        drained = list(self.events)
        self.events.clear()
        return drained


def is_terminal(status: OrderStatus) -> bool:
    """True when no further transition out of ``status`` is possible."""
    return status in TERMINAL_STATUSES


def next_statuses(status: OrderStatus) -> list[OrderStatus]:
    """List the legal onward moves, sorted for stable API output."""
    return sorted(ALLOWED_TRANSITIONS.get(status, frozenset()), key=lambda s: s.value)
