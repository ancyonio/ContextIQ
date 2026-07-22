"""HTTP entry points for the order service.

Handlers are plain functions taking a request dictionary and returning a
``(status, body)`` pair, which keeps them framework-agnostic and trivially
unit-testable. Business decisions live in the pricing engine, the state
machine and the payment gateway; this module only translates between HTTP and
those services.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
from typing import Any, Callable

from ..config import Settings
from ..models import Customer, Order, OrderStatus
from ..orders.state_machine import (
    IllegalTransition, OrderStateMachine, next_statuses,
)
from ..payments.gateway import (
    CircuitOpen, PaymentDeclined, PaymentGateway, PaymentUnavailable,
)
from ..pricing.rules import PricingEngine, line_from_catalog, validate_discount_code
from ..storage.repository import OrderRepository, RecordNotFound, unit_of_work


MAX_LINES_PER_ORDER = 40


class ValidationError(ValueError):
    """Raised when a request body fails shape or range checks."""

    def __init__(self, field: str, message: str):
        self.field = field
        super().__init__(f"{field}: {message}")


def json_error(status: int, code: str, message: str) -> tuple[int, dict[str, Any]]:
    """Uniform error envelope so clients only parse one shape."""
    return status, {"error": {"code": code, "message": message}}


def parse_order_request(body: dict[str, Any], currency: str) -> Order:
    """Validate an incoming create-order payload and build an aggregate."""
    lines = body.get("lines")
    if not isinstance(lines, list) or not lines:
        raise ValidationError("lines", "at least one line is required")
    if len(lines) > MAX_LINES_PER_ORDER:
        raise ValidationError("lines", f"at most {MAX_LINES_PER_ORDER} lines")
    customer_body = body.get("customer") or {}
    if not customer_body.get("id"):
        raise ValidationError("customer.id", "is required")
    order = Order(
        customer=Customer(
            id=str(customer_body["id"]),
            email=str(customer_body.get("email", "")),
            tier=str(customer_body.get("tier", "standard")),
            lifetime_orders=int(customer_body.get("lifetime_orders", 0)),
        ),
        currency=body.get("currency", currency),
        discount_code=body.get("discount_code"),
        idempotency_key=body.get("idempotency_key"),
    )
    for index, raw in enumerate(lines):
        if not raw.get("sku"):
            raise ValidationError(f"lines[{index}].sku", "is required")
        quantity = int(raw.get("quantity", 1))
        if quantity < 1:
            raise ValidationError(f"lines[{index}].quantity", "must be positive")
        try:
            unit_price = Decimal(str(raw["unit_price"]))
        except (KeyError, InvalidOperation) as exc:
            raise ValidationError(
                f"lines[{index}].unit_price", "must be a decimal amount"
            ) from exc
        order.add_line(line_from_catalog(
            sku=str(raw["sku"]), quantity=quantity, unit_price=unit_price,
            currency=order.currency, category=str(raw.get("category", "general")),
        ))
    if order.discount_code and not validate_discount_code(order.discount_code):
        raise ValidationError("discount_code", "is not a recognised promotion")
    return order


class OrderHandlers:
    """The HTTP surface: create, quote, pay, cancel and read an order."""

    def __init__(self, settings: Settings, repository: OrderRepository,
                 gateway: PaymentGateway, engine: PricingEngine | None = None,
                 machine: OrderStateMachine | None = None):
        self.settings = settings
        self.repository = repository
        self.gateway = gateway
        self.engine = engine or PricingEngine()
        self.machine = machine or OrderStateMachine()

    def create_order(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST /orders -- validate, price and persist a draft order."""
        try:
            order = parse_order_request(
                request.get("body") or {}, self.settings.default_currency
            )
        except ValidationError as exc:
            return json_error(422, "invalid_request", str(exc))
        if order.idempotency_key:
            existing = self.repository.find_by_idempotency_key(order.idempotency_key)
            if existing is not None:
                return 200, self._serialize(existing)
        priced = self.engine.price(order)
        with unit_of_work(self.repository.session):
            self.repository.save(order)
        body = self._serialize(order)
        body["pricing"] = _serialize_pricing(priced)
        return 201, body

    def quote_order(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST /orders/quote -- price a basket without persisting anything."""
        try:
            order = parse_order_request(
                request.get("body") or {}, self.settings.default_currency
            )
        except ValidationError as exc:
            return json_error(422, "invalid_request", str(exc))
        priced = self.engine.price(order)
        return 200, {
            "pricing": _serialize_pricing(priced),
            "explanation": self.engine.explain(order),
        }

    def pay_order(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST /orders/{id}/pay -- capture funds and advance the status."""
        order_id = str(request.get("path_params", {}).get("order_id", ""))
        try:
            order = self.repository.get(order_id)
        except RecordNotFound:
            return json_error(404, "not_found", f"no order {order_id}")
        priced = self.engine.price(order)
        try:
            self.machine.transition(
                order, OrderStatus.PENDING_PAYMENT, reason="checkout started"
            )
            result = self.gateway.charge(order, priced.total)
            self.machine.mark_paid(order, result.reference)
        except IllegalTransition as exc:
            return json_error(409, "illegal_transition", str(exc))
        except PaymentDeclined as exc:
            self.machine.cancel(order, reason=f"declined: {exc.code}")
            self._persist(order)
            return json_error(402, "payment_declined", str(exc))
        except CircuitOpen as exc:
            return json_error(503, "provider_unavailable", str(exc))
        except PaymentUnavailable as exc:
            return json_error(503, "payment_unavailable", str(exc))
        self._persist(order)
        return 200, {
            "order": self._serialize(order),
            "charge": {"reference": result.reference, "attempts": result.attempts},
        }

    def cancel_order(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """POST /orders/{id}/cancel -- customer-initiated cancellation."""
        order_id = str(request.get("path_params", {}).get("order_id", ""))
        reason = str((request.get("body") or {}).get("reason", "customer request"))
        try:
            order = self.repository.get(order_id)
            self.machine.cancel(order, reason=reason)
        except RecordNotFound:
            return json_error(404, "not_found", f"no order {order_id}")
        except IllegalTransition as exc:
            return json_error(409, "illegal_transition", str(exc))
        self._persist(order)
        return 200, self._serialize(order)

    def get_order(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """GET /orders/{id} -- read one order plus its legal next moves."""
        order_id = str(request.get("path_params", {}).get("order_id", ""))
        try:
            order = self.repository.get(order_id)
        except RecordNotFound:
            return json_error(404, "not_found", f"no order {order_id}")
        return 200, self._serialize(order)

    def healthz(self, request: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        """GET /healthz -- readiness plus payment breaker state."""
        return 200, {
            "status": "ok",
            "payments": self.gateway.health(),
            "orders_by_status": self.repository.count_by_status(),
        }

    def _persist(self, order: Order) -> None:
        """Write the order and drain the audit trail in one transaction."""
        with unit_of_work(self.repository.session):
            self.repository.save(order)
            for event in self.machine.drain_events():
                self.repository.record_transition(event)

    def _serialize(self, order: Order) -> dict[str, Any]:
        return {
            "id": order.id,
            "status": order.status.value,
            "currency": order.currency,
            "item_count": order.item_count(),
            "subtotal": str(order.subtotal().amount),
            "payment_reference": order.payment_reference,
            "next_statuses": [s.value for s in next_statuses(order.status)],
            "lines": [
                {"sku": line.sku, "quantity": line.quantity,
                 "unit_price": str(line.unit_price.amount)}
                for line in order.lines
            ],
        }


def _serialize_pricing(priced: Any) -> dict[str, Any]:
    """Flatten a :class:`~app.models.PricedOrder` for the JSON response."""
    return {
        "subtotal": str(priced.subtotal.amount),
        "discount": str(priced.discount.amount),
        "shipping": str(priced.shipping.amount),
        "tax": str(priced.tax.amount),
        "total": str(priced.total.amount),
        "applied_rules": list(priced.applied_rules),
    }


def build_routes(handlers: OrderHandlers) -> dict[str, Callable[..., Any]]:
    """Map ``METHOD /path`` strings to bound handler methods."""
    return {
        "POST /orders": handlers.create_order,
        "POST /orders/quote": handlers.quote_order,
        "POST /orders/{order_id}/pay": handlers.pay_order,
        "POST /orders/{order_id}/cancel": handlers.cancel_order,
        "GET /orders/{order_id}": handlers.get_order,
        "GET /healthz": handlers.healthz,
    }
