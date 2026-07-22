"""Domain value objects shared by every layer of the order service.

These types are deliberately dumb: they hold data and answer arithmetic
questions about themselves. All behaviour that needs the database, the clock
or the network lives in the service modules.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, ROUND_HALF_UP
from enum import Enum


CENTS = Decimal("0.01")


class OrderStatus(str, Enum):
    """Lifecycle positions an order can occupy.

    The string values are what gets persisted, so renaming a member is a
    migration, not a refactor.
    """

    DRAFT = "draft"
    PENDING_PAYMENT = "pending_payment"
    PAID = "paid"
    FULFILLING = "fulfilling"
    SHIPPED = "shipped"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


@dataclass(frozen=True)
class Money:
    """A currency-tagged decimal amount.

    Arithmetic between two different currencies raises rather than silently
    coercing -- mixing them up is the classic way to lose real money.
    """

    amount: Decimal
    currency: str = "USD"

    def __post_init__(self) -> None:
        object.__setattr__(self, "amount", Decimal(self.amount).quantize(CENTS, ROUND_HALF_UP))

    def add(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount + other.amount, self.currency)

    def subtract(self, other: "Money") -> "Money":
        self._assert_same_currency(other)
        return Money(self.amount - other.amount, self.currency)

    def scale(self, factor: Decimal | int) -> "Money":
        """Multiply the amount, rounding half-up to the nearest cent."""
        return Money(self.amount * Decimal(factor), self.currency)

    def is_zero(self) -> bool:
        return self.amount == Decimal("0.00")

    def to_minor_units(self) -> int:
        """Convert to integer cents, the wire format the gateway expects."""
        return int((self.amount * 100).to_integral_value(ROUND_HALF_UP))

    def _assert_same_currency(self, other: "Money") -> None:
        if other.currency != self.currency:
            raise ValueError(
                f"currency mismatch: {self.currency} vs {other.currency}"
            )


@dataclass
class Customer:
    """The buyer, plus the loyalty attributes pricing rules care about."""

    id: str
    email: str
    tier: str = "standard"
    lifetime_orders: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def is_loyal(self) -> bool:
        """Loyalty kicks in at ten completed orders or an explicit tier."""
        return self.lifetime_orders >= 10 or self.tier in ("gold", "platinum")


@dataclass
class OrderLine:
    """One SKU on an order, at the price captured when it was added."""

    sku: str
    quantity: int
    unit_price: Money
    category: str = "general"

    def subtotal(self) -> Money:
        return self.unit_price.scale(self.quantity)


@dataclass
class Order:
    """An order aggregate: lines, status and the payment reference."""

    id: str = field(default_factory=lambda: f"ord_{uuid.uuid4().hex[:12]}")
    customer: Customer | None = None
    lines: list[OrderLine] = field(default_factory=list)
    status: OrderStatus = OrderStatus.DRAFT
    currency: str = "USD"
    discount_code: str | None = None
    payment_reference: str | None = None
    idempotency_key: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime | None = None

    def add_line(self, line: OrderLine) -> None:
        """Append a line, merging quantities when the SKU is already present."""
        for existing in self.lines:
            if existing.sku == line.sku:
                existing.quantity += line.quantity
                return
        self.lines.append(line)

    def subtotal(self) -> Money:
        """Sum every line before discounts, shipping or tax."""
        total = Money(Decimal("0.00"), self.currency)
        for line in self.lines:
            total = total.add(line.subtotal())
        return total

    def item_count(self) -> int:
        return sum(line.quantity for line in self.lines)

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc)


@dataclass
class PricedOrder:
    """Result of running the pricing rules over an :class:`Order`."""

    order: Order
    subtotal: Money
    discount: Money
    shipping: Money
    tax: Money
    total: Money
    applied_rules: list[str] = field(default_factory=list)
