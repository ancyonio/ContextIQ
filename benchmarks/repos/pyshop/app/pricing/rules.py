"""Discount, shipping and tax rules applied to an order before checkout.

Rules are small objects with a uniform interface so they can be reordered or
feature-flagged without touching the engine. The engine itself is responsible
for one thing only: running the rules in priority order and never letting the
running total go below zero.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Protocol

from ..models import Money, Order, OrderLine, PricedOrder


FREE_SHIPPING_THRESHOLD = Decimal("75.00")
FLAT_SHIPPING_RATE = Decimal("6.95")
DEFAULT_TAX_RATE = Decimal("0.0825")

#: Promotional codes and the fraction they take off the subtotal.
PROMO_CODES: dict[str, Decimal] = {
    "WELCOME10": Decimal("0.10"),
    "SPRING15": Decimal("0.15"),
    "VIP25": Decimal("0.25"),
}


class PricingRule(Protocol):
    """Structural interface every rule satisfies."""

    name: str
    priority: int

    def applies(self, order: Order) -> bool:
        ...

    def discount(self, order: Order, running_subtotal: Money) -> Money:
        ...


class PromoCodeRule:
    """Percentage off the whole basket for a recognised promo code."""

    name = "promo_code"
    priority = 10

    def applies(self, order: Order) -> bool:
        code = (order.discount_code or "").upper()
        return code in PROMO_CODES

    def discount(self, order: Order, running_subtotal: Money) -> Money:
        code = (order.discount_code or "").upper()
        fraction = PROMO_CODES.get(code, Decimal("0"))
        return running_subtotal.scale(fraction)


class LoyaltyRule:
    """Five percent off for customers the loyalty model considers regulars."""

    name = "loyalty"
    priority = 20

    def applies(self, order: Order) -> bool:
        return order.customer is not None and order.customer.is_loyal()

    def discount(self, order: Order, running_subtotal: Money) -> Money:
        return running_subtotal.scale(Decimal("0.05"))


class BulkLineRule:
    """Ten percent off any single line with a quantity of ten or more."""

    name = "bulk_line"
    priority = 30

    def applies(self, order: Order) -> bool:
        return any(line.quantity >= 10 for line in order.lines)

    def discount(self, order: Order, running_subtotal: Money) -> Money:
        total = Money(Decimal("0.00"), order.currency)
        for line in order.lines:
            if line.quantity >= 10:
                total = total.add(line.subtotal().scale(Decimal("0.10")))
        return total


class PricingEngine:
    """Runs the configured rules and produces a :class:`PricedOrder`.

    Discounts stack additively but are clamped so the discounted subtotal can
    never dip below zero; shipping and tax are then computed on the discounted
    amount, which is what finance reconciles against.
    """

    def __init__(self, rules: list[PricingRule] | None = None,
                 tax_rate: Decimal = DEFAULT_TAX_RATE):
        self.rules = sorted(rules or default_rules(), key=lambda r: r.priority)
        self.tax_rate = tax_rate

    def price(self, order: Order) -> PricedOrder:
        """Compute the full money breakdown for ``order``."""
        subtotal = order.subtotal()
        discount = Money(Decimal("0.00"), order.currency)
        applied: list[str] = []
        for rule in self.rules:
            if not rule.applies(order):
                continue
            amount = rule.discount(order, subtotal)
            if amount.is_zero():
                continue
            discount = discount.add(amount)
            applied.append(rule.name)
        if discount.amount > subtotal.amount:
            discount = subtotal
        discounted = subtotal.subtract(discount)
        shipping = self.shipping_for(discounted)
        tax = discounted.add(shipping).scale(self.tax_rate)
        total = discounted.add(shipping).add(tax)
        return PricedOrder(
            order=order, subtotal=subtotal, discount=discount,
            shipping=shipping, tax=tax, total=total, applied_rules=applied,
        )

    def shipping_for(self, discounted_subtotal: Money) -> Money:
        """Flat rate carriage, waived once the basket clears the threshold."""
        if discounted_subtotal.amount >= FREE_SHIPPING_THRESHOLD:
            return Money(Decimal("0.00"), discounted_subtotal.currency)
        return Money(FLAT_SHIPPING_RATE, discounted_subtotal.currency)

    def explain(self, order: Order) -> list[str]:
        """Human-readable trace of which rules fired, for support tooling."""
        priced = self.price(order)
        lines = [f"subtotal {priced.subtotal.amount} {priced.subtotal.currency}"]
        for name in priced.applied_rules:
            lines.append(f"  rule {name} applied")
        lines.append(f"discount -{priced.discount.amount}")
        lines.append(f"shipping +{priced.shipping.amount}")
        lines.append(f"tax +{priced.tax.amount}")
        lines.append(f"total {priced.total.amount}")
        return lines


def default_rules() -> list[PricingRule]:
    """The rule set the service boots with."""
    return [PromoCodeRule(), LoyaltyRule(), BulkLineRule()]


def validate_discount_code(code: str | None) -> bool:
    """Check a promo code without pricing a whole order (used by the API)."""
    if not code:
        return False
    return code.strip().upper() in PROMO_CODES


def line_from_catalog(sku: str, quantity: int, unit_price: Decimal,
                      currency: str = "USD", category: str = "general") -> OrderLine:
    """Small factory so handlers do not construct Money by hand."""
    return OrderLine(
        sku=sku, quantity=quantity,
        unit_price=Money(unit_price, currency), category=category,
    )
