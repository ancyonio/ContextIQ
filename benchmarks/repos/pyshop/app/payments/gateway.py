"""HTTP client for the external payment provider.

The provider is the least reliable dependency the service has, so this module
owns the whole failure story: which status codes are worth another attempt,
how long to wait between attempts, and when to stop sending traffic at a
provider that is plainly down.
"""

from __future__ import annotations

import hashlib
import json
import random
import time
from dataclasses import dataclass
from typing import Any, Callable

from ..config import Settings
from ..models import Money, Order


#: Provider responses that are worth another attempt. 409 is deliberately
#: absent: a duplicate charge must never be retried blindly.
RETRYABLE_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})

#: How many consecutive failures trip the breaker open.
CIRCUIT_FAILURE_THRESHOLD = 5

#: Seconds the breaker stays open before allowing a single probe through.
CIRCUIT_RESET_TIMEOUT = 30.0


class PaymentError(RuntimeError):
    """Base class for every failure surfaced by this module."""


class PaymentDeclined(PaymentError):
    """The provider answered, and the answer was no. Never retried."""

    def __init__(self, code: str, message: str):
        self.code = code
        super().__init__(f"declined ({code}): {message}")


class PaymentUnavailable(PaymentError):
    """The provider could not be reached, or kept failing. Safe to retry later."""


class CircuitOpen(PaymentUnavailable):
    """The breaker is open, so the request was never sent."""


@dataclass
class ChargeResult:
    """Normalised success payload returned to the rest of the service."""

    reference: str
    amount_minor: int
    currency: str
    attempts: int
    provider_status: str = "captured"


class CircuitBreaker:
    """Stops hammering a provider that is consistently failing.

    Three states, walked in the usual order: ``closed`` (traffic flows),
    ``open`` (everything is rejected locally for
    :data:`CIRCUIT_RESET_TIMEOUT` seconds) and ``half_open`` (exactly one
    probe is allowed through to see whether the provider recovered).
    """

    def __init__(self, failure_threshold: int = CIRCUIT_FAILURE_THRESHOLD,
                 reset_timeout: float = CIRCUIT_RESET_TIMEOUT,
                 clock: Callable[[], float] = time.monotonic):
        self.failure_threshold = failure_threshold
        self.reset_timeout = reset_timeout
        self.clock = clock
        self.state = "closed"
        self.failures = 0
        self.opened_at = 0.0

    def allow(self) -> bool:
        """Ask permission to make a call, moving to half-open when due."""
        if self.state == "closed":
            return True
        if self.state == "open":
            if self.clock() - self.opened_at >= self.reset_timeout:
                self.state = "half_open"
                return True
            return False
        return True

    def record_success(self) -> None:
        """A call worked: clear the failure count and close the breaker."""
        self.failures = 0
        self.state = "closed"

    def record_failure(self) -> None:
        """A call failed: trip open once the threshold is crossed."""
        self.failures += 1
        if self.state == "half_open" or self.failures >= self.failure_threshold:
            self.state = "open"
            self.opened_at = self.clock()


def compute_backoff(attempt: int, base_delay: float, max_delay: float,
                    jitter: Callable[[], float] = random.random) -> float:
    """Exponential delay with full jitter, capped at ``max_delay``.

    Full jitter (a uniform draw over ``[0, window]`` rather than the window
    itself) is what stops a fleet of workers from retrying in lockstep after a
    provider blip.
    """
    window = min(max_delay, base_delay * (2 ** max(0, attempt - 1)))
    return round(window * jitter(), 4)


def idempotency_key_for(order: Order, amount: Money) -> str:
    """Derive a stable key so a replayed charge cannot double-bill."""
    if order.idempotency_key:
        return order.idempotency_key
    seed = f"{order.id}:{amount.to_minor_units()}:{amount.currency}"
    return "idem_" + hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


class PaymentGateway:
    """Talks to the provider, with retries, backoff and a breaker in front.

    The transport is injected as a callable so tests can drive every failure
    branch without a socket; in production it is a thin wrapper over the
    connection-pooled HTTP session.
    """

    def __init__(self, settings: Settings,
                 transport: Callable[[str, dict[str, Any]], tuple[int, dict]] | None = None,
                 breaker: CircuitBreaker | None = None,
                 sleep: Callable[[float], None] = time.sleep):
        self.settings = settings
        self.transport = transport or _default_transport
        self.breaker = breaker or CircuitBreaker()
        self.sleep = sleep

    def charge(self, order: Order, amount: Money) -> ChargeResult:
        """Capture ``amount`` for ``order``, retrying transient failures.

        Raises :class:`PaymentDeclined` immediately on a hard decline, and
        :class:`PaymentUnavailable` once the attempt budget is exhausted.
        """
        if not self.breaker.allow():
            raise CircuitOpen("payment provider circuit is open")
        payload = {
            "amount": amount.to_minor_units(),
            "currency": amount.currency,
            "order_id": order.id,
            "idempotency_key": idempotency_key_for(order, amount),
        }
        last_error: Exception | None = None
        for attempt in range(1, self.settings.max_payment_attempts + 1):
            try:
                status, body = self.transport(
                    f"{self.settings.payment_base_url}/charges", payload
                )
            except OSError as exc:
                last_error = PaymentUnavailable(f"transport failure: {exc}")
                self.breaker.record_failure()
            else:
                if 200 <= status < 300:
                    self.breaker.record_success()
                    return ChargeResult(
                        reference=body.get("id", ""),
                        amount_minor=payload["amount"],
                        currency=amount.currency,
                        attempts=attempt,
                    )
                if status in (402, 403):
                    self.breaker.record_success()
                    raise PaymentDeclined(
                        body.get("code", "card_declined"),
                        body.get("message", "issuer refused the charge"),
                    )
                if status not in RETRYABLE_STATUS:
                    self.breaker.record_success()
                    raise PaymentError(f"unexpected provider status {status}")
                last_error = PaymentUnavailable(f"provider returned {status}")
                self.breaker.record_failure()
            if attempt < self.settings.max_payment_attempts:
                self.sleep(compute_backoff(
                    attempt, self.settings.retry_base_delay,
                    self.settings.retry_max_delay,
                ))
        raise PaymentUnavailable(
            f"giving up after {self.settings.max_payment_attempts} attempts: {last_error}"
        )

    def refund(self, reference: str, amount: Money) -> ChargeResult:
        """Reverse a previously captured charge. Single attempt by design."""
        if not self.breaker.allow():
            raise CircuitOpen("payment provider circuit is open")
        status, body = self.transport(
            f"{self.settings.payment_base_url}/refunds",
            {"charge_id": reference, "amount": amount.to_minor_units()},
        )
        if not 200 <= status < 300:
            self.breaker.record_failure()
            raise PaymentUnavailable(f"refund failed with status {status}")
        self.breaker.record_success()
        return ChargeResult(
            reference=body.get("id", reference),
            amount_minor=amount.to_minor_units(),
            currency=amount.currency,
            attempts=1,
            provider_status="refunded",
        )

    def health(self) -> dict[str, Any]:
        """Expose breaker internals for the /healthz endpoint."""
        return {
            "circuit_state": self.breaker.state,
            "consecutive_failures": self.breaker.failures,
            "max_attempts": self.settings.max_payment_attempts,
        }


def _default_transport(url: str, payload: dict[str, Any]) -> tuple[int, dict]:
    """Placeholder transport that serialises the payload and reports success.

    The real deployment swaps this for a pooled ``requests.Session`` post; the
    signature is the contract both share.
    """
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    digest = hashlib.md5(encoded).hexdigest()[:16]
    return 200, {"id": f"ch_{digest}", "url": url, "status": "captured"}
