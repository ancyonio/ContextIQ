"""Background worker that drives paid orders through fulfilment.

The loop is deliberately boring: claim a batch, process each order, release
the claim, sleep. Everything interesting -- what a legal move is, how to talk
to the provider -- is delegated to the services this module composes.
"""

from __future__ import annotations

import logging
import signal
import socket
import time
from dataclasses import dataclass, field
from typing import Any

from ..config import Settings
from ..models import Order, OrderStatus
from ..orders.state_machine import (
    IllegalTransition, OrderStateMachine, is_terminal,
)
from ..payments.gateway import PaymentGateway, PaymentUnavailable
from ..storage.repository import OrderRepository, unit_of_work


LOGGER = logging.getLogger("pyshop.worker")

#: Consecutive batch failures tolerated before the loop backs off hard.
MAX_CONSECUTIVE_ERRORS = 3

#: How long the loop naps after a hard backoff, in seconds.
ERROR_BACKOFF_SECONDS = 15.0


@dataclass
class WorkerStats:
    """Counters exported to the metrics endpoint each cycle."""

    claimed: int = 0
    shipped: int = 0
    refunded: int = 0
    skipped: int = 0
    errors: int = 0
    cycles: int = 0
    last_cycle_started: float = field(default_factory=time.time)

    def as_dict(self) -> dict[str, Any]:
        return {
            "claimed": self.claimed, "shipped": self.shipped,
            "refunded": self.refunded, "skipped": self.skipped,
            "errors": self.errors, "cycles": self.cycles,
            "uptime_seconds": round(time.time() - self.last_cycle_started, 2),
        }


def worker_identity() -> str:
    """A stable-enough id for claim rows: hostname plus process start time."""
    return f"{socket.gethostname()}-{int(time.time())}"


class FulfillmentWorker:
    """Polls for claimable orders and advances them one step per cycle.

    A cycle is at-least-once: if the process dies mid-batch the claims are
    released by a janitor query and another worker replays them, which is why
    every transition here has to tolerate being applied twice.
    """

    def __init__(self, settings: Settings, repository: OrderRepository,
                 gateway: PaymentGateway, machine: OrderStateMachine | None = None,
                 shipper: Any | None = None):
        self.settings = settings
        self.repository = repository
        self.gateway = gateway
        self.machine = machine or OrderStateMachine()
        self.shipper = shipper or ConsoleShipper()
        self.stats = WorkerStats()
        self.worker_id = worker_identity()
        self._running = False
        self._consecutive_errors = 0

    def install_signal_handlers(self) -> None:
        """Ask for a graceful stop on SIGTERM so batches are not torn in half."""
        signal.signal(signal.SIGTERM, lambda *_: self.stop())
        signal.signal(signal.SIGINT, lambda *_: self.stop())

    def stop(self) -> None:
        """Request that the loop finish the current batch and exit."""
        LOGGER.info("worker %s asked to stop", self.worker_id)
        self._running = False

    def run_forever(self, max_cycles: int = 0) -> WorkerStats:
        """Poll until stopped. ``max_cycles`` bounds the loop for tests."""
        self._running = True
        cycles = 0
        while self._running:
            cycles += 1
            try:
                processed = self.run_once()
                self._consecutive_errors = 0
            except Exception:
                self._consecutive_errors += 1
                self.stats.errors += 1
                LOGGER.exception("worker cycle failed (%d in a row)",
                                 self._consecutive_errors)
                processed = 0
            if self._consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                LOGGER.error("backing off for %.1fs", ERROR_BACKOFF_SECONDS)
                time.sleep(ERROR_BACKOFF_SECONDS)
                self._consecutive_errors = 0
            elif processed == 0:
                time.sleep(self.settings.worker_poll_interval)
            if max_cycles and cycles >= max_cycles:
                break
        return self.stats

    def run_once(self) -> int:
        """Claim one batch and process it. Returns how many orders were seen."""
        self.stats.cycles += 1
        batch = self.repository.claim_batch(
            self.settings.worker_batch_size, self.worker_id
        )
        self.stats.claimed += len(batch)
        for order in batch:
            try:
                self.process_order(order)
            except PaymentUnavailable as exc:
                LOGGER.warning("order %s deferred: %s", order.id, exc)
                self.stats.skipped += 1
            except IllegalTransition as exc:
                LOGGER.error("order %s stuck: %s", order.id, exc)
                self.stats.errors += 1
            finally:
                self.repository.release(order.id)
        return len(batch)

    def process_order(self, order: Order) -> None:
        """Advance a single order one step along the happy path."""
        if is_terminal(order.status):
            self.stats.skipped += 1
            return
        if order.status == OrderStatus.PAID:
            self.machine.transition(
                order, OrderStatus.FULFILLING, reason="picked by worker"
            )
            self._flush(order)
            return
        if order.status == OrderStatus.FULFILLING:
            tracking = self.shipper.dispatch(order)
            self.machine.transition(
                order, OrderStatus.SHIPPED, reason=f"tracking {tracking}"
            )
            self.stats.shipped += 1
            self._flush(order)
            return
        self.stats.skipped += 1

    def refund_order(self, order: Order, reason: str) -> None:
        """Reverse the capture and mark the order refunded, in that order."""
        if not order.payment_reference:
            raise IllegalTransition(order.id, order.status, OrderStatus.REFUNDED)
        self.gateway.refund(order.payment_reference, order.subtotal())
        self.machine.refund(order, reason=reason)
        self.stats.refunded += 1
        self._flush(order)

    def _flush(self, order: Order) -> None:
        """Persist the order plus its audit events atomically."""
        with unit_of_work(self.repository.session):
            self.repository.save(order)
            for event in self.machine.drain_events():
                self.repository.record_transition(event)


class ConsoleShipper:
    """Stand-in carrier integration that logs instead of calling a courier."""

    def dispatch(self, order: Order) -> str:
        """Return a tracking number for ``order``, logging the handoff."""
        tracking = f"TRK{order.id[-8:].upper()}"
        LOGGER.info("dispatching order %s with %d items as %s",
                    order.id, order.item_count(), tracking)
        return tracking


def build_worker(settings: Settings, repository: OrderRepository,
                 gateway: PaymentGateway) -> FulfillmentWorker:
    """Compose a worker with signal handlers already installed."""
    worker = FulfillmentWorker(settings, repository, gateway)
    worker.install_signal_handlers()
    return worker
