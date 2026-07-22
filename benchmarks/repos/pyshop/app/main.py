"""Composition root: build the object graph and start a process.

``pyshop`` ships one binary with two modes -- ``api`` serves HTTP, ``worker``
runs the fulfilment loop. Both share the same wiring so a configuration
mistake fails identically in either mode.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from .api.handlers import OrderHandlers, build_routes
from .config import ConfigError, Settings, load_settings
from .payments.gateway import CircuitBreaker, PaymentGateway
from .pricing.rules import PricingEngine, default_rules
from .storage.repository import OrderRepository
from .workers.fulfillment import FulfillmentWorker, build_worker


LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"


def configure_logging(level: str = "INFO") -> None:
    """Set up root logging once, before anything else emits a record."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=LOG_FORMAT,
    )


def open_session(settings: Settings) -> Any:
    """Create the database session the repository will use.

    Real deployments hand back a SQLAlchemy scoped session; the signature is
    kept narrow (``execute``/``commit``/``rollback``) so tests can substitute
    a recording double.
    """
    from .storage import repository as repo_module

    session = getattr(repo_module, "SESSION_FACTORY", None)
    if session is not None:
        return session(settings.database_url)
    raise ConfigError(
        "no session factory configured; set repository.SESSION_FACTORY"
    )


def build_gateway(settings: Settings) -> PaymentGateway:
    """Assemble the payment client with a breaker sized from settings."""
    breaker = CircuitBreaker(
        failure_threshold=settings.max_payment_attempts,
        reset_timeout=settings.retry_max_delay * 4,
    )
    return PaymentGateway(settings, breaker=breaker)


def build_handlers(settings: Settings, session: Any) -> OrderHandlers:
    """Wire the HTTP layer over a live session."""
    repository = OrderRepository(session)
    engine = PricingEngine(default_rules())
    return OrderHandlers(settings, repository, build_gateway(settings), engine)


def serve_api(settings: Settings, session: Any) -> dict[str, Any]:
    """Return the route table a WSGI/ASGI adapter will mount."""
    handlers = build_handlers(settings, session)
    routes = build_routes(handlers)
    logging.getLogger("pyshop.api").info(
        "serving %d routes with config %s", len(routes), settings.redacted()
    )
    return routes


def serve_worker(settings: Settings, session: Any) -> FulfillmentWorker:
    """Build and start the fulfilment loop until it is asked to stop."""
    repository = OrderRepository(session)
    worker = build_worker(settings, repository, build_gateway(settings))
    worker.run_forever()
    return worker


def main(argv: list[str] | None = None) -> int:
    """Entry point: ``python -m app.main [api|worker]``."""
    args = list(sys.argv[1:] if argv is None else argv)
    mode = args[0] if args else "api"
    configure_logging()
    try:
        settings = load_settings()
        session = open_session(settings)
    except ConfigError as exc:
        logging.getLogger("pyshop").error("startup failed: %s", exc)
        return 2
    if mode == "worker":
        serve_worker(settings, session)
    elif mode == "api":
        serve_api(settings, session)
    else:
        logging.getLogger("pyshop").error("unknown mode %r", mode)
        return 64
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
