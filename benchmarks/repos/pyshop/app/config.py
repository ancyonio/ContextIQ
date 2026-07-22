"""Runtime configuration for the pyshop order service.

Every tunable the service has is declared here as a single frozen dataclass so
that the worker, the HTTP layer and the payment client all read the same
values. Nothing else in the codebase is allowed to call ``os.environ``
directly -- that rule keeps the twelve-factor surface auditable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any


DEFAULT_CURRENCY = "USD"
PAYMENT_TIMEOUT_SECONDS = 12.0
MAX_PAYMENT_ATTEMPTS = 5


class ConfigError(RuntimeError):
    """Raised when a required environment variable is missing or unparsable."""


@dataclass(frozen=True)
class Settings:
    """Immutable snapshot of the process environment.

    Built once at boot by :func:`load_settings` and then threaded through the
    application explicitly. Treating settings as a value rather than a global
    is what makes the fixtures in ``tests`` able to run several configurations
    inside one interpreter.
    """

    database_url: str
    payment_base_url: str
    payment_api_key: str
    default_currency: str = DEFAULT_CURRENCY
    payment_timeout: float = PAYMENT_TIMEOUT_SECONDS
    max_payment_attempts: int = MAX_PAYMENT_ATTEMPTS
    retry_base_delay: float = 0.25
    retry_max_delay: float = 8.0
    worker_batch_size: int = 25
    worker_poll_interval: float = 1.5
    feature_flags: dict[str, bool] = field(default_factory=dict)

    def flag(self, name: str, default: bool = False) -> bool:
        """Return a feature flag, falling back to ``default`` when unset."""
        return self.feature_flags.get(name, default)

    def redacted(self) -> dict[str, Any]:
        """Dump the settings for logging with the API key masked out."""
        data = {
            "database_url": _mask_credentials(self.database_url),
            "payment_base_url": self.payment_base_url,
            "payment_api_key": "***" + self.payment_api_key[-4:]
            if self.payment_api_key
            else "",
            "default_currency": self.default_currency,
            "payment_timeout": self.payment_timeout,
            "max_payment_attempts": self.max_payment_attempts,
            "worker_batch_size": self.worker_batch_size,
        }
        return data


def _mask_credentials(url: str) -> str:
    """Strip ``user:password@`` out of a database URL before it is logged."""
    if "@" not in url:
        return url
    scheme, _, rest = url.partition("://")
    _, _, host = rest.rpartition("@")
    return f"{scheme}://***@{host}" if scheme else f"***@{host}"


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number, got {raw!r}") from exc


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from exc


def _parse_flags(raw: str) -> dict[str, bool]:
    """Parse ``a=on,b=off`` style flag strings into a dictionary."""
    flags: dict[str, bool] = {}
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        name, _, value = chunk.partition("=")
        flags[name.strip()] = value.strip().lower() in ("1", "on", "true", "yes")
    return flags


def load_settings(environ: dict[str, str] | None = None) -> Settings:
    """Build a :class:`Settings` from the process environment.

    ``DATABASE_URL`` and ``PAYMENT_API_KEY`` are mandatory; everything else
    falls back to a documented default so a developer can boot the service
    with two variables set.
    """
    env = dict(os.environ if environ is None else environ)
    missing = [k for k in ("DATABASE_URL", "PAYMENT_API_KEY") if not env.get(k)]
    if missing:
        raise ConfigError("missing required environment: " + ", ".join(missing))
    os.environ.update(env)
    return Settings(
        database_url=env["DATABASE_URL"],
        payment_base_url=env.get("PAYMENT_BASE_URL", "https://payments.internal/v2"),
        payment_api_key=env["PAYMENT_API_KEY"],
        default_currency=env.get("DEFAULT_CURRENCY", DEFAULT_CURRENCY),
        payment_timeout=_env_float("PAYMENT_TIMEOUT_SECONDS", PAYMENT_TIMEOUT_SECONDS),
        max_payment_attempts=_env_int("MAX_PAYMENT_ATTEMPTS", MAX_PAYMENT_ATTEMPTS),
        retry_base_delay=_env_float("RETRY_BASE_DELAY", 0.25),
        retry_max_delay=_env_float("RETRY_MAX_DELAY", 8.0),
        worker_batch_size=_env_int("WORKER_BATCH_SIZE", 25),
        worker_poll_interval=_env_float("WORKER_POLL_INTERVAL", 1.5),
        feature_flags=_parse_flags(env.get("FEATURE_FLAGS", "")),
    )
