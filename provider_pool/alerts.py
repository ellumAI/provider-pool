"""Provider pool alert system.

Fires structured alert payloads when a slot changes health state:
  KEY_EXHAUSTED      — billing cap hit (24 h+ cooldown)
  KEY_RATE_LIMITED   — per-minute/day quota (short cooldown)
  KEY_AUTH_FAILED    — 401/403, key is dead
  KEY_UNHEALTHY      — repeated 5xx errors
  KEY_RECOVERED      — first success after a cooldown period
  QUOTA_WARNING      — accumulated token usage crossed a threshold
  ALL_KEYS_EXHAUSTED — no slots available at all

AlertBackend is an interface; compose multiple backends via
CompositeAlertBackend (log + webhook is the usual setup).
"""

from __future__ import annotations

import contextlib
import logging
from dataclasses import dataclass
from enum import Enum

import httpx

logger = logging.getLogger(__name__)


class SlotEvent(str, Enum):
    EXHAUSTED = "KEY_EXHAUSTED"
    RATE_LIMITED = "KEY_RATE_LIMITED"
    AUTH_FAILED = "KEY_AUTH_FAILED"
    UNHEALTHY = "KEY_UNHEALTHY"
    RECOVERED = "KEY_RECOVERED"
    QUOTA_WARNING = "QUOTA_WARNING"
    ALL_EXHAUSTED = "ALL_KEYS_EXHAUSTED"


@dataclass
class AlertPayload:
    event: SlotEvent
    key_id: str
    model: str | None
    message: str
    cooldown_seconds: float | None = None
    tokens_used: int | None = None

    def to_dict(self) -> dict:
        return {
            "event": self.event.value,
            "key_id": self.key_id,
            "model": self.model,
            "message": self.message,
            "cooldown_seconds": self.cooldown_seconds,
            "tokens_used": self.tokens_used,
        }


class AlertBackend:
    async def fire(self, payload: AlertPayload) -> None:
        raise NotImplementedError


class LogAlertBackend(AlertBackend):
    """Default backend — writes to the application log."""

    async def fire(self, payload: AlertPayload) -> None:
        level = (
            "error"
            if payload.event
            in (SlotEvent.EXHAUSTED, SlotEvent.AUTH_FAILED, SlotEvent.ALL_EXHAUSTED)
            else "warning"
        )
        msg = (
            f"[ProviderPool] {payload.event.value} | key={payload.key_id}"
            f" model={payload.model} | {payload.message}"
        )
        getattr(logger, level)(msg)


class WebhookAlertBackend(AlertBackend):
    """POST the alert payload as JSON to a configured URL (e.g. Slack, PagerDuty)."""

    def __init__(self, url: str, timeout: float = 5.0) -> None:
        self._url = url
        self._timeout = timeout

    async def fire(self, payload: AlertPayload) -> None:
        with contextlib.suppress(Exception):
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(self._url, json=payload.to_dict())
                if resp.status_code >= 400:
                    logger.warning(
                        f"Alert webhook returned {resp.status_code} for {payload.event}"
                    )


class CompositeAlertBackend(AlertBackend):
    """Fan-out to multiple backends; failures in one don't block others."""

    def __init__(self, backends: list[AlertBackend]) -> None:
        self._backends = backends

    async def fire(self, payload: AlertPayload) -> None:
        for backend in self._backends:
            with contextlib.suppress(Exception):
                await backend.fire(payload)


def build_alert_backend(webhook_url: str | None) -> AlertBackend:
    """Convenience factory: log-only by default, log+webhook if URL provided."""
    backends: list[AlertBackend] = [LogAlertBackend()]
    if webhook_url:
        backends.append(WebhookAlertBackend(webhook_url))
    return CompositeAlertBackend(backends) if len(backends) > 1 else backends[0]
