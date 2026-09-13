"""Adapter contract.

An adapter does exactly two things: fetch a provider payload, and turn it into
CostRows. It must not aggregate, filter by cost, or make attribution decisions.
Keeping adapters dumb is what lets new providers be added in an afternoon.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import httpx

from unalloc.core.models import CostRow

DEFAULT_TIMEOUT = 30.0

# Hard stop for cursor pagination. A provider that keeps saying has_more is a
# bug on their side, and looping forever against a billing API is worse.
MAX_PAGES = 1000


class Source(ABC):
    """Base class for every cost source."""

    name: str = "unknown"

    default_base_url: str | None = None
    """Public API root, for providers that have one. An unset or empty
    `base_url` falls back to this, so a blank env var does not disable it."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        token: str | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        aliases: dict[str, str] | None = None,
    ) -> None:
        self.base_url = (base_url or self.default_base_url or "").rstrip("/")
        self.token = token
        self.timeout = timeout
        self.aliases = aliases or {}

    # --- to implement per provider -------------------------------------

    @abstractmethod
    def fetch(self, start: datetime, end: datetime) -> list[CostRow]:
        """Call the provider and return normalized rows."""

    @abstractmethod
    def parse(self, payload: Any) -> list[CostRow]:
        """Turn a raw provider payload into CostRows.

        Kept separate from `fetch` so the fixtures in examples/ are a real test
        of the parsing contract and the tool is demoable with zero infra.
        """

    # --- shared plumbing ------------------------------------------------

    def from_fixture(self, path: str | Path) -> list[CostRow]:
        with Path(path).open() as handle:
            return self.parse(json.load(handle))

    def _headers(self) -> dict[str, str]:
        """Request headers. Override when a provider does not take Bearer auth."""
        headers = {"Accept": "application/json"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return headers

    def _get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        if not self.base_url:
            raise ValueError(
                f"{self.name}: no base_url configured. Set UNALLOC_{self.name.upper()}_URL "
                f"or run with --fixtures to demo without infrastructure."
            )
        with httpx.Client(timeout=self.timeout) as client:
            response = client.get(
                f"{self.base_url}{path}", params=params, headers=self._headers()
            )
            response.raise_for_status()
            return response.json()

    def _get_pages(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        """Follow `has_more` / `next_page` cursors and concatenate `data`.

        OpenAI's and Anthropic's admin reporting APIs both page this way. A
        30-day window grouped by project easily exceeds one page, and silently
        reading only the first page under-reports spend.
        """
        data: list[Any] = []
        query = dict(params)
        for _ in range(MAX_PAGES):
            payload = self._get(path, params=query)
            if not isinstance(payload, dict):
                break
            data.extend(payload.get("data") or [])
            cursor = payload.get("next_page")
            if not payload.get("has_more") or not cursor:
                break
            query["page"] = cursor
        return {"data": data}


def money(value: Any) -> Decimal:
    """Coerce a provider's cost field to Decimal without ever going via float.

    Providers are inconsistent: OpenCost sends JSON floats, LiteLLM sends
    strings, some send None for zero. str() before Decimal keeps float noise
    (0.30000000000000004) out of the ledger.
    """
    if value is None:
        return Decimal("0")
    try:
        return Decimal(str(value))
    except (InvalidOperation, ValueError):
        return Decimal("0")


def parse_ts(value: Any, default: datetime | None = None) -> datetime:
    """Parse the assorted timestamp formats providers emit, as UTC-aware."""
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, tz=UTC)
    if isinstance(value, str) and value:
        text = value.replace("Z", "+00:00")
        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            pass
        else:
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return default or datetime.now(tz=UTC)
