"""Anthropic organization cost report adapter.

Shape mirrors the OpenAI adapter closely: time buckets containing per-workspace
cost results. Workspace is the attribution dimension available here.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unalloc.core.models import CostRow
from unalloc.core.normalize import normalize_labels
from unalloc.sources.base import Source, money, parse_ts

ANTHROPIC_VERSION = "2023-06-01"


class AnthropicSource(Source):
    name = "anthropic"
    default_base_url = "https://api.anthropic.com/v1"

    def _headers(self) -> dict[str, str]:
        # The Admin API authenticates with x-api-key, not a Bearer token.
        headers = {"Accept": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if self.token:
            headers["x-api-key"] = self.token
        return headers

    def fetch(self, start: datetime, end: datetime) -> list[CostRow]:
        payload = self._get_pages(
            "/organizations/cost_report",
            params={
                "starting_at": start.date().isoformat(),
                "ending_at": end.date().isoformat(),
                "group_by[]": "workspace_id",
                "limit": 180,
            },
        )
        return self.parse(payload)

    def parse(self, payload: Any) -> list[CostRow]:
        buckets = payload.get("data", []) if isinstance(payload, dict) else payload
        if not isinstance(buckets, list):
            return []
        rows: list[CostRow] = []
        for bucket in buckets:
            if not isinstance(bucket, dict):
                continue
            start = parse_ts(bucket.get("starting_at"))
            end = parse_ts(bucket.get("ending_at"), default=start)
            for result in bucket.get("results") or []:
                if not isinstance(result, dict):
                    continue
                rows.append(self._row(result, start, end))
        return rows

    def _row(self, result: dict[str, Any], start: datetime, end: datetime) -> CostRow:
        raw_labels = {
            "workspace_id": result.get("workspace_id"),
            "workspace": result.get("workspace_name"),
            "description": result.get("description"),
            "provider": "anthropic",
        }
        return CostRow(
            source=self.name,
            name=str(result.get("description") or result.get("workspace_id") or "anthropic-usage"),
            amount_usd=money(result.get("amount")),
            start=start,
            end=end,
            labels=normalize_labels(raw_labels, self.aliases),
            meta={"currency": result.get("currency"), "token_type": result.get("token_type")},
        )
