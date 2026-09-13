"""LiteLLM adapter.

LiteLLM's proxy already attributes spend per virtual key, team and tag, which
makes it the single highest-signal LLM source. It is also the only one that can
give you request-level granularity, so where a team runs LiteLLM this adapter
should be preferred over the direct provider adapters to avoid double counting.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unalloc.core.models import CostRow
from unalloc.core.normalize import normalize_labels
from unalloc.sources.base import Source, money, parse_ts


class LiteLLMSource(Source):
    name = "litellm"

    def fetch(self, start: datetime, end: datetime) -> list[CostRow]:
        payload = self._get(
            "/spend/logs",
            params={
                "start_date": start.date().isoformat(),
                "end_date": end.date().isoformat(),
            },
        )
        return self.parse(payload)

    def parse(self, payload: Any) -> list[CostRow]:
        records = payload.get("data", payload) if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return []
        return [self._row(rec) for rec in records if isinstance(rec, dict)]

    def _row(self, rec: dict[str, Any]) -> CostRow:
        metadata = rec.get("metadata") or {}
        raw_labels: dict[str, Any] = dict(metadata) if isinstance(metadata, dict) else {}

        # Tags arrive as a list; index them so a `team:platform` style tag
        # becomes a real dimension instead of an opaque string.
        for tag in rec.get("request_tags") or []:
            text = str(tag)
            if ":" in text:
                key, _, value = text.partition(":")
                raw_labels.setdefault(key, value)
            else:
                raw_labels.setdefault(text, "true")

        for field in ("team_id", "user", "end_user", "api_key_alias", "model"):
            if rec.get(field):
                raw_labels.setdefault(field, rec[field])

        tokens = rec.get("total_tokens")
        if tokens is None:
            tokens = (rec.get("prompt_tokens") or 0) + (rec.get("completion_tokens") or 0)

        return CostRow(
            source=self.name,
            name=str(rec.get("model") or rec.get("api_key_alias") or "litellm-request"),
            amount_usd=money(rec.get("spend") if rec.get("spend") is not None else rec.get("cost")),
            start=parse_ts(rec.get("startTime") or rec.get("start_time")),
            end=parse_ts(rec.get("endTime") or rec.get("end_time")),
            labels=normalize_labels(raw_labels, self.aliases),
            quantity=money(tokens) or None,
            unit="tokens" if tokens else None,
            meta={
                "request_id": rec.get("request_id"),
                "prompt_tokens": rec.get("prompt_tokens"),
                "completion_tokens": rec.get("completion_tokens"),
                "cache_hit": rec.get("cache_hit"),
            },
        )
