"""OpenAI organization costs adapter.

Uses the org-level costs endpoint, which buckets spend by time and (optionally)
project. Project is the only attribution dimension the billing API exposes, so
if an org runs everything under one project this adapter will correctly report
near-100% unallocated. That is not a bug in unalloc; it is the finding.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unalloc.core.models import CostRow
from unalloc.core.normalize import normalize_labels
from unalloc.sources.base import Source, money, parse_ts


class OpenAISource(Source):
    name = "openai"
    default_base_url = "https://api.openai.com/v1"

    def fetch(self, start: datetime, end: datetime) -> list[CostRow]:
        payload = self._get_pages(
            "/organization/costs",
            params={
                "start_time": int(start.timestamp()),
                "end_time": int(end.timestamp()),
                "bucket_width": "1d",
                "group_by": "project_id",
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
            start = parse_ts(bucket.get("start_time"))
            end = parse_ts(bucket.get("end_time"), default=start)
            for result in bucket.get("results") or []:
                if not isinstance(result, dict):
                    continue
                rows.append(self._row(result, start, end))
        return rows

    def _row(self, result: dict[str, Any], start: datetime, end: datetime) -> CostRow:
        amount = result.get("amount") or {}
        raw_labels = {
            "project_id": result.get("project_id"),
            "project": result.get("project_name"),
            "line_item": result.get("line_item"),
            "provider": "openai",
        }
        return CostRow(
            source=self.name,
            name=str(result.get("line_item") or result.get("project_id") or "openai-usage"),
            amount_usd=money(amount.get("value") if isinstance(amount, dict) else amount),
            start=start,
            end=end,
            labels=normalize_labels(raw_labels, self.aliases),
            meta={"currency": (amount or {}).get("currency") if isinstance(amount, dict) else None},
        )
