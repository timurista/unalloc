"""OpenCost adapter.

Reads the /allocation endpoint, which returns one or more time windows, each a
map of allocation name -> allocation object. Since 1.121.0 OpenCost also
exposes per-model inference costs for vLLM/llm-d deployments; those arrive
through the same allocation shape, so they land in the ledger for free and get
tagged with `unalloc_layer=inference` where detectable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from unalloc.core.models import CostRow
from unalloc.core.normalize import normalize_labels
from unalloc.sources.base import Source, money, parse_ts


class OpenCostSource(Source):
    name = "opencost"

    def fetch(self, start: datetime, end: datetime) -> list[CostRow]:
        payload = self._get(
            "/allocation",
            params={
                "window": f"{start.date().isoformat()},{end.date().isoformat()}",
                "aggregate": "namespace,controller,label:team",
                "accumulate": "true",
                "includeIdle": "true",
            },
        )
        return self.parse(payload)

    def parse(self, payload: Any) -> list[CostRow]:
        rows: list[CostRow] = []
        for window in _windows(payload):
            for name, alloc in window.items():
                if not isinstance(alloc, dict):
                    continue
                rows.append(self._row(name, alloc))
        return rows

    def _row(self, name: str, alloc: dict[str, Any]) -> CostRow:
        props = alloc.get("properties") or {}
        raw_labels: dict[str, Any] = {}
        raw_labels.update(props.get("labels") or {})
        raw_labels.update(props.get("annotations") or {})
        # Structural identity is a legitimate fallback dimension, so promote it
        # into labels rather than hiding it in meta.
        for key in ("namespace", "controller", "controllerKind", "cluster", "node"):
            if props.get(key):
                raw_labels.setdefault(key, props[key])

        model = _inference_model(alloc, props)
        if model:
            raw_labels.setdefault("model", model)
            raw_labels.setdefault("unalloc_layer", "inference")

        total = alloc.get("totalCost")
        if total is None:
            total = sum(
                float(alloc.get(field) or 0)
                for field in (
                    "cpuCost",
                    "ramCost",
                    "gpuCost",
                    "pvCost",
                    "networkCost",
                    "loadBalancerCost",
                    "sharedCost",
                    "externalCost",
                )
            )

        return CostRow(
            source=self.name,
            name=alloc.get("name") or name,
            amount_usd=money(total),
            start=parse_ts(alloc.get("start") or alloc.get("window", {}).get("start")),
            end=parse_ts(alloc.get("end") or alloc.get("window", {}).get("end")),
            labels=normalize_labels(raw_labels, self.aliases),
            quantity=money(alloc.get("gpuHours")) or None,
            unit="gpu-hours" if alloc.get("gpuHours") else None,
            meta={
                "cpuCost": alloc.get("cpuCost"),
                "ramCost": alloc.get("ramCost"),
                "gpuCost": alloc.get("gpuCost"),
                "sharedCost": alloc.get("sharedCost"),
                "efficiency": alloc.get("totalEfficiency"),
            },
        )


def _windows(payload: Any) -> list[dict[str, Any]]:
    """OpenCost has shipped several response envelopes. Accept all of them."""
    data = payload.get("data") if isinstance(payload, dict) else payload
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    if isinstance(data, list):
        return [window for window in data if isinstance(window, dict)]
    return []


def _inference_model(alloc: dict[str, Any], props: dict[str, Any]) -> str | None:
    """Best-effort detection of the 1.121.0+ inference metadata.

    The field naming here is the least stable part of this adapter. If it moves
    upstream, only this function changes.
    """
    for container in (alloc, props):
        for key in ("model", "modelName", "llmModel", "inferenceModel"):
            value = container.get(key)
            if value:
                return str(value)
    return None
