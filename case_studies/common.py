"""Shared plumbing for the case studies.

Three things live here, and only these:

1. Pricing assumptions, stated once so every study and the paper agree.
2. Emitters that build payloads in the exact shapes the real providers return,
   so the studies exercise unalloc's adapters rather than a side door.
3. A thin wrapper that runs unalloc's attribution on a results directory and
   returns a JSON-serialisable summary.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

from unalloc.core.attribute import attribute, unallocated_rows
from unalloc.core.models import CostRow
from unalloc.sources import REGISTRY

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# --- pricing assumptions ------------------------------------------------------
# Deliberately round, public-list-price-order-of-magnitude numbers. The studies
# compare attribution *shares*, which do not depend on the absolute rate.

GPU_HOUR_USD = 3.00
"""One datacenter GPU (H100-class), on-demand, per hour."""

CPU_CORE_HOUR_USD = 0.031611
"""OpenCost's default CPU price."""

RAM_GB_HOUR_USD = 0.004237
"""OpenCost's default RAM price."""

WINDOW_START = datetime(2026, 8, 1, tzinfo=UTC)
WINDOW_END = datetime(2026, 9, 1, tzinfo=UTC)

# unalloc's CLI looks these filenames up inside --fixture-dir.
FIXTURE_FILES = {
    "opencost": "opencost_allocation.json",
    "litellm": "litellm_spend.json",
    "openai": "openai_costs.json",
    "anthropic": "anthropic_cost_report.json",
}


def iso(ts: datetime) -> str:
    return ts.astimezone(UTC).isoformat().replace("+00:00", "Z")


# --- payload emitters -----------------------------------------------------------


def opencost_allocation(
    name: str,
    *,
    namespace: str,
    controller: str,
    controller_kind: str = "deployment",
    labels: Mapping[str, str] | None = None,
    cpu_cost: float = 0.0,
    ram_cost: float = 0.0,
    gpu_cost: float = 0.0,
    shared_cost: float = 0.0,
    gpu_hours: float | None = None,
    model: str | None = None,
    efficiency: float | None = None,
    cluster: str = "case-study",
    start: datetime = WINDOW_START,
    end: datetime = WINDOW_END,
    include_total: bool = True,
) -> dict[str, Any]:
    """One entry of an OpenCost /allocation window, as OpenCost serialises it."""
    props: dict[str, Any] = {
        "namespace": namespace,
        "controller": controller,
        "controllerKind": controller_kind,
        "cluster": cluster,
        "labels": dict(labels or {}),
    }
    if model:
        props["modelName"] = model
    alloc: dict[str, Any] = {
        "name": name,
        "start": iso(start),
        "end": iso(end),
        "properties": props,
        "cpuCost": round(cpu_cost, 6),
        "ramCost": round(ram_cost, 6),
        "gpuCost": round(gpu_cost, 6),
        "sharedCost": round(shared_cost, 6),
    }
    if gpu_hours is not None:
        alloc["gpuHours"] = round(gpu_hours, 6)
    if efficiency is not None:
        alloc["totalEfficiency"] = round(efficiency, 4)
    if include_total:
        alloc["totalCost"] = round(cpu_cost + ram_cost + gpu_cost + shared_cost, 6)
    return alloc


def opencost_payload(allocations: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"code": 200, "data": [{a["name"]: a for a in allocations}]}


def litellm_record(
    request_id: str,
    *,
    model: str,
    spend: float,
    prompt_tokens: int,
    completion_tokens: int,
    start: datetime,
    duration_s: float = 1.0,
    team_id: str | None = None,
    api_key_alias: str | None = None,
    tags: Sequence[str] = (),
    metadata: Mapping[str, Any] | None = None,
    cache_hit: bool = False,
) -> dict[str, Any]:
    """One LiteLLM SpendLogs row, as /spend/logs returns it."""
    record: dict[str, Any] = {
        "request_id": request_id,
        "model": model,
        "spend": f"{spend:.6f}",
        "startTime": iso(start),
        "endTime": iso(start + timedelta(seconds=duration_s)),
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
        "request_tags": list(tags),
        "metadata": dict(metadata or {}),
        "cache_hit": cache_hit,
    }
    if team_id:
        record["team_id"] = team_id
    if api_key_alias:
        record["api_key_alias"] = api_key_alias
    return record


def litellm_payload(records: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return {"data": list(records)}


def openai_payload(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """OpenAI /organization/costs: one bucket spanning the window."""
    return {
        "object": "page",
        "data": [
            {
                "object": "bucket",
                "start_time": int(WINDOW_START.timestamp()),
                "end_time": int(WINDOW_END.timestamp()),
                "results": list(results),
            }
        ],
        "has_more": False,
    }


def anthropic_payload(results: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Anthropic /organizations/cost_report: one bucket spanning the window."""
    return {
        "data": [
            {
                "starting_at": iso(WINDOW_START),
                "ending_at": iso(WINDOW_END),
                "results": list(results),
            }
        ],
        "has_more": False,
    }


# --- IO -------------------------------------------------------------------------


def _default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, datetime):
        return iso(value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value))


def dump_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_default) + "\n")
    return path


def results_dir(study: str) -> Path:
    path = RESULTS / study
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_payloads(outdir: Path, **payloads: dict[str, Any]) -> list[Path]:
    """Write payloads under the filenames `unalloc --fixture-dir` expects."""
    return [dump_json(outdir / FIXTURE_FILES[name], body) for name, body in payloads.items()]


# --- running unalloc --------------------------------------------------------------


def load_rows(outdir: Path, sources: Sequence[str]) -> list[CostRow]:
    """Parse payloads through unalloc's real adapters."""
    rows: list[CostRow] = []
    for name in sources:
        rows.extend(REGISTRY[name]().from_fixture(outdir / FIXTURE_FILES[name]))
    return rows


def summarize(
    rows: Sequence[CostRow],
    dimension: str,
    *,
    fallback: Sequence[str] = (),
    backlog: int = 10,
) -> dict[str, Any]:
    """Attribution result as plain data: what the paper tables are built from."""
    report = attribute(rows, dimension, fallback_dimensions=tuple(fallback))
    return {
        "dimension": dimension,
        "fallback": list(fallback),
        "total_usd": report.total_usd,
        "unallocated_usd": report.unallocated_usd,
        "unallocated_pct": report.unallocated_pct,
        "fallback_usd": report.fallback_usd,
        "by_source": report.by_source,
        "buckets": [
            {
                "key": b.key,
                "amount_usd": b.amount_usd,
                "by_source": b.by_source,
                "rows": b.row_count,
            }
            for b in report.buckets
        ],
        "backlog": [
            {"source": r.source, "name": r.name, "amount_usd": r.amount_usd}
            for r in unallocated_rows(
                rows, dimension, fallback_dimensions=tuple(fallback), limit=backlog
            )
        ],
    }


def usd(value: Decimal | float) -> str:
    return f"${Decimal(str(value)):,.2f}"
