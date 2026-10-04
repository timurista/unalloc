"""Provenance: enough to rebuild a result later, or to know that you can't.

A report is only checkable if someone can say which bytes, which rules and
which tool revision produced it. A source path is not enough: a fixture or an
export can be rewritten at the same path, and a live billing API answers a
re-run with today's data, not last month's. So every input is recorded by
digest, sources that failed or were absent are recorded as such, and the
result itself gets a digest that a replay must match.

A digest identifies bytes. It says nothing about whether a meter was accurate
or an attribution rule was fair, and it can only rebuild a result when the
bytes it names were retained somewhere.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from unalloc.core.models import UNALLOCATED

LOADED = "loaded"
MISSING = "missing"
FAILED = "failed"


@dataclass(frozen=True, slots=True)
class SourceInput:
    """What one source contributed to a run, including nothing at all."""

    source: str
    status: str
    """`loaded`, `missing` (no input to read) or `failed` (the fetch raised)."""

    sha256: str | None = None
    """Digest of the exact payload bytes parsed, or None when nothing loaded."""

    bytes: int | None = None
    rows: int = 0
    detail: str = ""

    @property
    def loaded(self) -> bool:
        return self.status == LOADED


def _default(value: Any) -> Any:
    if isinstance(value, Decimal):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.asdict(value)
    raise TypeError(type(value))


def canonical(value: Any) -> bytes:
    """Stable bytes for a JSON-able value: sorted keys, no whitespace,
    Decimals as strings. Two equal results always hash the same."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), default=_default
    ).encode()


def digest(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def partial(inputs: list[SourceInput]) -> list[SourceInput]:
    """Inputs that did not load. Non-empty means the ledger is incomplete and
    every percentage computed from it covers the loaded sources only."""
    return [item for item in inputs if not item.loaded]


def conservation_errors(result: dict[str, Any]) -> list[str]:
    """Check a recorded result against its own totals.

    Buckets and per-source subtotals must each sum to the total: a dollar can
    be assigned once, never twice and never dropped. This holds for any
    attribution policy, so it can be checked without trusting the tool that
    wrote the file.
    """
    errors: list[str] = []
    total = Decimal(result["total_usd"])
    buckets = sum((Decimal(b["amount_usd"]) for b in result["buckets"]), Decimal("0"))
    sources = sum((Decimal(v) for v in result["by_source"].values()), Decimal("0"))
    if buckets != total:
        errors.append(f"buckets sum to {buckets}, total is {total}")
    if sources != total:
        errors.append(f"sources sum to {sources}, total is {total}")
    unallocated = sum(
        (Decimal(b["amount_usd"]) for b in result["buckets"] if b["key"] == UNALLOCATED),
        Decimal("0"),
    )
    if unallocated != Decimal(result["unallocated_usd"]):
        errors.append(
            f"unallocated bucket is {unallocated}, headline says {result['unallocated_usd']}"
        )
    return errors
