"""The one data structure every cost source normalizes into.

The entire design of unalloc rests on this: an adapter's only job is to turn a
provider's payload into CostRows. Attribution, reconciliation and rendering all
operate on CostRows and know nothing about where they came from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Any

# Label key used to bucket spend that carries no value for the requested
# attribution dimension. This is the number the tool exists to produce.
UNALLOCATED = "<unallocated>"


@dataclass(frozen=True, slots=True)
class CostRow:
    """A single unit of spend, normalized across infra and LLM providers.

    Money is Decimal, never float. Cost attribution that disagrees with the
    invoice by a penny per row is how you lose an argument with finance.
    """

    source: str
    """Adapter that produced this row: opencost, litellm, openai, anthropic."""

    name: str
    """Human-readable identity of the cost object (pod, key, project, model)."""

    amount_usd: Decimal
    """Cost in USD for the window [start, end)."""

    start: datetime
    end: datetime

    labels: dict[str, str] = field(default_factory=dict)
    """Normalized, lowercased label keys -> values. The attribution dimension
    is looked up here."""

    quantity: Decimal | None = None
    """Optional usage amount (tokens, core-hours). Not used for attribution,
    but carried through so unit-economics math is possible later."""

    unit: str | None = None
    """Unit for `quantity`, e.g. "tokens", "core-hours", "gpu-hours"."""

    meta: dict[str, Any] = field(default_factory=dict)
    """Anything source-specific worth keeping for debugging. Never read by
    core logic."""

    def label(self, dimension: str) -> str | None:
        """Return the value for `dimension`, or None if this row is unallocated."""
        value = self.labels.get(dimension)
        if value is None:
            return None
        value = value.strip()
        return value or None


@dataclass(frozen=True, slots=True)
class Invoice:
    """What the provider says you owe, for reconciliation against the ledger."""

    source: str
    amount_usd: Decimal
    start: datetime
    end: datetime
    note: str = ""
