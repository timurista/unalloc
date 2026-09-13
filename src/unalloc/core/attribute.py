"""Attribution: group a mixed ledger on one dimension and expose the gap.

Everything here is deliberately boring. The interesting claim unalloc makes is
not the grouping, it is that infra rows and LLM API rows can sit in the same
grouping at all.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal

from unalloc.core.models import UNALLOCATED, CostRow

ZERO = Decimal("0")


@dataclass(frozen=True, slots=True)
class Bucket:
    """Spend for one value of the attribution dimension."""

    key: str
    amount_usd: Decimal
    by_source: dict[str, Decimal] = field(default_factory=dict)
    row_count: int = 0

    @property
    def is_unallocated(self) -> bool:
        return self.key == UNALLOCATED


@dataclass(frozen=True, slots=True)
class AttributionReport:
    dimension: str
    total_usd: Decimal
    buckets: list[Bucket]
    by_source: dict[str, Decimal]
    unallocated_usd: Decimal
    row_count: int
    fallback_dimensions: tuple[str, ...] = ()
    fallback_usd: Decimal = ZERO
    """Spend that counts as allocated only because a fallback key matched.
    Reported separately so a fallback onto something that is not an owner
    (a chart name, a namespace shared by teams) cannot hide inside the
    headline percentage."""

    @property
    def unallocated_pct(self) -> Decimal:
        """The headline number. Zero total means zero gap, not a crash."""
        if self.total_usd == ZERO:
            return ZERO
        return (self.unallocated_usd / self.total_usd * 100).quantize(Decimal("0.1"))

    @property
    def allocated_usd(self) -> Decimal:
        return self.total_usd - self.unallocated_usd


def attribute(
    rows: Iterable[CostRow],
    dimension: str,
    *,
    fallback_dimensions: Sequence[str] = (),
) -> AttributionReport:
    """Group `rows` by `dimension`, bucketing label-less spend as unallocated.

    `fallback_dimensions` lets you accept a second-best key before giving up on
    a row: attributing on `team`, falling back to `namespace`, is usually more
    honest than reporting 60% unallocated when the data is merely inconsistent.
    Fallback hits are still counted as allocated, so keep the list short and
    defensible.
    """
    rows = list(rows)
    totals: dict[str, Decimal] = {}
    per_source: dict[str, dict[str, Decimal]] = {}
    counts: dict[str, int] = {}
    by_source: dict[str, Decimal] = {}
    total = ZERO
    via_fallback = ZERO

    for row in rows:
        key = row.label(dimension)
        if key is None:
            for alt in fallback_dimensions:
                key = row.label(alt)
                if key is not None:
                    via_fallback += row.amount_usd
                    break
        key = key or UNALLOCATED

        totals[key] = totals.get(key, ZERO) + row.amount_usd
        counts[key] = counts.get(key, 0) + 1
        per_source.setdefault(key, {})
        per_source[key][row.source] = per_source[key].get(row.source, ZERO) + row.amount_usd
        by_source[row.source] = by_source.get(row.source, ZERO) + row.amount_usd
        total += row.amount_usd

    buckets = [
        Bucket(
            key=key,
            amount_usd=amount,
            by_source=dict(sorted(per_source[key].items())),
            row_count=counts[key],
        )
        for key, amount in totals.items()
    ]
    # Largest spend first, but the unallocated bucket always sorts last so it
    # reads as the punchline rather than getting lost mid-table.
    buckets.sort(key=lambda b: (b.is_unallocated, -b.amount_usd))

    return AttributionReport(
        dimension=dimension,
        total_usd=total,
        buckets=buckets,
        by_source=dict(sorted(by_source.items())),
        unallocated_usd=totals.get(UNALLOCATED, ZERO),
        row_count=len(rows),
        fallback_dimensions=tuple(fallback_dimensions),
        fallback_usd=via_fallback,
    )


def unallocated_rows(
    rows: Iterable[CostRow],
    dimension: str,
    *,
    fallback_dimensions: Sequence[str] = (),
    limit: int = 20,
) -> list[CostRow]:
    """The rows that have no value for `dimension`, most expensive first.

    This is the actionable output: a labeling backlog sorted by dollars. It
    honours the same fallbacks as `attribute`, so the backlog always sums to
    the unallocated bucket of the matching report.
    """
    keys = (dimension, *fallback_dimensions)
    missing = [row for row in rows if all(row.label(key) is None for key in keys)]
    missing.sort(key=lambda r: r.amount_usd, reverse=True)
    return missing[:limit]
