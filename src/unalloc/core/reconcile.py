"""Reconcile the ledger against what the provider actually billed.

An attribution report that doesn't tie back to the invoice is a guess. This is
the step most tools skip, and it is the one that makes a FinOps team trust the
number.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from decimal import Decimal

from unalloc.core.models import CostRow, Invoice

ZERO = Decimal("0")

# Below this, a delta is rounding and metric-scrape jitter, not a real gap.
DEFAULT_TOLERANCE_PCT = Decimal("1.0")


@dataclass(frozen=True, slots=True)
class SourceReconciliation:
    source: str
    ledger_usd: Decimal
    invoice_usd: Decimal | None

    @property
    def delta_usd(self) -> Decimal | None:
        if self.invoice_usd is None:
            return None
        return self.ledger_usd - self.invoice_usd

    @property
    def delta_pct(self) -> Decimal | None:
        if self.invoice_usd is None or self.invoice_usd == ZERO:
            return None
        return ((self.ledger_usd - self.invoice_usd) / self.invoice_usd * 100).quantize(
            Decimal("0.1")
        )

    def status(self, tolerance_pct: Decimal = DEFAULT_TOLERANCE_PCT) -> str:
        """One of: matched, over, under, no-invoice."""
        if self.invoice_usd is None:
            return "no-invoice"
        pct = self.delta_pct
        if pct is None:
            return "no-invoice"
        if abs(pct) <= tolerance_pct:
            return "matched"
        return "over" if pct > 0 else "under"


@dataclass(frozen=True, slots=True)
class ReconciliationReport:
    per_source: list[SourceReconciliation]
    tolerance_pct: Decimal

    @property
    def ledger_total(self) -> Decimal:
        return sum((r.ledger_usd for r in self.per_source), ZERO)

    @property
    def invoice_total(self) -> Decimal:
        return sum(
            (r.invoice_usd for r in self.per_source if r.invoice_usd is not None), ZERO
        )

    @property
    def clean(self) -> bool:
        """True when every source with an invoice ties out within tolerance."""
        return all(
            r.status(self.tolerance_pct) in {"matched", "no-invoice"}
            for r in self.per_source
        )


def reconcile(
    rows: Iterable[CostRow],
    invoices: Iterable[Invoice],
    *,
    tolerance_pct: Decimal = DEFAULT_TOLERANCE_PCT,
) -> ReconciliationReport:
    ledger: dict[str, Decimal] = {}
    for row in rows:
        ledger[row.source] = ledger.get(row.source, ZERO) + row.amount_usd

    billed: dict[str, Decimal] = {}
    for invoice in invoices:
        billed[invoice.source] = billed.get(invoice.source, ZERO) + invoice.amount_usd

    sources = sorted(set(ledger) | set(billed))
    per_source = [
        SourceReconciliation(
            source=source,
            ledger_usd=ledger.get(source, ZERO),
            invoice_usd=billed.get(source),
        )
        for source in sources
    ]
    return ReconciliationReport(per_source=per_source, tolerance_pct=tolerance_pct)
