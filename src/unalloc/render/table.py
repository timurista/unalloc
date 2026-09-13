"""Rendering. The unallocated percentage is the thing people screenshot, so it
gets a banner rather than a row buried in a table.
"""

from __future__ import annotations

import dataclasses
import json
from decimal import Decimal
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from unalloc.core.attribute import AttributionReport
from unalloc.core.models import UNALLOCATED, CostRow
from unalloc.core.reconcile import ReconciliationReport

console = Console()


def usd(amount: Decimal) -> str:
    return f"${amount:,.2f}"


def _severity(pct: Decimal) -> str:
    if pct >= 25:
        return "bold red"
    if pct >= 10:
        return "bold yellow"
    return "bold green"


def render_report(report: AttributionReport, *, top: int = 20) -> None:
    style = _severity(report.unallocated_pct)
    console.print(
        Panel(
            Text.assemble(
                (f"{report.unallocated_pct}%", style),
                (" of ", "dim"),
                (usd(report.total_usd), "bold"),
                (" AI spend is unattributed", "dim"),
                ("\n", ""),
                (f"{usd(report.unallocated_usd)} has no '{report.dimension}' label", "dim"),
                *(
                    [
                        ("\n", ""),
                        (
                            f"{usd(report.fallback_usd)} attributed only via fallback "
                            f"({', '.join(report.fallback_dimensions)})",
                            "yellow",
                        ),
                    ]
                    if report.fallback_usd
                    else []
                ),
            ),
            title="unalloc",
            border_style=style.split()[-1],
        )
    )

    table = Table(title=f"Spend by {report.dimension}", header_style="bold")
    table.add_column(report.dimension)
    table.add_column("Cost", justify="right")
    table.add_column("Share", justify="right")
    table.add_column("Sources")
    table.add_column("Rows", justify="right")

    for bucket in report.buckets[:top]:
        share = (
            (bucket.amount_usd / report.total_usd * 100).quantize(Decimal("0.1"))
            if report.total_usd
            else Decimal("0")
        )
        label = (
            Text(UNALLOCATED, style="bold red")
            if bucket.is_unallocated
            else Text(bucket.key)
        )
        table.add_row(
            label,
            usd(bucket.amount_usd),
            f"{share}%",
            ", ".join(bucket.by_source),
            str(bucket.row_count),
        )
    console.print(table)

    sources = Table(title="By source", header_style="bold")
    sources.add_column("Source")
    sources.add_column("Cost", justify="right")
    for source, amount in report.by_source.items():
        sources.add_row(source, usd(amount))
    sources.add_row(Text("total", style="bold"), Text(usd(report.total_usd), style="bold"))
    console.print(sources)


def render_unallocated(rows: list[CostRow], dimension: str) -> None:
    table = Table(
        title=f"Labeling backlog — rows missing '{dimension}', by cost",
        header_style="bold",
    )
    table.add_column("#", justify="right", style="dim")
    table.add_column("Source")
    table.add_column("Cost object")
    table.add_column("Cost", justify="right")
    table.add_column("Labels it does have", style="dim")

    for index, row in enumerate(rows, start=1):
        available = ", ".join(sorted(row.labels)) or "none"
        table.add_row(str(index), row.source, row.name, usd(row.amount_usd), available)
    console.print(table)
    if not rows:
        console.print("[green]Nothing unallocated. Ship it.[/green]")


def render_reconciliation(report: ReconciliationReport) -> None:
    table = Table(title="Ledger vs invoice", header_style="bold")
    table.add_column("Source")
    table.add_column("Ledger", justify="right")
    table.add_column("Invoice", justify="right")
    table.add_column("Delta", justify="right")
    table.add_column("Status")

    palette = {"matched": "green", "over": "red", "under": "yellow", "no-invoice": "dim"}
    for item in report.per_source:
        status = item.status(report.tolerance_pct)
        delta = item.delta_usd
        table.add_row(
            item.source,
            usd(item.ledger_usd),
            usd(item.invoice_usd) if item.invoice_usd is not None else "—",
            f"{usd(delta)} ({item.delta_pct}%)" if delta is not None else "—",
            Text(status, style=palette[status]),
        )
    console.print(table)
    if not report.clean:
        console.print(
            "[yellow]The ledger does not tie out. Attribution built on this "
            "is not defensible yet.[/yellow]"
        )


def to_json(payload: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, Decimal):
            # String, not float: JSON floats would reintroduce the precision
            # loss the whole ledger is built to avoid.
            return str(value)
        if hasattr(value, "isoformat"):
            return value.isoformat()
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            return dataclasses.asdict(value)
        raise TypeError(type(value))

    return json.dumps(payload, indent=2, default=default)
