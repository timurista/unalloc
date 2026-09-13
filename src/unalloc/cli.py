"""unalloc command line interface.

Three commands, one question each:
  report      how much of our AI spend is unattributed?
  labels      which cost objects should we fix first?
  reconcile   does the ledger agree with the invoice?
"""

from __future__ import annotations

import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import typer

from unalloc import __version__
from unalloc.core.attribute import attribute, unallocated_rows
from unalloc.core.models import CostRow, Invoice
from unalloc.core.normalize import dimensions
from unalloc.core.reconcile import DEFAULT_TOLERANCE_PCT, reconcile
from unalloc.render import table as render
from unalloc.sources import REGISTRY

app = typer.Typer(
    add_completion=False,
    help="Find the AI spend nobody owns. Joins Kubernetes allocation data with "
    "LLM provider bills and reports what is unattributed.",
)

FIXTURE_DIR = Path(__file__).resolve().parents[2] / "examples" / "fixtures"

FIXTURE_FILES = {
    "opencost": "opencost_allocation.json",
    "litellm": "litellm_spend.json",
    "openai": "openai_costs.json",
    "anthropic": "anthropic_cost_report.json",
}

ENV_URL = {
    "opencost": "UNALLOC_OPENCOST_URL",
    "litellm": "UNALLOC_LITELLM_URL",
    "openai": "UNALLOC_OPENAI_URL",
    "anthropic": "UNALLOC_ANTHROPIC_URL",
}

ENV_TOKEN = {
    "litellm": "UNALLOC_LITELLM_KEY",
    "openai": "OPENAI_ADMIN_KEY",
    "anthropic": "ANTHROPIC_ADMIN_KEY",
}


def _window(days: int) -> tuple[datetime, datetime]:
    end = datetime.now(tz=UTC)
    return end - timedelta(days=days), end


def _load(
    sources: list[str],
    *,
    fixtures: bool,
    days: int,
    fixture_dir: Path,
) -> list[CostRow]:
    start, end = _window(days)
    rows: list[CostRow] = []
    for name in sources:
        if name not in REGISTRY:
            raise typer.BadParameter(
                f"unknown source '{name}'. Known: {', '.join(sorted(REGISTRY))}"
            )
        adapter = REGISTRY[name](
            base_url=os.environ.get(ENV_URL.get(name, "")) or None,
            token=os.environ.get(ENV_TOKEN.get(name, "")) or None,
        )
        if fixtures:
            path = fixture_dir / FIXTURE_FILES[name]
            if not path.exists():
                typer.secho(f"skipping {name}: no fixture at {path}", fg="yellow")
                continue
            rows.extend(adapter.from_fixture(path))
        else:
            try:
                rows.extend(adapter.fetch(start, end))
            except Exception as exc:
                typer.secho(f"{name}: {exc}", fg="red", err=True)
    return rows


SourcesOpt = typer.Option(
    ["opencost", "litellm", "openai", "anthropic"],
    "--source",
    "-s",
    help="Source to pull. Repeat the flag for several.",
)
FixturesOpt = typer.Option(
    False, "--fixtures", help="Run against bundled sample payloads. No infra needed."
)
DaysOpt = typer.Option(30, "--days", "-d", help="Lookback window in days.")
DimensionOpt = typer.Option(
    "team", "--dimension", "-D", help="Label key to attribute on."
)
FallbackOpt = typer.Option(
    [],
    "--fallback",
    "-f",
    help="Second-choice label keys, tried in order before a row is called "
    "unallocated. Keep this short and defensible.",
)
JsonOpt = typer.Option(False, "--json", help="Emit JSON instead of tables.")
FixtureDirOpt = typer.Option(FIXTURE_DIR, "--fixture-dir", help="Where fixtures live.")


@app.command()
def report(
    source: list[str] = SourcesOpt,
    dimension: str = DimensionOpt,
    fallback: list[str] = FallbackOpt,
    days: int = DaysOpt,
    fixtures: bool = FixturesOpt,
    fixture_dir: Path = FixtureDirOpt,
    top: int = typer.Option(20, "--top", help="Rows to show."),
    as_json: bool = JsonOpt,
) -> None:
    """Attribute joined spend and report the unallocated share."""
    rows = _load(source, fixtures=fixtures, days=days, fixture_dir=fixture_dir)
    if not rows:
        typer.secho("No cost rows loaded. Try --fixtures.", fg="yellow")
        raise typer.Exit(code=1)

    result = attribute(rows, dimension, fallback_dimensions=tuple(fallback))
    if as_json:
        typer.echo(
            render.to_json(
                {
                    "dimension": result.dimension,
                    "total_usd": result.total_usd,
                    "unallocated_usd": result.unallocated_usd,
                    "unallocated_pct": result.unallocated_pct,
                    "by_source": result.by_source,
                    "buckets": result.buckets,
                }
            )
        )
    else:
        render.render_report(result, top=top)


@app.command()
def labels(
    source: list[str] = SourcesOpt,
    dimension: str = DimensionOpt,
    days: int = DaysOpt,
    fixtures: bool = FixturesOpt,
    fixture_dir: Path = FixtureDirOpt,
    limit: int = typer.Option(20, "--limit", help="Rows in the backlog."),
    available: bool = typer.Option(
        False, "--available", help="List label keys present in the data instead."
    ),
) -> None:
    """Show the labeling backlog: unattributed cost objects, priciest first."""
    rows = _load(source, fixtures=fixtures, days=days, fixture_dir=fixture_dir)
    if available:
        for key, count in dimensions(rows).items():
            typer.echo(f"{count:>6}  {key}")
        return
    render.render_unallocated(unallocated_rows(rows, dimension, limit=limit), dimension)


@app.command(name="reconcile")
def reconcile_cmd(
    source: list[str] = SourcesOpt,
    days: int = DaysOpt,
    fixtures: bool = FixturesOpt,
    fixture_dir: Path = FixtureDirOpt,
    invoice: list[str] = typer.Option(
        [],
        "--invoice",
        "-i",
        help="Billed total per source as SOURCE=AMOUNT, e.g. openai=4210.55. "
        "Repeat the flag per source.",
    ),
    tolerance: float = typer.Option(
        float(DEFAULT_TOLERANCE_PCT),
        "--tolerance",
        help="Percent delta treated as rounding rather than a real gap.",
    ),
) -> None:
    """Check the ledger against what each provider actually billed."""
    rows = _load(source, fixtures=fixtures, days=days, fixture_dir=fixture_dir)
    start, end = _window(days)

    invoices: list[Invoice] = []
    for entry in invoice:
        if "=" not in entry:
            raise typer.BadParameter(f"expected SOURCE=AMOUNT, got '{entry}'")
        name, _, amount = entry.partition("=")
        invoices.append(
            Invoice(
                source=name.strip(),
                amount_usd=Decimal(amount.strip()),
                start=start,
                end=end,
            )
        )

    render.render_reconciliation(
        reconcile(rows, invoices, tolerance_pct=Decimal(str(tolerance)))
    )


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
