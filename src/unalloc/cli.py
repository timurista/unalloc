"""unalloc command line interface.

Four commands, one question each:
  report      how much of our AI spend is unattributed?
  labels      which cost objects should we fix first?
  reconcile   does the ledger agree with the invoice?
  verify      can an earlier report be rebuilt from what was kept?
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path

import typer

from unalloc import __version__
from unalloc.core.attribute import AttributionReport, attribute, unallocated_rows
from unalloc.core.models import CostRow, Invoice
from unalloc.core.normalize import DEFAULT_ALIASES, dimensions
from unalloc.core.provenance import (
    FAILED,
    LOADED,
    MISSING,
    SourceInput,
    canonical,
    conservation_errors,
    digest,
    partial,
)
from unalloc.core.reconcile import DEFAULT_TOLERANCE_PCT, reconcile
from unalloc.render import table as render
from unalloc.sources import REGISTRY

app = typer.Typer(
    add_completion=False,
    help="Find the AI spend nobody owns. Joins Kubernetes allocation data with "
    "LLM provider bills and reports what is unattributed.",
)

# Fixtures ship inside the package so `--fixtures` works from an installed
# wheel, not just from a clone. A clean-venv install is the only way to
# catch this, so `make check-wheel` does exactly that.
FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"

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
) -> tuple[list[CostRow], list[SourceInput], dict[str, bytes]]:
    """Load every requested source, recording what each one contributed.

    Returns the rows, one SourceInput per requested source (including the
    ones that failed or had no input), and the exact payload bytes that were
    parsed, keyed by source, so a caller can retain them for a later replay.
    """
    start, end = _window(days)
    rows: list[CostRow] = []
    inputs: list[SourceInput] = []
    payloads: dict[str, bytes] = {}
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
                typer.secho(f"skipping {name}: no fixture at {path}", fg="yellow", err=True)
                inputs.append(SourceInput(name, MISSING, detail=f"no input at {path.name}"))
                continue
            data = path.read_bytes()
            parsed = adapter.parse(json.loads(data))
        else:
            try:
                payload = adapter.fetch_payload(start, end)
            except Exception as exc:
                typer.secho(f"{name}: {exc}", fg="red", err=True)
                inputs.append(SourceInput(name, FAILED, detail=str(exc)[:200]))
                continue
            # Digest the canonical form, which is also what --keep-inputs
            # writes, so a retained file always matches its recorded digest.
            data = canonical(payload)
            parsed = adapter.parse(payload)
        rows.extend(parsed)
        payloads[name] = data
        inputs.append(
            SourceInput(name, LOADED, sha256=digest(data), bytes=len(data), rows=len(parsed))
        )
    return rows, inputs, payloads


def _result(result: AttributionReport) -> dict[str, object]:
    """The report body. Its canonical digest is what a replay must match."""
    return {
        "dimension": result.dimension,
        "total_usd": result.total_usd,
        "unallocated_usd": result.unallocated_usd,
        "unallocated_pct": result.unallocated_pct,
        "fallback_dimensions": result.fallback_dimensions,
        "fallback_usd": result.fallback_usd,
        "by_source": result.by_source,
        "buckets": result.buckets,
    }


def _warn_partial(inputs: list[SourceInput]) -> None:
    gaps = partial(inputs)
    if gaps:
        loaded = len(inputs) - len(gaps)
        names = ", ".join(f"{item.source} {item.status}" for item in gaps)
        typer.secho(
            f"partial ledger: {loaded} of {len(inputs)} sources loaded ({names}). "
            "Percentages cover the loaded sources only.",
            fg="yellow",
            err=True,
        )


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
    budget: float | None = typer.Option(
        None,
        "--budget",
        help="Exit with code 2 when the unallocated share exceeds this percent. "
        "Lets CI fail a deploy that ships unlabeled spend.",
    ),
    keep_inputs: Path | None = typer.Option(
        None,
        "--keep-inputs",
        help="Write the exact payloads this run parsed into DIR, named like "
        "fixtures, so `unalloc verify` can rebuild the result later.",
    ),
) -> None:
    """Attribute joined spend and report the unallocated share."""
    rows, inputs, payloads = _load(
        source, fixtures=fixtures, days=days, fixture_dir=fixture_dir
    )
    if not rows:
        typer.secho("No cost rows loaded. Try --fixtures.", fg="yellow")
        raise typer.Exit(code=1)

    if keep_inputs is not None:
        keep_inputs.mkdir(parents=True, exist_ok=True)
        for name, data in payloads.items():
            (keep_inputs / FIXTURE_FILES[name]).write_bytes(data)

    result = attribute(rows, dimension, fallback_dimensions=tuple(fallback))
    if as_json:
        body = _result(result)
        start, end = _window(days)
        body["provenance"] = {
            "unalloc_version": __version__,
            "generated_at": datetime.now(tz=UTC),
            "mode": "fixtures" if fixtures else "live",
            # Fixtures carry their own dates; --days does not filter them.
            "window": None if fixtures else {"start": start, "end": end},
            "aliases_sha256": digest(canonical(DEFAULT_ALIASES)),
            "inputs": inputs,
            "result_sha256": digest(canonical(_result(result))),
        }
        typer.echo(render.to_json(body))
    else:
        render.render_report(result, top=top)
    _warn_partial(inputs)

    if budget is not None and result.unallocated_pct > Decimal(str(budget)):
        typer.secho(
            f"unallocated {result.unallocated_pct}% exceeds budget {budget}%",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=2)


@app.command()
def labels(
    source: list[str] = SourcesOpt,
    dimension: str = DimensionOpt,
    fallback: list[str] = FallbackOpt,
    days: int = DaysOpt,
    fixtures: bool = FixturesOpt,
    fixture_dir: Path = FixtureDirOpt,
    limit: int = typer.Option(20, "--limit", help="Rows in the backlog."),
    available: bool = typer.Option(
        False, "--available", help="List label keys present in the data instead."
    ),
) -> None:
    """Show the labeling backlog: unattributed cost objects, priciest first."""
    rows, inputs, _ = _load(source, fixtures=fixtures, days=days, fixture_dir=fixture_dir)
    _warn_partial(inputs)
    if not rows:
        # Without this, an unreachable source reads as "Nothing unallocated".
        typer.secho("No cost rows loaded. Try --fixtures.", fg="yellow")
        raise typer.Exit(code=1)
    if available:
        for key, count in dimensions(rows).items():
            typer.echo(f"{count:>6}  {key}")
        return
    backlog = unallocated_rows(
        rows, dimension, fallback_dimensions=tuple(fallback), limit=limit
    )
    render.render_unallocated(backlog, dimension)


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
    rows, inputs, _ = _load(source, fixtures=fixtures, days=days, fixture_dir=fixture_dir)
    _warn_partial(inputs)
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
def verify(
    result_file: Path = typer.Argument(
        ..., help="A `report --json` output that carries a provenance block."
    ),
    inputs_dir: Path = typer.Option(
        ...,
        "--inputs",
        help="Directory of retained payloads, e.g. one written by --keep-inputs.",
    ),
) -> None:
    """Rebuild an earlier report from retained inputs and say whether it matches.

    Exit 0: reproduced. Exit 3: the inputs needed are missing or changed, so
    the result cannot be rebuilt. Exit 4: the inputs match but the result
    does not, so a rule or the tool changed in between. Exit 5: the recorded
    result does not conserve its own total.
    """
    recorded = json.loads(result_file.read_text())
    prov = recorded.get("provenance")
    if not prov:
        typer.secho(
            f"{result_file} has no provenance block; it predates `unalloc verify` "
            "or was not written with --json.",
            fg="red",
            err=True,
        )
        raise typer.Exit(code=1)

    broken = conservation_errors(recorded)
    if broken:
        for line in broken:
            typer.secho(f"not conserved: {line}", fg="red")
        raise typer.Exit(code=5)

    problems: list[str] = []
    rows: list[CostRow] = []
    for item in prov["inputs"]:
        name = item["source"]
        if item["status"] != LOADED:
            typer.echo(f"{name}: {item['status']} in the original run, not replayed")
            continue
        path = inputs_dir / FIXTURE_FILES[name]
        if not path.exists():
            problems.append(f"{name}: retained input missing at {path}")
            continue
        data = path.read_bytes()
        found = digest(data)
        if found != item["sha256"]:
            problems.append(
                f"{name}: input changed (recorded {item['sha256'][:19]}, found {found[:19]})"
            )
            continue
        rows.extend(REGISTRY[name]().parse(json.loads(data)))

    if problems:
        for line in problems:
            typer.secho(line, fg="red")
        typer.secho("cannot reconstruct: the recorded inputs are not all available", fg="red")
        raise typer.Exit(code=3)

    rebuilt = attribute(
        rows,
        recorded["dimension"],
        fallback_dimensions=tuple(recorded["fallback_dimensions"]),
    )
    found = digest(canonical(_result(rebuilt)))
    if found == prov["result_sha256"]:
        typer.secho(
            f"reproduced: {rebuilt.unallocated_pct}% unallocated of "
            f"{render.usd(rebuilt.total_usd)}, result {found[:19]}",
            fg="green",
        )
        return

    typer.secho("inputs match but the result differs", fg="red")
    typer.echo(
        f"  unallocated: recorded {recorded['unallocated_pct']}%, "
        f"rebuilt {rebuilt.unallocated_pct}%"
    )
    if prov["unalloc_version"] != __version__:
        typer.echo(f"  unalloc version: recorded {prov['unalloc_version']}, running {__version__}")
    aliases = digest(canonical(DEFAULT_ALIASES))
    if prov["aliases_sha256"] != aliases:
        typer.echo("  label alias table changed since the original run")
    raise typer.Exit(code=4)


@app.command()
def version() -> None:
    """Print the installed version."""
    typer.echo(__version__)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
