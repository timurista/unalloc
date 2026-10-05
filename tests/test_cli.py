from __future__ import annotations

import json

from typer.testing import CliRunner

from unalloc.cli import app

runner = CliRunner()


def test_budget_gate_fails_ci_when_unallocated_share_is_too_high():
    # Bundled fixtures are 73.5% unallocated on `team`.
    over = runner.invoke(app, ["report", "--fixtures", "-D", "team", "--json", "--budget", "50"])
    assert over.exit_code == 2
    assert "exceeds budget" in over.stderr

    under = runner.invoke(app, ["report", "--fixtures", "-D", "team", "--json", "--budget", "80"])
    assert under.exit_code == 0
    assert json.loads(under.stdout)["unallocated_pct"] == "73.5"


def test_labels_backlog_honours_fallbacks():
    plain = runner.invoke(app, ["labels", "--fixtures", "-D", "team", "--limit", "50"])
    rescued = runner.invoke(
        app, ["labels", "--fixtures", "-D", "team", "-f", "namespace", "--limit", "50"]
    )
    assert plain.exit_code == rescued.exit_code == 0
    # Three OpenCost rows lack `team`; two of them (the vLLM pod and the
    # embeddings job) carry a namespace, so only __idle__ stays on the backlog.
    # Count rows rather than match names: rich truncates to terminal width.
    assert plain.stdout.count("│ opencost") == 3
    assert rescued.stdout.count("│ opencost") == 1


def test_version_flag_matches_version_command():
    from unalloc import __version__

    for args in (["--version"], ["-V"], ["version"]):
        result = runner.invoke(app, args)
        assert result.exit_code == 0
        assert result.stdout.strip() == __version__


def test_reconcile_fixtures_uses_bundled_invoices():
    result = runner.invoke(app, ["reconcile", "--fixtures"])
    assert result.exit_code == 0
    # Anthropic ties out within tolerance; OpenAI billed more than its
    # costs endpoint returned, so the demo shows a real gap.
    assert "matched" in result.stdout
    assert "under" in result.stdout
    assert "does not tie out" in result.stdout


def test_explicit_invoice_replaces_bundled_invoices():
    result = runner.invoke(app, ["reconcile", "--fixtures", "-i", "openai=4031.36"])
    assert result.exit_code == 0
    assert "under" not in result.stdout
    assert "does not tie out" not in result.stdout


def test_fixture_sources_do_not_overlap():
    # The default demo pulls all four sources, so the gateway fixture must not
    # carry traffic that the direct provider fixtures also bill.
    from unalloc.cli import FIXTURE_DIR, FIXTURE_FILES

    gateway = json.loads((FIXTURE_DIR / FIXTURE_FILES["litellm"]).read_text())["data"]
    direct = ("openai/", "anthropic/")
    for record in gateway:
        model = record["model"]
        assert "/" in model and not model.startswith(direct), model
