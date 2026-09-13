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
