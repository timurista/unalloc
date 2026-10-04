"""A report is checkable only if it can be rebuilt, or shown not to be.

Each test changes one thing between the original run and the replay: the
bytes behind a path, a retained file, the recorded result, or a source that
never loaded. `verify` has to tell those cases apart rather than reproduce a
stale answer or bless a result it cannot rebuild.
"""

from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest
from typer.testing import CliRunner

from unalloc.cli import FIXTURE_DIR, FIXTURE_FILES, app
from unalloc.core.provenance import canonical, conservation_errors, digest

runner = CliRunner()


def _run(tmp_path: Path, *extra: str) -> tuple[Path, Path, dict]:
    kept = tmp_path / "kept"
    out = runner.invoke(
        app,
        ["report", "--fixtures", "-D", "team", "--json", "--keep-inputs", str(kept), *extra],
    )
    assert out.exit_code == 0, out.output
    result = tmp_path / "result.json"
    result.write_text(out.stdout)
    return result, kept, json.loads(out.stdout)


def _verify(result: Path, kept: Path):
    return runner.invoke(app, ["verify", str(result), "--inputs", str(kept)])


def test_report_records_every_input_by_digest(tmp_path):
    _, kept, body = _run(tmp_path)
    prov = body["provenance"]
    assert prov["mode"] == "fixtures"
    assert prov["window"] is None
    assert [i["source"] for i in prov["inputs"]] == list(FIXTURE_FILES)
    for item in prov["inputs"]:
        retained = (kept / FIXTURE_FILES[item["source"]]).read_bytes()
        bundled = (FIXTURE_DIR / FIXTURE_FILES[item["source"]]).read_bytes()
        assert retained == bundled
        assert item["sha256"] == digest(retained)
        assert item["status"] == "loaded" and item["rows"] > 0


def test_unchanged_inputs_reproduce_the_result(tmp_path):
    result, kept, body = _run(tmp_path, "-f", "namespace")
    out = _verify(result, kept)
    assert out.exit_code == 0, out.output
    assert f"{body['unallocated_pct']}% unallocated" in out.stdout


def test_same_path_new_bytes_is_detected_not_reproduced(tmp_path):
    # The case a path-only record cannot catch: the file is still there under
    # the same name, but it no longer says what the original run read.
    result, kept, _ = _run(tmp_path)
    path = kept / FIXTURE_FILES["opencost"]
    payload = json.loads(path.read_text())
    window = payload["data"][0]
    name = next(iter(window))
    window[name]["totalCost"] = float(window[name]["totalCost"]) + 1
    path.write_text(json.dumps(payload))

    out = _verify(result, kept)
    assert out.exit_code == 3
    assert "opencost: input changed" in out.stdout


def test_missing_retained_input_cannot_reconstruct(tmp_path):
    result, kept, _ = _run(tmp_path)
    (kept / FIXTURE_FILES["anthropic"]).unlink()
    out = _verify(result, kept)
    assert out.exit_code == 3
    assert "anthropic: retained input missing" in out.stdout


def test_changed_result_with_same_inputs_is_flagged(tmp_path):
    # Inputs are byte-identical, so the difference has to come from the rule
    # side. Simulate an older run whose rules put spend in a different bucket.
    result, kept, body = _run(tmp_path)
    body["provenance"]["result_sha256"] = digest(canonical({"older": "rules"}))
    body["provenance"]["unalloc_version"] = "0.0.1"
    result.write_text(json.dumps(body))

    out = _verify(result, kept)
    assert out.exit_code == 4
    assert "inputs match but the result differs" in out.stdout
    assert "recorded 0.0.1" in out.stdout


def test_result_that_counts_a_dollar_twice_is_rejected(tmp_path):
    result, kept, body = _run(tmp_path)
    first = body["buckets"][0]
    first["amount_usd"] = str(Decimal(first["amount_usd"]) * 2)
    result.write_text(json.dumps(body))

    out = _verify(result, kept)
    assert out.exit_code == 5
    assert "not conserved: buckets sum" in out.stdout


def test_missing_source_is_recorded_and_not_silently_replayed(tmp_path):
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    for name in ("opencost", "litellm", "openai"):
        (partial_dir / FIXTURE_FILES[name]).write_bytes(
            (FIXTURE_DIR / FIXTURE_FILES[name]).read_bytes()
        )
    kept = tmp_path / "kept"
    out = runner.invoke(
        app,
        [
            "report", "--fixtures", "--fixture-dir", str(partial_dir), "-D", "team",
            "--json", "--keep-inputs", str(kept),
        ],
    )
    assert out.exit_code == 0
    assert "partial ledger: 3 of 4 sources loaded (anthropic missing)" in out.stderr
    body = json.loads(out.stdout)
    anthropic = body["provenance"]["inputs"][-1]
    assert anthropic["status"] == "missing" and anthropic["sha256"] is None

    # A file that appears later for the missing source must not leak into the
    # replay: the original result never included it.
    (kept / FIXTURE_FILES["anthropic"]).write_bytes(
        (FIXTURE_DIR / FIXTURE_FILES["anthropic"]).read_bytes()
    )
    result = tmp_path / "result.json"
    result.write_text(out.stdout)
    replay = _verify(result, kept)
    assert replay.exit_code == 0, replay.output
    assert "anthropic: missing in the original run" in replay.stdout


def test_result_without_provenance_is_refused(tmp_path):
    legacy = tmp_path / "legacy.json"
    legacy.write_text(json.dumps({"dimension": "team"}))
    out = runner.invoke(app, ["verify", str(legacy), "--inputs", str(tmp_path)])
    assert out.exit_code == 1


@pytest.mark.parametrize(
    ("field", "message"),
    [("unallocated_usd", "headline says"), ("total_usd", "sources sum")],
)
def test_conservation_checks_name_the_broken_total(tmp_path, field, message):
    _, _, body = _run(tmp_path)
    body[field] = str(Decimal(body[field]) + 1)
    assert any(message in line for line in conservation_errors(body))
