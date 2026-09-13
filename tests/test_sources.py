from __future__ import annotations

from decimal import Decimal
from pathlib import Path

import pytest

from unalloc.sources import REGISTRY

FIXTURES = Path(__file__).resolve().parents[1] / "src" / "unalloc" / "fixtures"

CASES = [
    ("opencost", "opencost_allocation.json", 5),
    ("litellm", "litellm_spend.json", 4),
    ("openai", "openai_costs.json", 3),
    ("anthropic", "anthropic_cost_report.json", 3),
]


@pytest.mark.parametrize(("name", "fixture", "expected_rows"), CASES)
def test_adapter_parses_its_fixture(name, fixture, expected_rows):
    rows = REGISTRY[name]().from_fixture(FIXTURES / fixture)
    assert len(rows) == expected_rows
    assert all(row.source == name for row in rows)
    assert all(isinstance(row.amount_usd, Decimal) for row in rows)
    assert sum((r.amount_usd for r in rows), Decimal("0")) > 0


def test_opencost_promotes_namespace_into_labels():
    rows = REGISTRY["opencost"]().from_fixture(FIXTURES / "opencost_allocation.json")
    by_name = {row.name: row for row in rows}
    assert by_name["search/api"].labels["namespace"] == "search"
    assert by_name["search/api"].labels["team"] == "search"


def test_opencost_camel_case_cost_center_is_normalized():
    rows = REGISTRY["opencost"]().from_fixture(FIXTURES / "opencost_allocation.json")
    vllm = next(r for r in rows if r.name == "inference/vllm-llama-70b")
    assert vllm.labels["cost_center"] == "CC-4471"
    assert vllm.labels["unalloc_layer"] == "inference"


def test_opencost_label_wins_over_same_key_annotation():
    payload = {
        "data": [
            {
                "pod": {
                    "name": "pod",
                    "properties": {
                        "namespace": "ns",
                        "labels": {"team": "search"},
                        "annotations": {"team": "someone-else"},
                    },
                    "totalCost": 1.0,
                }
            }
        ]
    }
    (row,) = REGISTRY["opencost"]().parse(payload)
    assert row.labels["team"] == "search"


def test_litellm_tags_become_dimensions():
    rows = REGISTRY["litellm"]().from_fixture(FIXTURES / "litellm_spend.json")
    rerank = next(r for r in rows if r.labels.get("feature") == "rerank")
    assert rerank.labels["team"] == "search"
    assert rerank.unit == "tokens"


def test_money_never_goes_through_float():
    rows = REGISTRY["litellm"]().from_fixture(FIXTURES / "litellm_spend.json")
    assert any(r.amount_usd == Decimal("5412.75") for r in rows)
