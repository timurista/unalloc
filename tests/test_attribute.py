from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from unalloc.core.attribute import attribute, unallocated_rows
from unalloc.core.models import UNALLOCATED, CostRow

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def row(name, amount, labels, source="opencost"):
    return CostRow(
        source=source,
        name=name,
        amount_usd=Decimal(amount),
        start=NOW,
        end=NOW,
        labels=labels,
    )


def test_unlabeled_spend_lands_in_the_unallocated_bucket():
    report = attribute(
        [row("a", "100", {"team": "search"}), row("b", "300", {})], "team"
    )
    assert report.total_usd == Decimal("400")
    assert report.unallocated_usd == Decimal("300")
    assert report.unallocated_pct == Decimal("75.0")


def test_unallocated_bucket_always_sorts_last():
    report = attribute(
        [row("a", "1", {"team": "search"}), row("b", "9999", {})], "team"
    )
    assert report.buckets[-1].key == UNALLOCATED


def test_fallback_dimension_rescues_inconsistent_labels():
    rows = [row("a", "100", {"namespace": "search"})]
    assert attribute(rows, "team").unallocated_pct == Decimal("100.0")
    rescued = attribute(rows, "team", fallback_dimensions=("namespace",))
    assert rescued.unallocated_pct == Decimal("0")


def test_fallback_spend_is_reported_separately():
    rows = [row("a", "100", {"team": "search"}), row("b", "60", {"namespace": "shared"})]
    report = attribute(rows, "team", fallback_dimensions=("namespace",))
    assert report.unallocated_pct == Decimal("0")
    assert report.fallback_usd == Decimal("60")
    assert attribute(rows, "team").fallback_usd == Decimal("0")


def test_sources_are_joined_under_one_dimension():
    report = attribute(
        [
            row("pod", "100", {"team": "platform"}, source="opencost"),
            row("gpt-4o", "50", {"team": "platform"}, source="litellm"),
        ],
        "team",
    )
    bucket = report.buckets[0]
    assert bucket.by_source == {"litellm": Decimal("50"), "opencost": Decimal("100")}


def test_empty_ledger_does_not_divide_by_zero():
    assert attribute([], "team").unallocated_pct == Decimal("0")


def test_backlog_is_sorted_by_cost_descending():
    rows = [row("cheap", "5", {}), row("pricey", "900", {}), row("ok", "10", {"team": "x"})]
    backlog = unallocated_rows(rows, "team")
    assert [r.name for r in backlog] == ["pricey", "cheap"]
