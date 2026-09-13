from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal

from unalloc.core.models import CostRow, Invoice
from unalloc.core.reconcile import reconcile

NOW = datetime(2026, 8, 1, tzinfo=UTC)


def row(source, amount):
    return CostRow(
        source=source, name="x", amount_usd=Decimal(amount), start=NOW, end=NOW, labels={}
    )


def invoice(source, amount):
    return Invoice(source=source, amount_usd=Decimal(amount), start=NOW, end=NOW)


def test_small_delta_is_treated_as_rounding():
    report = reconcile([row("openai", "1000.00")], [invoice("openai", "1004.00")])
    assert report.per_source[0].status() == "matched"
    assert report.clean


def test_large_delta_is_flagged_with_direction():
    report = reconcile([row("openai", "1200.00")], [invoice("openai", "1000.00")])
    item = report.per_source[0]
    assert item.status() == "over"
    assert item.delta_usd == Decimal("200.00")
    assert not report.clean


def test_source_without_an_invoice_is_not_a_failure():
    report = reconcile([row("litellm", "500")], [])
    assert report.per_source[0].status() == "no-invoice"
    assert report.clean
