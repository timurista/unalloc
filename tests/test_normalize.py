from __future__ import annotations

from unalloc.core.normalize import canonical_key, normalize_labels


def test_canonical_key_strips_provider_noise():
    assert canonical_key("label_costCenter") == "cost_center"
    assert canonical_key("app.kubernetes.io/name") == "name"
    # canonical_key normalizes shape only; the alias table lives in normalize_labels
    assert canonical_key("  Team_ID ") == "team_id"


def test_aliases_collapse_spellings_onto_one_dimension():
    labels = normalize_labels({"team_id": "search", "environment": "prod"})
    assert labels == {"team": "search", "environment": "prod"}


def test_explicit_key_wins_over_alias():
    labels = normalize_labels({"team": "platform", "owner": "search"})
    assert labels["team"] == "platform"


def test_empty_values_are_dropped_not_stored_as_blank():
    assert normalize_labels({"team": "  ", "project": None, "env": "prod"}) == {
        "environment": "prod"
    }
