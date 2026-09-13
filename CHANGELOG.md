# Changelog

## 0.2.0 — 2026-09-13

Validated against five inference case studies; see `paper/unalloc-case-studies.pdf`.

### Fixed
- OpenAI and Anthropic adapters lost their default API URLs whenever the CLI passed an unset environment variable, so every live fetch failed.
- The Anthropic Admin API is now called with `x-api-key` and `anthropic-version` instead of a Bearer token.
- Provider cost APIs are paged with `has_more` / `next_page`; previously only the first page was read.
- OpenCost component costs are summed as `Decimal` when `totalCost` is absent, never through `float`.
- Stacked label prefixes are all stripped (`label_app_kubernetes_io_name` → `name`).
- Canonical-key collisions resolve by a fixed precedence instead of arrival order (explicit key, alias, rewritten key, path-derived key).
- OpenCost annotations no longer override labels with the same key.
- `unalloc labels` honours `--fallback`, and exits 1 when no rows load instead of printing "Nothing unallocated".

### Added
- `report --budget PCT` exits 2 when the unallocated share exceeds `PCT`, for CI gates.
- Reports show the spend attributed only through a fallback key (`fallback_usd` in `--json`).
- Case studies (`case_studies/`), a Typst paper (`paper/`), a ledger explorer, a companion notebook and a dev container with Postgres.

## 0.1.0

Initial release: OpenCost, LiteLLM, OpenAI and Anthropic adapters; `report`, `labels` and `reconcile`.
