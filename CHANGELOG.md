# Changelog

## Unreleased

### Added
- `case_studies/gpu_validation`: benchmark client, fake server, remote setup script, analysis and evidence renderer for validating the metering results with real vLLM on a GPU, plus a DigitalOcean runbook. First run: H100, vLLM 0.29.0, Qwen2.5-7B-Instruct, 6,241 requests, 0 errors.
- Blog addendum with the verified GPU specs, results and bring-up/teardown evidence.
- Paper §9, "Validation on a datacenter GPU", with the H100 results table and figure; abstract, methodology table and limitations updated to match.

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
