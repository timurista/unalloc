# Changelog

## 0.2.2 — 2026-09-21

### Fixed
- Paper Table 3: the time-share entries at the 8 and 16 requests/s loads printed the raw shares while the caption promised overhead redistributed; they are 4.7% and 5.3%. The 12–14 point headline result is unaffected — the figure and the divergence metric already redistributed.
- Paper: the configured load is now stated as the arrival rate of session-initial requests, distinguished throughout from completed throughput (3.7–26.9 requests/s), which includes the follow-up turns of multi-turn sessions.
- Paper: "none waited in the queue" is now scoped to what was measured — no queued request in any half-second telemetry sample.
- Paper: the CPU-versus-GPU gap is reported as a difference between two experiments rather than an effect with a measured cause; the fixed-overhead explanation is labelled a hypothesis, since model, software, hardware and workload all differ.
- Paper: the abstract marks the 66% and 61% distribution results as a constructed scenario built on synthetic allocations.
- Paper and README: OpenCost's OpenAI plugin and inference accounting are acknowledged, and the novelty claim narrowed to the implementation and the reproducible characterization of the failure modes. FOCUS added as related work.
- Paper: Shapley is described as a reference for a specified cost function rather than a ground truth for rental bills; the Pope, DistServe, ABACUS and Cost-Governed RAG comparisons are bounded to what those works support; CPU collective fractions are no longer called an upper bound for NVLink GPUs.
- Bibliography: official proceedings links, retrieval dates for mutable documentation, Shapley page range, and the ABACUS preprint dated to its 22 December 2024 submission.

### Added
- `bench.py --repeats n`: independent runs per load level, each with a fresh seed, with the load levels interleaved so session drift does not land on one of them. `analyze.py` groups repeats by rate and reports the median and range across them.
- Within-run stability: both meters recomputed in ten-second windows of each run, with quartiles and a bootstrap confidence interval for the mean, warm-up window excluded. The published single runs now carry error bars in the GPU figure and a confidence interval in §9; none of the intervals approaches zero.
- `tests/test_paper_numbers.py`: the GPU results table, its prose ranges and its scope claims are checked against `metrics.json`, so a stale table fails the build. This is the test that would have caught the redistribution error above.
- `paper/REFERENCE-AUDIT.md`: the 23-reference source audit behind the corrections.

## 0.2.1 — 2026-09-14

### Changed
- Paper prepared for arXiv: abstract leads with the attribution failures at the seams between ledgers and the H100 result; metering results are framed as disagreement between rules rather than over-charging, since none is a ground truth; new Related Work section (JouleShare, LLMVisor, PrefixShield, Cost-Governed RAG and others); defect catalogue moved to Appendix A; use of AI tools disclosed.
- `paper/build.py` fails on Typst that compiles but renders wrong (`~`, `;` after a citation, unescaped dollar amounts).

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
