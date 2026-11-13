# Environment-lifetime memory allocation (synthetic)

This is a small example that runs offline and uses only the standard library. It splits each
environment's **lifetime memory GB-seconds** among the requests that ran in that
environment. Any time not covered by a request becomes a residual.

> **Scope.** The example does synthetic arithmetic on a hand-written fixture. It is
> **not** validated AWS ingestion, **not** an AWS bill, and **not** a CPU billing model.
> It contains no prices and no dollar or savings figures. The case study is not wired into the `unalloc` CLI or its sources,
> and it changes no existing product behavior.

## Why environment lifetime and not request duration

The AWS Lambda docs (checked 2026-11-13) describe two facts this example models:

- [Lambda web functions](https://docs.aws.amazon.com/lambda/latest/dg/web-functions.html):
  memory is billed over the life of the environment.
- [Environment lifecycle](https://docs.aws.amazon.com/lambda/latest/dg/web-functions-env-lifecycle.html):
  an environment has no guaranteed lifetime.

So the sum of request durations is not the quantity that memory accrues over. Concurrent
requests share one environment's memory, and memory also accrues while no request is
running. This example only shows the arithmetic for those two effects. It reads no AWS data.

## Model

For each environment, the fixture gives:

- An observed lifetime `[lifetime_start_s, lifetime_end_s]`, which must have positive length.
- A synthetic constant memory capacity `memory_gb`, which must be greater than 0.
- `lifetime_gb_seconds = memory_gb × (lifetime_end_s − lifetime_start_s)`.

For each request, the fixture gives:

- An `environment` that must exist.
- A span `[start_s, end_s]` that must lie inside that environment's lifetime.

The example integrates each environment independently:

1. Cut the lifetime into non-overlapping slices at every lifetime boundary and every request
   boundary.
2. Check how many requests cover each slice:
   - If `n ≥ 1` requests cover the slice, the slice counts as busy. Each covering request
     receives `memory_gb × width / n`.
   - If no request covers the slice, it counts as uncovered. Its GB-seconds go to the
     **residual**.
3. Never share a slice across environments. Two requests on different environments get
   separate allocations, even when their wall-clock spans overlap.

### Reported quantities

The example reports each quantity per environment and as a total (except `overlap_seconds`, which is per-environment only):

| Field | Meaning |
|---|---|
| `raw_request_seconds` | Sum of request durations. Counts overlapping time more than once. |
| `union_busy_seconds` | Per-environment union of request spans. The total sums these environment-seconds, not a global wall-clock union. |
| `overlap_seconds` | `raw_request_seconds − union_busy_seconds`. |
| `uncovered_seconds` | Lifetime seconds that no request covers. |
| `lifetime_gb_seconds` | `memory_gb × lifetime seconds`. |
| `allocated_gb_seconds` | Sum of per-request allocations, equal to `memory_gb × union_busy_seconds`. |
| `residual_gb_seconds` | `memory_gb × uncovered_seconds`. |
| `reconciliation` | Checks that `allocated + residual == lifetime` exactly. The `difference` field must be `"0"`. |

The output also lists `allocated_gb_seconds` for each request.

### Residual: unobserved unless the fixture claims completeness

By default, the example labels the residual `unobserved`. A gap in request telemetry might
mean the environment sat idle, or it might mean a request went unrecorded.

The residual is labelled `idle_by_complete_synthetic_fixture` only when the fixture sets both
`"synthetic": true` and `"telemetry_complete": true`. The bundled fixture sets both, because
it is hand-written and lists every request by construction. When you run with
`--drop-request`, the example always sets `telemetry_complete` to false. In that case, the
residual is labelled `unobserved`.

### Zero-length requests

A request with `start_s == end_s` is accepted if it lies inside its environment's
lifetime, including the lifetime's endpoints. A zero-length request has the following
properties:

- It covers no slice.
- It receives exactly `0` GB-seconds.
- It adds `0` to `raw_request_seconds`.
- It is flagged `"zero_duration": true`.

### Exactness and validation

All arithmetic uses `fractions.Fraction`. The example accepts numbers in these forms:

- Decimal strings, such as `"1.5"`.
- Integers.
- JSON numbers, which the example parses as `Decimal`, so `0.1` stays exactly 1/10.
- In-memory `Decimal` or `Fraction` values.

The output writes each value as an exact string. A value that terminates in base 10 is written
as a decimal, such as `"28.75"`. Any other value is written as `"numerator/denominator"`, such
as `"155/6"`.

The example rejects the following inputs with a `FixtureError` that names the offending field:

- Binary floats, booleans, `NaN`, `sNaN`, `Infinity`, and the JSON `NaN` or `Infinity`
  literals.
- Malformed number strings.
- Negative times, and a memory capacity of 0 or less.
- Lifetimes where `end <= start`, and requests where `end < start`.
- Requests that fall outside their environment's lifetime.
- Unknown environments.
- Duplicate environment IDs, duplicate request IDs (checked across all environments), and
  duplicate JSON keys.
- IDs that are empty, not a string, or padded with whitespace.
- Missing or unknown fields, and a wrong `schema` value.

## Bundled fixture

The fixture is [`fixture.json`](fixture.json). All values are synthetic.

| Environment | Memory (GB) | Lifetime (s) | Requests | What it shows |
|---|---|---|---|---|
| `env-a` | 1.5 | [0, 100] | `a-1` [10,40], `a-2` [20,50], `a-3` [30,35], `a-4` [60,60], `a-5` [70,90] | Three-way overlap, a zero-length request, and gaps |
| `env-b` | 0.5 | [5, 65] | `b-1` [5,65], `b-2` [15,25], `b-3` [20,30] | Up to 3 concurrent requests. The thirds don't terminate as decimals. |
| `env-c` | 3 | [0, 30] | none | The whole lifetime is residual |

The bundled fixture produces these results, from [`fixture_output.json`](fixture_output.json):

| Environment | Raw request s | Union busy s | Lifetime GB-s | Allocated GB-s | Residual GB-s |
|---|---|---|---|---|---|
| `env-a` | 85 | 60 | 150 | 90 | 60 |
| `env-b` | 80 | 60 | 30 | 30 | 0 |
| `env-c` | 0 | 0 | 90 | 0 | 90 |
| **total** | **165** | **120** | **270** | **120** | **150** |

The per-request allocations are:

- `a-1` 28.75, `a-2` 28.75, `a-3` 2.5, `a-4` 0, `a-5` 30.
- `b-1` 155/6, `b-2` 25/12, `b-3` 25/12.

### Missing telemetry

The file [`fixture_output_missing_a-2.json`](fixture_output_missing_a-2.json) shows the result
when request `a-2` is removed:

- The residual for `env-a` grows from 60 to **75** GB-s. The extra 15 GB-s comes from
  [40, 50], the 10 seconds when only `a-2` was running, at 1.5 GB.
- The slices that `a-2` shared now go to `a-1` (41.25) and `a-3` (3.75).
- The lifetime GB-seconds don't change, and the reconciliation is still exact.
- The residual is labelled `unobserved`.

## Reproduce

Run these commands from the repository root. They need Python 3.9 or later, no third-party
packages, and no network access.

```sh
# Print the result JSON for the bundled fixture.
python3 -m case_studies.environment_lifetime

# Check that the stored outputs match the current code byte for byte. The command exits 1 on a mismatch.
python3 -m case_studies.environment_lifetime --check case_studies/environment_lifetime/fixture_output.json
python3 -m case_studies.environment_lifetime --drop-request a-2 \
    --check case_studies/environment_lifetime/fixture_output_missing_a-2.json

# Regenerate the stored outputs.
python3 -m case_studies.environment_lifetime --out case_studies/environment_lifetime/fixture_output.json
python3 -m case_studies.environment_lifetime --drop-request a-2 \
    --out case_studies/environment_lifetime/fixture_output_missing_a-2.json

# Run the tests with unittest from the standard library. pytest is not required.
python3 -m unittest discover -s case_studies/environment_lifetime -t . -v
```

The repository's `pytest` configuration only collects tests from `tests/`, so this suite
runs only when you invoke the `unittest` command above.

## Files

- `lifetime.py`: parsing, validation, exact integration, and reconciliation.
- `__main__.py`: the command-line interface (`--fixture`, `--drop-request`, `--out`, and
  `--check`).
- `fixture.json`: the synthetic input.
- `fixture_output.json` and `fixture_output_missing_a-2.json`: the stored outputs.
- `test_lifetime.py`: the `unittest` suite.
