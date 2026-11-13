"""Allocate synthetic environment-lifetime memory GB-seconds to requests.

    python3 -m case_studies.environment_lifetime
    python3 -m case_studies.environment_lifetime --out PATH
    python3 -m case_studies.environment_lifetime --check EXPECTED_JSON
    python3 -m case_studies.environment_lifetime --drop-request a-2 \\
        --check case_studies/environment_lifetime/fixture_output_missing_a-2.json

Offline synthetic arithmetic only: no network, no AWS ingestion, no prices.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from case_studies.environment_lifetime.lifetime import (
    FixtureError,
    compute,
    dumps_result,
    load_fixture,
    without_requests,
)

HERE = Path(__file__).resolve().parent
DEFAULT_FIXTURE = HERE / "fixture.json"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exact synthetic environment-lifetime GB-second allocation."
    )
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE)
    parser.add_argument(
        "--drop-request",
        action="append",
        default=[],
        metavar="ID",
        help="omit a request to simulate missing telemetry (repeatable)",
    )
    parser.add_argument("--out", type=Path, help="write result JSON here instead of stdout")
    parser.add_argument(
        "--check",
        type=Path,
        metavar="EXPECTED",
        help="exit 1 unless the result equals this file byte for byte",
    )
    args = parser.parse_args(argv)

    try:
        fixture = load_fixture(args.fixture)
        if args.drop_request:
            fixture = without_requests(fixture, args.drop_request)
    except FixtureError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    text = dumps_result(compute(fixture))

    if args.check is not None:
        expected = args.check.read_text(encoding="utf-8")
        if expected != text:
            print(f"MISMATCH: result differs from {args.check}", file=sys.stderr)
            return 1
        print(f"OK: result matches {args.check}")
        return 0
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}")
        return 0
    sys.stdout.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
