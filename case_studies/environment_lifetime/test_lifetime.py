"""unittest suite for the synthetic environment-lifetime allocation.

python3 -m unittest discover -s case_studies/environment_lifetime -t . -v
"""

from __future__ import annotations

import contextlib
import copy
import io
import json
import random
import unittest
from decimal import Decimal
from fractions import Fraction
from pathlib import Path

from case_studies.environment_lifetime import __main__ as cli
from case_studies.environment_lifetime.lifetime import (
    RESIDUAL_IDLE,
    RESIDUAL_UNOBSERVED,
    SCHEMA,
    FixtureError,
    compute,
    dumps_result,
    exact,
    format_exact,
    load_fixture,
    loads_fixture,
    parse_fixture,
    without_requests,
)

HERE = Path(__file__).resolve().parent
FIXTURE = HERE / "fixture.json"
OUTPUT = HERE / "fixture_output.json"
OUTPUT_MISSING = HERE / "fixture_output_missing_a-2.json"


def env(id_, memory="1", start="0", end="100"):
    return {"id": id_, "memory_gb": memory, "lifetime_start_s": start, "lifetime_end_s": end}


def req(id_, environment, start, end):
    return {"id": id_, "environment": environment, "start_s": start, "end_s": end}


def doc(envs, reqs, *, synthetic=True, complete=True):
    return {
        "schema": SCHEMA,
        "synthetic": synthetic,
        "telemetry_complete": complete,
        "environments": envs,
        "requests": reqs,
    }


def run(data):
    return compute(parse_fixture(data))


def by_id(rows):
    return {row["id"]: row for row in rows}


def q(text):
    """Parse an exact output string back to a Fraction."""
    return Fraction(text)


class ExactFormatting(unittest.TestCase):
    def test_terminating_and_non_terminating(self):
        self.assertEqual(format_exact(Fraction(0)), "0")
        self.assertEqual(format_exact(Fraction(100)), "100")
        self.assertEqual(format_exact(Fraction(115, 4)), "28.75")
        self.assertEqual(format_exact(Fraction(1, 3)), "1/3")
        self.assertEqual(format_exact(Fraction(155, 6)), "155/6")
        for value in (Fraction(1, 3), Fraction(115, 4), Fraction(10**30 + 1, 2**40)):
            self.assertEqual(q(format_exact(value)), value)

    def test_decimal_inputs_are_exact(self):
        self.assertEqual(exact("0.1", "x") * 3, Fraction(3, 10))
        self.assertEqual(exact(Decimal("1.5"), "x"), Fraction(3, 2))
        self.assertEqual(exact(Fraction(1, 3), "x"), Fraction(1, 3))


class BundledFixture(unittest.TestCase):
    def setUp(self):
        self.result = compute(load_fixture(FIXTURE))
        self.envs = by_id(self.result["environments"])
        self.reqs = by_id(self.result["requests"])

    def test_committed_outputs_are_current(self):
        self.assertEqual(dumps_result(self.result), OUTPUT.read_text(encoding="utf-8"))
        missing = compute(without_requests(load_fixture(FIXTURE), ["a-2"]))
        self.assertEqual(dumps_result(missing), OUTPUT_MISSING.read_text(encoding="utf-8"))

    def test_hand_computed_env_a(self):
        a = self.envs["env-a"]
        self.assertEqual(a["lifetime_gb_seconds"], "150")
        self.assertEqual(a["raw_request_seconds"], "85")
        self.assertEqual(a["union_busy_seconds"], "60")
        self.assertEqual(a["overlap_seconds"], "25")
        self.assertEqual(a["uncovered_seconds"], "40")
        self.assertEqual(a["allocated_gb_seconds"], "90")
        self.assertEqual(a["residual_gb_seconds"], "60")
        self.assertEqual(a["max_concurrency"], 3)
        self.assertEqual(self.reqs["a-1"]["allocated_gb_seconds"], "28.75")
        self.assertEqual(self.reqs["a-2"]["allocated_gb_seconds"], "28.75")
        self.assertEqual(self.reqs["a-3"]["allocated_gb_seconds"], "2.5")
        self.assertEqual(self.reqs["a-5"]["allocated_gb_seconds"], "30")

    def test_hand_computed_env_b_thirds(self):
        b = self.envs["env-b"]
        self.assertEqual(b["lifetime_gb_seconds"], "30")
        self.assertEqual(b["residual_gb_seconds"], "0")
        self.assertEqual(self.reqs["b-1"]["allocated_gb_seconds"], "155/6")
        self.assertEqual(self.reqs["b-2"]["allocated_gb_seconds"], "25/12")
        self.assertEqual(self.reqs["b-3"]["allocated_gb_seconds"], "25/12")

    def test_totals_and_reconciliation(self):
        t = self.result["totals"]
        self.assertEqual(t["lifetime_gb_seconds"], "270")
        self.assertEqual(t["allocated_gb_seconds"], "120")
        self.assertEqual(t["residual_gb_seconds"], "150")
        self.assertEqual(t["raw_request_seconds"], "165")
        self.assertEqual(t["union_busy_seconds"], "120")
        self.assertTrue(t["reconciliation"]["exact"])
        self.assertEqual(t["reconciliation"]["difference_gb_seconds"], "0")
        for e in self.result["environments"]:
            self.assertTrue(e["reconciliation"]["exact"], e["id"])
            self.assertEqual(
                q(e["allocated_gb_seconds"]) + q(e["residual_gb_seconds"]),
                q(e["lifetime_gb_seconds"]),
            )
            self.assertEqual(
                q(e["union_busy_seconds"]) + q(e["uncovered_seconds"]), q(e["lifetime_seconds"])
            )
            per_req = sum(
                q(r["allocated_gb_seconds"])
                for r in self.result["requests"]
                if r["environment"] == e["id"]
            )
            self.assertEqual(per_req, q(e["allocated_gb_seconds"]))

    def test_complete_synthetic_fixture_labels_residual_idle(self):
        self.assertEqual(self.result["residual_label"], RESIDUAL_IDLE)

    def test_no_prices_or_dollars_in_output(self):
        text = OUTPUT.read_text(encoding="utf-8")
        self.assertNotIn("$", text)
        keys = []

        def walk(node):
            if isinstance(node, dict):
                keys.extend(k.lower() for k in node)
                for value in node.values():
                    walk(value)
            elif isinstance(node, list):
                for value in node:
                    walk(value)

        walk(json.loads(text))
        for key in keys:
            for word in ("usd", "price", "cost", "dollar", "saving"):
                self.assertNotIn(word, key)


class Overlap(unittest.TestCase):
    def test_two_way_overlap_equal_share(self):
        r = run(
            doc(
                [env("e", memory="2", end="10")],
                [req("x", "e", "0", "6"), req("y", "e", "4", "10")],
            )
        )
        reqs = by_id(r["requests"])
        # [0,4] x alone 8, [4,6] shared 4 -> 2 each, [6,10] y alone 8
        self.assertEqual(reqs["x"]["allocated_gb_seconds"], "10")
        self.assertEqual(reqs["y"]["allocated_gb_seconds"], "10")
        e = r["environments"][0]
        self.assertEqual(e["raw_request_seconds"], "12")
        self.assertEqual(e["union_busy_seconds"], "10")
        self.assertEqual(e["overlap_seconds"], "2")
        self.assertEqual(e["residual_gb_seconds"], "0")

    def test_identical_spans_split_evenly_in_thirds(self):
        r = run(
            doc([env("e", memory="1", end="1")], [req(i, "e", "0", "1") for i in ("p", "q", "r")])
        )
        for row in r["requests"]:
            self.assertEqual(row["allocated_gb_seconds"], "1/3")
        self.assertTrue(r["totals"]["reconciliation"]["exact"])

    def test_nested_and_touching_spans(self):
        # Touching at 5 is not overlap; nested [2,3] shares with outer [0,5].
        r = run(
            doc(
                [env("e", end="10")],
                [
                    req("outer", "e", "0", "5"),
                    req("inner", "e", "2", "3"),
                    req("next", "e", "5", "10"),
                ],
            )
        )
        reqs = by_id(r["requests"])
        self.assertEqual(reqs["outer"]["allocated_gb_seconds"], "4.5")
        self.assertEqual(reqs["inner"]["allocated_gb_seconds"], "0.5")
        self.assertEqual(reqs["next"]["allocated_gb_seconds"], "5")


class MultipleEnvironments(unittest.TestCase):
    def test_same_wall_clock_on_different_envs_is_not_shared(self):
        r = run(
            doc(
                [env("e1", memory="1", end="10"), env("e2", memory="4", end="10")],
                [req("x", "e1", "0", "10"), req("y", "e2", "0", "10")],
            )
        )
        reqs = by_id(r["requests"])
        self.assertEqual(reqs["x"]["allocated_gb_seconds"], "10")
        self.assertEqual(reqs["y"]["allocated_gb_seconds"], "40")
        self.assertEqual(r["totals"]["residual_gb_seconds"], "0")

    def test_concurrency_on_two_envs(self):
        r = run(
            doc(
                [env("e1", memory="1", end="10"), env("e2", memory="3", end="10")],
                [
                    req("a", "e1", "0", "10"),
                    req("b", "e1", "0", "10"),
                    req("c", "e2", "0", "5"),
                    req("d", "e2", "0", "5"),
                    req("f", "e2", "0", "5"),
                ],
            )
        )
        envs = by_id(r["environments"])
        self.assertEqual(envs["e1"]["max_concurrency"], 2)
        self.assertEqual(envs["e2"]["max_concurrency"], 3)
        reqs = by_id(r["requests"])
        self.assertEqual(reqs["a"]["allocated_gb_seconds"], "5")
        self.assertEqual(reqs["c"]["allocated_gb_seconds"], "5")
        self.assertEqual(envs["e2"]["residual_gb_seconds"], "15")


class EmptyInputs(unittest.TestCase):
    def test_no_environments(self):
        r = run(doc([], []))
        self.assertEqual(r["totals"]["lifetime_gb_seconds"], "0")
        self.assertEqual(r["totals"]["environment_count"], 0)
        self.assertTrue(r["totals"]["reconciliation"]["exact"])

    def test_environment_without_requests_is_all_residual(self):
        r = run(doc([env("e", memory="2", start="5", end="15")], [], complete=False))
        e = r["environments"][0]
        self.assertEqual(e["lifetime_gb_seconds"], "20")
        self.assertEqual(e["allocated_gb_seconds"], "0")
        self.assertEqual(e["residual_gb_seconds"], "20")
        self.assertEqual(e["raw_request_seconds"], "0")
        self.assertEqual(e["residual_label"], RESIDUAL_UNOBSERVED)


class ZeroDuration(unittest.TestCase):
    def test_zero_length_request_gets_zero(self):
        r = run(doc([env("e", end="10")], [req("z", "e", "4", "4"), req("x", "e", "0", "10")]))
        reqs = by_id(r["requests"])
        self.assertTrue(reqs["z"]["zero_duration"])
        self.assertEqual(reqs["z"]["allocated_gb_seconds"], "0")
        self.assertEqual(reqs["x"]["allocated_gb_seconds"], "10")
        self.assertFalse(reqs["x"]["zero_duration"])

    def test_zero_length_at_lifetime_edges(self):
        r = run(
            doc(
                [env("e", start="2", end="10")],
                [req("s", "e", "2", "2"), req("t", "e", "10", "10")],
            )
        )
        self.assertEqual(r["totals"]["residual_gb_seconds"], "8")
        self.assertEqual(r["totals"]["allocated_gb_seconds"], "0")

    def test_zero_length_outside_lifetime_rejected(self):
        with self.assertRaisesRegex(FixtureError, "outside"):
            parse_fixture(doc([env("e", start="2", end="10")], [req("z", "e", "11", "11")]))


class InvalidIdentities(unittest.TestCase):
    def test_duplicate_environment(self):
        with self.assertRaisesRegex(FixtureError, "duplicate environment id 'e'"):
            parse_fixture(doc([env("e"), env("e")], []))

    def test_duplicate_request_same_env(self):
        with self.assertRaisesRegex(FixtureError, "duplicate request id 'x'"):
            parse_fixture(doc([env("e")], [req("x", "e", "0", "1"), req("x", "e", "2", "3")]))

    def test_duplicate_request_across_envs(self):
        with self.assertRaisesRegex(FixtureError, "duplicate request id 'x'"):
            parse_fixture(
                doc([env("e1"), env("e2")], [req("x", "e1", "0", "1"), req("x", "e2", "0", "1")])
            )

    def test_duplicate_json_key(self):
        text = json.dumps(doc([env("e")], []))[:-1] + ', "requests": []}'
        with self.assertRaisesRegex(FixtureError, "duplicate JSON key 'requests'"):
            loads_fixture(text)

    def test_bad_identity_values(self):
        for bad in ("", " e", "e ", 7, None, True):
            with self.subTest(bad=bad), self.assertRaisesRegex(FixtureError, "identity"):
                parse_fixture(doc([env(bad)], []))

    def test_unknown_environment(self):
        with self.assertRaisesRegex(FixtureError, "unknown environment 'nope'"):
            parse_fixture(doc([env("e")], [req("x", "nope", "0", "1")]))


class MalformedInputs(unittest.TestCase):
    def test_request_end_before_start(self):
        with self.assertRaisesRegex(FixtureError, "before"):
            parse_fixture(doc([env("e")], [req("x", "e", "5", "4")]))

    def test_lifetime_not_positive(self):
        for start, end in (("5", "5"), ("6", "5")):
            with (
                self.subTest(start=start, end=end),
                self.assertRaisesRegex(FixtureError, "must be >"),
            ):
                parse_fixture(doc([env("e", start=start, end=end)], []))

    def test_request_out_of_bounds(self):
        for s, e in (("0", "5"), ("95", "101"), ("-0", "200")):
            with self.subTest(span=(s, e)), self.assertRaisesRegex(FixtureError, "outside"):
                parse_fixture(doc([env("e", start="1", end="100")], [req("x", "e", s, e)]))

    def test_missing_unknown_and_wrong_shape(self):
        bad_env = env("e")
        del bad_env["memory_gb"]
        cases = [
            (doc([bad_env], []), "missing field"),
            (doc([{**env("e"), "price_usd": "1"}], []), "unknown field"),
            ({**doc([], []), "extra": 1}, "unknown field"),
            ({**doc([], []), "schema": "v0"}, "schema"),
            (doc("not-a-list", []), "expected a list"),
            (doc([], [["x"]]), "expected an object"),
            (doc([], [], synthetic="yes"), "true/false"),
            ([], "expected an object"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(FixtureError, message):
                parse_fixture(data)

    def test_malformed_number_strings(self):
        for bad in ("", "abc", "1/3", " 1", "1_000", "0x10"):
            with self.subTest(bad=bad), self.assertRaisesRegex(FixtureError, "malformed"):
                parse_fixture(doc([env("e", memory=bad)], []))

    def test_invalid_json(self):
        with self.assertRaisesRegex(FixtureError, "invalid JSON"):
            loads_fixture("{not json")


class NonFiniteAndNegative(unittest.TestCase):
    def test_non_finite_strings_and_decimals(self):
        for bad in ("NaN", "nan", "sNaN", "Infinity", "-inf", Decimal("NaN"), Decimal("Infinity")):
            with self.subTest(bad=bad), self.assertRaisesRegex(FixtureError, "non-finite"):
                parse_fixture(doc([env("e", end=bad)], []))

    def test_floats_rejected_even_when_finite(self):
        for bad in (float("nan"), float("inf"), 1.5):
            with self.subTest(bad=bad), self.assertRaisesRegex(FixtureError, "binary float"):
                parse_fixture(doc([env("e", memory=bad)], []))

    def test_json_nan_literal_rejected(self):
        text = json.dumps(doc([env("e")], [])).replace('"memory_gb": "1"', '"memory_gb": NaN')
        with self.assertRaisesRegex(FixtureError, "non-finite JSON literal 'NaN'"):
            loads_fixture(text)

    def test_json_numbers_parse_exactly(self):
        text = json.dumps(doc([env("e", end="10")], [])).replace(
            '"memory_gb": "1"', '"memory_gb": 0.1'
        )
        r = compute(loads_fixture(text))
        self.assertEqual(r["totals"]["lifetime_gb_seconds"], "1")  # 0.1 * 10, no float drift

    def test_negative_and_zero_rejected(self):
        cases = [
            (doc([env("e", memory="-1")], []), "must be > 0"),
            (doc([env("e", memory="0")], []), "must be > 0"),
            (doc([env("e", start="-5")], []), "negative"),
            (doc([env("e")], [req("x", "e", "-1", "2")]), "negative"),
            (doc([env("e", memory=True)], []), "boolean"),
        ]
        for data, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(FixtureError, message):
                parse_fixture(data)


class MissingTelemetry(unittest.TestCase):
    def test_dropping_a_request_grows_residual_exactly(self):
        full_fixture = load_fixture(FIXTURE)
        full = compute(full_fixture)
        partial = compute(without_requests(full_fixture, ["a-2"]))
        fa = by_id(full["environments"])["env-a"]
        pa = by_id(partial["environments"])["env-a"]
        # a-2 alone covered [40, 50] at 1.5 GB -> 15 GB-s moves to the residual.
        self.assertEqual(q(pa["residual_gb_seconds"]) - q(fa["residual_gb_seconds"]), 15)
        self.assertEqual(pa["residual_gb_seconds"], "75")
        self.assertEqual(pa["lifetime_gb_seconds"], fa["lifetime_gb_seconds"])
        self.assertTrue(pa["reconciliation"]["exact"])
        self.assertTrue(partial["totals"]["reconciliation"]["exact"])
        # Remaining requests absorb the shared slices a-2 used to take part in.
        reqs = by_id(partial["requests"])
        self.assertEqual(reqs["a-1"]["allocated_gb_seconds"], "41.25")
        self.assertEqual(reqs["a-3"]["allocated_gb_seconds"], "3.75")
        self.assertNotIn("a-2", reqs)
        self.assertEqual(partial["residual_label"], RESIDUAL_UNOBSERVED)
        self.assertFalse(partial["telemetry_complete"])

    def test_drop_unknown_request_rejected(self):
        with self.assertRaisesRegex(FixtureError, "unknown request"):
            without_requests(load_fixture(FIXTURE), ["zzz"])

    def test_residual_label_requires_synthetic_and_complete(self):
        data = doc([env("e")], [])
        for synthetic, complete, label in (
            (True, True, RESIDUAL_IDLE),
            (True, False, RESIDUAL_UNOBSERVED),
            (False, True, RESIDUAL_UNOBSERVED),
            (False, False, RESIDUAL_UNOBSERVED),
        ):
            with self.subTest(synthetic=synthetic, complete=complete):
                d = {**data, "synthetic": synthetic, "telemetry_complete": complete}
                self.assertEqual(run(d)["residual_label"], label)


class OrderInvariance(unittest.TestCase):
    def test_shuffled_inputs_give_identical_output(self):
        data = json.loads(FIXTURE.read_text(encoding="utf-8"))
        expected = dumps_result(compute(parse_fixture(data)))
        rng = random.Random(20261113)
        for _ in range(25):
            shuffled = copy.deepcopy(data)
            rng.shuffle(shuffled["environments"])
            rng.shuffle(shuffled["requests"])
            self.assertEqual(dumps_result(compute(parse_fixture(shuffled))), expected)


class CommandLine(unittest.TestCase):
    def _main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    def test_check_passes_on_committed_outputs(self):
        self.assertEqual(self._main("--check", str(OUTPUT))[0], 0)
        self.assertEqual(self._main("--drop-request", "a-2", "--check", str(OUTPUT_MISSING))[0], 0)

    def test_check_fails_on_mismatch(self):
        code, _, err = self._main("--drop-request", "a-2", "--check", str(OUTPUT))
        self.assertEqual(code, 1)
        self.assertIn("MISMATCH", err)

    def test_stdout_is_result_json(self):
        code, out, _ = self._main()
        self.assertEqual(code, 0)
        self.assertEqual(out, OUTPUT.read_text(encoding="utf-8"))

    def test_bad_drop_is_reported(self):
        code, _, err = self._main("--drop-request", "zzz")
        self.assertEqual(code, 2)
        self.assertIn("unknown request", err)


if __name__ == "__main__":
    unittest.main()
