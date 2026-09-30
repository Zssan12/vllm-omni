#!/usr/bin/env python3
"""Read-only audit of this run. Standard library only; never writes result files.

p50 is the median; p95 is nearest rank (ceil(0.95*n)). Group quantile
differences are not quantiles of paired per-request treatment effects.
Use --json for every request ID, source line, timestamp and missing endpoint.
"""

import argparse
import hashlib
import json
import re
import statistics
import sys
from collections import Counter, defaultdict
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

D = Decimal
CLIPS = {"a3.wav": 8, "a4.wav": 8, "a0.wav": 3, "a1.wav": 3}
TIMING = re.compile(
    r"OMNI_TIMING (?:(forward) src=(\d+) dst=(\d+)|(submit) stage=(\d+)) "
    r"req=(\S+) t=(\d+\.\d+)"
)
COMMIT = re.compile(
    r"Duplex committed turn session=(\S+) request=(\S+) response=\S+ audio_bytes=(\d+)"
)
# terminal_ms was rounded to 0.1 ms by the client; permit only half that unit.
TERMINAL_ROUNDING_S = D("0.00005")


def physical_lines(path):
    # Bare CRs occur inside progress output. Count LF lines, like sed / rg -n.
    return path.read_bytes().decode("utf-8").split("\n")


def summarize(values):
    values = sorted(v for v in values if v is not None)
    if not values:
        return {"n": 0, "p50": None, "p95": None}
    return {
        "n": len(values),
        "p50": statistics.median(values),
        "p95": values[(95 * len(values) + 99) // 100 - 1],
    }


def read_client(path, mode):
    rows, summaries = [], []
    for line, text in enumerate(physical_lines(path), 1):
        if not text.strip():
            continue
        item = json.loads(text, parse_float=D, parse_constant=D)
        if "summary" in item:
            summaries.append(item)
        else:
            item["commit_wall"] = D(item["commit_wall"])
            item["terminal_ms"] = D(item["terminal_ms"])
            item["client_line"] = line
            rows.append(item)
    errors = []
    if len(summaries) != 1 or summaries[0].get("mode") != mode:
        errors.append("expected one matching client summary")
    expected = {(clip, rep) for clip, n in CLIPS.items() for rep in range(n + 2)}
    actual = [(r["clip"], r["rep"]) for r in rows]
    if set(actual) != expected or len(actual) != len(expected):
        errors.append("client schedule differs from 2 warmups + 8/8/3/3 measured turns")
    for i, row in enumerate(rows):
        prefix = f"client line {row['client_line']}: "
        if row["mode"] != mode or row["warmup"] is not (row["rep"] < 2):
            errors.append(prefix + "mode/warmup mismatch")
        if row["errors"] or row["terminal"] not in ("response.done", "response.listen"):
            errors.append(prefix + "client error or missing terminal")
        blocked = (
            row["terminal"] == "response.listen"
            and not row["has_text"] and not row["has_audio"]
        )
        if row["blocked"] is not blocked:
            errors.append(prefix + "inconsistent blocked flag")
        if not row["commit_wall"].is_finite() or not row["terminal_ms"].is_finite() or row["terminal_ms"] <= 0:
            errors.append(prefix + "invalid client clock/duration")
        if i and row["commit_wall"] <= rows[i - 1]["commit_wall"]:
            errors.append(prefix + "non-increasing commit time")
    return rows, summaries, errors


def parse_server(lines):
    events, commits, warmups, errors = [], defaultdict(list), [], []
    warmup_start = None
    for line, text in enumerate(lines, 1):
        if "Duplex warmup starting:" in text:
            if warmup_start is not None:
                errors.append(f"server line {line}: nested warmup")
            warmup_start = line
        if "Duplex warmup finished" in text:
            if warmup_start is None:
                errors.append(f"server line {line}: warmup finish without start")
            else:
                warmups.append((warmup_start, line))
            warmup_start = None
        match = COMMIT.search(text)
        if match:
            commits[match[2]].append(
                {"line": line, "session": match[1], "audio_bytes": int(match[3])}
            )
        if "OMNI_TIMING " not in text:
            continue
        match = TIMING.search(text)
        if not match:
            errors.append(f"server line {line}: malformed timing record")
            continue
        events.append({
            "kind": match[1] or match[4],
            "stage": int(match[2] or match[5]),
            "dst": int(match[3]) if match[3] else None,
            "req": match[6], "t": D(match[7]), "line": line,
        })
    if warmup_start is not None:
        errors.append("unfinished server warmup")
    if any(b["t"] < a["t"] for a, b in zip(events, events[1:])):
        errors.append("server timing clock regressed in log order")
    return events, commits, warmups, errors


def match_and_measure(rows, events, commits, warmups, mode):
    """Require a unique request in a bounded window; never choose a nearest ID."""
    errors, matched, used, sessions = [], [], set(), set()
    by_request = defaultdict(list)
    for event in events:
        by_request[event["req"]].append(event)
    starts = [e for e in events if e["kind"] == "forward" and e["stage"] == 0]
    duplicate_starts = {k: n for k, n in Counter(e["req"] for e in starts).items() if n != 1}
    if duplicate_starts:
        errors.append(f"duplicate ASR forwards: {duplicate_starts}")
    for i, row in enumerate(rows):
        label = f"client line {row['client_line']}"
        commit = row["commit_wall"]
        end = commit + row["terminal_ms"] / 1000 + TERMINAL_ROUNDING_S
        next_commit = rows[i + 1]["commit_wall"] if i + 1 < len(rows) else D("Infinity")
        if end >= next_commit:
            errors.append(f"{label}: client windows overlap")
        candidates = [e for e in starts if commit <= e["t"] <= end and e["t"] < next_commit]
        if len(candidates) != 1:
            errors.append(f"{label}: expected one ASR forward in window, got {len(candidates)}")
            continue
        start = candidates[0]
        req = start["req"]
        if req in used:
            errors.append(f"{label}: server request reused: {req}")
            continue
        used.add(req)
        request_events = by_request[req]
        server_commits = commits.get(req, [])
        if len(server_commits) != 1 or server_commits[0]["line"] >= start["line"]:
            errors.append(f"{label}: missing/duplicate/out-of-order server commit")
            continue
        server_commit = server_commits[0]
        if server_commit["session"] in sessions:
            errors.append(f"{label}: session reused")
        sessions.add(server_commit["session"])
        if any(e["t"] < commit or e["t"] > end for e in request_events):
            errors.append(f"{label}: same-request timing lies outside client window")
        if start["dst"] != 1:
            errors.append(f"{label}: unexpected ASR forward destination")

        def endpoint(kind, stage, required=True):
            found = [e for e in request_events if e["kind"] == kind and e["stage"] == stage]
            if len(found) != (1 if required else 0):
                errors.append(f"{label}: {kind} stage={stage}, count={len(found)}, required={required}")
            return found[0] if len(found) == 1 else None

        submit1 = endpoint("submit", 1)
        judge_forward = aura_submit = None
        rejection = mode == "on" and row["blocked"]
        if mode == "on":
            judge_forward = endpoint("forward", 1, not rejection)
            aura_submit = endpoint("submit", 2, not rejection)
            if judge_forward and judge_forward["dst"] != 2:
                errors.append(f"{label}: unexpected judge destination")
            if rejection:
                endpoint("submit", 3, False)  # no Talker; stage 4 can be pre-submitted
        else:
            aura_submit = submit1
        ordered = [start, submit1] + ([judge_forward, aura_submit] if mode == "on" else [])
        ordered = [e for e in ordered if e is not None]
        if any(b["t"] < a["t"] or b["line"] <= a["line"] for a, b in zip(ordered, ordered[1:])):
            errors.append(f"{label}: gate endpoint order invalid")
        matched.append({
            **row, "req": req, "server_commit": server_commit,
            "window_end": end, "asr_forward": start, "judge_submit": submit1 if mode == "on" else None,
            "judge_forward": judge_forward, "aura_submit": aura_submit,
            "server_events": request_events,
            "commit_to_asr_forward_ms": (start["t"] - commit) * 1000,
            "gate_ms": (aura_submit["t"] - start["t"]) * 1000 if aura_submit else None,
            "judge_ms": (judge_forward["t"] - submit1["t"]) * 1000 if judge_forward and submit1 else None,
            "missing_reason": "rejected: no judge forward or AURA submit" if rejection else None,
        })
    unassigned = []
    for req in sorted((set(by_request) | set(commits)) - used):
        es = by_request.get(req, [])
        cs = commits.get(req, [])
        all_lines = [e["line"] for e in es] + [c["line"] for c in cs]
        is_warmup = bool(es and cs) and all(e["t"] < rows[0]["commit_wall"] for e in es) and any(
            all(lo < line < hi for line in all_lines) for lo, hi in warmups
        )
        unassigned.append({
            "req": req, "reason": "server startup warmup" if is_warmup else "unexplained",
            "lines": all_lines,
        })
        if not is_warmup:
            errors.append(f"unexplained unmatched server request: {req}")
    return matched, {
        "client_rows": len(rows), "matched_rows": len(matched),
        "unique_matched_requests": len(used), "unique_matched_sessions": len(sessions),
        "unmatched_client_rows": len(rows) - len(matched),
        "asr_forwards": len(starts), "unique_asr_requests": len({e["req"] for e in starts}),
        "duplicate_asr_forwards": duplicate_starts, "unassigned_server_requests": unassigned,
        "server_warmup_line_ranges": warmups,
        "client_warmups_excluded": sum(r["warmup"] for r in matched),
        "measured_rows": sum(not r["warmup"] for r in matched),
        "measured_rejected": sum(r["blocked"] and not r["warmup"] for r in matched),
        "commit_to_asr_forward_ms": summarize(r["commit_to_asr_forward_ms"] for r in matched),
    }, errors


def group_summary(rows):
    return {
        "attempts": len(rows), "blocked": sum(r["blocked"] for r in rows),
        "has_audio": sum(r["has_audio"] for r in rows),
        **{key: summarize(r[key] for r in rows) for key in (
            "gate_ms", "judge_ms", "first_text_ms", "first_audio_ms", "terminal_ms"
        )},
    }


def analyze(directory):
    report = {"status": "valid", "errors": [], "inputs_sha256": {}, "modes": {}}
    for mode in ("off", "on"):
        client_path, server_path = directory / f"onoff-{mode}.jsonl", directory / f"server-{mode}.log"
        for path in (client_path, server_path):
            report["inputs_sha256"][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        rows, summaries, errors = read_client(client_path, mode)
        if not rows:
            raise ValueError(f"{client_path}: no client rows")
        events, commits, warmups, parse_errors = parse_server(physical_lines(server_path))
        matches, audit, match_errors = match_and_measure(rows, events, commits, warmups, mode)
        measured = [r for r in matches if not r["warmup"]]
        groups = {clip: group_summary([r for r in measured if r["clip"] == clip]) for clip in CLIPS}
        groups["questions_a3_a4"] = group_summary([r for r in measured if r["clip"] in ("a3.wav", "a4.wav")])
        groups["all_measured"] = group_summary(measured)
        # Deliberately no all-clips on/off difference: a0 has no on-gate endpoint.
        if len(summaries) == 1:
            for clip in CLIPS:
                saved = summaries[0]["summary"].get(clip, {})
                computed = groups[clip]
                for key, value in (("n", computed["attempts"]), ("blocked", computed["blocked"]), ("spoke", computed["has_audio"])):
                    if saved.get(key) != value:
                        errors.append(f"{clip}: stored summary {key} disagrees with rows")
                for metric in ("first_text_ms", "first_audio_ms", "terminal_ms"):
                    stats = computed[metric]
                    if stats["n"]:
                        old = saved.get(metric, {})
                        if old.get("n") != stats["n"] or any(
                            abs(old.get(q, D("Infinity")) - stats[q]) > D("0.05") for q in ("p50", "p95")
                        ):
                            errors.append(f"{clip}: stored summary {metric} disagrees with rows")
        report["errors"].extend(f"{mode}: {e}" for e in errors + parse_errors + match_errors)
        report["modes"][mode] = {"audit": audit, "groups": groups, "rows": matches}
    differences = {}
    for clip in (*CLIPS, "questions_a3_a4"):
        a, b = (report["modes"][m]["groups"][clip]["gate_ms"] for m in ("off", "on"))
        differences[clip] = {
            "off_n": a["n"], "on_n": b["n"],
            **{q: b[q] - a[q] if a[q] is not None and b[q] is not None else None for q in ("p50", "p95")},
        }
    report["gate_group_quantile_differences_ms"] = differences
    report["definitions"] = {
        "p50": "median (average of the middle two for even n)",
        "p95": "nearest rank: sorted[ceil(0.95*n)-1]",
        "off_gate_ms": "1000*(submit stage=1 - forward src=0), same request ID",
        "on_gate_ms": "1000*(submit stage=2 - forward src=0), same request ID",
        "judge_ms": "1000*(forward src=1 - submit stage=1), on mode, same request ID",
        "difference": "Q(on gate)-Q(off gate); NOT Q(on_i-off_i); different sessions, sequential runs",
        "display_rounding": "Decimal ROUND_HALF_UP to 0.001 ms, only after computing quantiles/differences",
        "pairing": "unique ASR forward within [commit_wall, commit_wall+terminal_ms/1000+0.00005] and before next commit",
        "server_line_numbers": "physical LF-delimited lines (CR progress updates do not increment)",
    }
    if report["errors"]:
        report["status"] = "invalid: do not quote aggregates"
    return report


def display(report):
    print(f"Audit: {report['status']}")
    for mode, data in report["modes"].items():
        a = data["audit"]
        print(f"{mode}: matched={a['matched_rows']}/{a['client_rows']}, warmups={a['client_warmups_excluded']}, "
              f"measured={a['measured_rows']}, rejected={a['measured_rejected']}, "
              f"unassigned_server_requests={json.dumps(a['unassigned_server_requests'])}")
    if report["errors"]:
        print("\n".join(report["errors"]))
        return
    print("\nAll values in ms. p50=median; p95=nearest rank. Delta=group quantile difference.")
    print("| Group | off gate n; p50 / p95 | on gate n; p50 / p95 | on judge n; p50 / p95 | delta p50 / p95 |")
    print("|---|---:|---:|---:|---:|")

    def pair(s):
        if s["p50"] is None:
            return "NA"
        return " / ".join(str(s[q].quantize(D("0.001"), rounding=ROUND_HALF_UP)) for q in ("p50", "p95"))

    for group in (*CLIPS, "questions_a3_a4"):
        off = report["modes"]["off"]["groups"][group]["gate_ms"]
        on = report["modes"]["on"]["groups"][group]
        gate, judge = on["gate_ms"], on["judge_ms"]
        delta = report["gate_group_quantile_differences_ms"][group]
        print(f"| {group} | {off['n']}; {pair(off)} | {gate['n']}; {pair(gate)} | "
              f"{judge['n']}; {pair(judge)} | {pair(delta)} |")


def self_test(directory):
    """Adversarial audit tests on in-memory copies; original logs stay untouched."""
    from copy import deepcopy
    import unittest

    rows, _, _ = read_client(directory / "onoff-on.jsonl", "on")
    events, commits, warmups, _ = parse_server(physical_lines(directory / "server-on.log"))
    baseline = analyze(directory)
    target = baseline["modes"]["on"]["rows"][2]
    reject = baseline["modes"]["on"]["rows"][22]

    class AuditTests(unittest.TestCase):
        def audit(self, changed):
            return match_and_measure(rows, changed, commits, warmups, "on")

        def test_actual_run_coverage_and_warmups(self):
            self.assertEqual(baseline["status"], "valid")
            for mode in ("off", "on"):
                audit = baseline["modes"][mode]["audit"]
                self.assertEqual((audit["matched_rows"], audit["client_warmups_excluded"], audit["measured_rows"]), (30, 8, 22))
                self.assertEqual(audit["unassigned_server_requests"][0]["reason"], "server startup warmup")

        def test_duplicate_forward_fails(self):
            self.assertTrue(self.audit(events + [target["asr_forward"]])[2])

        def test_two_different_candidates_fail(self):
            extra = {**target["asr_forward"], "req": "unrelated-request"}
            self.assertTrue(self.audit(events + [extra])[2])

        def test_missing_forward_does_not_borrow_next_request(self):
            changed = [e for e in events if e != target["asr_forward"]]
            matched, _, errors = self.audit(changed)
            self.assertTrue(errors)
            self.assertNotIn(target["client_line"], [r["client_line"] for r in matched])

        def test_missing_accepted_submit_fails(self):
            self.assertTrue(self.audit([e for e in events if e != target["aura_submit"]])[2])

        def test_duplicate_submit_fails(self):
            self.assertTrue(self.audit(events + [target["aura_submit"]])[2])

        def test_rejection_is_missing_not_zero(self):
            self.assertTrue(reject["blocked"])
            self.assertIsNone(reject["gate_ms"])
            self.assertIsNone(reject["judge_ms"])
            group = baseline["modes"]["on"]["groups"]["a0.wav"]
            self.assertEqual((group["attempts"], group["gate_ms"]["n"]), (3, 0))

        def test_rejection_with_aura_submit_fails(self):
            extra = {**reject["judge_submit"], "stage": 2}
            self.assertTrue(self.audit(events + [extra])[2])

        def test_backwards_endpoint_fails(self):
            changed = deepcopy(events)
            e = next(e for e in changed if e["line"] == target["aura_submit"]["line"])
            e["t"] = target["asr_forward"]["t"] - D("0.001")
            self.assertTrue(self.audit(changed)[2])

        def test_unexplained_unmatched_request_fails(self):
            extra = {**events[-1], "req": "orphan", "t": events[-1]["t"] + 100}
            self.assertTrue(self.audit(events + [extra])[2])

        def test_malformed_timing_and_clock_regression_fail(self):
            self.assertTrue(parse_server(["OMNI_TIMING invalid"])[3])
            self.assertTrue(parse_server([
                "OMNI_TIMING forward src=0 dst=1 req=r t=2.0",
                "OMNI_TIMING submit stage=1 req=r t=1.0",
            ])[3])

        def test_quantiles_and_group_difference_are_not_paired(self):
            off, on = list(map(D, (0, 100, 101))), list(map(D, (50, 51, 102)))
            group_delta = summarize(on)["p50"] - summarize(off)["p50"]
            paired_delta = summarize([b - a for a, b in zip(off, on)])["p50"]
            self.assertEqual((group_delta, paired_delta), (D(-49), D(1)))
            self.assertEqual(summarize(list(map(D, range(1, 21))))["p95"], D(19))
            self.assertEqual(summarize([D(1), D(2)])["p50"], D("1.5"))

    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(AuditTests))
    return 0 if result.wasSuccessful() else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", help="emit full audit to stdout; never writes files")
    parser.add_argument("--self-test", action="store_true", help="run in-memory corruption/coverage tests; no file writes")
    args = parser.parse_args()
    try:
        if args.self_test:
            return self_test(args.directory)
        report = analyze(args.directory)
    except (OSError, ValueError, KeyError, TypeError, ArithmeticError) as exc:
        print(f"Analysis failed; no statistics accepted: {exc}", file=sys.stderr)
        return 2
    if args.json:
        # Decimal subtraction preserves the original microsecond precision.
        # JSON exports Decimal as strings to avoid binary-float roundoff.
        print(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    else:
        display(report)
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
