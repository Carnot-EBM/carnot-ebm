"""REQ-REPORT-8433: bounded private validation precedes terminal publication.

Workers own only synthetic checkpoints. Required tests use private paths;
their receipts and coverage shards stay separate from historical evidence.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any

from carnot.reporting import block_checkpoint_qualification_8433 as q
from carnot.reporting import v726_feature_panel_runner as consumers
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.v709_execution import child, execute
from carnot.verify import block_feedback_state_8433 as state

Json = dict[str, Any]


def run(root: Path, output: Path, private: Path) -> int:
    """Keep a complete terminal failure when owned validation cannot qualify readiness."""
    raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
    plan = q.manifest(private)
    atomic_json(
        raw / "execution_manifest.json",
        dict(
            commands=plan,
            owned=q.OWNED,
            task_cap_s=4800,
            validation_reserve_s=900,
            full_repository_suite="not_run",
        ),
    )
    work = q.measure(raw, private / "checkpoints", root)
    receipts = execute(plan, raw / "validation")
    if (private / "coverage.json").is_file():
        shards = raw / "coverage_shards"
        shards.mkdir()
        for source in private.glob(".coverage*"):
            shutil.copyfile(source, shards / source.name)
        shutil.copyfile(private / "coverage.json", raw / "coverage.json")
        work["coverage_shards"] = [q.reference(p) for p in sorted(shards.iterdir())]
    candidate = raw / "preterminal.json"
    atomic_json(raw / "measurement.json", work)
    atomic_json(candidate, q.build(work, receipts, raw, output))
    argv = [sys.executable, "-u", str(q.ROOT / q.CLI)]
    for name, args, expected in [
        ("cold", ["--cold-replay", str(candidate)], 0),
        ("missing", ["--cold-replay", str(raw / "absent.json")], 1),
        ("error", ["--deliberate-error"], 7),
    ]:
        receipts.append(
            child(name, [*argv, *args], raw / "controls", expected=expected, deadline=90)
        )
    changed = deepcopy(json.loads(candidate.read_bytes()))
    changed["block_store_ready_score"] = 999
    changed["reproducibility_checksum"] = q.checksum(changed)
    tamper = raw / "rehashed_tamper.json"
    atomic_json(tamper, changed)
    receipts.append(
        child(
            "rehashed_tamper",
            [*argv, "--cold-replay", str(tamper)],
            raw / "controls",
            expected=1,
            deadline=90,
        )
    )
    for field in ("coefficient", "cursor", "release", "rng"):
        cfg = state.config(8, "uniform")
        original = next(
            r
            for r in work["rows"]
            if r["capacity"] == 8 and r["arm"] == "uniform" and r["barrier"] == "none"
        )
        if not original.get("final_state"):
            continue
        changed_state = deepcopy(q.read(original["final_state"]))
        if field == "coefficient":
            changed_state["head"]["parameters"][0] += 0.2
        elif field == "cursor":
            changed_state["cursor"] -= 20
        elif field == "release":
            changed_state["applied_feedback_ids"][0] = "forged"
        else:
            changed_state["admission_rng"][1][0] ^= 1
        tamper_state = raw / ("rehashed_" + field + ".json")
        atomic_json(tamper_state, changed_state)
        atomic_json(raw / "corruption_configuration.json", cfg)
        atomic_json(
            raw / ("rehashed_" + field + "_envelope.json"),
            dict(state=changed_state, sha256=canonical_hash(changed_state)),
        )
        receipts.append(
            child(
                "corruption_" + field,
                [
                    *argv,
                    "--state-replay",
                    str(raw / ("rehashed_" + field + "_envelope.json")),
                    "--configuration",
                    str(raw / "corruption_configuration.json"),
                ],
                raw / "controls",
                expected=1,
                deadline=60,
            )
        )
    atomic_json(candidate, q.build(work, receipts, raw, output))
    checks, findings = consumers.scan(candidate, raw / "preterminal_validators")
    receipts.extend(checks)
    work["adversarial_findings"] = findings
    work["ended_monotonic_ns"] = time.monotonic_ns()
    atomic_json(raw / "measurement.json", work)
    value = q.build(work, receipts, raw, output)

    def validator(path: Path) -> Json:
        """Failed scanner qualification stays visible in an honest disqualified terminal."""
        terminal_checks, flags = consumers.scan(path, raw / "terminal_validators")
        terminal_accounting = (
            value["verdict_class"] == "disqualified"
            and value["block_store_ready_score"] == 0
            and value["required_checks_passed"] is False
        )
        return dict(
            passed=q.replay(path)
            and (all(r["passed"] for r in terminal_checks) or terminal_accounting),
            findings=flags,
            receipts=terminal_checks,
        )

    publication = publish_primary(output, value, validator)
    reader = reader_receipt(
        q.TASK,
        output.parent,
        field="block_store_ready_score",
        expected=value["block_store_ready_score"],
    )
    atomic_json(
        Path(value["terminal_validation_sidecar_path"]),
        dict(
            publication=publication,
            consumer=reader,
            primary_sha256=publication["primary_sha256"],
            validation_receipts=receipts,
            adversarial_findings=findings,
        ),
    )
    q.progress("finished", value["completed_count"], value["censored_count"])
    return 0


def main(argv: list[str] | None = None) -> int:
    """Expose dated execution and bounded worker/replay modes without model loads."""
    q.progress("start")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261012"], default="20261012")
    parser.add_argument("--root", type=Path, default=q.ROOT)
    parser.add_argument("--output", type=Path, default=q.ROOT / "results" / (q.NAME + ".json"))
    parser.add_argument("--worker", type=Path)
    parser.add_argument("--configuration", type=Path)
    parser.add_argument("--crash", default="")
    parser.add_argument("--state-replay", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    if args.deliberate_error:
        return 7
    if args.cold_replay:
        passed = q.replay(args.cold_replay)
        q.progress("replay_passed" if passed else "reduction_drift")
        return int(not passed)
    if args.worker or args.state_replay:
        try:
            cfg = json.loads(args.configuration.read_bytes())
            if args.state_replay:
                primitive = json.loads(args.state_replay.read_bytes())
                if set(primitive) == {"state", "sha256"}:
                    if primitive["sha256"] != canonical_hash(primitive["state"]):
                        return 1
                    primitive = primitive["state"]
                return int(not state.replay_state(primitive, cfg))
            result = state.drive(args.worker, cfg, args.crash)
            atomic_json(args.worker / "final.json", result)
            return 0
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            return 1
    with TemporaryDirectory(prefix="carnot8433-", dir="/var/tmp") as directory:
        return run(args.root, args.output.absolute(), Path(directory))
