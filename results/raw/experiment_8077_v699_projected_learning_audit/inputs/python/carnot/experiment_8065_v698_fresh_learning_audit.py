"""REQ-REPORT-8065: authenticate primitives and publish a terminal learning audit."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tempfile
import time
from typing import Any

from carnot import experiment_8064_v698_fresh_feedback_learning as upstream
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.primary_publication import publish_primary, read_bound_sidecar
from carnot.reporting.v686_contract_validation import run_check
from carnot.verify import fresh_learning_audit_8065 as a

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME = "experiment_8065_v698_fresh_learning_audit"
TASK = "exp8065-fresh-learning-audit"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = f"python/carnot/{NAME}.py"
TEST = "tests/python/test_fresh_learning_audit_8065.py"
OWNED = [MODULE, CLI, "python/carnot/verify/fresh_learning_audit_8065.py"]
START = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Show real work and elapsed time without inventing inference activity."""
    print(
        f"[exp8065] {phase} elapsed_s={time.monotonic() - START:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def failure(path: Path, field: str, expected: Any, observed: Any, *, owned: bool = False) -> Json:
    """Keep exact operands so a missing artifact differs from failed arithmetic."""
    return dict(
        check=field,
        upstream=path.stem,
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op="==",
        expected=expected,
        observed=observed,
        classification="owned" if owned else "external",
    )


def labels(data: Json, key: str) -> Json:
    """An explicit accessor separates delayed stream labels from retention targets."""
    if key in data:
        return dict(data[key])
    ref = "target_reference" if key == "labels" else "retention_target"
    return {
        r["family_id"]: r["eligible_y"] for r in json.loads(checked(data[ref]).read_text())["rows"]
    }


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate the sealed CPU branch and current terminal trajectory binding."""
    data, failures = upstream.load_inputs(root, raw / "custody")
    path = root / "results" / (upstream.NAME + ".json")
    try:
        value = json.loads(path.read_text())
        side = Path(value["terminal_validation_sidecar_path"])
        binding = json.loads(side.read_text())["publication"]
        report = read_bound_sidecar(path, Path(binding["sidecar_path"]))
        for field, expected, observed in [
            ("experiment_id", 8064, value.get("experiment_id")),
            ("task_id", upstream.TASK, value.get("task_id")),
            ("learning_trajectory_ready_score", 1, value.get("learning_trajectory_ready_score")),
            ("required_checks_passed", True, value.get("required_checks_passed")),
            ("verifier_is_oracle", False, value.get("verifier_is_oracle")),
            ("flagged_adversarial", False, value.get("flagged_adversarial")),
            ("primary_sha256", sha256_file(path), binding.get("primary_sha256")),
            ("report.passed", True, report["report"]["passed"]),
        ]:
            if expected != observed:
                failures.append(failure(path, field, expected, observed))
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            checked(ref)
        data["references"] += [reference(p) for p in (path, side, Path(binding["sidecar_path"]))]
        data["trajectory"] = value["trajectory_directory"]
        original = json.loads((Path(data["trajectory"]).parent / "plan.json").read_text())["data"]
        for key in ("head", "sources", "seeds", "retention"):
            if data[key] != original[key]:
                failures.append(failure(path, key, original[key], data[key]))
        health = root / "results/raw" / NAME / "preflight/repository_health.json"
        if health.is_file():
            data["current_repository_health"] = json.loads(health.read_text())["receipts"]
            data["references"] += [
                reference(p)
                for p in [health, *(Path(r["log_path"]) for r in data["current_repository_health"])]
            ]
        for p in (
            ROOT / "python/carnot/verify/learning_retention_audit_8026.py",
            ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md",
        ):
            data["references"].append(reference(p))
    except (OSError, KeyError, ValueError) as error:
        failures.append(failure(path, "authenticated_8064", True, str(error)))
    return data, failures


def manifest(private: Path) -> list[Json]:
    """Reuse the explicit bounded validation contract with this module's includes."""
    specs = upstream.manifest(private)
    replacements = dict(zip([*upstream.OWNED, upstream.TEST], [*OWNED, TEST], strict=True))
    for spec in specs:
        spec["argv"] = [replace_paths(item, replacements) for item in spec["argv"]]
    config = private / "coverage.ini"
    config.write_text(
        "[run]\nparallel=true\ndata_file="
        + str(private / ".coverage")
        + "\ninclude=\n"
        + "".join("    " + str(ROOT / p) + "\n" for p in OWNED)
    )
    return specs


def replace_paths(item: str, mapping: dict[str, str]) -> str:
    """Retain each deadline and argv while replacing only owned file names."""
    for old, new in mapping.items():
        item = item.replace(old, new)
    return item


def journal_events(path: Path) -> list[Any]:
    """Cold-read ordered transactions without opening a writer or resuming work."""
    db = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = db.execute("SELECT kind,payload FROM events ORDER BY seq").fetchall()
    db.close()
    return [(kind, json.loads(payload)) for kind, payload in rows]


def recovery_state(expected: list[Any], prefix: list[Any], actual: list[Any]) -> Json:
    """Derive outstanding labels and the next update from chronological events."""
    issued = {r["slot"] for k, r in prefix if k == "issue"}
    released = {r["slot"] for k, r in prefix if k == "release"}
    consumed = [r["slot"] for k, r in prefix if k == "consume"]
    candidates = [r for k, r in prefix if k == "candidate"]
    next_update = next((r for k, r in actual[len(prefix) :] if k == "commit"), None)
    original_next = next((r for k, r in expected[len(prefix) :] if k == "commit"), None)
    a.math.equal("recovered_prefix", expected[: len(prefix)], prefix)
    a.math.equal("recovered_predictions_and_states", expected, actual)
    a.math.equal("recovered_next_update", original_next, next_update)
    return dict(
        pending_ids=sorted(issued - released),
        consumed_ids=consumed,
        pending_candidate_ids=sorted(set(candidates[-1]["fresh_slots"]) - set(consumed))
        if candidates
        else [],
        prefix_hash=canonical_hash(prefix),
        next_update_hash=canonical_hash(next_update),
        predictions_hash=canonical_hash([r for k, r in actual if k == "issue"]),
    )


def verify_recovery(raw: Path) -> list[Json]:
    """Rebuild recovery operands from journals instead of trusting passed flags."""
    data = json.loads((raw / "input.json").read_text())
    baseline = raw / "uninterrupted" / f"seed-{data['seeds'][0]}" / "ledger.sqlite"
    expected = journal_events(baseline)
    rows: list[Json] = json.loads((raw / "receipts.json").read_text())["rows"]
    a.math.equal(
        "recovery_boundaries",
        [(k, s) for k in ("candidate", "consume", "commit") for s in ("before", "after")],
        [(r["event"], r["boundary"]) for r in rows],
    )
    for row in rows:
        count = next(i for i, (kind, _) in enumerate(expected) if kind == row["event"]) + int(
            row["boundary"] == "after"
        )
        ledger = raw / (row["event"] + "-" + row["boundary"]) / baseline.parent.name / baseline.name
        state = recovery_state(expected, expected[:count], journal_events(ledger))
        state.update(
            uninterrupted_sha256=sha256_file(baseline), recovered_sha256=sha256_file(ledger)
        )
        for field, observed in state.items():
            a.math.equal("recovery." + field, observed, row[field])
        a.math.equal("recovery_bytes", state["uninterrupted_sha256"], state["recovered_sha256"])
        a.math.equal("recovery_kill", -9, row["exit_code"])
        a.math.equal(
            "recovery_passed",
            True,
            row["passed"] and all(r["passed"] for r in row["validation_receipts"]),
        )
    return rows


def recovery(data: Json, raw: Path) -> list[Json]:
    """SIGKILL private journals at six boundaries and compare exact resumed bytes."""
    raw.mkdir(parents=True, exist_ok=True)
    d = {k: data[k] for k in ("head", "sources", "seeds")}
    d.update(seeds=[data["seeds"][0]], labels=labels(data, "labels"))
    src = raw / "input.json"
    atomic_json(src, d)
    worker = raw / "worker.py"
    worker.write_text(
        "from pathlib import Path\nimport json,os,signal,sys\nfrom carnot.verify import fresh_feedback_8064 as m\nd=json.loads(Path(sys.argv[1]).read_text())\noriginal=m.Journal.emit\ndef emit(self,kind,row):\n if kind==sys.argv[3] and sys.argv[4]=='before': os.kill(os.getpid(),signal.SIGKILL)\n original(self,kind,row)\n if kind==sys.argv[3] and sys.argv[4]=='after': os.kill(os.getpid(),signal.SIGKILL)\nm.Journal.emit=emit\nm.measure(d,Path(sys.argv[2]))\n"
    )
    py = str(ROOT / ".venv/bin/python")
    baseline = raw / "uninterrupted"

    def run(name: str, folder: Path, event: str, side: str, expected: int) -> Json:
        progress("subprocess_before_" + name)
        row = run_check(
            ROOT,
            dict(
                name=name,
                argv=[py, "-u", str(worker), str(src), str(folder), event, side],
                deadline_s=60,
                expected_exit=expected,
            ),
            raw / "private_logs",
            raw / "logs",
        )
        progress("subprocess_after_" + name, int(row["passed"]), 0)
        return row

    receipts = [run("baseline", baseline, "none", "none", 0)]
    base = baseline / f"seed-{d['seeds'][0]}" / "ledger.sqlite"

    expected = journal_events(base)
    rows = []
    for event in ("candidate", "consume", "commit"):
        for side in ("before", "after"):
            folder = raw / (event + "-" + side)
            death = run(event + "_" + side + "_kill", folder, event, side, -9)
            ledger = folder / base.parent.name / base.name
            prefix = journal_events(ledger)
            resumed = run(event + "_" + side + "_resume", folder, "none", "none", 0)
            actual = journal_events(ledger)
            state = recovery_state(expected, prefix, actual)
            row = dict(
                event=event,
                boundary=side,
                exit_code=death["exit_code"],
                **state,
                uninterrupted_sha256=sha256_file(base),
                recovered_sha256=sha256_file(ledger),
                passed=all(r["passed"] for r in [*receipts, death, resumed])
                and prefix == expected[: len(prefix)]
                and actual == expected
                and sha256_file(base) == sha256_file(ledger),
                validation_receipts=[death, resumed],
                scope="Private first-seed restart control; no scientific sample credit",
            )
            rows.append(row)
            progress("recovery_boundary", len(rows), 6 - len(rows))
    atomic_json(raw / "receipts.json", dict(rows=rows, baseline_receipts=receipts))
    return rows


def measure(data: Json, raw: Path) -> Json:
    """Copy historical primitives, reconstruct them, and seal evaluator predictions."""
    progress("benchmark_before_independent_reconstruction")
    shutil.copytree(Path(data["trajectory"]), raw / "trajectory")
    inputs = json.loads((raw / "trajectory/inputs.json").read_text())
    for key in ("head", "sources", "seeds"):
        a.math.equal("input." + key, data[key], inputs[key])
    a.math.equal("original_timeline", 256, len(inputs["sources"]))
    a.math.equal("unique_sources", 256, len({r["source_cluster_id"] for r in inputs["sources"]}))
    result = a.reconstruct(raw / "trajectory", labels(data, "labels"))
    progress("benchmark_after_independent_reconstruction", len(result["issued_prediction_rows"]), 0)
    progress("retention_predictions_before", 0, len(data["retention"]))
    result["retention_rows"] = a.retention(
        data, result["final_head_seals"], raw, lambda: labels(data, "retention_labels")
    )
    progress("retention_predictions_after", len(data["retention"]), 0)
    result.update(a.comparisons(result["later_source_rows"], result["retention_rows"]))
    result["recovery_rows"] = recovery(data, raw / "recovery")
    a.math.equal("recovery_complete", True, all(r["passed"] for r in result["recovery_rows"]))
    atomic_json(raw / "independent_reduction.json", result)
    return result


def build(
    data: Json,
    failures: list[Json],
    raw: Path,
    receipts: list[Json],
    coverage: Json,
    fixture: bool,
    started: int,
) -> Json:
    """A valid negative trajectory stays terminal and receives no benefit score."""
    complete = (raw / "independent_reduction.json").is_file()
    result = (
        json.loads((raw / "independent_reduction.json").read_text())
        if complete
        else {
            "rows": [],
            "later_source_rows": [],
            "retention_rows": [],
            "per_seed_false_accept_rows": [],
            "recovery_rows": [],
            "primary_hypothesis_results": [],
            "admission_reuse_count": 0,
        }
    )
    passed = (
        bool(receipts)
        and all(r["passed"] for r in receipts)
        and all(p in coverage and coverage[p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    benefit = bool(
        complete
        and result["primary_hypothesis_results"][0]["qualified_benefit"]
        and passed
        and not fixture
        and not failures
    )
    owned = any(r.get("classification") == "owned" for r in failures) or bool(
        receipts and not passed
    )
    kind = (
        "disqualified"
        if owned
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "positive"
        if benefit
        else "null"
        if passed and complete
        else "disqualified"
    )
    gates = list(failures)
    for r in receipts:
        if not r["passed"]:
            gates.append(
                failure(
                    Path(r["log_path"]),
                    r["name"] + ".exit_code",
                    r["expected_exit"],
                    r["exit_code"],
                    owned=True,
                )
            )
    if complete:
        h = result["primary_hypothesis_results"][0]
        checks = [
            ("support_count", ">=", 80, h["support_count"]),
            ("support_passed", "==", True, h["support_passed"]),
            ("safety_passed", "==", True, h["safety_passed"]),
            ("beneficial_changed_sources", ">=", 5, h["beneficial_changed_sources"]),
            ("cost_gain", ">=", 0.02, h["tests"][0]["gain"]),
            ("margin_p", "<", 0.05, h["tests"][0]["raw_p"]),
        ]
        checks += [("later_class_" + str(c), ">=", 10, n) for c, n in enumerate(h["class_counts"])]
        checks += [("retention_count", ">=", 48, h["retention_support_count"])]
        checks += [
            ("retention_class_" + str(c), ">=", 8, n)
            for c, n in enumerate(h["retention_class_counts"])
        ]
        checks += [
            ("cost_noninferiority." + arm, "<=", 0.02, v)
            for arm, v in h["noninferiority_cost_differences"].items()
        ]
        for retained in h["retention_checks"]:
            checks += [
                ("retention." + retained["arm"] + ".brier", "<=", 0.01, retained["brier_drift"]),
                ("retention." + retained["arm"] + ".cost", "<=", 0.02, retained["cost_drift"]),
            ]
        fa = {(r["seed"], r["arm"]): r["numerator"] for r in result["per_seed_false_accept_rows"]}
        checks += [
            (f"false_accept.{seed}.{arm}", "<=", fa[seed, arm], v)
            for (seed, name), v in fa.items()
            if name == "fresh_admission"
            for arm in a.ARMS[:3]
        ]
        for field, op, expected, observed in checks:
            ok = observed is not None and (
                observed >= expected
                if op == ">="
                else observed <= expected
                if op == "<="
                else observed < expected
                if op == "<"
                else observed == expected
            )
            if not ok:
                gates.append(
                    dict(
                        failure(
                            raw / "independent_reduction.json", "H3." + field, expected, observed
                        ),
                        op=op,
                        classification="scientific",
                    )
                )
    feedback = [
        r for r in result.get("feedback_release_rows", []) if r["seed"] == data.get("seeds", [0])[0]
    ]
    completed = sum(r["eligible"] for r in feedback)
    n = len(data.get("sources", [])) or 256
    counts = dict(
        intended=n,
        eligible=completed,
        independent=completed,
        completed=completed,
        censored=n - len(feedback),
        excluded=len(feedback) - completed,
        failed=0,
    )
    value: Json = dict(
        experiment_id=8065,
        task_id=TASK,
        milestone="2026.10.698",
        schema="carnot.v698.fresh_learning_audit.v1",
        run_date="20261003",
        honest_verdict="complete_" + kind + "_fresh_learning_audit",
        verdict_class=kind,
        verifier_is_oracle=fixture,
        flagged_adversarial=False,
        claim_scope="Independent audit of one historically exposed development stream. Private recovery controls add no sources. Conditional H3 benefit is separate from lifelong learning and future safety.",
        generalized_learning_benefit_score=0,
        learning_audit_ready_score=int(passed and complete and not failures and not fixture),
        learning_benefit_score=int(benefit),
        required_checks_passed=passed,
        validation_receipts=receipts,
        coverage_statement_counts={k: v["summary"] for k, v in coverage.items()},
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        substrate_declaration=dict(
            inference_substrate="aggregation_from_upstream_artifacts",
            no_model_load=True,
            MODEL_SPECS=[],
        ),
        config=a.CONFIG,
        random_seed=a.CONFIG["seed"],
        gate_check_summary=gates,
        historical_exposure="Original stream256 and retention64 were used in prior development studies; no independent environment or pretraining nonexposure is claimed.",
        sample_size_budget=dict(counts, seeds_are_independent=False, independent_datasets=1),
        repository_health=data.get("repository_health", []),
        current_repository_health=data.get("current_repository_health", []),
        phase_spans=data.get("phase_spans", []),
        terminal_validation_sidecar_path=str(raw.parent / "terminal_validation.json"),
        raw_directory=str(raw),
        independent_reduction_hash=canonical_hash(result),
        methodology_note="Independent cubic geometry, calibrated BCE, guard and event replay. Common update-role issued states after matched opportunity; original-slot margin bootstrap; both guarded retention arms.",
        source_artifact_hashes=data.get("references", []),
        **result,
    )
    value.update({k + "_count": v for k, v in counts.items()})
    if failures and not owned:
        value["honest_verdict"] = "complete_blocked_" + str(failures[0]["check"]).replace(".", "_")
    ended = time.monotonic_ns()
    value["duration_s"] = (ended - started) / 1e9
    value["current_work_receipt"] = build_current_work_receipt(
        run_id=str(started),
        owner_pid=os.getpid(),
        events=[],
        inference_substrate=value["inference_substrate"],
        inference_substrate_details=dict(
            operation="independent arithmetic and private restart controls; no pretrained model"
        ),
        inference_substrate_class="no_model_load",
        execution_venue="host",
        started_monotonic_ns=started,
        ended_monotonic_ns=ended,
        phase_spans=value["phase_spans"],
    )
    value["model_invocation_counts"] = value["current_work_receipt"]["invocation_counts"]
    value["code_config_hashes"] = [
        reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            "python/carnot/verify/learning_retention_audit_8026.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
        ]
    ]
    value["raw_shard_hashes"] = [reference(p) for p in sorted(raw.rglob("*")) if p.is_file()]
    value["reproducibility_checksum"] = canonical_hash(
        dict(config=a.CONFIG, code=value["code_config_hashes"], raw=value["raw_shard_hashes"])
    )
    value["field_principles"] = {
        k: "Bind this field to primitive bytes and original chronology; repetitions, private controls and exposed development cannot supply independent lifelong benefit."
        for k in value
    }
    return value


def replay(path: Path) -> Json:
    """Cold-reduce primitive bytes and reject forged readiness or edited metrics."""
    value = json.loads(path.read_text())
    for ref in (
        value["raw_shard_hashes"] + value["code_config_hashes"] + value["source_artifact_hashes"]
    ):
        checked(ref)
    raw = Path(value["raw_directory"])
    plan = json.loads((raw / "plan.json").read_text())
    validation = json.loads((raw / "validation.json").read_text())
    receipts, coverage = validation["receipts"], validation["coverage"]
    a.math.equal("validation_receipts", receipts, value["validation_receipts"])
    passed = (
        bool(receipts)
        and [r["name"] for r in receipts] == plan["validation_manifest"]
        and all(r["passed"] for r in receipts)
        and all(p in coverage and coverage[p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    a.math.equal("required_checks_passed", passed, value["required_checks_passed"])
    complete = (raw / "independent_reduction.json").is_file()
    if complete:
        result = a.reconstruct(raw / "trajectory", labels(plan["data"], "labels"))
        seal = json.loads((raw / "retention_prediction_seal.json").read_text())
        a.math.equal("retention_seal", result["final_head_seals"], seal["final_head_seals"])
        a.math.equal("labels_opened", False, seal["labels_opened"])
        with tempfile.TemporaryDirectory(prefix="carnot-8065-cold-") as temp:
            result["retention_rows"] = a.retention(
                plan["data"],
                result["final_head_seals"],
                Path(temp),
                lambda: labels(plan["data"], "retention_labels"),
            )
            a.math.equal(
                "retention_prediction_bytes",
                seal,
                json.loads((Path(temp) / "retention_prediction_seal.json").read_text()),
            )
        result.update(a.comparisons(result["later_source_rows"], result["retention_rows"]))
        result["recovery_rows"] = verify_recovery(raw / "recovery")
        a.math.equal("recovery_passed", True, all(r["passed"] for r in result["recovery_rows"]))
        for field, observed in result.items():
            a.math.equal("reduction." + field, observed, value[field])
        a.math.equal(
            "independent_reduction_hash",
            canonical_hash(result),
            value["independent_reduction_hash"],
        )
    ready = int(passed and complete and not plan["failures"] and not plan["fixture"])
    benefit = int(ready and value["primary_hypothesis_results"][0]["qualified_benefit"])
    a.math.equal("learning_audit_ready_score", ready, value["learning_audit_ready_score"])
    a.math.equal("learning_benefit_score", benefit, value["learning_benefit_score"])
    a.math.equal("oracle_scope", plan["fixture"], value["verifier_is_oracle"])
    return dict(passed=True, sha256=sha256_file(path))


def terminal_commands(path: Path) -> list[Json]:
    """Declare cold arithmetic and both shared readers before publication."""
    py = str(ROOT / ".venv/bin/python")
    return [
        dict(name=n, argv=argv, deadline_s=120, expected_exit=0)
        for n, argv in [
            ("cold_replay", [py, "-u", str(ROOT / CLI), "--cold-replay", str(path)]),
            ("adversarial", [py, str(ROOT / "scripts/adversarial_verify.py"), "--json", str(path)]),
            (
                "strict_rows",
                [py, str(ROOT / "scripts/verdict_row_consistency_lint.py"), "--strict", str(path)],
            ),
        ]
    ]


def terminal(path: Path) -> Json:
    """All three children must exit normally before checked bytes are published."""
    raw = Path(json.loads(path.read_text())["terminal_validation_sidecar_path"]).parent
    receipts = []
    with tempfile.TemporaryDirectory(prefix="carnot-8065-terminal-") as temp:
        for spec in terminal_commands(path):
            progress("subprocess_before_" + spec["name"])
            receipts.append(run_check(ROOT, spec, Path(temp), raw / "terminal_logs" / path.name))
            progress("subprocess_after_" + spec["name"], len(receipts), 3 - len(receipts))
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze commands, audit once, validate, and publish one terminal result."""
    os.environ["PYTHONUNBUFFERED"] = "1"
    progress("start_preconditions")
    started = time.monotonic_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = (args.fixture_output or args.output).absolute()
        if output.exists():
            raise ValueError("existing_primary_preserved")
        raw = output.parent / "raw" / output.stem / str(time.time_ns())
        with tempfile.TemporaryDirectory(prefix="carnot-8065-") as temp:
            private = Path(temp)
            specs = manifest(private)
            atomic_json(
                raw / "validation_commands.json",
                dict(
                    commands=specs,
                    terminal_commands=terminal_commands(raw.parent / "terminal_candidate.json"),
                    code={p: sha256_file(ROOT / p) for p in [*OWNED, TEST]},
                ),
            )
            progress("inputs_before")
            data, failures = (
                (json.loads(args.fixture_input.read_text()), [])
                if args.fixture_input
                else load_inputs(args.root, raw)
            )
            frozen = time.monotonic_ns()
            receipts = []
            if not args.fixture_output:
                receipt = run_check(ROOT, specs[0], private, raw / "validation_logs")
                receipts.append(receipt)
                if not receipt["passed"]:
                    failures.append(
                        failure(
                            Path(receipt["log_path"]), "python_environment", 0, receipt["exit_code"]
                        )
                    )
            atomic_json(
                raw / "plan.json",
                dict(
                    data=data,
                    failures=failures,
                    fixture=bool(args.fixture_input),
                    validation_manifest=[r["name"] for r in specs],
                ),
            )
            progress("inputs_after", len(data.get("sources", [])), 0)
            if not failures:
                try:
                    measure(data, raw)
                except (OSError, KeyError, ValueError, IndexError, TimeoutError) as error:
                    failures.append(
                        failure(
                            raw / "trajectory",
                            "independent_reconstruction",
                            True,
                            str(error),
                            owned=True,
                        )
                    )
                    atomic_json(
                        raw / "plan.json",
                        dict(
                            data=data,
                            failures=failures,
                            fixture=bool(args.fixture_input),
                            validation_manifest=[r["name"] for r in specs],
                        ),
                    )
            measured = time.monotonic_ns()
            os.environ["CARNOT_8065_COVERAGE_CONFIG"] = str(private / "coverage.ini")
            progress("validation_before", 0, len(specs))
            if not args.fixture_output:
                receipts += [
                    run_check(ROOT, spec, private, raw / "validation_logs") for spec in specs[1:]
                ]
            coverage = (
                json.loads((private / "coverage.json").read_text())["files"]
                if (private / "coverage.json").is_file()
                else {}
            )
            atomic_json(raw / "validation.json", dict(receipts=receipts, coverage=coverage))
            data["phase_spans"] = [
                dict(phase="freeze", duration_s=(frozen - started) / 1e9),
                dict(phase="audit", duration_s=(measured - frozen) / 1e9),
                dict(phase="validation", duration_s=(time.monotonic_ns() - measured) / 1e9),
            ]
            value = build(
                data, failures, raw, receipts, coverage, bool(args.fixture_input), started
            )
            progress("validation_after", len(receipts), 0)
            progress("publication_before")
            publication = publish_primary(output, value, terminal)
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(publication=publication, owned_invocation_exit=0),
            )
            progress("complete", value["completed_count"], 0)
        return 0
    except (OSError, KeyError, ValueError, TimeoutError) as error:
        progress("rejected_" + str(error))
        return 1
