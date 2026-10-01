"""REQ-REPORT-7998: randomized delayed feedback over cached public sources.

Only small spline coefficients change. This producer completes a mechanism and
raw trajectories; independent Exp7999 must decide natural benefit and retention.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_7997_v693_typed_development_decisions as upstream
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import selective_feedback_7998 as m
from carnot.verify import sparse_energy_7996 as sparse

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME = "experiment_7998_v693_selective_feedback_learning"
TASK = "exp7998-selective-feedback-learning"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/selective_feedback_7998.py",
    "python/carnot/reporting/selective_validation_7998.py",
    f"scripts/experiments/{NAME}.py",
]
TESTS = ["tests/python/test_selective_feedback_7998.py", f"tests/python/test_{NAME}.py"]
reference, checked = upstream.reference, upstream.checked


def progress(phase: str) -> None:
    """Flushed boundaries expose real progress without extending measured work."""
    print(f"[exp7998] phase={phase}", flush=True)


def authenticate(root: Path) -> tuple[list[Json], Json]:
    """Check named role gates as exact fields after immutable producer hashes.

    Missing fields are contract errors. A present zero is a failed external
    prerequisite and records its own producer path, hash and comparison.
    """
    failures, plan = upstream.authenticate(root)
    if failures:
        return failures, plan
    for eid, fields in {
        7995: ["calibration_capture_ready_score", "stream_capture_ready_score"],
        7996: [],
    }.items():
        producer = plan["upstream"][eid]
        path = root / "results" / (upstream.INPUTS[eid][0] + ".json")
        for field in fields + ["verdict_class"]:
            if field not in producer:
                raise ValueError("upstream_contract:" + field)
            expected = ["positive", "circular_positive", "null"] if field == "verdict_class" else 1
            check = upstream.custody.operand(eid, path, field, expected, producer[field])
            if field == "verdict_class":
                check.update(op="in", passed=producer[field] in expected)
            plan["checks"].append(check)
        next(r for r in plan["refs"] if r["producer_id"] == eid)["imported_fields"] += fields + [
            "verdict_class"
        ]
    return [r for r in plan["checks"] if not r["passed"]], plan


def past_shuffle(reveals: list[Json]) -> tuple[Any, list[Json]]:
    """Choose deterministic diagnostic labels only from the already due prefix.

    Reversing the complete stream would assign future-origin targets to early
    slots. Each recorded sampled origin instead stays at or before its receipt.
    """
    indexed = {r["family_id"]: r for r in reveals}
    rows: list[Json] = []

    def label(identity: str) -> int | None:
        requested = indexed[identity]
        available = [r for r in reveals if r["origin_slot"] <= requested["origin_slot"]]
        draw = int(canonical_hash(dict(identity=identity, seed=69398)).split(":")[1][:13], 16)
        sampled = available[draw % len(available)]
        rows.append(
            dict(
                family_id=identity,
                origin_slot=requested["origin_slot"],
                due_slot=requested["due_slot"],
                sampled_origin_slot=sampled["origin_slot"],
                sampled_family_id=sampled["family_id"],
                numerator=sampled["y"],
                denominator=int(sampled["y"] is not None),
                eligibility=sampled["y"] is not None,
                failure_status=False,
                censor_status=sampled["y"] is None,
            )
        )
        return sampled["y"]  # type: ignore[no-any-return]

    return label, rows


def health_receipt(path: Path) -> Json:
    """Authenticate a prior real suite receipt without rerunning the full suite.

    Its exit status and original log remain historical diagnostics. Importing
    them never asserts that this invocation ran or passed that command.
    """
    value = json.loads(path.read_text())
    receipt = value["receipt"]
    expected = [str(ROOT / ".venv/bin/pytest"), "tests/python", "-q"]
    if receipt["command_argv"] != expected or receipt["log_sha256"] != sha256_file(
        Path(receipt["log_path"])
    ):
        raise ValueError("health_receipt_drift")
    checked(value["producer_configuration"])
    return dict(
        receipt,
        imported_from_prior_attempt=True,
        producer_execution_date=value["producer_execution_date"],
        source_receipt=reference(path),
        scope="prior_repository_health_attempt",
    )


def base(failures: list[Json]) -> Json:
    """Use established terminal metadata while naming this invocation's scope."""
    value = upstream.base(failures)
    value.update(
        experiment_id=7998,
        task_id=TASK,
        schema="carnot.selective_feedback_learning.v1",
        honest_verdict="complete_blocked_selective_feedback_learning"
        if failures
        else "complete_null_selective_feedback_learning",
        random_seed=69398,
        config=m.CONFIG,
        learning_measurement_ready_score=0,
        claim_scope="Completed randomized delayed-feedback mechanism over fallible source-support labels. Natural benefit and retention are reserved for Exp7999. Public pretraining and wider historical exposure remain unknown.",
        replay_scope="One recorded development dataset; acquisition schedules are algorithm repetitions, not independent sources.",
        algorithm_seeds=list(range(101, 121)),
        effective_independent_sources=0,
        acquisition_rows=[],
        reveal_rows=[],
        update_rows=[],
        checkpoint_rows=[],
        expected_label_budget={},
        realized_label_counts={},
        inverse_propensity_weights=[],
        future_label_mutation_rows=[],
        issued_predictions=[],
        pending_feedback=[],
        coefficient_touches=0,
        durable_write_bytes=0,
        mechanism_checks={},
        checkpoints={},
    )
    return value


def measure(heads: Json, public: Json, labels: Json, raw: Path, fixture: bool = False) -> Json:
    """Seal calibration candidates and acquisition before the stream label vault.

    The callback keeps the full evaluator roster outside the learner. It opens
    stream targets only on the first due selected receipt, after durable issue.
    """
    value = base([])
    head = copy.deepcopy(next(h for h in heads["spline"] if h["seed"] == 17))
    head["temperature"] = 1.0
    progress("calibration_prediction_begin")
    candidates = {
        str(t): [
            float(sparse.predict(dict(head, temperature=t), sparse.inputs([r]))[0])
            if r["status"] == "completed" and r["q"] is not None and r["features"] is not None
            else None
            for r in public["calibration"]
        ]
        for t in m.CONFIG["temperatures"]
    }
    atomic_json(
        raw / "calibration_candidates.json",
        dict(head=head, public=public["calibration"], candidates=candidates),
    )
    progress("calibration_candidates_sealed_target_open")
    calibration = upstream.read_targets(labels["calibration"])
    scores = []
    for t in m.CONFIG["temperatures"]:
        errors = [
            (p - r["y"]) ** 2
            for p, r in zip(candidates[str(t)], calibration, strict=True)
            if r["y"] is not None and p is not None
        ]
        scores.append(dict(temperature=t, numerator=sum(errors), denominator=len(errors)))
    head["temperature"] = min(
        enumerate(scores), key=lambda item: (item[1]["numerator"] / item[1]["denominator"], item[0])
    )[1]["temperature"]
    atomic_json(raw / "frozen_head.json", head)
    atomic_json(raw / "public_stream.json", dict(rows=public["stream"]))
    seeds = [101] if fixture else m.CONFIG["algorithm_seeds"]
    acquisition = [r for seed in seeds for r in m.acquire(head, public["stream"], seed)]
    atomic_json(raw / "acquisition.json", dict(rows=acquisition))
    value.update(calibration_rows=scores, acquisition_rows=acquisition, algorithm_seeds=seeds)
    for name in ("calibration_candidates", "frozen_head", "public_stream", "acquisition"):
        value["checkpoints"][name] = reference(raw / (name + ".json"))
    value["label_access_events"] = [
        dict(
            role="calibration",
            purpose="temperature_only",
            preceded_by=value["checkpoints"]["calibration_candidates"],
        )
    ]
    vault: Json = {}

    def label(identity: str) -> int | None:
        if not vault:
            vault.update({r["family_id"]: r["y"] for r in upstream.read_targets(labels["stream"])})
            value["label_access_events"].append(
                dict(
                    role="stream",
                    purpose="due_selected_callback_only",
                    acquisition_seal=value["checkpoints"]["acquisition"],
                )
            )
        return vault[identity]  # type: ignore[no-any-return]

    progress("acquisition_frozen_benchmark_begin")
    trajectories: Json = {}
    for seed in seeds:
        for arm in m.ARMS:
            schedule = [r for r in acquisition if r["seed"] == seed and r["arm"] == arm]
            key = f"{arm}-{seed}"
            result = m.stream(head, public["stream"], schedule, label, raw / "states" / key)
            atomic_json(
                raw / (key + ".json"),
                dict(trajectory=result, state_directory=str(raw / "states" / key)),
            )
            trajectories[key] = result
            value["checkpoints"][key] = reference(raw / (key + ".json"))
            for field in ("issued_predictions", "reveal_rows", "update_rows", "checkpoint_rows"):
                value[field].extend(result[field])
            value["pending_feedback"].extend(
                dict(r, arm=arm, seed=seed) for r in result["pending_feedback"]
            )
            value["durable_write_bytes"] += result["durable_write_bytes"]
            progress("checkpoint_" + key)
    progress("benchmark_end_mechanism_audits_begin")
    first = trajectories["targeted_ipw-101"]
    selected = [r for r in acquisition if r["arm"] == "targeted_ipw" and r["seed"] == 101]
    duplicate_state = copy.deepcopy(first["final_state"])
    duplicate_before = canonical_hash(duplicate_state)
    duplicate_receipt = first["reveal_rows"][0]
    duplicate = m.apply_feedback(
        duplicate_state, duplicate_receipt, public["stream"][duplicate_receipt["origin_slot"]]
    )
    duplicate_ok = duplicate is None and canonical_hash(duplicate_state) == duplicate_before
    known = {r["family_id"]: r["y"] for r in first["reveal_rows"]}

    def observed(identity: str) -> int | None:
        return known[identity]  # type: ignore[no-any-return]

    auditdir = raw / "audits"
    never_targets = dict(known)
    never_targets.update({r["family_id"]: -1 for r in selected if not r["selected"]})
    never = m.stream(
        head, public["stream"], selected, never_targets.__getitem__, auditdir / "never_selected"
    )
    never_ok = never == first
    m.stream(head, public["stream"], selected, observed, auditdir / "restart", stop_at=128)
    restart = m.stream(head, public["stream"], selected, observed, auditdir / "restart")
    dense = m.stream(head, public["stream"], selected, observed, auditdir / "dense", dense=True)
    mutated = dict(known)
    for r in selected:
        if r["slot"] >= 128 and r["family_id"] in mutated and mutated[r["family_id"]] is not None:
            mutated[r["family_id"]] = 1 - mutated[r["family_id"]]
    future = m.stream(head, public["stream"], selected, mutated.__getitem__, auditdir / "future")
    future_ok = future["issued_predictions"][:149] == first["issued_predictions"][:149]
    value["future_label_mutation_rows"] = [
        dict(
            cutoff_origin_slot=128,
            compared_predictions=149,
            numerator=int(future_ok),
            denominator=1,
            eligibility=True,
            failure_status=False,
            censor_status=False,
        )
    ]
    shuffled_label, shuffle_receipts = past_shuffle(first["reveal_rows"])
    shuffle = m.stream(head, public["stream"], selected, shuffled_label, auditdir / "past_shuffle")
    atomic_json(
        raw / "past_label_shuffle_diagnostic.json",
        dict(
            scope="diagnostic_only_due_past_selected_labels",
            shuffle_receipts=shuffle_receipts,
            trajectory=shuffle,
        ),
    )
    value["past_label_shuffle_diagnostic"] = reference(raw / "past_label_shuffle_diagnostic.json")
    parity = max(
        abs(a - b)
        for a, b in zip(first["parity_parameters"], dense["parity_parameters"], strict=True)
    )
    value["positive_control_results"] = m.controls()
    value["mechanism_checks"] = dict(
        restart_at_128=restart == first,
        dense_gradient_error=parity,
        future_label_mutation_invariant=future_ok,
        never_selected_mutation_invariant=never_ok,
        duplicate_feedback_idempotent=duplicate_ok,
        passed=restart == first
        and parity < 1e-10
        and future_ok
        and never_ok
        and duplicate_ok
        and value["positive_control_results"]["passed"],
    )
    value["rows"] = value["issued_predictions"]
    value["coefficient_touches"] = sum(r["coefficient_touches"] for r in value["update_rows"])
    value["inverse_propensity_weights"] = [
        dict(arm=r["arm"], seed=r["seed"], source=r["family_id"], weight=r["weight"])
        for r in value["update_rows"]
    ]
    for arm in m.ARMS:
        schedules = [r for r in acquisition if r["arm"] == arm]
        value["expected_label_budget"][arm] = dict(
            per_seed=sum(r["pi"] for r in schedules) / len(seeds),
            all_schedules=sum(r["pi"] for r in schedules),
            matching="expected_not_realized",
            equal_cost_competitor=arm != "full_feedback",
        )
        value["realized_label_counts"][arm] = dict(
            selected=sum(r["selected"] for r in schedules),
            revealed=sum(r["arm"] == arm for r in value["reveal_rows"]),
            updated=sum(r["arm"] == arm for r in value["update_rows"]),
            by_seed={
                str(seed): sum(r["selected"] for r in schedules if r["seed"] == seed)
                for seed in seeds
            },
        )
    sources = public["stream"]
    eligible = [
        r
        for r in acquisition
        if r["seed"] == 101 and r["arm"] == "targeted_ipw" and r["eligibility"]
    ]
    value["effective_independent_sources"] = len({r["source_cluster_id"] for r in eligible})
    value["sample_size_budget"] = dict(
        intended=len(sources),
        eligible=len(eligible),
        started=sum(r["status"] != "excluded" for r in sources),
        completed=len(eligible),
        excluded=sum(r["status"] == "excluded" for r in sources),
        failed=sum(r["status"] == "failed" for r in sources),
        censored=sum(r["status"] == "censored" for r in sources),
        independent=value["effective_independent_sources"],
        seeds_are_independent=False,
    )
    value["trained_head_specs"] = [
        dict(
            arm=arm,
            acquisition_seeds=seeds,
            pretrained=False,
            initial_seed=17,
            parameter_count=109,
            optimizer="online_sparse_gradient_descent",
            temperature=head["temperature"],
            learning_rate=0.01,
            l2=0.001,
            fitted_current_steps=sum(r["arm"] == arm for r in value["update_rows"]),
        )
        for arm in m.ARMS
    ]
    frozen_predictions = trajectories["frozen_no_write-101"]["issued_predictions"]
    baseline_costs = [
        (
            0.25
            if frozen_predictions[r["origin_slot"]]["action"] == "escalate"
            else (
                5 * r["y"]
                if frozen_predictions[r["origin_slot"]]["action"] == "accept"
                else 1 - r["y"]
            )
        )
        for r in trajectories["full_feedback-101"]["reveal_rows"]
        if r["y"] is not None
    ]
    value["genuine_headroom"] = sum(baseline_costs) > 0
    value["headroom_scope"] = dict(
        role="stream",
        source="due_full_feedback_only",
        numerator=sum(baseline_costs),
        denominator=len(baseline_costs),
        benefit_inference=False,
    )
    value["benefit_assessed"] = False
    value["verifier_is_oracle"] = fixture
    return value


def replay(value: Json) -> Json:
    """Reconstruct predictions from frozen coefficients and due receipt rows.

    Hash checks bind producer shards; sequential recomputation rejects a changed
    row even when an attacker adjusts an aggregate to conceal the change.
    """
    if value["retention_labels_opened"] or (
        value["verdict_class"] in ("blocked", "disqualified")
        and value["learning_measurement_ready_score"]
    ):
        raise ValueError("unsafe_readiness")
    if not value["checkpoints"]:
        return dict(passed=True, blocked=True)
    for ref in (
        list(value["checkpoints"].values())
        + value["code_config_hashes"]
        + value["raw_shard_hashes"]
    ):
        checked(ref)
    head = json.loads(checked(value["checkpoints"]["frozen_head"]).read_text())
    sources = json.loads(checked(value["checkpoints"]["public_stream"]).read_text())["rows"]
    acquisition = [r for seed in value["algorithm_seeds"] for r in m.acquire(head, sources, seed)]
    if acquisition != value["acquisition_rows"]:
        raise ValueError("acquisition_drift")
    combined: Json = {
        k: [] for k in ("issued_predictions", "reveal_rows", "update_rows", "checkpoint_rows")
    }
    for seed in value["algorithm_seeds"]:
        for arm in m.ARMS:
            trajectory = json.loads(checked(value["checkpoints"][f"{arm}-{seed}"]).read_text())[
                "trajectory"
            ]
            state = dict(head=copy.deepcopy(head), next_slot=0, seen_label_ids=[])
            reveals = {r["due_slot"]: r for r in trajectory["reveal_rows"]}
            actual_updates = []
            for t, prediction in enumerate(trajectory["issued_predictions"]):
                state["next_slot"] = t
                if prediction["head_checksum"] != canonical_hash(state["head"]):
                    raise ValueError("head_drift")
                p = (
                    float(sparse.predict(state["head"], sparse.inputs([sources[t]]))[0])
                    if prediction["eligibility"]
                    else None
                )
                if p != prediction["probability"] or m.action(p) != prediction["action"]:
                    raise ValueError("prediction_drift")
                if t in reveals:
                    update = m.apply_feedback(state, reveals[t], sources[t - 20])
                    if update is not None:
                        actual_updates.append(update)
            if (
                actual_updates != trajectory["update_rows"]
                or state["head"] != trajectory["final_state"]["head"]
            ):
                raise ValueError("update_drift")
            for field in combined:
                combined[field].extend(trajectory[field])
    if (
        any(combined[k] != value[k] for k in combined)
        or value["rows"] != value["issued_predictions"]
    ):
        raise ValueError("trajectory_drift")
    return dict(passed=True, rows=len(value["rows"]))


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Owned failures zero readiness; historical full-suite health stays explicit."""
    coverage_ok = set(counts) == set(OWNED) and all(
        c["num_statements"] > 0 and c["missing_lines"] == 0 for c in counts.values()
    )
    value.update(
        validation_receipts=receipts,
        coverage_statement_counts=counts,
        repository_health=dict(
            current=[
                r
                for r in receipts
                if not r["required"] and not r.get("imported_from_prior_attempt")
            ],
            historical=[r for r in receipts if r.get("imported_from_prior_attempt")],
        ),
    )
    if any(not r["passed"] for r in receipts if r["required"]) or (
        receipts
        and value["checkpoints"]
        and (not coverage_ok or not value["mechanism_checks"]["passed"])
    ):
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_selective_feedback_learning",
            learning_measurement_ready_score=0,
        )
    elif receipts and coverage_ok and value["checkpoints"] and not value["verifier_is_oracle"]:
        value["learning_measurement_ready_score"] = 1
    value["acceptance_gate_results"] = dict(
        readiness=bool(value["learning_measurement_ready_score"]),
        benefit=False,
        independent_benefit_audit="Exp7999",
    )


def terminal_check(candidate: Path) -> Json:
    """Unmodified independent validators inspect actual candidate bytes."""
    replay(json.loads(candidate.read_text()))
    commands = [
        CommandSpec(
            name,
            (str(ROOT / ".venv/bin/python"), "-u", str(ROOT / script), flag, str(candidate)),
            "terminal",
            60,
        )
        for name, script, flag in [
            ("adversarial", "scripts/adversarial_verify.py", "--json"),
            ("strict_rows", "scripts/verdict_row_consistency_lint.py", "--strict"),
        ]
    ]
    receipts = run_commands(
        ROOT, commands, log_dir=candidate.parent / "terminal_logs", heartbeat_s=15
    )
    return dict(
        passed=all(r["passed"] for r in receipts),
        receipts=receipts,
        flagged_adversarial=not receipts[0]["passed"],
    )


def publish(output: Path, value: Json, scratch: Path) -> None:
    """Only checked final bytes become the primary selected by both readers."""
    raw = output.parent / "raw" / output.stem
    value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
    value["field_principles"] = {
        k: "Bind current work to raw evidence; readiness does not assert benefit." for k in value
    }
    candidate = scratch / "candidate.json"
    atomic_json(candidate, value)
    first = terminal_check(candidate)
    if not first["passed"]:
        value["flagged_adversarial"] = first["flagged_adversarial"]
        value["historical_required_failures"].append(first)
        apply_validation(
            value,
            value["validation_receipts"]
            + [dict(name="terminal_failure", required=True, passed=False)],
            value["coverage_statement_counts"],
        )
    receipt = publish_primary(output, value, terminal_check)
    atomic_json(raw / "terminal_validation.json", receipt)
    reader = reader_receipt(
        TASK,
        output.parent,
        field="learning_measurement_ready_score",
        expected=value["learning_measurement_ready_score"],
    )
    atomic_json(raw / "primary_resolution.json", reader)
    if (
        not reader["passed"]
        or reader["gate_path"] != str(output)
        or reader["gate_sha256"] != sha256_file(output)
    ):
        raise ValueError("primary_resolution")


def main(argv: list[str] | None = None) -> int:
    """Run real private or natural CLI paths with bounded child supervision."""
    from carnot.reporting import selective_validation_7998 as validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261001", choices=["20261001"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--repository-health-receipt", type=Path)
    parser.add_argument("--check-health-receipt", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin")
    try:
        if args.check_health_receipt:
            health_receipt(args.check_health_receipt)
            progress("prior_health_receipt_verified")
            return 0
        if args.cold_replay:
            replay(json.loads(args.cold_replay.read_text()))
            progress("replay_passed")
            return 0
        scratch = Path(tempfile.mkdtemp(prefix="carnot-7998-"))
        raw = scratch / "evidence"
        frozen = [reference(ROOT / p) for p in OWNED + TESTS]
        atomic_json(raw / "task_configuration.json", dict(config=m.CONFIG, code=frozen))
        manifest = (
            validation.freeze(raw, scratch, args.repository_health_receipt)
            if not args.validation_worker
            else {}
        )
        progress("configuration_and_commands_frozen")
        if args.fixture_input:
            fixture = json.loads(args.fixture_input.read_text())
            heads, public, labels = fixture["heads"], fixture["public"], {}
            for role, targets in fixture["targets"].items():
                atomic_json(raw / "evaluator" / (role + ".json"), dict(rows=targets))
                labels[role] = reference(raw / "evaluator" / (role + ".json"))
            failures, plan = [], dict(checks=[], refs=[], upstream={})
        else:
            failures, plan = authenticate(args.root)
            if not failures:
                heads, public, labels, roles = upstream.load_public(plan)
        value = (
            base(failures)
            if failures
            else measure(heads, public, labels, raw, bool(args.fixture_input))
        )
        value.update(
            preconditions_checked=plan["checks"],
            cited_upstream_artifacts=plan["refs"],
            code_config_hashes=frozen,
        )
        if manifest:
            receipts = validation.execute(manifest, raw)
            value["validation_command_manifest_path"] = str(raw / "validation_manifest.json")
            apply_validation(value, receipts, validation.coverage_counts(scratch))
        value["raw_shard_hashes"] = list(value["checkpoints"].values()) + [
            reference(raw / "task_configuration.json")
        ]
        if not failures:
            value["raw_shard_hashes"] += list(labels.values()) + [
                value["past_label_shuffle_diagnostic"]
            ]
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=m.CONFIG, code=frozen, raw=value["raw_shard_hashes"], upstream=plan["refs"])
        )
        value["duration_s"] = time.monotonic() - began
        value["phase_spans"] = [
            dict(
                phase="configuration_measurement_and_owned_validation",
                duration_s=value["duration_s"],
            )
        ]
        value["duration_scope"] = (
            "Invocation through owned checks; final-byte validator durations remain in the sidecar."
        )
        progress("publish_begin")
        publish(args.output.absolute(), value, scratch)
        progress("publish_end")
        return 0
    except (ValueError, OSError, KeyError, TypeError) as error:
        print(f"[exp7998] failed={error}", flush=True)
        return 1
