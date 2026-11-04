"""REQ-REPORT-8025: persist causal updates and measure later issued decisions.

This CPU experiment uses historically exposed cached sources. A complete null
trajectory prepares measurement, but cannot establish deployment learning.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8019_v695_eligible_targets as eligible
from carnot import experiment_8020_v695_qualified_energy_fit as fitted
from carnot.experiment_8007_v694_conditioning_diagnosis import copy_evidence
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import causal_online_8025 as m

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8025_v695_causal_online_updates"
TASK = "exp8025-causal-online-updates"
MODEL_SPECS: list[str] = []
OWNED = [
    f"python/carnot/{NAME}.py",
    "python/carnot/verify/causal_online_8025.py",
    f"scripts/experiments/{NAME}.py",
]
TEST = "tests/python/test_causal_online_8025.py"


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Freeze calibrated starting state before opening the public stream manifest."""
    refs, failures, producers = [], [], {}
    for name, field in (
        (fitted.NAME, "energy_fit_ready_score"),
        (eligible.NAME, "stream_targets_ready_score"),
    ):
        path = root / "results" / (name + ".json")
        value = json.loads(path.read_text()) if path.is_file() else {}
        gate = dict(
            upstream_id=value.get("task_id", name),
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            expected=1,
            observed=value.get(field, "MISSING_CONTRACT_FIELD"),
            passed=value.get(field) == 1
            and value.get("verdict_class") in {"null", "positive"}
            and value.get("flagged_adversarial") is False,
        )
        if not gate["passed"]:
            failures.append(gate)
        else:
            refs.append(copy_evidence(reference(path), raw))
        producers[name] = value
    if failures:
        return dict(references=refs), failures
    try:
        primary, targets = producers[fitted.NAME], producers[eligible.NAME]
        if not targets["support_by_role"]["stream"]["passed"]:
            return dict(references=refs), [
                dict(
                    gate,
                    artifact_field="support_by_role.stream.passed",
                    expected=True,
                    observed=False,
                    passed=False,
                )
            ]
        m.progress("small_head_load_before")
        heads = [(r, json.loads(checked(r).read_text())) for r in primary["head_checkpoints"]]
        ref, h = next(
            (r, h) for r, h in heads if h["arm"] == "conditioned_energy" and h["seed"] == 17
        )
        calibration_ref = copy_evidence(primary["calibration_checkpoint"], raw)
        calibration = json.loads(checked(calibration_ref).read_text())["maps"][
            "conditioned_energy-17"
        ]
        if not h["converged"] or not calibration["converged"] or len(h["parameters"]) != 110:
            raise ValueError("initial_head_contract")
        head = dict(
            arm=h["arm"],
            parameters=h["parameters"],
            geometry=h["geometry"],
            decay_scale=1.0,
            calibration=calibration["parameters"],
        )
        refs += [copy_evidence(ref, raw), calibration_ref]
        atomic_json(raw / "starting_head.json", dict(head=head, config=m.CONFIG))
        m.progress("small_head_load_after")
        public_ref = copy_evidence(targets["public_manifests"]["stream"], raw)
        sources = json.loads(checked(public_ref).read_text())["rows"]
        if len(sources) != 256 or [r["slot"] for r in sources] != list(range(256)):
            raise ValueError("original_stream_slots")
        target_ref = copy_evidence(targets["evaluator_manifests"]["stream"], raw)
        refs += [public_ref, target_ref, copy_evidence(targets["exclusion_manifest"], raw)]
        for row in targets["historical_failure_logs"]:
            refs.append(copy_evidence(dict(path=row["path"], sha256=row["hash"]), raw))
        return dict(
            head=head,
            sources=sources,
            target_reference=target_ref,
            seeds=m.CONFIG["seeds"],
            references=refs,
        ), []
    except (ValueError, KeyError, OSError, StopIteration) as error:
        return dict(references=refs), [
            dict(
                gate,
                artifact_field="calibrated_head_and_stream_contract",
                expected="valid immutable head and original supported slots",
                observed=str(error),
                passed=False,
            )
        ]


def base(failures: list[Json]) -> Json:
    """Terminal metadata gives null science and failed prerequisites distinct roles."""
    return dict(
        experiment_id=8025,
        task_id=TASK,
        milestone="2026.10.695",
        run_date="20261002",
        schema="carnot.v695.causal_online_updates.v1",
        honest_verdict="complete_blocked_causal_online_updates"
        if failures
        else "complete_null_causal_online_updates",
        verdict_class="blocked" if failures else "null",
        claim_scope="One historically exposed development stream; schedule sensitivity is not independent datasets. Generator fixed; retention unopened; no deployment or retention benefit claim.",
        gate_check_summary=failures,
        random_seed=69525,
        algorithm_seeds=m.CONFIG["seeds"],
        verifier_is_oracle=False,
        flagged_adversarial=False,
        acceptance_gate_results=dict(trajectory=False, owned_checks=False, natural_benefit=False),
        genuine_headroom=dict(measured=False),
        positive_control_results={},
        generalized_learning_benefit_score=0,
        learning_measurement_ready_score=0,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        continuous_self_learning_task=True,
        retained_labels_opened=False,
        rows=[],
        issued_rows=[],
        released_feedback_rows=[],
        update_rows=[],
        per_arm_budget={},
        checkpoint_sequence=[],
        overlap_rows=[],
        later_cost_rows=[],
        hot_update_ns=[],
        durable_transaction_ns=[],
        bytes_written=0,
        checkpoint_references=[],
        sample_size_budget=dict(
            intended=256,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=256,
            independent=0,
            seeds_are_independent=False,
        ),
        cited_upstream_artifacts=[],
        raw_shard_hashes=[],
        code_config_hashes=[],
        validation_receipts=[],
        coverage_statement_counts={},
        repository_health=[],
        phase_spans=[],
        config=m.CONFIG,
        methodology_note="Calibrated BCE sparse gradients use only due feedback. Loss priorities use the original issued decision and probability, never a recomputed training loss. CPU arithmetic and synchronous transactions have separate clocks. No time-series or hardware speed claim is imported.",
    )


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse bounded repository validation instead of duplicating its supervision."""
    commands = eligible.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(
        config.read_text().replace(eligible.NAME, NAME) + "    " + str(ROOT / OWNED[1]) + "\n"
    )
    return [
        replace(
            c,
            argv=tuple(a.replace(eligible.NAME, NAME).replace(eligible.TEST, TEST) for a in c.argv),
        )
        for c in commands
    ]


def apply_validation(value: Json, receipts: list[Json], counts: Json) -> None:
    """Readiness requires actual owned checks; broad health is reported separately."""
    owned = [r for r in receipts if r["scope"] == "owned"]
    good = bool(owned) and all(r["passed"] for r in owned) and len(counts) == len(OWNED)
    good = good and all(
        r["num_statements"] > 0 and r["missing_lines"] == 0 for r in counts.values()
    )
    value["validation_receipts"] = owned
    value["repository_health"] = [r for r in receipts if r["scope"] == "repository_health"]
    value["coverage_statement_counts"] = counts
    value["acceptance_gate_results"]["owned_checks"] = good
    if not good and value["verdict_class"] != "blocked":
        value.update(
            verdict_class="disqualified",
            honest_verdict="complete_disqualified_causal_online_updates",
        )
    value["learning_measurement_ready_score"] = int(
        good and value["acceptance_gate_results"]["trajectory"] and value["verdict_class"] == "null"
    )


def replay(path: Path) -> Json:
    """A fresh process recomputes metrics without trusting producer aggregates."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["acceptance_gate_results"]["trajectory"]:
        reduced = m.reduce(Path(value["trajectory_directory"]))
        for k, v in reduced.items():
            if value[k] != v:
                raise ValueError("reduction_drift:" + k)
    if value["learning_measurement_ready_score"] and (
        value["verdict_class"] != "null" or not value["acceptance_gate_results"]["owned_checks"]
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True)


def terminal(path: Path) -> Json:
    """Existing adversarial and row readers inspect the exact candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Run the original stream or isolated CLI fixtures with bounded owned checks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        print(json.dumps(replay(args.cold_replay)), flush=True)
        return 0
    started = time.monotonic()
    output = (args.output or args.root / "results" / (NAME + ".json")).absolute()
    raw = output.parent / "raw" / output.stem
    raw.mkdir(parents=True, exist_ok=True)
    m.progress("preconditions_before")
    with tempfile.TemporaryDirectory(prefix="carnot-8025-") as temporary:
        scratch = Path(temporary)
        commands = validation_plan(scratch) if not args.validation_worker else []
        atomic_json(
            raw / "validation_manifest.json",
            dict(
                commands=[asdict(c) for c in commands], config=m.CONFIG, artifact_guard_enabled=True
            ),
        )
        data, failures = (
            (json.loads(args.fixture_input.read_text()), [])
            if args.fixture_input
            else load_inputs(args.root, raw)
        )
        value = base(failures)
        value["cited_upstream_artifacts"] = data.get("references", [])
        m.progress("preconditions_after")
        if not failures:
            measured = time.monotonic()
            value.update(m.measure(data, raw))
            value["trajectory_directory"] = str(raw)
            value["phase_spans"].append(
                dict(phase="trajectory", duration_s=time.monotonic() - measured)
            )
            value["acceptance_gate_results"]["trajectory"] = True
            value["acceptance_gate_results"].update(
                delayed_release=True,
                equal_update_budget=True,
                durable_reload=True,
                unknown_targets_not_negative=True,
                generator_fixed=True,
                retained_labels_unopened=True,
            )
            value["trained_head_specs"] = [
                dict(
                    arm=arm,
                    seed=seed,
                    parameters=110,
                    pretrained=False,
                    optimizer="calibrated_BCE_sparse_SGD",
                    current_updates=budget["updates"],
                )
                for label, budget in value["per_arm_budget"].items()
                for arm, seed in [(label.split("/")[0], int(label.split("/")[1]))]
            ]
            value["checkpoint_references"] = list(
                {r["sha256"]: r for r in value["checkpoint_sequence"]}.values()
            )
            value["genuine_headroom"] = dict(
                measured=True,
                scope="released later decisions",
                frozen_cost=sum(
                    r["actual_cost"]
                    for r in value["later_cost_rows"]
                    if r["arm"] == "frozen_no_write"
                    and r["seed"] == data["seeds"][0]
                    and r["eligibility"]
                ),
            )
            value["positive_control_results"] = dict(
                scope="isolated numerical and artificial causal controls; not natural benefit",
                sparse_dense_equivalence=True,
                future_mutation=True,
                positive_decision_transition=True,
                oracle_controls_are_natural_evidence=False,
            )
            value["hot_update_measurement"] = dict(
                target_ns=1000,
                target_is_not_a_promise=True,
                samples=len(value["hot_update_ns"]),
                observed_min_ns=min(value["hot_update_ns"], default=None),
                below_target_count=sum(n < 1000 for n in value["hot_update_ns"]),
                scope="CPU sparse arithmetic only; gradient construction, snapshots and durability excluded",
            )
        if args.fixture_input:
            value["verifier_is_oracle"] = True
            value["claim_scope"] = (
                "Artificial CLI protocol fixture; no natural or generalized benefit."
            )
        if not args.validation_worker:
            m.progress("owned_validation_before")
            receipts = run_commands(
                ROOT,
                commands,
                log_dir=raw / "validation_logs",
                heartbeat_s=30,
                extra_env=dict(
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                ),
            )
            coverage = (
                json.loads((scratch / "coverage.json").read_text())
                if (scratch / "coverage.json").is_file()
                else {}
            )
            counts = {p: r["summary"] for p, r in coverage.get("files", {}).items() if p in OWNED}
            apply_validation(value, receipts, counts)
            m.progress("owned_validation_after")
        value["raw_shard_hashes"] = [
            reference(p)
            for p in raw.rglob("*")
            if p.is_file() and (p.suffix in {".json", ".sqlite"} or p.name.endswith(".log"))
        ]
        value["code_config_hashes"] = [reference(ROOT / p) for p in OWNED]
        value["code_config_hashes"] += [
            reference(Path(module.__file__))
            for module in (
                m.conditioned,
                fitted,
                eligible,
                __import__("carnot.verify.sparse_energy_7996", fromlist=["sparse"]),
                __import__("carnot.experiment_8021_v695_typed_decision_test", fromlist=["action"]),
            )
        ]
        value["reproducibility_checksum"] = canonical_hash(
            dict(config=m.CONFIG, raw=value["raw_shard_hashes"], code=value["code_config_hashes"])
        )
        value["duration_s"] = time.monotonic() - started
        value["phase_spans"].append(
            dict(phase="invocation_through_owned_checks", duration_s=value["duration_s"])
        )
        value["terminal_validation_sidecar_path"] = str(raw / "terminal_validation.json")
        value["field_principles"] = {
            k: f"Record {k} for this owned invocation; preserve its numerator, denominator and scope."
            for k in value
        }
        value["field_principles"].update(
            honest_verdict="Terminal completion describes owned work; null does not mean unfinished.",
            learning_measurement_ready_score="One requires a complete valid trajectory and all owned checks; benefit is separate.",
            generalized_learning_benefit_score="Historical development exposure cannot establish deployment benefit.",
            sample_size_budget="Count original slots and source groups; twenty schedules do not multiply independent n.",
            issued_rows="Commit current probabilities, actions and state hashes before releasing any feedback at that slot.",
            released_feedback_rows="The evaluator releases only origin plus twenty; unknowns are not negatives.",
            update_rows="Only selected past eligible IDs drive identical calibrated gradients and sparse writes.",
            later_cost_rows="Score issued later decisions against frozen no-write on the same released labels.",
            hot_update_ns="CPU arithmetic excludes gradients, snapshots and synchronous transactions; one microsecond is only a target.",
            durable_transaction_ns="Synchronous SQLite transaction wall time includes commit overhead separately.",
            checkpoint_references="Every published head checkpoint remains content-addressed under durable raw evidence.",
            overlap_rows="Spline support overlap and past counterfactual transitions diagnose locality without retention labels.",
            model_invocation_counts="Current pretrained loads and generations are zero; CPU small-head updates are separate.",
            repository_health="One bounded broad diagnostic cannot replace or redefine required owned checks.",
        )
        m.progress("publication_before")
        receipt = publish_primary(
            output, value, lambda p: replay(p) if args.validation_worker else terminal(p)
        )
        post = replay(output) if args.validation_worker else terminal(output)
        readers = reader_receipt(
            TASK,
            output.parent,
            field="learning_measurement_ready_score",
            expected=value["learning_measurement_ready_score"],
        )
        atomic_json(
            raw / "terminal_validation.json",
            dict(publication=receipt, published=post, readers=readers),
        )
        if not post["passed"] or not readers["passed"]:
            raise ValueError("published_validation_failed")
        m.progress("publication_after")
    return 0
