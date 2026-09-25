"""Run and cold-read the CPU source-atom energy experiment (REQ-REPORT-7660)."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting.experiment_7303_validation_scope import (
    CommandSpec,
    build_scoped_commands,
    reduce_required_checks,
    run_commands,
)
from carnot.reporting.experiment_7646_source_features import read_jsonl
from carnot.reporting.experiment_7660_atom_energy import action, fit_heads, join_role, score


ROOT = Path(__file__).resolve().parents[2]
RAW = Path("results/raw/experiment_7660_v668_atom_energy")
OUTPUT = Path("results/experiment_7660_v668_atom_energy.json")
UPSTREAM = Path("results/raw/experiment_7659_v668_atom_corpus")
PROTOCOL = Path("results/raw/experiment_7602_v664_evidence_requalification/protocol.json")
HEADS = RAW / "heads.json"
ROLES = ("fit", "tune", "policy")
MODEL_SPECS: list[str] = []
SEED = 7660


def _check(check: str, upstream: str, path: str, field: str, expected: Any, observed: Any) -> dict:
    return {
        "check": check,
        "upstream": upstream,
        "path": path,
        "field": field,
        "operator": "==",
        "expected": expected,
        "observed": observed,
        "passed": expected == observed,
    }


def authenticate(root: Path) -> tuple[dict, dict, list[dict], dict]:
    """Authenticate only fit/tune/policy producers and the CPU resource."""

    checks = [
        _check(
            "cpu_available",
            "host",
            "/proc/self",
            "cpu_count_positive",
            True,
            (os.cpu_count() or 0) > 0,
        )
    ]
    hashes: dict[str, Any] = {"producers": {}, "pre_gate_receipts": {}, "missing_inputs": []}
    for relative, upstream in (
        (PROTOCOL, "Exp7602"),
        (UPSTREAM / "manifest.json", "Exp7659"),
        (Path("results/experiment_7659_v668_atom_corpus.json"), "Exp7659"),
    ):
        path = root / relative
        checks.append(
            _check("input_exists", upstream, str(relative), "exists", True, path.is_file())
        )
        if path.is_file():
            hashes["producers"][str(relative)] = sha256_file(path)
        else:
            hashes["missing_inputs"].append(str(relative))
    if hashes["missing_inputs"]:
        return {}, {}, checks, hashes
    protocol = json.loads((root / PROTOCOL).read_text())
    manifest = json.loads((root / UPSTREAM / "manifest.json").read_text())
    prior = json.loads((root / "results/experiment_7659_v668_atom_corpus.json").read_text())
    checks.append(
        _check(
            "qualified_corpus",
            "Exp7659",
            "results/experiment_7659_v668_atom_corpus.json",
            "atom_features_ready_score",
            1,
            prior.get("atom_features_ready_score"),
        )
    )
    checks.append(
        _check(
            "unflagged_corpus",
            "Exp7659",
            "results/experiment_7659_v668_atom_corpus.json",
            "flagged_adversarial",
            False,
            prior.get("flagged_adversarial"),
        )
    )
    for role in ROLES:
        info = manifest["roles"][role]
        for field, relative, expected, bucket in (
            ("feature_sha256", info["feature_path"], info["feature_sha256"], "producers"),
            (
                "evaluator_sha256",
                protocol["reader_sidecars"]["evaluator_stores"][role]["path"],
                protocol["reader_sidecars"]["evaluator_stores"][role]["sha256"],
                "pre_gate_receipts",
            ),
        ):
            path = root / relative
            observed = sha256_file(path) if path.is_file() else None
            checks.append(
                _check(
                    "input_hash",
                    "Exp7659" if field == "feature_sha256" else "Exp7602",
                    relative,
                    field,
                    expected,
                    observed,
                )
            )
            if observed is None:
                hashes["missing_inputs"].append(relative)
            else:
                hashes[bucket][relative] = observed
    return protocol, manifest, checks, hashes


def load_roles(root: Path, protocol: dict, manifest: dict) -> tuple[dict, dict]:
    """Open evaluator labels only after feature-role and byte authentication."""

    roles: dict[str, list[dict]] = {}
    controls: dict[str, dict[str, dict]] = {}
    for role in ROLES:
        info = manifest["roles"][role]
        features = read_jsonl(root / info["feature_path"])
        labels = read_jsonl(root / protocol["reader_sidecars"]["evaluator_stores"][role]["path"])
        original = [row for row in features if row["arm"] == "original_source"]
        if (
            len(original) != protocol["role_counts"][role]
            or [r["unit_id"] for r in original] != info["group_ids"]
        ):
            raise ValueError("role_roster_mismatch")
        if len(features) != 3 * len(original):
            raise ValueError("feature_arm_count_invalid")
        by_arm = {(row["unit_id"], row["arm"]): row for row in features}
        if len(by_arm) != len(features):
            raise ValueError("duplicate_feature_arm")
        joined = join_role(original, labels, role)
        for item in joined:
            unit = item["feature"]["unit_id"]
            erased = by_arm[(unit, "evidence_erasure")]
            deranged = by_arm[(unit, "within_role_derangement")]
            if deranged["source_group_id"] != info["derangement"][unit]:
                raise ValueError("derangement_mismatch")
            item["erased_feature"] = erased
        roles[role] = joined
        controls[role] = by_arm
    fit_ids = set(protocol["learning_schedule"]["fit_optimization_ids"])
    anchor_ids = set(protocol["learning_schedule"]["fit_anchor_ids"])
    observed = {row["feature"]["unit_id"] for row in roles["fit"]}
    if (
        fit_ids & anchor_ids
        or observed != fit_ids | anchor_ids
        or len(fit_ids) != 64
        or len(anchor_ids) != 16
    ):
        raise ValueError("fit_partition_mismatch")
    return roles, controls


def _policy_cost(prediction: float, label: int, thresholds: tuple[float, float]) -> float:
    choice = action(prediction, thresholds)
    return 5.0 * label if choice == "accept" else 1.0 - label if choice == "reject" else 0.2


def freeze_policy(policy: list[dict], head: dict) -> tuple[tuple[float, float], list[dict]]:
    """Select typed cutoffs on policy20 using fixed costs only."""

    candidates = ((0.04, 0.8), (0.05, 0.5), (0.1, 0.9), (0.2, 0.8))
    records = []
    for thresholds in candidates:
        cost = sum(
            _policy_cost(score(row["feature"], row["probability"], head), row["label"], thresholds)
            for row in policy
        ) / len(policy)
        records.append({"thresholds": list(thresholds), "mean_cost": cost})
    chosen = min(records, key=lambda item: item["mean_cost"])
    return tuple(chosen["thresholds"]), records


def comparison_rows(
    roles: dict, controls: dict, bundle: dict, thresholds: tuple[float, float]
) -> list[dict]:
    """Emit paired arms without multiplying independent source groups."""

    result = []
    for role in ROLES:
        for item in roles[role]:
            feature = item["feature"]
            unit = feature["unit_id"]
            for arm in (
                "identity",
                "scalar",
                "cheap_atom",
                "atom",
                "source_erased",
                "source_deranged",
            ):
                selected_feature = (
                    controls[role][(unit, "within_role_derangement")]
                    if arm == "source_deranged"
                    else item["erased_feature"]
                    if arm == "source_erased"
                    else feature
                )
                head_name = bundle["selected"] if arm == "source_deranged" else arm
                p = score(selected_feature, item["probability"], bundle["heads"][head_name])
                label = item["label"]
                result.append(
                    {
                        "unit_id": unit,
                        "role": role,
                        "partition": feature["partition"],
                        "arm": arm,
                        "label": label,
                        "probability": p,
                        "baseline_probability": item["probability"],
                        "typed_action": action(p, thresholds),
                        "brier": (p - label) ** 2,
                        "decision_cost": _policy_cost(p, label, thresholds),
                        "raw_metrics": {"error_probability": p, "brier": (p - label) ** 2},
                        "counts": {"independent_group": 1, "paired_view": 1},
                        "exclusions": [],
                        "excluded": False,
                        "censored": selected_feature["censored"],
                        "provenance": {
                            "feature_sha256": selected_feature["source_sha256"],
                            "dataset_label": "Exp7602 isolated evaluator store",
                        },
                    }
                )
    return result


def cold_reduce(path: Path) -> dict:
    """Independently replay probabilities and actions from immutable inputs."""

    artifact = json.loads(path.read_text())
    for bucket in ("producers", "pre_gate_receipts"):
        for relative, expected in artifact["source_artifact_hashes"][bucket].items():
            if sha256_file(ROOT / relative) != expected:
                raise ValueError("source_hash_mismatch")
    head_path = ROOT / artifact["head_manifest_path"]
    if sha256_file(head_path) != artifact["head_manifest_sha256"]:
        raise ValueError("head_hash_mismatch")
    bundle = json.loads(head_path.read_text())
    protocol = json.loads((ROOT / PROTOCOL).read_text())
    manifest = json.loads((ROOT / UPSTREAM / "manifest.json").read_text())
    roles, controls = load_roles(ROOT, protocol, manifest)
    rows = comparison_rows(roles, controls, bundle, tuple(bundle["thresholds"]))
    if rows != artifact["rows"]:
        raise ValueError("row_reduction_mismatch")
    return {
        "passed": True,
        "independent_groups": len({r["unit_id"] for r in rows}),
        "paired_rows": len(rows),
    }


def _gate(name: str, passed: bool | None, operands: dict, principle: str) -> dict:
    return {"gate": name, "passed": passed, "measured_operands": operands, "principle": principle}


def _terminal_artifact(
    run_date: str,
    started: float,
    checks: list[dict],
    hashes: dict,
    bundle: dict,
    rows: list[dict],
    receipts: list[dict],
    spans: list[dict],
    scope: dict,
    policy_records: list[dict],
) -> dict:
    required = reduce_required_checks(receipts) if receipts else {"required_checks_passed": False}
    authenticated = all(check["passed"] for check in checks)
    valid = authenticated and required["required_checks_passed"] and len(rows) == 720
    ready = valid and bool(bundle["heads"])
    klass = "blocked" if not authenticated else "null" if valid else "disqualified"
    verdict = {
        "blocked": "complete_blocked_missing_external_evidence",
        "null": "complete_null_energy_head_ready",
        "disqualified": "complete_disqualified_required_validation",
    }[klass]
    gates = [
        _gate(
            "validity",
            valid,
            {
                "authenticated": authenticated,
                "required_checks_passed": required["required_checks_passed"],
                "paired_rows": len(rows),
            },
            "Every input and required receipt must authenticate.",
        ),
        _gate(
            "readiness",
            ready,
            {
                "frozen_heads": len(bundle["heads"]),
                "identity_fallback": bundle["selected"] == "identity",
            },
            "A valid replayable identity fallback can open evaluation.",
        ),
        _gate(
            "coverage",
            valid,
            {"covered_training_groups": 64 if rows else 0},
            "Source atoms are partial; group accounting is distinct from semantic proof.",
        ),
        _gate(
            "probability_benefit",
            None,
            {"held_out_groups_opened": 0},
            "Fit and tune diagnostics are not held-out benefit.",
        ),
        _gate(
            "utility",
            None,
            {"held_out_policy_groups": 0},
            "Policy20 freezes actions but cannot confirm utility.",
        ),
        _gate(
            "retention",
            None,
            {"delayed_feedback_events": 0},
            "Retention needs later feedback and replay.",
        ),
        _gate(
            "freshness",
            False,
            {"fresh_confirmatory_groups": 0},
            "All inherited source groups were previously exposed.",
        ),
    ]
    checksum = hashlib.sha256(
        json.dumps(
            {"hashes": hashes, "heads": bundle, "reducer": sha256_file(ROOT / __file__)},
            sort_keys=True,
        ).encode()
    ).hexdigest()
    training_rows = [
        {
            "unit_id": row["unit_id"],
            "role": row["role"],
            "partition": row["partition"],
            "objective": (
                "regularized_binary_log_loss"
                if row["partition"] == "fit_optimization"
                else "untouched_anchor_diagnostic"
                if row["role"] == "fit"
                else "brier_setting_selection"
                if row["role"] == "tune"
                else "typed_action_cost_selection"
            ),
            "arm": row["arm"],
            "chosen_configuration": bundle["selected"],
            "label": row["label"],
            "probability": row["probability"],
            "brier": row["brier"],
            "typed_action": row["typed_action"],
            "decision_cost": row["decision_cost"],
            "costs": {"accept": "5*y", "reject": "1-y", "escalate": 0.2},
        }
        for row in rows
        if row["arm"] == bundle["selected"]
    ]
    return {
        "schema": "carnot.exp7660.v668.atom_energy.v1",
        "experiment_id": "exp7660-v668-atom-energy",
        "milestone": "2026.09.668",
        "run_date": run_date,
        "honest_verdict": verdict,
        "verdict_class": klass,
        "flagged_adversarial": False,
        "gate_check_summary": [c for c in checks if not c["passed"]],
        "acceptance_gate_results": gates,
        "rows": rows,
        "sample_size_budget": {
            "intended": 120,
            "observed": len({r["unit_id"] for r in rows}),
            "eligible": len({r["unit_id"] for r in rows}),
            "excluded": 0,
            "censored": len({r["unit_id"] for r in rows if r["censored"]}),
            "prior_exposure": "All 120 inherited learning groups were previously exposed.",
            "claim_limits": "Paired arms do not enlarge N; no held-out benefit is claimed.",
        },
        "inference_substrate": "cpu_conditional_energy_on_cached_source_atoms_no_llm",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "MODEL_SPECS": MODEL_SPECS,
        "planned_MODEL_SPECS": [],
        "model_specs": [],
        "model_specs_declaration": "no model loaded or planned in current CPU work",
        "model_invoked": False,
        "invocation_counts": {
            key: 0
            for key in ("loads", "forwards", "generations", "tokens", "attempted", "cancelled")
        },
        "execution_venue": "host",
        "execution_venue_details": {
            "hostname": platform.node(),
            "owned_pid": os.getpid(),
            "gpu_uuid": None,
        },
        "phase_spans": spans,
        "duration_s": time.monotonic() - started,
        "random_seed": {"fit": SEED, "source_derangement": "Exp7659 fixed within-role mapping"},
        "reproducibility_checksum": "sha256:" + checksum,
        "source_artifact_hashes": hashes,
        "preconditions_checked": checks,
        "validation_receipts": {
            "frozen_affected_scope": scope,
            "required_commands": receipts,
            **required,
            "unrelated_repository_suite_debt": [],
        },
        "verifier_is_oracle": False,
        "field_principles": {
            "rows": "One independent group per role, with paired arms.",
            "sample_size_budget": "Arms and seeds cannot enlarge independent N.",
            "energy_ready_score": "Readiness requires valid frozen replayable heads, even identity.",
            "honest_verdict": "Completion is separate from scientific benefit.",
            "flagged_adversarial": "A flagged terminal reader closes readiness.",
            "duration_s": "Actual CPU duration; no model floor applies.",
            "inference_substrate_class": "No current model load or generation.",
        },
        "energy_ready_score": int(ready),
        "head_manifest_path": str(HEADS),
        "head_manifest_sha256": sha256_file(ROOT / HEADS) if (ROOT / HEADS).is_file() else None,
        "target_is_witness_output": False,
        "training_rows": training_rows,
        "training_summary": {
            "fit64_objective": "regularized binary log loss",
            "fit16_anchor_groups": 16,
            "tune20_settings": bundle["settings"],
            "selected_arm": bundle["selected"],
            "policy20_costs": {"accept": "5*y", "reject": "1-y", "escalate": 0.2},
            "policy20_threshold_candidates": policy_records,
        },
        "historical_model_id": "unsloth/Qwen3.8-27B-GGUF",
        "retirement": "Retire unchanged V667 source/claim grammar after repeated zero-check failure; external absence is not scientific disproof.",
    }


def run_experiment(root: Path, run_date: str, output: Path) -> dict:  # pragma: no cover - task E2E
    """Fit, validate, cold-read, and atomically publish one terminal artifact."""

    root = root.resolve()
    started = time.monotonic()
    raw = root / RAW
    raw.mkdir(parents=True, exist_ok=True)
    spans: list[dict] = []

    def progress(phase: str, event: str, detail: str = "") -> None:
        print(
            f"[exp7660] {phase} {event} elapsed_s={time.monotonic() - started:.3f} {detail}",
            flush=True,
        )

    def close(phase: str, begin: float, units: int, checkpoint: Path) -> None:
        end = time.monotonic() - started
        spans.append(
            {
                "phase": phase,
                "start_offset_s": begin,
                "end_offset_s": end,
                "duration_s": end - begin,
                "completed_units": units,
                "heartbeat_times_s": [begin, end],
                "checkpoint": str(checkpoint.relative_to(root)),
            }
        )
        progress(phase, "complete", f"units={units}")

    progress("preconditions", "start", f"root={root}")
    begin = time.monotonic() - started
    protocol, manifest, checks, hashes = authenticate(root)
    atomic_json(raw / "preconditions.json", {"checks": checks, "hashes": hashes})
    close("preconditions", begin, len(checks), raw / "preconditions.json")
    tests = ["tests/python/test_experiment_7660_v668_atom_energy.py"]
    modules = ["python/carnot/reporting/experiment_7660_atom_energy.py"]
    static = [
        "python/carnot/experiment_7660_v668_atom_energy.py",
        "scripts/experiments/experiment_7660_v668_atom_energy.py",
    ]
    bundle: dict = {"heads": {}, "settings": [], "selected": "identity"}
    rows: list[dict] = []
    policy_records: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="exp7660-", dir="/tmp") as private_dir:
        private = Path(private_dir)
        basetemp = private / "basetemp"
        basetemp.mkdir()
        commands = build_scoped_commands(
            root,
            tests,
            modules,
            static_paths=static,
            basetemp=basetemp,
            coverage_file=private / ".coverage",
        )
        commands.append(
            CommandSpec(
                "orchestration_mypy",
                (str(root / ".venv/bin/mypy"), static[0]),
                "changed_orchestration",
            )
        )
        scope = {
            "tests": tests,
            "changed_modules": modules,
            "static_paths": static,
            "commands": [{"name": cmd.name, "argv": list(cmd.argv)} for cmd in commands],
        }
        atomic_json(raw / "frozen_validation_manifest.json", scope)
        progress("fit", "start")
        begin = time.monotonic() - started
        if all(check["passed"] for check in checks):
            roles, controls = load_roles(root, protocol, manifest)
            fit_ids = set(protocol["learning_schedule"]["fit_optimization_ids"])
            fit = [row for row in roles["fit"] if row["feature"]["unit_id"] in fit_ids]
            if len(fit) != 64 or len(roles["tune"]) != 20 or len(roles["policy"]) != 20:
                raise ValueError("learning_role_count_invalid")
            bundle = fit_heads(fit, roles["tune"])
            chosen = bundle["heads"][bundle["selected"]]
            thresholds, policy_records = freeze_policy(roles["policy"], chosen)
            bundle["thresholds"] = list(thresholds)
            bundle["fit_group_ids"] = [r["feature"]["unit_id"] for r in fit]
            bundle["anchor_group_ids"] = [
                r["feature"]["unit_id"]
                for r in roles["fit"]
                if r["feature"]["unit_id"] not in fit_ids
            ]
            bundle["derangement"] = {role: manifest["roles"][role]["derangement"] for role in ROLES}
            atomic_json(root / HEADS, bundle)
            rows = comparison_rows(roles, controls, bundle, thresholds)
            atomic_json(
                raw / "fit_checkpoint.json",
                {
                    "completed_groups": 120,
                    "head_sha256": sha256_file(root / HEADS),
                    "paired_rows": len(rows),
                },
            )
        else:
            atomic_json(
                raw / "fit_checkpoint.json",
                {"completed_groups": 0, "blocked_checks": [c for c in checks if not c["passed"]]},
            )
        close("fit", begin, len(rows) // 6, raw / "fit_checkpoint.json")
        progress("affected_validation", "before_subprocess")
        begin = time.monotonic() - started
        receipts = run_commands(
            root,
            commands,
            log_dir=raw / "validation/affected",
            extra_env={"JAX_PLATFORMS": "cpu", "COVERAGE_FILE": str(private / ".coverage")},
            heartbeat_s=30,
        )
        close("affected_validation", begin, len(receipts), raw / "validation/affected")
        artifact = _terminal_artifact(
            run_date, started, checks, hashes, bundle, rows, receipts, spans, scope, policy_records
        )
        candidate = raw / "exact_terminal_candidate.json"
        atomic_json(candidate, artifact)
        python = str(root / ".venv/bin/python")
        readers = [
            CommandSpec(
                "cold_reduce",
                (
                    python,
                    "-u",
                    "-m",
                    "carnot.experiment_7660_v668_atom_energy",
                    "--cold-reduce",
                    str(candidate),
                ),
                "exact_candidate",
            ),
            CommandSpec(
                "adversarial_verify",
                (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
                "exact_candidate",
            ),
            CommandSpec(
                "verdict_row_consistency",
                (
                    python,
                    "-u",
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(candidate),
                ),
                "exact_candidate",
            ),
        ]
        progress("terminal_readers", "before_subprocess")
        begin = time.monotonic() - started
        terminal = run_commands(
            root,
            readers,
            log_dir=raw / "validation/terminal",
            extra_env={"JAX_PLATFORMS": "cpu"},
            heartbeat_s=30,
        )
        close("terminal_readers", begin, len(terminal), raw / "validation/terminal")
        artifact["validation_receipts"]["terminal_readers"] = terminal
        artifact["validation_receipts"]["exact_terminal_candidate_sha256"] = sha256_file(candidate)
        artifact["flagged_adversarial"] = not terminal[1]["passed"]
        if not all(r["passed"] for r in terminal):
            artifact["verdict_class"] = "disqualified"
            artifact["honest_verdict"] = "complete_disqualified_terminal_reader"
            artifact["energy_ready_score"] = 0
            for gate in artifact["acceptance_gate_results"]:
                if gate["gate"] in {"validity", "readiness", "coverage"}:
                    gate["passed"] = False
        artifact["duration_s"] = time.monotonic() - started
        progress("publish", "before_atomic_write")
        atomic_json(output, artifact)
        progress("publish", "complete", f"verdict={artifact['honest_verdict']}")
        return artifact


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260925")
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--cold-reduce", type=Path)
    args = parser.parse_args(argv)
    if args.cold_reduce:
        print(json.dumps(cold_reduce(args.cold_reduce), sort_keys=True), flush=True)
        return 0
    root = ROOT.resolve()
    run_experiment(root, args.date, root / args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - exercised through CLI
    raise SystemExit(main())
