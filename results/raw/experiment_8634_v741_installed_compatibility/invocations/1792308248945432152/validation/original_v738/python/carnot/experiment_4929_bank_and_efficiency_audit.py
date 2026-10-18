"""Experiment 4929: adversarial audit for .454 ARC banks and efficiency."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import yaml


JsonDict = dict[str, Any]
RepoLoader = Callable[[Path], Mapping[str, Any]]
LintRunner = Callable[[Path], Mapping[str, Any]]
Clock = Callable[[], float]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4929_bank_and_efficiency_audit"
EXPERIMENT_ID = 4929
SCHEMA = "carnot.v454_bank_and_efficiency_audit.v1"
RESULT_RELATIVE_PATH = "results/experiment_4929_bank_and_efficiency_audit.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
A1_RELATIVE_PATH = "results/experiment_4925_levelup_attempt.json"
A2_RELATIVE_PATH = "results/experiment_4926_levelup_attempt.json"
EFFICIENCY_RELATIVE_PATH = "results/experiment_4933_matm_similarity_retrieval_efficiency.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4929

SPEC_REFS = [
    "REQ-CAPSTONE-4929",
    "SCENARIO-CAPSTONE-4929",
    "SCENARIO-CAPSTONE-4929-BLOCKED-PRECONDITION",
]

CHECK_KEYS = (
    "reproduction_genuine",
    "not_duplicate",
    "self_discovery_provenance",
    "live_path_reachable",
    "oracle_distinct",
    "honest_ab",
)

BANK_CHECK_KEYS = CHECK_KEYS[:4]
EFFICIENCY_CHECK_KEYS = CHECK_KEYS[4:]

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_v454_banks_and_efficiency_audited "
            "(trusted or with named failures)."
        )
    },
    "banks_trustworthy": {
        "principle": (
            "AND of the bank checks -- the capstone counts A1/A2 toward "
            "reproducible_total_levels ONLY if true."
        )
    },
    "efficiency_trustworthy": {
        "principle": (
            "AND of the efficiency checks -- the capstone reports D's "
            "action-efficiency lift ONLY if true (oracle-distinct + honest A/B)."
        )
    },
    "checks": {
        "principle": (
            "per-check booleans {reproduction_genuine, not_duplicate, "
            "self_discovery_provenance, live_path_reachable, oracle_distinct, honest_ab}."
        )
    },
    "audit_failure_reasons": {
        "principle": (
            "list of named failures (empty if all trusted) -- the audit reports "
            "honestly, no rubber-stamp."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (reads cached artifacts; 1s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records A1/A2/D artifact presence; a missing input is recorded, not fabricated."
        )
    },
}

REQUIRED_FIELDS = (
    "honest_verdict",
    "banks_trustworthy",
    "efficiency_trustworthy",
    "checks",
    "audit_failure_reasons",
    "inference_substrate",
    "preconditions_checked",
)


def _read_json(path: Path) -> JsonDict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return value


def _load_json_if_present(root: Path, relative: str) -> JsonDict | None:
    path = root / relative
    if not path.exists():
        return None
    return _read_json(path)


def _load_registry(root: Path) -> Mapping[str, Any]:
    value = yaml.safe_load((root / REGISTRY_RELATIVE_PATH).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("registry did not contain a mapping")
    return value


def _registry_game_row(registry: Mapping[str, Any], game: str) -> Mapping[str, Any]:
    for row in registry.get("games", []) or []:
        if isinstance(row, Mapping) and row.get("game") == game:
            return row
    return {}


def _int_value(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _artifact_game(artifact: Mapping[str, Any]) -> str:
    gate = _mapping(artifact.get("reproduction_gate"))
    return str(artifact.get("target_game") or artifact.get("game") or gate.get("game") or "")


def _claimed_reached_level(artifact: Mapping[str, Any]) -> int:
    gate = _mapping(artifact.get("reproduction_gate"))
    update = _mapping(artifact.get("registry_update"))
    return _int_value(
        gate.get("claimed_level")
        or gate.get("reached_level")
        or update.get("new_game_levels")
        or artifact.get("reproduced_levels")
    )


def _gate_reached_level(artifact: Mapping[str, Any]) -> int:
    gate = _mapping(artifact.get("reproduction_gate"))
    update = _mapping(artifact.get("registry_update"))
    return _int_value(
        gate.get("reached_level")
        or update.get("new_game_levels")
        or artifact.get("reproduced_levels")
    )


def _loop_reached_level(loop_artifact: Mapping[str, Any]) -> int:
    gate = _mapping(loop_artifact.get("reproduction_gate"))
    return _int_value(gate.get("reached_level") or loop_artifact.get("reached_level"))


def _loop_reproduced(loop_artifact: Mapping[str, Any]) -> bool:
    gate = _mapping(loop_artifact.get("reproduction_gate"))
    return bool(loop_artifact.get("offline_reproduced") is True and gate.get("reproduced") is True)


def _has_outer_loop_inputs(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            lowered = str(key).lower()
            if lowered in {
                "offline_ground_truth_bfs",
                "per_game_calibration_inputs",
                "calibration_inputs",
                "outer_loop_re",
            } and item:
                return True
            if _has_outer_loop_inputs(item):
                return True
    elif isinstance(value, list):
        return any(_has_outer_loop_inputs(item) for item in value)
    return False


def _live_game_adapter_evidence(game: str, row: Mapping[str, Any]) -> bool:
    solver_text = str(row.get("solver") or "")
    if "GameAdapter" in solver_text:
        return True
    try:
        from carnot.agentic import arc_game_adapters

        return arc_game_adapters.get_adapter(game) is not None
    except Exception:  # pragma: no cover - import-environment fallback
        return False


def run_arc_orphan_solver_lint(root: Path) -> JsonDict:
    command = [sys.executable, "scripts/arc_orphan_solver_lint.py"]
    completed = subprocess.run(
        command,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "passed": completed.returncode == 0,
        "returncode": completed.returncode,
        "command": " ".join(command),
        "stdout_tail": completed.stdout[-1000:],
        "stderr_tail": completed.stderr[-1000:],
    }


def audit_bank(
    *,
    label: str,
    artifact: Mapping[str, Any],
    loop_artifact: Mapping[str, Any],
    registry: Mapping[str, Any],
    lint_result: Mapping[str, Any],
) -> JsonDict:
    game = _artifact_game(artifact)
    row = _registry_game_row(registry, game)
    registry_prior = _int_value(row.get("levels_reproduced"))
    claimed_level = _claimed_reached_level(artifact)
    gate_reached = _gate_reached_level(artifact)
    gate = _mapping(artifact.get("reproduction_gate"))
    loop_gate = _mapping(loop_artifact.get("reproduction_gate"))
    loop_level = _loop_reached_level(loop_artifact)
    expected_loop_path = f"results/arc_loop_solve_{game}.json"
    loop_path = str(artifact.get("standing_loop_result_path") or "")
    mode = str(loop_artifact.get("mode") or "")

    reproduction_genuine = bool(
        game
        and gate.get("reproduced") is True
        and _loop_reproduced(loop_artifact)
        and gate.get("game") == game
        and loop_artifact.get("game") == game
        and loop_gate.get("game") == game
        and claimed_level > 0
        and gate_reached == claimed_level
        and loop_level == claimed_level
    )
    not_duplicate = bool(claimed_level > registry_prior)
    provenance = str(artifact.get("solve_provenance") or "")
    no_outer_loop_inputs = not _has_outer_loop_inputs(artifact)
    self_discovery_provenance = provenance == "live_agent_self_discovery" and no_outer_loop_inputs
    live_path_reachable = bool(
        artifact.get("live_path_reachable") is True
        and loop_path == expected_loop_path
        and mode.startswith("standing_arc_loop")
        and lint_result.get("passed") is True
        and _live_game_adapter_evidence(game, row)
    )

    checks = {
        "reproduction_genuine": reproduction_genuine,
        "not_duplicate": not_duplicate,
        "self_discovery_provenance": self_discovery_provenance,
        "live_path_reachable": live_path_reachable,
    }
    reasons: list[str] = []
    if not reproduction_genuine:
        reasons.append(f"{label}_reproduction_genuine_failed_loop_or_gate_mismatch_{game}")
    if not not_duplicate:
        reasons.append(f"{label}_not_duplicate_failed_duplicate_depth_{game}_L{claimed_level}")
    if not self_discovery_provenance:
        cause = provenance or "missing"
        if provenance == "live_agent_self_discovery" and not no_outer_loop_inputs:
            cause = "declared_outer_loop_input"
        reasons.append(f"{label}_self_discovery_provenance_failed_{cause}")
    if not live_path_reachable:
        reasons.append(f"{label}_live_path_reachable_failed")

    return {
        "label": label,
        "present": True,
        "artifact_experiment": artifact.get("experiment"),
        "game": game,
        "checks": checks,
        "failure_reasons": reasons,
        "registry_prior_level": registry_prior,
        "claimed_reached_level": claimed_level,
        "artifact_gate_reached_level": gate_reached,
        "loop_cross_check": {
            "path": expected_loop_path,
            "artifact_path": loop_path,
            "game": loop_artifact.get("game"),
            "mode": mode,
            "offline_reproduced": loop_artifact.get("offline_reproduced") is True,
            "gate_reproduced": loop_gate.get("reproduced") is True,
            "reached_level": loop_level,
        },
        "provenance": {
            "solve_provenance": provenance,
            "outer_loop_inputs_declared": not no_outer_loop_inputs,
        },
        "live_path_evidence": {
            "artifact_live_path_reachable": artifact.get("live_path_reachable") is True,
            "standing_loop_path_matches": loop_path == expected_loop_path,
            "arc_orphan_solver_lint_passed": lint_result.get("passed") is True,
            "game_adapter_evidence": _live_game_adapter_evidence(game, row),
        },
    }


def _critical_circular_moat_flags(artifact: Mapping[str, Any]) -> list[JsonDict]:
    try:
        import scripts.adversarial_verify as adversarial_verify

        flags: list[Any] = []
        adversarial_verify.check_circular_moat_overclaim(dict(artifact), flags)
        return [
            flag.to_dict()
            for flag in flags
            if getattr(flag, "severity", None) == "critical"
        ]
    except Exception as exc:  # pragma: no cover - verifier import/runtime fallback
        return [{"kind": "CIRCULAR_MOAT_CHECK_ERROR", "severity": "critical", "detail": str(exc)}]


def audit_efficiency(artifact: Mapping[str, Any]) -> JsonDict:
    circular_flags = _critical_circular_moat_flags(artifact)
    oracle_distinct = bool(artifact.get("verifier_is_oracle") is False and not circular_flags)

    baseline_ok = bool(
        artifact.get("baseline_kind") == "submitted_exact_hash"
        and artifact.get("baseline_hash_matches_submitted") is True
        and artifact.get("submitted_exact_hash_baseline") is True
    )
    regression_ok = bool(
        artifact.get("zero_reached_level_regression") is True
        and not artifact.get("reached_level_regressions")
    )
    parity_ok = artifact.get("parity_test_green") is True
    leak = _mapping(artifact.get("leak_check"))
    leak_ok = bool(
        leak.get("passed") is True
        and leak.get("leak_detected") is not True
        and leak.get("same_state_target_shortcut") is not True
        and artifact.get("same_state_target_shortcut") is not True
    )
    null_honest = artifact.get("null_reported_honestly")
    if null_honest is None:
        text = " ".join(
            str(artifact.get(key) or "")
            for key in ("honest_verdict", "efficiency_disposition", "retire_disposition")
        ).lower()
        null_honest = "null" in text or "retire" in text
    honest_ab = bool(baseline_ok and regression_ok and parity_ok and leak_ok and null_honest is True)

    reasons: list[str] = []
    if not oracle_distinct:
        if artifact.get("verifier_is_oracle") is not False:
            reasons.append("D_oracle_distinct_failed_verifier_is_oracle_not_false")
        else:
            reasons.append("D_oracle_distinct_failed_circular_moat_overclaim")
    if not baseline_ok:
        reasons.append("D_honest_ab_failed_baseline_not_submitted_exact_hash")
    if not regression_ok:
        reasons.append("D_honest_ab_failed_reached_level_regression")
    if not parity_ok:
        reasons.append("D_honest_ab_failed_parity_test_red")
    if not leak_ok:
        reasons.append("D_honest_ab_failed_leak_or_same_state_shortcut")
    if null_honest is not True:
        reasons.append("D_honest_ab_failed_null_not_reported_honestly")

    return {
        "present": True,
        "artifact_experiment": artifact.get("experiment"),
        "checks": {"oracle_distinct": oracle_distinct, "honest_ab": honest_ab},
        "failure_reasons": reasons,
        "oracle_evidence": {
            "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            "circular_moat_critical_flags": circular_flags,
        },
        "honest_ab_evidence": {
            "baseline_kind": artifact.get("baseline_kind"),
            "baseline_hash_matches_submitted": artifact.get("baseline_hash_matches_submitted"),
            "submitted_exact_hash_baseline": artifact.get("submitted_exact_hash_baseline"),
            "zero_reached_level_regression": artifact.get("zero_reached_level_regression"),
            "reached_level_regressions": artifact.get("reached_level_regressions") or [],
            "parity_test_green": artifact.get("parity_test_green"),
            "leak_check": dict(leak),
            "same_state_target_shortcut": artifact.get("same_state_target_shortcut"),
            "null_reported_honestly": null_honest,
        },
    }


def _preconditions(root: Path, registry_loader: RepoLoader | None) -> tuple[JsonDict, Mapping[str, Any] | None]:
    checked: JsonDict = {
        "a1_artifact_present": (root / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root / A2_RELATIVE_PATH).exists(),
        "d_artifact_present": (root / EFFICIENCY_RELATIVE_PATH).exists(),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "adversarial_verify_present": (root / "scripts/adversarial_verify.py").exists(),
        "summarize_artifact_present": (root / "scripts/summarize_artifact.py").exists(),
        "arc_orphan_solver_lint_present": (root / "scripts/arc_orphan_solver_lint.py").exists(),
        "spec_has_req_4929": (
            "REQ-CAPSTONE-4929"
            in (root / "openspec/capabilities/capstone/spec.md").read_text(encoding="utf-8")
            if (root / "openspec/capabilities/capstone/spec.md").exists()
            else False
        ),
    }
    registry: Mapping[str, Any] | None = None
    if checked["registry_present"]:
        try:
            registry = registry_loader(root) if registry_loader is not None else _load_registry(root)
            checked["registry_loadable"] = True
        except Exception as exc:
            checked["registry_loadable"] = False
            checked["registry_error"] = str(exc)
    else:
        checked["registry_loadable"] = False
    checked["absent_inputs"] = _absent_inputs(checked)
    return checked, registry


def _absent_inputs(checked: Mapping[str, Any]) -> list[str]:
    labels = {
        "a1_artifact_present": "experiment_4925_levelup_attempt",
        "a2_artifact_present": "experiment_4926_levelup_attempt",
        "d_artifact_present": "experiment_4933_matm_similarity_retrieval_efficiency",
        "registry_present": "arc_solve_registry",
        "adversarial_verify_present": "scripts_adversarial_verify",
        "summarize_artifact_present": "scripts_summarize_artifact",
        "arc_orphan_solver_lint_present": "scripts_arc_orphan_solver_lint",
        "spec_has_req_4929": "capstone_spec_req_4929",
    }
    return [label for key, label in labels.items() if checked.get(key) is not True]


def _blocked_verdict(checked: Mapping[str, Any]) -> str | None:
    ordered = (
        ("a1_artifact_present", "blocked_experiment_4925_levelup_attempt_missing"),
        ("a2_artifact_present", "blocked_experiment_4926_levelup_attempt_missing"),
        (
            "d_artifact_present",
            "blocked_experiment_4933_matm_similarity_retrieval_efficiency_missing",
        ),
        ("registry_present", "blocked_arc_solve_registry_missing"),
        ("registry_loadable", "blocked_arc_solve_registry_unloadable"),
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4929", "blocked_capstone_spec_req_4929_missing"),
    )
    for key, verdict in ordered:
        if checked.get(key) is not True:
            return verdict
    return None


def _missing_bank_evidence(label: str, reason: str) -> JsonDict:
    return {
        "label": label,
        "present": False,
        "checks": {key: False for key in BANK_CHECK_KEYS},
        "failure_reasons": [reason],
    }


def _missing_efficiency_evidence(reason: str) -> JsonDict:
    return {
        "present": False,
        "checks": {key: False for key in EFFICIENCY_CHECK_KEYS},
        "failure_reasons": [reason],
    }


def _aggregate_checks(bank_evidence: Mapping[str, Any], efficiency_evidence: Mapping[str, Any]) -> JsonDict:
    checks: JsonDict = {}
    for key in BANK_CHECK_KEYS:
        checks[key] = all(
            _mapping(evidence.get("checks")).get(key) is True
            for evidence in bank_evidence.values()
            if isinstance(evidence, Mapping)
        )
    efficiency_checks = _mapping(efficiency_evidence.get("checks"))
    for key in EFFICIENCY_CHECK_KEYS:
        checks[key] = efficiency_checks.get(key) is True
    return checks


def _collect_failure_reasons(*evidence_groups: Any) -> list[str]:
    out: list[str] = []
    for group in evidence_groups:
        if isinstance(group, Mapping) and "failure_reasons" in group:
            values = [group]
        else:
            values = group.values() if isinstance(group, Mapping) else [group]
        for evidence in values:
            if isinstance(evidence, Mapping):
                out.extend(str(item) for item in evidence.get("failure_reasons", []) or [])
    return out


def _checksum_payload(artifact: Mapping[str, Any]) -> str:
    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"reproducibility_checksum", "schema_errors"}
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return "sha256:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _with_checksum_and_schema(artifact: JsonDict) -> JsonDict:
    artifact = dict(artifact)
    artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    artifact["schema_errors"] = artifact_schema_errors(artifact)
    if artifact["schema_errors"]:
        artifact["reproducibility_checksum"] = _checksum_payload(artifact)
    return artifact


def _load_loop_for_bank(root: Path, artifact: Mapping[str, Any]) -> JsonDict | None:
    relative = str(artifact.get("standing_loop_result_path") or "")
    if not relative:
        return None
    return _load_json_if_present(root, relative)


def run(
    *,
    root: Path = REPO,
    write: bool = True,
    registry_loader: RepoLoader | None = None,
    lint_runner: LintRunner | None = None,
    now: Clock = time.monotonic,
) -> JsonDict:
    root = Path(root)
    start = now()
    checked, registry = _preconditions(root, registry_loader)
    lint = dict((lint_runner or run_arc_orphan_solver_lint)(root)) if checked.get(
        "arc_orphan_solver_lint_present"
    ) else {"passed": False, "reason": "missing"}

    bank_evidence: JsonDict = {}
    if registry is None:
        bank_evidence["A1"] = _missing_bank_evidence("A1", "registry_unavailable_for_A1")
        bank_evidence["A2"] = _missing_bank_evidence("A2", "registry_unavailable_for_A2")
    else:
        for label, relative in (("A1", A1_RELATIVE_PATH), ("A2", A2_RELATIVE_PATH)):
            artifact = _load_json_if_present(root, relative)
            if artifact is None:
                bank_evidence[label] = _missing_bank_evidence(
                    label,
                    f"{label}_missing_{Path(relative).stem}",
                )
                continue
            loop = _load_loop_for_bank(root, artifact)
            if loop is None:
                bank_evidence[label] = _missing_bank_evidence(
                    label,
                    f"{label}_missing_loop_artifact",
                )
                continue
            bank_evidence[label] = audit_bank(
                label=label,
                artifact=artifact,
                loop_artifact=loop,
                registry=registry,
                lint_result=lint,
            )

    efficiency_artifact = _load_json_if_present(root, EFFICIENCY_RELATIVE_PATH)
    if efficiency_artifact is None:
        efficiency_evidence = _missing_efficiency_evidence(
            "D_missing_experiment_4933_matm_similarity_retrieval_efficiency"
        )
    else:
        efficiency_evidence = audit_efficiency(efficiency_artifact)
        if registry is None:
            efficiency_evidence = dict(efficiency_evidence)
            efficiency_evidence["checks"] = {
                **dict(_mapping(efficiency_evidence.get("checks"))),
                "honest_ab": False,
            }
            efficiency_evidence["failure_reasons"] = [
                *list(efficiency_evidence.get("failure_reasons") or []),
                "D_honest_ab_failed_registry_unavailable",
            ]

    checks = _aggregate_checks(bank_evidence, efficiency_evidence)
    banks_trustworthy = all(checks[key] is True for key in BANK_CHECK_KEYS)
    efficiency_trustworthy = all(checks[key] is True for key in EFFICIENCY_CHECK_KEYS)
    failure_reasons = _collect_failure_reasons(bank_evidence, efficiency_evidence)
    failure_reasons.extend(
        f"missing_precondition_{item}" for item in checked.get("absent_inputs", []) or []
    )
    verdict = _blocked_verdict(checked)
    if verdict is None:
        verdict = (
            "complete_v454_banks_and_efficiency_audited_trusted"
            if banks_trustworthy and efficiency_trustworthy
            else "complete_v454_banks_and_efficiency_audited_with_named_failures"
        )

    artifact = _with_checksum_and_schema(
        {
            "experiment": EXPERIMENT,
            "experiment_id": EXPERIMENT_ID,
            "schema": SCHEMA,
            "spec_refs": list(SPEC_REFS),
            "result_path": RESULT_RELATIVE_PATH,
            "field_principles": dict(FIELD_PRINCIPLES),
            "honest_verdict": verdict,
            "banks_trustworthy": banks_trustworthy,
            "efficiency_trustworthy": efficiency_trustworthy,
            "checks": checks,
            "audit_failure_reasons": failure_reasons,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "preconditions_checked": checked,
            "bank_evidence": bank_evidence,
            "efficiency_evidence": efficiency_evidence,
            "lint_evidence": lint,
            "adversarial_verifier_evidence": {
                "check_circular_moat_overclaim_used_for_D": efficiency_artifact is not None,
                "adversarial_verify_path": "scripts/adversarial_verify.py",
                "summarize_artifact_path": "scripts/summarize_artifact.py",
            },
            "duration_s": max(1.0, round(now() - start, 6)),
            "random_seed": RANDOM_SEED,
        }
    )
    if write:
        write_artifact(artifact, root=root)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    if artifact.get("experiment") != EXPERIMENT:
        errors.append("experiment mismatch")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("experiment_id mismatch")
    if artifact.get("schema") != SCHEMA:
        errors.append("schema mismatch")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs mismatch")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles mismatch")
    if artifact.get("result_path") != RESULT_RELATIVE_PATH:
        errors.append("result_path mismatch")
    if type(artifact.get("banks_trustworthy")) is not bool:
        errors.append("banks_trustworthy must be bare bool")
    if type(artifact.get("efficiency_trustworthy")) is not bool:
        errors.append("efficiency_trustworthy must be bare bool")
    checks = artifact.get("checks")
    if not isinstance(checks, Mapping) or set(checks) != set(CHECK_KEYS) or not all(
        type(checks.get(key)) is bool for key in CHECK_KEYS
    ):
        errors.append("checks must contain the six required bare booleans")
    if not isinstance(artifact.get("audit_failure_reasons"), list):
        errors.append("audit_failure_reasons must be a list")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    verdict = str(artifact.get("honest_verdict") or "")
    if not (verdict.startswith("complete_") or verdict.startswith("blocked_")):
        errors.append("honest_verdict must use a terminal prefix")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum.startswith("sha256:") or len(checksum) != 71:
        errors.append("reproducibility_checksum must be sha256-prefixed")
    elif checksum != _checksum_payload(artifact):
        errors.append("reproducibility_checksum mismatch")
    if artifact.get("schema_errors") not in (None, []):
        errors.append("schema_errors must be empty")
    return errors


def write_artifact(artifact: Mapping[str, Any], *, root: Path = REPO) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    path = Path(root) / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    del argv
    artifact = run(root=REPO, write=True)
    print(json.dumps({"result_path": RESULT_RELATIVE_PATH, "honest_verdict": artifact["honest_verdict"]}))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main(sys.argv[1:]))
