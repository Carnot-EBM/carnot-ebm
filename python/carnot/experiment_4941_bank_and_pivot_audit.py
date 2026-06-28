"""Experiment 4941: adversarial audit for .455 ARC banks and pivot readiness."""

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
EXPERIMENT = "experiment_4941_bank_and_pivot_audit"
EXPERIMENT_ID = 4941
SCHEMA = "carnot.v455_bank_and_pivot_audit.v1"
RESULT_RELATIVE_PATH = "results/experiment_4941_bank_and_pivot_audit.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
A1_RELATIVE_PATH = "results/experiment_4936_levelup_attempt.json"
A2_RELATIVE_PATH = "results/experiment_4937_levelup_attempt.json"
PIVOT_RELATIVE_PATH = "results/experiment_4940_distributional_energy_verifier_executable_spec.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4941

SPEC_REFS = [
    "REQ-CAPSTONE-4941",
    "SCENARIO-CAPSTONE-4941",
    "SCENARIO-CAPSTONE-4941-BLOCKED-PRECONDITION",
]

CHECK_KEYS = (
    "reproduction_genuine",
    "not_duplicate",
    "self_discovery_provenance",
    "live_path_reachable",
    "oracle_distinct_design",
    "honest_readiness",
)

BANK_CHECK_KEYS = CHECK_KEYS[:4]
PIVOT_CHECK_KEYS = CHECK_KEYS[4:]
REQUIRED_ARXIV_IDS = {"2605.18871", "2504.16828", "2502.01989"}

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_v455_banks_and_pivot_audited "
            "(trusted or with named failures)."
        )
    },
    "banks_trustworthy": {
        "principle": (
            "AND of the bank checks over CLAIMED banks -- the capstone counts "
            "A1/A2 toward reproducible_total_levels ONLY if true (vacuously true "
            "if no bank claimed)."
        )
    },
    "pivot_readiness_trustworthy": {
        "principle": (
            "AND of the pivot-readiness checks -- the capstone states D's "
            "readiness ONLY if true (oracle-distinct design + honest gate + no over-claim)."
        )
    },
    "checks": {
        "principle": (
            "per-check booleans {reproduction_genuine, not_duplicate, "
            "self_discovery_provenance, live_path_reachable, "
            "oracle_distinct_design, honest_readiness}."
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
    "pivot_readiness_trustworthy",
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


def _loop_reached_level(loop_artifact: Mapping[str, Any] | None) -> int:
    gate = _mapping((loop_artifact or {}).get("reproduction_gate"))
    return _int_value(gate.get("reached_level") or (loop_artifact or {}).get("reached_level"))


def _loop_reproduced(loop_artifact: Mapping[str, Any] | None) -> bool:
    gate = _mapping((loop_artifact or {}).get("reproduction_gate"))
    return bool(
        loop_artifact is not None
        and loop_artifact.get("offline_reproduced") is True
        and gate.get("reproduced") is True
    )


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


def _claims_bank(artifact: Mapping[str, Any]) -> bool:
    verdict = str(artifact.get("honest_verdict") or "")
    return bool(
        _int_value(artifact.get("new_levels_banked")) > 0
        or artifact.get("offline_reproduced") is True
        or verdict.startswith("success_")
    )


def _registry_prior_level(artifact: Mapping[str, Any], registry: Mapping[str, Any], game: str) -> int:
    update = _mapping(artifact.get("registry_update"))
    if "prior_game_levels" in update:
        return _int_value(update.get("prior_game_levels"))
    return _int_value(_registry_game_row(registry, game).get("levels_reproduced"))


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
    loop_artifact: Mapping[str, Any] | None,
    registry: Mapping[str, Any],
    lint_result: Mapping[str, Any],
) -> JsonDict:
    game = _artifact_game(artifact)
    row = _registry_game_row(registry, game)
    bank_claimed = _claims_bank(artifact)
    if not bank_claimed:
        return {
            "label": label,
            "present": True,
            "bank_claimed": False,
            "artifact_experiment": artifact.get("experiment"),
            "game": game,
            "checks": {key: True for key in BANK_CHECK_KEYS},
            "failure_reasons": [],
            "claim_status": "no_bank_claimed_honest_dead_end",
            "claimed_reached_level": _claimed_reached_level(artifact),
            "registry_prior_level": _registry_prior_level(artifact, registry, game),
            "loop_artifact_present": loop_artifact is not None,
        }

    registry_prior = _registry_prior_level(artifact, registry, game)
    claimed_level = _claimed_reached_level(artifact)
    gate_reached = _gate_reached_level(artifact)
    gate = _mapping(artifact.get("reproduction_gate"))
    loop_gate = _mapping((loop_artifact or {}).get("reproduction_gate"))
    loop_level = _loop_reached_level(loop_artifact)
    expected_loop_path = f"results/arc_loop_solve_{game}.json"
    loop_path = str(artifact.get("standing_loop_result_path") or "")
    mode = str((loop_artifact or {}).get("mode") or "")

    reproduction_genuine = bool(
        game
        and artifact.get("offline_reproduced") is True
        and gate.get("reproduced") is True
        and _loop_reproduced(loop_artifact)
        and gate.get("game") == game
        and (loop_artifact or {}).get("game") == game
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
    if loop_artifact is None:
        reasons.append(f"{label}_missing_loop_artifact")
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
        "bank_claimed": True,
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
            "game": (loop_artifact or {}).get("game"),
            "mode": mode,
            "offline_reproduced": (loop_artifact or {}).get("offline_reproduced") is True,
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
        repo_text = str(REPO)
        if repo_text not in sys.path:  # pragma: no cover - direct script execution guard
            sys.path.insert(0, repo_text)
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


def _verifier_design_principle_declared(artifact: Mapping[str, Any]) -> bool:
    principle = _mapping(_mapping(artifact.get("field_principles")).get("verifier_is_oracle")).get(
        "principle"
    )
    text = str(principle or "").lower()
    return "design target" in text and "oracle-distinct" in text


def _citation_metadata_real(artifact: Mapping[str, Any]) -> bool:
    citations = _mapping(artifact.get("citations"))
    for arxiv_id in REQUIRED_ARXIV_IDS:
        citation = _mapping(citations.get(arxiv_id))
        if citation.get("http_status") != 200:
            return False
        if arxiv_id not in str(citation.get("url") or ""):
            return False
        if not str(citation.get("title") or "").strip():
            return False
    return True


def _validation_gate_precise(artifact: Mapping[str, Any]) -> bool:
    gate = _mapping(artifact.get("validation_gate"))
    return bool(
        gate.get("beats_self_consistency_ci95_excludes_zero_required") is True
        and gate.get("oracle_distinct_required") is True
        and gate.get("no_model_identity_shortcut_required") is True
        and gate.get("verifier_is_oracle_required_value") is False
        and gate.get("claimed_met") is False
    )


def _contains_matm_reproposal(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key).lower()
            if "matm" in key_text and any(word in key_text for word in ("proposed", "retrieval")):
                return True
            if "matm" in key_text and _mapping(item).get("proposed") is True:
                return True
            if _contains_matm_reproposal(item):
                return True
    elif isinstance(value, list):
        return any(_contains_matm_reproposal(item) for item in value)
    elif isinstance(value, str):
        text = value.lower()
        if "matm" in text and "retired" not in text and any(
            word in text for word in ("proposed", "re-proposed", "similarity-keyed")
        ):
            return True
    return False


def audit_pivot_readiness(artifact: Mapping[str, Any]) -> JsonDict:
    design = _mapping(artifact.get("design_spec"))
    verifier_column = _mapping(design.get("decomposed_energy_verifier_column"))
    circular_flags = _critical_circular_moat_flags(artifact)

    verifier_false = artifact.get("verifier_is_oracle") is False
    design_target_declared = _verifier_design_principle_declared(artifact)
    model_identity_forbidden = verifier_column.get("model_identity_features_allowed") is False
    oracle_labels_forbidden = verifier_column.get("oracle_labels_allowed_in_verifier") is False
    oracle_distinct_design = bool(
        verifier_false
        and design_target_declared
        and model_identity_forbidden
        and oracle_labels_forbidden
        and not circular_flags
    )

    arxiv_ids_exact = set(artifact.get("arxiv_ids_cited") or []) == REQUIRED_ARXIV_IDS
    citations_real = _citation_metadata_real(artifact)
    validation_precise = _validation_gate_precise(artifact)
    moat_not_claimed = artifact.get("moat_proven_claimed") is False
    matm_not_reproposed = not _contains_matm_reproposal(artifact)
    honest_readiness = bool(
        arxiv_ids_exact
        and citations_real
        and validation_precise
        and moat_not_claimed
        and matm_not_reproposed
    )

    reasons: list[str] = []
    if not verifier_false:
        reasons.append("D_oracle_distinct_design_failed_verifier_is_oracle_not_false")
    if not design_target_declared:
        reasons.append("D_oracle_distinct_design_failed_design_target_not_declared")
    if not model_identity_forbidden:
        reasons.append("D_oracle_distinct_design_failed_model_identity_shortcut_allowed")
    if not oracle_labels_forbidden:
        reasons.append("D_oracle_distinct_design_failed_oracle_labels_allowed")
    if circular_flags:
        reasons.append("D_oracle_distinct_design_failed_circular_moat_overclaim")
    if not arxiv_ids_exact:
        reasons.append("D_honest_readiness_failed_arxiv_ids_not_exact")
    if not citations_real:
        reasons.append("D_honest_readiness_failed_citation_metadata_not_real")
    if not validation_precise:
        reasons.append("D_honest_readiness_failed_validation_gate_not_precise")
    if not moat_not_claimed:
        reasons.append("D_honest_readiness_failed_moat_proven_claimed")
    if not matm_not_reproposed:
        reasons.append("D_honest_readiness_failed_matm_reproposed")

    return {
        "present": True,
        "artifact_experiment": artifact.get("experiment"),
        "checks": {
            "oracle_distinct_design": oracle_distinct_design,
            "honest_readiness": honest_readiness,
        },
        "failure_reasons": reasons,
        "oracle_distinct_design_evidence": {
            "verifier_is_oracle": artifact.get("verifier_is_oracle"),
            "design_target_declared": design_target_declared,
            "model_identity_features_allowed": verifier_column.get(
                "model_identity_features_allowed"
            ),
            "oracle_labels_allowed_in_verifier": verifier_column.get(
                "oracle_labels_allowed_in_verifier"
            ),
            "circular_moat_critical_flags": circular_flags,
        },
        "honest_readiness_evidence": {
            "arxiv_ids_cited": artifact.get("arxiv_ids_cited") or [],
            "required_arxiv_ids": sorted(REQUIRED_ARXIV_IDS),
            "citation_metadata_real": citations_real,
            "validation_gate": dict(_mapping(artifact.get("validation_gate"))),
            "moat_proven_claimed": artifact.get("moat_proven_claimed"),
            "matm_reproposed": not matm_not_reproposed,
        },
    }


def _preconditions(root: Path, registry_loader: RepoLoader | None) -> tuple[JsonDict, Mapping[str, Any] | None]:
    spec_path = root / "openspec/capabilities/capstone/spec.md"
    checked: JsonDict = {
        "a1_artifact_present": (root / A1_RELATIVE_PATH).exists(),
        "a2_artifact_present": (root / A2_RELATIVE_PATH).exists(),
        "d_artifact_present": (root / PIVOT_RELATIVE_PATH).exists(),
        "registry_present": (root / REGISTRY_RELATIVE_PATH).exists(),
        "adversarial_verify_present": (root / "scripts/adversarial_verify.py").exists(),
        "summarize_artifact_present": (root / "scripts/summarize_artifact.py").exists(),
        "arc_orphan_solver_lint_present": (root / "scripts/arc_orphan_solver_lint.py").exists(),
        "spec_has_req_4941": (
            "REQ-CAPSTONE-4941" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
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
        "a1_artifact_present": "experiment_4936_levelup_attempt",
        "a2_artifact_present": "experiment_4937_levelup_attempt",
        "d_artifact_present": "experiment_4940_distributional_energy_verifier_executable_spec",
        "registry_present": "arc_solve_registry",
        "adversarial_verify_present": "scripts_adversarial_verify",
        "summarize_artifact_present": "scripts_summarize_artifact",
        "arc_orphan_solver_lint_present": "scripts_arc_orphan_solver_lint",
        "spec_has_req_4941": "capstone_spec_req_4941",
    }
    return [label for key, label in labels.items() if checked.get(key) is not True]


def _blocked_verdict(checked: Mapping[str, Any]) -> str | None:
    ordered = (
        ("a1_artifact_present", "blocked_experiment_4936_levelup_attempt_missing"),
        ("a2_artifact_present", "blocked_experiment_4937_levelup_attempt_missing"),
        (
            "d_artifact_present",
            "blocked_experiment_4940_distributional_energy_verifier_executable_spec_missing",
        ),
        ("registry_present", "blocked_arc_solve_registry_missing"),
        ("registry_loadable", "blocked_arc_solve_registry_unloadable"),
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4941", "blocked_capstone_spec_req_4941_missing"),
    )
    for key, verdict in ordered:
        if checked.get(key) is not True:
            return verdict
    return None


def _missing_bank_evidence(label: str, reason: str) -> JsonDict:
    return {
        "label": label,
        "present": False,
        "bank_claimed": False,
        "checks": {key: False for key in BANK_CHECK_KEYS},
        "failure_reasons": [reason],
    }


def _missing_pivot_evidence(reason: str) -> JsonDict:
    return {
        "present": False,
        "checks": {key: False for key in PIVOT_CHECK_KEYS},
        "failure_reasons": [reason],
    }


def _aggregate_checks(bank_evidence: Mapping[str, Any], pivot_evidence: Mapping[str, Any]) -> JsonDict:
    checks: JsonDict = {}
    bank_values = [value for value in bank_evidence.values() if isinstance(value, Mapping)]
    claimed = [value for value in bank_values if value.get("bank_claimed") is True]
    missing_bank = any(value.get("present") is not True for value in bank_values)
    for key in BANK_CHECK_KEYS:
        if missing_bank:
            checks[key] = False
        elif claimed:
            checks[key] = all(_mapping(value.get("checks")).get(key) is True for value in claimed)
        else:
            checks[key] = True
    pivot_checks = _mapping(pivot_evidence.get("checks"))
    for key in PIVOT_CHECK_KEYS:
        checks[key] = pivot_checks.get(key) is True
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
            bank_evidence[label] = audit_bank(
                label=label,
                artifact=artifact,
                loop_artifact=loop,
                registry=registry,
                lint_result=lint,
            )

    pivot_artifact = _load_json_if_present(root, PIVOT_RELATIVE_PATH)
    if pivot_artifact is None:
        pivot_evidence = _missing_pivot_evidence(
            "D_missing_experiment_4940_distributional_energy_verifier_executable_spec"
        )
    else:
        pivot_evidence = audit_pivot_readiness(pivot_artifact)

    checks = _aggregate_checks(bank_evidence, pivot_evidence)
    banks_trustworthy = all(checks[key] is True for key in BANK_CHECK_KEYS)
    pivot_trustworthy = all(checks[key] is True for key in PIVOT_CHECK_KEYS)
    failure_reasons = _collect_failure_reasons(bank_evidence, pivot_evidence)
    failure_reasons.extend(
        f"missing_precondition_{item}" for item in checked.get("absent_inputs", []) or []
    )
    verdict = _blocked_verdict(checked)
    if verdict is None:
        verdict = (
            "complete_v455_banks_and_pivot_audited_trusted"
            if banks_trustworthy and pivot_trustworthy
            else "complete_v455_banks_and_pivot_audited_with_named_failures"
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
            "pivot_readiness_trustworthy": pivot_trustworthy,
            "checks": checks,
            "audit_failure_reasons": failure_reasons,
            "inference_substrate": INFERENCE_SUBSTRATE,
            "preconditions_checked": checked,
            "bank_evidence": bank_evidence,
            "pivot_readiness_evidence": pivot_evidence,
            "lint_evidence": lint,
            "adversarial_verifier_evidence": {
                "check_circular_moat_overclaim_used_for_D": pivot_artifact is not None,
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
    if type(artifact.get("pivot_readiness_trustworthy")) is not bool:
        errors.append("pivot_readiness_trustworthy must be bare bool")
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
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
            }
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI boundary
    raise SystemExit(main(sys.argv[1:]))
