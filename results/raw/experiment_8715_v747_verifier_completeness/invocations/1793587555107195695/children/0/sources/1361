"""Experiment 4974: adversarial audit for .458 ARC banks and pivot turnkey."""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from carnot import experiment_4963_bank_and_pivot_audit as prior_audit


JsonDict = dict[str, Any]
RepoLoader = Callable[[Path], Mapping[str, Any]]
LintRunner = Callable[[Path], Mapping[str, Any]]
Clock = Callable[[], float]

REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4974_bank_and_pivot_audit"
EXPERIMENT_ID = 4974
SCHEMA = "carnot.v458_bank_and_pivot_audit.v1"
RESULT_RELATIVE_PATH = "results/experiment_4974_bank_and_pivot_audit.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
A1_RELATIVE_PATH = "results/experiment_4969_levelup_attempt.json"
A2_RELATIVE_PATH = "results/experiment_4970_levelup_attempt.json"
PIVOT_RELATIVE_PATH = "results/experiment_4973_distributional_energy_verifier_turnkey.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 4974

PIVOT_ENTRYPOINT_COMMAND = (
    ".venv/bin/python python/carnot/experiment_4973_distributional_energy_verifier_turnkey.py"
)
THREE_DRY_RUN_COLUMNS = ("self_consistency", "decomposed_energy_verifier", "oracle")

SPEC_REFS = [
    "REQ-CAPSTONE-4974",
    "SCENARIO-CAPSTONE-4974",
    "SCENARIO-CAPSTONE-4974-BLOCKED-PRECONDITION",
]

CHECK_KEYS = prior_audit.CHECK_KEYS
BANK_CHECK_KEYS = prior_audit.BANK_CHECK_KEYS
PIVOT_CHECK_KEYS = prior_audit.PIVOT_CHECK_KEYS
REQUIRED_ARXIV_IDS = {
    "2504.01005",
    "2504.00891",
    "2509.24460",
    "2605.18871",
    "2504.16828",
    "2502.01989",
    "2508.16665",
    "2508.10539",
    "2502.11157",
}

FIELD_PRINCIPLES = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; complete_v458_banks_and_pivot_audited "
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
            "AND of the pivot-turnkey checks -- the capstone states D's readiness "
            "ONLY if true (oracle-distinct design + honest gate + genuine turnkey "
            "wiring + no over-claim)."
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

_read_json = prior_audit._read_json
_load_json_if_present = prior_audit._load_json_if_present
_load_registry = prior_audit._load_registry
_int_value = prior_audit._int_value
_mapping = prior_audit._mapping
_contains_matm_reproposal = prior_audit._contains_matm_reproposal
_missing_bank_evidence = prior_audit._missing_bank_evidence
_missing_pivot_evidence = prior_audit._missing_pivot_evidence
_aggregate_checks = prior_audit._aggregate_checks
_collect_failure_reasons = prior_audit._collect_failure_reasons
audit_bank = prior_audit.audit_bank
run_arc_orphan_solver_lint = prior_audit.run_arc_orphan_solver_lint


def _critical_circular_moat_flags(artifact: Mapping[str, Any]) -> list[JsonDict]:
    try:
        repo_text = str(REPO)
        if repo_text not in sys.path:  # pragma: no cover - import environment guard
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


def _artifact_path_exists(root: Path, path_text: str) -> bool:
    if not path_text:
        return False
    path = Path(path_text)
    if path.is_absolute():
        return path.exists()
    return (root / path).exists()


def _turnkey_rows_have_required_columns(dry_run: Mapping[str, Any]) -> bool:
    rows = dry_run.get("rows")
    if dry_run.get("columns") != list(THREE_DRY_RUN_COLUMNS):
        return False
    if dry_run.get("full_benchmark_run") is not False:
        return False
    if not isinstance(rows, list) or _int_value(dry_run.get("n_rows")) < 3 or len(rows) < 3:
        return False
    for row in rows:
        if not isinstance(row, Mapping):
            return False
        if not all(column in row for column in THREE_DRY_RUN_COLUMNS):
            return False
        verifier = _mapping(row.get("decomposed_energy_verifier"))
        oracle = _mapping(row.get("oracle"))
        if verifier.get("verifier_is_oracle") is not False:
            return False
        if oracle.get("oracle_used_for_correctness_only") is not True:
            return False
    return True


def _turnkey_wiring_evidence(artifact: Mapping[str, Any], *, root: Path) -> JsonDict:
    turnkey = _mapping(artifact.get("turnkey_spec"))
    dry_run = _mapping(artifact.get("dry_run_three_columns"))
    preconditions = _mapping(artifact.get("preconditions_checked"))
    source_artifacts = _mapping(artifact.get("source_artifacts"))
    post_pointer = _mapping(artifact.get("post_sprint_first_experiment_pointer"))
    loader = str(turnkey.get("real_loader") or "")
    source_loader = str(source_artifacts.get("domain_slice") or "")
    loader_present = _artifact_path_exists(root, loader)
    loader_declared_consistently = bool(loader and (not source_loader or source_loader == loader))
    dry_run_rows_ok = _turnkey_rows_have_required_columns(dry_run)
    entrypoint_ok = bool(
        turnkey.get("entrypoint_command") == PIVOT_ENTRYPOINT_COMMAND
        and post_pointer.get("entrypoint_command") == PIVOT_ENTRYPOINT_COMMAND
    )
    preconditions_ok = bool(
        preconditions.get("domain_slice_present") is True
        and preconditions.get("domain_slice_valid") is True
        and _int_value(preconditions.get("domain_slice_rows")) >= 3
        and preconditions.get("self_consistency_saturated") is False
        and preconditions.get("blocked_resource") is None
        and preconditions.get("real_benchmark_executed") is False
        and preconditions.get("model_load") is False
        and preconditions.get("training_launched") is False
        and preconditions.get("scripts_research_conductor_modified") is False
    )
    return {
        "pivot_turnkey": artifact.get("pivot_turnkey") is True,
        "three_column_dry_run_ok": artifact.get("three_column_dry_run_ok") is True,
        "entrypoint_ok": entrypoint_ok,
        "loader": loader,
        "loader_present": loader_present,
        "loader_declared_consistently": loader_declared_consistently,
        "dry_run_rows_have_required_columns": dry_run_rows_ok,
        "preconditions_ok": preconditions_ok,
    }


def _turnkey_wiring_genuine(artifact: Mapping[str, Any], *, root: Path) -> bool:
    evidence = _turnkey_wiring_evidence(artifact, root=root)
    return all(value is True for key, value in evidence.items() if key != "loader")


def audit_pivot_readiness(artifact: Mapping[str, Any], *, root: Path = REPO) -> JsonDict:
    design = _mapping(artifact.get("design_spec")) or _mapping(artifact.get("turnkey_spec"))
    verifier_column = _mapping(design.get("decomposed_energy_verifier_column"))
    circular_flags = _critical_circular_moat_flags(artifact)
    turnkey_evidence = _turnkey_wiring_evidence(artifact, root=root)

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
    turnkey_wiring = _turnkey_wiring_genuine(artifact, root=root)
    moat_not_claimed = artifact.get("moat_proven_claimed") is False
    matm_not_reproposed = not _contains_matm_reproposal(artifact)
    honest_readiness = bool(
        arxiv_ids_exact
        and citations_real
        and validation_precise
        and turnkey_wiring
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
    if not turnkey_wiring:
        reasons.append("D_honest_readiness_failed_pivot_turnkey_wiring_not_genuine")
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
        "turnkey_wiring_evidence": turnkey_evidence,
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
        "spec_has_req_4974": (
            "REQ-CAPSTONE-4974" in spec_path.read_text(encoding="utf-8")
            if spec_path.exists()
            else False
        ),
    }
    registry: Mapping[str, Any] | None = None
    if checked["registry_present"]:
        try:
            registry = registry_loader(root) if registry_loader is not None else _load_registry(root)
            if not isinstance(registry, Mapping):
                raise ValueError("registry did not contain a mapping")
            checked["registry_loadable"] = True
        except Exception as exc:
            checked["registry_loadable"] = False
            checked["registry_error"] = str(exc)
            registry = None
    else:
        checked["registry_loadable"] = False
    checked["absent_inputs"] = _absent_inputs(checked)
    return checked, registry


def _absent_inputs(checked: Mapping[str, Any]) -> list[str]:
    labels = {
        "a1_artifact_present": "experiment_4969_levelup_attempt",
        "a2_artifact_present": "experiment_4970_levelup_attempt",
        "d_artifact_present": "experiment_4973_distributional_energy_verifier_turnkey",
        "registry_present": "arc_solve_registry",
        "adversarial_verify_present": "scripts_adversarial_verify",
        "summarize_artifact_present": "scripts_summarize_artifact",
        "arc_orphan_solver_lint_present": "scripts_arc_orphan_solver_lint",
        "spec_has_req_4974": "capstone_spec_req_4974",
    }
    return [label for key, label in labels.items() if checked.get(key) is not True]


def _blocked_verdict(checked: Mapping[str, Any]) -> str | None:
    ordered = (
        ("a1_artifact_present", "blocked_experiment_4969_levelup_attempt_missing"),
        ("a2_artifact_present", "blocked_experiment_4970_levelup_attempt_missing"),
        (
            "d_artifact_present",
            "blocked_experiment_4973_distributional_energy_verifier_turnkey_missing",
        ),
        ("registry_present", "blocked_arc_solve_registry_missing"),
        ("registry_loadable", "blocked_arc_solve_registry_unloadable"),
        ("adversarial_verify_present", "blocked_scripts_adversarial_verify_missing"),
        ("summarize_artifact_present", "blocked_scripts_summarize_artifact_missing"),
        ("arc_orphan_solver_lint_present", "blocked_scripts_arc_orphan_solver_lint_missing"),
        ("spec_has_req_4974", "blocked_capstone_spec_req_4974_missing"),
    )
    for key, verdict in ordered:
        if checked.get(key) is not True:
            return verdict
    return None


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
            "D_missing_experiment_4973_distributional_energy_verifier_turnkey"
        )
    else:
        pivot_evidence = audit_pivot_readiness(pivot_artifact, root=root)

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
            "complete_v458_banks_and_pivot_audited_trusted"
            if banks_trustworthy and pivot_trustworthy
            else "complete_v458_banks_and_pivot_audited_with_named_failures"
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
