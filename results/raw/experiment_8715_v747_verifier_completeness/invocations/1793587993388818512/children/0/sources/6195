"""Build the Exp 3198 repair-gate decision v5 artifact.

Spec refs: REQ-VERIFY-3198, SCENARIO-VERIFY-3198.

This module is a gate, not a repair runner. It exists so the conductor and any
downstream repair ladder can make one cheap deterministic decision from the
checked-in `.296` evidence instead of trying an expensive model call and only
then discovering that the clean local SOTA verifier was skipped, blocked, or
flagged.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260527"
SCHEMA_VERSION = "carnot.repair_gate_decision.v5"
EXPERIMENT_ID = "exp3198"
ARTIFACT = "experiment_3198_repair_gate_decision_v5"

OUTPUT_REL_PATH = Path("results/experiment_3198_repair_gate_decision_v5.json")
SPEC_REL_PATH = Path("openspec/capabilities/verification/spec.md")
SCRIPT_REL_PATH = REPO_ROOT / "scripts" / "experiment_3198_repair_gate_decision_v5.py"

EXP3192_REL_PATH = Path("results/experiment_3192_receipt_adversarial_contract_v4.json")
EXP3193_REL_PATH = Path("results/experiment_3193_llama_cpp_cuda_offload_health_probe_v1.json")
EXP3194_REL_PATH = Path("results/experiment_3194_clean_live_sota_verifier_rerun_v11.json")
EXP3195_REL_PATH = Path("results/experiment_3195_adaptive_verification_granularity_policy_v1.json")
EXP3196_REL_PATH = Path("results/experiment_3196_gencp_domain_preview_repair_compiler_v1.json")
EXP3197_REL_PATH = Path("results/experiment_3197_exverus_inductive_certificate_expansion_v1.json")

FALSE_ACCEPT_GATE = 0.10
MAX_REPAIR_ROWS = 2
MAX_ATTEMPTS_PER_ROW = 2

REQUIRED_FIELDS = {
    "schema_version",
    "experiment_id",
    "source_artifacts",
    "receipt_gate_state",
    "clean_verifier_state",
    "adaptive_policy_state",
    "domain_preview_state",
    "invariant_certificate_state",
    "repair_gate_state",
    "repair_allowed_scope",
    "blocker_reasons",
    "downstream_gated_skip_expected",
    "honest_verdict",
}

ALLOWED_REPAIR_GATE_STATES = {
    "blocked_missing_required_upstream",
    "blocked_clean_verifier_gate_skipped",
    "blocked_upstream_adversarially_flagged",
    "blocked_receipt_precondition",
    "blocked_clean_verifier_not_eligible",
    "blocked_adaptive_policy_not_ready",
    "blocked_domain_preview_not_ready",
    "blocked_invariant_certificate_not_ready",
    "blocked_other_precondition",
    "unblocked_for_bounded_repair_ladder",
}

SOURCE_REL_PATHS: tuple[tuple[str, Path, bool, str], ...] = (
    ("agents_repo_instructions", Path("AGENTS.md"), True, "text"),
    ("codex_repo_workflow", Path("CODEX.md"), True, "text"),
    ("claude_authenticity_rules", Path("CLAUDE.md"), True, "text"),
    ("verification_openspec", SPEC_REL_PATH, True, "text"),
    ("exp3192_receipt_adversarial_contract_v4", EXP3192_REL_PATH, True, "json"),
    ("exp3193_llama_cpp_cuda_offload_health_probe_v1", EXP3193_REL_PATH, True, "json"),
    ("exp3194_clean_live_sota_verifier_rerun_v11", EXP3194_REL_PATH, True, "json"),
    ("exp3195_adaptive_verification_granularity_policy_v1", EXP3195_REL_PATH, True, "json"),
    ("exp3196_gencp_domain_preview_repair_compiler_v1", EXP3196_REL_PATH, True, "json"),
    (
        "exp3197_exverus_inductive_certificate_expansion_v1",
        EXP3197_REL_PATH,
        True,
        "json",
    ),
    ("exp3198_module", Path("python/carnot/verify/repair_gate_decision_v5.py"), False, "python"),
    ("exp3198_script", Path("scripts/experiment_3198_repair_gate_decision_v5.py"), False, "python"),
    (
        "exp3198_tests",
        Path("tests/python/test_experiment_3198_repair_gate_decision_v5.py"),
        False,
        "python",
    ),
)

DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest tests/python/test_experiment_3198_repair_gate_decision_v5.py -q -o addopts=''",
    ".venv/bin/coverage erase",
    ".venv/bin/coverage run --source=python/carnot/verify/repair_gate_decision_v5.py -m pytest -o addopts='' tests/python/test_experiment_3198_repair_gate_decision_v5.py -q",
    ".venv/bin/coverage report --include='python/carnot/verify/repair_gate_decision_v5.py' --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_3198_repair_gate_decision_v5.py",
    ".venv/bin/pytest tests/python -q",
)


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """REQ-VERIFY-3198: aggregate `.296` gate evidence without new execution."""

    root_path = Path(root)
    payloads = load_payloads(root_path)
    sources = source_artifacts(root_path)
    receipt_state = receipt_gate_state(payloads["exp3192"], payloads["exp3193"])
    clean_state = clean_verifier_state(payloads["exp3194"])
    adaptive_state = adaptive_policy_state(payloads["exp3195"])
    domain_state = domain_preview_state(payloads["exp3196"])
    invariant_state = invariant_certificate_state(payloads["exp3197"])
    blockers = blocker_reasons(sources, payloads)
    state = repair_gate_state(
        receipt_state=receipt_state,
        clean_verifier_state=clean_state,
        adaptive_policy_state=adaptive_state,
        domain_preview_state=domain_state,
        invariant_certificate_state=invariant_state,
        blockers=blockers,
    )
    artifact: JsonDict = {
        "artifact": ARTIFACT,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": ["REQ-VERIFY-3198", "SCENARIO-VERIFY-3198"],
        "source_artifacts": sources,
        "source_checksums": {
            row["path"]: row["sha256"] for row in sources if row.get("sha256") is not None
        },
        "source_gate_summary": source_gate_summary(payloads),
        "receipt_gate_state": receipt_state,
        "clean_verifier_state": clean_state,
        "adaptive_policy_state": adaptive_state,
        "domain_preview_state": domain_state,
        "invariant_certificate_state": invariant_state,
        "repair_gate_state": state,
        "repair_allowed_scope": repair_allowed_scope(state, payloads),
        "blocker_reasons": blockers,
        "downstream_gated_skip_expected": state != "unblocked_for_bounded_repair_ladder",
        "inference_substrate": inference_substrate(payloads),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "scripts_research_conductor_modified": False,
        "ops_docs_reconciliation_left_to_conductor": True,
        "honest_verdict": "",
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Build, validate, and persist the v5 repair-gate JSON artifact."""

    root_path = Path(root)
    output = Path(output_path)
    if not output.is_absolute():
        output = root_path / output
    artifact = build_artifact(root_path, tests_run=tests_run)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def load_payloads(root: Path) -> dict[str, JsonDict]:
    """Load every required `.296` upstream artifact that exists locally."""

    return {
        "exp3192": read_json_object(root / EXP3192_REL_PATH),
        "exp3193": read_json_object(root / EXP3193_REL_PATH),
        "exp3194": read_json_object(root / EXP3194_REL_PATH),
        "exp3195": read_json_object(root / EXP3195_REL_PATH),
        "exp3196": read_json_object(root / EXP3196_REL_PATH),
        "exp3197": read_json_object(root / EXP3197_REL_PATH),
    }


def read_json_object(path: Path) -> JsonDict:
    """Return a JSON object from disk, or empty evidence for missing/malformed input."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def source_artifacts(root: Path) -> list[JsonDict]:
    """Describe every instruction, spec, and upstream evidence file used by the gate."""

    rows: list[JsonDict] = []
    for role, rel_path, required, source_type in SOURCE_REL_PATHS:
        path = root / rel_path
        payload = read_json_object(path) if source_type == "json" else {}
        rows.append(
            {
                "role": role,
                "path": rel_path.as_posix(),
                "required": required,
                "source_type": source_type,
                "present": path.is_file(),
                "readable_json_object": bool(payload) if source_type == "json" else None,
                "sha256": sha256_file(path),
                "summary": source_summary(payload) if source_type == "json" and payload else {},
            }
        )
    return rows


def source_summary(payload: Mapping[str, Any]) -> JsonDict:
    """Keep compact status fields beside provenance so blockers are auditable."""

    return {
        "experiment_id": payload.get("experiment_id") or payload.get("experiment"),
        "status": payload.get("status"),
        "schema": payload.get("schema") or payload.get("schema_version"),
        "honest_verdict": payload.get("honest_verdict"),
        "repair_call_ready": payload.get("repair_call_ready"),
        "clean_rerun_allowed": payload.get("clean_rerun_allowed"),
        "flagged_adversarial": payload.get("flagged_adversarial"),
    }


def receipt_gate_state(contract: Mapping[str, Any], offload: Mapping[str, Any]) -> str:
    """Classify receipt/offload eligibility before the clean verifier may run."""

    if not contract:
        return "missing_receipt_contract_artifact"
    if not offload:
        return "missing_offload_probe_artifact"
    current = mapping_value(contract.get("current_evidence_assessment"))
    substrate = str(
        offload.get("substrate_classification")
        or current.get("substrate_classification")
        or "unknown_substrate"
    )
    if (
        contract.get("receipt_adversarial_contract_v4_ready") is True
        and current.get("clean_rerun_allowed") is True
        and offload.get("clean_rerun_allowed") is True
        and offload.get("headline_claim_allowed") is True
        and substrate == "full_local_sota_receipt"
        and int_value(offload.get("receipt_count")) > 0
        and offload.get("flagged_adversarial") is not True
    ):
        return "eligible_full_local_sota_receipt"
    return f"blocked_{substrate}"


def clean_verifier_state(clean: Mapping[str, Any]) -> str:
    """Classify whether Exp 3194 produced usable unflagged clean verifier evidence."""

    if not clean:
        return "missing_clean_verifier_artifact"
    if clean.get("schema") == "blocked_gate_check_v1" or clean.get("status") == "blocked":
        if clean.get("blocked_at_layer") == "conductor_pre_gate":
            return "blocked_gate_skipped_conductor_pre_gate"
        return "blocked_clean_verifier_artifact"
    if clean.get("gated_skip") is True:
        return "blocked_clean_verifier_gated_skip"
    if clean.get("flagged_adversarial") is True:
        return "blocked_clean_verifier_adversarially_flagged"
    false_accept_rate = finite_rate(clean.get("false_accept_rate"))
    accepted_false = clean.get("known_false_accepts_accepted") or []
    if (
        clean.get("clean_live_sota_verifier_rerun_v11_ready") is True
        and clean.get("metrics_computed") is True
        and clean.get("headline_claim_allowed") is True
        and false_accept_rate is not None
        and false_accept_rate <= FALSE_ACCEPT_GATE
        and not accepted_false
    ):
        return "eligible_unflagged_clean_verifier"
    return "blocked_clean_verifier_not_eligible"


def adaptive_policy_state(policy: Mapping[str, Any]) -> str:
    """Classify whether adaptive scheduling evidence is ready and non-risk-increasing."""

    if not policy:
        return "missing_adaptive_policy_artifact"
    if policy.get("adaptive_verification_granularity_policy_v1_ready") is not True:
        return "blocked_adaptive_policy_not_ready"
    if policy.get("source_errors"):
        return "blocked_adaptive_policy_source_errors"
    if int_value(policy.get("exact_rows_used")) <= 0:
        return "blocked_adaptive_policy_no_exact_rows"
    risk = finite_nonnegative_number(policy.get("false_accept_risk_increase"))
    if risk is None or risk > 0.0:
        return "blocked_adaptive_policy_false_accept_risk"
    return "ready_adaptive_schedule"


def domain_preview_state(preview: Mapping[str, Any]) -> str:
    """Classify whether GenCP preview domains are bounded enough for a later repair."""

    if not preview:
        return "missing_domain_preview_artifact"
    if preview.get("source_errors"):
        return "blocked_domain_preview_source_errors"
    if int_value(preview.get("preview_domain_count")) <= 0:
        return "blocked_domain_preview_empty"
    average_size = finite_nonnegative_number(preview.get("average_candidate_domain_size"))
    if average_size is None or average_size <= 0.0:
        return "blocked_domain_preview_unbounded"
    return "ready_bounded_domain_preview"


def invariant_certificate_state(certificate: Mapping[str, Any]) -> str:
    """Classify whether ExVerus records cover exact guards and anti-overfit tests."""

    if not certificate:
        return "missing_invariant_certificate_artifact"
    if certificate.get("source_errors"):
        return "blocked_invariant_certificate_source_errors"
    record_count = int_value(certificate.get("invariant_record_count"))
    exact_guard_count = int_value(certificate.get("exact_guard_count"))
    anti_overfit_count = int_value(certificate.get("anti_overfit_test_count"))
    linked_count = int_value(certificate.get("linked_domain_preview_count"))
    if record_count <= 0:
        return "blocked_invariant_certificate_empty"
    if exact_guard_count < record_count:
        return "blocked_invariant_certificate_exact_guard_gap"
    if anti_overfit_count < record_count:
        return "blocked_invariant_certificate_anti_overfit_gap"
    if linked_count <= 0:
        return "blocked_invariant_certificate_no_domain_links"
    return "ready_exact_guard_coverage"


def blocker_reasons(
    sources: Sequence[Mapping[str, Any]],
    payloads: Mapping[str, Mapping[str, Any]],
) -> list[JsonDict]:
    """Return all fail-closed blocker reasons as deterministic structured rows."""

    blockers: list[JsonDict] = []
    blockers.extend(source_blockers(sources))
    blockers.extend(receipt_blockers(payloads["exp3192"], payloads["exp3193"]))
    blockers.extend(clean_verifier_blockers(payloads["exp3194"]))
    blockers.extend(adaptive_policy_blockers(payloads["exp3195"]))
    blockers.extend(domain_preview_blockers(payloads["exp3196"]))
    blockers.extend(invariant_certificate_blockers(payloads["exp3197"]))
    return blockers


def source_blockers(sources: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Expose absent or malformed required sources instead of inferring success."""

    blockers: list[JsonDict] = []
    for row in sources:
        structured = row.get("source_type") == "json"
        malformed = structured and row.get("readable_json_object") is not True
        if row.get("required") is True and (row.get("present") is not True or malformed):
            blockers.append(
                blocker(
                    "missing_required_upstream",
                    str(row.get("path") or ""),
                    "present/readable_json_object",
                    True,
                    {
                        "present": row.get("present"),
                        "readable_json_object": row.get("readable_json_object"),
                    },
                    "required upstream artifact or instruction source is missing or malformed",
                )
            )
    return blockers


def receipt_blockers(contract: Mapping[str, Any], offload: Mapping[str, Any]) -> list[JsonDict]:
    """Return receipt/offload blockers, including CPU fallback and adversarial flags."""

    blockers: list[JsonDict] = []
    if contract:
        current = mapping_value(contract.get("current_evidence_assessment"))
        current_substrate = str(current.get("substrate_classification") or "")
        if contract.get("receipt_adversarial_contract_v4_ready") is not True:
            blockers.append(
                blocker(
                    "exp3192_contract_not_ready",
                    EXP3192_REL_PATH,
                    "receipt_adversarial_contract_v4_ready",
                    True,
                    contract.get("receipt_adversarial_contract_v4_ready"),
                    "receipt/adversarial v4 contract is not ready",
                )
            )
        if current_substrate == "cpu_fallback_receipt_only":
            blockers.append(
                blocker(
                    "exp3192_current_evidence_cpu_fallback_only",
                    EXP3192_REL_PATH,
                    "current_evidence_assessment.substrate_classification",
                    "full_local_sota_receipt",
                    current_substrate,
                    "CPU fallback proves wiring only and cannot unlock repair",
                )
            )
        if current.get("clean_rerun_allowed") is not True:
            blockers.append(
                blocker(
                    "exp3192_clean_rerun_not_allowed",
                    EXP3192_REL_PATH,
                    "current_evidence_assessment.clean_rerun_allowed",
                    True,
                    current.get("clean_rerun_allowed"),
                    "receipt contract assessment does not allow clean rerun",
                )
            )
    if offload:
        substrate = str(offload.get("substrate_classification") or "")
        if offload.get("clean_rerun_allowed") is not True:
            blockers.append(
                blocker(
                    "exp3193_clean_rerun_not_allowed",
                    EXP3193_REL_PATH,
                    "clean_rerun_allowed",
                    True,
                    offload.get("clean_rerun_allowed"),
                    offload.get("blocker_reasons") or "offload probe did not allow clean rerun",
                )
            )
        if offload.get("headline_claim_allowed") is not True:
            blockers.append(
                blocker(
                    "exp3193_headline_claim_not_allowed",
                    EXP3193_REL_PATH,
                    "headline_claim_allowed",
                    True,
                    offload.get("headline_claim_allowed"),
                    "headline verifier metrics are not allowed from this substrate",
                )
            )
        if substrate != "full_local_sota_receipt":
            blockers.append(
                blocker(
                    "exp3193_nonfull_substrate",
                    EXP3193_REL_PATH,
                    "substrate_classification",
                    "full_local_sota_receipt",
                    substrate,
                    "repair requires full local SOTA CUDA/offload receipt evidence",
                )
            )
        if int_value(offload.get("receipt_count")) <= 0:
            blockers.append(
                blocker(
                    "exp3193_no_offload_receipt",
                    EXP3193_REL_PATH,
                    "receipt_count",
                    "positive",
                    offload.get("receipt_count"),
                    "no CUDA/offload receipt exists",
                )
            )
        if offload.get("flagged_adversarial") is True:
            blockers.append(
                blocker(
                    "exp3193_adversarially_flagged",
                    EXP3193_REL_PATH,
                    "flagged_adversarial",
                    False,
                    True,
                    "offload probe is adversarially flagged and cannot unlock repair",
                )
            )
    return blockers


def clean_verifier_blockers(clean: Mapping[str, Any]) -> list[JsonDict]:
    """Return clean verifier blockers, including conductor gate skips."""

    blockers: list[JsonDict] = []
    if not clean:
        return blockers
    if clean.get("schema") == "blocked_gate_check_v1" or clean.get("status") == "blocked":
        blockers.append(
            blocker(
                "exp3194_gate_skipped",
                EXP3194_REL_PATH,
                "status",
                "eligible clean verifier artifact",
                clean.get("status") or clean.get("honest_verdict"),
                clean.get("gate_check_summary") or "clean verifier was blocked or gate-skipped",
            )
        )
        return blockers
    if clean.get("gated_skip") is True:
        blockers.append(
            blocker(
                "exp3194_gated_skip",
                EXP3194_REL_PATH,
                "gated_skip",
                False,
                True,
                clean.get("gate_reasons") or "clean verifier gated skip",
            )
        )
    if clean.get("flagged_adversarial") is True:
        blockers.append(
            blocker(
                "exp3194_adversarially_flagged",
                EXP3194_REL_PATH,
                "flagged_adversarial",
                False,
                True,
                "clean verifier is adversarially flagged",
            )
        )
    if clean.get("clean_live_sota_verifier_rerun_v11_ready") is not True:
        blockers.append(
            blocker(
                "exp3194_not_ready",
                EXP3194_REL_PATH,
                "clean_live_sota_verifier_rerun_v11_ready",
                True,
                clean.get("clean_live_sota_verifier_rerun_v11_ready"),
                "clean verifier artifact is not marked ready",
            )
        )
    if clean.get("metrics_computed") is not True:
        blockers.append(
            blocker(
                "exp3194_metrics_not_computed",
                EXP3194_REL_PATH,
                "metrics_computed",
                True,
                clean.get("metrics_computed"),
                "clean verifier metrics were not computed",
            )
        )
    if clean.get("headline_claim_allowed") is not True:
        blockers.append(
            blocker(
                "exp3194_headline_claim_not_allowed",
                EXP3194_REL_PATH,
                "headline_claim_allowed",
                True,
                clean.get("headline_claim_allowed"),
                "clean verifier metrics are not headline eligible",
            )
        )
    false_accept_rate = finite_rate(clean.get("false_accept_rate"))
    if false_accept_rate is None or false_accept_rate > FALSE_ACCEPT_GATE:
        blockers.append(
            blocker(
                "exp3194_false_accept_gate_failed",
                EXP3194_REL_PATH,
                "false_accept_rate",
                f"finite <= {FALSE_ACCEPT_GATE}",
                clean.get("false_accept_rate"),
                "clean verifier false-accept evidence is missing or above gate",
            )
        )
    if clean.get("known_false_accepts_accepted"):
        blockers.append(
            blocker(
                "exp3194_known_false_accepts_accepted",
                EXP3194_REL_PATH,
                "known_false_accepts_accepted",
                [],
                clean.get("known_false_accepts_accepted"),
                "known false-accept rows must remain rejected",
            )
        )
    return blockers


def adaptive_policy_blockers(policy: Mapping[str, Any]) -> list[JsonDict]:
    """Return adaptive scheduling blockers."""

    blockers: list[JsonDict] = []
    if not policy:
        return blockers
    if policy.get("adaptive_verification_granularity_policy_v1_ready") is not True:
        blockers.append(
            blocker(
                "exp3195_policy_not_ready",
                EXP3195_REL_PATH,
                "adaptive_verification_granularity_policy_v1_ready",
                True,
                policy.get("adaptive_verification_granularity_policy_v1_ready"),
                "adaptive policy artifact is not ready",
            )
        )
    if policy.get("source_errors"):
        blockers.append(
            blocker(
                "exp3195_source_errors",
                EXP3195_REL_PATH,
                "source_errors",
                [],
                policy.get("source_errors"),
                "adaptive policy has source errors",
            )
        )
    if int_value(policy.get("exact_rows_used")) <= 0:
        blockers.append(
            blocker(
                "exp3195_no_exact_rows",
                EXP3195_REL_PATH,
                "exact_rows_used",
                "positive",
                policy.get("exact_rows_used"),
                "adaptive policy used no exact rows",
            )
        )
    risk = finite_nonnegative_number(policy.get("false_accept_risk_increase"))
    if risk is None or risk > 0.0:
        blockers.append(
            blocker(
                "exp3195_false_accept_risk_increase",
                EXP3195_REL_PATH,
                "false_accept_risk_increase",
                0.0,
                policy.get("false_accept_risk_increase"),
                "adaptive scheduling may increase known false-accept risk",
            )
        )
    return blockers


def domain_preview_blockers(preview: Mapping[str, Any]) -> list[JsonDict]:
    """Return bounded domain-preview blockers."""

    blockers: list[JsonDict] = []
    if not preview:
        return blockers
    if preview.get("source_errors"):
        blockers.append(
            blocker(
                "exp3196_source_errors",
                EXP3196_REL_PATH,
                "source_errors",
                [],
                preview.get("source_errors"),
                "domain preview has source errors",
            )
        )
    if int_value(preview.get("preview_domain_count")) <= 0:
        blockers.append(
            blocker(
                "exp3196_preview_domain_missing",
                EXP3196_REL_PATH,
                "preview_domain_count",
                "positive",
                preview.get("preview_domain_count"),
                "no bounded preview domains are available",
            )
        )
    average_size = finite_nonnegative_number(preview.get("average_candidate_domain_size"))
    if average_size is None or average_size <= 0.0:
        blockers.append(
            blocker(
                "exp3196_average_domain_size_invalid",
                EXP3196_REL_PATH,
                "average_candidate_domain_size",
                "finite positive",
                preview.get("average_candidate_domain_size"),
                "candidate domains are missing or unbounded",
            )
        )
    return blockers


def invariant_certificate_blockers(certificate: Mapping[str, Any]) -> list[JsonDict]:
    """Return invariant guard and anti-overfit coverage blockers."""

    blockers: list[JsonDict] = []
    if not certificate:
        return blockers
    record_count = int_value(certificate.get("invariant_record_count"))
    exact_guard_count = int_value(certificate.get("exact_guard_count"))
    anti_overfit_count = int_value(certificate.get("anti_overfit_test_count"))
    linked_count = int_value(certificate.get("linked_domain_preview_count"))
    if certificate.get("source_errors"):
        blockers.append(
            blocker(
                "exp3197_source_errors",
                EXP3197_REL_PATH,
                "source_errors",
                [],
                certificate.get("source_errors"),
                "invariant certificate has source errors",
            )
        )
    if record_count <= 0:
        blockers.append(
            blocker(
                "exp3197_invariant_records_missing",
                EXP3197_REL_PATH,
                "invariant_record_count",
                "positive",
                certificate.get("invariant_record_count"),
                "no invariant records are available",
            )
        )
    if exact_guard_count < record_count:
        blockers.append(
            blocker(
                "exp3197_exact_guard_coverage_insufficient",
                EXP3197_REL_PATH,
                "exact_guard_count",
                f">= invariant_record_count ({record_count})",
                certificate.get("exact_guard_count"),
                "each invariant record needs an exact guard",
            )
        )
    if anti_overfit_count < record_count:
        blockers.append(
            blocker(
                "exp3197_anti_overfit_coverage_insufficient",
                EXP3197_REL_PATH,
                "anti_overfit_test_count",
                f">= invariant_record_count ({record_count})",
                certificate.get("anti_overfit_test_count"),
                "each invariant record needs an anti-overfit test",
            )
        )
    if linked_count <= 0:
        blockers.append(
            blocker(
                "exp3197_domain_links_missing",
                EXP3197_REL_PATH,
                "linked_domain_preview_count",
                "positive",
                certificate.get("linked_domain_preview_count"),
                "invariant guards must link back to preview domains",
            )
        )
    return blockers


def blocker(
    code: str,
    source_artifact: str | Path,
    field: str,
    expected: Any,
    actual: Any,
    detail: Any,
) -> JsonDict:
    """Build one machine-readable blocker row with stable keys."""

    source = source_artifact.as_posix() if isinstance(source_artifact, Path) else source_artifact
    return {
        "code": code,
        "source_artifact": source,
        "field": field,
        "expected": expected,
        "actual": actual,
        "detail": detail,
    }


def repair_gate_state(
    *,
    receipt_state: str,
    clean_verifier_state: str,
    adaptive_policy_state: str,
    domain_preview_state: str,
    invariant_certificate_state: str,
    blockers: Sequence[Mapping[str, Any]],
) -> str:
    """Collapse layer states to the single downstream repair gate state."""

    if has_blocker(blockers, "missing_required_upstream"):
        return "blocked_missing_required_upstream"
    if clean_verifier_state == "blocked_gate_skipped_conductor_pre_gate":
        return "blocked_clean_verifier_gate_skipped"
    if any(str(row.get("code", "")).endswith("adversarially_flagged") for row in blockers):
        return "blocked_upstream_adversarially_flagged"
    if receipt_state != "eligible_full_local_sota_receipt":
        return "blocked_receipt_precondition"
    if clean_verifier_state != "eligible_unflagged_clean_verifier":
        return "blocked_clean_verifier_not_eligible"
    if adaptive_policy_state != "ready_adaptive_schedule":
        return "blocked_adaptive_policy_not_ready"
    if domain_preview_state != "ready_bounded_domain_preview":
        return "blocked_domain_preview_not_ready"
    if invariant_certificate_state != "ready_exact_guard_coverage":
        return "blocked_invariant_certificate_not_ready"
    if blockers:
        return "blocked_other_precondition"
    return "unblocked_for_bounded_repair_ladder"


def repair_allowed_scope(state: str, payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict | None:
    """Return the exact bounded repair scope, or None when repair is blocked."""

    if state != "unblocked_for_bounded_repair_ladder":
        return None
    preview_count = int_value(payloads["exp3196"].get("preview_domain_count"))
    invariant_count = int_value(payloads["exp3197"].get("invariant_record_count"))
    max_rows = max(0, min(MAX_REPAIR_ROWS, preview_count, invariant_count))
    return {
        "enabled": True,
        "source_domain_preview_artifact": EXP3196_REL_PATH.as_posix(),
        "source_invariant_certificate_artifact": EXP3197_REL_PATH.as_posix(),
        "row_selection": "intersection_of_preview_domains_and_invariant_guards",
        "max_distinct_rows": max_rows,
        "max_attempts_per_row": MAX_ATTEMPTS_PER_ROW,
        "max_total_repair_attempts": max_rows * MAX_ATTEMPTS_PER_ROW,
        "requires_mandated_local_sota": True,
        "requires_clean_verifier_unflagged": True,
        "requires_exact_authority_acceptance": True,
        "requires_anti_overfit_guard": True,
        "no_headline_claim_from_gate_alone": True,
        "allowed_row_budget_source": {
            "preview_domain_count": preview_count,
            "invariant_record_count": invariant_count,
        },
    }


def source_gate_summary(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Record compact upstream facts beside the final machine states."""

    exp3192_current = mapping_value(payloads["exp3192"].get("current_evidence_assessment"))
    return {
        "exp3192_clean_rerun_allowed": exp3192_current.get("clean_rerun_allowed"),
        "exp3192_substrate_classification": exp3192_current.get("substrate_classification"),
        "exp3193_clean_rerun_allowed": payloads["exp3193"].get("clean_rerun_allowed"),
        "exp3193_substrate_classification": payloads["exp3193"].get("substrate_classification"),
        "exp3193_flagged_adversarial": payloads["exp3193"].get("flagged_adversarial"),
        "exp3194_status": payloads["exp3194"].get("status"),
        "exp3194_schema": payloads["exp3194"].get("schema"),
        "exp3194_flagged_adversarial": payloads["exp3194"].get("flagged_adversarial"),
        "exp3195_false_accept_risk_increase": payloads["exp3195"].get(
            "false_accept_risk_increase"
        ),
        "exp3196_preview_domain_count": payloads["exp3196"].get("preview_domain_count"),
        "exp3197_invariant_record_count": payloads["exp3197"].get("invariant_record_count"),
    }


def inference_substrate(payloads: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    """Declare that Exp 3198 aggregates only and performs no model or repair work."""

    return {
        "kind": "deterministic_repair_gate_decision_v5",
        "aggregation_only": True,
        "no_live_inference": True,
        "no_llm_calls": True,
        "executes_models": False,
        "executes_repairs": False,
        "executes_verifiers": False,
        "downloads_models": False,
        "live_model_calls": 0,
        "new_live_model_calls": 0,
        "repair_calls": 0,
        "source_exp3193_receipt_count": int_value(payloads["exp3193"].get("receipt_count")),
        "source_exp3194_live_call_count": int_value(payloads["exp3194"].get("live_call_count")),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the terminal v5 gate shape before writing it to disk."""

    missing = sorted(REQUIRED_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"Exp 3198 artifact missing required fields: {missing}")
    state = str(artifact.get("repair_gate_state") or "")
    if state not in ALLOWED_REPAIR_GATE_STATES:
        raise ValueError(f"repair_gate_state must be an allowed repair gate state, got {state!r}")
    blockers = artifact.get("blocker_reasons")
    blocker_rows = blockers if isinstance(blockers, list) else []
    scope = artifact.get("repair_allowed_scope")
    unblocked = state == "unblocked_for_bounded_repair_ladder"
    if unblocked and blocker_rows:
        raise ValueError("unblocked repair gate cannot include blocker reasons")
    if unblocked and not valid_repair_scope(scope):
        raise ValueError("unblocked repair gate requires a positive repair scope")
    if unblocked and artifact.get("downstream_gated_skip_expected") is not False:
        raise ValueError("unblocked repair gate must not expect downstream skip")
    if not unblocked and scope is not None:
        raise ValueError("blocked repair gate must set repair scope to null")
    if not unblocked and not blocker_rows:
        raise ValueError("blocked repair gate must include blocker reasons")
    if not unblocked and artifact.get("downstream_gated_skip_expected") is not True:
        raise ValueError("blocked repair gate must expect downstream skip")
    substrate = mapping_value(artifact.get("inference_substrate"))
    if substrate.get("live_model_calls") or substrate.get("repair_calls"):
        raise ValueError("Exp 3198 must not perform live model or repair calls")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith("complete:"):
        raise ValueError("Exp 3198 honest_verdict must start with complete:")


def valid_repair_scope(scope: Any) -> bool:
    """Return true only for a finite positive bounded repair scope."""

    repair_scope = mapping_value(scope)
    return (
        repair_scope.get("enabled") is True
        and int_value(repair_scope.get("max_total_repair_attempts")) > 0
        and int_value(repair_scope.get("max_attempts_per_row")) > 0
        and int_value(repair_scope.get("max_distinct_rows")) > 0
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the truthful terminal verdict string for conductor consumers."""

    state = str(artifact.get("repair_gate_state") or "blocked_other_precondition")
    blockers = artifact.get("blocker_reasons")
    blocker_count = len(blockers) if isinstance(blockers, list) else 0
    skip = artifact.get("downstream_gated_skip_expected")
    if state == "unblocked_for_bounded_repair_ladder":
        scope = mapping_value(artifact.get("repair_allowed_scope"))
        return (
            "complete: repair_gate_state=unblocked_for_bounded_repair_ladder; "
            f"max_distinct_rows={scope.get('max_distinct_rows')}; "
            f"max_total_repair_attempts={scope.get('max_total_repair_attempts')}"
        )
    return (
        f"complete: repair_gate_state={state}; blocker_count={blocker_count}; "
        f"downstream_gated_skip_expected={str(skip).lower()}"
    )


def has_blocker(blockers: Sequence[Mapping[str, Any]], code: str) -> bool:
    """Return whether a blocker code is present."""

    return any(row.get("code") == code for row in blockers)


def mapping_value(value: Any) -> JsonDict:
    """Normalize optional object-shaped JSON values to dictionaries."""

    return dict(value) if isinstance(value, Mapping) else {}


def finite_rate(value: Any) -> float | None:
    """Return a finite rate in [0, 1], or None for malformed metric evidence."""

    if not isinstance(value, (int, float)):
        return None
    rate = float(value)
    if not math.isfinite(rate) or rate < 0.0 or rate > 1.0:
        return None
    return rate


def finite_nonnegative_number(value: Any) -> float | None:
    """Return a finite nonnegative number, or None when the JSON value is malformed."""

    if not isinstance(value, (int, float)):
        return None
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        return None
    return number


def int_value(value: Any) -> int:
    """Convert nonnegative integer counters while failing malformed values to zero."""

    return int(value) if isinstance(value, int) and value >= 0 else 0


def sha256_file(path: Path) -> str | None:
    """Checksum local source bytes so the decision traces to exact evidence."""

    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
