"""Exp 3230 FR-11 KAN-CL certificate boundary audit.

Spec refs: REQ-LEARN-3230, SCENARIO-LEARN-3230,
SCENARIO-LEARN-3230-BLOCKED.

This module defines the certificates a future KAN-CL sidecar would need before
promotion into FR-11 continuous self-learning.  It is deliberately an audit of
checked-in evidence only.  It does not train KAN weights, update base model
weights, or grant sidecar verifier authority from empirical nonforgetting
metrics alone.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260528"
MILESTONE = "2026.05.298"
SCHEMA_VERSION = "1.0"
EXPERIMENT_ID = "experiment_3230_kan_cl_certificate_boundary_audit_v2"
SCHEMA = "carnot.fr11.kan_cl_certificate_boundary_audit.v2"
INFERENCE_SUBSTRATE = "checked_in_artifact_certificate_boundary_audit_no_training"
OUTPUT_REL_PATH = Path("results/experiment_3230_kan_cl_certificate_boundary_audit_v2.json")
EXP3201_REL_PATH = Path("results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json")
EXP3216_REL_PATH = Path(
    "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json"
)
SPEC_REL_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_REL_PATH = Path("python/carnot/eval/fr11_kan_cl_certificate_boundary_audit_v2.py")
TEST_REL_PATH = Path(
    "tests/python/test_experiment_3230_kan_cl_certificate_boundary_audit_v2.py"
)
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_")
MUTATION_FLAGS = (
    "executes_live_model_inference",
    "model_weight_learning",
    "model_weight_training",
    "model_weight_mutation",
    "base_model_weights_updated",
    "kan_model_weight_training",
    "hidden_state_mutation_claimed",
)
REQUIRED_ARTIFACT_FIELDS = {
    "schema_version",
    "experiment_id",
    "milestone",
    "continuous_self_learning_task",
    "source_artifacts",
    "certificate_requirements",
    "requirement_evidence_matrix",
    "missing_certificate_count",
    "per_knot_budget_defined",
    "pwa_milp_abstraction_ready",
    "certificate_boundary_ready",
    "kan_sidecar_promotion_allowed",
    "model_weight_update_claimed",
    "inference_substrate",
    "conductor_file_modified",
    "active_roadmap_modified",
    "honest_verdict",
}
DEFAULT_TESTS_RUN = (
    ".venv/bin/pytest -o addopts='' "
    "tests/python/test_experiment_3230_kan_cl_certificate_boundary_audit_v2.py -q",
    ".venv/bin/coverage run -m pytest -o addopts='' "
    "tests/python/test_experiment_3230_kan_cl_certificate_boundary_audit_v2.py -q",
    ".venv/bin/coverage report "
    "--include='python/carnot/eval/fr11_kan_cl_certificate_boundary_audit_v2.py' "
    "--fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_3230_kan_cl_certificate_boundary_audit_v2.py",
    ".venv/bin/pytest tests/python -q",
)
SOURCE_ARTIFACTS = (
    ("agents_repo_instructions", Path("AGENTS.md"), False),
    ("codex_repo_workflow", Path("CODEX.md"), False),
    ("claude_no_hidden_weight_update_rules", Path("CLAUDE.md"), False),
    ("research_program", Path("research-program.md"), False),
    ("research_references", Path("research-references.md"), False),
    ("self_learning_openspec", SPEC_REL_PATH, False),
    ("exp3201_kan_cl_sidecar_audit", EXP3201_REL_PATH, True),
    ("exp3216_nonforgetting_queue", EXP3216_REL_PATH, True),
    ("exp3230_module", MODULE_REL_PATH, False),
    ("exp3230_tests", TEST_REL_PATH, False),
)


def read_json_object(path: Path) -> JsonDict:
    """Read a JSON object, treating unavailable evidence as absent."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):  # pragma: no cover - defensive bad evidence path
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def load_sources(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the prior sidecar audit and nonforgetting queue artifacts."""

    root_path = Path(root)
    return {
        "exp3201": read_json_object(root_path / EXP3201_REL_PATH),
        "exp3216": read_json_object(root_path / EXP3216_REL_PATH),
    }


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal Exp 3230 certificate boundary artifact."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    sources = load_sources(root_path)
    blocker = source_blocker(sources)
    if blocker:
        artifact = blocked_artifact(root_path, blocker, start, now_s, tests_run)
        validate_artifact(artifact)
        return artifact

    exp3201 = sources["exp3201"]
    exp3216 = sources["exp3216"]
    matrix = requirement_evidence_matrix(exp3201, exp3216)
    artifact = {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_artifacts": source_artifacts(root_path),
        "certificate_requirements": certificate_requirements(),
        "requirement_evidence_matrix": matrix,
        "missing_certificate_count": missing_certificate_count(matrix),
        "per_knot_budget_defined": per_knot_budget_defined(matrix),
        "pwa_milp_abstraction_ready": pwa_milp_abstraction_ready(matrix),
        "certificate_boundary_ready": certificate_boundary_ready(matrix),
        "kan_sidecar_promotion_allowed": kan_sidecar_promotion_allowed(matrix, exp3201, exp3216),
        "model_weight_update_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "source_safety": source_safety_summary(exp3201, exp3216),
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(start, now_s),
        "honest_verdict": honest_verdict(matrix),
    }
    validate_artifact(artifact)
    return artifact


def blocked_artifact(
    root: Path,
    blocker: str,
    started_s: float,
    now_s: float | None,
    tests_run: Sequence[str] | None,
) -> JsonDict:
    """Return a schema-complete fail-closed artifact for unsafe sources."""

    matrix = blocked_evidence_matrix(blocker)
    return {
        "artifact": EXPERIMENT_ID,
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "continuous_self_learning_task": True,
        "source_artifacts": source_artifacts(root),
        "certificate_requirements": certificate_requirements(),
        "requirement_evidence_matrix": matrix,
        "missing_certificate_count": missing_certificate_count(matrix),
        "per_knot_budget_defined": False,
        "pwa_milp_abstraction_ready": False,
        "certificate_boundary_ready": False,
        "kan_sidecar_promotion_allowed": False,
        "model_weight_update_claimed": False,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "conductor_file_modified": False,
        "active_roadmap_modified": False,
        "source_safety": {"blocked_reason": blocker},
        "blocked_reason": blocker,
        "tests_run": list(tests_run or DEFAULT_TESTS_RUN),
        "duration_s": duration(started_s, now_s),
        "honest_verdict": (
            "complete: blocked kan-cl certificate boundary audit; "
            f"{blocker}; kan_sidecar_promotion_allowed=false; "
            "model_weight_update_claimed=false"
        ),
    }


def source_blocker(sources: Mapping[str, Any]) -> str:
    """REQ-LEARN-3230-1: require terminal, no-live, no-mutation sources."""

    checks = []
    for key in ("exp3201", "exp3216"):
        payload = sources.get(key, {})
        safe_mapping = isinstance(payload, Mapping)
        checks.extend(
            [
                (not safe_mapping or not is_terminal(payload), f"{key}_missing_or_not_terminal"),
                (
                    safe_mapping and detected_model_weight_update(payload),
                    f"{key}_model_weight_update_claimed",
                ),
                (
                    safe_mapping and source_claims_live_or_mutation(payload),
                    f"{key}_live_inference_or_weight_update_claimed",
                ),
                (
                    safe_mapping and payload.get("conductor_file_modified") is True,
                    f"{key}_conductor_file_modified",
                ),
                (
                    safe_mapping and payload.get("active_roadmap_modified") is True,
                    f"{key}_active_roadmap_modified",
                ),
            ]
        )
    for blocked, reason in checks:
        if blocked:
            return reason
    return ""


def is_terminal(payload: Mapping[str, Any]) -> bool:
    """Return whether an artifact verdict is terminal enough to consume."""

    return str(payload.get("honest_verdict") or "").startswith(TERMINAL_PREFIXES)


def source_claims_live_or_mutation(payload: Mapping[str, Any]) -> bool:
    """Return whether a source claims fresh inference or hidden mutation."""

    substrate = payload.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return True
    return int(substrate.get("fresh_live_inference_calls") or 0) != 0 or any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS
    )


def detected_model_weight_update(payload: Mapping[str, Any]) -> bool:
    """Return whether source evidence claims a model-weight update."""

    direct_flags = (
        "model_weight_update_claimed",
        "model_weight_update_performed",
        "base_model_weights_updated",
    )
    if any(payload.get(flag) is True for flag in direct_flags):
        return True
    substrate = payload.get("inference_substrate", {})
    return isinstance(substrate, Mapping) and any(
        substrate.get(flag) is True for flag in MUTATION_FLAGS if "weight" in flag
    )


def certificate_requirements() -> list[JsonDict]:
    """REQ-LEARN-3230-2: deterministic KAN sidecar promotion requirements."""

    return [
        {
            "requirement_id": "bounded_input_domains",
            "description": "Every KAN-CL sidecar input feature must have a bounded domain.",
            "evidence_key": "certificate_boundary.bounded_input_domains",
            "promotion_critical": True,
        },
        {
            "requirement_id": "per_knot_budget",
            "description": "Each knot or template anchor needs an explicit nonforgetting budget.",
            "evidence_key": "certificate_boundary.per_knot_budget",
            "promotion_critical": True,
        },
        {
            "requirement_id": "monotonicity_or_lipschitz_evidence",
            "description": "Each promoted knot needs monotonicity or local Lipschitz evidence.",
            "evidence_key": "certificate_boundary.monotonicity_lipschitz_evidence",
            "promotion_critical": True,
        },
        {
            "requirement_id": "pwa_milp_abstraction",
            "description": "The sidecar must expose PWA segments, error bounds, and MILP checks.",
            "evidence_key": "certificate_boundary.pwa_milp_abstraction",
            "promotion_critical": True,
        },
        {
            "requirement_id": "nonforgetting_budget_check",
            "description": "Aggregate held-out, drift, and negative-control budget checks must pass.",
            "evidence_key": "exp3201 metrics plus exp3216 nonforgetting_queue",
            "promotion_critical": True,
        },
        {
            "requirement_id": "model_weight_immutability",
            "description": "The audit must claim no KAN, controller, or base-model weight update.",
            "evidence_key": "inference_substrate and model_weight_update flags",
            "promotion_critical": True,
        },
    ]


def requirement_evidence_matrix(exp3201: Mapping[str, Any], exp3216: Mapping[str, Any]) -> list[JsonDict]:
    """REQ-LEARN-3230-3: map each candidate requirement to artifact evidence."""

    boundary = exp3201.get("certificate_boundary", {})
    boundary = boundary if isinstance(boundary, Mapping) else {}
    evidence = {
        "bounded_input_domains": bounded_input_domain_evidence(boundary),
        "per_knot_budget": per_knot_budget_evidence(boundary),
        "monotonicity_or_lipschitz_evidence": monotonicity_lipschitz_evidence(boundary),
        "pwa_milp_abstraction": pwa_milp_evidence(boundary),
        "nonforgetting_budget_check": nonforgetting_budget_evidence(exp3201, exp3216),
        "model_weight_immutability": model_weight_immutability_evidence(exp3201, exp3216),
    }
    matrix: list[JsonDict] = []
    for requirement in certificate_requirements():
        req_id = requirement["requirement_id"]
        row = dict(requirement)
        row.update(evidence[req_id])
        matrix.append(row)
    return matrix


def bounded_input_domain_evidence(boundary: Mapping[str, Any]) -> JsonDict:
    """Return sidecar input-domain certificate evidence."""

    item = boundary.get("bounded_input_domains", {})
    domains = item.get("domains", []) if isinstance(item, Mapping) else []
    present = isinstance(domains, Sequence) and not isinstance(domains, (str, bytes)) and bool(domains)
    return evidence_row(
        present,
        ["results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json"],
        "bounded input domains are present" if present else "no bounded sidecar input domains",
        "bounded numeric or categorical domain limits for every sidecar input feature",
    )


def per_knot_budget_evidence(boundary: Mapping[str, Any]) -> JsonDict:
    """Return per-knot budget evidence, not aggregate queue evidence."""

    item = boundary.get("per_knot_budget", {})
    budget = item.get("budget", {}) if isinstance(item, Mapping) else {}
    present = isinstance(item, Mapping) and item.get("defined") is True and bool(budget)
    return evidence_row(
        present,
        ["results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json"],
        "per-knot budget is defined" if present else "only aggregate queue evidence is present",
        "per-knot or per-template nonforgetting budget",
    )


def monotonicity_lipschitz_evidence(boundary: Mapping[str, Any]) -> JsonDict:
    """Return per-knot monotonicity or Lipschitz certificate evidence."""

    item = boundary.get("monotonicity_lipschitz_evidence", {})
    checks = item.get("checks", []) if isinstance(item, Mapping) else []
    present = isinstance(item, Mapping) and item.get("defined") is True and bool(checks)
    return evidence_row(
        present,
        ["results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json"],
        "per-knot monotonicity/Lipschitz checks are present"
        if present
        else "no per-knot monotonicity or Lipschitz checks",
        "per-knot monotonicity or local Lipschitz evidence",
    )


def pwa_milp_evidence(boundary: Mapping[str, Any]) -> JsonDict:
    """Return sidecar-specific PWA/MILP abstraction evidence."""

    item = boundary.get("pwa_milp_abstraction", {})
    if isinstance(item, Mapping):
        present = (
            item.get("ready") is True
            and bool(item.get("segments"))
            and bool(item.get("error_bounds"))
            and bool(item.get("property_checks"))
        )
    else:
        present = False
    return evidence_row(
        present,
        ["results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json"],
        "sidecar PWA/MILP abstraction is ready"
        if present
        else "no sidecar-linked PWA/MILP abstraction certificate",
        "PWA segments, error bounds, and MILP-compatible property checks",
    )


def nonforgetting_budget_evidence(exp3201: Mapping[str, Any], exp3216: Mapping[str, Any]) -> JsonDict:
    """Return aggregate nonforgetting budget evidence from Exp 3201 and 3216."""

    present = (
        int(exp3201.get("heldout_replay_count") or 0) > 0
        and int(exp3201.get("drift_replay_count") or 0) > 0
        and int(exp3201.get("negative_control_regression_count") or 0) == 0
        and int(exp3201.get("locality_violation_count") or 0) == 0
        and exp3216.get("nonforgetting_queue_defined") is True
        and exp3216.get("nonforgetting_budget_exceeded") is not True
    )
    return evidence_row(
        present,
        [
            "results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json",
            "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json",
        ],
        "aggregate nonforgetting replay and queue checks pass"
        if present
        else "aggregate nonforgetting replay or queue checks are missing or over budget",
        "held-out/drift replay counts, zero regressions, and queue within budget",
    )


def model_weight_immutability_evidence(exp3201: Mapping[str, Any], exp3216: Mapping[str, Any]) -> JsonDict:
    """Return evidence that no training or hidden mutation was claimed."""

    present = (
        not detected_model_weight_update(exp3201)
        and not detected_model_weight_update(exp3216)
        and not source_claims_live_or_mutation(exp3201)
        and not source_claims_live_or_mutation(exp3216)
    )
    return evidence_row(
        present,
        [
            "results/experiment_3201_kan_cl_nonforgetting_sidecar_audit_v1.json",
            "results/experiment_3216_fr11_grounded_continuation_nonforgetting_queue_v1.json",
        ],
        "sources deny live inference, KAN training, and model-weight updates"
        if present
        else "a source claims live inference or model-weight mutation",
        "no model-weight, KAN-weight, or hidden-state update claim",
    )


def evidence_row(
    present: bool,
    sources: Sequence[str],
    summary: str,
    missing_evidence: str,
) -> JsonDict:
    """Build one normalized evidence matrix row payload."""

    return {
        "evidence_status": "present" if present else "missing",
        "source_artifacts": list(sources),
        "evidence_summary": summary,
        "missing_evidence": "" if present else missing_evidence,
    }


def blocked_evidence_matrix(blocker: str) -> list[JsonDict]:
    """Return an all-missing matrix when source evidence is unsafe."""

    return [
        {
            **requirement,
            "evidence_status": "missing",
            "source_artifacts": [],
            "evidence_summary": f"source evidence blocked before certificate audit: {blocker}",
            "missing_evidence": requirement["description"],
        }
        for requirement in certificate_requirements()
    ]


def missing_certificate_count(matrix: Sequence[Mapping[str, Any]]) -> int:
    """REQ-LEARN-3230-4: count missing certificate rows."""

    return sum(1 for row in matrix if row.get("evidence_status") == "missing")


def matrix_row(matrix: Sequence[Mapping[str, Any]], requirement_id: str) -> Mapping[str, Any]:
    """Return one evidence row by stable requirement ID."""

    return next(row for row in matrix if row.get("requirement_id") == requirement_id)


def per_knot_budget_defined(matrix: Sequence[Mapping[str, Any]]) -> bool:
    """REQ-LEARN-3230-5: distinguish per-knot budgets from aggregate queues."""

    return matrix_row(matrix, "per_knot_budget").get("evidence_status") == "present"


def pwa_milp_abstraction_ready(matrix: Sequence[Mapping[str, Any]]) -> bool:
    """REQ-LEARN-3230-6: report sidecar-specific PWA/MILP readiness."""

    return matrix_row(matrix, "pwa_milp_abstraction").get("evidence_status") == "present"


def certificate_boundary_ready(matrix: Sequence[Mapping[str, Any]]) -> bool:
    """REQ-LEARN-3230-7: all critical certificates must be present."""

    critical_missing = any(
        row.get("promotion_critical") is True and row.get("evidence_status") != "present"
        for row in matrix
    )
    return (
        not critical_missing
        and per_knot_budget_defined(matrix)
        and pwa_milp_abstraction_ready(matrix)
    )


def kan_sidecar_promotion_allowed(
    matrix: Sequence[Mapping[str, Any]],
    exp3201: Mapping[str, Any],
    exp3216: Mapping[str, Any],
) -> bool:
    """REQ-LEARN-3230-8: promotion is false unless all boundaries are present."""

    return (
        certificate_boundary_ready(matrix)
        and not detected_model_weight_update(exp3201)
        and not detected_model_weight_update(exp3216)
        and exp3201.get("conductor_file_modified") is not True
        and exp3216.get("conductor_file_modified") is not True
        and exp3201.get("active_roadmap_modified") is not True
        and exp3216.get("active_roadmap_modified") is not True
    )


def source_safety_summary(exp3201: Mapping[str, Any], exp3216: Mapping[str, Any]) -> JsonDict:
    """Summarize source safety gates in the terminal artifact."""

    return {
        "exp3201_terminal": is_terminal(exp3201),
        "exp3216_terminal": is_terminal(exp3216),
        "exp3201_model_weight_update_claimed": detected_model_weight_update(exp3201),
        "exp3216_model_weight_update_claimed": detected_model_weight_update(exp3216),
        "exp3201_live_or_mutation_claimed": source_claims_live_or_mutation(exp3201),
        "exp3216_live_or_mutation_claimed": source_claims_live_or_mutation(exp3216),
    }


def source_artifacts(root: Path) -> list[JsonDict]:
    """Return source artifact provenance with checksums when files exist."""

    artifacts: list[JsonDict] = []
    for artifact_id, rel_path, required in SOURCE_ARTIFACTS:
        path = root / rel_path
        artifacts.append(
            {
                "id": artifact_id,
                "path": rel_path.as_posix(),
                "required": required,
                "exists": path.exists(),
                "sha256": sha256_file(path),
            }
        )
    return artifacts


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Assert the required Exp 3230 schema and promotion-boundary invariants."""

    assert REQUIRED_ARTIFACT_FIELDS <= artifact.keys()
    assert isinstance(artifact["schema_version"], str)
    assert artifact["experiment_id"] == EXPERIMENT_ID
    assert artifact["milestone"] == MILESTONE
    assert artifact["continuous_self_learning_task"] is True
    assert isinstance(artifact["source_artifacts"], list)
    assert isinstance(artifact["certificate_requirements"], list)
    assert isinstance(artifact["requirement_evidence_matrix"], list)
    assert artifact["missing_certificate_count"] == missing_certificate_count(
        artifact["requirement_evidence_matrix"]
    )
    assert artifact["per_knot_budget_defined"] == per_knot_budget_defined(
        artifact["requirement_evidence_matrix"]
    )
    assert artifact["pwa_milp_abstraction_ready"] == pwa_milp_abstraction_ready(
        artifact["requirement_evidence_matrix"]
    )
    assert artifact["certificate_boundary_ready"] == certificate_boundary_ready(
        artifact["requirement_evidence_matrix"]
    )
    assert artifact["kan_sidecar_promotion_allowed"] is (
        artifact["certificate_boundary_ready"]
        and artifact["model_weight_update_claimed"] is False
        and artifact["conductor_file_modified"] is False
        and artifact["active_roadmap_modified"] is False
    )
    assert artifact["model_weight_update_claimed"] is False
    assert artifact["inference_substrate"] == INFERENCE_SUBSTRATE
    assert artifact["conductor_file_modified"] is False
    assert artifact["active_roadmap_modified"] is False
    assert isinstance(artifact["honest_verdict"], str)


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
    tests_run: Sequence[str] | None = None,
) -> Path:
    """Write the Exp 3230 JSON artifact with stable key ordering."""

    root_path = Path(root)
    artifact = build_artifact(root_path, started_s=started_s, now_s=now_s, tests_run=tests_run)
    path = Path(output_path) if output_path is not None else root_path / OUTPUT_REL_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def honest_verdict(matrix: Sequence[Mapping[str, Any]]) -> str:
    """Return the concise terminal verdict for the certificate audit."""

    missing = missing_certificate_count(matrix)
    per_knot = per_knot_budget_defined(matrix)
    pwa_ready = pwa_milp_abstraction_ready(matrix)
    boundary = certificate_boundary_ready(matrix)
    promotion = boundary
    return (
        "complete: kan-cl certificate boundary audit; "
        f"missing_certificate_count={missing}; "
        f"per_knot_budget_defined={str(per_knot).lower()}; "
        f"pwa_milp_abstraction_ready={str(pwa_ready).lower()}; "
        f"certificate_boundary_ready={str(boundary).lower()}; "
        f"kan_sidecar_promotion_allowed={str(promotion).lower()}; "
        "model_weight_update_claimed=false"
    )


def duration(started_s: float, now_s: float | None = None) -> float:
    """Return elapsed seconds rounded for reproducible artifacts."""

    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def sha256_file(path: Path) -> str | None:
    """Return a SHA-256 checksum when the source artifact exists."""

    if not path.exists():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()
