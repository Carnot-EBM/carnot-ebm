"""Exp 3131 KAN PWA/MILP verifier abstraction audit.

Spec refs: REQ-KAN-3131, SCENARIO-KAN-3131.

This audit is intentionally narrow. It does not introduce a new verifier tier
or claim any deployed improvement. It inspects the existing tiny KAN-style
PWA/MILP fixture from Exp 2876, records the local and propagated global error
accounting, and checks the bounded property through the local MILP-compatible
path that already exists in `carnot.verify.kan_pwa_milp_corrigendum`.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any

from carnot.verify.kan_pwa_milp_corrigendum import (
    CorrigendumFixture,
    CorrigendumSolveResult,
    build_corrigendum_fixture,
    solve_property,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260526"
ARTIFACT = "experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.kan_pwa_milp_verifier_abstraction_audit.v1"
RESULT_PATH = REPO_ROOT / "results" / ARTIFACT_FILENAME
TERMINAL_SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "kan_pwa_milp_audit_v1_ready",
        "kan_code_present",
        "abstraction_count",
        "local_error_bound_summary",
        "global_error_bound_summary",
        "milp_property_check_count",
        "milp_property_pass_count",
        "implementation_blockers",
        "tests_run",
        "source_artifacts",
        "inference_substrate",
        "honest_verdict",
    }
)
KAN_CODE_PATHS = (
    Path("python/carnot/verify/kan_pwa_milp_corrigendum.py"),
    Path("python/carnot/verify/kan_pwa_milp_tiny.py"),
    Path("python/carnot/verify/pwa_kan.py"),
)
SOURCE_ARTIFACT_PATHS = (
    (Path("python/carnot/verify/kan_pwa_milp_corrigendum.py"), "existing_two_unit_pwa_milp_fixture"),
    (Path("python/carnot/verify/kan_pwa_milp_tiny.py"), "prior_one_unit_pwa_property_fixture"),
    (Path("openspec/capabilities/kan/spec.md"), "req_kan_3131_schema_anchor"),
    (Path("_bmad/architecture.md"), "kaem_architecture_context"),
    (Path("research-references.md"), "kan_pwa_milp_research_context"),
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime hooks for writing the deterministic Exp 3131 artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    backend_name: str | None = None
    tests_run: Sequence[str] = field(default_factory=tuple)

    def artifact_path(self) -> Path:
        """Return the target JSON artifact path."""

        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def start_time(self) -> float:
        """Return the deterministic or live start time."""

        return self.clock() if self.started_at is None else self.started_at


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and write the Exp 3131 audit artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    artifact = build_experiment_artifact(active, duration_s=_round(active.clock() - started))
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def build_experiment_artifact(config: ExperimentConfig, *, duration_s: float) -> JsonDict:
    """Build the complete Exp 3131 artifact from local PWA/MILP code."""

    code_present = kan_code_present(config.repo_root)
    blockers = [] if code_present else implementation_blockers_for_missing_code()
    fixture = build_source_fixture()
    local_summary = local_error_bound_summary(fixture)
    global_summary = global_error_bound_summary(fixture)
    property_checks = milp_property_checks(fixture, backend_name=config.backend_name)
    abstraction_count = len(fixture.units) if code_present else 0
    ready = compute_readiness(code_present, abstraction_count, property_checks, blockers)
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "kan_pwa_milp_audit_v1_ready": ready,
        "kan_code_present": code_present,
        "abstraction_count": abstraction_count,
        "local_error_bound_summary": local_summary,
        "global_error_bound_summary": global_summary,
        "milp_property_check_count": len(property_checks),
        "milp_property_pass_count": sum(
            1 for check in property_checks if check.get("property_verified") is True
        ),
        "milp_property_checks": property_checks,
        "implementation_blockers": blockers,
        "tests_run": list(config.tests_run),
        "source_artifacts": source_artifacts(config.repo_root),
        "inference_substrate": inference_substrate(),
        "honest_verdict": honest_verdict(ready),
        "claim_boundary": claim_boundary(),
        "field_principles": field_principles(),
        "duration_s": duration_s,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_source_fixture() -> CorrigendumFixture:
    """Return the existing tiny two-unit PWA/MILP fixture."""

    return build_corrigendum_fixture()


def kan_code_present(repo_root: Path = REPO_ROOT) -> bool:
    """Return true only when the local KAN/PWA verifier files are present."""

    return all((repo_root / path).is_file() for path in KAN_CODE_PATHS)


def local_error_bound_summary(fixture: CorrigendumFixture) -> JsonDict:
    """Summarize per-unit local PWA approximation errors."""

    per_unit = [
        {
            "name": unit.name,
            "local_error_bound": unit.local_error_bound,
            "segment_count": len(unit.segments),
            "segment_local_error_bounds": [
                segment.local_error_bound for segment in unit.segments
            ],
        }
        for unit in fixture.units
    ]
    return {
        "procedure": fixture.bound_procedures()["local_error_bound"],
        "unit_count": len(fixture.units),
        "segment_count": sum(unit["segment_count"] for unit in per_unit),
        "max_local_error_bound": fixture.local_error_bound,
        "per_unit": per_unit,
    }


def global_error_bound_summary(fixture: CorrigendumFixture) -> JsonDict:
    """Summarize propagated output-level error accounting."""

    contributions = [
        {
            "name": unit.name,
            "output_weight": unit.output_weight,
            "local_error_bound": unit.local_error_bound,
            "contribution": abs(unit.output_weight) * unit.local_error_bound,
        }
        for unit in fixture.units
    ]
    return {
        "procedure": fixture.bound_procedures()["global_error_bound"],
        "global_error_bound": fixture.global_error_bound,
        "bounds_distinct_by_construction": fixture.bounds_distinct_by_construction,
        "weighted_contributions": contributions,
    }


def milp_property_checks(
    fixture: CorrigendumFixture,
    *,
    backend_name: str | None = None,
) -> list[JsonDict]:
    """Run the existing bounded property check and return audit-safe fields."""

    result = solve_property(fixture, backend_name=backend_name)
    return [milp_property_check_payload(fixture, result)]


def milp_property_check_payload(
    fixture: CorrigendumFixture,
    result: CorrigendumSolveResult,
) -> JsonDict:
    """Convert the solver result into the Exp 3131 property-check schema."""

    return {
        "property_statement": (
            "For all x in "
            f"[{fixture.property_lower_x}, {fixture.property_upper_x}], "
            f"weighted PWA upper envelope <= {fixture.property_threshold}."
        ),
        "property_threshold": fixture.property_threshold,
        "property_verified": result.property_verified,
        "certified_upper_bound": result.certified_upper_bound,
        "witness_x": result.witness_x,
        "milp_backend_available": result.milp_backend_available,
        "milp_backend_name": result.milp_backend_name,
        "solver_status": result.solver_status,
        "exact_enumeration_used_only_as_fallback": result.exact_enumeration_used_only_as_fallback,
        "counterexample_or_certificate": result.counterexample_or_certificate,
    }


def compute_readiness(
    code_present: bool,
    abstraction_count: int,
    property_checks: Sequence[Mapping[str, Any]],
    implementation_blockers: Sequence[str],
) -> bool:
    """Return the terminal readiness bit without promoting broader verifier claims."""

    return (
        code_present
        and abstraction_count > 0
        and not implementation_blockers
        and len(property_checks) > 0
        and all(check.get("property_verified") is True for check in property_checks)
        and all(check.get("solver_status") == "optimal" for check in property_checks)
    )


def implementation_blockers_for_missing_code() -> list[str]:
    """Name the exact minimum boundary required before implementation claims."""

    return [
        "python/carnot/verify/kan_pwa_milp_corrigendum.py",
        "python/carnot/verify/kan_pwa_milp_tiny.py",
        "openspec/capabilities/kan/spec.md: REQ-KAN-3131",
        "tests/python/test_experiment_3131_kan_pwa_milp_verifier_abstraction_audit_v1.py",
    ]


def inference_substrate() -> JsonDict:
    """Declare the execution substrate so solver-only evidence stays bounded."""

    return {
        "mode": "cpu_pwa_milp_abstraction_audit",
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "trained_kan_weight_verification": False,
        "hardware_execution": False,
        "deployed_verifier_improvement_claim": False,
        "solver_only_abstraction_accounting": True,
    }


def source_artifacts(repo_root: Path = REPO_ROOT) -> list[JsonDict]:
    """Return concrete file provenance for the KAN audit evidence."""

    return [_source_artifact(repo_root, path, role) for path, role in SOURCE_ARTIFACT_PATHS]


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when the artifact omits fields or overstates the evidence."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    blockers = _sequence(artifact.get("implementation_blockers"))
    if artifact.get("kan_code_present") is False and not blockers:
        raise ValueError("missing KAN code requires implementation blockers")
    abstraction_count = int(artifact.get("abstraction_count", 0))
    if artifact.get("kan_code_present") is True and abstraction_count <= 0:
        raise ValueError("abstraction_count must be positive when KAN code is present")
    checks = _sequence(artifact.get("milp_property_checks"))
    check_count = int(artifact.get("milp_property_check_count", 0))
    pass_count = int(artifact.get("milp_property_pass_count", 0))
    if check_count != len(checks):
        raise ValueError("milp property check count mismatch")
    if pass_count != sum(1 for check in checks if check.get("property_verified") is True):
        raise ValueError("milp property pass count mismatch")
    if not _sequence(artifact.get("source_artifacts")):
        raise ValueError("source_artifacts must cite concrete files")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if substrate.get("live_model_inference") is not False:
        raise ValueError("live model inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")
    if substrate.get("hardware_execution") is not False:
        raise ValueError("hardware execution must remain false")
    if substrate.get("deployed_verifier_improvement_claim") is not False:
        raise ValueError("deployed verifier improvement claims are forbidden")
    if artifact.get("kan_pwa_milp_audit_v1_ready") is True:
        if artifact.get("kan_code_present") is not True:
            raise ValueError("ready audit requires KAN code")
        if blockers:
            raise ValueError("ready audit cannot have implementation blockers")
        if check_count == 0 or pass_count != check_count:
            raise ValueError("ready audit requires every MILP property check to pass")


def honest_verdict(ready: bool) -> str:
    """Return the terminal verdict string required by the conductor schema."""

    if ready:
        return "complete_kan_pwa_milp_abstraction_audit_v1_z3_property_passed_no_deployed_claim"
    return "complete_kan_pwa_milp_abstraction_audit_v1_design_boundary_only"


def claim_boundary() -> JsonDict:
    """State what this audit does and does not prove."""

    return {
        "proves": "bounded two-unit PWA abstraction accounting and one local property check",
        "does_not_prove": [
            "general KAN verifier soundness",
            "trained-network soundness",
            "deployed verifier improvement",
            "hardware execution",
            "live LLM inference",
        ],
    }


def field_principles() -> JsonDict:
    """Map required fields to the evidence discipline they enforce."""

    return {
        "kan_pwa_milp_audit_v1_ready": "KAN architecture work must produce a concrete audit",
        "kan_code_present": "implementation claims require code",
        "abstraction_count": "PWA abstraction scale must be visible",
        "local_error_bound_summary": "unit-level approximation error must be explicit",
        "global_error_bound_summary": "network-level claims need propagated bounds",
        "milp_property_check_count": "verification claims need property checks",
        "milp_property_pass_count": "pass/fail must be separated",
        "implementation_blockers": "design-only output must say what is missing",
        "tests_run": "verifier/abstraction code must be checked",
        "source_artifacts": "KAN evidence must trace to concrete files",
        "inference_substrate": "solver-only work must declare no live LLM inference",
        "honest_verdict": "terminal verdict must use a success prefix unless blocked",
    }


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash deterministic audit payload fields, excluding duration."""

    payload = {
        key: value
        for key, value in artifact.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _source_artifact(repo_root: Path, path: Path, role: str) -> JsonDict:
    full_path = repo_root / path
    return {
        "path": str(path),
        "role": role,
        "exists": full_path.is_file(),
        "sha256": _sha256_file(full_path),
    }


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _round(value: float) -> float:
    return round(float(value), 6)


def main() -> None:  # pragma: no cover
    """CLI entrypoint for writing the requested result artifact."""

    artifact = run_experiment()
    print(
        json.dumps(
            {
                "artifact": str(RESULT_PATH),
                "kan_pwa_milp_audit_v1_ready": artifact["kan_pwa_milp_audit_v1_ready"],
                "milp_property_pass_count": artifact["milp_property_pass_count"],
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
