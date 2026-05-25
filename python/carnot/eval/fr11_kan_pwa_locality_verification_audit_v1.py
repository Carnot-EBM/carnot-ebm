"""Exp 3062 KAN/PWA locality verification audit.

This audit deliberately separates two ideas that are easy to blur: Exp 3047
does provide exact, numeric evidence about which controller anchors changed,
but it does not expose trained KAN spline weights that a PWA/MILP verifier can
certify.  The smallest bounded object here is therefore the controller anchor
delta set, not a KAN model.  The artifact records the exact controller bound
while refusing to promote it into a KAN model-weight verification claim.

Spec refs: REQ-LEARN-3062, SCENARIO-LEARN-3062,
SCENARIO-LEARN-3062-BLOCKED.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import time
from typing import Any


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260525"
ARTIFACT = "experiment_3062_kan_pwa_locality_verification_audit_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
SCHEMA = "carnot.fr11.kan_pwa_locality_verification_audit.v1"
EXP3047_ARTIFACT_REL_PATH = Path("results/experiment_3047_kan_locality_nonforgetting_probe_v2.json")
EXP3061_ARTIFACT_REL_PATH = Path(
    "results/experiment_3061_fr11_delayed_regression_solver_self_model_pilot_v1.json"
)
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
BLOCKED_VERDICT = "blocked_missing_kan_locality_or_delayed_regression_source"
REQUIRED_ARTIFACT_FIELDS = frozenset(
    {
        "kan_pwa_verification_ready",
        "locality_bound",
        "approximation_error_bound",
        "prior_retention_bound",
        "verification_path",
        "tests_or_checks_run",
        "promotion_decision",
        "spec_updates",
        "inference_substrate",
        "honest_verdict",
    }
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic hooks for Exp 3062."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    exp3047_artifact_path: Path | None = None
    exp3061_artifact_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_or_checks_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / ARTIFACT_FILENAME

    def resolved_exp3047_artifact_path(self) -> Path:
        return self.exp3047_artifact_path or self.repo_root / EXP3047_ARTIFACT_REL_PATH

    def resolved_exp3061_artifact_path(self) -> Path:
        return self.exp3061_artifact_path or self.repo_root / EXP3061_ARTIFACT_REL_PATH


@dataclass(frozen=True)
class SourceBundle:
    """Loaded source artifacts that bound the controller locality claim."""

    exp3047_artifact: JsonDict
    exp3061_artifact: JsonDict


@dataclass(frozen=True)
class LocalityObject:
    """Smallest source object with enough structure for an exact bound."""

    object_type: str
    locality_bound: float
    nonlocal_change_bound: float
    total_anchor_count: int
    changed_anchor_count: int
    anchored_prior_count: int
    changed_anchor_keys: tuple[str, ...]
    anchored_prior_keys: tuple[str, ...]

    def as_serializable(self) -> JsonDict:
        return {
            "object_type": self.object_type,
            "locality_bound": self.locality_bound,
            "nonlocal_change_bound": self.nonlocal_change_bound,
            "total_anchor_count": self.total_anchor_count,
            "changed_anchor_count": self.changed_anchor_count,
            "anchored_prior_count": self.anchored_prior_count,
            "changed_anchor_keys": list(self.changed_anchor_keys),
            "anchored_prior_keys": list(self.anchored_prior_keys),
            "bound_definition": (
                "locality_bound = 1 - changed_controller_anchor_count / "
                "total_controller_anchor_count"
            ),
        }


@dataclass(frozen=True)
class VerificationAudit:
    """Bound result for the controller-only locality evidence."""

    kan_pwa_verification_ready: bool
    locality_bound: float
    prior_retention_bound: float
    approximation_error_bound: float
    verification_path: str
    exact_controller_anchor_bound_available: bool
    promotion_decision: str
    claim_promotion_useful: bool
    smallest_locality_object: LocalityObject


def run_experiment(config: ExperimentConfig | None = None) -> JsonDict:
    """Build, validate, and persist the Exp 3062 audit artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    sources = load_source_bundle(active)
    blocker = precondition_blocker(sources)
    duration_s = _round(active.clock() - started)
    if blocker is not None:
        artifact = blocked_artifact(active, blocker, duration_s)
        validate_artifact(artifact)
        _write_json(active.artifact_path(), artifact)
        return artifact

    audit = measure_verification_audit(sources)
    artifact = complete_artifact(active, sources, audit, duration_s)
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def load_source_bundle(config: ExperimentConfig) -> SourceBundle:
    """Load Exp 3047 locality evidence and Exp 3061 delayed-regression evidence."""

    return SourceBundle(
        exp3047_artifact=_read_json(config.resolved_exp3047_artifact_path()),
        exp3061_artifact=_read_json(config.resolved_exp3061_artifact_path()),
    )


def precondition_blocker(sources: SourceBundle) -> str | None:
    """Return the first source condition that prevents a bounded audit."""

    exp3047_blocker = _source_blocker(
        "exp3047",
        sources.exp3047_artifact,
        ready_field="kan_locality_probe_ready",
        not_ready_reason="locality_not_ready",
    )
    if exp3047_blocker is not None:
        return exp3047_blocker
    return _source_blocker(
        "exp3061",
        sources.exp3061_artifact,
        ready_field="fr11_delayed_regression_ready",
        not_ready_reason="delayed_regression_not_ready",
    )


def identify_smallest_locality_object(sources: SourceBundle) -> LocalityObject:
    """Extract the controller anchor delta set from Exp 3047."""

    report = _mapping(sources.exp3047_artifact.get("locality_report"))
    total_anchor_count = int(report.get("total_anchor_count", 0))
    changed_anchor_count = int(sources.exp3047_artifact.get("changed_anchor_count", 0))
    anchored_prior_count = int(sources.exp3047_artifact.get("anchored_prior_count", 0))
    if total_anchor_count <= 0:
        raise ValueError("cannot identify locality object without controller anchors")

    locality_bound = _round(float(sources.exp3047_artifact.get("locality_metric", 0.0)))
    nonlocal_change_bound = _round(changed_anchor_count / total_anchor_count)
    return LocalityObject(
        object_type="controller_anchor_delta_set",
        locality_bound=locality_bound,
        nonlocal_change_bound=nonlocal_change_bound,
        total_anchor_count=total_anchor_count,
        changed_anchor_count=changed_anchor_count,
        anchored_prior_count=anchored_prior_count,
        changed_anchor_keys=tuple(
            str(item) for item in _sequence(report.get("changed_anchor_keys"))
        ),
        anchored_prior_keys=tuple(
            str(item) for item in _sequence(report.get("anchored_prior_keys"))
        ),
    )


def measure_verification_audit(sources: SourceBundle) -> VerificationAudit:
    """Measure the exact controller bound and the KAN/PWA promotion boundary."""

    blocker = precondition_blocker(sources)
    if blocker is not None:
        raise ValueError(f"cannot audit locality with blocked sources: {blocker}")

    locality_object = identify_smallest_locality_object(sources)
    prior_retention_bound = min(
        float(sources.exp3047_artifact.get("prior_retention_delta", 0.0)),
        float(sources.exp3061_artifact.get("prior_retention_delta", 0.0)),
    )
    return VerificationAudit(
        kan_pwa_verification_ready=False,
        locality_bound=locality_object.locality_bound,
        prior_retention_bound=_round(prior_retention_bound),
        approximation_error_bound=0.0,
        verification_path="exact_controller_anchor_audit",
        exact_controller_anchor_bound_available=True,
        promotion_decision="controller_locality_evidence_only",
        claim_promotion_useful=False,
        smallest_locality_object=locality_object,
    )


def complete_artifact(
    config: ExperimentConfig,
    sources: SourceBundle,
    audit: VerificationAudit,
    duration_s: float,
) -> JsonDict:
    """Build the complete artifact without implying trained-KAN verification."""

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "kan_pwa_verification_ready": audit.kan_pwa_verification_ready,
        "locality_bound": audit.locality_bound,
        "approximation_error_bound": audit.approximation_error_bound,
        "prior_retention_bound": audit.prior_retention_bound,
        "verification_path": audit.verification_path,
        "tests_or_checks_run": list(config.tests_or_checks_run),
        "promotion_decision": audit.promotion_decision,
        "spec_updates": spec_updates(),
        "inference_substrate": inference_substrate(trained_kan_weight_verification=False),
        "honest_verdict": "complete_kan_pwa_locality_exact_controller_audit_not_promoted",
        "smallest_locality_object": audit.smallest_locality_object.as_serializable(),
        "exact_controller_anchor_bound_available": audit.exact_controller_anchor_bound_available,
        "claim_promotion_useful": audit.claim_promotion_useful,
        "source_artifacts": source_artifacts(sources, config),
        "field_principles": field_principles(),
        "duration_s": duration_s,
    }


def blocked_artifact(config: ExperimentConfig, reason: str, duration_s: float) -> JsonDict:
    """Build a fail-closed artifact when source evidence is unavailable."""

    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "kan_pwa_verification_ready": False,
        "locality_bound": 0.0,
        "approximation_error_bound": 0.0,
        "prior_retention_bound": 0.0,
        "verification_path": "blocked_missing_source",
        "tests_or_checks_run": list(config.tests_or_checks_run),
        "promotion_decision": "blocked",
        "spec_updates": spec_updates(),
        "inference_substrate": inference_substrate(trained_kan_weight_verification=False),
        "honest_verdict": BLOCKED_VERDICT,
        "blocked_reason": reason,
        "smallest_locality_object": {},
        "exact_controller_anchor_bound_available": False,
        "claim_promotion_useful": False,
        "source_artifacts": {},
        "field_principles": field_principles(),
        "duration_s": duration_s,
    }


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3062 artifact overclaims the available evidence."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        raise ValueError("inference_substrate must be a mapping")
    if substrate.get("live_llm_inference") is not False:
        raise ValueError("live LLM inference must remain false")
    if (
        substrate.get("model_weight_training") is not False
        or substrate.get("model_weight_mutation") is not False
    ):
        raise ValueError("model weights must not be trained or mutated")

    if artifact.get("promotion_decision") not in {"blocked", "controller_locality_evidence_only"}:
        raise ValueError("promotion_decision must stay controller-only or blocked")
    if artifact.get("claim_promotion_useful") is True:
        raise ValueError("claim_promotion_useful must remain false for controller-only evidence")
    if (
        artifact.get("kan_pwa_verification_ready") is True
        and substrate.get("trained_kan_weight_verification") is not True
    ):
        raise ValueError("trained KAN weight verification is required before KAN/PWA readiness")

    ready = artifact.get("honest_verdict") != BLOCKED_VERDICT
    if not ready:
        return
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must use a terminal success prefix")
    if float(artifact.get("locality_bound") or 0.0) < 0.0:
        raise ValueError("locality_bound must be nonnegative")
    if float(artifact.get("prior_retention_bound") or 0.0) < 0.0:
        raise ValueError("prior_retention_bound must be nonnegative")
    if float(artifact.get("approximation_error_bound") or 0.0) < 0.0:
        raise ValueError("approximation_error_bound must be nonnegative")
    if not str(artifact.get("verification_path", "")):
        raise ValueError("verification_path must be explicit")


def inference_substrate(*, trained_kan_weight_verification: bool) -> JsonDict:
    """Return the execution boundary so exact CPU audit work is not inflated."""

    return {
        "mode": "exact_cpu_controller_anchor_audit",
        "cached_artifacts_only": True,
        "live_llm_inference": False,
        "live_model_inference": False,
        "local_gguf_inference": False,
        "exact_cpu_verifier": True,
        "controller_weight_update": False,
        "trace_memory_update": False,
        "model_weight_training": False,
        "model_weight_mutation": False,
        "kan_model_weight_training": False,
        "trained_kan_weight_verification": trained_kan_weight_verification,
        "training_scope": "audit_only_no_learning",
    }


def source_artifacts(sources: SourceBundle, config: ExperimentConfig) -> JsonDict:
    """Return source provenance with checksums for replayable audit inputs."""

    exp3047_path = config.resolved_exp3047_artifact_path()
    exp3061_path = config.resolved_exp3061_artifact_path()
    return {
        "exp3047_artifact": str(_relative_to(config.repo_root, exp3047_path)),
        "exp3047_ready": sources.exp3047_artifact.get("kan_locality_probe_ready") is True,
        "exp3047_sha256": _sha256_file(exp3047_path),
        "exp3061_artifact": str(_relative_to(config.repo_root, exp3061_path)),
        "exp3061_ready": sources.exp3061_artifact.get("fr11_delayed_regression_ready") is True,
        "exp3061_sha256": _sha256_file(exp3061_path),
    }


def spec_updates() -> list[str]:
    """Return the OpenSpec delta that anchors this implementation."""

    return ["openspec/capabilities/self-learning/spec.md: REQ-LEARN-3062"]


def field_principles() -> JsonDict:
    """Return compact reasons for fields that keep the verdict honest."""

    return {
        "kan_pwa_verification_ready": "KAN verification claims require trained-KAN exact or bounded evidence",
        "locality_bound": "locality claims need a numeric bound",
        "approximation_error_bound": "PWA abstraction error must be visible even when zero by exact audit",
        "prior_retention_bound": "nonforgetting must be bounded",
        "verification_path": "implementation or audit path must be explicit",
        "tests_or_checks_run": "formulaic verifier changes need execution evidence",
        "promotion_decision": "controller evidence must not imply model-weight learning",
        "spec_updates": "implementation changes must be spec-anchored",
        "inference_substrate": "exact CPU verifier work must not be confused with live LLM inference",
        "honest_verdict": "terminal verdict must start with a success prefix unless blocked",
    }


def _source_blocker(
    prefix: str,
    artifact: Mapping[str, Any],
    *,
    ready_field: str,
    not_ready_reason: str,
) -> str | None:
    if not artifact:
        return f"{prefix}_artifact_missing_or_empty"
    if artifact.get("_malformed") is True:
        return f"{prefix}_artifact_malformed"
    if not str(artifact.get("honest_verdict", "")).startswith(TERMINAL_SUCCESS_PREFIXES):
        return f"{prefix}_not_terminal"
    if artifact.get(ready_field) is not True:
        return f"{prefix}_{not_ready_reason}"
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping):
        return f"{prefix}_inference_substrate_missing"
    if substrate.get("live_llm_inference") is not False:
        return f"{prefix}_live_llm_inference_claimed"
    if (
        substrate.get("model_weight_training") is True
        or substrate.get("model_weight_mutation") is True
    ):
        return f"{prefix}_model_weight_learning_claimed"
    return None


def _read_json(path: Path) -> JsonDict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    except (OSError, json.JSONDecodeError):
        return {"_malformed": True}
    return dict(payload) if isinstance(payload, Mapping) else {"_malformed": True}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        return ""


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _sequence(value: Any) -> list[Any]:
    return list(value) if isinstance(value, Sequence) and not isinstance(value, str | bytes) else []


def _round(value: float) -> float:
    return round(float(value), 6)


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.resolve().relative_to(root.resolve())
    except ValueError:
        return path
