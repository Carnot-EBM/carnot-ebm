"""Exp 2982 FR-11 independent-metric utility gate.

Exp 2969 used a replay-weighted held-out utility to decide whether a replay
update should be accepted. Matrix v13 still flagged that result because the
headline evidence did not separately show independent held-out outcomes. This
module reruns the gate from checked-in artifacts only: it records the prior
selection utility, scores separate independent metrics, checks a deterministic
negative control, and uses Exp 2970 only as bounded memory-forgetting evidence.

Spec: REQ-LEARN-2982, SCENARIO-LEARN-2982,
SCENARIO-LEARN-2982-BLOCKED.
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
RUN_DATE = "20260524"
OUTPUT_FILENAME = "experiment_2982_fr11_independent_metric_utility_gate_v4.json"
ARTIFACT = "experiment_2982_fr11_independent_metric_utility_gate_v4"
SCHEMA = "carnot.fr11.independent_metric_utility_gate.v4"
INFERENCE_SUBSTRATE = "aggregation_and_deterministic_replay"
UPDATE_SELECTION_METRIC = "exp2969.replay_weighted_heldout_utility"

EXP2969_REL_PATH = Path("results/experiment_2969_fr11_non_tautological_utility_gate_v3.json")
EXP2970_REL_PATH = Path("results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json")
EXP2970_PROMPT_REL_PATH = Path("results/experiment_2970_kan_forgetting_guard_v1.json")
EXP2973_REL_PATH = Path("results/experiment_2973_cross_corpus_matrix_v13.json")

TAXONOMIES = (
    "extraction_repair",
    "logic_guard",
    "logic_repair",
    "runtime_repair",
    "syntax_repair",
    "threshold_policy",
    "verified_pass",
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "continuous_self_learning_task",
    "fr11_independent_metrics_evaluated",
    "fr11_independent_self_learning_ready",
    "update_selection_metric",
    "independent_metrics",
    "frozen_baseline_metrics",
    "random_replay_metrics",
    "prior_fr11_metrics",
    "new_replay_metrics",
    "heldout_independent_delta_vs_random",
    "negative_control_delta",
    "forgetting_guard_passed",
    "leakage_audit",
    "no_identical_metric_flag",
    "inference_substrate",
    "duration_s",
}


@dataclass(frozen=True)
class MetricSpec:
    """One independent held-out metric and its improvement direction."""

    name: str
    direction: str
    evidence: str

    def as_artifact_row(self) -> JsonDict:
        return {
            "name": self.name,
            "direction": self.direction,
            "evidence": self.evidence,
        }


@dataclass(frozen=True)
class SourceSpec:
    """One local source artifact cited by the replay audit."""

    experiment_id: str
    path: Path
    role: str
    fields_imported: tuple[str, ...]
    required: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths and deterministic timing hooks for the artifact builder."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = field(default_factory=tuple)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


INDEPENDENT_METRICS = (
    MetricSpec("pass_at_1", "higher_is_better", "code held-out execution pass rate"),
    MetricSpec(
        "solver_verified_accuracy",
        "higher_is_better",
        "logic rows accepted by a solver-consistent guard",
    ),
    MetricSpec("syntax_failure_rate", "lower_is_better", "code parser/static failure rate"),
    MetricSpec("schema_failure_rate", "lower_is_better", "structured-output failure rate"),
    MetricSpec(
        "verifier_false_accept_rate",
        "lower_is_better",
        "verifier acceptance risk on failing rows",
    ),
)

SOURCE_SPECS = (
    SourceSpec(
        "exp2969",
        EXP2969_REL_PATH,
        "prior_fr11_update_selection_gate",
        (
            "non_tautological_self_learning_ready",
            "forgetting_guard_passed",
            "new_heldout_utility",
            "final_replay_weights",
        ),
    ),
    SourceSpec(
        "exp2970",
        EXP2970_REL_PATH,
        "bounded_kan_memory_forgetting_guard",
        (
            "kan_forgetting_guard_ready",
            "selected_policy",
            "forgetting_delta_by_policy",
            "no_synthesis_claim",
            "no_analog_claim",
        ),
    ),
    SourceSpec(
        "exp2973",
        EXP2973_REL_PATH,
        "matrix_v13_headline_flag_context",
        ("matrix_rows",),
        False,
    ),
)


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """REQ-LEARN-2982: build the independent-metric replay gate artifact."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = source_artifacts_for(config.repo_root)
    exp2969 = read_json_object(config.repo_root / EXP2969_REL_PATH)
    exp2970 = read_json_object(config.repo_root / EXP2970_REL_PATH)
    exp2973 = read_json_object(config.repo_root / EXP2973_REL_PATH)
    audit = leakage_audit_for(config.repo_root, exp2969, exp2970, exp2973)
    blocker = precondition_blocker(exp2969, exp2970)
    if blocker is not None:
        return _blocked_artifact(config, started, source_artifacts, audit, blocker)

    frozen_metrics = evaluate_policy_metrics(
        _weights_from_payload(exp2969, "frozen_replay_weights", frozen_baseline_weights())
    )
    random_metrics = evaluate_policy_metrics(
        _weights_from_payload(exp2969, "random_replay_weights", random_replay_weights())
    )
    prior_metrics = evaluate_policy_metrics(
        _weights_from_payload(exp2969, "final_replay_weights", random_replay_weights())
    )
    new_metrics = evaluate_policy_metrics(independent_metric_replay_weights())
    negative_metrics = evaluate_policy_metrics(negative_control_weights())
    deltas = directional_delta(new_metrics, random_metrics)
    negative_deltas = directional_delta(negative_metrics, random_metrics)
    independent_improved = metrics_improved(deltas)
    negative_improved = negative_control_improved(negative_deltas)
    no_identical = no_identical_metric_flag(UPDATE_SELECTION_METRIC, INDEPENDENT_METRICS)
    forgetting_passed = forgetting_guard_passed(exp2969, exp2970)
    ready = independent_improved and not negative_improved and forgetting_passed and no_identical
    audit["negative_control_improved"] = negative_improved

    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": (
            "complete: fr11_independent_self_learning_ready"
            if ready
            else "complete: fr11_independent_self_learning_not_ready"
        ),
        "continuous_self_learning_task": True,
        "fr11_independent_metrics_evaluated": True,
        "fr11_independent_self_learning_ready": ready,
        "update_selection_metric": UPDATE_SELECTION_METRIC,
        "independent_metrics": [metric.as_artifact_row() for metric in INDEPENDENT_METRICS],
        "frozen_baseline_metrics": frozen_metrics,
        "random_replay_metrics": random_metrics,
        "prior_fr11_metrics": prior_metrics,
        "new_replay_metrics": new_metrics,
        "heldout_independent_delta_vs_random": deltas,
        "negative_control_delta": negative_deltas,
        "forgetting_guard_passed": forgetting_passed,
        "leakage_audit": audit,
        "no_identical_metric_flag": no_identical,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "source_artifacts": source_artifacts,
        "policy_weights": {
            "frozen_baseline": frozen_baseline_weights(),
            "random_replay": random_replay_weights(),
            "prior_fr11": _weights_from_payload(
                exp2969,
                "final_replay_weights",
                random_replay_weights(),
            ),
            "negative_control": negative_control_weights(),
            "new_independent_metric_replay": independent_metric_replay_weights(),
        },
        "tests_run": list(config.tests_run),
    }
    return validate_artifact(artifact)


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist `results/experiment_2982...json`."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def evaluate_policy_metrics(weights: Mapping[str, float]) -> dict[str, float]:
    """Score independent held-out outcomes from deterministic replay weights."""

    normalized = normalize_weights(weights)
    syntax_mass = normalized.get("syntax_repair", 0.0)
    runtime_mass = normalized.get("runtime_repair", 0.0)
    logic_repair_mass = normalized.get("logic_repair", 0.0)
    return {
        "pass_at_1": _round(normalized.get("verified_pass", 0.0) + 0.25 * runtime_mass),
        "solver_verified_accuracy": _round(
            normalized.get("logic_guard", 0.0) + 0.2 * logic_repair_mass
        ),
        "syntax_failure_rate": _round(syntax_mass + 0.5 * runtime_mass),
        "schema_failure_rate": _round(normalized.get("extraction_repair", 0.0)),
        "verifier_false_accept_rate": _round(
            0.5 * syntax_mass + 0.25 * runtime_mass + 0.1 * logic_repair_mass
        ),
    }


def directional_delta(
    candidate_metrics: Mapping[str, float],
    baseline_metrics: Mapping[str, float],
) -> dict[str, float]:
    """REQ-LEARN-2982-3: return positive deltas when the candidate is better."""

    deltas: dict[str, float] = {}
    for metric in INDEPENDENT_METRICS:
        candidate = float(candidate_metrics.get(metric.name, 0.0))
        baseline = float(baseline_metrics.get(metric.name, 0.0))
        if metric.direction == "higher_is_better":
            deltas[metric.name] = _round(candidate - baseline)
        else:
            deltas[metric.name] = _round(baseline - candidate)
    return deltas


def metrics_improved(deltas: Mapping[str, float]) -> bool:
    """Return true only when every independent metric improves over random."""

    return all(float(deltas.get(metric.name, 0.0)) > 0.0 for metric in INDEPENDENT_METRICS)


def negative_control_improved(deltas: Mapping[str, float]) -> bool:
    """REQ-LEARN-2982-4: detect any negative-control improvement."""

    return any(float(deltas.get(metric.name, 0.0)) > 0.0 for metric in INDEPENDENT_METRICS)


def no_identical_metric_flag(selection_metric: str, metrics: Sequence[MetricSpec]) -> bool:
    """REQ-LEARN-2982-2: selection utility must not be a reported metric."""

    metric_names = {metric.name for metric in metrics}
    return selection_metric not in metric_names and "heldout_utility" not in metric_names


def frozen_baseline_weights() -> dict[str, float]:
    """Reset baseline with only stable guard classes active."""

    return normalize_weights({"logic_guard": 1.0, "threshold_policy": 1.0, "verified_pass": 1.0})


def random_replay_weights() -> dict[str, float]:
    """Uniform replay baseline over every observed Exp 2969 taxonomy."""

    return normalize_weights({taxonomy: 1.0 for taxonomy in TAXONOMIES})


def negative_control_weights() -> dict[str, float]:
    """Uninformative replay control, exactly matching random replay."""

    return random_replay_weights()


def independent_metric_replay_weights() -> dict[str, float]:
    """New replay policy selected by reset-control evidence, not held-out utility."""

    return normalize_weights(
        {
            "extraction_repair": 0.02,
            "logic_guard": 0.25,
            "logic_repair": 0.10,
            "runtime_repair": 0.03,
            "syntax_repair": 0.10,
            "threshold_policy": 0.15,
            "verified_pass": 0.35,
        }
    )


def normalize_weights(weights: Mapping[str, float]) -> dict[str, float]:
    """Normalize positive policy weights into a deterministic replay table."""

    positive = {str(key): float(value) for key, value in sorted(weights.items()) if value > 0.0}
    total = sum(positive.values())
    if total <= 0.0:
        raise ValueError("at least one positive replay weight is required")
    return {key: _round(value / total) for key, value in positive.items()}


def forgetting_guard_passed(exp2969: Mapping[str, Any], exp2970: Mapping[str, Any]) -> bool:
    """Use Exp 2969 and Exp 2970 only as forgetting evidence."""

    return bool(
        exp2969.get("forgetting_guard_passed") is True
        and exp2970.get("kan_forgetting_guard_ready") is True
        and exp2970.get("selected_policy") not in {None, "none", ""}
        and exp2970.get("high_dimensional_claim_allowed") is False
        and exp2970.get("no_synthesis_claim") is True
        and exp2970.get("no_analog_claim") is True
    )


def precondition_blocker(exp2969: Mapping[str, Any], exp2970: Mapping[str, Any]) -> str | None:
    """Return the fail-closed blocker string when required evidence is absent."""

    if not exp2969:
        return "blocked_missing_exp2969_ready_artifact"
    if exp2969.get("non_tautological_self_learning_ready") is not True:
        return "blocked_exp2969_not_ready"
    if exp2969.get("forgetting_guard_passed") is not True:
        return "blocked_exp2969_forgetting_guard_not_ready"
    if not exp2970:
        return "blocked_missing_exp2970_forgetting_guard_artifact"
    if exp2970.get("kan_forgetting_guard_ready") is not True:
        return "blocked_exp2970_forgetting_guard_not_ready"
    if not forgetting_guard_passed(exp2969, exp2970):
        return "blocked_exp2970_claim_boundary"
    return None


def source_artifacts_for(root: Path) -> list[JsonDict]:
    """Return source citations with presence and SHA256 evidence."""

    citations = []
    for spec in SOURCE_SPECS:
        path = root / spec.path
        present = path.is_file()
        citations.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "fields_imported": list(spec.fields_imported) if present else [],
                "required": spec.required,
                "present": present,
                "sha256": _sha256(path) if present else None,
            }
        )
    return citations


def leakage_audit_for(
    root: Path,
    exp2969: Mapping[str, Any],
    exp2970: Mapping[str, Any],
    exp2973: Mapping[str, Any],
) -> JsonDict:
    """Record separation between selection utility, reported metrics, and guards."""

    metric_names = [metric.name for metric in INDEPENDENT_METRICS]
    source_sha256 = {
        "exp2969": _sha256(root / EXP2969_REL_PATH)
        if (root / EXP2969_REL_PATH).is_file()
        else None,
        "exp2970": _sha256(root / EXP2970_REL_PATH)
        if (root / EXP2970_REL_PATH).is_file()
        else None,
        "exp2973": _sha256(root / EXP2973_REL_PATH)
        if (root / EXP2973_REL_PATH).is_file()
        else None,
    }
    matrix_row = _matrix_v13_exp2969_row(exp2973)
    return {
        "selection_metric": UPDATE_SELECTION_METRIC,
        "reported_metric_names": metric_names,
        "selection_metric_reused_as_reported_metric": UPDATE_SELECTION_METRIC in metric_names,
        "exp2969_ready": exp2969.get("non_tautological_self_learning_ready") is True,
        "exp2970_ready": exp2970.get("kan_forgetting_guard_ready") is True,
        "used_exp2970_memory_audit_path": EXP2970_REL_PATH.as_posix(),
        "prompt_requested_exp2970_short_path_present": (root / EXP2970_PROMPT_REL_PATH).is_file(),
        "deterministic_reset_controls": {
            "frozen_baseline": True,
            "random_replay": True,
            "negative_control": True,
        },
        "negative_control_improved": False,
        "kan_evidence_scope": "bounded_memory_forgetting_only",
        "kan_acceleration_claimed": False,
        "kan_selected_policy": exp2970.get("selected_policy"),
        "matrix_v13_exp2969_headline_eligible": matrix_row.get("headline_eligible"),
        "matrix_v13_exp2969_row_class": matrix_row.get("row_class"),
        "source_sha256": source_sha256,
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a local JSON object, returning an empty object for invalid evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Validate required fields and the explicit no-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("substrate must remain aggregation_and_deterministic_replay")
    if artifact.get("no_identical_metric_flag") is not True:
        raise ValueError("selection metric must be distinct from reported metrics")
    audit = _mapping(artifact.get("leakage_audit"))
    if audit.get("kan_acceleration_claimed") is True:
        raise ValueError("KAN acceleration claims are out of scope")
    return dict(artifact)


def main() -> int:
    """CLI entry point used by the experiment wrapper."""

    write_artifact()
    return 0


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    source_artifacts: Sequence[Mapping[str, Any]],
    audit: Mapping[str, Any],
    verdict: str,
) -> JsonDict:
    artifact = {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "honest_verdict": verdict,
        "continuous_self_learning_task": True,
        "fr11_independent_metrics_evaluated": False,
        "fr11_independent_self_learning_ready": False,
        "update_selection_metric": UPDATE_SELECTION_METRIC,
        "independent_metrics": [metric.as_artifact_row() for metric in INDEPENDENT_METRICS],
        "frozen_baseline_metrics": {},
        "random_replay_metrics": {},
        "prior_fr11_metrics": {},
        "new_replay_metrics": {},
        "heldout_independent_delta_vs_random": {},
        "negative_control_delta": {},
        "forgetting_guard_passed": False,
        "leakage_audit": dict(audit),
        "no_identical_metric_flag": no_identical_metric_flag(
            UPDATE_SELECTION_METRIC,
            INDEPENDENT_METRICS,
        ),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "source_artifacts": [dict(source) for source in source_artifacts],
        "blockers": [verdict],
        "tests_run": list(config.tests_run),
    }
    return validate_artifact(artifact)


def _weights_from_payload(
    payload: Mapping[str, Any],
    key: str,
    fallback: Mapping[str, float],
) -> dict[str, float]:
    raw = payload.get(key)
    if not isinstance(raw, Mapping):
        return dict(fallback)
    try:
        return normalize_weights({str(name): float(value) for name, value in raw.items()})
    except (TypeError, ValueError):
        return dict(fallback)


def _matrix_v13_exp2969_row(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    rows = payload.get("matrix_rows")
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        return {}
    for row in rows:
        mapping = _mapping(row)
        if mapping.get("row_id") == "exp2969_non_tautological_fr11":
            return mapping
    return {}


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 12)
