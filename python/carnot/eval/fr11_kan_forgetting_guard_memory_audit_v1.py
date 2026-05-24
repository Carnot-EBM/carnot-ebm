"""Exp 2970 KAN constraint-memory forgetting guard audit.

This is a bounded software audit. It reuses the deterministic Exp 2933
constraint-memory fixture to compare four local memory-update policies and
checks whether KAN/per-knot memory can be treated as FR-11 evidence without
hiding old-domain forgetting. The hardware fields are copied from no-claim
complexity accounting artifacts only; this module does not run synthesis,
program a board, or claim analog acceleration.

Spec: REQ-LEARN-2970,
      SCENARIO-LEARN-2970,
      SCENARIO-LEARN-2970-BLOCKED.
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
OUTPUT_FILENAME = "experiment_2970_kan_forgetting_guard_memory_audit_v1.json"
RUN_DATE = "20260524"
RANDOM_SEED = 2933
INFERENCE_SUBSTRATE = "deterministic_wiring"
FORGETTING_THRESHOLD = 0.05

EXP2969_REL_PATH = Path("results/experiment_2969_fr11_non_tautological_utility_gate_v3.json")
EXP2933_REL_PATH = Path("results/experiment_2933_kan_cl_per_knot_self_learning_v1.json")
EXP2893_REL_PATH = Path("results/experiment_2893_kan_hardware_complexity_accounting_v1.json")

FILES_CHANGED = (
    "openspec/capabilities/self-learning/spec.md",
    "python/carnot/eval/fr11_kan_forgetting_guard_memory_audit_v1.py",
    "tests/python/test_experiment_2970_kan_forgetting_guard_memory_audit_v1.py",
    "results/experiment_2970_kan_forgetting_guard_memory_audit_v1.json",
)

REQUIRED_ARTIFACT_FIELDS = {
    "honest_verdict",
    "kan_forgetting_guard_ready",
    "source_artifacts",
    "policies_compared",
    "current_domain_utility",
    "old_domain_utility",
    "forgetting_delta_by_policy",
    "selected_policy",
    "high_dimensional_claim_allowed",
    "hardware_cost_fields",
    "no_synthesis_claim",
    "no_analog_claim",
    "files_changed",
    "inference_substrate",
    "duration_s",
}


@dataclass(frozen=True)
class SourceSpec:
    """One local source artifact cited by the Exp 2970 audit."""

    experiment_id: str
    path: Path
    role: str
    fields_imported: tuple[str, ...]
    required: bool


@dataclass(frozen=True)
class ExperimentConfig:
    """Runtime paths, deterministic clock, and file provenance for the artifact."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    files_changed: Sequence[str] = field(default_factory=lambda: FILES_CHANGED)

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME


SOURCE_SPECS = (
    SourceSpec(
        "exp2969",
        EXP2969_REL_PATH,
        "fr11_non_tautological_readiness_gate",
        ("non_tautological_self_learning_ready", "forgetting_guard_passed"),
        True,
    ),
    SourceSpec(
        "exp2933",
        EXP2933_REL_PATH,
        "kan_per_knot_fixture_reference",
        ("kan_cl_self_learning_ready", "dataset_manifest", "baselines"),
        False,
    ),
    SourceSpec(
        "exp2893",
        EXP2893_REL_PATH,
        "rm_bop_nabs_cost_reference",
        (
            "complexity_metrics.rm_count",
            "complexity_metrics.bop_count",
            "complexity_metrics.nabs_count",
        ),
        False,
    ),
)


def _default_kan_helpers() -> Any:
    from carnot.eval import fr11_kan_cl_per_knot_self_learning_v1

    return fr11_kan_cl_per_knot_self_learning_v1


KAN_HELPERS_IMPORTER: Callable[[], Any] = _default_kan_helpers


class _PersistentRBFPolicy:
    """Local RBF/per-knot memory that keeps all previously updated centers."""

    def __init__(self, helper: Any, centers: Any) -> None:
        self.memory = helper.RBFImportanceMemory(centers=centers)

    def fit_old(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.memory.update(rows, rule_by_id)

    def fit_current(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.memory.update(rows, rule_by_id)

    def predict(self, row: Any) -> float:
        return float(self.memory.predict_proba(row.features))


class _FrozenPolicy(_PersistentRBFPolicy):
    """Old-domain memory with no current-domain update."""

    def fit_current(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        return None


class _EagerOverwritePolicy:
    """Eager memory rewrite that exposes old-domain forgetting after update."""

    def __init__(self, helper: Any, centers: Any) -> None:
        self.helper = helper
        self.centers = centers
        self.memory = helper.RBFImportanceMemory(centers=centers)

    def fit_old(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.memory = self.helper.RBFImportanceMemory(centers=self.centers)
        self.memory.update(rows, rule_by_id)

    def fit_current(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.memory = self.helper.RBFImportanceMemory(centers=self.centers)
        self.memory.update(rows, rule_by_id)

    def predict(self, row: Any) -> float:
        return float(self.memory.predict_proba(row.features))


class _AdapterStylePolicy:
    """Frozen old-domain base memory plus a routed current-domain adapter."""

    def __init__(self, helper: Any, centers: Any) -> None:
        self.base = helper.RBFImportanceMemory(centers=centers)
        self.adapter = helper.RBFImportanceMemory(centers=centers)
        self.current_domains: set[str] = set()

    def fit_old(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.base.update(rows, rule_by_id)

    def fit_current(self, rows: Sequence[Any], rule_by_id: Mapping[str, Any]) -> None:
        self.current_domains.update(str(row.constraint_id) for row in rows)
        self.adapter.update(rows, rule_by_id)

    def predict(self, row: Any) -> float:
        if str(row.constraint_id) in self.current_domains:
            return float(self.adapter.predict_proba(row.features))
        return float(self.base.predict_proba(row.features))


def build_policy_comparison(helper: Any | None = None) -> list[JsonDict]:
    """REQ-LEARN-2970-2: compare four policies on identical fixture slices."""

    helper = helper or KAN_HELPERS_IMPORTER()
    stream = helper.build_constraint_stream(RANDOM_SEED)
    old_ids = tuple(rule.constraint_id for rule in stream.rules[:-1])
    source_current_id = stream.rules[-1].constraint_id
    current_rule = helper.ConstraintRule("logic_inverse_current", (10, 11), (8, 9))
    current_id = current_rule.constraint_id
    rule_by_id = dict(stream.rule_by_id)
    rule_by_id[current_id] = current_rule
    old_train = _rows_for_constraints(stream.train_by_constraint, old_ids)
    old_holdout = _rows_for_constraints(stream.holdout_by_constraint, old_ids)
    current_train = _remap_current_rows(
        helper,
        stream.train_by_constraint[source_current_id],
        current_rule,
    )
    current_holdout = _remap_current_rows(
        helper,
        stream.holdout_by_constraint[source_current_id],
        current_rule,
    )
    policy_specs = (
        ("frozen", _FrozenPolicy(helper, stream.centers), 0),
        ("eager_update", _EagerOverwritePolicy(helper, stream.centers), 0),
        ("per_knot_importance_update", _PersistentRBFPolicy(helper, stream.centers), 0),
        ("adapter_style_update", _AdapterStylePolicy(helper, stream.centers), 1),
    )
    rows: list[JsonDict] = []
    for policy_name, policy, extra_memory_tables in policy_specs:
        policy.fit_old(old_train, rule_by_id)
        old_before = _utility(old_holdout, policy.predict)
        policy.fit_current(current_train, rule_by_id)
        current_after = _utility(current_holdout, policy.predict)
        old_after = _utility(old_holdout, policy.predict)
        forgetting_delta = _round(max(0.0, old_before - old_after))
        rows.append(
            {
                "policy_name": policy_name,
                "current_domain": current_id,
                "old_domains": list(old_ids),
                "current_domain_utility": current_after,
                "old_domain_utility": old_after,
                "old_domain_utility_before_current_update": old_before,
                "forgetting_delta": forgetting_delta,
                "forgetting_threshold": FORGETTING_THRESHOLD,
                "forgetting_guard_passed": forgetting_delta <= FORGETTING_THRESHOLD,
                "high_dimensional_extrapolation": "out_of_scope",
                "extra_memory_tables": extra_memory_tables,
            }
        )
    return rows


def select_policy(policies: Sequence[Mapping[str, Any]]) -> str:
    """Select the strongest utility-improving policy that keeps old-domain guard slices."""

    frozen_current = next(
        (
            float(row["current_domain_utility"])
            for row in policies
            if row["policy_name"] == "frozen"
        ),
        0.0,
    )
    candidates = [
        row
        for row in policies
        if row.get("policy_name") != "frozen"
        and bool(row.get("forgetting_guard_passed"))
        and float(row.get("current_domain_utility", 0.0)) > frozen_current
    ]
    if not candidates:
        return "none"
    ranked = sorted(
        candidates,
        key=lambda row: (
            -float(row["current_domain_utility"]),
            float(row["forgetting_delta"]),
            int(row.get("extra_memory_tables", 0)),
            str(row["policy_name"]),
        ),
    )
    return str(ranked[0]["policy_name"])


def build_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build the Exp 2970 artifact, failing closed when preconditions are absent."""

    config = config or ExperimentConfig()
    started = config.start_time()
    source_artifacts = source_artifacts_for(config.repo_root)
    blocker = exp2969_blocker(config.repo_root)
    if blocker is not None:
        return _blocked_artifact(config, started, blocker, source_artifacts, [blocker])
    try:
        helper = KAN_HELPERS_IMPORTER()
    except ImportError as exc:
        blocker = "blocked_missing_kan_import"
        return _blocked_artifact(config, started, blocker, source_artifacts, [str(exc)])

    policies = build_policy_comparison(helper)
    selected_policy = select_policy(policies)
    artifact = {
        "run_date": RUN_DATE,
        "honest_verdict": (
            "complete: kan_forgetting_guard_ready"
            if selected_policy != "none"
            else "complete: kan_forgetting_guard_not_ready"
        ),
        "kan_forgetting_guard_ready": selected_policy != "none",
        "source_artifacts": source_artifacts,
        "policies_compared": policies,
        "current_domain_utility": _metric_by_policy(policies, "current_domain_utility"),
        "old_domain_utility": _metric_by_policy(policies, "old_domain_utility"),
        "forgetting_delta_by_policy": _metric_by_policy(policies, "forgetting_delta"),
        "selected_policy": selected_policy,
        "high_dimensional_claim_allowed": False,
        "hardware_cost_fields": hardware_cost_fields(config.repo_root),
        "no_synthesis_claim": True,
        "no_analog_claim": True,
        "files_changed": list(config.files_changed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "forgetting_threshold": FORGETTING_THRESHOLD,
        "out_of_scope": ["high_dimensional_extrapolation", "synthesis", "analog_acceleration"],
    }
    return validate_artifact(artifact)


def write_artifact(config: ExperimentConfig | None = None) -> JsonDict:
    """Build and persist `results/experiment_2970...json`."""

    config = config or ExperimentConfig()
    artifact = build_artifact(config)
    output_path = config.artifact_path()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> JsonDict:
    """Validate required fields and the explicit no-claim boundary."""

    missing = REQUIRED_ARTIFACT_FIELDS - set(artifact)
    if missing:
        raise ValueError(f"missing required fields: {sorted(missing)}")
    if artifact.get("high_dimensional_claim_allowed") is not False:
        raise ValueError("high-dimensional extrapolation must remain out of scope")
    if (
        artifact.get("no_synthesis_claim") is not True
        or artifact.get("no_analog_claim") is not True
    ):
        raise ValueError("claim boundary requires no synthesis and no analog claim")
    return dict(artifact)


def source_artifacts_for(root: Path) -> list[JsonDict]:
    """Return stable source-artifact citations with presence and SHA evidence."""

    citations = []
    for spec in SOURCE_SPECS:
        absolute_path = root / spec.path
        present = absolute_path.exists()
        citations.append(
            {
                "experiment_id": spec.experiment_id,
                "path": spec.path.as_posix(),
                "role": spec.role,
                "fields_imported": list(spec.fields_imported),
                "required": spec.required,
                "present": present,
                "sha256": _sha256(absolute_path) if present else None,
            }
        )
    return citations


def exp2969_blocker(root: Path) -> str | None:
    """Return the fail-closed precondition blocker, or None when Exp 2969 is ready."""

    payload = read_json_object(root / EXP2969_REL_PATH)
    if not payload:
        return "blocked_missing_exp2969_ready_artifact"
    if payload.get("non_tautological_self_learning_ready") is not True:
        return "blocked_exp2969_not_ready"
    return None


def hardware_cost_fields(root: Path) -> JsonDict:
    """REQ-LEARN-2970-5: derive RM/BOP/NABS fields without hardware claims."""

    payload = read_json_object(root / EXP2893_REL_PATH)
    metrics = payload.get("complexity_metrics")
    required = ("rm_count", "bop_count", "nabs_count")
    derivable = isinstance(metrics, Mapping) and all(name in metrics for name in required)
    return {
        "derivable": derivable,
        "source_artifact": EXP2893_REL_PATH.as_posix(),
        "rm_count": int(metrics["rm_count"]) if derivable else None,
        "bop_count": int(metrics["bop_count"]) if derivable else None,
        "nabs_count": int(metrics["nabs_count"]) if derivable else None,
        "memory_table_entries": _optional_int(metrics, "memory_table_entries")
        if derivable
        else None,
        "pwa_regions": _optional_int(metrics, "pwa_regions") if derivable else None,
        "milp_constraints": _optional_int(metrics, "milp_constraints") if derivable else None,
        "field_scope": "platform_independent_accounting_only",
        "no_synthesis_claim": True,
        "no_analog_claim": True,
    }


def read_json_object(path: Path) -> JsonDict:
    """Read a local JSON object, returning an empty object for invalid evidence."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def _blocked_artifact(
    config: ExperimentConfig,
    started: float,
    honest_verdict: str,
    source_artifacts: Sequence[Mapping[str, Any]],
    blockers: Sequence[str],
) -> JsonDict:
    artifact = {
        "run_date": RUN_DATE,
        "honest_verdict": honest_verdict,
        "kan_forgetting_guard_ready": False,
        "source_artifacts": [dict(source) for source in source_artifacts],
        "policies_compared": [],
        "current_domain_utility": {},
        "old_domain_utility": {},
        "forgetting_delta_by_policy": {},
        "selected_policy": "none",
        "high_dimensional_claim_allowed": False,
        "hardware_cost_fields": hardware_cost_fields(config.repo_root),
        "no_synthesis_claim": True,
        "no_analog_claim": True,
        "files_changed": list(config.files_changed),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": _elapsed(config, started),
        "blockers": list(blockers),
    }
    return validate_artifact(artifact)


def _rows_for_constraints(
    rows_by_constraint: Mapping[str, Sequence[Any]], ids: Sequence[str]
) -> tuple[Any, ...]:
    rows: list[Any] = []
    for constraint_id in ids:
        rows.extend(rows_by_constraint[constraint_id])
    return tuple(rows)


def _remap_current_rows(helper: Any, rows: Sequence[Any], rule: Any) -> tuple[Any, ...]:
    return tuple(
        helper.ConstraintExample(
            example_id=f"{rule.constraint_id}:{row.split}:{index}",
            constraint_id=rule.constraint_id,
            split=row.split,
            features=row.features,
            label=1 - int(row.label),
        )
        for index, row in enumerate(rows)
    )


def _utility(rows: Sequence[Any], predict: Callable[[Any], float]) -> float:
    correct = [(float(predict(row)) >= 0.5) == bool(row.label) for row in rows]
    return _round(sum(float(value) for value in correct) / len(correct))


def _metric_by_policy(policies: Sequence[Mapping[str, Any]], key: str) -> dict[str, float]:
    return {str(row["policy_name"]): float(row[key]) for row in policies}


def _optional_int(metrics: object, key: str) -> int | None:
    if isinstance(metrics, Mapping) and key in metrics:
        return int(metrics[key])
    return None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _elapsed(config: ExperimentConfig, started: float) -> float:
    return _round(config.clock() - started)


def _round(value: float) -> float:
    return round(float(value), 12)


def main() -> int:  # pragma: no cover
    write_artifact()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
