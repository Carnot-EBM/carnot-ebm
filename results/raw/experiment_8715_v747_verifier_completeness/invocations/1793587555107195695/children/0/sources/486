"""Exp 3044 SMT/SAT validator-tree exactness upgrade.

Spec refs: REQ-VERIFY-3044, SCENARIO-VERIFY-3044.

This module keeps the scope intentionally small: it runs deterministic
integer-constraint fixtures through Z3, writes row-level evidence, and leaves
non-exact or fallback regions visible. That row separation is what downstream
self-learning code needs; it can consume verified and correction-set feedback
without accidentally treating unresolved or fallback-only rows as exact truth.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:  # pragma: no cover - unit tests exercise dependency absence via injection.
    import z3 as _z3
except Exception:  # pragma: no cover
    _z3 = None


JsonDict = dict[str, Any]
RUN_DATE = "20260525"
SCHEMA = "carnot.smt_sat_validator_tree_exactness_upgrade.v1"
ARTIFACT = "experiment_3044_smt_sat_validator_tree_exactness_upgrade_v1"
ARTIFACT_FILENAME = f"{ARTIFACT}.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
EXACT_VALIDATOR_PATH = Path("python/carnot/eval/smt_sat_validator_tree_exactness_upgrade_v1.py")
OUTPUT_REL_PATH = Path("results") / ARTIFACT_FILENAME
EVIDENCE_REL_PATH = Path("results/smt_sat_validator_tree_exactness_3044/exact_validator_rows.jsonl")
TRANSCRIPT_REL_PATH = Path(
    "results/smt_sat_validator_tree_exactness_3044/exact_validator_transcript.json"
)
INFERENCE_SUBSTRATE = {
    "mode": "deterministic_z3_cpu_validator_tree",
    "live_llm_inference": False,
    "local_gguf_inference": False,
    "z3_solver_used": True,
    "hardware_acceleration": False,
}
SUCCESS_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
REQUIRED_ARTIFACT_FIELDS = (
    "validator_tree_exactness_ready",
    "exact_validator_path",
    "tests_or_checks_run",
    "verified_count",
    "unresolved_count",
    "fallback_only_count",
    "correction_sets",
    "spec_updates",
    "model_specs",
    "inference_substrate",
    "honest_verdict",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and clock hooks for a deterministic Exp 3044 run."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    evidence_path: Path | None = None
    transcript_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter
    tests_run: Sequence[str] = ()

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def artifact_path(self) -> Path:
        return self.output_path or self.repo_root / OUTPUT_REL_PATH

    def resolved_evidence_path(self) -> Path:
        return self.evidence_path or self.repo_root / EVIDENCE_REL_PATH

    def resolved_transcript_path(self) -> Path:
        return self.transcript_path or self.repo_root / TRANSCRIPT_REL_PATH


def build_validator_fixtures() -> list[JsonDict]:
    """Return tiny deterministic fixtures spanning every required row class."""

    relation = {
        "constraint_id": "sum_relation",
        "target": "total",
        "terms": {"a": 1, "b": 1},
        "constant": 0,
    }
    return [
        {
            "item_id": "sat-sum-ok",
            "row_kind": "exact",
            "candidate": {"a": 2, "b": 3, "total": 5},
            "mutable_fields": ["total"],
            "constraints": [relation],
        },
        {
            "item_id": "sat-sum-bad",
            "row_kind": "exact",
            "candidate": {"a": 2, "b": 3, "total": 6},
            "mutable_fields": ["total"],
            "constraints": [relation],
        },
        {
            "item_id": "semantic-boundary",
            "row_kind": "irrelevant",
            "reason": "non_authoritative_semantic_boundary",
        },
        {
            "item_id": "quantifier-text",
            "row_kind": "unresolved",
            "reason": "text_quantifier_not_encoded_in_tiny_smt_path",
        },
        {
            "item_id": "enumerator-fallback",
            "row_kind": "fallback_only",
            "fallback_path": "results/raw/experiment_3004/enumerator_fallback.json",
        },
    ]


def evaluate_fixtures(
    fixtures: Sequence[Mapping[str, Any]],
    *,
    z3_module: Any = _z3,
) -> list[JsonDict]:
    """Evaluate fixtures and return row-level exactness evidence."""

    return [evaluate_fixture(fixture, z3_module=z3_module) for fixture in fixtures]


def evaluate_fixture(fixture: Mapping[str, Any], *, z3_module: Any = _z3) -> JsonDict:
    """Classify one validator-tree fixture without using model self-judgment."""

    kind = str(fixture["row_kind"])
    base = {
        "row_id": str(fixture["item_id"]),
        "item_id": str(fixture["item_id"]),
        "row_kind": kind,
        "llm_judge_used": False,
        "live_llm_inference": False,
        "fallback_promoted_as_exact": False,
    }
    if kind == "irrelevant":
        return base | {
            "classification": "irrelevant",
            "solver_status": "not_applicable",
            "exact_checked": False,
            "bound_status": "clipped_irrelevant_to_exact_authority",
            "allowed_claim_wording": "Report as irrelevant to exact authority, not verified.",
        }
    if kind == "unresolved":
        return base | {
            "classification": "unresolved",
            "solver_status": "not_encoded",
            "exact_checked": False,
            "unresolved_reason": str(fixture["reason"]),
            "allowed_claim_wording": "Do not promote; exact SMT/SAT authority is absent.",
        }
    if kind == "fallback_only":
        return base | {
            "classification": "fallback_only",
            "solver_status": "not_run",
            "exact_checked": False,
            "fallback_path": str(fixture["fallback_path"]),
            "allowed_claim_wording": "Fallback-only row cannot be promoted as exact authority.",
        }
    if z3_module is None:
        return base | {
            "classification": "unresolved",
            "solver_status": "z3_unavailable",
            "exact_checked": False,
            "unresolved_reason": "z3_solver_unavailable",
            "allowed_claim_wording": "Do not promote; exact SMT/SAT solver is unavailable.",
        }
    return base | _evaluate_exact_fixture(fixture, z3_module)


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Run Exp 3044, persist row evidence, and write the terminal artifact."""

    active = config or ExperimentConfig()
    started = active.start_time()
    rows = evaluate_fixtures(build_validator_fixtures(), z3_module=z3_module)
    _write_jsonl(active.resolved_evidence_path(), rows)
    _write_json(
        active.resolved_transcript_path(),
        {
            "schema": f"{SCHEMA}.transcript",
            "row_count": len(rows),
            "solver": "z3" if z3_module is not None else "unavailable",
            "rows": rows,
        },
    )
    artifact = _build_artifact(
        active,
        rows,
        duration_s=round(active.clock() - started, 6),
        z3_solver_used=z3_module is not None,
    )
    validate_artifact(artifact)
    _write_json(active.artifact_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Raise when an Exp 3044 artifact violates the exactness contract."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("model_specs") != []:
        raise ValueError("model_specs must remain empty because no live LLM was used")
    substrate = artifact.get("inference_substrate")
    if not isinstance(substrate, Mapping) or substrate.get("live_llm_inference") is not False:
        raise ValueError("inference_substrate must disclose no live LLM inference")
    if artifact.get("validator_tree_exactness_ready") is not True:
        if not str(artifact.get("honest_verdict", "")).startswith("blocked_"):
            raise ValueError("honest_verdict must use a blocked_ prefix when not ready")
        return
    if int(artifact.get("verified_count") or 0) <= 0:
        raise ValueError("verified_count must be positive when ready")
    if not artifact.get("correction_sets"):
        raise ValueError("correction_sets must be non-empty when ready")
    if int(artifact.get("fallback_only_count") or 0) <= 0:
        raise ValueError("fallback_only_count must remain visible when ready")
    if int(artifact.get("unresolved_count") or 0) <= 0:
        raise ValueError("unresolved_count must remain visible when ready")
    if artifact.get("exact_validator_present") is not True:
        raise ValueError("exact_validator_path must exist when ready")
    if artifact.get("exact_evidence_present") is not True:
        raise ValueError("exact evidence file must exist when ready")
    if artifact.get("exact_evidence_path") != EVIDENCE_REL_PATH.as_posix():
        raise ValueError("exact evidence path must be the Exp 3044 JSONL path")
    if not str(artifact.get("honest_verdict", "")).startswith(SUCCESS_PREFIXES):
        raise ValueError("honest_verdict must start with a terminal success prefix")


def load_jsonl(path: Path) -> list[JsonDict]:
    """Load a JSONL evidence file written by this module."""

    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest for a file that should already exist."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def _evaluate_exact_fixture(fixture: Mapping[str, Any], z3_module: Any) -> JsonDict:
    variables = _z3_variables(fixture, z3_module)
    assignments = _candidate_assignments(fixture, variables, z3_module)
    constraints = _constraint_assertions(fixture, variables, z3_module)
    status = _solver_status(
        [row["assertion"] for row in assignments] + [row["assertion"] for row in constraints],
        z3_module,
    )
    if status == "sat":
        return {
            "classification": "verified",
            "solver_status": "sat",
            "exact_checked": True,
            "exact_authority": "z3_solver",
            "constraint_ids": [str(row["id"]) for row in constraints],
            "correction_set": None,
        }
    if status == "unknown":  # pragma: no cover - tiny integer fixtures are decidable.
        return {
            "classification": "unresolved",
            "solver_status": "unknown",
            "exact_checked": False,
            "unresolved_reason": "z3_returned_unknown",
        }
    correction = _minimal_correction_set(assignments, constraints, variables, z3_module)
    return {
        "classification": "correction_set",
        "solver_status": "unsat",
        "exact_checked": True,
        "exact_authority": "z3_solver",
        "constraint_ids": [str(row["id"]) for row in constraints],
        "correction_set": correction,
    }


def _z3_variables(fixture: Mapping[str, Any], z3_module: Any) -> dict[str, Any]:
    names = set(_mapping(fixture["candidate"]))
    for constraint in _mapping_list(fixture["constraints"]):
        names.add(str(constraint["target"]))
        names.update(str(name) for name in _mapping(constraint["terms"]))
    return {name: z3_module.Int(name) for name in sorted(names)}


def _candidate_assignments(
    fixture: Mapping[str, Any],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> list[JsonDict]:
    mutable = set(_string_list(fixture["mutable_fields"]))
    return [
        {
            "id": f"candidate.{field}",
            "field": field,
            "mutable": field in mutable,
            "assertion": variables[field] == z3_module.IntVal(int(value)),
        }
        for field, value in sorted(_mapping(fixture["candidate"]).items())
    ]


def _constraint_assertions(
    fixture: Mapping[str, Any],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for constraint in _mapping_list(fixture["constraints"]):
        expression = z3_module.IntVal(int(constraint.get("constant", 0)))
        for name, coefficient in sorted(_mapping(constraint["terms"]).items()):
            expression += int(coefficient) * variables[str(name)]
        rows.append(
            {
                "id": str(constraint["constraint_id"]),
                "assertion": variables[str(constraint["target"])] == expression,
            }
        )
    return rows


def _solver_status(assertions: Sequence[Any], z3_module: Any) -> str:
    solver = z3_module.Solver()
    solver.add(*assertions)
    status = solver.check()
    if status == z3_module.sat:
        return "sat"
    if status == z3_module.unsat:
        return "unsat"
    return "unknown"  # pragma: no cover - tiny integer fixtures are decidable.


def _minimal_correction_set(
    assignments: Sequence[Mapping[str, Any]],
    constraints: Sequence[Mapping[str, Any]],
    variables: Mapping[str, Any],
    z3_module: Any,
) -> JsonDict:
    mutable = [row for row in assignments if row["mutable"]]
    constraint_assertions = [row["assertion"] for row in constraints]
    for row in mutable:
        kept = [entry["assertion"] for entry in assignments if entry["id"] != row["id"]]
        solver = z3_module.Solver()
        solver.add(*(kept + constraint_assertions))
        if solver.check() == z3_module.sat:
            model = solver.model()
            field = str(row["field"])
            return {
                "candidate_fields": [field],
                "minimal_assignment_ids": [str(row["id"])],
                "suggested_assignments": {
                    field: model.eval(variables[field], model_completion=True).as_long()
                },
                "failing_constraint_ids": [str(entry["id"]) for entry in constraints],
            }
    return {  # pragma: no cover - fixtures keep one mutable repairable field.
        "candidate_fields": [],
        "minimal_assignment_ids": [],
        "suggested_assignments": {},
        "failing_constraint_ids": [str(entry["id"]) for entry in constraints],
    }


def _build_artifact(
    config: ExperimentConfig,
    rows: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    z3_solver_used: bool,
) -> JsonDict:
    evidence_path = config.resolved_evidence_path()
    transcript_path = config.resolved_transcript_path()
    counts = _counts(rows)
    correction_sets = [
        row["correction_set"] for row in rows if row.get("classification") == "correction_set"
    ]
    exact_validator_present = (REPO_ROOT / EXACT_VALIDATOR_PATH).is_file()
    evidence_present = evidence_path.is_file()
    substrate = dict(INFERENCE_SUBSTRATE) | {"z3_solver_used": z3_solver_used}
    ready = (
        exact_validator_present
        and evidence_present
        and z3_solver_used
        and counts["verified"] > 0
        and counts["unresolved"] > 0
        and counts["fallback_only"] > 0
        and bool(correction_sets)
        and substrate["live_llm_inference"] is False
    )
    return {
        "schema": SCHEMA,
        "artifact": ARTIFACT,
        "run_date": RUN_DATE,
        "validator_tree_exactness_ready": ready,
        "exact_validator_path": EXACT_VALIDATOR_PATH.as_posix(),
        "exact_validator_present": exact_validator_present,
        "exact_evidence_path": str(_relative_to(config.repo_root, evidence_path)),
        "exact_evidence_present": evidence_present,
        "exact_evidence_sha256": sha256_file(evidence_path),
        "transcript_path": str(_relative_to(config.repo_root, transcript_path)),
        "transcript_sha256": sha256_file(transcript_path),
        "tests_or_checks_run": list(config.tests_run),
        "verified_count": counts["verified"],
        "irrelevant_count": counts["irrelevant"],
        "unresolved_count": counts["unresolved"],
        "fallback_only_count": counts["fallback_only"],
        "correction_set_count": counts["correction_set"],
        "correction_sets": correction_sets,
        "row_counts": counts,
        "exact_rows": [dict(row) for row in rows],
        "spec_updates": [
            "openspec/capabilities/verification/spec.md#REQ-VERIFY-3044",
            "openspec/capabilities/verification/spec.md#SCENARIO-VERIFY-3044",
        ],
        "model_specs": [],
        "inference_substrate": substrate,
        "source_context": {
            "exp3030": "results/experiment_3030_validator_frontier_corrigendum_v2.json",
            "exp3033": "results/experiment_3033_fr11_nonforgetting_negative_control_stress_v1.json",
        },
        "duration_s": duration_s,
        "honest_verdict": _honest_verdict(ready, counts),
    }


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    return {
        name: sum(1 for row in rows if row.get("classification") == name)
        for name in ("verified", "irrelevant", "unresolved", "fallback_only", "correction_set")
    }


def _honest_verdict(ready: bool, counts: Mapping[str, int]) -> str:
    if ready:
        return (
            "complete: validator_tree_exactness_ready=true; "
            f"verified={counts['verified']}; unresolved={counts['unresolved']}; "
            f"fallback_only={counts['fallback_only']}; correction_sets={counts['correction_set']}"
        )
    return "blocked_exact_validator_evidence_unavailable"


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(dict(row), sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _relative_to(root: Path, path: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _mapping_list(value: Any) -> list[JsonDict]:
    return [dict(item) for item in value if isinstance(item, Mapping)]


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, (list, tuple)) else []
