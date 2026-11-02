"""Exp 2966 deterministic skill-labeled exact logic frontier materializer.

Spec: REQ-BENCH-2966, SCENARIO-BENCH-2966.

This module deliberately does not call a language model.  It creates a compact
set of natural-language logic items whose reference formalizations are already
written in SMT-LIB and then asks Z3 to execute those references.  The point is
to give later live LLM tasks a small frontier with unambiguous skill labels,
expected statuses, and solver-backed answers before asking a model to emit Z3.
"""

from __future__ import annotations

import hashlib
import json
import textwrap
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:  # pragma: no cover - exercised indirectly by passing z3_module=None.
    import z3 as _z3
except Exception:  # pragma: no cover - dependency absence is environment-specific.
    _z3 = None


JsonDict = dict[str, Any]
RUN_DATE = "20260524"
RANDOM_SEED = 2966
REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_FILENAME = "experiment_2966_logic_frontier_materializer_v1.json"
MANIFEST_FILENAME = "logic_frontier_2966_manifest.json"
INFERENCE_SUBSTRATE = "deterministic_wiring"
SKILL_LABELS: tuple[str, ...] = (
    "symbolization",
    "quantifier handling",
    "countermodel construction",
    "satisfiability",
    "validity",
    "answer extraction",
)
MODEL_SPECS: tuple[JsonDict, ...] = (
    {"name": "Qwen3.6-35B-A3B", "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF", "gpu": 0},
    {"name": "Gemma4-31B-it", "hf_id": "unsloth/gemma-4-31B-it-GGUF", "gpu": 0},
    {"name": "Gemma4-26B-A4B-it", "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF", "gpu": 0},
)
MANDATED_MODEL_IDS = tuple(str(spec["hf_id"]) for spec in MODEL_SPECS)
REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "honest_verdict",
    "preconditions_checked",
    "z3_import_ok",
    "logic_frontier_materialized",
    "n_items",
    "skill_labels",
    "reference_formalizations_executed",
    "reference_z3_execution_rate",
    "reference_solver_accuracy",
    "manifest_path",
    "manifest_sha256",
    "model_specs_for_downstream_live_use",
    "inference_substrate",
    "duration_s",
)


@dataclass(frozen=True)
class ExperimentConfig:
    """Paths and clocks for the deterministic Exp 2966 materializer."""

    repo_root: Path = REPO_ROOT
    output_path: Path | None = None
    manifest_path: Path | None = None
    started_at: float | None = None
    clock: Callable[[], float] = time.perf_counter

    def start_time(self) -> float:
        return self.clock() if self.started_at is None else self.started_at

    def resolved_output_path(self) -> Path:
        return self.output_path or self.repo_root / "results" / OUTPUT_FILENAME

    def resolved_manifest_path(self) -> Path:
        return self.manifest_path or self.repo_root / "data" / "research" / MANIFEST_FILENAME


@dataclass(frozen=True)
class LogicFrontierItem:
    """One natural-language logic item with its exact reference Z3 check."""

    item_id: str
    prompt: str
    expected_label: str
    check_kind: str
    expected_solver_status: str
    skill_labels: tuple[str, ...]
    reference_smt2: str
    expected_answer_values: Mapping[str, str] = field(default_factory=dict)

    def to_manifest_record(self) -> JsonDict:
        unknown = sorted(set(self.skill_labels) - set(SKILL_LABELS))
        if unknown:
            raise ValueError(f"unknown skill label(s): {unknown}")
        if self.expected_solver_status not in {"sat", "unsat"}:
            raise ValueError("expected_solver_status must be sat or unsat")
        return {
            "item_id": self.item_id,
            "prompt": self.prompt,
            "expected_label": self.expected_label,
            "check_kind": self.check_kind,
            "expected_solver_status": self.expected_solver_status,
            "skill_labels": list(self.skill_labels),
            "reference_z3": {
                "format": "smt2",
                "assertions": self.reference_smt2,
            },
            "expected_answer_values": dict(self.expected_answer_values),
        }


def build_logic_frontier_items() -> list[LogicFrontierItem]:
    """Return the fixed 24-item exact logic frontier used by Exp 2966."""

    return [
        _item(
            "lf-2966-001",
            "All cats are mammals. Milo is a cat. Therefore Milo is a mammal.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "quantifier handling", "validity"),
            """
            (declare-sort Entity 0)
            (declare-const milo Entity)
            (declare-fun Cat (Entity) Bool)
            (declare-fun Mammal (Entity) Bool)
            (assert (forall ((x Entity)) (=> (Cat x) (Mammal x))))
            (assert (Cat milo))
            (assert (not (Mammal milo)))
            """,
        ),
        _item(
            "lf-2966-002",
            "Every pilot is licensed. Ada is licensed. Ada need not be a pilot.",
            "not_entailed",
            "validity",
            "sat",
            (
                "symbolization",
                "quantifier handling",
                "countermodel construction",
                "validity",
            ),
            """
            (declare-sort Entity 0)
            (declare-const ada Entity)
            (declare-fun Pilot (Entity) Bool)
            (declare-fun Licensed (Entity) Bool)
            (assert (forall ((x Entity)) (=> (Pilot x) (Licensed x))))
            (assert (Licensed ada))
            (assert (not (Pilot ada)))
            """,
        ),
        _item(
            "lf-2966-003",
            "No robot is organic. Rho is a robot. Rho cannot also be organic.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "quantifier handling", "satisfiability"),
            """
            (declare-sort Entity 0)
            (declare-const rho Entity)
            (declare-fun Robot (Entity) Bool)
            (declare-fun Organic (Entity) Bool)
            (assert (forall ((x Entity)) (not (and (Robot x) (Organic x)))))
            (assert (Robot rho))
            (assert (Organic rho))
            """,
        ),
        _item(
            "lf-2966-004",
            "Some student is an athlete, but Nia being a student does not force Nia to be one.",
            "satisfiable_countermodel",
            "satisfiability",
            "sat",
            (
                "symbolization",
                "quantifier handling",
                "countermodel construction",
                "satisfiability",
            ),
            """
            (declare-sort Entity 0)
            (declare-const nia Entity)
            (declare-fun Student (Entity) Bool)
            (declare-fun Athlete (Entity) Bool)
            (assert (Student nia))
            (assert (exists ((x Entity)) (and (Student x) (Athlete x))))
            (assert (not (Athlete nia)))
            """,
        ),
        _item(
            "lf-2966-005",
            "Archivists are careful; careful people are trusted; Ivo is an archivist.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "quantifier handling", "validity"),
            """
            (declare-sort Entity 0)
            (declare-const ivo Entity)
            (declare-fun Archivist (Entity) Bool)
            (declare-fun Careful (Entity) Bool)
            (declare-fun Trusted (Entity) Bool)
            (assert (forall ((x Entity)) (=> (Archivist x) (Careful x))))
            (assert (forall ((x Entity)) (=> (Careful x) (Trusted x))))
            (assert (Archivist ivo))
            (assert (not (Trusted ivo)))
            """,
        ),
        _item(
            "lf-2966-006",
            "If some device is encrypted and all encrypted things are audited, something is audited.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "quantifier handling", "validity"),
            """
            (declare-sort Entity 0)
            (declare-fun Device (Entity) Bool)
            (declare-fun Encrypted (Entity) Bool)
            (declare-fun Audited (Entity) Bool)
            (assert (exists ((x Entity)) (and (Device x) (Encrypted x))))
            (assert (forall ((x Entity)) (=> (Encrypted x) (Audited x))))
            (assert (forall ((x Entity)) (not (Audited x))))
            """,
        ),
        _item(
            "lf-2966-007",
            "Access requires a badge; access without a badge is inconsistent.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "satisfiability"),
            """
            (declare-const access Bool)
            (declare-const badge Bool)
            (assert (=> access badge))
            (assert access)
            (assert (not badge))
            """,
        ),
        _item(
            "lf-2966-008",
            "A token is red or blue. It is not red. It must be blue.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "validity"),
            """
            (declare-const red Bool)
            (declare-const blue Bool)
            (assert (or red blue))
            (assert (not red))
            (assert (not blue))
            """,
        ),
        _item(
            "lf-2966-009",
            "Exactly one of switch A or switch B may be on; one-on states are satisfiable.",
            "satisfiable",
            "satisfiability",
            "sat",
            ("symbolization", "satisfiability"),
            """
            (declare-const a_on Bool)
            (declare-const b_on Bool)
            (assert (or a_on b_on))
            (assert (not (and a_on b_on)))
            """,
        ),
        _item(
            "lf-2966-010",
            "A flag cannot be both true and false.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "satisfiability"),
            """
            (declare-const flag Bool)
            (assert flag)
            (assert (not flag))
            """,
        ),
        _item(
            "lf-2966-011",
            "If dry then safe. Not dry does not prove not safe.",
            "not_entailed",
            "validity",
            "sat",
            ("symbolization", "countermodel construction", "validity"),
            """
            (declare-const dry Bool)
            (declare-const safe Bool)
            (assert (=> dry safe))
            (assert (not dry))
            (assert (not safe))
            """,
        ),
        _item(
            "lf-2966-012",
            "A scheduled task cannot be both queued and running.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "satisfiability"),
            """
            (declare-const queued Bool)
            (declare-const running Bool)
            (assert (not (and queued running)))
            (assert queued)
            (assert running)
            """,
        ),
        _item(
            "lf-2966-013",
            "Three plus four gives the answer.",
            "answer=7",
            "answer_extraction",
            "sat",
            ("symbolization", "satisfiability", "answer extraction"),
            """
            (declare-const left Int)
            (declare-const right Int)
            (declare-const answer Int)
            (assert (= left 3))
            (assert (= right 4))
            (assert (= answer (+ left right)))
            """,
            {"answer": "7"},
        ),
        _item(
            "lf-2966-014",
            "Three boxes with four items each produce twelve items.",
            "answer=12",
            "answer_extraction",
            "sat",
            ("symbolization", "satisfiability", "answer extraction"),
            """
            (declare-const boxes Int)
            (declare-const per_box Int)
            (declare-const answer Int)
            (assert (= boxes 3))
            (assert (= per_box 4))
            (assert (= answer (* boxes per_box)))
            """,
            {"answer": "12"},
        ),
        _item(
            "lf-2966-015",
            "Ten units minus four used units leaves six.",
            "answer=6",
            "answer_extraction",
            "sat",
            ("symbolization", "satisfiability", "answer extraction"),
            """
            (declare-const start Int)
            (declare-const used Int)
            (declare-const answer Int)
            (assert (= start 10))
            (assert (= used 4))
            (assert (= answer (- start used)))
            """,
            {"answer": "6"},
        ),
        _item(
            "lf-2966-016",
            "If five approvals are needed and five are present, the numeric decision is one.",
            "answer=1",
            "answer_extraction",
            "sat",
            ("symbolization", "satisfiability", "answer extraction"),
            """
            (declare-const needed Int)
            (declare-const present Int)
            (declare-const answer Int)
            (assert (= needed 5))
            (assert (= present 5))
            (assert (= answer (ite (>= present needed) 1 0)))
            """,
            {"answer": "1"},
        ),
        _item(
            "lf-2966-017",
            "A false arithmetic claim that three plus four equals eight is impossible.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "satisfiability", "answer extraction"),
            """
            (declare-const answer Int)
            (assert (= answer (+ 3 4)))
            (assert (= answer 8))
            """,
        ),
        _item(
            "lf-2966-018",
            "All admins are employees. Sam being an employee does not force Sam to be admin.",
            "not_entailed",
            "validity",
            "sat",
            (
                "symbolization",
                "quantifier handling",
                "countermodel construction",
                "validity",
            ),
            """
            (declare-sort Entity 0)
            (declare-const sam Entity)
            (declare-fun Admin (Entity) Bool)
            (declare-fun Employee (Entity) Bool)
            (assert (forall ((x Entity)) (=> (Admin x) (Employee x))))
            (assert (Employee sam))
            (assert (not (Admin sam)))
            """,
        ),
        _item(
            "lf-2966-019",
            "Some manager approves a request; all approvers are trained; no trained person is impossible.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "quantifier handling", "satisfiability"),
            """
            (declare-sort Entity 0)
            (declare-fun Manager (Entity) Bool)
            (declare-fun Approves (Entity) Bool)
            (declare-fun Trained (Entity) Bool)
            (assert (exists ((x Entity)) (and (Manager x) (Approves x))))
            (assert (forall ((x Entity)) (=> (Approves x) (Trained x))))
            (assert (forall ((x Entity)) (not (Trained x))))
            """,
        ),
        _item(
            "lf-2966-020",
            "Squares are rectangles and rectangles are polygons; a square is a polygon.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "quantifier handling", "validity"),
            """
            (declare-sort Entity 0)
            (declare-const shape Entity)
            (declare-fun Square (Entity) Bool)
            (declare-fun Rectangle (Entity) Bool)
            (declare-fun Polygon (Entity) Bool)
            (assert (forall ((x Entity)) (=> (Square x) (Rectangle x))))
            (assert (forall ((x Entity)) (=> (Rectangle x) (Polygon x))))
            (assert (Square shape))
            (assert (not (Polygon shape)))
            """,
        ),
        _item(
            "lf-2966-021",
            "No guest is staff. A guest marked as staff is inconsistent.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "quantifier handling", "satisfiability"),
            """
            (declare-sort Entity 0)
            (declare-const guest1 Entity)
            (declare-fun Guest (Entity) Bool)
            (declare-fun Staff (Entity) Bool)
            (assert (forall ((x Entity)) (not (and (Guest x) (Staff x)))))
            (assert (Guest guest1))
            (assert (Staff guest1))
            """,
        ),
        _item(
            "lf-2966-022",
            "Subscribed users are notified. Ben is subscribed, so Ben is notified.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "validity"),
            """
            (declare-const subscribed Bool)
            (declare-const notified Bool)
            (assert (=> subscribed notified))
            (assert subscribed)
            (assert (not notified))
            """,
        ),
        _item(
            "lf-2966-023",
            "Exactly one delivery window is selected; selecting both windows is inconsistent.",
            "unsatisfiable",
            "satisfiability",
            "unsat",
            ("symbolization", "satisfiability"),
            """
            (declare-const morning Bool)
            (declare-const evening Bool)
            (assert (or morning evening))
            (assert (not (and morning evening)))
            (assert morning)
            (assert evening)
            """,
        ),
        _item(
            "lf-2966-024",
            "Some poet exists and all poets are writers, so at least one writer exists.",
            "entailed",
            "validity",
            "unsat",
            ("symbolization", "quantifier handling", "validity"),
            """
            (declare-sort Entity 0)
            (declare-fun Poet (Entity) Bool)
            (declare-fun Writer (Entity) Bool)
            (assert (exists ((x Entity)) (Poet x)))
            (assert (forall ((x Entity)) (=> (Poet x) (Writer x))))
            (assert (forall ((x Entity)) (not (Writer x))))
            """,
        ),
    ]


def execute_reference_formalizations(
    items: Sequence[LogicFrontierItem],
    *,
    z3_module: Any = _z3,
) -> list[JsonDict]:
    """Execute every reference formalization and return per-item Z3 evidence."""

    return [execute_reference_formalization(item, z3_module=z3_module) for item in items]


def execute_reference_formalization(
    item: LogicFrontierItem,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Execute one SMT-LIB reference and compare status plus answer fields."""

    base = {
        "item_id": item.item_id,
        "expected_solver_status": item.expected_solver_status,
        "expected_label": item.expected_label,
        "expected_answer_values": dict(item.expected_answer_values),
    }
    if z3_module is None:
        return base | _failed_execution("z3_unavailable")

    try:
        solver = z3_module.Solver()
        solver.add(z3_module.parse_smt2_string(item.reference_smt2))
        actual_status = str(solver.check())
    except Exception as exc:
        return base | _failed_execution(f"{type(exc).__name__}: {exc}")

    status_matches = actual_status == item.expected_solver_status
    actual_answer_values: dict[str, str] = {}
    answer_matches = True
    if actual_status == "sat" and item.expected_answer_values:
        model = solver.model()
        for symbol_name, expected_value in item.expected_answer_values.items():
            actual_value = str(model.eval(z3_module.Int(symbol_name), model_completion=True))
            actual_answer_values[symbol_name] = actual_value
            answer_matches = answer_matches and actual_value == expected_value
    elif item.expected_answer_values:
        answer_matches = False

    return base | {
        "z3_executed": True,
        "z3_error": None,
        "actual_solver_status": actual_status,
        "solver_status_matches_expected": status_matches,
        "actual_answer_values": actual_answer_values,
        "answer_extraction_matches_expected": answer_matches,
        "reference_passed": bool(status_matches and answer_matches),
    }


def aggregate_execution_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize execution and solver-reference accuracy over item results."""

    if not rows:
        return {
            "reference_formalizations_executed": 0,
            "reference_z3_execution_rate": 0.0,
            "reference_solver_accuracy": 0.0,
        }
    total = len(rows)
    executed = sum(bool(row.get("z3_executed")) for row in rows)
    passed = sum(bool(row.get("reference_passed")) for row in rows)
    return {
        "reference_formalizations_executed": executed,
        "reference_z3_execution_rate": _rate(executed, total),
        "reference_solver_accuracy": _rate(passed, total),
    }


def skill_label_counts(items: Sequence[LogicFrontierItem]) -> dict[str, int]:
    """Count how often each required LogicSkills-style skill appears."""

    counts = Counter({label: 0 for label in SKILL_LABELS})
    for item in items:
        for label in item.skill_labels:
            counts[label] += 1
    return dict(counts)


def build_manifest(
    items: Sequence[LogicFrontierItem], results: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Build the downstream manifest without including a self-referential hash."""

    result_by_id = {str(row["item_id"]): dict(row) for row in results}
    manifest_items: list[JsonDict] = []
    for item in items:
        record = item.to_manifest_record()
        record["reference_execution"] = result_by_id[item.item_id]
        manifest_items.append(record)
    return {
        "schema": "carnot.logic_frontier.v1",
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "n_items": len(manifest_items),
        "skill_labels": list(SKILL_LABELS),
        "skill_label_counts": skill_label_counts(items),
        "model_specs_for_downstream_live_use": list(MODEL_SPECS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "items": manifest_items,
    }


def check_preconditions(z3_module: Any = _z3) -> list[JsonDict]:
    """Record dependency and provenance checks before materializing the frontier."""

    z3_ok = z3_module is not None
    return [
        {
            "name": "z3_import",
            "ok": z3_ok,
            "detail": (
                f"z3-solver {z3_module.get_version_string()}" if z3_ok else "missing z3-solver"
            ),
        },
        {
            "name": "live_llm_invocation",
            "ok": True,
            "detail": "not invoked; Exp 2966 uses deterministic reference wiring only",
        },
        {
            "name": "downstream_model_specs_recorded",
            "ok": True,
            "detail": ",".join(MANDATED_MODEL_IDS),
        },
    ]


def run_experiment(
    config: ExperimentConfig | None = None,
    *,
    z3_module: Any = _z3,
) -> JsonDict:
    """Materialize the exact logic frontier and write the terminal artifact."""

    cfg = config or ExperimentConfig()
    started_at = cfg.start_time()
    duration = lambda: cfg.clock() - started_at
    preconditions = check_preconditions(z3_module)
    if not preconditions[0]["ok"]:
        artifact = _blocked_artifact(preconditions, duration())
        write_json(cfg.resolved_output_path(), artifact)
        return artifact

    items = build_logic_frontier_items()
    results = execute_reference_formalizations(items, z3_module=z3_module)
    metrics = aggregate_execution_metrics(results)
    manifest = build_manifest(items, results)
    manifest_text = stable_json(manifest)
    manifest_path = cfg.resolved_manifest_path()
    write_text(manifest_path, manifest_text)
    artifact = {
        "honest_verdict": "complete: exact skill-labeled logic frontier materialized",
        "preconditions_checked": preconditions,
        "z3_import_ok": True,
        "logic_frontier_materialized": True,
        "n_items": len(items),
        "skill_labels": list(SKILL_LABELS),
        "reference_formalizations_executed": metrics["reference_formalizations_executed"],
        "reference_z3_execution_rate": metrics["reference_z3_execution_rate"],
        "reference_solver_accuracy": metrics["reference_solver_accuracy"],
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_text(manifest_text),
        "model_specs_for_downstream_live_use": list(MODEL_SPECS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration(),
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "skill_label_counts": skill_label_counts(items),
        "per_item_results": results,
    }
    validate_artifact(artifact)
    write_json(cfg.resolved_output_path(), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 2966 terminal artifact enough for conductor use."""

    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be deterministic_wiring")
    recorded_ids = {
        str(spec.get("hf_id")) for spec in artifact.get("model_specs_for_downstream_live_use", [])
    }
    if set(MANDATED_MODEL_IDS) - recorded_ids:
        raise ValueError("missing mandated downstream model specs")
    if artifact.get("logic_frontier_materialized"):
        n_items = int(artifact["n_items"])
        if not 20 <= n_items <= 30:
            raise ValueError("materialized artifact must contain 20-30 items")
        if artifact["reference_formalizations_executed"] != n_items:
            raise ValueError("materialized artifact requires all references to execute")
        if artifact["reference_z3_execution_rate"] != 1.0:
            raise ValueError("materialized artifact requires full Z3 execution")
        if artifact["reference_solver_accuracy"] != 1.0:
            raise ValueError("materialized artifact requires exact reference solver accuracy")


def write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write stable JSON with parent directory creation."""

    return write_text(path, stable_json(payload))


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def stable_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _item(
    item_id: str,
    prompt: str,
    expected_label: str,
    check_kind: str,
    expected_solver_status: str,
    skill_labels: tuple[str, ...],
    reference_smt2: str,
    expected_answer_values: Mapping[str, str] | None = None,
) -> LogicFrontierItem:
    return LogicFrontierItem(
        item_id=item_id,
        prompt=prompt,
        expected_label=expected_label,
        check_kind=check_kind,
        expected_solver_status=expected_solver_status,
        skill_labels=skill_labels,
        reference_smt2=textwrap.dedent(reference_smt2).strip() + "\n",
        expected_answer_values=dict(expected_answer_values or {}),
    )


def _failed_execution(error: str) -> JsonDict:
    return {
        "z3_executed": False,
        "z3_error": error,
        "actual_solver_status": None,
        "solver_status_matches_expected": False,
        "actual_answer_values": {},
        "answer_extraction_matches_expected": False,
        "reference_passed": False,
    }


def _blocked_artifact(preconditions: list[JsonDict], duration_s: float) -> JsonDict:
    artifact = {
        "honest_verdict": "blocked_dependency: z3 import failed",
        "preconditions_checked": preconditions,
        "z3_import_ok": False,
        "logic_frontier_materialized": False,
        "n_items": 0,
        "skill_labels": list(SKILL_LABELS),
        "reference_formalizations_executed": 0,
        "reference_z3_execution_rate": 0.0,
        "reference_solver_accuracy": 0.0,
        "manifest_path": "",
        "manifest_sha256": "",
        "model_specs_for_downstream_live_use": list(MODEL_SPECS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": duration_s,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
    }
    validate_artifact(artifact)
    return artifact


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def main() -> None:  # pragma: no cover - covered through run_experiment tests.
    run_experiment()


if __name__ == "__main__":  # pragma: no cover
    main()
