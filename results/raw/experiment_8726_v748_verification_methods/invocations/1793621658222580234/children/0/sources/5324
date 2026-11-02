"""Cold-read current V681 science and retain failed custody (REQ-REPORT-7849)."""

from __future__ import annotations

import json
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file


NUMBERS = (7838, 7840, 7841, 7842, 7843, 7844, 7846, 7848)
AUTHORITY = "docs/research-notes/v681-authority-snapshots/active.yaml"
BASE = "results/raw/experiment_7849_v681_independent_audit"
MANIFEST = f"{BASE}/validation_command_manifest.json"
SCORES = {
    7838: "source_boundary_ready_score",
    7840: "set_heads_ready_score",
    7841: "decision_evidence_ready_score",
    7842: "qwen_evidence_ready_score",
    7843: "learning_evidence_ready_score",
    7844: "selective_evidence_ready_score",
    7846: "service_evidence_ready_score",
    7848: "length_control_ready_score",
}


def _operand(
    number: int, path: Path, field: str, expected: Any, observed: Any, op: str = "=="
) -> dict[str, Any]:
    """Record the exact failed field and source bytes, including absent files."""
    return {
        "upstream_id": f"Exp{number}",
        "path": str(path),
        "hash": sha256_file(path) if path.is_file() else None,
        "artifact_field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
    }


def producer_failures(
    number: int, task_id: str, path: Path, data: dict[str, Any]
) -> list[dict[str, Any]]:
    """Reject a slug where the numeric ID belongs and every false gate."""
    failures = []
    for field, expected in (
        ("experiment_id", number),
        ("task_id", task_id),
        ("milestone", "2026.09.681"),
        ("flagged_adversarial", False),
        (SCORES[number], 1),
    ):
        if data.get(field) != expected:
            failures.append(_operand(number, path, field, expected, data.get(field)))
    if data.get("run_date") not in ("20260928", "20260929"):
        failures.append(
            _operand(number, path, "run_date", ["20260928", "20260929"], data.get("run_date"), "in")
        )
    allowed = ("positive", "circular_positive", "null")
    if data.get("verdict_class") not in allowed:
        failures.append(
            _operand(number, path, "verdict_class", list(allowed), data.get("verdict_class"), "in")
        )
    if data.get("required_validation_failures"):
        failures.append(
            _operand(
                number,
                path,
                "required_validation_failures",
                [],
                data["required_validation_failures"],
            )
        )
    return failures


def inspect_sources(root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Inspect declared producers, never substituting administrative receipts."""
    authority = yaml.safe_load((root / AUTHORITY).read_text())
    tasks = {task["id"].split("-", 1)[0]: task for task in authority["tasks"]}
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for number in NUMBERS:
        task = tasks[f"exp{number}"]
        path = root / task["deliverable"]
        receipts = sorted((root / "results").glob(f"experiment_{number}_*.json"))
        receipts = [item for item in receipts if item != path]
        state = "missing"
        data: dict[str, Any] = {}
        if path.is_file():
            try:
                loaded = json.loads(path.read_bytes())
                data = loaded if isinstance(loaded, dict) else {}
                state = str(data.get("verdict_class") or "disqualified")
            except (UnicodeError, ValueError):
                state = "disqualified"
        elif receipts:
            state = "conductor_only"
        source = {
            "upstream_id": f"Exp{number}",
            "task_id": task["id"],
            "path": str(path),
            "sha256": sha256_file(path) if path.is_file() else None,
            "date": data.get("run_date"),
            "role": "science_producer",
            "state": state,
            "eligibility": False,
            "conductor_receipts": [
                {"path": str(item), "sha256": sha256_file(item), "role": "explanation_only"}
                for item in receipts
            ],
        }
        if not data:
            failures.append(
                _operand(number, path, "science_producer", "qualified current artifact", state)
            )
        else:
            failures.extend(producer_failures(number, task["id"], path, data))
            if not any(f["upstream_id"] == f"Exp{number}" for f in failures):
                source["state"] = "eligible"
                source["eligibility"] = True
            for gate in data.get("gate_check_summary", []):
                if isinstance(gate, dict) and {"path", "expected", "observed"} <= gate.keys():
                    failures.append({**gate, "upstream_id": f"Exp{number}"})
        for receipt in receipts:
            try:
                gate_data = json.loads(receipt.read_bytes())
            except (UnicodeError, ValueError):
                gate_data = {}
            for gate in gate_data.get("gates_evaluated", []):
                if gate.get("passed") is False:
                    upstream_path = Path(gate["artifact_path"])
                    failures.append(
                        _operand(
                            number,
                            upstream_path,
                            gate["artifact_field"],
                            gate.get("expected"),
                            gate.get("actual"),
                            gate.get("op", "=="),
                        )
                    )
        sources.append(source)
    return sources, failures


def check_primitive_rows(rows: list[dict[str, Any]], *, intended: int) -> list[str]:
    """Fail leakage, chronology, lost rows and counterfeit independent N."""
    failures: list[str] = []
    families = [str(row.get("family_id")) for row in rows]
    if len(rows) < intended:
        failures.append("lost_censoring_rows")
    if len(families) != len(set(families)):
        failures.append("seed_pseudoreplication")
    admissions = [
        row.get("admission_family_id") for row in rows if row.get("role") == "online_admission"
    ]
    if len(admissions) != len(set(admissions)) or any(
        row.get("role") == "online_admission"
        and row.get("admission_family_id") != row.get("family_id")
        for row in rows
    ):
        failures.append("admission_family")
    for row in rows:
        feature_names = " ".join(str(key).lower() for key in row.get("features", {}))
        if any(
            token in feature_names
            for token in ("label", "gold", "annotation", "source", "confidence", "target")
        ):
            failures.append("private_feature")
        if (
            row.get("feedback_step") is not None
            and row.get("prediction_step", 0) >= row["feedback_step"]
        ):
            failures.append("future_feedback")
        if row.get("status") == "dropped" and row.get("feedback_replayed"):
            failures.append("dropped_feedback_replayed")
        if row.get("shuffle_label_step", 0) > row.get("prediction_step", 0):
            failures.append("future_shuffle_label")
        if row.get("label") not in (0, 1, None):
            failures.append("unknown_label_loss")
    return sorted(set(failures))


def _action(risk: float) -> str:
    """Use frozen V681 typed costs and send every tied minimum to escalation."""
    costs = {"accept": 5 * risk, "reject": 1 - risk, "escalate": 0.25}
    best = min(costs.values())
    if costs["escalate"] == best:
        return "escalate"
    return min(("accept", "reject"), key=costs.__getitem__)


def _cost(action: str, label: int) -> float:
    """Charge actual decisions against held-out labels, not expected risk."""
    return {"accept": float(5 * label), "reject": float(1 - label), "escalate": 0.25}[action]


def reduce_length(root: Path) -> dict[str, Any]:
    """Join the available length predictions to raw labels as an ineligible fact."""
    base = root / "results/raw/experiment_7848_v681_length_shortcut/current"
    prediction_path = base / "evaluation_predictions.json"
    label_path = (
        root / "results/raw/experiment_7727_v673_development_corpus/evaluation_evaluator.jsonl"
    )
    if not prediction_path.is_file() or not label_path.is_file():
        return {
            "eligible": False,
            "independent_n": 0,
            "means": {},
            "rows": [],
            "status": "missing_raw",
        }
    predictions = json.loads(prediction_path.read_bytes())
    labels = [json.loads(line) for line in label_path.read_text().splitlines() if line]
    by_label = {row["family_id"]: row["label"] for row in labels}
    raw = predictions["rows"]
    families = [row["family_id"] for row in raw]
    if len(families) != len(set(families)) or set(families) != set(by_label):
        raise ValueError("length_family_join")
    rows = []
    for row in raw:
        label = by_label[row["family_id"]]
        if label not in (0, 1):
            raise ValueError("length_label_invalid")
        arms = {}
        for arm, risk in (("length", row["length_risk"]), ("prevalence", row["prevalence_risk"])):
            if not isinstance(risk, (float, int)) or not 0 <= risk <= 1:
                raise ValueError("length_probability_invalid")
            chosen = _action(float(risk))
            arms[arm] = {
                "risk": risk,
                "action": chosen,
                "brier": (risk - label) ** 2,
                "cost": _cost(chosen, label),
            }
        arms["always_escalate"] = {"risk": None, "action": "escalate", "brier": None, "cost": 0.25}
        rows.append(
            {
                "family_id": row["family_id"],
                "role": "evaluation",
                "seed": 7848,
                "status": "completed",
                "arm": "paired_intact_family",
                "label": label,
                "answer_bytes": row["answer_bytes"],
                "source_bytes": row["source_bytes"],
                "stratum": row["stratum"],
                "arms": arms,
                "prediction_path": str(prediction_path),
                "evaluator_path": str(label_path),
            }
        )
    means = {
        arm: {
            "brier": sum(row["arms"][arm]["brier"] for row in rows) / len(rows)
            if arm != "always_escalate"
            else None,
            "cost": sum(row["arms"][arm]["cost"] for row in rows) / len(rows),
            "count": len(rows),
        }
        for arm in ("length", "prevalence", "always_escalate")
    }
    return {
        "eligible": False,
        "reason": "Exp7848 required validation failed",
        "independent_n": len(families),
        "means": means,
        "rows": rows,
        "prediction_sha256": sha256_file(prediction_path),
        "label_sha256": sha256_file(label_path),
    }


def _negative_mutations() -> list[dict[str, Any]]:
    """Exercise private attacks without altering the observed evidence."""
    clean = {
        "family_id": "f",
        "role": "online_admission",
        "seed": 68101,
        "features": {"length": 1},
        "label": 1,
        "probability": 0.5,
        "prediction_step": 1,
        "feedback_step": 2,
        "shuffle_label_step": 0,
        "admission_family_id": "f",
        "status": "completed",
        "feedback_replayed": False,
    }
    mutations = {
        "label_leakage": ([{**clean, "features": {"gold_label": 1}}], 1, "private_feature"),
        "future_feedback": ([{**clean, "feedback_step": 0}], 1, "future_feedback"),
        "lost_censoring": ([clean], 2, "lost_censoring_rows"),
        "seed_pseudoreplication": ([clean, {**clean, "seed": 68102}], 2, "seed_pseudoreplication"),
        "dropped_replay": (
            [{**clean, "status": "dropped", "feedback_replayed": True}],
            1,
            "dropped_feedback_replayed",
        ),
    }
    return [
        {
            "name": name,
            "observed": check_primitive_rows(rows, intended=count),
            "rejected": expected in check_primitive_rows(rows, intended=count),
        }
        for name, (rows, count, expected) in mutations.items()
    ]


def build_candidate(root: Path, output_root: Path, date: str) -> dict[str, Any]:
    """Build a terminal blocked result from current bytes before validation."""
    started = time.monotonic()
    sources, failures = inspect_sources(root)
    length = reduce_length(root)
    source_hashes = [
        {
            "path": str(root / AUTHORITY),
            "sha256": sha256_file(root / AUTHORITY),
            "role": "immutable_v681_authority",
            "date": "20260928",
            "eligibility": True,
        },
        *sources,
        *[
            {**receipt, "date": None, "eligibility": False}
            for source in sources
            for receipt in source["conductor_receipts"]
        ],
    ]
    if length["independent_n"]:
        for path, digest, role in (
            (
                length["rows"][0]["prediction_path"],
                length["prediction_sha256"],
                "ineligible_primitive_predictions",
            ),
            (
                length["rows"][0]["evaluator_path"],
                length["label_sha256"],
                "exposed_evaluator_labels",
            ),
        ):
            source_hashes.append(
                {
                    "path": path,
                    "sha256": digest,
                    "role": role,
                    "date": "20260929",
                    "eligibility": False,
                }
            )
    historical = json.loads(
        (root / "results/experiment_7835_v680_independent_evidence_audit.json").read_bytes()
    )
    old_failures = [
        {
            "name": row["name"],
            "exit_code": row["exit_code"],
            "log_path": row["log_path"],
            "log_sha256": row["log_sha256"],
            "artifact_path": str(
                root / "results/experiment_7835_v680_independent_evidence_audit.json"
            ),
        }
        for row in historical["validation_receipts"]
        if row.get("classification") == "required"
        and (row.get("exit_code") != 0 or row.get("timed_out", False))
    ]
    rows = []
    for source in sources:
        diagnostic = length if source["upstream_id"] == "Exp7848" else None
        rows.append(
            {
                "upstream_id": source["upstream_id"],
                "path": source["path"],
                "sha256": source["sha256"],
                "arm": source["task_id"],
                "family": None,
                "seed": None,
                "status": source["state"],
                "intended": 64 if source["upstream_id"] == "Exp7848" else None,
                "eligible": int(source["eligibility"]),
                "started": int(source["sha256"] is not None),
                "completed": 0,
                "censored": 0,
                "excluded": int(not source["eligibility"]),
                "independent_n": 0,
                "raw_diagnostic": diagnostic,
            }
        )
    manifest_path = root / MANIFEST
    inputs = {
        "sources": source_hashes,
        "manifest": sha256_file(manifest_path),
        "seed": 7849,
        "module": sha256_file(Path(__file__)),
        "date": date,
    }
    elapsed = time.monotonic() - started
    result: dict[str, Any] = {
        "experiment_id": 7849,
        "task_id": "exp7849-independent-audit",
        "milestone": "2026.09.681",
        "run_date": date,
        "honest_verdict": "complete_blocked_required_v681_science",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "gate_check_summary": failures,
        "rows": rows,
        "branch_dispositions": [
            {
                "upstream_id": row["upstream_id"],
                "state": row["state"],
                "path": row["path"],
                "eligibility": row["eligibility"],
            }
            for row in sources
        ],
        "recomputed_metrics": {
            "length_control_ineligible": length["means"],
            "qualified_science": None,
        },
        "mutation_results": _negative_mutations(),
        "sample_size_budget": {
            "intended": 8,
            "eligible": 0,
            "started": sum(bool(x["sha256"]) for x in sources),
            "completed": 0,
            "censored": 0,
            "excluded": 8,
            "independent_n": 0,
        },
        "acceptance_gate_results": {
            "validity": False,
            "readiness": 0,
            "probability_quality": None,
            "decision_benefit": None,
            "retention": None,
            "efficiency": None,
        },
        "duration_s": elapsed,
        "phase_spans": [
            {
                "phase": "preconditions_and_raw_join",
                "duration_s": elapsed,
                "completed_units": len(sources),
            }
        ],
        "random_seed": 7849,
        "reproducibility_checksum": canonical_hash(inputs),
        "source_artifact_hashes": source_hashes,
        "preconditions_checked": failures,
        "validation_receipts": [],
        "validation_command_manifest_path": str(manifest_path),
        "observed_child_commands": [],
        "repository_health": {"status": "pending", "historical_required_failures": old_failures},
        "verifier_is_oracle": False,
        "claim_scope": {
            "fixtures": "circular_positive_only",
            "natural_annotations": "exposed_development_only",
            "fresh_generalization_eligible": False,
        },
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "aggregation",
        "planned_inference_substrate_class": "aggregation",
        "actual_inference_substrate_class": "aggregation",
        "MODEL_SPECS": [],
        "model_specs": [],
        "model_invocation_counts": {
            "model_loads_attempted": 0,
            "generation_calls_attempted": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "model_file_hashes": [],
        },
        "independent_evidence_ready_score": 0,
        "import_call_closure": [
            "carnot.reporting.v681_independent_audit",
            "carnot.reporting.current_work_receipt",
            "carnot.reporting.experiment_7303_validation_scope",
        ],
        "output_root": str(output_root),
    }
    result["field_principles"] = {
        key: "Exact V681 provenance and claim scope; absent science cannot open a benefit gate."
        for key in result
    }
    return result


def cold_replay(path: Path, root: Path) -> list[str]:
    """Re-read identities, source bytes, branch rows and closed child logs."""
    try:
        result = json.loads(path.read_bytes())
    except (OSError, ValueError):
        return ["candidate_unreadable"]
    errors = []
    for key, expected in (
        ("experiment_id", 7849),
        ("task_id", "exp7849-independent-audit"),
        ("milestone", "2026.09.681"),
        ("run_date", "20260929"),
    ):
        if result.get(key) != expected:
            errors.append(key)
    for source in result.get("source_artifact_hashes", []):
        source_path = Path(source["path"])
        observed = sha256_file(source_path) if source_path.is_file() else None
        if source.get("sha256") != observed:
            errors.append("source_bytes_changed")
    current, _failures = inspect_sources(root)
    if [(row["upstream_id"], row["state"], row["sha256"]) for row in current] != [
        (row.get("upstream_id"), row.get("status"), row.get("sha256"))
        for row in result.get("rows", [])
    ]:
        errors.append("rows_changed")
    length = reduce_length(root)
    if result.get("recomputed_metrics", {}).get("length_control_ineligible") != length["means"]:
        errors.append("metric_changed")
    for receipt in result.get("validation_receipts", []):
        log_path = Path(receipt["log_path"])
        if not log_path.is_file() or sha256_file(log_path) != receipt.get("log_sha256"):
            errors.append("validation_log_changed")
    return sorted(set(errors))
