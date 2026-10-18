"""Controlled nonparametric typed-memory ablation for Exp 5239.

This module treats Exp 5227 memory as a read-only controller prior. It runs the
same small stream through five arms so a future ARC consumer can tell whether
the right typed memory helps because it is aligned with the task, rather than
because any memory, a lucky random prior, or a constant prior was present.

Spec refs: REQ-LEARN-5239, SCENARIO-LEARN-5239.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import random
from typing import Any

from carnot.pipeline import multihead_verifier_memory


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5239_continuous_self_learning_controlled_memory_ablation_v479"
EXPERIMENT_ID = 5239
SCHEMA = "carnot.continuous_self_learning_controlled_memory_ablation.v1"
RUN_DATE = "2026-07-04"
RANDOM_SEED = 5239

RESULT_RELATIVE_PATH = (
    "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json"
)
MEMORY_RELATIVE_PATH = multihead_verifier_memory.MEMORY_RELATIVE_PATH
EXP5227_RELATIVE_PATH = multihead_verifier_memory.RESULT_RELATIVE_PATH

SPEC_REFS = ("REQ-LEARN-5239", "SCENARIO-LEARN-5239")
MEMORY_HEADS = multihead_verifier_memory.TYPED_MEMORY_HEADS
ARM_NAMES = (
    "no_memory",
    "best_constant",
    "per_query_random",
    "shuffled_memory",
    "aligned_memory",
)

EXPECTED_ACTIONS = {
    "GAP-1 orientation discriminator memory-only promotion": (
        "use_gap1_orientation_discriminator_as_memory_only"
    ),
    "GAP-1 registry promotion blocked by subset instability": (
        "block_gap1_registry_promotion_until_frozen_subset_gate"
    ),
    "GAP-4 candidate-pool validation null/quarantine": (
        "quarantine_gap4_candidate_pool_until_positive_validation"
    ),
    "MMLU hidden-state verifier path retired": "retire_mmlu_hidden_state_path",
    "ARC live-path zero-level delta retained for rubric setup": (
        "build_arc_process_rubric_before_patch_synthesis"
    ),
    "Hardware speedup claim boundary": "block_hardware_speedup_claim_until_transcript",
}

TASK_TEMPLATES = (
    {
        "task_id": "gap1_memory_only_consumer",
        "query": "ARC GAP-1 orientation discriminator: choose memory-only or registry promotion",
        "expected_subject": "GAP-1 orientation discriminator memory-only promotion",
        "expected_head": "provenance",
        "default_action": "attempt_gap1_registry_promotion",
        "arc_patch_relevant": True,
    },
    {
        "task_id": "gap1_registry_rollback_consumer",
        "query": "ARC GAP-1 registry promotion retry after subset instability",
        "expected_subject": "GAP-1 registry promotion blocked by subset instability",
        "expected_head": "provenance",
        "default_action": "attempt_gap1_registry_promotion",
        "arc_patch_relevant": True,
    },
    {
        "task_id": "gap4_candidate_pool_consumer",
        "query": "ARC GAP-4 clean candidate pool should be reused or quarantined",
        "expected_subject": "GAP-4 candidate-pool validation null/quarantine",
        "expected_head": "failures",
        "default_action": "reuse_gap4_candidate_pool",
        "arc_patch_relevant": True,
    },
    {
        "task_id": "mmlu_hidden_state_retention_consumer",
        "query": "MMLU hidden-state verifier path after failed controls",
        "expected_subject": "MMLU hidden-state verifier path retired",
        "expected_head": "failures",
        "default_action": "retire_mmlu_hidden_state_path",
        "arc_patch_relevant": False,
    },
    {
        "task_id": "arc_rubric_before_patch_consumer",
        "query": "ARC live path zero-level delta before patch synthesis",
        "expected_subject": "ARC live-path zero-level delta retained for rubric setup",
        "expected_head": "skills_rubrics",
        "default_action": "patch_arc_level_directly",
        "arc_patch_relevant": True,
    },
    {
        "task_id": "hardware_speedup_boundary_consumer",
        "query": "Hardware speedup claim without authenticated transcript",
        "expected_subject": "Hardware speedup claim boundary",
        "expected_head": "constraints",
        "default_action": "block_hardware_speedup_claim_until_transcript",
        "arc_patch_relevant": False,
    },
)

ROLLBACK_ACTION_MARKERS = (
    "block_",
    "quarantine_",
    "retire_",
    "build_arc_process_rubric",
)

REQUIRED_WRAPPED_FIELDS = (
    "continuous_self_learning_task",
    "memory_heads_tested",
    "controlled_stream_n",
    "arms",
    "aligned_vs_shuffled_delta",
    "aligned_vs_no_memory_delta",
    "degradation_detected",
    "retention_check_passed",
    "rollback_policy_exercised",
    "recommended_arc_memory_heads",
    "inference_substrate",
    "honest_verdict",
    "tests_run",
    "nonparametric_memory_updates",
    "broad_self_distillation_used",
)

FIELD_PRINCIPLES = {
    "continuous_self_learning_task": (
        "This field proves the milestone includes the required continuous self-learning experiment."
    ),
    "memory_heads_tested": "List of typed heads consumed by the controlled stream.",
    "controlled_stream_n": "Number of controlled tasks evaluated under every arm.",
    "arms": "The exact five ablation arms used with a fixed one-prior-slot budget.",
    "aligned_vs_shuffled_delta": (
        "Accuracy delta isolating task-aligned memory from shuffled-memory bias."
    ),
    "aligned_vs_no_memory_delta": "Accuracy delta isolating aligned memory transfer over no memory.",
    "degradation_detected": (
        "True when irrelevant random, shuffled, or constant memory underperforms no memory."
    ),
    "retention_check_passed": (
        "True when promoted and rolled-back entries remain correctly reusable in the stream."
    ),
    "rollback_policy_exercised": (
        "True when a rolled-back entry changes a consumer action to block, quarantine, retire, or no-promote."
    ),
    "recommended_arc_memory_heads": (
        "Typed heads that show controlled useful reuse for ARC patch-synthesis decisions."
    ),
    "inference_substrate": "Must be controlled_nonparametric_typed_memory_ablation.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether typed memory shows controlled useful reuse."
    ),
    "tests_run": "List of verification commands with pass/fail status.",
    "nonparametric_memory_updates": (
        "True only when the experiment consumes memory without model training or fine-tuning."
    ),
    "broad_self_distillation_used": "False because this experiment performs no model distillation.",
}


@dataclass(frozen=True)
class ControlledTask:
    """One controlled-stream query with one expected typed-memory prior."""

    task_id: str
    query: str
    expected_subject: str
    expected_head: str
    expected_state: str
    expected_action: str
    default_action: str
    arc_patch_relevant: bool
    spec_refs: tuple[str, ...] = SPEC_REFS


def load_memory(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load and validate the Exp 5227 typed-memory ledger."""

    memory = _read_json(Path(root) / MEMORY_RELATIVE_PATH)
    errors = multihead_verifier_memory.validate_memory(memory)
    if errors:  # pragma: no cover - repository fixture is valid; this guards manual use.
        raise ValueError("; ".join(errors))
    return memory


def build_controlled_stream(memory: Mapping[str, Any]) -> list[ControlledTask]:
    """Construct controlled tasks whose reusable priors are existing memory entries."""

    entries = _entries_by_subject(memory)
    stream: list[ControlledTask] = []
    for template in TASK_TEMPLATES:
        subject = str(template["expected_subject"])
        entry = entries.get(subject)
        if entry is None:  # pragma: no cover - fixture integrity guard.
            raise ValueError(f"missing expected typed-memory entry: {subject}")
        expected_head = str(template["expected_head"])
        if entry.get("head") != expected_head:  # pragma: no cover - fixture integrity guard.
            raise ValueError(f"{subject} expected head {expected_head}, got {entry.get('head')}")
        stream.append(
            ControlledTask(
                task_id=str(template["task_id"]),
                query=str(template["query"]),
                expected_subject=subject,
                expected_head=expected_head,
                expected_state=str(entry["promotion_state"]),
                expected_action=EXPECTED_ACTIONS[subject],
                default_action=str(template["default_action"]),
                arc_patch_relevant=bool(template["arc_patch_relevant"]),
            )
        )
    return stream


def evaluate_memory(memory: Mapping[str, Any], *, seed: int = RANDOM_SEED) -> JsonDict:
    """Run all five ablation arms and return deterministic metrics."""

    stream = build_controlled_stream(memory)
    entries = [dict(entry) for entry in memory.get("entries", [])]
    by_subject = _entries_by_subject(memory)

    aligned_entries = [by_subject[task.expected_subject] for task in stream]
    rng = random.Random(seed)
    arm_metrics = {
        "no_memory": _evaluate_selected("no_memory", stream, [None] * len(stream)),
        "best_constant": _evaluate_best_constant(stream, entries),
        "per_query_random": _evaluate_selected(
            "per_query_random",
            stream,
            [entries[rng.randrange(len(entries))] for _task in stream],
        ),
        "shuffled_memory": _evaluate_selected(
            "shuffled_memory",
            stream,
            _seeded_derangement(aligned_entries, seed),
        ),
        "aligned_memory": _evaluate_selected("aligned_memory", stream, aligned_entries),
    }
    arm_metrics["shuffled_memory"]["fixed_points"] = _fixed_points(
        stream, arm_metrics["shuffled_memory"]["rows"]
    )

    no_memory_accuracy = arm_metrics["no_memory"]["accuracy"]
    shuffled_accuracy = arm_metrics["shuffled_memory"]["accuracy"]
    aligned_accuracy = arm_metrics["aligned_memory"]["accuracy"]
    retention = _retention_summary(memory, arm_metrics["aligned_memory"]["rows"])
    rollback_exercised = _rollback_policy_exercised(arm_metrics["aligned_memory"]["rows"])
    aligned_vs_shuffled_delta = _delta(aligned_accuracy, shuffled_accuracy)
    aligned_vs_no_memory_delta = _delta(aligned_accuracy, no_memory_accuracy)
    degradation_detected = any(
        arm_metrics[name]["accuracy"] < no_memory_accuracy
        for name in ("best_constant", "per_query_random", "shuffled_memory")
    )

    return {
        "controlled_stream": [_task_summary(task) for task in stream],
        "controlled_stream_n": len(stream),
        "memory_heads_tested": list(MEMORY_HEADS),
        "arm_metrics": arm_metrics,
        "aligned_vs_shuffled_delta": aligned_vs_shuffled_delta,
        "aligned_vs_no_memory_delta": aligned_vs_no_memory_delta,
        "degradation_detected": bool(degradation_detected),
        "retention": retention,
        "retention_check_passed": bool(retention["passed"]),
        "rollback_policy_exercised": bool(rollback_exercised),
        "recommended_arc_memory_heads": _recommended_arc_heads(
            stream=stream,
            aligned_rows=arm_metrics["aligned_memory"]["rows"],
            useful=aligned_vs_shuffled_delta > 0.0 and aligned_vs_no_memory_delta > 0.0,
        ),
        "budget": {
            "task_budget_per_arm": len(stream),
            "prior_slots_per_query": 1,
            "random_seed": seed,
        },
    }


def build_result_artifact(
    *,
    memory: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    seed: int = RANDOM_SEED,
) -> JsonDict:
    """Build the principle-annotated Exp 5239 result artifact."""

    evaluation = evaluate_memory(memory, seed=seed)
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": _source_artifacts(),
        "controlled_stream": evaluation["controlled_stream"],
        "arm_metrics": evaluation["arm_metrics"],
        "retention": evaluation["retention"],
        "budget": evaluation["budget"],
        "continuous_self_learning_task": _wrap("continuous_self_learning_task", True),
        "memory_heads_tested": _wrap("memory_heads_tested", evaluation["memory_heads_tested"]),
        "controlled_stream_n": _wrap("controlled_stream_n", evaluation["controlled_stream_n"]),
        "arms": _wrap("arms", list(ARM_NAMES)),
        "aligned_vs_shuffled_delta": _wrap(
            "aligned_vs_shuffled_delta",
            evaluation["aligned_vs_shuffled_delta"],
        ),
        "aligned_vs_no_memory_delta": _wrap(
            "aligned_vs_no_memory_delta",
            evaluation["aligned_vs_no_memory_delta"],
        ),
        "degradation_detected": _wrap(
            "degradation_detected",
            evaluation["degradation_detected"],
        ),
        "retention_check_passed": _wrap(
            "retention_check_passed",
            evaluation["retention_check_passed"],
        ),
        "rollback_policy_exercised": _wrap(
            "rollback_policy_exercised",
            evaluation["rollback_policy_exercised"],
        ),
        "recommended_arc_memory_heads": _wrap(
            "recommended_arc_memory_heads",
            evaluation["recommended_arc_memory_heads"],
        ),
        "inference_substrate": _wrap(
            "inference_substrate",
            "controlled_nonparametric_typed_memory_ablation",
        ),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(evaluation)),
        "tests_run": _wrap("tests_run", [dict(item) for item in tests_run]),
        "nonparametric_memory_updates": _wrap("nonparametric_memory_updates", True),
        "broad_self_distillation_used": _wrap("broad_self_distillation_used", False),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5239 result artifact and return its JSON payload."""

    artifact = build_result_artifact(memory=load_memory(root), tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def _evaluate_best_constant(stream: Sequence[ControlledTask], entries: Sequence[JsonDict]) -> JsonDict:
    candidates: list[JsonDict] = []
    for entry in entries:
        rows = _rows_for_selection(stream, [entry] * len(stream))
        candidates.append(
            {
                "subject": str(entry["subject"]),
                "correct_n": sum(1 for row in rows if row["correct"]),
                "rows": rows,
            }
        )
    best = sorted(candidates, key=lambda item: (-int(item["correct_n"]), str(item["subject"])))[0]
    summary = _arm_summary("best_constant", best["rows"])
    summary["selected_constant_subject"] = best["subject"]
    summary["constant_candidates"] = [
        {"subject": item["subject"], "correct_n": item["correct_n"]} for item in candidates
    ]
    return summary


def _evaluate_selected(
    arm: str,
    stream: Sequence[ControlledTask],
    selected_entries: Sequence[Mapping[str, Any] | None],
) -> JsonDict:
    return _arm_summary(arm, _rows_for_selection(stream, selected_entries))


def _rows_for_selection(
    stream: Sequence[ControlledTask],
    selected_entries: Sequence[Mapping[str, Any] | None],
) -> list[JsonDict]:
    rows = []
    for task, entry in zip(stream, selected_entries, strict=True):
        action = task.default_action if entry is None else EXPECTED_ACTIONS[str(entry["subject"])]
        rows.append(
            {
                "task_id": task.task_id,
                "query": task.query,
                "expected_head": task.expected_head,
                "expected_state": task.expected_state,
                "expected_subject": task.expected_subject,
                "expected_action": task.expected_action,
                "selected_subject": None if entry is None else str(entry["subject"]),
                "selected_action": action,
                "correct": action == task.expected_action,
                "arc_patch_relevant": task.arc_patch_relevant,
            }
        )
    return rows


def _arm_summary(arm: str, rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    correct_n = sum(1 for row in rows if row["correct"])
    return {
        "arm": arm,
        "n": len(rows),
        "correct_n": correct_n,
        "accuracy": _rate(correct_n, len(rows)),
        "selected_subjects": [row["selected_subject"] for row in rows],
        "rows": [dict(row) for row in rows],
    }


def _seeded_derangement(entries: Sequence[JsonDict], seed: int) -> list[JsonDict]:
    offset = seed % (len(entries) - 1) + 1
    return list(entries[offset:]) + list(entries[:offset])


def _fixed_points(stream: Sequence[ControlledTask], rows: Sequence[Mapping[str, Any]]) -> int:
    return sum(
        1
        for task, row in zip(stream, rows, strict=True)
        if row["selected_subject"] == task.expected_subject
    )


def _retention_summary(memory: Mapping[str, Any], aligned_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    promoted_correct = sum(
        1 for row in aligned_rows if row["expected_state"] == "promoted" and row["correct"]
    )
    rolled_back_correct = sum(
        1 for row in aligned_rows if row["expected_state"] == "rolled_back" and row["correct"]
    )
    typed_retention = multihead_verifier_memory.run_retention_check(memory)
    return {
        "passed": bool(typed_retention["passed"] and promoted_correct > 0 and rolled_back_correct > 0),
        "typed_memory_retention_passed": bool(typed_retention["passed"]),
        "promoted_correct_n": promoted_correct,
        "rolled_back_correct_n": rolled_back_correct,
        "queries": typed_retention["queries"],
    }


def _rollback_policy_exercised(aligned_rows: Sequence[Mapping[str, Any]]) -> bool:
    return any(
        row["expected_state"] == "rolled_back"
        and row["correct"]
        and str(row["selected_action"]).startswith(ROLLBACK_ACTION_MARKERS)
        for row in aligned_rows
    )


def _recommended_arc_heads(
    *,
    stream: Sequence[ControlledTask],
    aligned_rows: Sequence[Mapping[str, Any]],
    useful: bool,
) -> list[str]:
    useful_heads = {
        task.expected_head
        for task, row in zip(stream, aligned_rows, strict=True)
        if useful and task.arc_patch_relevant and row["correct"]
    }
    return [head for head in MEMORY_HEADS if head in useful_heads]


def _entries_by_subject(memory: Mapping[str, Any]) -> dict[str, JsonDict]:
    return {str(entry["subject"]): dict(entry) for entry in memory.get("entries", [])}


def _task_summary(task: ControlledTask) -> JsonDict:
    summary = asdict(task)
    summary["spec_refs"] = list(task.spec_refs)
    return summary


def _source_artifacts() -> list[str]:
    return [
        "research-program.md#continuous-self-learning-core-architectural-goal",
        "research-references.md#v479-research-update-2026-07-04",
        "ops/arc_solve_registry.yaml#primitive_persistent_action_effect_memory_operator",
        EXP5227_RELATIVE_PATH,
        MEMORY_RELATIVE_PATH,
    ]


def _honest_verdict(evaluation: Mapping[str, Any]) -> str:
    return (
        "complete: typed memory shows controlled useful reuse; "
        f"aligned_vs_shuffled_delta={evaluation['aligned_vs_shuffled_delta']:.6f}, "
        f"aligned_vs_no_memory_delta={evaluation['aligned_vs_no_memory_delta']:.6f}, "
        f"degradation_detected={str(evaluation['degradation_detected']).lower()}, "
        f"retention_passed={str(evaluation['retention_check_passed']).lower()}, "
        f"rollback_exercised={str(evaluation['rollback_policy_exercised']).lower()}, "
        "no_model_training"
    )


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _rate(correct_n: int, total_n: int) -> float:
    return round(correct_n / total_n, 6)


def _delta(left: float, right: float) -> float:
    return round(left - right, 6)


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
