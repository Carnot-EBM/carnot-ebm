"""Deterministic typed-memory retention and interference audit for Exp 5261.

This module audits verifier memory as controller state rather than as model
training. The fixtures deliberately avoid live LLM calls: they replay cached
memory-policy situations that exercise useful retention, irrelevant memory,
conflicts, stale entries, shuffled controls, promotion thresholds, and rollback
after harmful memory.

Spec refs: REQ-LEARN-5261, SCENARIO-LEARN-5261.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.pipeline import multihead_verifier_memory
from carnot.pipeline.verifier_memory import DEFAULT_PROMOTION_THRESHOLD, decide_promotion
from carnot.provenance_receipts import receipt_bytes, receipt_exists


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5261_typed_memory_interference_audit_v481"
EXPERIMENT_ID = 5261
SCHEMA = "carnot.typed_memory_interference_audit.v481"
RUN_DATE = "2026-07-05"
RANDOM_SEED = 5261
RESULT_RELATIVE_PATH = "results/experiment_5261_typed_memory_interference_audit_v481.json"
INFERENCE_SUBSTRATE = "cached_fixture_replay_no_llm"
SPEC_REFS = ("REQ-LEARN-5261", "SCENARIO-LEARN-5261")
TYPED_MEMORY_HEADS = multihead_verifier_memory.TYPED_MEMORY_HEADS
ROLLBACK_ACTION_PREFIXES = ("rollback_", "block_", "quarantine_", "retire_")

SOURCE_ARTIFACTS = (
    "results/verifier_memory_v477.json",
    "results/typed_multihead_verifier_memory_v478.json",
    "results/experiment_5239_continuous_self_learning_controlled_memory_ablation_v479.json",
    "results/experiment_5249_cross_model_typed_memory_transfer_v480.json",
    "research-references.md#v481-research-update-2026-07-05",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal verdict starts with complete: or blocked_ and states whether "
        "the memory policy is ready, blocked, harmful, or needs redesign."
    ),
    "inference_substrate": (
        "Declares cached fixture replay with no live LLM calls, preventing a "
        "false SOTA inference claim."
    ),
    "retention_rate": (
        "Useful aligned memories retained after deterministic distractor insertion."
    ),
    "interference_rate": (
        "Unrelated tasks receiving an action from irrelevant, stale, conflicting, "
        "or shuffled memory."
    ),
    "harmful_memory_rollback_passed": (
        "True only when harmful memory rolls back into a safe block/quarantine/"
        "retire action on the degradation fixture."
    ),
    "promotion_eviction_policy": (
        "Promotion, hold, rollback, stale eviction, and conflict eviction receipts."
    ),
    "fixture_checksums": (
        "Stable SHA-256 receipts for deterministic fixtures and cached source artifacts."
    ),
}

REQUIRED_WRAPPED_FIELDS = tuple(FIELD_PRINCIPLES)


@dataclass(frozen=True)
class AuditMemory:
    """One deterministic typed-memory fixture row.

    The row mirrors the existing typed verifier-memory schema: it has a fixed
    head, a subject, evidence, a promotion/rollback signal, and optional
    invalidation. Extra audit-only fields make the stability-plasticity cases
    explicit without changing the durable memory schema.
    """

    subject: str
    head: str
    task_slot: str
    fixture_kind: str
    action: str
    evidence: tuple[str, ...]
    guard_passed: bool
    heldout_delta: float | None
    stale: bool = False
    conflicts_with: str | None = None
    invalidated_by: str | None = None
    harmful: bool = False


@dataclass(frozen=True)
class AuditTask:
    """One cached replay task used to score memory retrieval behavior."""

    task_id: str
    task_slot: str
    task_kind: str
    expected_action: str
    default_action: str
    expected_subject: str | None = None
    requires_harmful_rollback: bool = False


@dataclass(frozen=True)
class AuditFixtureSet:
    """Complete deterministic fixture set for the Exp 5261 audit."""

    memories: tuple[AuditMemory, ...]
    tasks: tuple[AuditTask, ...]
    seed: int = RANDOM_SEED

    @property
    def memory_kinds(self) -> tuple[str, ...]:
        """Return the required fixture kinds, including the shuffled arm."""

        return tuple(sorted({memory.fixture_kind for memory in self.memories} | {"shuffled"}))


def typed_memory_schema_invariants() -> tuple[str, ...]:
    """Describe the current typed-memory invariants the audit preserves."""

    return (
        "fixed typed heads: constraints, provenance, failures, skills_rubrics",
        "deterministic memory identifiers derived from typed content",
        "evidence-gated promotion above the held-out delta threshold",
        "invalidation-gated rollback for null, harmful, or refuted memories",
        "idempotent duplicate collapse by stable memory id",
        "test-gold leakage rejection before durable memory write",
    )


def build_deterministic_fixtures() -> AuditFixtureSet:
    """Build deterministic aligned, distractor, conflict, stale, and shuffled fixtures."""

    memories = (
        AuditMemory(
            subject="GAP-1 orientation discriminator memory-only promotion",
            head="provenance",
            task_slot="gap1_orientation",
            fixture_kind="aligned",
            action="use_gap1_orientation_discriminator_as_memory_only",
            evidence=("results/verifier_memory_v477.json",),
            guard_passed=True,
            heldout_delta=0.041797,
        ),
        AuditMemory(
            subject="Hardware speedup claim boundary",
            head="constraints",
            task_slot="hardware_reporting",
            fixture_kind="aligned",
            action="block_speedup_claim_until_transcript",
            evidence=("results/experiment_5217_hardware_continuity_v477.json",),
            guard_passed=True,
            heldout_delta=0.05,
        ),
        AuditMemory(
            subject="MMLU hidden-state verifier path retired",
            head="failures",
            task_slot="mmlu_hidden_state",
            fixture_kind="aligned",
            action="retire_mmlu_hidden_state_path",
            evidence=(
                "results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json",
            ),
            guard_passed=True,
            heldout_delta=0.0,
            invalidated_by="results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json",
        ),
        AuditMemory(
            subject="ARC harmful direct patch rollback",
            head="skills_rubrics",
            task_slot="arc_patch",
            fixture_kind="aligned",
            action="block_arc_patch_until_positive_validation",
            evidence=(
                "results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json",
            ),
            guard_passed=True,
            heldout_delta=-0.20,
            invalidated_by="results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json",
            harmful=True,
        ),
        AuditMemory(
            subject="GAP-1 contradictory registry promotion stale copy",
            head="provenance",
            task_slot="gap1_orientation",
            fixture_kind="conflicting",
            action="promote_gap1_registry_now",
            evidence=("results/experiment_5222_gap1_gate_field_registry_promotion_v478.json",),
            guard_passed=True,
            heldout_delta=0.03,
            conflicts_with="GAP-1 orientation discriminator memory-only promotion",
        ),
        AuditMemory(
            subject="Hardware speedup outdated smoke-only shortcut",
            head="constraints",
            task_slot="hardware_reporting",
            fixture_kind="stale",
            action="accept_hardware_speedup_from_smoke_only",
            evidence=("results/experiment_5217_hardware_continuity_v477.json",),
            guard_passed=True,
            heldout_delta=None,
            stale=True,
            invalidated_by="authenticated transcript still missing",
        ),
        AuditMemory(
            subject="Citation-style irrelevant held memory",
            head="failures",
            task_slot="citation_support",
            fixture_kind="irrelevant",
            action="apply_citation_support_memory",
            evidence=("results/experiment_5252_halluhard_provenance_memory_microbench_v480.json",),
            guard_passed=True,
            heldout_delta=0.01,
        ),
    )
    tasks = (
        AuditTask(
            task_id="gap1_orientation_holdout",
            task_slot="gap1_orientation",
            task_kind="useful",
            expected_action="use_gap1_orientation_discriminator_as_memory_only",
            default_action="attempt_gap1_registry_promotion",
            expected_subject="GAP-1 orientation discriminator memory-only promotion",
        ),
        AuditTask(
            task_id="hardware_reporting_holdout",
            task_slot="hardware_reporting",
            task_kind="useful",
            expected_action="block_speedup_claim_until_transcript",
            default_action="accept_hardware_speedup_claim",
            expected_subject="Hardware speedup claim boundary",
        ),
        AuditTask(
            task_id="mmlu_hidden_state_holdout",
            task_slot="mmlu_hidden_state",
            task_kind="useful",
            expected_action="retire_mmlu_hidden_state_path",
            default_action="retry_mmlu_hidden_state_probe",
            expected_subject="MMLU hidden-state verifier path retired",
        ),
        AuditTask(
            task_id="arc_patch_harmful_holdout",
            task_slot="arc_patch",
            task_kind="useful",
            expected_action="block_arc_patch_until_positive_validation",
            default_action="patch_arc_level_directly",
            expected_subject="ARC harmful direct patch rollback",
            requires_harmful_rollback=True,
        ),
        AuditTask(
            task_id="range_constraint_unrelated",
            task_slot="numeric_range",
            task_kind="unrelated",
            expected_action="answer_without_memory",
            default_action="answer_without_memory",
        ),
        AuditTask(
            task_id="prompt_style_unrelated",
            task_slot="prompt_style",
            task_kind="unrelated",
            expected_action="answer_without_memory",
            default_action="answer_without_memory",
        ),
        AuditTask(
            task_id="proof_obligation_unrelated",
            task_slot="proof_obligation",
            task_kind="unrelated",
            expected_action="answer_without_memory",
            default_action="answer_without_memory",
        ),
    )
    return AuditFixtureSet(memories=memories, tasks=tasks)


def evaluate_promotion_eviction_policy(memories: Sequence[AuditMemory]) -> JsonDict:
    """Apply current promotion/rollback rules and audit-local eviction gates."""

    raw_decisions = [_decision_for_memory(memory) for memory in memories]
    promoted_subjects = {
        str(decision["subject"])
        for decision in raw_decisions
        if decision["effective_state"] == "promoted"
    }
    decisions = []
    for decision, memory in zip(raw_decisions, memories, strict=True):
        eviction_reason = _eviction_reason(memory, promoted_subjects)
        active = eviction_reason is None and (
            decision["effective_state"] == "promoted"
            or (
                decision["effective_state"] == "rolled_back"
                and str(memory.action).startswith(ROLLBACK_ACTION_PREFIXES)
            )
        )
        decisions.append(
            {
                **decision,
                "eviction_reason": eviction_reason,
                "active": bool(active),
            }
        )
    return {
        "promotion_threshold": DEFAULT_PROMOTION_THRESHOLD,
        "decisions": decisions,
        "promotion_summary": dict(Counter(str(item["effective_state"]) for item in decisions)),
        "eviction_summary": _eviction_summary(decisions),
        "active_subjects": sorted(str(item["subject"]) for item in decisions if item["active"]),
    }


def evaluate_audit(fixtures: AuditFixtureSet) -> JsonDict:
    """Score retention, interference, shuffled controls, eviction, and rollback."""

    policy = evaluate_promotion_eviction_policy(fixtures.memories)
    active = _active_memories(fixtures.memories, policy)
    useful_tasks = [task for task in fixtures.tasks if task.task_kind == "useful"]
    unrelated_tasks = [task for task in fixtures.tasks if task.task_kind == "unrelated"]

    aligned_rows = [_score_task(task, _select_memory(task, active)) for task in useful_tasks]
    shuffled_rows = [
        _score_task(task, memory)
        for task, memory in zip(
            useful_tasks,
            _derange([row["memory"] for row in aligned_rows], fixtures.seed),
            strict=True,
        )
    ]
    unrelated_rows = [_score_task(task, _select_memory(task, active)) for task in unrelated_tasks]
    retention_rate = _rate(sum(row["correct"] for row in aligned_rows), len(aligned_rows))
    interference_rate = _rate(
        sum(row["selected_subject"] is not None for row in unrelated_rows),
        len(unrelated_rows),
    )
    harmful_rollback_passed = any(
        task.requires_harmful_rollback
        and row["correct"]
        and str(row["selected_action"]).startswith(ROLLBACK_ACTION_PREFIXES)
        for task, row in zip(useful_tasks, aligned_rows, strict=True)
    )
    ready = bool(
        retention_rate == 1.0
        and interference_rate == 0.0
        and harmful_rollback_passed
        and policy["eviction_summary"]["evicted_by_reason"] == {"conflicting": 1, "stale": 1}
    )
    return {
        "retention_rate": retention_rate,
        "interference_rate": interference_rate,
        "harmful_memory_rollback_passed": bool(harmful_rollback_passed),
        "aligned_accuracy": retention_rate,
        "shuffled_accuracy": _rate(
            sum(row["correct"] for row in shuffled_rows), len(shuffled_rows)
        ),
        "aligned_rows": [_public_row(row) for row in aligned_rows],
        "shuffled_rows": [_public_row(row) for row in shuffled_rows],
        "unrelated_rows": [_public_row(row) for row in unrelated_rows],
        "promotion_summary": _ordered_counts(policy["promotion_summary"]),
        "eviction_summary": policy["eviction_summary"],
        "promotion_eviction_policy": policy,
        "memory_policy_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the principle-wrapped Exp 5261 artifact from cached fixtures."""

    fixtures = build_deterministic_fixtures()
    audit = evaluate_audit(fixtures)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "typed_memory_schema_invariants": list(typed_memory_schema_invariants()),
        "fixture_memory_kinds": list(fixtures.memory_kinds),
        "audit_rows": {
            "aligned": audit["aligned_rows"],
            "shuffled": audit["shuffled_rows"],
            "unrelated": audit["unrelated_rows"],
        },
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(audit)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "memory_policy_ready": bool(audit["memory_policy_ready"]),
        "memory_policy_ready_principle": (
            "Bare readiness gate for future cached/no-LLM self-learning experiments; "
            "true only when retention, interference, eviction, and harmful rollback pass."
        ),
        "retention_rate": _wrap("retention_rate", audit["retention_rate"]),
        "interference_rate": _wrap("interference_rate", audit["interference_rate"]),
        "harmful_memory_rollback_passed": _wrap(
            "harmful_memory_rollback_passed",
            audit["harmful_memory_rollback_passed"],
        ),
        "promotion_eviction_policy": _wrap(
            "promotion_eviction_policy",
            {
                "promotion_summary": audit["promotion_summary"],
                "eviction_summary": audit["eviction_summary"],
                "active_subjects": audit["promotion_eviction_policy"]["active_subjects"],
                "decisions": audit["promotion_eviction_policy"]["decisions"],
            },
        ),
        "fixture_checksums": _wrap(
            "fixture_checksums",
            fixture_checksums(fixtures=fixtures, root=root),
        ),
        "tests_run": [dict(row) for row in tests_run],
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    validate_artifact(artifact)
    return artifact


def fixture_checksums(*, fixtures: AuditFixtureSet, root: Path | str = REPO_ROOT) -> JsonDict:
    """Return stable checksums for fixture rows and cached source artifacts."""

    fixture_payload = {
        "memories": [asdict(memory) for memory in fixtures.memories],
        "tasks": [asdict(task) for task in fixtures.tasks],
        "memory_kinds": list(fixtures.memory_kinds),
        "seed": fixtures.seed,
    }
    return {
        "fixture_set_sha256": _sha256_json(fixture_payload),
        "memories": {memory.subject: _sha256_json(asdict(memory)) for memory in fixtures.memories},
        "tasks": {task.task_id: _sha256_json(asdict(task)) for task in fixtures.tasks},
        "source_artifacts": _source_artifact_checksums(Path(root)),
    }


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the Exp 5261 artifact schema used by tests and the conductor."""

    for field in REQUIRED_WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if not isinstance(wrapped, Mapping) or "value" not in wrapped or "principle" not in wrapped:
            raise ValueError(f"{field} must be principle-wrapped")  # pragma: no cover
    verdict = str(_wrapped_value(artifact, "honest_verdict"))
    if not (verdict.startswith("complete:") or verdict.startswith("blocked_")):
        raise ValueError("honest_verdict terminal prefix invalid")  # pragma: no cover
    if _wrapped_value(artifact, "inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError(
            "inference_substrate must be cached_fixture_replay_no_llm"
        )  # pragma: no cover
    if not isinstance(artifact.get("memory_policy_ready"), bool):
        raise ValueError("memory_policy_ready must be a bare bool")  # pragma: no cover
    if not artifact.get("memory_policy_ready_principle"):
        raise ValueError("missing memory_policy_ready_principle")  # pragma: no cover
    if not isinstance(artifact.get("tests_run"), list):
        raise ValueError("tests_run must be a bare list")  # pragma: no cover
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5261 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def _decision_for_memory(memory: AuditMemory) -> JsonDict:
    decision = decide_promotion(
        deterministic_guard_result={
            "passed": memory.guard_passed,
            "checks": {"fixture_guard_passed": memory.guard_passed},
            "no_test_gold_leak": True,
        },
        heldout_delta=memory.heldout_delta,
    )
    return {
        "subject": memory.subject,
        "head": memory.head,
        "task_slot": memory.task_slot,
        "fixture_kind": memory.fixture_kind,
        "action": memory.action,
        "effective_state": decision.promotion_state,
        "promotion_reason": decision.reason,
        "rollback_reason": decision.rollback_reason,
        "heldout_delta": memory.heldout_delta,
        "stale": memory.stale,
        "conflicts_with": memory.conflicts_with,
        "harmful": memory.harmful,
    }


def _eviction_reason(memory: AuditMemory, promoted_subjects: set[str]) -> str | None:
    if memory.stale:
        return "stale"
    if memory.conflicts_with and memory.conflicts_with in promoted_subjects:
        return "conflicting"
    return None


def _eviction_summary(decisions: Sequence[Mapping[str, Any]]) -> JsonDict:
    reasons = Counter(
        str(decision["eviction_reason"])
        for decision in decisions
        if decision["eviction_reason"] is not None
    )
    return {
        "evicted_n": sum(reasons.values()),
        "evicted_by_reason": _ordered_counts(dict(reasons)),
        "evicted_subjects": sorted(
            str(decision["subject"])
            for decision in decisions
            if decision["eviction_reason"] is not None
        ),
    }


def _active_memories(
    memories: Sequence[AuditMemory],
    policy: Mapping[str, Any],
) -> tuple[AuditMemory, ...]:
    active_subjects = set(policy["active_subjects"])
    return tuple(memory for memory in memories if memory.subject in active_subjects)


def _select_memory(task: AuditTask, active_memories: Sequence[AuditMemory]) -> AuditMemory | None:
    candidates = [memory for memory in active_memories if memory.task_slot == task.task_slot]
    return sorted(candidates, key=lambda memory: memory.subject)[0] if candidates else None


def _score_task(task: AuditTask, memory: AuditMemory | None) -> JsonDict:
    selected_action = task.default_action if memory is None else memory.action
    return {
        "task": task,
        "memory": memory,
        "task_id": task.task_id,
        "task_slot": task.task_slot,
        "task_kind": task.task_kind,
        "expected_subject": task.expected_subject,
        "selected_subject": None if memory is None else memory.subject,
        "selected_action": selected_action,
        "expected_action": task.expected_action,
        "correct": selected_action == task.expected_action,
    }


def _public_row(row: Mapping[str, Any]) -> JsonDict:
    return {key: value for key, value in row.items() if key not in {"task", "memory"}}


def _derange(memories: Sequence[AuditMemory | None], seed: int) -> tuple[AuditMemory | None, ...]:
    offset = seed % (len(memories) - 1) + 1
    return tuple(memories[offset:]) + tuple(memories[:offset])


def _honest_verdict(audit: Mapping[str, Any]) -> str:
    if audit["memory_policy_ready"]:
        return (
            "complete: memory policy ready for cached fixture replay; "
            f"retention_rate={audit['retention_rate']:.6f}, "
            f"interference_rate={audit['interference_rate']:.6f}, "
            "stale_conflict_eviction_passed=true, harmful_rollback_passed=true, "
            "live_cross_model_memory_still_unclaimed"
        )
    return (
        "complete: memory policy needs redesign before future continuous self-learning experiments"
    )


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _wrapped_value(artifact: Mapping[str, Any], field: str) -> Any:
    value = artifact.get(field)
    return value.get("value") if isinstance(value, Mapping) else None


def _rate(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6)


def _ordered_counts(counts: Mapping[str, int]) -> JsonDict:
    return {key: int(counts[key]) for key in sorted(counts)}


def _source_artifact_checksums(root: Path) -> JsonDict:
    checksums: JsonDict = {}
    for source in SOURCE_ARTIFACTS:
        path_text = source.split("#", 1)[0]
        path = root / path_text
        checksums[source] = (
            _sha256_bytes(receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH))
            if receipt_exists(path, artifact_relative_path=RESULT_RELATIVE_PATH)
            else None
        )
    return checksums


def _sha256_json(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    )


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return _sha256_json(stable)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
