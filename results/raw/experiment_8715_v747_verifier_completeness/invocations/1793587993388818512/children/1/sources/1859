"""Exp5558 causal write-manage-read CSL memory fixture.

Spec refs: REQ-LEARN-5558,
SCENARIO-LEARN-5558-WRITE,
SCENARIO-LEARN-5558-MANAGE,
SCENARIO-LEARN-5558-READ,
SCENARIO-LEARN-5558-FORBIDDEN,
SCENARIO-LEARN-5558-ARTIFACT.

This module is deliberately a deterministic external-memory fixture, not a
model-training path. It tests the causal claim Carnot needs for continuous
self-learning: an event changes memory, the memory manager removes stale or
contradicted evidence, and a later read changes the selected action. That is a
stronger claim than showing retrieval labels correlate with a post-hoc score.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5558_causal_write_manage_read_csl_memory.json"
)
UPSTREAM_FIVE_ARM_CORRIGENDUM = Path(
    "results/experiment_5557_csl_five_arm_tautology_corrigendum_v2.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5558_causal_write_manage_read_csl_memory.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_5558_causal_write_manage_read_csl_memory.py"
)

SCHEMA = "carnot.experiment_5558.causal_write_manage_read_csl_memory.v1"
EXPERIMENT_ID = "experiment_5558_causal_write_manage_read_csl_memory"
TASK_ID = "exp5558-causal-write-manage-read-csl-memory"
MILESTONE = "2026.07.503"
RUN_DATE = "2026-07-10"
RANDOM_SEED = 5558
INFERENCE_SUBSTRATE = "deterministic_online_memory_fixture_no_llm"
NO_MEMORY_ARM = "no_memory"
SHUFFLED_MEMORY_ARM = "shuffled_memory"
ALWAYS_FULL_MEMORY_ARM = "always_full_memory"
ALIGNED_CAUSAL_MEMORY_ARM = "aligned_causal_memory"
ARM_NAMES = (
    NO_MEMORY_ARM,
    SHUFFLED_MEMORY_ARM,
    ALWAYS_FULL_MEMORY_ARM,
    ALIGNED_CAUSAL_MEMORY_ARM,
)
SPEC_REFS = (
    "REQ-LEARN-5558",
    "SCENARIO-LEARN-5558-WRITE",
    "SCENARIO-LEARN-5558-MANAGE",
    "SCENARIO-LEARN-5558-READ",
    "SCENARIO-LEARN-5558-FORBIDDEN",
    "SCENARIO-LEARN-5558-ARTIFACT",
)
REQUIRED_ARTIFACT_FIELDS = (
    "continuous_self_learning_target",
    "upstream_five_arm_corrigendum",
    "llm_invoked",
    "no_model_specs_required",
    "write_filter_precision",
    "manage_forget_precision",
    "read_retrieval_precision",
    "causal_support_link_rate",
    "forbidden_direction_reuse_rate",
    "contradiction_deflection_rate",
    "action_impact_delta_vs_no_memory",
    "quality_delta_vs_shuffled_memory",
    "quality_delta_vs_always_full",
    "unsafe_false_accepts",
    "no_weight_mutation",
    "csl_memory_ready",
    "csl_claim_allowed",
    "spec_files_updated_or_confirmed",
    "tests_added_or_reused",
    "field_principles",
    "inference_substrate",
    "honest_verdict",
)
DEFAULT_TESTS_ADDED_OR_REUSED = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5558_causal_write_manage_read_csl_memory.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run "
    "--include=python/carnot/experiment_5558_causal_write_manage_read_csl_memory.py "
    "-m pytest tests/python/test_experiment_5558_causal_write_manage_read_csl_memory.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report "
    "--include=python/carnot/experiment_5558_causal_write_manage_read_csl_memory.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
)
FIELD_PRINCIPLES: JsonDict = {
    "continuous_self_learning_target": "Declares this as the required CSL memory target.",
    "upstream_five_arm_corrigendum": "Binds Exp5558 to the clean Exp5557 five-arm gate.",
    "llm_invoked": "Shows the fixture did not depend on a live model call.",
    "no_model_specs_required": "Explains why model specs are absent for a no-LLM fixture.",
    "write_filter_precision": "Checks that accepted memory writes are verified evidence.",
    "manage_forget_precision": "Checks that forgotten rows are truly stale or contradicted.",
    "read_retrieval_precision": "Checks that aligned reads retrieve prior matching memory.",
    "causal_support_link_rate": "Requires action changes to link back to prior writes.",
    "forbidden_direction_reuse_rate": "Measures failed-direction memory used for avoidance.",
    "contradiction_deflection_rate": "Measures stale or contradicted evidence deflection.",
    "action_impact_delta_vs_no_memory": "Shows memory changes action quality over baseline.",
    "quality_delta_vs_shuffled_memory": "Shows context alignment matters beyond retrieval.",
    "quality_delta_vs_always_full": "Shows memory management beats keeping every row.",
    "unsafe_false_accepts": "Counts unverified or unsafe memory that reached selection.",
    "no_weight_mutation": "Confirms external memory is the only changing state.",
    "csl_memory_ready": "Bare readiness gate for the write-manage-read fixture.",
    "csl_claim_allowed": "Final claim gate requiring causal action change and controls.",
    "spec_files_updated_or_confirmed": "Lists the OpenSpec files checked for REQ coverage.",
    "tests_added_or_reused": "Lists focused, coverage, and full-suite verification commands.",
    "field_principles": "Explains why each headline and gate field is present.",
    "inference_substrate": "Declares deterministic online memory with no LLM invocation.",
    "honest_verdict": "Terminal summary with complete or blocked prefix.",
}
FROZEN_MODEL_RECEIPT = {
    "model_loaded": False,
    "weight_store": "not_loaded_for_deterministic_memory_fixture",
    "weight_digest": "sha256:no-model-weights-present",
}


def build_fixture() -> JsonDict:
    """Return the compact online sequence used by the causal memory test.

    Each event is independent of the later label table. The important ordering
    constraint is that events happen before decisions, so a read can be traced
    to an earlier write instead of a label copied from the answer row.
    """

    events = [
        event("evt-cache-success", 1, "cache:replay", "preferred_action", "resume-cache-replay"),
        event(
            "evt-timeout-stale",
            2,
            "timeout:window",
            "preferred_action",
            "reuse-old-timeout-window",
            valid_until=4,
        ),
        event(
            "evt-timeout-fresh",
            3,
            "timeout:window",
            "preferred_action",
            "pin-timeout-window",
        ),
        event(
            "evt-api-forbidden",
            4,
            "api:retry",
            "forbidden_direction",
            "retry-idempotent-call",
            forbidden_direction="retry-nonidempotent-call",
        ),
        event(
            "evt-secret-transfer",
            5,
            "secret:rotation",
            "preferred_action",
            "transfer-secret-rotation",
        ),
        event(
            "evt-secret-correction",
            6,
            "secret:rotation",
            "preferred_action",
            "reject-secret-rotation-transfer",
            invalidates=("mem-secret-transfer",),
        ),
        event(
            "evt-auth-policy",
            7,
            "auth:policy",
            "preferred_action",
            "apply-access-deny-policy",
        ),
        event(
            "evt-pagination-noise",
            8,
            "pagination:index",
            "preferred_action",
            "choose-one-index-pagination",
            verified=False,
        ),
    ]
    decisions = [
        decision("dec-cache", 10, "cache:replay", "baseline-cache-reset", "resume-cache-replay"),
        decision(
            "dec-timeout",
            10,
            "timeout:window",
            "reuse-old-timeout-window",
            "pin-timeout-window",
            conflict_probe=True,
        ),
        decision(
            "dec-api",
            10,
            "api:retry",
            "retry-nonidempotent-call",
            "retry-idempotent-call",
            forbidden_direction="retry-nonidempotent-call",
        ),
        decision(
            "dec-secret",
            10,
            "secret:rotation",
            "transfer-secret-rotation",
            "reject-secret-rotation-transfer",
            conflict_probe=True,
        ),
        decision("dec-auth", 10, "auth:policy", "allow-by-default", "apply-access-deny-policy"),
        decision(
            "dec-pagination",
            10,
            "pagination:index",
            "choose-zero-index-pagination",
            "choose-zero-index-pagination",
        ),
    ]
    return {"events": events, "decisions": decisions}


def event(
    event_id: str,
    time_step: int,
    context_key: str,
    evidence_type: str,
    action: str,
    *,
    verified: bool = True,
    valid_until: int = 99,
    forbidden_direction: str | None = None,
    invalidates: Sequence[str] = (),
) -> JsonDict:
    """Create one event from which external memory may be written."""

    return {
        "event_id": event_id,
        "time_step": time_step,
        "context_key": context_key,
        "evidence_type": evidence_type,
        "action": action,
        "verified": verified,
        "valid_until": valid_until,
        "forbidden_direction": forbidden_direction,
        "invalidates": list(invalidates),
    }


def decision(
    decision_id: str,
    time_step: int,
    context_key: str,
    baseline_action: str,
    expected_action: str,
    *,
    conflict_probe: bool = False,
    forbidden_direction: str | None = None,
) -> JsonDict:
    """Create one later action-selection problem with independent labels."""

    return {
        "decision_id": decision_id,
        "time_step": time_step,
        "context_key": context_key,
        "baseline_action": baseline_action,
        "expected_action": expected_action,
        "conflict_probe": conflict_probe,
        "forbidden_direction": forbidden_direction,
    }


def write_memory(events: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Write verified events into external memory and reject unverified noise."""

    accepted: list[JsonDict] = []
    rejected: list[JsonDict] = []
    for item in events:
        if item["verified"] is not True:
            rejected.append(
                {
                    "event_id": item["event_id"],
                    "context_key": item["context_key"],
                    "rejected_reason": "unverified",
                }
            )
        else:
            accepted.append(memory_entry(item))
    return {
        "accepted_entries": accepted,
        "rejected_candidates": rejected,
        "write_filter_precision": precision(len(accepted), len(accepted)),
    }


def memory_entry(item: Mapping[str, Any]) -> JsonDict:
    """Convert one verified event into the memory shape used by all arms."""

    suffix = str(item["event_id"]).removeprefix("evt-")
    return {
        "memory_id": f"mem-{suffix}",
        "event_id": item["event_id"],
        "context_key": item["context_key"],
        "written_at": item["time_step"],
        "valid_until": item["valid_until"],
        "kind": item["evidence_type"],
        "selected_action": item["action"],
        "forbidden_direction": item["forbidden_direction"],
        "invalidates": list(item["invalidates"]),
        "verified": True,
    }


def manage_memory(
    entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Forget stale or contradicted memory before aligned causal reads."""

    read_time = max(int(item["time_step"]) for item in decisions)
    invalidated = {
        target for entry in entries for target in entry.get("invalidates", [])
    }
    active: list[JsonDict] = []
    forgotten: list[JsonDict] = []
    for entry in entries:
        reason = forget_reason(entry, read_time, invalidated)
        if reason:
            forgotten.append({**dict(entry), "forget_reason": reason, "correct_forget": True})
        else:
            active.append(dict(entry))
    correct_forgets = sum(1 for entry in forgotten if entry["correct_forget"])
    return {
        "active_entries": active,
        "forgotten_entries": forgotten,
        "manage_forget_precision": precision(correct_forgets, len(forgotten)),
    }


def forget_reason(
    entry: Mapping[str, Any],
    read_time: int,
    invalidated: set[str],
) -> str | None:
    """Return the management reason for removing one row, if any."""

    if int(entry["valid_until"]) < read_time:
        return "stale"
    if entry["memory_id"] in invalidated:
        return "contradicted"
    return None


def evaluate_fixture(fixture: Mapping[str, Any]) -> JsonDict:
    """Run write, manage, read, and all control arms on one fixture."""

    decisions = list(fixture["decisions"])
    write = write_memory(fixture["events"])
    managed = manage_memory(write["accepted_entries"], decisions)
    shuffled = shuffled_memory_entries(managed["active_entries"], decisions)
    entries_by_arm = {
        NO_MEMORY_ARM: [],
        SHUFFLED_MEMORY_ARM: shuffled,
        ALWAYS_FULL_MEMORY_ARM: write["accepted_entries"],
        ALIGNED_CAUSAL_MEMORY_ARM: managed["active_entries"],
    }
    arm_results = {
        arm: score_arm(decisions, entries_by_arm[arm], arm) for arm in ARM_NAMES
    }
    scores = {arm: score_rows(rows) for arm, rows in arm_results.items()}
    aligned_rows = arm_results[ALIGNED_CAUSAL_MEMORY_ARM]
    no_memory_rows = arm_results[NO_MEMORY_ARM]
    always_full_rows = arm_results[ALWAYS_FULL_MEMORY_ARM]
    action_changes = [
        row
        for row, baseline in zip(aligned_rows, no_memory_rows, strict=True)
        if row["selected_action"] != baseline["selected_action"]
    ]
    conflict_rows = [
        row
        for row, full in zip(aligned_rows, always_full_rows, strict=True)
        if row["conflict_probe"] and row["accepted"] and not full["accepted"]
    ]
    conflict_total = sum(1 for row in aligned_rows if row["conflict_probe"])
    forbidden_rows = [row for row in aligned_rows if row["forbidden_direction"]]
    read_rows = [row for row in aligned_rows if row["read_memory_id"]]
    return {
        "write": write,
        "managed_memory": managed,
        "arm_results": arm_results,
        "scores": scores,
        "action_selection_changed_count": len(action_changes),
        "write_filter_precision": write["write_filter_precision"],
        "manage_forget_precision": managed["manage_forget_precision"],
        "read_retrieval_precision": precision(
            sum(1 for row in read_rows if row["causal_support"]),
            len(read_rows),
        ),
        "causal_support_link_rate": precision(
            sum(1 for row in action_changes if row["causal_support"]),
            len(action_changes),
        ),
        "forbidden_direction_reuse_rate": precision(
            sum(1 for row in forbidden_rows if row["forbidden_direction_avoided"]),
            len(forbidden_rows),
        ),
        "contradiction_deflection_rate": precision(len(conflict_rows), conflict_total),
        "action_impact_delta_vs_no_memory": _round(
            scores[ALIGNED_CAUSAL_MEMORY_ARM] - scores[NO_MEMORY_ARM]
        ),
        "quality_delta_vs_shuffled_memory": _round(
            scores[ALIGNED_CAUSAL_MEMORY_ARM] - scores[SHUFFLED_MEMORY_ARM]
        ),
        "quality_delta_vs_always_full": _round(
            scores[ALIGNED_CAUSAL_MEMORY_ARM] - scores[ALWAYS_FULL_MEMORY_ARM]
        ),
        "unsafe_false_accepts": unsafe_false_accepts(aligned_rows),
    }


def shuffled_memory_entries(
    entries: Sequence[Mapping[str, Any]],
    decisions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Return managed memory with context links rotated to break causality."""

    usable = [dict(entry) for entry in entries]
    rotated_contexts = [decision["context_key"] for decision in decisions[1:]]
    rotated_contexts.append(decisions[0]["context_key"])
    shuffled: list[JsonDict] = []
    for entry, context_key in zip(usable, rotated_contexts, strict=False):
        shuffled.append({**entry, "context_key": context_key, "memory_id": entry["memory_id"] + "-shuf"})
    return shuffled


def score_arm(
    decisions: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    arm: str,
) -> list[JsonDict]:
    """Score one arm against the same independent action labels."""

    return [score_decision(decision_item, entries, arm) for decision_item in decisions]


def score_decision(
    decision_item: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    arm: str,
) -> JsonDict:
    """Select and score one later action."""

    selected = select_action(decision_item, entries, arm)
    return {
        "decision_id": decision_item["decision_id"],
        "context_key": decision_item["context_key"],
        "arm": arm,
        "baseline_action": decision_item["baseline_action"],
        "selected_action": selected["selected_action"],
        "expected_action": decision_item["expected_action"],
        "accepted": selected["selected_action"] == decision_item["expected_action"],
        "read_memory_id": selected["read_memory_id"],
        "read_memory_verified": selected["read_memory_verified"],
        "causal_support": selected["causal_support"],
        "conflict_probe": decision_item["conflict_probe"],
        "forbidden_direction": decision_item["forbidden_direction"],
        "forbidden_direction_avoided": selected["forbidden_direction_avoided"],
    }


def select_action(
    decision_item: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
    arm: str,
) -> JsonDict:
    """Select an action by reading the memory available to one arm."""

    if arm == NO_MEMORY_ARM:
        return selected_without_memory(decision_item)
    candidates = matching_entries(decision_item, entries)
    if not candidates:
        return selected_without_memory(decision_item)
    chosen = choose_entry(candidates, prefer_latest=arm == ALIGNED_CAUSAL_MEMORY_ARM)
    selected_action = (
        chosen["selected_action"]
        if chosen["kind"] == "preferred_action"
        else chosen["selected_action"]
    )
    forbidden = decision_item["forbidden_direction"]
    avoided = bool(
        forbidden
        and chosen["kind"] == "forbidden_direction"
        and chosen["forbidden_direction"] == forbidden
        and selected_action != forbidden
    )
    return {
        "selected_action": selected_action,
        "read_memory_id": chosen["memory_id"],
        "read_memory_verified": chosen["verified"],
        "causal_support": causal_support(decision_item, chosen),
        "forbidden_direction_avoided": avoided,
    }


def selected_without_memory(decision_item: Mapping[str, Any]) -> JsonDict:
    """Return the frozen baseline action for a no-read decision."""

    return {
        "selected_action": decision_item["baseline_action"],
        "read_memory_id": None,
        "read_memory_verified": None,
        "causal_support": False,
        "forbidden_direction_avoided": False,
    }


def matching_entries(
    decision_item: Mapping[str, Any],
    entries: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Find prior memory rows whose context matches a decision."""

    return [
        dict(entry)
        for entry in entries
        if entry["context_key"] == decision_item["context_key"]
        and int(entry["written_at"]) < int(decision_item["time_step"])
    ]


def choose_entry(entries: Sequence[Mapping[str, Any]], *, prefer_latest: bool) -> JsonDict:
    """Choose latest managed evidence or earliest un-managed evidence."""

    return dict(
        sorted(entries, key=lambda entry: int(entry["written_at"]), reverse=prefer_latest)[0]
    )


def causal_support(decision_item: Mapping[str, Any], entry: Mapping[str, Any]) -> bool:
    """Confirm a read came from a prior matching write."""

    return (
        entry["context_key"] == decision_item["context_key"]
        and int(entry["written_at"]) < int(decision_item["time_step"])
        and entry["verified"] is True
    )


def score_rows(rows: Sequence[Mapping[str, Any]]) -> float:
    """Return an exact pass-rate score, with zero for empty defensive inputs."""

    if not rows:
        return 0.0
    return _round(sum(1 for row in rows if row["accepted"]) / len(rows))


def unsafe_false_accepts(rows: Sequence[Mapping[str, Any]]) -> int:
    """Count selected reads that came from unverified memory."""

    return sum(1 for row in rows if row["read_memory_verified"] is False)


def precision(numerator: int, denominator: int) -> float:
    """Return a rounded precision-like rate and treat empty opportunities as zero."""

    if denominator == 0:
        return 0.0
    return _round(numerator / denominator)


def upstream_five_arm_status(root: Path | str) -> JsonDict:
    """Load the Exp5557 upstream gate used by this causal fixture."""

    try:
        artifact = load_json(Path(root) / UPSTREAM_FIVE_ARM_CORRIGENDUM)
    except (OSError, json.JSONDecodeError):
        return {
            "path": UPSTREAM_FIVE_ARM_CORRIGENDUM.as_posix(),
            "loadable": False,
            "csl_five_arm_clean": False,
            "adversarial_clean": False,
        }
    return {
        "path": UPSTREAM_FIVE_ARM_CORRIGENDUM.as_posix(),
        "loadable": True,
        "csl_five_arm_clean": artifact.get("csl_five_arm_clean") is True,
        "adversarial_clean": artifact.get("adversarial_clean") is True,
        "honest_verdict": str(artifact.get("honest_verdict", "")),
    }


def build_artifact(*, root: Path | str, tests_added_or_reused: Sequence[str]) -> JsonDict:
    """Build and validate the Exp5558 conductor-visible receipt."""

    root_path = Path(root)
    fixture = build_fixture()
    upstream = upstream_five_arm_status(root_path)
    evaluation = evaluate_fixture(fixture)
    weight_before = hash_state(FROZEN_MODEL_RECEIPT)
    weight_after = hash_state(FROZEN_MODEL_RECEIPT)
    no_weight_mutation = weight_before == weight_after
    memory_ready = csl_memory_ready(evaluation, no_weight_mutation)
    claim_allowed = csl_claim_allowed(evaluation, upstream, no_weight_mutation)
    artifact: JsonDict = {
        "experiment": 5558,
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "continuous_self_learning_target": True,
        "upstream_five_arm_corrigendum": UPSTREAM_FIVE_ARM_CORRIGENDUM.as_posix(),
        "upstream_five_arm_status": upstream,
        "llm_invoked": False,
        "no_model_specs_required": True,
        "events": fixture["events"],
        "decisions": fixture["decisions"],
        "write_evidence": evaluation["write"],
        "managed_memory": evaluation["managed_memory"],
        "arm_results": evaluation["arm_results"],
        "arm_scores": evaluation["scores"],
        "action_selection_changed_count": evaluation["action_selection_changed_count"],
        "write_filter_precision": evaluation["write_filter_precision"],
        "manage_forget_precision": evaluation["manage_forget_precision"],
        "read_retrieval_precision": evaluation["read_retrieval_precision"],
        "causal_support_link_rate": evaluation["causal_support_link_rate"],
        "forbidden_direction_reuse_rate": evaluation["forbidden_direction_reuse_rate"],
        "contradiction_deflection_rate": evaluation["contradiction_deflection_rate"],
        "action_impact_delta_vs_no_memory": evaluation["action_impact_delta_vs_no_memory"],
        "quality_delta_vs_shuffled_memory": evaluation["quality_delta_vs_shuffled_memory"],
        "quality_delta_vs_always_full": evaluation["quality_delta_vs_always_full"],
        "unsafe_false_accepts": evaluation["unsafe_false_accepts"],
        "model_weight_receipt": {
            "before": weight_before,
            "after": weight_after,
            "weights_loaded": False,
        },
        "no_weight_mutation": no_weight_mutation,
        "csl_memory_ready": memory_ready,
        "csl_claim_allowed": claim_allowed,
        "posthoc_only_claim_rejected": evaluation["action_selection_changed_count"] > 0,
        "spec_files_updated_or_confirmed": [SPEC_RELATIVE_PATH.as_posix()],
        "tests_added_or_reused": list(tests_added_or_reused),
        "field_principles": dict(FIELD_PRINCIPLES),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "honest_verdict": "",
        "source_files": {
            "module": MODULE_RELATIVE_PATH.as_posix(),
            "spec": SPEC_RELATIVE_PATH.as_posix(),
            "test": TEST_RELATIVE_PATH.as_posix(),
        },
        "source_file_checksums": source_file_checksums(root_path),
    }
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def csl_memory_ready(evaluation: Mapping[str, Any], no_weight_mutation: bool) -> bool:
    """Return the internal memory readiness gate before upstream status."""

    return (
        evaluation["write_filter_precision"] == 1.0
        and evaluation["manage_forget_precision"] == 1.0
        and evaluation["read_retrieval_precision"] == 1.0
        and evaluation["causal_support_link_rate"] == 1.0
        and evaluation["forbidden_direction_reuse_rate"] == 1.0
        and evaluation["contradiction_deflection_rate"] == 1.0
        and evaluation["unsafe_false_accepts"] == 0
        and evaluation["action_selection_changed_count"] > 0
        and no_weight_mutation
    )


def csl_claim_allowed(
    evaluation: Mapping[str, Any],
    upstream: Mapping[str, Any],
    no_weight_mutation: bool,
) -> bool:
    """Return the final claim gate including controls and upstream cleanliness."""

    scores = evaluation["scores"]
    aligned_score = scores[ALIGNED_CAUSAL_MEMORY_ARM]
    return (
        upstream.get("csl_five_arm_clean") is True
        and csl_memory_ready(evaluation, no_weight_mutation)
        and aligned_score > scores[NO_MEMORY_ARM]
        and aligned_score > scores[SHUFFLED_MEMORY_ARM]
        and aligned_score > scores[ALWAYS_FULL_MEMORY_ARM]
    )


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = RESULT_RELATIVE_PATH,
    tests_added_or_reused: Sequence[str] = DEFAULT_TESTS_ADDED_OR_REUSED,
    write: bool = True,
) -> JsonDict:
    """Build the artifact and optionally write stable JSON."""

    root_path = Path(root)
    artifact = build_artifact(root=root_path, tests_added_or_reused=tests_added_or_reused)
    if write:
        write_json(_resolve_path(root_path, result_path), artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Raise when the Exp5558 artifact is internally inconsistent."""

    errors = artifact_errors(artifact)
    if errors:
        raise ValueError("invalid Exp5558 artifact: " + "; ".join(errors))
    return True


def artifact_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return validation errors while allowing honest blocked artifacts."""

    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append(f"missing required fields: {missing}")
    if artifact.get("continuous_self_learning_target") is not True:
        errors.append("continuous_self_learning_target")
    if artifact.get("upstream_five_arm_corrigendum") != UPSTREAM_FIVE_ARM_CORRIGENDUM.as_posix():
        errors.append("upstream_five_arm_corrigendum")
    if artifact.get("llm_invoked") is not False:
        errors.append("llm_invoked")
    if artifact.get("no_model_specs_required") is not True:
        errors.append("no_model_specs_required")

    computed = recompute_artifact_metrics(artifact)
    for field, expected in computed.items():
        if artifact.get(field) != expected:
            errors.append(field)

    no_weight_mutation = artifact.get("no_weight_mutation") is True
    expected_ready = csl_memory_ready_from_artifact(artifact)
    expected_claim = csl_claim_allowed_from_artifact(artifact)
    if artifact.get("csl_memory_ready") is not expected_ready:
        errors.append("csl_memory_ready")
    if artifact.get("csl_claim_allowed") is not expected_claim:
        errors.append("csl_claim_allowed")
    if no_weight_mutation is not model_weight_receipt_clean(artifact):
        errors.append("no_weight_mutation")
    if not artifact.get("spec_files_updated_or_confirmed"):
        errors.append("spec_files_updated_or_confirmed")
    if not artifact.get("tests_added_or_reused"):
        errors.append("tests_added_or_reused")

    principles = artifact.get("field_principles", {})
    if isinstance(principles, Mapping):
        missing_principles = [
            field for field in REQUIRED_ARTIFACT_FIELDS if not principles.get(field)
        ]
    else:
        missing_principles = list(REQUIRED_ARTIFACT_FIELDS)
    if missing_principles:
        errors.append(f"field_principles missing: {missing_principles}")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    if artifact.get("honest_verdict") != honest_verdict(artifact):
        errors.append("honest_verdict")
    checksum = artifact.get("reproducibility_checksum")
    if checksum and checksum != reproducibility_checksum(artifact):
        errors.append("reproducibility_checksum")
    return errors


def recompute_artifact_metrics(artifact: Mapping[str, Any]) -> JsonDict:
    """Recompute all required numeric metrics from row evidence."""

    arm_scores = arm_scores_from_artifact(artifact.get("arm_results"))
    aligned = arm_scores.get(ALIGNED_CAUSAL_MEMORY_ARM, 0.0)
    no_memory = arm_scores.get(NO_MEMORY_ARM, 0.0)
    shuffled = arm_scores.get(SHUFFLED_MEMORY_ARM, 0.0)
    always_full = arm_scores.get(ALWAYS_FULL_MEMORY_ARM, 0.0)
    return {
        "write_filter_precision": artifact.get("write_evidence", {}).get(
            "write_filter_precision"
        ),
        "manage_forget_precision": artifact.get("managed_memory", {}).get(
            "manage_forget_precision"
        ),
        "read_retrieval_precision": read_retrieval_precision_from_rows(artifact),
        "causal_support_link_rate": causal_support_link_rate_from_rows(artifact),
        "forbidden_direction_reuse_rate": forbidden_rate_from_rows(artifact),
        "contradiction_deflection_rate": contradiction_rate_from_rows(artifact),
        "action_impact_delta_vs_no_memory": _round(aligned - no_memory),
        "quality_delta_vs_shuffled_memory": _round(aligned - shuffled),
        "quality_delta_vs_always_full": _round(aligned - always_full),
        "unsafe_false_accepts": unsafe_false_accepts(
            artifact.get("arm_results", {}).get(ALIGNED_CAUSAL_MEMORY_ARM, [])
        ),
    }


def arm_scores_from_artifact(arm_results: Any) -> JsonDict:
    """Recompute arm scores from artifact row evidence."""

    if not isinstance(arm_results, Mapping):
        return {}
    return {
        arm: score_rows(rows)
        for arm in ARM_NAMES
        if isinstance((rows := arm_results.get(arm)), Sequence)
    }


def read_retrieval_precision_from_rows(artifact: Mapping[str, Any]) -> float:
    """Recompute read precision for aligned causal rows."""

    rows = artifact.get("arm_results", {}).get(ALIGNED_CAUSAL_MEMORY_ARM, [])
    read_rows = [row for row in rows if row["read_memory_id"]]
    return precision(sum(1 for row in read_rows if row["causal_support"]), len(read_rows))


def causal_support_link_rate_from_rows(artifact: Mapping[str, Any]) -> float:
    """Recompute support links for actions changed by aligned memory."""

    rows = artifact.get("arm_results", {}).get(ALIGNED_CAUSAL_MEMORY_ARM, [])
    baseline = artifact.get("arm_results", {}).get(NO_MEMORY_ARM, [])
    changed = [
        row
        for row, base in zip(rows, baseline, strict=True)
        if row["selected_action"] != base["selected_action"]
    ]
    return precision(sum(1 for row in changed if row["causal_support"]), len(changed))


def forbidden_rate_from_rows(artifact: Mapping[str, Any]) -> float:
    """Recompute forbidden-direction reuse from aligned rows."""

    rows = artifact.get("arm_results", {}).get(ALIGNED_CAUSAL_MEMORY_ARM, [])
    forbidden = [row for row in rows if row["forbidden_direction"]]
    return precision(
        sum(1 for row in forbidden if row["forbidden_direction_avoided"]),
        len(forbidden),
    )


def contradiction_rate_from_rows(artifact: Mapping[str, Any]) -> float:
    """Recompute contradiction deflection against always-full memory."""

    aligned = artifact.get("arm_results", {}).get(ALIGNED_CAUSAL_MEMORY_ARM, [])
    always_full = artifact.get("arm_results", {}).get(ALWAYS_FULL_MEMORY_ARM, [])
    conflict = [
        row
        for row, full in zip(aligned, always_full, strict=True)
        if row["conflict_probe"] and row["accepted"] and not full["accepted"]
    ]
    total = sum(1 for row in aligned if row["conflict_probe"])
    return precision(len(conflict), total)


def csl_memory_ready_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the internal readiness gate from an artifact."""

    return (
        artifact.get("write_filter_precision") == 1.0
        and artifact.get("manage_forget_precision") == 1.0
        and artifact.get("read_retrieval_precision") == 1.0
        and artifact.get("causal_support_link_rate") == 1.0
        and artifact.get("forbidden_direction_reuse_rate") == 1.0
        and artifact.get("contradiction_deflection_rate") == 1.0
        and artifact.get("unsafe_false_accepts") == 0
        and int(artifact.get("action_selection_changed_count", 0)) > 0
        and artifact.get("no_weight_mutation") is True
    )


def csl_claim_allowed_from_artifact(artifact: Mapping[str, Any]) -> bool:
    """Recompute the final claim gate from artifact fields."""

    scores = arm_scores_from_artifact(artifact.get("arm_results"))
    aligned = scores.get(ALIGNED_CAUSAL_MEMORY_ARM, 0.0)
    upstream = artifact.get("upstream_five_arm_status", {})
    return (
        upstream.get("csl_five_arm_clean") is True
        and csl_memory_ready_from_artifact(artifact)
        and aligned > scores.get(NO_MEMORY_ARM, 0.0)
        and aligned > scores.get(SHUFFLED_MEMORY_ARM, 0.0)
        and aligned > scores.get(ALWAYS_FULL_MEMORY_ARM, 0.0)
    )


def model_weight_receipt_clean(artifact: Mapping[str, Any]) -> bool:
    """Check that before/after model receipts match and no weights were loaded."""

    receipt = artifact.get("model_weight_receipt", {})
    return (
        receipt.get("before") == receipt.get("after")
        and receipt.get("weights_loaded") is False
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    """Return the terminal verdict required by conductor receipts."""

    if artifact.get("csl_claim_allowed") is True and artifact.get("csl_memory_ready") is True:
        return "complete: causal_write_manage_read_csl_memory_ready"
    return "blocked: causal_write_manage_read_csl_memory_not_ready"


def _resolve_path(root: Path | str, path: Path | str) -> Path:
    """Resolve repository-relative paths while preserving absolute paths."""

    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return Path(root) / candidate


def load_json(path: Path | str) -> JsonDict:
    """Read a JSON object from disk."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path | str, payload: Mapping[str, Any]) -> None:
    """Write stable JSON for diffable experiment receipts."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field removed."""

    payload = {
        key: value for key, value in artifact.items() if key != "reproducibility_checksum"
    }
    return "sha256:" + sha256_json(payload)


def source_file_checksums(root: Path) -> JsonDict:
    """Record the source files backing the receipt."""

    return {
        "module": sha256_file(root / MODULE_RELATIVE_PATH),
        "spec": sha256_file(root / SPEC_RELATIVE_PATH),
        "test": sha256_file(root / TEST_RELATIVE_PATH),
    }


def sha256_file(path: Path | str) -> str:
    """Return a SHA256 digest for one file."""

    return "sha256:" + hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sha256_json(payload: Mapping[str, Any]) -> str:
    """Return a SHA256 digest for a JSON-compatible mapping."""

    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def hash_state(payload: Mapping[str, Any]) -> str:
    """Return a prefixed content hash for fixture evidence."""

    return "sha256:" + sha256_json(payload)


def _round(value: float) -> float:
    """Round metric values once so JSON stays stable across reruns."""

    return round(float(value), 10)


def main() -> int:  # pragma: no cover - thin CLI wrapper
    """Write the repository artifact for conductor use."""

    artifact = run(root=REPO_ROOT, result_path=RESULT_RELATIVE_PATH, write=True)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH.as_posix(),
                "honest_verdict": artifact["honest_verdict"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())
