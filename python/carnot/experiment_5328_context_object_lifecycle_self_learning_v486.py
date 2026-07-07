"""Exp5328: deterministic context-object lifecycle fixture.

Spec refs: REQ-LEARN-5328, SCENARIO-LEARN-5328, SCENARIO-LEARN-5329.

The fixture models context as explicit JSON-like objects rather than hidden
model state. Each lifecycle action either proposes a bank update through the
Exp5312 transition-memory verifier, or checks retrieval/answer-time use without
mutating the bank. That gives downstream experiments a stable policy-learning
surface for create, revise, fold, mask, archive, retrieve, commit, and rollback
actions while preserving the hard boundary that model weights never change.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot.pipeline.memory_transition_verifier import (
    MemoryTransitionProposal,
    MemoryTransitionVerifier,
)


JsonDict = dict[str, Any]
ContextBank = Mapping[str, Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5328_context_object_lifecycle_self_learning_v486"
EXPERIMENT_ID = 5328
MILESTONE = "v486"
SCHEMA = "carnot.experiment_5328.context_object_lifecycle_self_learning.v486"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5328
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5328_context_object_lifecycle_self_learning_v486.json"
)
EXP5312_RELATIVE_PATH = Path(
    "results/experiment_5312_trustmem_transition_verifier_self_learning_v485.json"
)
EXP5313_RELATIVE_PATH = Path(
    "results/experiment_5313_gated_memory_transition_policy_rollout_v485.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5328_context_object_lifecycle_self_learning_v486.py"
)
VERIFIER_RELATIVE_PATH = Path("python/carnot/pipeline/memory_transition_verifier.py")
INFERENCE_SUBSTRATE = "deterministic_context_lifecycle_fixture"
SPEC_REFS = (
    "REQ-LEARN-5328",
    "SCENARIO-LEARN-5328",
    "SCENARIO-LEARN-5329",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

LIFECYCLE_ACTION_SET = (
    "create",
    "revise",
    "fold",
    "mask",
    "archive",
    "retrieve",
    "commit",
    "rollback",
)
OBJECT_TYPES = (
    "answer_context",
    "archival_record",
    "evidence_receipt",
    "folded_summary",
    "ghost_claim",
    "masked_note",
    "patch_receipt",
    "policy_rule",
)
CURRENT_LABELS = (
    "archived",
    "committed",
    "current",
    "historical",
    "masked",
    "retrieved",
    "rolled_back",
    "transition",
)
TRANSITION_LABELS = (
    "archived",
    "committed",
    "corrupted",
    "created",
    "folded",
    "ghosted",
    "masked",
    "omitted",
    "pruned",
    "retrieved",
    "revised",
    "rolled_back",
    "stale_retrieved",
)
BANK_FAILURE_FAMILY = "bank"
RETRIEVAL_FAILURE_FAMILY = "retrieval"
ANSWER_FAILURE_FAMILY = "answer"
FAILURE_FAMILIES = (
    BANK_FAILURE_FAMILY,
    RETRIEVAL_FAILURE_FAMILY,
    ANSWER_FAILURE_FAMILY,
)
BANK_MUTATION_ACTIONS = {
    "archive",
    "commit",
    "create",
    "fold",
    "mask",
    "prune",
    "revise",
    "rollback",
}

FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5328 artifact so Exp5329/Exp5330 gates cannot "
        "confuse lifecycle learning with earlier transition-memory verifier experiments."
    ),
    "milestone": (
        "Binds the fixture to milestone v486 where context-object lifecycle "
        "learning becomes the downstream target."
    ),
    "status": (
        "Reports whether the lifecycle fixture is usable by Exp5329 and Exp5330 "
        "rather than merely present as code."
    ),
    "honest_verdict": (
        "Terminal Exp5328 verdict; starts with complete: or blocked_ and states "
        "whether lifecycle safety and recoverability gates passed."
    ),
    "inference_substrate": (
        "Declares deterministic context lifecycle fixture evaluation with no live "
        "LLM, API judge, model generation, fine-tuning, adapter update, or weight "
        "mutation."
    ),
    "continuous_self_learning_target": (
        "Bare gate showing the fixture trains or evaluates policy actions for "
        "continuous self-learning rather than static documentation."
    ),
    "no_weight_mutation": (
        "Bare gate confirming only deterministic context-object bank state changed, "
        "never model weights or adapters."
    ),
    "context_object_count": (
        "Bare count of stable context object IDs exercised by the lifecycle fixture."
    ),
    "lifecycle_action_set": (
        "Lists every lifecycle action so downstream policy learners know the "
        "available action vocabulary."
    ),
    "bank_failure_detection_rate": (
        "Bare numeric rate over bank-maintenance failure modes, separated from "
        "retrieval and answer-time failures."
    ),
    "retrieval_failure_detection_rate": (
        "Bare numeric rate over stale or masked retrieval failures, separated from "
        "bank and answer-time failures."
    ),
    "answer_failure_detection_rate": (
        "Bare numeric rate over unsafe answer-time context use, separated from bank "
        "and retrieval failures."
    ),
    "rollback_success_rate": (
        "Bare numeric rate showing rejected unsafe commits preserved the pre-action "
        "bank and safe rollback/recovery restored usable context."
    ),
    "context_lifecycle_fixture_ready": (
        "Bare downstream gate for Exp5329 and Exp5330; true only when object IDs, "
        "sidecars, failure scores, safe commits, rollback, recoverability, unsafe "
        "rejection, tests, and no-weight-mutation checks all pass."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the fixture "
        "and artifact are usable by Exp5329 and Exp5330."
    ),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "lifecycle_action_set",
    "tests_run",
)
BARE_NUMERIC_FIELDS = (
    "bank_failure_detection_rate",
    "retrieval_failure_detection_rate",
    "answer_failure_detection_rate",
    "rollback_success_rate",
)


@dataclass(frozen=True)
class LifecycleCase:
    """One deterministic context-object lifecycle action.

    The case keeps bank maintenance, retrieval, and answer-time checks in one
    record so the fixture can prove where a failure was detected. Bank mutation
    actions carry a proposed and expected bank for the transition verifier;
    retrieval and answer cases keep the bank unchanged and check whether the
    selected context may safely reach the answer path.
    """

    case_id: str
    action: str
    prior_state: ContextBank
    proposed_state: ContextBank
    expected_state: ContextBank
    protected_keys: tuple[str, ...] = ()
    safe_expected: bool = True
    failure_family: str | None = None
    failure_mode: str | None = None
    retrieval_object_ids: tuple[str, ...] = ()
    answer_context_object_ids: tuple[str, ...] = ()
    recover_object_id: str | None = None
    rollback_expected: bool = False


def build_lifecycle_fixture() -> tuple[LifecycleCase, ...]:
    """Return the deterministic lifecycle panel used by tests and artifact build."""

    objects = _fixture_objects()
    runtime_v1 = objects["runtime_v1"]
    runtime_v2 = objects["runtime_v2"]
    runtime_stale = objects["runtime_stale"]
    sensor = objects["sensor"]
    rubric = objects["rubric"]
    rubric_corrupt = objects["rubric_corrupt"]
    folded = objects["folded"]
    secret = objects["secret"]
    secret_masked = objects["secret_masked"]
    policy = objects["policy"]
    policy_archived = objects["policy_archived"]
    patch_pending = objects["patch_pending"]
    patch_committed = objects["patch_committed"]
    patch_corrupt = objects["patch_corrupt"]
    patch_rolled_back = objects["patch_rolled_back"]
    ghost = objects["ghost"]

    return (
        LifecycleCase(
            case_id="life-create-runtime",
            action="create",
            prior_state={},
            proposed_state={runtime_v1["object_id"]: runtime_v1},
            expected_state={runtime_v1["object_id"]: runtime_v1},
        ),
        LifecycleCase(
            case_id="life-revise-runtime",
            action="revise",
            prior_state={runtime_v1["object_id"]: runtime_v1},
            proposed_state={runtime_v2["object_id"]: runtime_v2},
            expected_state={runtime_v2["object_id"]: runtime_v2},
        ),
        LifecycleCase(
            case_id="life-fold-runtime-rubric",
            action="fold",
            prior_state={runtime_v2["object_id"]: runtime_v2, rubric["object_id"]: rubric},
            proposed_state={
                runtime_v2["object_id"]: runtime_v2,
                rubric["object_id"]: rubric,
                folded["object_id"]: folded,
            },
            expected_state={
                runtime_v2["object_id"]: runtime_v2,
                rubric["object_id"]: rubric,
                folded["object_id"]: folded,
            },
            protected_keys=(runtime_v2["object_id"], rubric["object_id"]),
        ),
        LifecycleCase(
            case_id="life-mask-secret",
            action="mask",
            prior_state={secret["object_id"]: secret},
            proposed_state={secret_masked["object_id"]: secret_masked},
            expected_state={secret_masked["object_id"]: secret_masked},
        ),
        LifecycleCase(
            case_id="life-archive-policy",
            action="archive",
            prior_state={policy["object_id"]: policy},
            proposed_state={policy_archived["object_id"]: policy_archived},
            expected_state={policy_archived["object_id"]: policy_archived},
        ),
        LifecycleCase(
            case_id="life-retrieve-archive-recover",
            action="retrieve",
            prior_state={policy_archived["object_id"]: policy_archived},
            proposed_state={policy_archived["object_id"]: policy_archived},
            expected_state={policy_archived["object_id"]: policy_archived},
            retrieval_object_ids=(policy_archived["object_id"],),
            answer_context_object_ids=(policy_archived["object_id"],),
            recover_object_id=policy_archived["object_id"],
        ),
        LifecycleCase(
            case_id="life-commit-patch",
            action="commit",
            prior_state={patch_pending["object_id"]: patch_pending},
            proposed_state={patch_committed["object_id"]: patch_committed},
            expected_state={patch_committed["object_id"]: patch_committed},
        ),
        LifecycleCase(
            case_id="life-rollback-corrupt-patch",
            action="rollback",
            prior_state={patch_corrupt["object_id"]: patch_corrupt},
            proposed_state={patch_rolled_back["object_id"]: patch_rolled_back},
            expected_state={patch_rolled_back["object_id"]: patch_rolled_back},
            rollback_expected=True,
            recover_object_id=patch_rolled_back["object_id"],
        ),
        LifecycleCase(
            case_id="life-ghost-memory",
            action="create",
            prior_state={runtime_v2["object_id"]: runtime_v2},
            proposed_state={runtime_v2["object_id"]: runtime_v2, ghost["object_id"]: ghost},
            expected_state={runtime_v2["object_id"]: runtime_v2},
            protected_keys=(runtime_v2["object_id"],),
            safe_expected=False,
            failure_family=BANK_FAILURE_FAMILY,
            failure_mode="ghost_memory",
        ),
        LifecycleCase(
            case_id="life-omission-sensor-rule",
            action="fold",
            prior_state={runtime_v2["object_id"]: runtime_v2},
            proposed_state={runtime_v2["object_id"]: runtime_v2},
            expected_state={runtime_v2["object_id"]: runtime_v2, sensor["object_id"]: sensor},
            protected_keys=(runtime_v2["object_id"],),
            safe_expected=False,
            failure_family=BANK_FAILURE_FAMILY,
            failure_mode="omission",
        ),
        LifecycleCase(
            case_id="life-corrupt-rubric",
            action="revise",
            prior_state={rubric["object_id"]: rubric},
            proposed_state={rubric_corrupt["object_id"]: rubric_corrupt},
            expected_state={rubric["object_id"]: rubric},
            protected_keys=(rubric["object_id"],),
            safe_expected=False,
            failure_family=BANK_FAILURE_FAMILY,
            failure_mode="corruption",
        ),
        LifecycleCase(
            case_id="life-unsafe-prune-runtime",
            action="prune",
            prior_state={runtime_v2["object_id"]: runtime_v2},
            proposed_state={},
            expected_state={runtime_v2["object_id"]: runtime_v2},
            protected_keys=(runtime_v2["object_id"],),
            safe_expected=False,
            failure_family=BANK_FAILURE_FAMILY,
            failure_mode="unsafe_prune",
            rollback_expected=True,
        ),
        LifecycleCase(
            case_id="life-stale-retrieval",
            action="retrieve",
            prior_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            proposed_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            expected_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            safe_expected=False,
            failure_family=RETRIEVAL_FAILURE_FAMILY,
            failure_mode="stale_retrieval",
            retrieval_object_ids=(runtime_stale["object_id"],),
        ),
        LifecycleCase(
            case_id="life-mask-retrieval-leak",
            action="mask",
            prior_state={secret_masked["object_id"]: secret_masked},
            proposed_state={secret_masked["object_id"]: secret_masked},
            expected_state={secret_masked["object_id"]: secret_masked},
            safe_expected=False,
            failure_family=RETRIEVAL_FAILURE_FAMILY,
            failure_mode="mask_leakage",
            retrieval_object_ids=(secret_masked["object_id"],),
        ),
        LifecycleCase(
            case_id="life-answer-stale-context",
            action="retrieve",
            prior_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            proposed_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            expected_state={
                runtime_v2["object_id"]: runtime_v2,
                runtime_stale["object_id"]: runtime_stale,
            },
            safe_expected=False,
            failure_family=ANSWER_FAILURE_FAMILY,
            failure_mode="answer_stale_context",
            retrieval_object_ids=(runtime_stale["object_id"],),
            answer_context_object_ids=(runtime_stale["object_id"],),
        ),
        LifecycleCase(
            case_id="life-answer-corrupt-context",
            action="commit",
            prior_state={rubric_corrupt["object_id"]: rubric_corrupt},
            proposed_state={rubric_corrupt["object_id"]: rubric_corrupt},
            expected_state={rubric["object_id"]: rubric},
            safe_expected=False,
            failure_family=ANSWER_FAILURE_FAMILY,
            failure_mode="answer_corruption",
            answer_context_object_ids=(rubric_corrupt["object_id"],),
        ),
    )


def build_lifecycle_verifier() -> MemoryTransitionVerifier:
    """Return the Exp5312 transition verifier used for lifecycle bank writes."""

    return MemoryTransitionVerifier(threshold=1.0)


def evaluate_lifecycle_fixture(cases: Sequence[LifecycleCase]) -> JsonDict:
    """Evaluate lifecycle safety, recoverability, and separated failure rates."""

    rows = [_evaluate_case(case) for case in cases]
    failure_counts = {
        family: _failure_count(rows, family)
        for family in FAILURE_FAMILIES
    }
    rollback_rows = [row for row in rows if bool(row["rollback_expected"])]
    rollback_success_rate = _rate(
        sum(1 for row in rollback_rows if bool(row["rollback_success"])),
        len(rollback_rows),
    )
    no_weight_mutation = all(row["model_weights_mutated"] is False for row in rows)
    action_set_complete = set(LIFECYCLE_ACTION_SET).issubset(lifecycle_action_counts(cases))
    object_schema_valid = all(_object_schema_valid(obj) for case in cases for obj in objects_in_case(case))
    safe_rows_ok = all(
        bool(row["accepted"])
        for row in rows
        if row["failure_family"] is None and row["action"] != "retrieve"
    )
    safe_retrieval_ok = all(
        bool(row["accepted"] and row["recovered_from_sidecar"])
        for row in rows
        if row["case_id"] == "life-retrieve-archive-recover"
    )
    unsafe_rows_ok = all(
        bool(row["detected_failure"] and not row["accepted"])
        for row in rows
        if row["failure_family"] in FAILURE_FAMILIES
    )
    ready = bool(
        action_set_complete
        and object_schema_valid
        and safe_rows_ok
        and safe_retrieval_ok
        and unsafe_rows_ok
        and _detection_rate(failure_counts[BANK_FAILURE_FAMILY]) == 1.0
        and _detection_rate(failure_counts[RETRIEVAL_FAILURE_FAMILY]) == 1.0
        and _detection_rate(failure_counts[ANSWER_FAILURE_FAMILY]) == 1.0
        and rollback_success_rate == 1.0
        and no_weight_mutation
    )
    return {
        "lifecycle_rows": rows,
        "context_object_ids": stable_context_object_ids(cases),
        "context_object_count": context_object_count(cases),
        "lifecycle_action_counts": lifecycle_action_counts(cases),
        "failure_counts": failure_counts,
        "bank_failure_detection_rate": _detection_rate(
            failure_counts[BANK_FAILURE_FAMILY]
        ),
        "retrieval_failure_detection_rate": _detection_rate(
            failure_counts[RETRIEVAL_FAILURE_FAMILY]
        ),
        "answer_failure_detection_rate": _detection_rate(
            failure_counts[ANSWER_FAILURE_FAMILY]
        ),
        "rollback_success_rate": rollback_success_rate,
        "transition_verifier_reuse": {
            "verifier_path": str(VERIFIER_RELATIVE_PATH),
            "bank_mutation_rows": sum(
                1 for row in rows if bool(row["transition_verifier_reused"])
            ),
        },
        "no_weight_mutation": no_weight_mutation,
        "context_lifecycle_fixture_ready": ready,
    }


def lifecycle_action_counts(cases: Sequence[LifecycleCase]) -> dict[str, int]:
    """Count lifecycle actions in a stable order for auditability."""

    counts = Counter(case.action for case in cases)
    ordered_actions = (*LIFECYCLE_ACTION_SET, "prune")
    return {action: counts[action] for action in ordered_actions if counts[action]}


def stable_context_object_ids(cases: Sequence[LifecycleCase]) -> list[str]:
    """Return every stable context object ID exercised by the fixture."""

    object_ids = {obj["object_id"] for case in cases for obj in objects_in_case(case)}
    return sorted(object_ids)


def context_object_count(cases: Sequence[LifecycleCase]) -> int:
    """Return the bare object count used by the result artifact."""

    return len(stable_context_object_ids(cases))


def objects_in_case(case: LifecycleCase) -> tuple[JsonDict, ...]:
    """Return every context object appearing in a lifecycle case."""

    objects: dict[str, JsonDict] = {}
    for bank in (case.prior_state, case.proposed_state, case.expected_state):
        for object_id, obj in bank.items():
            objects[str(object_id)] = dict(obj)
    return tuple(objects[object_id] for object_id in sorted(objects))


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5328 artifact from deterministic fixture replay."""

    cases = build_lifecycle_fixture()
    evaluation = evaluate_lifecycle_fixture(cases)
    ready = bool(evaluation["context_lifecycle_fixture_ready"] and tests_run)
    status = "ready_for_exp5329_exp5330" if ready else "blocked_tests_or_fixture_not_ready"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5312_RELATIVE_PATH), str(EXP5313_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(ready, evaluation)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "context_object_count": evaluation["context_object_count"],
        "lifecycle_action_set": _wrap("lifecycle_action_set", list(LIFECYCLE_ACTION_SET)),
        "bank_failure_detection_rate": evaluation["bank_failure_detection_rate"],
        "retrieval_failure_detection_rate": evaluation["retrieval_failure_detection_rate"],
        "answer_failure_detection_rate": evaluation["answer_failure_detection_rate"],
        "rollback_success_rate": evaluation["rollback_success_rate"],
        "context_lifecycle_fixture_ready": ready,
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "context_object_ids": evaluation["context_object_ids"],
        "lifecycle_action_counts": evaluation["lifecycle_action_counts"],
        "failure_counts": evaluation["failure_counts"],
        "lifecycle_rows": evaluation["lifecycle_rows"],
        "transition_verifier_reuse": evaluation["transition_verifier_reuse"],
        "weight_mutation_receipt": _weight_mutation_receipt(evaluation),
        "methodology_note": (
            "Detection rates are deterministic fixture gates over enumerated "
            "lifecycle failure rows, not statistical classifier-performance claims."
        ),
        "field_principles": dict(FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields consumed by Exp5329/Exp5330 gates."""

    for field in WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if (
            not isinstance(wrapped, Mapping)
            or "value" not in wrapped
            or wrapped.get("principle") != FIELD_PRINCIPLES[field]
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if not str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_target") is not True:
        raise ValueError("continuous_self_learning_target must be bare true")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    if isinstance(artifact.get("context_object_count"), bool) or not isinstance(
        artifact.get("context_object_count"),
        int,
    ):
        raise ValueError("context_object_count must be a bare integer")
    if artifact["lifecycle_action_set"]["value"] != list(LIFECYCLE_ACTION_SET):
        raise ValueError("lifecycle_action_set mismatch")
    for field in BARE_NUMERIC_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ValueError(f"{field} must be bare numeric")
        if float(value) != 1.0:
            raise ValueError(f"{field} must be 1.0")
    if not isinstance(artifact.get("context_lifecycle_fixture_ready"), bool):
        raise ValueError("context_lifecycle_fixture_ready must be bare bool")
    if artifact["context_lifecycle_fixture_ready"] and not artifact["tests_run"]["value"]:
        raise ValueError("tests_run must record commands for a ready fixture")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5328 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for deterministic source inputs."""

    root_path = Path(root)
    return {
        "exp5312": _sha256_file(root_path / EXP5312_RELATIVE_PATH),
        "exp5313": _sha256_file(root_path / EXP5313_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "transition_verifier": _sha256_file(root_path / VERIFIER_RELATIVE_PATH),
    }


def _evaluate_case(case: LifecycleCase) -> JsonDict:
    if case.action in BANK_MUTATION_ACTIONS:
        transition = _evaluate_bank_action(case)
    else:
        transition = _evaluate_non_mutating_action(case)
    detected_failure = _detect_failure(case, transition)
    accepted = bool(case.safe_expected and not detected_failure and transition["accepted"])
    answer_context_allowed = bool(accepted and _answer_context_safe(case))
    rollback_success = _rollback_success(case, transition, detected_failure)
    return {
        "case_id": case.case_id,
        "action": case.action,
        "failure_family": case.failure_family,
        "failure_mode": case.failure_mode,
        "safe_expected": case.safe_expected,
        "accepted": accepted,
        "detected_failure": detected_failure,
        "transition_verifier_reused": transition["transition_verifier_reused"],
        "coverage_score": transition["coverage_score"],
        "preservation_score": transition["preservation_score"],
        "faithfulness_score": transition["faithfulness_score"],
        "rejection_reasons": transition["rejection_reasons"],
        "prior_state": _json_ready(case.prior_state),
        "committed_state": _json_ready(transition["committed_state"]),
        "committed_state_changed": transition["committed_state"] != case.prior_state,
        "retrieval_object_ids": list(case.retrieval_object_ids),
        "answer_context_object_ids": list(case.answer_context_object_ids),
        "answer_context_allowed": answer_context_allowed,
        "recovered_from_sidecar": _recovered_from_sidecar(case),
        "sidecar_preserved": _sidecar_preserved(case, transition["committed_state"]),
        "rollback_expected": case.rollback_expected,
        "rollback_success": rollback_success,
        "model_weights_mutated": False,
    }


def _evaluate_bank_action(case: LifecycleCase) -> JsonDict:
    verifier = build_lifecycle_verifier()
    proposal = MemoryTransitionProposal(
        transition_id=f"t5328-{case.case_id}",
        label=case.action if case.safe_expected else str(case.failure_mode),
        source_stress_event_id=case.case_id,
        prior_state=case.prior_state,
        proposed_state=case.proposed_state,
        expected_state=case.expected_state,
        protected_keys=case.protected_keys,
        safe_expected=case.safe_expected,
    )
    decision, committed_state = verifier.commit_if_safe(case.prior_state, proposal)
    return {
        "accepted": bool(decision.accepted),
        "committed_state": committed_state,
        "transition_verifier_reused": True,
        "coverage_score": decision.coverage_score,
        "preservation_score": decision.preservation_score,
        "faithfulness_score": decision.faithfulness_score,
        "rejection_reasons": list(decision.rejection_reasons),
    }


def _evaluate_non_mutating_action(case: LifecycleCase) -> JsonDict:
    safe = _retrieval_context_safe(case) and _answer_context_safe(case)
    return {
        "accepted": bool(case.safe_expected and safe),
        "committed_state": _copy_bank(case.prior_state),
        "transition_verifier_reused": False,
        "coverage_score": 1.0 if safe else 0.0,
        "preservation_score": 1.0,
        "faithfulness_score": 1.0 if safe else 0.0,
        "rejection_reasons": [] if safe else [f"{case.failure_family}_failure_detected"],
    }


def _detect_failure(case: LifecycleCase, transition: Mapping[str, Any]) -> bool:
    if case.failure_family == BANK_FAILURE_FAMILY:
        return not bool(transition["accepted"])
    if case.failure_family == RETRIEVAL_FAILURE_FAMILY:
        return not _retrieval_context_safe(case)
    if case.failure_family == ANSWER_FAILURE_FAMILY:
        return not _answer_context_safe(case)
    return False


def _retrieval_context_safe(case: LifecycleCase) -> bool:
    bank = case.proposed_state or case.prior_state
    for object_id in case.retrieval_object_ids:
        obj = bank.get(object_id)
        if obj is None:
            return False
        if obj.get("current_label") in {"historical", "masked"}:
            return False
    return True


def _answer_context_safe(case: LifecycleCase) -> bool:
    bank = case.proposed_state or case.prior_state
    for object_id in case.answer_context_object_ids:
        obj = bank.get(object_id)
        if obj is None:
            return False
        if obj.get("current_label") in {"historical", "masked"}:
            return False
        if obj.get("transition_label") in {"corrupted", "ghosted", "stale_retrieved"}:
            return False
    return True


def _rollback_success(
    case: LifecycleCase,
    transition: Mapping[str, Any],
    detected_failure: bool,
) -> bool:
    if not case.rollback_expected:
        return False
    if case.safe_expected:
        return bool(transition["accepted"] and _recovered_from_sidecar(case))
    return bool(detected_failure and transition["committed_state"] == case.prior_state)


def _recovered_from_sidecar(case: LifecycleCase) -> bool:
    if case.recover_object_id is None:
        return False
    obj = case.proposed_state.get(case.recover_object_id) or case.prior_state.get(
        case.recover_object_id
    )
    if obj is None:
        return False
    sidecar = obj.get("recoverable_sidecar", {})
    return bool(sidecar.get("recoverable") and sidecar.get("restore_payload"))


def _sidecar_preserved(case: LifecycleCase, committed_state: ContextBank) -> bool:
    state = committed_state if case.safe_expected else case.prior_state
    objects = state.values() or case.prior_state.values()
    return all(bool(obj.get("recoverable_sidecar", {}).get("recoverable")) for obj in objects)


def _failure_count(rows: Sequence[Mapping[str, Any]], family: str) -> JsonDict:
    family_rows = [row for row in rows if row["failure_family"] == family]
    return {
        "detected": sum(1 for row in family_rows if bool(row["detected_failure"])),
        "total": len(family_rows),
    }


def _detection_rate(counts: Mapping[str, Any]) -> float:
    return _rate(int(counts["detected"]), int(counts["total"]))


def _fixture_objects() -> dict[str, JsonDict]:
    runtime_v1 = _context_object(
        "ctx.runtime.receipt",
        "evidence_receipt",
        "current",
        (),
        "created",
        "native_cuda_cli",
        ("ar-update-runtime",),
    )
    runtime_v2 = _context_object(
        "ctx.runtime.receipt",
        "evidence_receipt",
        "current",
        ("current",),
        "revised",
        "native_cuda_cli_with_gpu_receipt",
        ("ar-update-runtime", "revise-runtime-receipt"),
    )
    runtime_stale = _context_object(
        "ctx.runtime.receipt.v0",
        "evidence_receipt",
        "historical",
        ("current",),
        "stale_retrieved",
        "cpu_only_offload_receipt",
        ("stale-query-outdated",),
    )
    sensor = _context_object(
        "ctx.sensor.reject_rule",
        "policy_rule",
        "current",
        (),
        "created",
        "reject_unsupported_sensor_claim",
        ("ttl-update-sensor",),
    )
    rubric = _context_object(
        "ctx.arc.rubric",
        "policy_rule",
        "current",
        (),
        "created",
        "reject_lexical_only_support",
        ("lru-update-rubric",),
    )
    rubric_corrupt = _context_object(
        "ctx.arc.rubric",
        "policy_rule",
        "current",
        ("current",),
        "corrupted",
        "accept_lexical_only_support",
        ("unsupported-corruption",),
    )
    folded = _context_object(
        "ctx.folded.runtime_rubric",
        "folded_summary",
        "current",
        (),
        "folded",
        "runtime_cli_and_arc_rubric_summary",
        ("fold-runtime-rubric",),
        source_object_ids=("ctx.runtime.receipt", "ctx.arc.rubric"),
    )
    secret = _context_object(
        "ctx.operator.secret",
        "masked_note",
        "current",
        (),
        "created",
        "operator_private_note",
        ("operator-note",),
    )
    secret_masked = _context_object(
        "ctx.operator.secret",
        "masked_note",
        "masked",
        ("current",),
        "masked",
        "MASKED",
        ("operator-note", "mask-secret"),
        restore_payload="operator_private_note",
    )
    policy = _context_object(
        "ctx.archive.policy",
        "archival_record",
        "current",
        (),
        "created",
        "legacy_safe_archive_rule",
        ("archive-policy",),
    )
    policy_archived = _context_object(
        "ctx.archive.policy",
        "archival_record",
        "archived",
        ("current",),
        "archived",
        "legacy_safe_archive_rule",
        ("archive-policy", "archive-action"),
        restore_payload="legacy_safe_archive_rule",
    )
    patch_pending = _context_object(
        "ctx.patch.autofix",
        "patch_receipt",
        "transition",
        (),
        "created",
        "candidate_patch_pending",
        ("patch-proposal",),
    )
    patch_committed = _context_object(
        "ctx.patch.autofix",
        "patch_receipt",
        "committed",
        ("transition",),
        "committed",
        "candidate_patch_passed",
        ("patch-proposal", "commit-check"),
    )
    patch_corrupt = _context_object(
        "ctx.patch.autofix",
        "patch_receipt",
        "current",
        ("committed",),
        "corrupted",
        "bad_patch_injected",
        ("harmful-injection-autopatch",),
        restore_payload="candidate_patch_passed",
    )
    patch_rolled_back = _context_object(
        "ctx.patch.autofix",
        "patch_receipt",
        "rolled_back",
        ("committed", "corrupted"),
        "rolled_back",
        "candidate_patch_passed",
        ("harmful-injection-autopatch", "rollback-action"),
        restore_payload="candidate_patch_passed",
    )
    ghost = _context_object(
        "ctx.ghost.unsourced",
        "ghost_claim",
        "current",
        (),
        "ghosted",
        "claim_without_evidence",
        (),
    )
    return {
        "runtime_v1": runtime_v1,
        "runtime_v2": runtime_v2,
        "runtime_stale": runtime_stale,
        "sensor": sensor,
        "rubric": rubric,
        "rubric_corrupt": rubric_corrupt,
        "folded": folded,
        "secret": secret,
        "secret_masked": secret_masked,
        "policy": policy,
        "policy_archived": policy_archived,
        "patch_pending": patch_pending,
        "patch_committed": patch_committed,
        "patch_corrupt": patch_corrupt,
        "patch_rolled_back": patch_rolled_back,
        "ghost": ghost,
    }


def _context_object(
    object_id: str,
    object_type: str,
    current_label: str,
    historical_labels: tuple[str, ...],
    transition_label: str,
    payload: str,
    evidence_ids: tuple[str, ...],
    *,
    source_object_ids: tuple[str, ...] = (),
    restore_payload: str | None = None,
) -> JsonDict:
    return {
        "object_id": object_id,
        "object_type": object_type,
        "current_label": current_label,
        "historical_labels": historical_labels,
        "transition_label": transition_label,
        "payload": payload,
        "evidence_ids": evidence_ids,
        "recoverable_sidecar": {
            "sidecar_id": f"sidecar:{object_id}:{transition_label}",
            "recoverable": True,
            "payload_sha256": _sha256_text(restore_payload or payload),
            "restore_label": current_label,
            "restore_payload": restore_payload or payload,
            "source_object_ids": source_object_ids,
        },
    }


def _object_schema_valid(obj: Mapping[str, Any]) -> bool:
    return bool(
        obj.get("object_id")
        and obj.get("object_type") in OBJECT_TYPES
        and obj.get("current_label") in CURRENT_LABELS
        and isinstance(obj.get("historical_labels"), tuple)
        and obj.get("transition_label") in TRANSITION_LABELS
        and obj.get("recoverable_sidecar", {}).get("recoverable") is True
    )


def _copy_bank(bank: ContextBank) -> JsonDict:
    return deepcopy({str(key): dict(value) for key, value in bank.items()})


def _weight_mutation_receipt(evaluation: Mapping[str, Any]) -> JsonDict:
    return {
        "no_weight_mutation": bool(evaluation["no_weight_mutation"]),
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": ["deterministic_context_object_bank"],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "cross_model_transfer_claim": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _honest_verdict(ready: bool, evaluation: Mapping[str, Any]) -> str:
    if ready:
        return (
            "complete: deterministic context-object lifecycle fixture ready for "
            "Exp5329 and Exp5330; safe actions commit, unsafe actions reject, "
            f"rollback succeeds at {evaluation['rollback_success_rate']:.1f}, "
            "and no model weights mutate"
        )
    return "blocked_tests_or_fixture_not_ready: fixture gates or recorded tests missing"


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _checksum(payload: Mapping[str, Any]) -> str:
    stable = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    encoded = json.dumps(_json_ready(stable), sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value
