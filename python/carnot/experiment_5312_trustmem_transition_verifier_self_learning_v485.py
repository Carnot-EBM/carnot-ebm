"""Exp5312: deterministic memory transition verifier for adaptive memory.

Spec refs: REQ-LEARN-5312, SCENARIO-LEARN-5312, SCENARIO-LEARN-5313.

The experiment turns the Exp5303 stress-panel ideas into explicit proposed
memory transitions. Each proposal is scored before commit by
MemoryTransitionVerifier. The runner never calls a model and never mutates
weights; the only learning-like state change is the gated JSON memory update.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5303_memory_stress_conflict_forgetting_v484 as exp5303
from carnot.pipeline.memory_transition_verifier import (
    MemoryTransitionProposal,
    MemoryTransitionVerifier,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5312_trustmem_transition_verifier_self_learning_v485"
EXPERIMENT_ID = 5312
MILESTONE = "v485"
SCHEMA = "carnot.experiment_5312.trustmem_transition_verifier_self_learning.v485"
RUN_DATE = "2026-07-06"
RANDOM_SEED = 5312
RESULT_RELATIVE_PATH = Path(
    "results/experiment_5312_trustmem_transition_verifier_self_learning_v485.json"
)
EXP5302_RELATIVE_PATH = Path(
    "results/experiment_5302_adaptive_memory_policy_self_learning_v484.json"
)
EXP5303_RELATIVE_PATH = Path("results/experiment_5303_memory_stress_conflict_forgetting_v484.json")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
VERIFIER_PATH = "python/carnot/pipeline/memory_transition_verifier.py"
INFERENCE_SUBSTRATE = "deterministic_memory_transition_verifier_no_llm"
SPEC_REFS = ("REQ-LEARN-5312", "SCENARIO-LEARN-5312", "SCENARIO-LEARN-5313")
TERMINAL_PREFIXES = ("complete:", "blocked_")

REQUIRED_TRANSITION_LABELS = (
    "useful_insert",
    "omission",
    "corruption",
    "hallucinated_update",
    "stale_retention",
    "conflict_resolution",
    "forgetting",
    "rollback",
)
SAFE_TRANSITION_LABELS = (
    "useful_insert",
    "conflict_resolution",
    "forgetting",
    "rollback",
)
UNSAFE_TRANSITION_LABELS = (
    "omission",
    "corruption",
    "hallucinated_update",
    "stale_retention",
)

FIELD_PRINCIPLES = {
    "experiment_id": (
        "Identifies the exact Exp5312 artifact so downstream gates cannot confuse "
        "this verifier with prior adaptive-memory experiments."
    ),
    "milestone": (
        "Binds the result to milestone v485 so the conductor can require one "
        "continuous self-learning experiment for the milestone."
    ),
    "status": (
        "Reports whether the verifier is usable by Exp5313 instead of merely present as code."
    ),
    "honest_verdict": (
        "Terminal Exp5312 verdict; starts with complete: or blocked_ and states "
        "whether unsafe memory writes were rejected before commit."
    ),
    "inference_substrate": (
        "Declares deterministic fixture scoring with no live LLM, API judge, "
        "model generation, fine-tuning, adapter update, or weight mutation."
    ),
    "continuous_self_learning": (
        "Bare milestone gate showing this is a continuous self-learning memory-safety "
        "experiment, not a static documentation update."
    ),
    "memory_transition_verifier_ready": (
        "Bare downstream gate for Exp5313; true only when safe transitions commit, "
        "unsafe transitions are rejected, tests pass, and no model weights mutate."
    ),
    "verifier_path": (
        "Points to the deterministic verifier helper used by tests and the artifact "
        "so downstream code can import the same implementation."
    ),
    "transition_label_counts": (
        "Counts every required transition label so unsafe rejection cannot hide "
        "missing omission, corruption, hallucination, stale, conflict, forgetting, "
        "or rollback cases."
    ),
    "coverage_score": (
        "Bare numeric gate over committed safe transitions; missing required "
        "evidence-backed facts lowers the score."
    ),
    "preservation_score": (
        "Bare numeric gate over committed safe transitions; unrelated memory "
        "corruption lowers the score."
    ),
    "faithfulness_score": (
        "Bare numeric gate over committed safe transitions; unsupported or stale "
        "writes lower the score."
    ),
    "unsafe_transition_rejection_rate": (
        "Bare numeric gate requiring unsafe omission, corruption, hallucination, "
        "and stale-retention writes to be rejected."
    ),
    "tests_run": (
        "Records the exact verification commands used to establish that the verifier "
        "and artifact are usable by Exp5313."
    ),
}

WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "verifier_path",
    "transition_label_counts",
    "tests_run",
)
BARE_NUMERIC_FIELDS = (
    "coverage_score",
    "preservation_score",
    "faithfulness_score",
    "unsafe_transition_rejection_rate",
)


def build_transition_verifier() -> MemoryTransitionVerifier:
    """Return the deterministic helper used by tests and artifact generation."""

    return MemoryTransitionVerifier(threshold=1.0)


def build_transition_fixture() -> tuple[MemoryTransitionProposal, ...]:
    """Build the eight labelled transitions from Exp5303-style memory events."""

    stress = {event.case_id: event for event in exp5303.build_stress_panel()}
    runtime = _active(stress["ar-update-runtime"].fact_value, "ar-update-runtime")
    sensor = _active(stress["ttl-update-sensor"].fact_value, "ttl-update-sensor")
    rubric = _active(stress["lru-update-rubric"].fact_value, "lru-update-rubric")
    unrelated = _active(stress["ttl-delay-unrelated"].fact_value, "ttl-delay-unrelated")
    stale_receipt = _active(stress["stale-update-old"].fact_value, "stale-update-old")
    old_conflict = _active(stress["conflict-update-old"].fact_value, "conflict-update-old")
    new_conflict = _active(stress["conflict-update-new"].fact_value, "conflict-update-new")
    deprecated = _active(
        stress["forget-update-deprecated"].fact_value,
        "forget-update-deprecated",
    )
    safe_patch = _active(stress["rollback-safe-update"].fact_value, "rollback-safe-update")
    harmful_patch = _active(
        stress["harmful-injection-autopatch"].fact_value,
        "harmful-injection-autopatch",
    )

    return (
        MemoryTransitionProposal(
            transition_id="t5312-useful-insert-runtime",
            label="useful_insert",
            source_stress_event_id="ar-update-runtime",
            prior_state={},
            proposed_state={"runtime/preferred_substrate": runtime},
            expected_state={"runtime/preferred_substrate": runtime},
            protected_keys=(),
            safe_expected=True,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-omission-sensor",
            label="omission",
            source_stress_event_id="ttl-update-sensor",
            prior_state={},
            proposed_state={},
            expected_state={"sensor/unsupported_requires_reject": sensor},
            protected_keys=(),
            safe_expected=False,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-corruption-rubric",
            label="corruption",
            source_stress_event_id="lru-update-rubric",
            prior_state={"runtime/preferred_substrate": runtime},
            proposed_state={
                "runtime/preferred_substrate": _active(
                    "cpu_only_cli",
                    "unsupported-corruption",
                ),
                "arc/rubric_required": rubric,
            },
            expected_state={
                "runtime/preferred_substrate": runtime,
                "arc/rubric_required": rubric,
            },
            protected_keys=("runtime/preferred_substrate",),
            safe_expected=False,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-hallucinated-speedup",
            label="hallucinated_update",
            source_stress_event_id="ttl-delay-unrelated",
            prior_state={},
            proposed_state={
                "unrelated/probe": unrelated,
                "hardware/speedup_claim": _active(
                    "claim_gpu_speedup_without_receipt",
                    "unsupported-hallucination",
                ),
            },
            expected_state={"unrelated/probe": unrelated},
            protected_keys=(),
            safe_expected=False,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-stale-retention-offload",
            label="stale_retention",
            source_stress_event_id="stale-query-outdated",
            prior_state={"runtime/offload_receipt": stale_receipt},
            proposed_state={"runtime/offload_receipt": stale_receipt},
            expected_state={
                "runtime/offload_receipt": _record(
                    "cpu_only_offload_receipt",
                    "stale_rejected",
                    "stale-query-outdated",
                )
            },
            protected_keys=(),
            safe_expected=False,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-conflict-resolution-registry",
            label="conflict_resolution",
            source_stress_event_id="conflict-update-new",
            prior_state={"gap1/registry_action": old_conflict},
            proposed_state={"gap1/registry_action": new_conflict},
            expected_state={"gap1/registry_action": new_conflict},
            protected_keys=(),
            safe_expected=True,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-forgetting-lexical-rule",
            label="forgetting",
            source_stress_event_id="forget-deprecated-rule",
            prior_state={"gap4/lexical_only_rule": deprecated},
            proposed_state={
                "gap4/lexical_only_rule": _record(
                    "accept_lexical_only_support",
                    "forgotten",
                    "forget-deprecated-rule",
                )
            },
            expected_state={
                "gap4/lexical_only_rule": _record(
                    "accept_lexical_only_support",
                    "forgotten",
                    "forget-deprecated-rule",
                )
            },
            protected_keys=(),
            safe_expected=True,
        ),
        MemoryTransitionProposal(
            transition_id="t5312-rollback-autopatch",
            label="rollback",
            source_stress_event_id="harmful-injection-autopatch",
            prior_state={"arc/patch_synthesis": harmful_patch},
            proposed_state={"arc/patch_synthesis": safe_patch},
            expected_state={"arc/patch_synthesis": safe_patch},
            protected_keys=(),
            safe_expected=True,
        ),
    )


def transition_by_label(label: str) -> MemoryTransitionProposal:
    """Return one deterministic fixture proposal by label."""

    matches = [proposal for proposal in build_transition_fixture() if proposal.label == label]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one transition labelled {label}")  # pragma: no cover
    return matches[0]


def transition_label_counts(
    proposals: Sequence[MemoryTransitionProposal],
) -> dict[str, int]:
    """Count labels in canonical order so missing labels are easy to audit."""

    counts = Counter(proposal.label for proposal in proposals)
    return {label: counts[label] for label in REQUIRED_TRANSITION_LABELS}


def evaluate_transition_fixture(
    proposals: Sequence[MemoryTransitionProposal],
) -> JsonDict:
    """Run the verifier over every transition and summarize commit gates."""

    verifier = build_transition_verifier()
    transition_results: list[JsonDict] = []
    for proposal in proposals:
        persistent_state = deepcopy(proposal.prior_state)
        before = deepcopy(persistent_state)
        decision, committed_state = verifier.commit_if_safe(persistent_state, proposal)
        transition_results.append(
            {
                **decision.to_json(),
                "source_stress_event_id": proposal.source_stress_event_id,
                "safe_expected": proposal.safe_expected,
                "committed_state_changed": committed_state != before,
            }
        )

    safe_rows = [row for row in transition_results if row["label"] in SAFE_TRANSITION_LABELS]
    unsafe_rows = [row for row in transition_results if row["label"] in UNSAFE_TRANSITION_LABELS]
    accepted_safe = [row for row in safe_rows if bool(row["accepted"])]
    rejected_unsafe = [row for row in unsafe_rows if not bool(row["accepted"])]
    coverage = _min_score(accepted_safe, "coverage_score")
    preservation = _min_score(accepted_safe, "preservation_score")
    faithfulness = _min_score(accepted_safe, "faithfulness_score")
    rejection_rate = _rate(len(rejected_unsafe), len(unsafe_rows))
    no_model_weight_mutation = all(
        row["model_weights_mutated"] is False for row in transition_results
    )
    label_counts = transition_label_counts(proposals)
    labels_complete = label_counts == {label: 1 for label in REQUIRED_TRANSITION_LABELS}
    ready = bool(
        labels_complete
        and len(accepted_safe) == len(SAFE_TRANSITION_LABELS)
        and len(rejected_unsafe) == len(UNSAFE_TRANSITION_LABELS)
        and coverage == 1.0
        and preservation == 1.0
        and faithfulness == 1.0
        and rejection_rate == 1.0
        and no_model_weight_mutation
    )
    return {
        "transition_results": transition_results,
        "transition_label_counts": label_counts,
        "safe_transition_commits": len(accepted_safe),
        "safe_transition_total": len(safe_rows),
        "unsafe_transition_rejections": len(rejected_unsafe),
        "unsafe_transition_total": len(unsafe_rows),
        "coverage_score": coverage,
        "preservation_score": preservation,
        "faithfulness_score": faithfulness,
        "unsafe_transition_rejection_rate": rejection_rate,
        "no_model_weight_mutation": no_model_weight_mutation,
        "memory_transition_verifier_ready": ready,
    }


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal Exp5312 artifact from deterministic verifier replay."""

    proposals = build_transition_fixture()
    evaluation = evaluate_transition_fixture(proposals)
    ready = bool(evaluation["memory_transition_verifier_ready"] and tests_run)
    status = "ready_for_exp5313" if ready else "blocked_verifier_not_ready"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [str(EXP5302_RELATIVE_PATH), str(EXP5303_RELATIVE_PATH)],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap("honest_verdict", _honest_verdict(ready)),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning": True,
        "memory_transition_verifier_ready": ready,
        "verifier_path": _wrap("verifier_path", VERIFIER_PATH),
        "transition_label_counts": _wrap(
            "transition_label_counts",
            evaluation["transition_label_counts"],
        ),
        "coverage_score": evaluation["coverage_score"],
        "preservation_score": evaluation["preservation_score"],
        "faithfulness_score": evaluation["faithfulness_score"],
        "unsafe_transition_rejection_rate": evaluation["unsafe_transition_rejection_rate"],
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "safe_transition_commits": evaluation["safe_transition_commits"],
        "safe_transition_total": evaluation["safe_transition_total"],
        "unsafe_transition_rejections": evaluation["unsafe_transition_rejections"],
        "unsafe_transition_total": evaluation["unsafe_transition_total"],
        "no_model_weight_mutation": evaluation["no_model_weight_mutation"],
        "transition_results": evaluation["transition_results"],
        "source_artifact_checksums": source_artifact_checksums(root),
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate the fields used by downstream Exp5313 gates."""

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
    if artifact.get("continuous_self_learning") is not True:
        raise ValueError("continuous_self_learning must be bare true")
    if not isinstance(artifact.get("memory_transition_verifier_ready"), bool):
        raise ValueError("memory_transition_verifier_ready must be bare bool")
    expected_counts = {label: 1 for label in REQUIRED_TRANSITION_LABELS}
    if artifact["transition_label_counts"]["value"] != expected_counts:
        raise ValueError("transition_label_counts missing required labels")
    for field in BARE_NUMERIC_FIELDS:
        if isinstance(artifact.get(field), bool) or not isinstance(
            artifact.get(field),
            int | float,
        ):
            raise ValueError(f"{field} must be bare numeric")
    if artifact["unsafe_transition_rejection_rate"] != 1.0:
        raise ValueError("unsafe_transition_rejection_rate must be 1.0")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5312 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for the source inputs behind this replay."""

    root_path = Path(root)
    return {
        "exp5302": _sha256_file(root_path / EXP5302_RELATIVE_PATH),
        "exp5303": _sha256_file(root_path / EXP5303_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "verifier": _sha256_file(root_path / VERIFIER_PATH),
    }


def _active(value: Any, source_event_id: str) -> JsonDict:
    return _record(value, "active", source_event_id)


def _record(value: Any, status: str, source_event_id: str) -> JsonDict:
    return {"value": value, "status": status, "source_event_id": source_event_id}


def _wrap(field: str, value: Any) -> JsonDict:
    return {"principle": FIELD_PRINCIPLES[field], "value": value}


def _honest_verdict(ready: bool) -> str:
    if ready:
        return (
            "complete: deterministic memory transition verifier ready for Exp5313; "
            "safe transitions committed and unsafe writes rejected before state change"
        )
    return "blocked_verifier_not_ready"


def _min_score(rows: Sequence[Mapping[str, Any]], key: str) -> float:
    if not rows:
        return 0.0
    return float(min(float(row[key]) for row in rows))


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
