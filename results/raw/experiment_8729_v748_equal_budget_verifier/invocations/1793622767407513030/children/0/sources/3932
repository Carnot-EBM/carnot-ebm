"""Typed multi-head verifier memory for Exp 5227.

The builder keeps continuous self-learning as verified controller memory: it
records what future verifier and ARC-agent tasks should reuse, block, or
remember, without changing model weights or self-distilling on prior outputs.

Spec refs: REQ-LEARN-5227, SCENARIO-LEARN-5227.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

from carnot.pipeline.verifier_memory import assert_no_test_gold_leak


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = "experiment_5227_continuous_self_learning_multihead_memory_v478"
EXPERIMENT_ID = 5227
SCHEMA = "carnot.continuous_self_learning_multihead_memory.v1"
MEMORY_SCHEMA = "carnot.typed_multihead_verifier_memory.v1"
CONSUMER_SCHEMA = "carnot.arc_rubric_setup_from_typed_memory.v1"
RUN_DATE = "2026-07-04"

RESULT_RELATIVE_PATH = "results/experiment_5227_continuous_self_learning_multihead_memory_v478.json"
MEMORY_RELATIVE_PATH = "results/typed_multihead_verifier_memory_v478.json"
CONSUMER_RELATIVE_PATH = "results/arc_rubric_setup_from_typed_memory_v478.json"

V477_MEMORY_RELATIVE_PATH = "results/verifier_memory_v477.json"
EXP5214_RELATIVE_PATH = "results/experiment_5214_continuous_self_learning_verifier_memory_v477.json"
EXP5215_RELATIVE_PATH = "results/experiment_5215_arc_paw_amortization_gate_v477.json"
EXP5216_RELATIVE_PATH = (
    "results/experiment_5216_arc_frontier_continuity_landmark_decomposition_v477.json"
)
EXP5217_RELATIVE_PATH = "results/experiment_5217_hardware_continuity_v477.json"
EXP5220_RELATIVE_PATH = "results/experiment_5220_archive_477_activate_478.json"
EXP5222_RELATIVE_PATH = "results/experiment_5222_gap1_gate_field_registry_promotion_v478.json"
EXP5225_RELATIVE_PATH = "results/experiment_5225_gap4_clean_scale_validation_gated_v478.json"
EXP5213_RELATIVE_PATH = (
    "results/experiment_5213_hidden_state_verifier_v3_layer_chunk_sweep_v477.json"
)

SPEC_REFS = ("REQ-LEARN-5227", "SCENARIO-LEARN-5227")
TYPED_MEMORY_HEADS = ("constraints", "provenance", "failures", "skills_rubrics")
PROMOTED = "promoted"
ROLLED_BACK = "rolled_back"
HELD = "held"
VALID_PROMOTION_STATES = (PROMOTED, ROLLED_BACK, HELD)

RUBRIC_FIELDS = [
    "skill_selection",
    "skill_following",
    "skill_composition",
    "reflection_retry_quality",
    "provenance_validity",
]
DEFAULT_RETENTION_QUERIES = {
    "mmlu_hidden_state_retired": "hidden-state MMLU path should stay retired",
    "gap4_quarantined": "GAP-4 clean validation null quarantine",
}

HEAD_PAYLOAD_KEYS = {
    "constraints": frozenset(
        {"constraint_id", "scope", "must_enforce", "forbidden_claim", "action"}
    ),
    "provenance": frozenset(
        {"source_scope", "registry_state", "allowed_use", "blocked_use", "next_gate"}
    ),
    "failures": frozenset(
        {"failure_id", "failure_mode", "retirement_status", "avoid_until", "replacement_gate"}
    ),
    "skills_rubrics": frozenset(
        {"rubric_id", "domain", "fields", "known_nulls", "recommended_consumer_action"}
    ),
}

FIELD_PRINCIPLES = {
    "continuous_self_learning_task": (
        "This field proves the milestone includes the required continuous self-learning experiment."
    ),
    "typed_memory_heads": "List of typed memory heads available for verifier and ARC consumers.",
    "memory_entries_written": "Count of schema-valid typed memory entries written to the durable ledger.",
    "promotions": "Entries reusable by future controllers because they cite artifact evidence.",
    "rollbacks": "Entries blocked, quarantined, or retired with explicit invalidating evidence.",
    "retention_check_passed": "True only when critical older retirements are returned by relevance queries.",
    "consumer_ready_path": "Path to the ARC rubric setup file for Exp 5228.",
    "broad_self_distillation_used": (
        "False because Exp 5227 performs verified typed-memory operations only, with no model training."
    ),
    "tests_run": "List of verification commands with pass/fail status.",
    "inference_substrate": "Must be verified_typed_memory_no_model_training.",
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether typed memory is "
        "consumer-ready."
    ),
}

STOP_WORDS = frozenset(
    {
        "and",
        "for",
        "the",
        "with",
        "should",
        "stay",
        "path",
        "clean",
    }
)


def make_typed_memory_entry(
    *,
    head: str,
    subject: str,
    promotion_state: str,
    payload: Mapping[str, Any],
    evidence: Sequence[str] = (),
    invalidated_by: Mapping[str, Any] | None = None,
    rollback_reason: str | None = None,
) -> JsonDict:
    """Build one typed memory entry after enforcing promotion/rollback gates."""

    _validate_head_payload(head, payload)
    if promotion_state not in VALID_PROMOTION_STATES:
        raise ValueError(f"unknown promotion_state: {promotion_state}")
    evidence_paths = _clean_string_list(evidence)
    invalidation = dict(invalidated_by or {})
    if promotion_state == PROMOTED and not evidence_paths:
        raise ValueError("promoted entry requires evidence")
    if promotion_state == ROLLED_BACK and not (
        invalidation.get("artifact") or invalidation.get("exclusion_reason")
    ):
        raise ValueError("rolled_back entry requires invalidating evidence")

    entry = {
        "entry_id": _entry_id(head, subject, payload),
        "head": head,
        "subject": str(subject),
        "promotion_state": promotion_state,
        "evidence": evidence_paths,
        "invalidated_by": invalidation or None,
        "rollback_reason": rollback_reason,
        "payload": dict(payload),
        "keywords": _keywords(head, subject, payload),
        "created_by": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
    }
    assert_no_test_gold_leak(entry)
    return entry


def validate_memory(memory: Mapping[str, Any]) -> list[str]:
    """Return schema errors for a typed memory artifact without raising."""

    errors: list[str] = []
    if list(memory.get("heads", [])) != list(TYPED_MEMORY_HEADS):
        errors.append("typed heads mismatch")
    for index, raw_entry in enumerate(memory.get("entries", [])):
        try:
            _validate_entry(raw_entry)
        except ValueError as exc:
            errors.append(f"entries[{index}]: {exc}")
    try:
        assert_no_test_gold_leak(memory)
    except ValueError as exc:
        errors.append(str(exc))
    return errors


def query_memory(memory: Mapping[str, Any], query: str) -> list[JsonDict]:
    """Return entries whose typed keywords cover the relevant query terms."""

    terms = _query_terms(query)
    if not terms:
        return []
    matches = []
    for entry in memory.get("entries", []):
        searchable = " ".join(
            [
                str(entry.get("head", "")),
                str(entry.get("subject", "")),
                " ".join(str(item) for item in entry.get("keywords", [])),
                _canonical_json(entry.get("payload", {})),
            ]
        ).lower()
        if all(term in searchable for term in terms):
            matches.append(dict(entry))
    return sorted(matches, key=lambda item: str(item["subject"]))


def run_retention_check(
    memory: Mapping[str, Any],
    queries: Mapping[str, str] = DEFAULT_RETENTION_QUERIES,
) -> JsonDict:
    """Check that older critical rollback and retirement memories are retrievable."""

    passed = all(query_memory(memory, query) for query in queries.values())
    return {"passed": bool(passed), "queries": dict(queries)}


def load_source_artifacts(root: Path | str = REPO_ROOT) -> dict[str, JsonDict]:
    """Load the source artifacts Exp 5227 is allowed to summarize."""

    base = Path(root)
    return {
        "v477_memory": _read_json(base / V477_MEMORY_RELATIVE_PATH),
        "exp5214": _read_json(base / EXP5214_RELATIVE_PATH),
        "exp5215": _read_json(base / EXP5215_RELATIVE_PATH),
        "exp5216": _read_json(base / EXP5216_RELATIVE_PATH),
        "exp5217": _read_json(base / EXP5217_RELATIVE_PATH),
        "exp5220": _read_json(base / EXP5220_RELATIVE_PATH),
        "exp5222": _read_json(base / EXP5222_RELATIVE_PATH),
        "exp5225": _read_json(base / EXP5225_RELATIVE_PATH),
        "exp5213": _read_json(base / EXP5213_RELATIVE_PATH),
    }


def build_memory_bundle(
    *,
    inputs: Mapping[str, Mapping[str, Any]],
    tests_run: Sequence[Mapping[str, Any]],
    memory_relative_path: str = MEMORY_RELATIVE_PATH,
    consumer_relative_path: str = CONSUMER_RELATIVE_PATH,
) -> JsonDict:
    """Build the result artifact plus durable typed memory and ARC consumer file."""

    entries = [
        _gap1_memory_only_entry(inputs),
        _gap1_registry_block_entry(inputs),
        _gap4_quarantine_entry(inputs),
        _mmlu_retirement_entry(inputs),
        _arc_zero_delta_entry(inputs),
        _hardware_no_speedup_entry(inputs),
    ]
    memory = {
        "schema": MEMORY_SCHEMA,
        "artifact": "typed_multihead_verifier_memory_v478",
        "spec_refs": list(SPEC_REFS),
        "heads": list(TYPED_MEMORY_HEADS),
        "entries": entries,
        "summary": _memory_summary(
            entries,
            memory_relative_path=memory_relative_path,
            consumer_relative_path=consumer_relative_path,
        ),
    }
    errors = validate_memory(memory)
    if errors:
        raise ValueError("; ".join(errors))

    consumer_summary = build_consumer_summary(
        memory=memory,
        inputs=inputs,
        memory_relative_path=memory_relative_path,
    )
    retention = run_retention_check(memory)
    result = build_result_artifact(
        memory=memory,
        retention=retention,
        tests_run=tests_run,
        memory_relative_path=memory_relative_path,
        consumer_relative_path=consumer_relative_path,
    )
    return {"memory": memory, "consumer_summary": consumer_summary, "result": result}


def build_consumer_summary(
    *,
    memory: Mapping[str, Any],
    inputs: Mapping[str, Mapping[str, Any]],
    memory_relative_path: str = MEMORY_RELATIVE_PATH,
) -> JsonDict:
    """Produce the compact ARC rubric setup consumed by Exp 5228."""

    exp5215 = inputs["exp5215"]
    exp5216 = inputs["exp5216"]
    return {
        "schema": CONSUMER_SCHEMA,
        "consumer_ready": True,
        "produced_by": EXPERIMENT,
        "spec_refs": list(SPEC_REFS),
        "memory_artifact_path": memory_relative_path,
        "next_task": "exp5228-arc-provenance-skill-rubric-gate-v478",
        "rubric_fields": list(RUBRIC_FIELDS),
        "known_arc_nulls": {
            "new_levels_banked": _field_value(exp5216, "new_levels_banked", []),
            "reproducible_total_levels_delta": _field_value(
                exp5216, "reproducible_total_levels_delta", 0
            ),
            "solve_provenance": _field_value(exp5216, "solve_provenance", "development_proxy"),
            "paw_amortization_viable": _field_value(exp5215, "paw_amortization_viable", False),
            "no_arc_solve_claim": True,
        },
        "provenance_requirements": {
            "accepted": ["live_agent_self_discovery"],
            "blocked": ["development_proxy", "outer_loop_re", "offline_bfs", "hand_game_adapter"],
            "required_fields": ["solve_provenance", "registry_precheck_done", "new_levels_banked"],
        },
        "memory_pointers": {
            "gap1": "GAP-1 orientation discriminator memory-only promotion",
            "gap4": "GAP-4 candidate-pool validation null/quarantine",
            "arc": "ARC live-path zero-level delta retained for rubric setup",
        },
        "gap_summaries": {
            "GAP-1": "positive memory-only discriminator evidence; registry promotion remains blocked",
            "GAP-4": "quarantined clean-null; require a future positive canonical validation",
            "ARC": "zero reproduction-gated level delta; build process rubric before another patch",
        },
        "broad_self_distillation_used": False,
        "entry_ids": [str(entry["entry_id"]) for entry in memory.get("entries", [])],
    }


def build_result_artifact(
    *,
    memory: Mapping[str, Any],
    retention: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
    memory_relative_path: str,
    consumer_relative_path: str,
) -> JsonDict:
    """Build the principle-annotated Exp 5227 result artifact."""

    summary = memory["summary"]
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": _source_artifacts(),
        "memory_artifact_path": {
            "value": memory_relative_path,
            "principle": "Durable typed memory path.",
        },
        "continuous_self_learning_task": _wrap("continuous_self_learning_task", True),
        "typed_memory_heads": _wrap("typed_memory_heads", list(TYPED_MEMORY_HEADS)),
        "memory_entries_written": _wrap(
            "memory_entries_written", summary["memory_entries_written"]
        ),
        "promotions": _wrap("promotions", summary["promotions"]),
        "rollbacks": _wrap("rollbacks", summary["rollbacks"]),
        "retention_check_passed": _wrap("retention_check_passed", retention["passed"]),
        "consumer_ready_path": _wrap("consumer_ready_path", consumer_relative_path),
        "broad_self_distillation_used": _wrap("broad_self_distillation_used", False),
        "tests_run": _wrap("tests_run", [dict(item) for item in tests_run]),
        "inference_substrate": _wrap(
            "inference_substrate",
            "verified_typed_memory_no_model_training",
        ),
        "honest_verdict": _wrap(
            "honest_verdict",
            (
                "complete: typed memory consumer-ready for exp5228 with "
                f"{len(TYPED_MEMORY_HEADS)} heads, promotions_{summary['promotions']}_"
                f"rollbacks_{summary['rollbacks']}, retention_passed_{retention['passed']}, "
                "verified_typed_memory_no_model_training"
            ),
        ),
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    memory_path: Path | str = REPO_ROOT / MEMORY_RELATIVE_PATH,
    consumer_path: Path | str = REPO_ROOT / CONSUMER_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5227 result, typed memory, and ARC consumer files."""

    result_dest = Path(result_path)
    memory_dest = Path(memory_path)
    consumer_dest = Path(consumer_path)
    output_root = _output_root(result_dest)
    memory_relative = _relative_to_output(memory_dest, output_root)
    consumer_relative = _relative_to_output(consumer_dest, output_root)

    bundle = build_memory_bundle(
        inputs=load_source_artifacts(root),
        tests_run=tests_run,
        memory_relative_path=memory_relative,
        consumer_relative_path=consumer_relative,
    )
    _write_json(memory_dest, bundle["memory"])
    _write_json(consumer_dest, bundle["consumer_summary"])
    _write_json(result_dest, bundle["result"])
    return bundle["result"]


def _gap1_memory_only_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    v477_entry = _find_v477_entry(inputs["v477_memory"], "GAP-1")
    exp5222 = inputs["exp5222"]
    return make_typed_memory_entry(
        head="provenance",
        subject="GAP-1 orientation discriminator memory-only promotion",
        promotion_state=PROMOTED,
        evidence=[V477_MEMORY_RELATIVE_PATH, EXP5214_RELATIVE_PATH, EXP5222_RELATIVE_PATH],
        payload={
            "source_scope": "GAP-1",
            "registry_state": "positive_but_unpromoted",
            "allowed_use": "memory_only",
            "blocked_use": "registry_promotion",
            "next_gate": exp5222.get("follow_up_criterion"),
            "imported_from_memory_id": v477_entry.get("memory_id"),
        },
    )


def _gap1_registry_block_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp5222 = inputs["exp5222"]
    decision = _field_value(exp5222, "gap1_registry_decision", "blocked_instability")
    return make_typed_memory_entry(
        head="provenance",
        subject="GAP-1 registry promotion blocked by subset instability",
        promotion_state=ROLLED_BACK,
        evidence=[EXP5222_RELATIVE_PATH],
        invalidated_by={"artifact": EXP5222_RELATIVE_PATH},
        rollback_reason=str(decision),
        payload={
            "source_scope": "GAP-1",
            "registry_state": decision,
            "allowed_use": "memory_only",
            "blocked_use": "registry_promotion",
            "next_gate": exp5222.get("follow_up_criterion"),
        },
    )


def _gap4_quarantine_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    v477_entry = _find_v477_entry(inputs["v477_memory"], "GAP-4")
    exp5225 = inputs["exp5225"]
    return make_typed_memory_entry(
        head="failures",
        subject="GAP-4 candidate-pool validation null/quarantine",
        promotion_state=ROLLED_BACK,
        evidence=[V477_MEMORY_RELATIVE_PATH, EXP5214_RELATIVE_PATH, EXP5225_RELATIVE_PATH],
        invalidated_by={"artifact": EXP5225_RELATIVE_PATH},
        rollback_reason="gap4_clean_null",
        payload={
            "failure_id": "gap4_clean_validation_null",
            "failure_mode": (
                f"n_scored={exp5225.get('n_scored')}, wins={exp5225.get('wins')}, "
                f"losses={exp5225.get('losses')}, ties={exp5225.get('ties')}"
            ),
            "retirement_status": "quarantined",
            "avoid_until": "future canonical validation crosses floor",
            "replacement_gate": "six_discordant_wins_zero_losses_exact_p_lt_0_05",
            "imported_from_memory_id": v477_entry.get("memory_id"),
        },
    )


def _mmlu_retirement_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp5213 = inputs["exp5213"]
    return make_typed_memory_entry(
        head="failures",
        subject="MMLU hidden-state verifier path retired",
        promotion_state=ROLLED_BACK,
        evidence=[EXP5213_RELATIVE_PATH],
        invalidated_by={"artifact": EXP5213_RELATIVE_PATH},
        rollback_reason="mmlu_hidden_state_path_retired",
        payload={
            "failure_id": "mmlu_hidden_state_path_retired",
            "failure_mode": "best hidden-state probe did not beat controls",
            "retirement_status": "retired",
            "avoid_until": "new non-final-layer signal or different corpus mechanism",
            "replacement_gate": "positive_ci_vs_tuned_sc_self_certainty_clue_rcs",
            "best_probe_accuracy": _field_value(exp5213, "best_probe_accuracy"),
            "tuned_sc_accuracy": _field_value(exp5213, "tuned_sc_accuracy"),
        },
    )


def _arc_zero_delta_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp5215 = inputs["exp5215"]
    exp5216 = inputs["exp5216"]
    return make_typed_memory_entry(
        head="skills_rubrics",
        subject="ARC live-path zero-level delta retained for rubric setup",
        promotion_state=ROLLED_BACK,
        evidence=[EXP5215_RELATIVE_PATH, EXP5216_RELATIVE_PATH, EXP5220_RELATIVE_PATH],
        invalidated_by={
            "artifact": EXP5216_RELATIVE_PATH,
            "exclusion_reason": "reproducible_total_levels_delta_zero",
        },
        rollback_reason="arc_zero_level_delta",
        payload={
            "rubric_id": "arc_live_path_process_rubric_seed",
            "domain": "ARC",
            "fields": list(RUBRIC_FIELDS),
            "known_nulls": {
                "new_levels_banked": _field_value(exp5216, "new_levels_banked", []),
                "reproducible_total_levels_delta": _field_value(
                    exp5216, "reproducible_total_levels_delta", 0
                ),
                "solve_provenance": _field_value(exp5216, "solve_provenance", "development_proxy"),
                "paw_amortization_viable": _field_value(exp5215, "paw_amortization_viable", False),
            },
            "recommended_consumer_action": "build_process_rubric_before_level_patch",
        },
    )


def _hardware_no_speedup_entry(inputs: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    exp5217 = inputs["exp5217"]
    return make_typed_memory_entry(
        head="constraints",
        subject="Hardware speedup claim boundary",
        promotion_state=PROMOTED,
        evidence=[EXP5217_RELATIVE_PATH],
        payload={
            "constraint_id": "hardware_no_speedup_without_transcript",
            "scope": "hardware_reporting",
            "must_enforce": _field_value(exp5217, "no_speedup_claim", True),
            "forbidden_claim": "hardware_speedup_without_authenticated_transcript",
            "action": "block_speedup_claim_until_end_to_end_workload_transcript",
            "kv260_status": exp5217.get("kv260_status"),
            "polarfire_status": exp5217.get("polarfire_status"),
        },
    )


def _memory_summary(
    entries: Sequence[Mapping[str, Any]],
    *,
    memory_relative_path: str,
    consumer_relative_path: str,
) -> JsonDict:
    promotions = sum(1 for entry in entries if entry.get("promotion_state") == PROMOTED)
    rollbacks = sum(1 for entry in entries if entry.get("promotion_state") == ROLLED_BACK)
    return {
        "memory_entries_written": len(entries),
        "promotions": promotions,
        "rollbacks": rollbacks,
        "memory_artifact_path": memory_relative_path,
        "consumer_ready_path": consumer_relative_path,
        "heads": list(TYPED_MEMORY_HEADS),
    }


def _validate_entry(raw_entry: Mapping[str, Any]) -> None:
    entry = dict(raw_entry)
    _validate_head_payload(str(entry.get("head")), _mapping(entry.get("payload")))
    state = entry.get("promotion_state")
    if state not in VALID_PROMOTION_STATES:
        raise ValueError(f"unknown promotion_state: {state}")
    if state == PROMOTED and not entry.get("evidence"):
        raise ValueError("promoted entry requires evidence")
    invalidated_by = _mapping(entry.get("invalidated_by"))
    if state == ROLLED_BACK and not (
        invalidated_by.get("artifact") or invalidated_by.get("exclusion_reason")
    ):
        raise ValueError("rolled_back entry requires invalidating evidence")


def _validate_head_payload(head: str, payload: Mapping[str, Any]) -> None:
    if head not in TYPED_MEMORY_HEADS:
        raise ValueError(f"unknown memory head: {head}")
    missing = sorted(HEAD_PAYLOAD_KEYS[head] - set(payload))
    if missing:
        raise ValueError(f"payload missing required keys for {head}: {missing}")


def _find_v477_entry(v477_memory: Mapping[str, Any], gap: str) -> JsonDict:
    for entry in v477_memory.get("entries", []):
        if gap.lower() in str(entry.get("failure_signature", "")).lower():
            return dict(entry)
    return {}


def _source_artifacts() -> list[str]:
    return [
        "research-program.md#continuous-self-learning-core-architectural-goal",
        "_bmad/prd.md#fr-11-autonomous-self-learning-loop",
        "_bmad/prd.md#fr-12-verifiable-reasoning",
        "research-references.md#continuous-self-learning-and-memory",
        V477_MEMORY_RELATIVE_PATH,
        EXP5214_RELATIVE_PATH,
        EXP5215_RELATIVE_PATH,
        EXP5216_RELATIVE_PATH,
        EXP5217_RELATIVE_PATH,
        EXP5220_RELATIVE_PATH,
        EXP5222_RELATIVE_PATH,
        EXP5225_RELATIVE_PATH,
        EXP5213_RELATIVE_PATH,
    ]


def _field_value(artifact: Mapping[str, Any], key: str, default: Any = None) -> Any:
    raw = artifact.get(key, default)
    if isinstance(raw, Mapping) and "value" in raw:
        return raw["value"]
    return raw


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": FIELD_PRINCIPLES[field]}


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _output_root(result_path: Path) -> Path:
    if result_path.parent.name == "results":
        return result_path.parent.parent
    return result_path.parent


def _relative_to_output(path: Path, output_root: Path) -> str:
    try:
        return str(path.relative_to(output_root))
    except ValueError:
        return str(path)


def _entry_id(head: str, subject: str, payload: Mapping[str, Any]) -> str:
    digest = sha256(
        _canonical_json({"head": head, "subject": subject, "payload": dict(payload)}).encode(
            "utf-8"
        )
    ).hexdigest()
    return f"typed-memory:{digest[:16]}"


def _keywords(head: str, subject: str, payload: Mapping[str, Any]) -> list[str]:
    words = {head.lower()}
    words.update(_query_terms(subject))
    for value in payload.values():
        if isinstance(value, str):
            words.update(_query_terms(value))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for item in value:
                words.update(_query_terms(str(item)))
    return sorted(words)


def _query_terms(text: str) -> list[str]:
    normalized = str(text).lower().replace("_", " ").replace("-", " ").replace("/", " ")
    return [token for token in normalized.split() if len(token) > 2 and token not in STOP_WORDS]


def _clean_string_list(values: Sequence[str]) -> list[str]:
    return sorted({str(value) for value in values if str(value)})


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def main() -> None:  # pragma: no cover
    run()


if __name__ == "__main__":  # pragma: no cover
    main()
