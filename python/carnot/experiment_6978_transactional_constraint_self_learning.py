"""Run chronological transactional constraint-policy self-learning.

The live model receives a fresh context for each arm-event call. Exact outcome
evaluation stays outside generation. Only the transactional arm can write, and
it can write only after the current exact certificate becomes durable.

Spec refs: REQ-LEARN-6978 and SCENARIO-LEARN-6978-*.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.inference.sota_models import cached_sota_pair
from carnot.learning.constraint_policy_store import (
    ForcedInterruption,
    PolicyStore,
    canonical_bytes,
    sha256_bytes,
)
from carnot.learning.continual_constraint_memory import (
    ARMS,
    CONSTRAINT_IR_SCHEMA,
    build_prompt,
    build_update_proposal,
    build_writer_input,
    classify_error,
    find_forbidden_paths,
    initial_memory_records,
    lookup_memory,
    replay_safety_score,
    visible_predecessors,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO_ROOT / "results/experiment_6978_transactional_constraint_self_learning.json"
TRANSACTION_ROOT = (
    REPO_ROOT / "results/checkpoints/experiment_6978_transactional_constraint_self_learning"
)
SOURCE_PATHS = {
    "lease": REPO_ROOT / "results/experiment_6973_lease_aware_gguf_runtime.json",
    "admissibility": REPO_ROOT / "results/experiment_6974_claim_provenance_duration_lint.json",
    "selection": REPO_ROOT / "results/experiment_6976_exact_candidate_certification.json",
    "fixture": REPO_ROOT / "results/experiment_6967_certified_error_headroom_fixture.json",
}
EXPERIMENT_ID = 6978
SCHEMA_VERSION = "carnot.exp6978.transactional_constraint_self_learning.v1"
RUN_DATE = "20260904"
RANDOM_SEED = 6_978_202_609_04
INFERENCE_SUBSTRATE = "live_local_qwen36_transactional_chronological_constraint_learning"
QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
EXPECTED_EXP6967_SHA256 = "sha256:1685ad1bff1b82aae3a17f80d341e0593d99879809bb9afb3268060100e54fee"
TOKEN_CAP = 128
HELD_FUTURE_START = 6
MEMORY_LOOKUP_LIMIT = 4
MAX_MEMORY_STATE_BYTES = 8_192
ROLLBACK_THRESHOLD = 0.0
TEMPERATURE = 0.35
TOP_P = 0.9
TOP_K = 40
REPEAT_PENALTY = 1.05

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment_id",
    "run_date",
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "live_duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes",
    "stream_hash",
    "event_order_hash",
    "arm_config_rows",
    "budget_rows",
    "rows",
    "per_event_results",
    "prompt_visibility_rows",
    "exact_outcome_rows",
    "memory_lookup_rows",
    "update_proposal_rows",
    "transaction_journal_rows",
    "commit_rows",
    "rollback_rows",
    "restart_recovery_rows",
    "held_future_rows",
    "chronological_gain_over_readonly",
    "plasticity_score",
    "stability_score",
    "max_forgetting",
    "memory_state_bytes",
    "leakage_check_rows",
    "self_learning_run_complete_score",
    "transactional_learning_positive_score",
    "checkpoint_rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "schema": "A versioned schema lets a cold audit reject incompatible rows.",
    "experiment_id": "A fixed task identity prevents evidence from another run entering the result.",
    "run_date": "The execution date distinguishes this protocol run from earlier blocked attempts.",
    "field_principles": "A reason for each field makes the scientific contract reviewable.",
    "preconditions_checked": "Explicit gates stop live calls when required evidence or tools are absent.",
    "inference_substrate": "The substrate declares local Qwen generation plus external policy state.",
    "duration_s": "Total wall time exposes incomplete or synthetic execution.",
    "live_duration_s": "Model-call time separates live generation from exact checking and storage.",
    "source_artifact_hashes": "Hashes bind the run to exact upstream evidence bytes.",
    "MODEL_SPECS": "The exact model declaration prevents a legacy smoke model entering comparisons.",
    "models_used": "The observed roster proves which model produced the compared rows.",
    "model_file_hashes": "Weight hashes detect silent model-byte changes during the run.",
    "stream_hash": "One digest binds the selected chronological fixture stream.",
    "event_order_hash": "The order digest detects event reordering that could change learning credit.",
    "arm_config_rows": "Arm contracts make read and write authority explicit before inference.",
    "budget_rows": "Matched assigned budgets prevent compute differences from posing as learning.",
    "rows": "All 72 terminal units are the authority for aggregate claims.",
    "per_event_results": "Paired event summaries expose wins, losses, and ties without pooling.",
    "prompt_visibility_rows": "Visibility receipts prove each prompt uses predecessor-only evidence.",
    "exact_outcome_rows": "Independent exact outcomes keep model self-reports outside authority.",
    "memory_lookup_rows": "Lookup receipts show which earlier policy records could affect each call.",
    "update_proposal_rows": "Proposal rows show that writes use only allowed post-outcome fields.",
    "transaction_journal_rows": "Hash-linked phases prove prepare became durable before commit.",
    "commit_rows": "Commit receipts bind each active update to parent and new state hashes.",
    "rollback_rows": "Rollback receipts prove harmful state can return to exact parent bytes.",
    "restart_recovery_rows": "Fresh-store receipts prove interrupted prepares and final state recover.",
    "held_future_rows": "Held-future pairs measure later utility after the six-event learning prefix.",
    "chronological_gain_over_readonly": "Paired later successes measure learning rather than write volume.",
    "plasticity_score": "Recurring-error recovery measures adaptation to newly observed failures.",
    "stability_score": "Retention of read-only successes measures resistance to harmful change.",
    "max_forgetting": "The largest cumulative lost-success count exposes catastrophic forgetting.",
    "memory_state_bytes": "Exact bytes enforce the preregistered small-state budget.",
    "leakage_check_rows": "Leakage checks keep confidence and later outcomes outside prompts and writes.",
    "self_learning_run_complete_score": "One requires all rows, matched budgets, replay, and leakage checks.",
    "transactional_learning_positive_score": "One requires two later gains, no losses, rollback, and bounded state.",
    "checkpoint_rows": "Durable stage receipts prove prompt and raw bytes precede outcome access.",
    "random_seed": "A fixed seed sequence makes matched stochastic calls addressable.",
    "reproducibility_checksum": "A timing-free digest detects drift in plans, outputs, and reductions.",
    "gate_check_summary": "Expected and observed values make any blocked prerequisite actionable.",
    "verifier_is_oracle": "False states that the memory policy is distinct from the exact evaluator.",
    "verdict_class": "A closed class prevents a completed null from being reported as positive.",
    "honest_verdict": "A class-consistent terminal prefix gives downstream automation a stable state.",
}

GenerateFn = Callable[[JsonDict, str], JsonDict]
EvaluateFn = Callable[[str, JsonDict, JsonDict, str], JsonDict]


def _canonical_json(value: Any) -> str:
    """Return stable compact JSON for plans and command output."""

    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_json(value: Any) -> str:
    """Hash one JSON value with the policy store's canonical encoding."""

    return sha256_bytes(canonical_bytes(value))


def _sha256_path(path: Path) -> str | None:
    """Hash a source or model file without loading it into memory."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def _write_json_atomic(path: Path, value: Any) -> None:
    """Publish complete evidence bytes through one filesystem rename."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(value, indent=2, sort_keys=True).encode("utf-8") + b"\n")
        handle.flush()
        import os

        os.fsync(handle.fileno())
    temporary.replace(path)


def _gate_check(check: str, expected: Any, observed: Any, passed: bool | None = None) -> JsonDict:
    """Record one prerequisite with explicit expected and observed values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def _gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Keep every prerequisite and identify the first failed one."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "checks": rows,
        "passed": failed is None,
    }


def resolve_model_specs(
    resolver: Callable[..., list[dict[str, Any]] | None] = cached_sota_pair,
) -> list[JsonDict]:
    """Resolve the mandated cached pair and retain only its exact Qwen row."""

    pair = resolver(gpu_indices=(0, 1)) or []
    qwen = next((dict(row) for row in pair if row.get("hf_id") == QWEN_ID), None)
    if qwen is None:
        return [
            {
                "name": "Qwen3.6-35B-A3B",
                "hf_id": QWEN_ID,
                "model_path": "",
                "gpu_indices": [0, 1],
                "headline_eligible": True,
                "resolution_method": "cached_sota_pair(gpu_indices=(0, 1))",
            }
        ]
    qwen.pop("gpu", None)
    qwen.update(
        {
            "gpu_indices": [0, 1],
            "headline_eligible": True,
            "resolution_method": "cached_sota_pair(gpu_indices=(0, 1))",
        }
    )
    return [qwen]


MODEL_SPECS = resolve_model_specs()


def _stream_observation(fixture: Mapping[str, Any]) -> JsonDict:
    """Recompute event, prompt, witness, and predecessor integrity."""

    events = sorted(
        (dict(row) for row in fixture.get("chronological_event_rows", [])),
        key=lambda row: row.get("event_ordinal", -1),
    )
    prompts = {
        str(row.get("pair_id")): row
        for row in fixture.get("prompt_visible_rows", [])
        if row.get("split") == "chronological"
    }
    witnesses = {
        str(row.get("pair_id")): row
        for row in fixture.get("exact_witness_rows", [])
        if row.get("subject_kind") == "slice_pair" and row.get("split") == "chronological"
    }
    chain_ok = True
    for ordinal, event in enumerate(events):
        predecessor = events[ordinal - 1] if ordinal else None
        chain_ok = chain_ok and event.get("event_ordinal") == ordinal
        chain_ok = chain_ok and event.get("predecessor_event_id") == (
            predecessor.get("event_id") if predecessor else None
        )
        if predecessor is not None:
            chain_ok = chain_ok and event.get("predecessor_record_hash") == predecessor.get(
                "event_hash"
            )
        prompt = prompts.get(str(event.get("pair_id")), {})
        chain_ok = chain_ok and prompt.get("prompt_record_hash") == event.get("prompt_record_hash")
    return {
        "event_count": len(events),
        "prompt_count": len(prompts),
        "witness_count": len(witnesses),
        "unique_event_ids": len({row.get("event_id") for row in events}),
        "unique_pair_ids": len({row.get("pair_id") for row in events}),
        "all_later_outcome_flags_false": all(
            row.get("later_outcome_exists") is False for row in events
        ),
        "chain_ok": chain_ok,
    }


def collect_preconditions(
    *,
    lease_artifact: Mapping[str, Any],
    admissibility_artifact: Mapping[str, Any],
    selection_artifact: Mapping[str, Any],
    fixture_artifact: Mapping[str, Any],
    source_paths: Mapping[str, Path],
    fixture_hash: str | None,
    model_spec: Mapping[str, Any],
    transaction_root: Path,
    z3_available: bool,
) -> tuple[list[JsonDict], JsonDict]:
    """Check all external resources before any model call."""

    source_hashes = {name: _sha256_path(Path(path)) for name, path in source_paths.items()}
    stream = _stream_observation(fixture_artifact)
    time_zero = {
        "public_pair": next(
            (
                dict(row)
                for row in fixture_artifact.get("prompt_visible_rows", [])
                if row.get("split") == "chronological"
            ),
            {},
        ),
        "memory": initial_memory_records(),
    }
    model_path = Path(str(model_spec.get("model_path", "")))
    checks = [
        _gate_check(
            "lease_aware_runtime_ready_score",
            1,
            lease_artifact.get("lease_aware_runtime_ready_score"),
        ),
        _gate_check(
            "fixture_admissibility_ready_score",
            1,
            admissibility_artifact.get("fixture_admissibility_ready_score"),
        ),
        _gate_check(
            "selected_policy_ready_score",
            1,
            selection_artifact.get("selected_policy_ready_score"),
        ),
        _gate_check(
            "selected_schedule",
            "direct",
            dict(selection_artifact.get("selected_policy", {})).get("schedule_id"),
        ),
        _gate_check(
            "chronological_event_stream_ready_score",
            1,
            fixture_artifact.get("chronological_event_stream_ready_score"),
        ),
        _gate_check("exp6967_artifact_hash", EXPECTED_EXP6967_SHA256, fixture_hash),
        _gate_check(
            "exact_24_event_stream",
            {
                "event_count": 24,
                "prompt_count": 24,
                "witness_count": 24,
                "unique_event_ids": 24,
                "unique_pair_ids": 24,
                "all_later_outcome_flags_false": True,
                "chain_ok": True,
            },
            stream,
        ),
        _gate_check("exact_qwen_model_id", QWEN_ID, model_spec.get("hf_id")),
        _gate_check("qwen_cache", True, model_path.is_file()),
        _gate_check("z3_available", True, bool(z3_available)),
        _gate_check("time_zero_forbidden_fields", [], find_forbidden_paths(time_zero)),
    ]
    transaction_ok = False
    try:
        transaction_root.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=transaction_root) as directory:
            store = PolicyStore(
                Path(directory) / "probe",
                initial_records=initial_memory_records(),
                max_state_bytes=MAX_MEMORY_STATE_BYTES,
            )
            transaction_ok = store.state_size <= MAX_MEMORY_STATE_BYTES
    except (OSError, ValueError):
        transaction_ok = False
    checks.append(_gate_check("writable_transaction_directory", True, transaction_ok))
    return checks, source_hashes


def build_plan(
    fixture: Mapping[str, Any],
    selection: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    """Freeze all causal inputs, budgets, rotations, and memory limits."""

    events = sorted(
        (deepcopy(dict(row)) for row in fixture.get("chronological_event_rows", [])),
        key=lambda row: row.get("event_ordinal", -1),
    )
    if len(events) != 24 or [row.get("event_ordinal") for row in events] != list(range(24)):
        raise ValueError("Exp6978 requires exactly 24 ordered events")
    if dict(selection.get("selected_policy", {})).get("schedule_id") != "direct":
        raise ValueError("selected schedule is not direct")
    pairs = {
        str(row["pair_id"]): deepcopy(dict(row))
        for row in fixture.get("prompt_visible_rows", [])
        if row.get("split") == "chronological"
    }
    labels = {
        str(row["pair_id"]): deepcopy(dict(row))
        for row in fixture.get("exact_witness_rows", [])
        if row.get("subject_kind") == "slice_pair" and row.get("split") == "chronological"
    }
    planned_events: list[JsonDict] = []
    for event in events:
        ordinal = int(event["event_ordinal"])
        pair_id = str(event["pair_id"])
        if pair_id not in pairs or pair_id not in labels:
            raise ValueError(f"event lacks prompt or exact witness:{pair_id}")
        seed = RANDOM_SEED + ordinal
        order = [ARMS[(ordinal + offset) % len(ARMS)] for offset in range(len(ARMS))]
        planned_events.append(
            {
                "event": event,
                "pair": pairs[pair_id],
                "expected_label": labels[pair_id]["expected_label"],
                "source_certificate_hash": event.get("source_certificate_hash"),
                "arm_order": order,
                "arm_seeds": {arm: seed for arm in ARMS},
                "token_cap": TOKEN_CAP,
            }
        )
    plan: JsonDict = {
        "schema": "carnot.exp6978.frozen_plan.v1",
        "model_spec": deepcopy(dict(model_spec)),
        "schedule_id": "direct",
        "selected_policy_hash": selection.get("selected_policy_hash"),
        "token_cap": TOKEN_CAP,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "repeat_penalty": REPEAT_PENALTY,
        "memory_schema": "carnot.constraint_policy_store.v1",
        "initial_memory": initial_memory_records(),
        "memory_lookup_limit": MEMORY_LOOKUP_LIMIT,
        "max_memory_state_bytes": MAX_MEMORY_STATE_BYTES,
        "rollback_threshold": ROLLBACK_THRESHOLD,
        "held_future_start": HELD_FUTURE_START,
        "exact_evaluator": "z3_plus_bounded_enumerator_from_exp6976",
        "stream_hash": fixture.get("split_hashes", {}).get("chronological"),
        "event_order_hash": _sha256_json(
            [
                {
                    "event_id": row["event"]["event_id"],
                    "event_hash": row["event"]["event_hash"],
                    "prompt_record_hash": row["event"]["prompt_record_hash"],
                }
                for row in planned_events
            ]
        ),
        "prompt_contract_hash": _sha256_json(
            {
                "schedule": "direct",
                "schema": CONSTRAINT_IR_SCHEMA,
                "initial_memory": initial_memory_records(),
                "token_cap": TOKEN_CAP,
            }
        ),
        "events": planned_events,
    }
    plan["plan_hash"] = _sha256_json(plan)
    return plan


def evaluate_completion(
    raw_completion: str,
    event: JsonDict,
    pair: JsonDict,
    expected_label: str,
) -> JsonDict:
    """Run the independent Exp6976 exact executor on one durable completion."""

    from carnot import experiment_6975_delayed_constraint_candidate_bank as bank
    from carnot import experiment_6976_exact_candidate_certification as cert

    attempt = {
        "attempt_key": f"{QWEN_ID}|{pair['pair_id']}|direct|exp6978",
        "ordinal": event["event_ordinal"],
        "hf_id": QWEN_ID,
        "pair_id": pair["pair_id"],
        "split": "chronological",
        "formulation_family": event["formulation_family"],
        "schedule_id": "direct",
        "candidate_raw_text": raw_completion,
        "candidate_raw_sha256": bank.sha256_text(raw_completion),
        "parser_diagnostic": bank.parse_syntax(raw_completion),
        "call_status": "complete",
        "exception_type": None,
    }
    result = cert.certify_candidate(attempt, pair)
    row = deepcopy(result["candidate_row"])
    row["exact_success"] = bool(
        row.get("authorities_agree") is True and row.get("certified_relation") == expected_label
    )
    digest_payload = {
        "event_id": event["event_id"],
        "raw_sha256": attempt["candidate_raw_sha256"],
        "expected_label": expected_label,
        "candidate": row,
        "agreement": result["solver_agreement_row"],
        "witness": result["exact_witness_row"],
    }
    row["exact_certificate_digest"] = _sha256_json(digest_payload)
    row["terminal"] = bool(row.get("terminal"))
    return row


def _sequence(counter: list[int]) -> int:
    """Allocate one monotonic causal sequence number."""

    counter[0] += 1
    return counter[0]


def _fixture_proposal(error_class: str, ordinal: int) -> JsonDict:
    """Build one valid proposal for restart and rollback safety fixtures."""

    writer_input = build_writer_input(
        atomic_error_class=error_class,
        exact_certificate_digest="sha256:" + hashlib.sha256(error_class.encode()).hexdigest(),
        schedule_metadata={"schedule_id": "direct", "token_cap": TOKEN_CAP},
        outcome="non_equivalent",
    )
    return build_update_proposal(
        writer_input,
        event_id=f"fixture-event-{ordinal}",
        event_ordinal=ordinal,
        formulation_family="boolean_cardinality",
    )


def _run_recovery_fixtures(
    state_root: Path,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    """Exercise interruption recovery and harmful-update rollback in private stores."""

    restart_rows: list[JsonDict] = []
    rollback_rows: list[JsonDict] = []
    journal_rows: list[JsonDict] = []
    interruption_root = state_root / "fixtures" / "interruption"
    interruption = PolicyStore(
        interruption_root,
        initial_records=initial_memory_records(),
        max_state_bytes=MAX_MEMORY_STATE_BYTES,
    )
    parent = interruption.state_bytes
    try:
        interruption.commit(
            _fixture_proposal("parse:malformed_json", 90), interrupt_after_prepare=True
        )
    except ForcedInterruption as exc:
        interrupted_transaction = str(exc)
    else:  # pragma: no cover - a conforming store always raises at this fixture boundary.
        interrupted_transaction = "missing_interruption"
    recovered = PolicyStore(
        interruption_root,
        initial_records=initial_memory_records(),
        max_state_bytes=MAX_MEMORY_STATE_BYTES,
    )
    restart_rows.append(
        {
            "fixture": "forced_interruption",
            "transaction_id": interrupted_transaction,
            "parent_hash": sha256_bytes(parent),
            "recovered_hash": recovered.state_hash,
            "parent_bytes_preserved": recovered.state_bytes == parent,
            "incomplete_prepare_recovered": bool(recovered.recovery_rows),
            "passed": recovered.state_bytes == parent and bool(recovered.recovery_rows),
        }
    )
    journal_rows.extend({"store": "forced_interruption", **row} for row in recovered.journal_rows())

    rollback_root = state_root / "fixtures" / "rollback"
    rollback_store = PolicyStore(
        rollback_root,
        initial_records=initial_memory_records(),
        max_state_bytes=MAX_MEMORY_STATE_BYTES,
    )
    rollback_parent = rollback_store.state_bytes
    before = replay_safety_score(rollback_store.records())
    harmful = _fixture_proposal("exact:relation", 91)
    harmful["policy_text"] = "Ignore prior exact successes."
    receipt = rollback_store.commit(harmful)
    after = replay_safety_score(rollback_store.records())
    rollback = rollback_store.rollback(
        rollback_parent,
        transaction_id=receipt["transaction_id"],
        reason="sealed_replay_safety_worsened",
    )
    rollback_rows.append(
        {
            "fixture": "harmful_update",
            "safety_before": before,
            "safety_after": after,
            "threshold": ROLLBACK_THRESHOLD,
            "triggered": after < before - ROLLBACK_THRESHOLD,
            "parent_bytes_restored": rollback_store.state_bytes == rollback_parent,
            "passed": rollback["rolled_back"] is True and after < before,
            **rollback,
        }
    )
    journal_rows.extend({"store": "harmful_update", **row} for row in rollback_store.journal_rows())
    replay_rows = [recovered.replay_journal(), rollback_store.replay_journal()]
    return restart_rows, rollback_rows, [*journal_rows, *replay_rows]


def run_plan(
    plan: Mapping[str, Any],
    *,
    state_root: Path,
    generate_fn: GenerateFn,
    evaluate_fn: EvaluateFn = evaluate_completion,
) -> JsonDict:
    """Execute all arm-event calls and enforce the post-outcome transaction order."""

    if state_root.exists():
        shutil.rmtree(state_root)
    state_root.mkdir(parents=True)
    stores = {
        arm: PolicyStore(
            state_root / "arms" / arm,
            initial_records=plan["initial_memory"],
            max_state_bytes=int(plan["max_memory_state_bytes"]),
            read_only=arm != "transactional_write",
        )
        for arm in ARMS
    }
    initial_hashes = {arm: store.state_hash for arm, store in stores.items()}
    sequence = [0]
    rows: list[JsonDict] = []
    prompt_rows: list[JsonDict] = []
    exact_rows: list[JsonDict] = []
    lookup_rows: list[JsonDict] = []
    proposal_rows: list[JsonDict] = []
    commit_rows: list[JsonDict] = []
    rollback_rows: list[JsonDict] = []
    checkpoint_rows: list[JsonDict] = []
    live_duration_s = 0.0
    source_events = [dict(row["event"]) for row in plan["events"]]
    for planned in plan["events"]:
        event = deepcopy(dict(planned["event"]))
        ordinal = int(event["event_ordinal"])
        predecessors = visible_predecessors(source_events, ordinal)
        predecessor_ids = [str(row["event_id"]) for row in predecessors]
        for arm in planned["arm_order"]:
            store = stores[str(arm)]
            memory = (
                []
                if arm == "frozen"
                else lookup_memory(
                    store.records(),
                    formulation_family=str(event["formulation_family"]),
                    limit=int(plan["memory_lookup_limit"]),
                )
            )
            prompt, visibility = build_prompt(
                event=event,
                pair=planned["pair"],
                memory_records=memory,
                schedule_id=str(plan["schedule_id"]),
                predecessor_ids=predecessor_ids,
            )
            event_dir = state_root / "durable" / str(arm) / f"{ordinal:02d}"
            prompt_path = event_dir / "prompt.json"
            prompt_sequence = _sequence(sequence)
            _write_json_atomic(
                prompt_path,
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "prompt": prompt,
                    "prompt_hash": visibility["prompt_hash"],
                    "sequence": prompt_sequence,
                },
            )
            call_row = {
                "event_id": event["event_id"],
                "event_ordinal": ordinal,
                "arm": arm,
                "model_id": plan["model_spec"]["hf_id"],
                "random_seed": planned["arm_seeds"][arm],
                "token_cap": planned["token_cap"],
                "schedule_id": plan["schedule_id"],
            }
            generated = deepcopy(generate_fn(call_row, prompt))
            raw = str(generated["raw_completion"])
            raw_path = event_dir / "raw_completion.json"
            raw_sequence = _sequence(sequence)
            raw_hash = sha256_bytes(raw.encode("utf-8"))
            _write_json_atomic(
                raw_path,
                {
                    "event_id": event["event_id"],
                    "arm": arm,
                    "raw_completion": raw,
                    "raw_hash": raw_hash,
                    "sequence": raw_sequence,
                },
            )
            exact_event = {**event, "active_arm": arm}
            outcome = deepcopy(
                evaluate_fn(
                    raw,
                    exact_event,
                    deepcopy(dict(planned["pair"])),
                    str(planned["expected_label"]),
                )
            )
            outcome_sequence = _sequence(sequence)
            error_class = classify_error(outcome)
            writer_input: JsonDict | None = None
            proposal: JsonDict | None = None
            commit: JsonDict | None = None
            rollback: JsonDict | None = None
            commit_sequence: int | None = None
            state_before = store.state_hash
            if arm == "transactional_write" and error_class != "none":
                writer_input = build_writer_input(
                    atomic_error_class=error_class,
                    exact_certificate_digest=str(outcome["exact_certificate_digest"]),
                    schedule_metadata={
                        "schedule_id": plan["schedule_id"],
                        "token_cap": plan["token_cap"],
                    },
                    outcome=str(planned["expected_label"]),
                )
                proposal = build_update_proposal(
                    writer_input,
                    event_id=str(event["event_id"]),
                    event_ordinal=ordinal,
                    formulation_family=str(event["formulation_family"]),
                )
                proposal_sequence = _sequence(sequence)
                parent_bytes = store.state_bytes
                safety_before = replay_safety_score(store.records())
                commit = store.commit(proposal)
                commit_sequence = _sequence(sequence)
                safety_after = replay_safety_score(store.records())
                if safety_after < safety_before - float(plan["rollback_threshold"]):
                    restored = store.rollback(
                        parent_bytes,
                        transaction_id=str(commit["transaction_id"]),
                        reason="sealed_replay_safety_worsened",
                    )
                    rollback = {
                        "fixture": "live_commit",
                        "event_id": event["event_id"],
                        "safety_before": safety_before,
                        "safety_after": safety_after,
                        "passed": restored["rolled_back"],
                        **restored,
                    }
                    rollback_rows.append(rollback)
                proposal_rows.append(
                    {
                        "event_id": event["event_id"],
                        "event_ordinal": ordinal,
                        "arm": arm,
                        "writer_input": writer_input,
                        "proposal": proposal,
                        "proposal_sequence": proposal_sequence,
                        "after_outcome": proposal_sequence > outcome_sequence,
                        "forbidden_paths": find_forbidden_paths(writer_input),
                    }
                )
                commit_rows.append(
                    {
                        "event_id": event["event_id"],
                        "event_ordinal": ordinal,
                        "arm": arm,
                        "outcome_sequence": outcome_sequence,
                        "commit_sequence": commit_sequence,
                        "after_outcome": commit_sequence > outcome_sequence,
                        "rolled_back": rollback is not None,
                        **commit,
                    }
                )
            live_duration_s += float(
                generated.get(
                    "live_duration_s",
                    float(generated.get("latency_ms", 0.0)) / 1_000,
                )
            )
            row = {
                **call_row,
                "arm_order": list(planned["arm_order"]),
                "attempted": True,
                "terminal": outcome.get("terminal") is True,
                "context_id": generated["context_id"],
                "fresh_context": True,
                "prompt_hash": visibility["prompt_hash"],
                "raw_completion_hash": raw_hash,
                "prompt_tokens": int(generated.get("prompt_tokens", 0)),
                "completion_tokens": int(generated.get("completion_tokens", 0)),
                "parse_success": outcome.get("parse_success") is True,
                "exact_success": outcome.get("exact_success") is True,
                "certified_relation": outcome.get("certified_relation"),
                "error_class": error_class,
                "memory_hit": bool(memory),
                "memory_record_keys": [str(record["policy_key"]) for record in memory],
                "write": proposal is not None,
                "commit": commit is not None,
                "rollback": rollback is not None,
                "latency_ms": float(generated.get("latency_ms", 0.0)),
                "state_hash_before": state_before,
                "state_hash_after": store.state_hash,
                "state_bytes": store.state_size,
                "prompt_sequence": prompt_sequence,
                "raw_sequence": raw_sequence,
                "outcome_sequence": outcome_sequence,
                "commit_sequence": commit_sequence,
            }
            rows.append(row)
            prompt_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_ordinal": ordinal,
                    "arm": arm,
                    "visible_predecessor_ids": predecessor_ids,
                    "visible_predecessor_ordinals": list(range(ordinal)),
                    "expected_predecessor_count": ordinal,
                    "prompt_hash": visibility["prompt_hash"],
                    "prompt_path": str(prompt_path),
                    "forbidden_paths": visibility["forbidden_paths"],
                    "passed": len(predecessor_ids) == ordinal and not visibility["forbidden_paths"],
                }
            )
            lookup_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_ordinal": ordinal,
                    "arm": arm,
                    "state_hash_before": state_before,
                    "record_keys": [str(record["policy_key"]) for record in memory],
                    "hit": bool(memory),
                    "all_sources_precede_event": all(
                        int(record.get("source_event_ordinal", -1)) < ordinal for record in memory
                    ),
                }
            )
            exact_rows.append(
                {
                    "event_id": event["event_id"],
                    "event_ordinal": ordinal,
                    "arm": arm,
                    "raw_sequence": raw_sequence,
                    "outcome_sequence": outcome_sequence,
                    "raw_was_durable": raw_sequence < outcome_sequence,
                    "expected_label": planned["expected_label"],
                    **outcome,
                }
            )
            checkpoint_rows.extend(
                [
                    {
                        "event_id": event["event_id"],
                        "arm": arm,
                        "stage": "prompt_durable",
                        "sequence": prompt_sequence,
                        "path": str(prompt_path),
                        "content_hash": visibility["prompt_hash"],
                    },
                    {
                        "event_id": event["event_id"],
                        "arm": arm,
                        "stage": "raw_completion_durable",
                        "sequence": raw_sequence,
                        "path": str(raw_path),
                        "content_hash": raw_hash,
                    },
                    {
                        "event_id": event["event_id"],
                        "arm": arm,
                        "stage": "exact_outcome",
                        "sequence": outcome_sequence,
                        "content_hash": outcome["exact_certificate_digest"],
                    },
                ]
            )
    write_store = stores["transactional_write"]
    main_journal = [{"store": "transactional_write", **row} for row in write_store.journal_rows()]
    main_replay = write_store.replay_journal()
    final_bytes = write_store.state_bytes
    restarted = PolicyStore(
        write_store.root,
        initial_records=plan["initial_memory"],
        max_state_bytes=int(plan["max_memory_state_bytes"]),
    )
    restart_rows = [
        {
            "fixture": "final_write_store_restart",
            "expected_hash": sha256_bytes(final_bytes),
            "recovered_hash": restarted.state_hash,
            "bytes_match": restarted.state_bytes == final_bytes,
            "passed": restarted.state_bytes == final_bytes,
        }
    ]
    fixture_restart, fixture_rollback, fixture_evidence = _run_recovery_fixtures(state_root)
    fixture_journal = [row for row in fixture_evidence if "phase" in row]
    fixture_replays = [row for row in fixture_evidence if "phase" not in row]
    restart_rows.extend(fixture_restart)
    rollback_rows.extend(fixture_rollback)
    journal_ok = bool(main_replay["passed"]) and all(
        row.get("passed") is True for row in fixture_replays
    )
    return {
        "rows": rows,
        "prompt_visibility_rows": prompt_rows,
        "exact_outcome_rows": exact_rows,
        "memory_lookup_rows": lookup_rows,
        "update_proposal_rows": proposal_rows,
        "transaction_journal_rows": [*main_journal, *fixture_journal],
        "commit_rows": commit_rows,
        "rollback_rows": rollback_rows,
        "restart_recovery_rows": restart_rows,
        "checkpoint_rows": checkpoint_rows,
        "live_duration_s": live_duration_s,
        "final_state_bytes": restarted.state_size,
        "final_state_hash": restarted.state_hash,
        "journal_replay_passed": journal_ok,
        "initial_state_hashes": initial_hashes,
        "store_paths": {arm: str(store.root) for arm, store in stores.items()},
    }


def _budgets_match(rows: Sequence[Mapping[str, Any]]) -> bool:
    """Check assigned model, seed, token cap, and arm coverage for every event."""

    by_event: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_event[int(row.get("event_ordinal", -1))].append(row)
    return len(by_event) == 24 and all(
        len(event_rows) == 3
        and {row.get("arm") for row in event_rows} == set(ARMS)
        and len({row.get("model_id") for row in event_rows}) == 1
        and len({row.get("random_seed") for row in event_rows}) == 1
        and {row.get("token_cap") for row in event_rows} == {TOKEN_CAP}
        for event_rows in by_event.values()
    )


def completion_score(
    rows: Sequence[Mapping[str, Any]],
    *,
    budgets_match: bool,
    journal_ok: bool,
    leakage_ok: bool,
) -> int:
    """Return bare one only for a complete and auditable 72-row run."""

    identities = {
        (row.get("event_ordinal"), row.get("arm")) for row in rows if row.get("terminal") is True
    }
    return int(
        len(rows) == 72
        and len(identities) == 72
        and all(row.get("terminal") is True for row in rows)
        and budgets_match
        and journal_ok
        and leakage_ok
    )


def positive_score(
    *,
    chronological_gain: int,
    lost_read_only_successes: int,
    rollback_passed: bool,
    memory_state_bytes: int,
    max_state_bytes: int,
) -> int:
    """Return bare one only for held-future utility with no stability loss."""

    return int(
        chronological_gain >= 2
        and lost_read_only_successes == 0
        and rollback_passed
        and memory_state_bytes <= max_state_bytes
    )


def _per_event_results(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Pair all three arms on each event without changing event order."""

    result: list[JsonDict] = []
    for ordinal in range(24):
        members = [row for row in rows if row.get("event_ordinal") == ordinal]
        by_arm = {str(row["arm"]): row for row in members}
        result.append(
            {
                "event_ordinal": ordinal,
                "event_id": members[0]["event_id"] if members else None,
                "arm_exact_success": {
                    arm: by_arm.get(arm, {}).get("exact_success") for arm in ARMS
                },
                "arm_parse_success": {
                    arm: by_arm.get(arm, {}).get("parse_success") for arm in ARMS
                },
                "terminal": len(members) == 3
                and all(row.get("terminal") is True for row in members),
            }
        )
    return result


def _metric_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[JsonDict], int, float, float, int]:
    """Compute held-future gain, plasticity, stability, and forgetting from rows."""

    held: list[JsonDict] = []
    seen_write_errors: set[str] = set()
    plastic_opportunities = 0
    plastic_successes = 0
    lost_count = 0
    max_forgetting = 0
    read_only_success_count = 0
    retained_count = 0
    for ordinal in range(24):
        members = {str(row["arm"]): row for row in rows if row.get("event_ordinal") == ordinal}
        write = members.get("transactional_write", {})
        read_only = members.get("read_only", {})
        error_class = str(read_only.get("error_class", "none"))
        recurring = error_class != "none" and error_class in seen_write_errors
        if ordinal >= HELD_FUTURE_START:
            if recurring:
                plastic_opportunities += 1
                plastic_successes += int(write.get("exact_success") is True)
            read_success = read_only.get("exact_success") is True
            write_success = write.get("exact_success") is True
            if read_success:
                read_only_success_count += 1
                retained_count += int(write_success)
                if not write_success:
                    lost_count += 1
            max_forgetting = max(max_forgetting, lost_count)
            held.append(
                {
                    "event_ordinal": ordinal,
                    "event_id": read_only.get("event_id"),
                    "read_only_exact_success": read_success,
                    "transactional_write_exact_success": write_success,
                    "paired_delta": int(write_success) - int(read_success),
                    "read_only_error_class": error_class,
                    "newly_recurring_error_class": recurring,
                }
            )
        if str(write.get("error_class", "none")) != "none":
            seen_write_errors.add(str(write["error_class"]))
    gain = sum(int(row["paired_delta"]) for row in held)
    plasticity = plastic_successes / plastic_opportunities if plastic_opportunities else 0.0
    stability = retained_count / read_only_success_count if read_only_success_count else 1.0
    return held, gain, plasticity, stability, max_forgetting


def _budget_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Summarize assigned and consumed tokens without treating consumption as budget."""

    return [
        {
            "arm": arm,
            "attempted_call_count": sum(row.get("arm") == arm for row in rows),
            "terminal_row_count": sum(
                row.get("arm") == arm and row.get("terminal") is True for row in rows
            ),
            "assigned_token_cap_per_event": TOKEN_CAP,
            "assigned_token_budget_total": sum(
                int(row.get("token_cap", 0)) for row in rows if row.get("arm") == arm
            ),
            "prompt_tokens_used": sum(
                int(row.get("prompt_tokens", 0)) for row in rows if row.get("arm") == arm
            ),
            "completion_tokens_used": sum(
                int(row.get("completion_tokens", 0)) for row in rows if row.get("arm") == arm
            ),
        }
        for arm in ARMS
    ]


def _checksum_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Select timing-free evidence for the reproducibility checksum."""

    return {
        "schema": artifact["schema"],
        "MODEL_SPECS": artifact["MODEL_SPECS"],
        "model_file_hashes": artifact["model_file_hashes"],
        "stream_hash": artifact["stream_hash"],
        "event_order_hash": artifact["event_order_hash"],
        "rows": [
            {
                key: row.get(key)
                for key in (
                    "event_id",
                    "event_ordinal",
                    "arm",
                    "random_seed",
                    "token_cap",
                    "prompt_hash",
                    "raw_completion_hash",
                    "parse_success",
                    "exact_success",
                    "error_class",
                    "state_hash_after",
                )
            }
            for row in artifact["rows"]
        ],
        "transaction_journal_rows": artifact["transaction_journal_rows"],
        "scores": {
            "complete": artifact["self_learning_run_complete_score"],
            "positive": artifact["transactional_learning_positive_score"],
            "gain": artifact["chronological_gain_over_readonly"],
            "plasticity": artifact["plasticity_score"],
            "stability": artifact["stability_score"],
            "max_forgetting": artifact["max_forgetting"],
        },
        "random_seed": artifact["random_seed"],
    }


def reduce_run(
    plan: Mapping[str, Any],
    run: Mapping[str, Any],
    *,
    run_date: str,
    duration_s: float,
    live_duration_s: float,
    source_artifact_hashes: Mapping[str, Any],
    model_file_hashes: Mapping[str, Any],
    preconditions_checked: Sequence[Mapping[str, Any]],
) -> JsonDict:
    """Reduce the complete run from arm-event and transaction evidence."""

    rows = [deepcopy(dict(row)) for row in run.get("rows", [])]
    prompt_rows = [deepcopy(dict(row)) for row in run.get("prompt_visibility_rows", [])]
    proposal_rows = [deepcopy(dict(row)) for row in run.get("update_proposal_rows", [])]
    budget_rows = _budget_rows(rows)
    budgets_match = _budgets_match(rows)
    leakage_rows = [
        {
            "check": "time_zero_initial_memory",
            "observed_forbidden_paths": find_forbidden_paths(plan.get("initial_memory", [])),
            "passed": not find_forbidden_paths(plan.get("initial_memory", [])),
        },
        {
            "check": "chronological_prompt_frontier",
            "observed_failure_count": sum(row.get("passed") is not True for row in prompt_rows),
            "passed": bool(prompt_rows) and all(row.get("passed") is True for row in prompt_rows),
        },
        {
            "check": "prompt_denied_fields",
            "observed_failure_count": sum(bool(row.get("forbidden_paths")) for row in prompt_rows),
            "passed": all(not row.get("forbidden_paths") for row in prompt_rows),
        },
        {
            "check": "writer_denied_fields",
            "observed_failure_count": sum(
                bool(row.get("forbidden_paths")) for row in proposal_rows
            ),
            "passed": all(not row.get("forbidden_paths") for row in proposal_rows),
        },
        {
            "check": "arm_store_isolation",
            "observed_store_count": len(set(run.get("store_paths", {}).values())),
            "passed": len(set(run.get("store_paths", {}).values())) == 3,
        },
    ]
    leakage_ok = all(row["passed"] is True for row in leakage_rows)
    journal_ok = run.get("journal_replay_passed") is True
    complete = completion_score(
        rows,
        budgets_match=budgets_match,
        journal_ok=journal_ok,
        leakage_ok=leakage_ok,
    )
    held, gain, plasticity, stability, max_forgetting = _metric_rows(rows)
    lost = sum(
        row["read_only_exact_success"] and not row["transactional_write_exact_success"]
        for row in held
    )
    rollback_passed = any(
        row.get("fixture") == "harmful_update" and row.get("passed") is True
        for row in run.get("rollback_rows", [])
    )
    state_bytes = int(run.get("final_state_bytes", 0))
    positive = (
        positive_score(
            chronological_gain=gain,
            lost_read_only_successes=lost,
            rollback_passed=rollback_passed,
            memory_state_bytes=state_bytes,
            max_state_bytes=int(plan["max_memory_state_bytes"]),
        )
        if complete
        else 0
    )
    if complete and positive:
        verdict_class = "positive"
        verdict = "complete_positive_transactional_constraint_self_learning"
    elif complete:
        verdict_class = "null"
        verdict = "complete_null_transactional_constraint_self_learning"
    else:
        verdict_class = "partial"
        verdict = "complete_partial_transactional_constraint_self_learning"
    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions_checked],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "live_duration_s": float(live_duration_s),
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "MODEL_SPECS": [deepcopy(dict(plan["model_spec"]))],
        "models_used": [plan["model_spec"]["hf_id"]] if rows else [],
        "model_file_hashes": deepcopy(dict(model_file_hashes)),
        "stream_hash": plan["stream_hash"],
        "event_order_hash": plan["event_order_hash"],
        "arm_config_rows": [
            {
                "arm": "frozen",
                "memory_read": False,
                "memory_write": False,
                "initial_state_hash": run.get("initial_state_hashes", {}).get("frozen"),
                "store_path": run.get("store_paths", {}).get("frozen"),
            },
            {
                "arm": "read_only",
                "memory_read": True,
                "memory_write": False,
                "initial_state_hash": run.get("initial_state_hashes", {}).get("read_only"),
                "store_path": run.get("store_paths", {}).get("read_only"),
            },
            {
                "arm": "transactional_write",
                "memory_read": True,
                "memory_write": True,
                "initial_state_hash": run.get("initial_state_hashes", {}).get(
                    "transactional_write"
                ),
                "store_path": run.get("store_paths", {}).get("transactional_write"),
            },
        ],
        "budget_rows": budget_rows,
        "rows": rows,
        "per_event_results": _per_event_results(rows),
        "prompt_visibility_rows": prompt_rows,
        "exact_outcome_rows": deepcopy(list(run.get("exact_outcome_rows", []))),
        "memory_lookup_rows": deepcopy(list(run.get("memory_lookup_rows", []))),
        "update_proposal_rows": proposal_rows,
        "transaction_journal_rows": deepcopy(list(run.get("transaction_journal_rows", []))),
        "commit_rows": deepcopy(list(run.get("commit_rows", []))),
        "rollback_rows": deepcopy(list(run.get("rollback_rows", []))),
        "restart_recovery_rows": deepcopy(list(run.get("restart_recovery_rows", []))),
        "held_future_rows": held,
        "chronological_gain_over_readonly": gain,
        "plasticity_score": plasticity,
        "stability_score": stability,
        "max_forgetting": max_forgetting,
        "memory_state_bytes": state_bytes,
        "leakage_check_rows": leakage_rows,
        "self_learning_run_complete_score": complete,
        "transactional_learning_positive_score": positive,
        "checkpoint_rows": deepcopy(list(run.get("checkpoint_rows", []))),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(preconditions_checked),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _sha256_json(_checksum_projection(artifact))
    return artifact


def build_blocked_artifact(
    *,
    run_date: str,
    checks: Sequence[Mapping[str, Any]],
    source_artifact_hashes: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    """Build the complete blocked schema without inventing live rows."""

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": 0.0,
        "live_duration_s": 0.0,
        "source_artifact_hashes": deepcopy(dict(source_artifact_hashes)),
        "MODEL_SPECS": [deepcopy(dict(model_spec))],
        "models_used": [],
        "model_file_hashes": {},
        "stream_hash": None,
        "event_order_hash": None,
        "arm_config_rows": [],
        "budget_rows": [],
        "rows": [],
        "per_event_results": [],
        "prompt_visibility_rows": [],
        "exact_outcome_rows": [],
        "memory_lookup_rows": [],
        "update_proposal_rows": [],
        "transaction_journal_rows": [],
        "commit_rows": [],
        "rollback_rows": [],
        "restart_recovery_rows": [],
        "held_future_rows": [],
        "chronological_gain_over_readonly": 0,
        "plasticity_score": 0.0,
        "stability_score": 0.0,
        "max_forgetting": 0,
        "memory_state_bytes": 0,
        "leakage_check_rows": [],
        "self_learning_run_complete_score": 0,
        "transactional_learning_positive_score": 0,
        "checkpoint_rows": [],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": _gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "blocked_transactional_constraint_self_learning",
    }
    artifact["reproducibility_checksum"] = _sha256_json(_checksum_projection(artifact))
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute required structure and scores from row evidence."""

    errors = [
        f"missing required field:{field}"
        for field in REQUIRED_ARTIFACT_FIELDS
        if field not in artifact
    ]
    if errors:
        return errors
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        errors.append("field_principles must cover every required field")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    for field in ("self_learning_run_complete_score", "transactional_learning_positive_score"):
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int) or value not in {0, 1}:
            errors.append(f"{field} must be a bare integer")
    verdict_class = artifact.get("verdict_class")
    if verdict_class not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class is invalid")
    verdict = str(artifact.get("honest_verdict", ""))
    prefixes = {
        "positive": "complete_positive",
        "circular_positive": "complete_circular",
        "null": "complete_null",
        "blocked": "blocked_",
        "disqualified": "complete_disqualified",
        "partial": "complete_partial",
    }
    if verdict_class in prefixes and not verdict.startswith(prefixes[str(verdict_class)]):
        errors.append("honest_verdict prefix disagrees with verdict_class")
    if verdict_class == "blocked":
        if artifact.get("rows"):
            errors.append("blocked artifact must not contain arm-event rows")
        if artifact.get("gate_check_summary", {}).get("failed_check") is None:
            errors.append("blocked artifact must name a failed check")
    else:
        rows = artifact.get("rows", [])
        budgets_match = _budgets_match(rows)
        leakage_ok = all(
            row.get("passed") is True for row in artifact.get("leakage_check_rows", [])
        )
        journal_ok = bool(artifact.get("transaction_journal_rows")) and all(
            row.get("passed") is True for row in artifact.get("restart_recovery_rows", [])
        )
        expected_complete = completion_score(
            rows,
            budgets_match=budgets_match,
            journal_ok=journal_ok,
            leakage_ok=leakage_ok,
        )
        if artifact.get("self_learning_run_complete_score") != expected_complete:
            errors.append("self_learning_run_complete_score disagrees with rows")
        held, gain, plasticity, stability, forgetting = _metric_rows(rows)
        if artifact.get("held_future_rows") != held:
            errors.append("held_future_rows disagree with rows")
        for field, expected in (
            ("chronological_gain_over_readonly", gain),
            ("plasticity_score", plasticity),
            ("stability_score", stability),
            ("max_forgetting", forgetting),
        ):
            if artifact.get(field) != expected:
                errors.append(f"{field} disagrees with rows")
        lost = sum(
            row["read_only_exact_success"] and not row["transactional_write_exact_success"]
            for row in held
        )
        rollback_passed = any(
            row.get("fixture") == "harmful_update" and row.get("passed") is True
            for row in artifact.get("rollback_rows", [])
        )
        expected_positive = (
            positive_score(
                chronological_gain=gain,
                lost_read_only_successes=lost,
                rollback_passed=rollback_passed,
                memory_state_bytes=int(artifact.get("memory_state_bytes", 0)),
                max_state_bytes=MAX_MEMORY_STATE_BYTES,
            )
            if expected_complete
            else 0
        )
        if artifact.get("transactional_learning_positive_score") != expected_positive:
            errors.append("transactional_learning_positive_score disagrees with rows")
    expected_checksum = _sha256_json(_checksum_projection(artifact))
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum mismatch")
    return errors


def _read_object(path: Path) -> JsonDict:
    """Read one JSON object or return an empty value for blocked preflight."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _load_live_model(model_path: str) -> Any:  # pragma: no cover - live GGUF boundary.
    """Load the exact Qwen file across the two frozen CUDA devices."""

    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=8_192,
        n_batch=512,
        n_ubatch=512,
        n_gpu_layers=-1,
        main_gpu=0,
        split_mode=1,
        tensor_split=[0.5, 0.5],
        seed=RANDOM_SEED,
        verbose=True,
    )


def _live_generator(model: Any) -> GenerateFn:  # pragma: no cover - live token boundary.
    """Return a generator that resets model context and grammar on every call."""

    from llama_cpp import LlamaGrammar

    call_index = [0]

    def generate(row: JsonDict, prompt: str) -> JsonDict:
        started = time.perf_counter()
        call_index[0] += 1
        model.reset()
        model.set_seed(int(row["random_seed"]))
        grammar = LlamaGrammar.from_json_schema(
            _canonical_json(CONSTRAINT_IR_SCHEMA), verbose=False
        )
        response = model.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "Use public evidence only. Emit the requested JSON certificate.",
                },
                {"role": "user", "content": prompt},
            ],
            max_tokens=TOKEN_CAP,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            top_k=TOP_K,
            repeat_penalty=REPEAT_PENALTY,
            seed=int(row["random_seed"]),
            grammar=grammar,
        )
        usage = response.get("usage", {})
        elapsed = time.perf_counter() - started
        return {
            "raw_completion": str(response["choices"][0]["message"]["content"]),
            "prompt_tokens": int(usage.get("prompt_tokens", 0)),
            "completion_tokens": int(usage.get("completion_tokens", 0)),
            "context_id": _sha256_json(
                {
                    "call_index": call_index[0],
                    "event_id": row["event_id"],
                    "arm": row["arm"],
                    "seed": row["random_seed"],
                }
            ),
            "latency_ms": elapsed * 1_000,
            "live_duration_s": elapsed,
        }

    return generate


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    transaction_root: Path = TRANSACTION_ROOT,
) -> JsonDict:
    """Check prerequisites, run live Qwen calls, reduce rows, and write the artifact."""

    started = time.perf_counter()
    sources = {name: _read_object(path) for name, path in SOURCE_PATHS.items()}
    model_spec = deepcopy(MODEL_SPECS[0])
    fixture_hash = _sha256_path(SOURCE_PATHS["fixture"])
    try:
        import z3  # noqa: F401

        z3_available = True
    except ImportError:
        z3_available = False
    checks, source_hashes = collect_preconditions(
        lease_artifact=sources["lease"],
        admissibility_artifact=sources["admissibility"],
        selection_artifact=sources["selection"],
        fixture_artifact=sources["fixture"],
        source_paths=SOURCE_PATHS,
        fixture_hash=fixture_hash,
        model_spec=model_spec,
        transaction_root=transaction_root,
        z3_available=z3_available,
    )
    if not all(row["passed"] is True for row in checks):
        artifact = build_blocked_artifact(
            run_date=run_date,
            checks=checks,
            source_artifact_hashes=source_hashes,
            model_spec=model_spec,
        )
        artifact["duration_s"] = time.perf_counter() - started
        artifact["reproducibility_checksum"] = _sha256_json(_checksum_projection(artifact))
        if validate_artifact(artifact):
            raise RuntimeError(f"blocked artifact validation failed:{validate_artifact(artifact)}")
        _write_json_atomic(result_path, artifact)
        return artifact
    plan = build_plan(sources["fixture"], sources["selection"], model_spec)
    model_path = Path(str(model_spec["model_path"]))
    model_hash_before = _sha256_path(model_path)
    model = _load_live_model(str(model_path))
    try:
        run_rows = run_plan(
            plan,
            state_root=transaction_root / "run",
            generate_fn=_live_generator(model),
        )
    finally:
        model.close()
        del model
        gc.collect()
    model_hash_after = _sha256_path(model_path)
    checks.append(_gate_check("model_hash_immutable", model_hash_before, model_hash_after))
    artifact = reduce_run(
        plan,
        run_rows,
        run_date=run_date,
        duration_s=time.perf_counter() - started,
        live_duration_s=float(run_rows["live_duration_s"]),
        source_artifact_hashes=source_hashes,
        model_file_hashes={QWEN_ID: model_hash_after},
        preconditions_checked=checks,
    )
    errors = validate_artifact(artifact)
    if errors:
        raise RuntimeError(f"artifact validation failed:{errors}")
    _write_json_atomic(result_path, artifact)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run Exp6978 or validate an existing result artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--transaction-root", type=Path, default=TRANSACTION_ROOT)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = _read_object(args.result_path)
        errors = validate_artifact(artifact)
        print(_canonical_json({"ok": not errors, "errors": errors}))
        return int(bool(errors))
    artifact = run(
        run_date=args.date,
        result_path=args.result_path,
        transaction_root=args.transaction_root,
    )
    _write_json_atomic(args.result_path, artifact)
    print(
        _canonical_json(
            {"honest_verdict": artifact["honest_verdict"], "result": str(args.result_path)}
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - command boundary.
    raise SystemExit(main())
