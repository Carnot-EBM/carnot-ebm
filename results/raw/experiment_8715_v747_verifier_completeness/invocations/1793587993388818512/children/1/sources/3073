"""Requalify the recovered V663 selector against exact historical bytes.

This module does not repeat the V663 protocol implementation. It authenticates
that implementation and adds the V664 custody rules needed to publish its
current successful selection without changing the old blocked receipt.

Spec refs: REQ-VERIFY-7602 and SCENARIO-VERIFY-7602-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Any

from carnot import experiment_7588_v663_evidence_protocol as v663
from carnot.experiment_7358_v646_validation_contract import (
    AffectedManifest,
    PlannedCommand,
    build_command_plan,
    run_categorized_commands,
    validate_command_plan,
)
from carnot.reporting import experiment_7303_validation_scope as validation_scope
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260924"
MILESTONE = "2026.09.664"
EXPERIMENT_ID = "exp7602-v664-evidence-requalification"
SCHEMA = "carnot.exp7602.v664.evidence_requalification.v1"
RESULT_PATH = Path("results/experiment_7602_v664_evidence_requalification.json")
RAW_DIR = Path("results/raw/experiment_7602_v664_evidence_requalification")
MODULE_PATH = Path("python/carnot/experiment_7602_v664_evidence_requalification.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7602_v664_evidence_requalification.py")
TEST_PATH = Path("tests/python/test_experiment_7602_v664_evidence_requalification.py")
SPEC_PATH = Path("openspec/capabilities/verification/spec.md")
V662_PATH = v663.SOURCE_PATH
V663_MODULE_PATH = v663.MODULE_PATH
V663_RESULT_PATH = v663.RESULT_PATH
HISTORICAL_MODEL_ID = v663.HISTORICAL_MODEL_ID
RANDOM_SEED = 7_602_001
ZERO_INVOCATION_COUNTS = deepcopy(v663.ZERO_INVOCATION_COUNTS)
AFFECTED_MANIFEST = AffectedManifest(
    experiment_id=EXPERIMENT_ID,
    test_paths=(TEST_PATH.as_posix(),),
    changed_modules=(MODULE_PATH.as_posix(),),
    static_paths=(WRAPPER_PATH.as_posix(),),
)


def precondition(
    check: str,
    upstream: str,
    path: Path,
    field: str,
    expected: Any,
    observed: Any,
    *,
    op: str = "eq",
) -> JsonDict:
    """Record exact operands so an external block can be resolved without guessing."""

    if op == "eq":
        passed = observed == expected
    elif op == "in":
        passed = observed in expected
    else:
        raise ValueError("precondition_operator_invalid")
    return {
        "check": check,
        "upstream": upstream,
        "path": str(path.resolve()),
        "field": field,
        "op": op,
        "expected": expected,
        "observed": observed,
        "passed": passed,
    }


def restore_authenticated_group(row: Mapping[str, Any]) -> JsonDict:
    """Restore source, question, and answer only when prompt delimiters are unique."""

    prompt = row.get("scorer_prompt") or row.get("context")
    if not isinstance(prompt, str):
        raise ValueError("original_text_contract_invalid")
    for marker in (v663._QUESTION_MARKER, v663._ANSWER_MARKER, v663._OPTION_MARKER):
        if prompt.count(marker) != 1:
            raise ValueError("parser_delimiter_collision")
    restored = v663.restore_original_text({**dict(row), "context": prompt, "response": ""})
    source = restored.get("context")
    question = restored.get("question")
    answer = restored.get("response")
    if not all(  # pragma: no cover - the shipped parser rejects this state first.
        isinstance(value, str) and value for value in (source, question, answer)
    ):
        raise ValueError("original_text_contract_invalid")
    rebuilt = (
        v663._PROMPT_PREFIX
        + str(source)
        + v663._QUESTION_MARKER
        + str(question)
        + v663._ANSWER_MARKER
        + str(answer)
        + v663._OPTION_MARKER
        + v663._PROMPT_SUFFIX
    )
    if rebuilt != prompt:  # pragma: no cover - unique delimiters make this invariant exact.
        raise ValueError("scorer_prompt_roundtrip_invalid")
    restored["authenticated_scorer_prompt_sha256"] = canonical_hash(prompt)
    restored["question_sha256"] = canonical_hash(question)
    return restored


def select_requalified_roles(roles: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    """Run the shipped selector after strict lossless prompt restoration."""

    restored = {
        role: [restore_authenticated_group(row) for row in rows] for role, rows in roles.items()
    }
    return v663.select_roles(restored)


def build_learning_schedule(selected: Mapping[str, Any]) -> JsonDict:
    """Freeze fit anchors and causal online access by the existing salted order."""

    scored = selected["scored"]
    fit_ids = [str(row["source_id"]) for row in scored["fit"]]
    online_ids = [str(row["source_id"]) for row in scored["online"]]
    blocks = []
    for start in range(0, len(online_ids), 8):
        block = online_ids[start : start + 8]
        if len(block) != 8:
            raise ValueError("online_block_incomplete")
        blocks.append(
            {
                "block_index": start // 8,
                "update_ids": block[:4],
                "admission_ids": block[4:],
                "label_release_lag": 8,
            }
        )
    return {
        "fit_optimization_ids": fit_ids[:64],
        "fit_anchor_ids": fit_ids[64:],
        "fit_partition_rule": "first_64_and_last_16_in_salted_source_hash_order",
        "tune_role": "hyperparameter_selection_only",
        "policy_role": "strongest_comparator_selection_only",
        "online_blocks": blocks,
        "evaluation_role": "evaluator_only",
        "admission_labels_trainable": False,
        "evaluation_labels_trainable": False,
        "feedback_lag": 8,
    }


def build_model_record(row: Mapping[str, Any], *, partition: str) -> JsonDict:
    """Build one complete predictor input without labels or baseline probabilities."""

    restored = restore_authenticated_group(row)
    source = str(restored["context"])
    question = str(restored["question"])
    answer = str(restored["response"])
    return {
        "component_hash": str(restored["source_id"]),
        "role": str(restored["role"]),
        "source_role": str(restored.get("source_role") or restored["role"]),
        "official_split": str(restored["official_split"]),
        "learning_partition": partition,
        "complete_source": source,
        "complete_question": question,
        "complete_answer": answer,
        "source_sha256": v663._text_sha256(source),
        "question_sha256": v663._text_sha256(question),
        "answer_sha256": v663._text_sha256(answer),
        "source_sentences": v663.segment_lossless(source, "S"),
        "question_sentences": v663.segment_lossless(question, "S"),
        "answer_sentences": v663.segment_lossless(answer, "R"),
        "evidence_feature_names": list(v663.EVIDENCE_FEATURE_NAMES),
        "allowed_relations": sorted(v663._RELATIONS),
        "maximum_proposed_links": 6,
        "fresh_confirmatory_claim_allowed": False,
        "historically_exposed": True,
        "labels_accessible": False,
        "raw_probability_accessible": False,
    }


def validate_model_record(record: Mapping[str, Any], source_row: Mapping[str, Any]) -> bool:
    """Reject any predictor label access or byte difference from the selected source."""

    forbidden = {"label", "probability", "raw_probability_offset", "human_label"}
    if forbidden & set(record):
        raise ValueError("predictor_label_access")
    restored = restore_authenticated_group(source_row)
    expected = {
        "complete_source": str(restored["context"]),
        "complete_question": str(restored["question"]),
        "complete_answer": str(restored["response"]),
    }
    names = {
        "complete_source": "source",
        "complete_question": "question",
        "complete_answer": "answer",
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise ValueError(f"{names[field]}_byte_identity_mismatch")
    for field, segment_field in (
        ("complete_source", "source_sentences"),
        ("complete_question", "question_sentences"),
        ("complete_answer", "answer_sentences"),
    ):
        v663.roundtrip_segments(str(record[field]), record.get(segment_field, []))
    if (
        record.get("labels_accessible") is not False
        or record.get("raw_probability_accessible") is not False
    ):
        raise ValueError("predictor_label_access")
    return True


def _partition_maps(schedule: Mapping[str, Any]) -> tuple[dict[str, str], dict[str, JsonDict]]:
    partition = {
        **{str(value): "fit_optimization" for value in schedule["fit_optimization_ids"]},
        **{str(value): "fit_old_distribution_anchor" for value in schedule["fit_anchor_ids"]},
    }
    online: dict[str, JsonDict] = {}
    for block in schedule["online_blocks"]:
        for phase, key in (("update", "update_ids"), ("admission", "admission_ids")):
            for position, component in enumerate(block[key]):
                online[str(component)] = {
                    "phase": phase,
                    "block_index": int(block["block_index"]),
                    "position_in_phase": position,
                }
    return partition, online


def build_role_records(
    selected: Mapping[str, Any],
) -> tuple[dict[str, list[JsonDict]], dict[str, list[JsonDict]]]:
    """Build role-local predictor inputs and separate evaluator-only label stores."""

    schedule = build_learning_schedule(selected)
    fit_partition, online_partition = _partition_maps(schedule)
    rows_by_role = {**selected["scored"], "pilot": selected["pilot"]}
    model_rows: dict[str, list[JsonDict]] = {}
    evaluator_rows: dict[str, list[JsonDict]] = {}
    for role, rows in rows_by_role.items():
        model_rows[role] = []
        evaluator_rows[role] = []
        for row in rows:
            component = str(row["source_id"])
            if role == "fit":
                partition = fit_partition[component]
            elif role == "online":
                partition = f"online_{online_partition[component]['phase']}"
            else:
                partition = f"{role}_only"
            model = build_model_record(row, partition=partition)
            validate_model_record(model, row)
            model_rows[role].append(model)
            online_phase = online_partition.get(component, {}).get("phase")
            training_allowed = partition == "fit_optimization" or online_phase == "update"
            evaluator_rows[role].append(
                {
                    "component_hash": component,
                    "role": role,
                    "learning_partition": partition,
                    "label": int(row["label"]),
                    "raw_probability": float(row["probability"]),
                    "online_phase": online_phase,
                    "online_block_index": online_partition.get(component, {}).get("block_index"),
                    "label_release_lag": 8 if role == "online" else None,
                    "training_allowed": training_allowed,
                    "evaluator_only": role in {"evaluation", "pilot"},
                }
            )
    validate_label_join(model_rows, evaluator_rows)
    return model_rows, evaluator_rows


def validate_label_join(
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    evaluator_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> bool:
    """Require an exact one-to-one identity join while labels remain separate."""

    if set(model_rows) != set(evaluator_rows):
        raise ValueError("label_join_identity_mismatch")
    for role in model_rows:
        models = list(model_rows[role])
        labels = list(evaluator_rows[role])
        if [row.get("component_hash") for row in models] != [
            row.get("component_hash") for row in labels
        ]:
            raise ValueError("label_join_identity_mismatch")
        if any(row.get("label") not in {0, 1} for row in labels):
            raise ValueError("label_join_value_invalid")
        if any({"label", "probability", "raw_probability"} & set(row) for row in models):
            raise ValueError("predictor_label_access")
    return True


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, root: Path) -> JsonDict:
    """Write one role store atomically and return its exact byte receipt."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    return {
        "path": v663._path_label(path, root),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "rows": len(rows),
    }


def freeze_requalification_files(
    raw_dir: Path,
    selected: Mapping[str, Any],
    *,
    root: Path,
) -> JsonDict:
    """Freeze the exact selected contracts and isolated stores by role."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    model_rows, evaluator_rows = build_role_records(selected)
    model_receipts = {
        role: _write_jsonl(raw_dir / f"{role}_model_inputs.jsonl", rows, root=root)
        for role, rows in model_rows.items()
    }
    evaluator_receipts = {
        role: _write_jsonl(raw_dir / f"{role}_evaluator_store.jsonl", rows, root=root)
        for role, rows in evaluator_rows.items()
    }
    protocol = v663.build_protocol(selected)
    schedule = build_learning_schedule(selected)
    protocol.update(
        {
            "schema": "carnot.exp7602.v664.evidence_requalification_manifest.v1",
            "fit_partition_counts": {"optimization": 64, "old_distribution_anchor": 16},
            "learning_schedule": schedule,
            "model_records_include_labels": False,
            "model_records_include_raw_probabilities": False,
            "role_separated_sidecars": True,
            "reader_sidecars": {
                "model_inputs": model_receipts,
                "evaluator_stores": evaluator_receipts,
            },
            "roundtrip_failure_count": 0,
        }
    )
    protocol_path = raw_dir / "protocol.json"
    atomic_json(protocol_path, protocol)
    rows = v663.build_protocol_rows(selected, protocol)
    reduction = v663.reduce_protocol_rows(rows)
    selected_ids_sha256 = canonical_hash(selected["selected_ids"])
    return {
        "protocol": protocol,
        "protocol_path": str(protocol_path.resolve()),
        "protocol_path_label": v663._path_label(protocol_path, root),
        "protocol_sha256": sha256_file(protocol_path),
        "protocol_bytes": protocol_path.stat().st_size,
        "raw_sidecars": protocol["reader_sidecars"],
        "model_rows": model_rows,
        "evaluator_rows": evaluator_rows,
        "rows": rows,
        "row_reduction": reduction,
        "selected_ids_sha256": selected_ids_sha256,
    }


def _load_object(path: Path) -> JsonDict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def historical_failure_record(path: Path, *, root: Path) -> JsonDict:
    """Preserve the old receipt while admitting that its process state is gone."""

    artifact = _load_object(path)
    failure = artifact.get("gate_check_summary", {}).get("first_failure", {})
    available = path.is_file() and bool(artifact)
    return {
        "artifact_path": v663._path_label(path, root),
        "artifact_sha256": sha256_file(path) if available else None,
        "terminal_artifact_bytes_preserved": available,
        "preserved_verdict": artifact.get("honest_verdict"),
        "preserved_verdict_class": artifact.get("verdict_class"),
        "preserved_failed_check": failure.get("check") if isinstance(failure, Mapping) else None,
        "preserved_observed_failure": (
            failure.get("observed") if isinstance(failure, Mapping) else None
        ),
        "exact_runtime_bytes_reconstructable": False,
        "exact_process_state_available": False,
        "historical_cause": None,
        "cause_note": "The terminal receipt is exact; the historical process state is unavailable.",
    }


def _git_text(root: Path, *args: str) -> str:
    completed = subprocess.run(  # noqa: S603 - fixed git executable and bounded arguments.
        ("git", *args),
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    return completed.stdout.strip()


def _committed_source_record(
    root: Path, relative: Path, producer: str
) -> tuple[JsonDict, JsonDict]:
    path = root / relative
    expected_blob = _git_text(root, "rev-parse", f"HEAD:{relative.as_posix()}")
    observed_blob = _git_text(root, "hash-object", relative.as_posix()) if path.is_file() else None
    check = precondition(
        f"{producer}_committed_bytes",
        "git_commit",
        path,
        "git_blob_sha1",
        expected_blob,
        observed_blob,
    )
    record = {
        "producer": producer,
        "path": relative.as_posix(),
        "sha256": sha256_file(path) if path.is_file() else None,
        "bytes": path.stat().st_size if path.is_file() else 0,
        "git_blob_sha1": observed_blob,
        "commit": _git_text(root, "log", "-1", "--format=%H", "--", relative.as_posix()),
        "source_class": "authenticated_producer",
    }
    return check, record


def collect_preconditions(  # pragma: no cover - production source and Git boundary.
    root: Path,
) -> tuple[list[JsonDict], list[JsonDict], dict[str, list[JsonDict]], JsonDict]:
    """Authenticate V662 custody plus the preserved and committed V663 bytes."""

    root = root.resolve()
    checks, hashes, roles = v663.collect_preconditions(root)
    historical = historical_failure_record(root / V663_RESULT_PATH, root=root)
    checks.extend(
        [
            precondition(
                "v663_preserved_verdict",
                "exp7588",
                root / V663_RESULT_PATH,
                "honest_verdict",
                "complete_blocked_selected_role_roster",
                historical.get("preserved_verdict"),
            ),
            precondition(
                "v663_preserved_failure",
                "exp7588",
                root / V663_RESULT_PATH,
                "gate_check_summary.first_failure.observed",
                "source_group_incomplete",
                historical.get("preserved_observed_failure"),
            ),
        ]
    )
    for relative, producer in (
        (V663_MODULE_PATH, "v663_module"),
        (V663_RESULT_PATH, "v663_failure_receipt"),
    ):
        check, record = _committed_source_record(root, relative, producer)
        checks.append(check)
        hashes.append(record)
    spec_path = root / SPEC_PATH
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    checks.append(
        precondition(
            "driving_requirement",
            "verification_spec",
            spec_path,
            "REQ-*",
            "REQ-VERIFY-7602",
            "REQ-VERIFY-7602" if "REQ-VERIFY-7602" in spec_text else "missing",
        )
    )
    declared = (root / RESULT_PATH).resolve()
    checks.append(
        precondition(
            "declared_output_path",
            "exp7602",
            declared,
            "resolved_path",
            str(root / RESULT_PATH),
            str(declared),
        )
    )
    hashes.append(
        {
            "producer": "v664_capability_spec",
            "path": SPEC_PATH.as_posix(),
            "sha256": sha256_file(spec_path) if spec_path.is_file() else None,
            "bytes": spec_path.stat().st_size if spec_path.is_file() else 0,
            "source_class": "conductor_pre_gate",
        }
    )
    return checks, hashes, roles, historical


def build_selector_snapshot(
    checks: Sequence[Mapping[str, Any]], selected: Mapping[str, Any]
) -> JsonDict:
    """Reduce fresh-process selection to exact counts and a roster checksum."""

    return {
        "all_preconditions_passed": bool(checks)
        and all(row.get("passed") is True for row in checks),
        "source_role_counts": deepcopy(v663.SOURCE_ROLE_COUNTS),
        "scored_role_counts": {role: len(selected["scored"][role]) for role in v663.ROLE_COUNTS},
        "pilot_count": len(selected["pilot"]),
        "selected_unique": len(
            {str(row["source_id"]) for rows in selected["scored"].values() for row in rows}
            | {str(row["source_id"]) for row in selected["pilot"]}
        ),
        "selected_ids_sha256": canonical_hash(selected["selected_ids"]),
        "salt": selected["salt"],
    }


REQUIRED_PRINCIPLE_FIELDS = (
    "honest_verdict",
    "verdict_class",
    "flagged_adversarial",
    "gate_check_summary",
    "acceptance_gate_results",
    "rows",
    "sample_size_budget",
    "inference_substrate",
    "inference_substrate_class",
    "MODEL_SPECS",
    "invocation_counts",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "source_artifact_hashes",
    "validation_receipts",
    "verifier_is_oracle",
    "field_principles",
    "evidence_protocol_ready_score",
    "protocol_path",
    "role_counts",
    "historical_failure_reconstruction",
    "fresh_confirmatory_claim_allowed",
)


def field_principles() -> dict[str, str]:
    """Keep the audit reason for each governed terminal field beside its value."""

    return {
        "honest_verdict": "Use a complete_ terminal prefix; completion alone is not scientific benefit.",
        "verdict_class": "Exactly positive | circular_positive | null | blocked | disqualified | partial; partial is unfinished owned work only.",
        "flagged_adversarial": "Persist the terminal reader result; flagged evidence never opens readiness.",
        "gate_check_summary": "Every block names check, upstream, path, field, operator, expected, and observed.",
        "acceptance_gate_results": "Validity, readiness, benefit, retention, and freshness have separate principles and results.",
        "rows": "Each unit and arm keeps absolute metrics, numerator, denominator, seed, direction, censoring, and provenance.",
        "sample_size_budget": "Count independent groups; repeated arms, seeds, or windows never multiply them.",
        "inference_substrate": "Describe current execution; historical GPU evidence is not a current model call.",
        "inference_substrate_class": "Record planned and actual classes; blocked_no_run means no model work.",
        "MODEL_SPECS": "No-model tasks use an empty list and name historical identity separately.",
        "invocation_counts": "Count current loads, forwards, generations, and tokens separately from history.",
        "duration_s": "Measure current monotonic time and phase spans without inherited or padded duration.",
        "random_seed": "Persist every stochastic-stage seed even when the selection salt is deterministic.",
        "reproducibility_checksum": "Bind configuration, immutable evidence, sidecars, and reduction.",
        "source_artifact_hashes": "Separate producers, conductor pre-gates, and missing producers.",
        "validation_receipts": "Bind commands, exits, worktree, log hashes, and independent terminal readers.",
        "verifier_is_oracle": "Exact fixtures cannot establish learned semantic correctness or an oracle-distinct gain.",
        "field_principles": "Carry these one-line reasons inside the terminal artifact.",
        "evidence_protocol_ready_score": "One requires exact sidecars and a valid terminal cold replay.",
        "protocol_path": "Name the exact authenticated role and feature contract.",
        "role_counts": "Freeze 240 scored groups and eight disjoint fit-role pilots.",
        "historical_failure_reconstruction": "Separate preserved receipt bytes from unavailable historical process state.",
        "fresh_confirmatory_claim_allowed": "Every selected group was historically exposed, so this remains false.",
    }


def _gate(
    check: str,
    category: str,
    expected: Any,
    observed: Any,
    principle: str,
) -> JsonDict:
    return {
        "check": check,
        "category": category,
        "op": "eq",
        "expected": expected,
        "observed": observed,
        "passed": observed == expected,
        "principle": principle,
    }


def _gate_summary(gates: Sequence[Mapping[str, Any]]) -> JsonDict:
    failed = [deepcopy(dict(row)) for row in gates if row.get("passed") is not True]
    return {
        "passed": not failed,
        "failed_count": len(failed),
        "failed_checks": [str(row["check"]) for row in failed],
        "first_failure": failed[0] if failed else None,
    }


def reproducibility_checksum(value: Mapping[str, Any]) -> str:
    """Hash the terminal content without recursively hashing the checksum itself."""

    stable = deepcopy(dict(value))
    stable.pop("reproducibility_checksum", None)
    return canonical_hash(stable)


def _validation_passed(receipts: Sequence[Mapping[str, Any]]) -> bool:
    return bool(receipts) and all(
        row.get("passed") is True and row.get("exit_code") == 0 and row.get("timed_out") is not True
        for row in receipts
    )


def _terminal_outcomes(receipts: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    names = {
        "declared_entrypoint_cold_replay",
        "independent_reduction",
        "adversarial_verify",
        "verdict_row_consistency_strict",
    }
    return [
        {
            "name": row.get("name"),
            "passed": row.get("passed"),
            "exit_code": row.get("exit_code"),
            "log_sha256": row.get("log_sha256"),
        }
        for row in receipts
        if row.get("name") in names
    ]


def build_artifact(
    bundle: Mapping[str, Any],
    *,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    validation_receipts: Sequence[Mapping[str, Any]],
    duration_s: float,
    historical_failure: Mapping[str, Any],
    selector_replay: Mapping[str, Any],
    require_validation: bool = True,
) -> JsonDict:
    """Build a protocol-ready null while preserving fixture circularity."""

    protocol = bundle["protocol"]
    expected_counts = {**v663.ROLE_COUNTS, "pilot": v663.PILOT_COUNT}
    reduction = dict(bundle["row_reduction"])
    preconditions_ok = bool(preconditions) and all(
        row.get("passed") is True for row in preconditions
    )
    validation_ok = _validation_passed(validation_receipts) if require_validation else True
    selector_ok = (
        selector_replay.get("all_preconditions_passed") is True
        and selector_replay.get("scored_role_counts") == v663.ROLE_COUNTS
        and selector_replay.get("pilot_count") == v663.PILOT_COUNT
        and selector_replay.get("selected_unique") == v663.SCORED_GROUPS + v663.PILOT_COUNT
        and selector_replay.get("selected_ids_sha256") == bundle["selected_ids_sha256"]
        and selector_replay.get("salt") == v663.SELECTION_SALT
    )
    role_ok = (
        protocol.get("role_counts") == expected_counts
        and reduction.get("unique_units") == v663.SCORED_GROUPS + v663.PILOT_COUNT
        and reduction.get("all_arms_complete") is True
    )
    isolated = (
        protocol.get("model_records_include_labels") is False
        and protocol.get("model_records_include_raw_probabilities") is False
        and set(bundle["raw_sidecars"]["model_inputs"]) == set(expected_counts)
        and set(bundle["raw_sidecars"]["evaluator_stores"]) == set(expected_counts)
    )
    schedule = protocol["learning_schedule"]
    retention_ok = (
        len(schedule["fit_optimization_ids"]) == 64
        and len(schedule["fit_anchor_ids"]) == 16
        and len(schedule["online_blocks"]) == 10
        and schedule["feedback_lag"] == 8
        and schedule["admission_labels_trainable"] is False
        and schedule["evaluation_labels_trainable"] is False
    )
    gates = [
        _gate(
            "authenticated_exact_bytes",
            "validity",
            True,
            preconditions_ok,
            "Only authenticated V662 and committed V663 bytes can seed requalification.",
        ),
        _gate(
            "scoped_and_terminal_validation",
            "validity",
            True,
            validation_ok,
            "Scoped checks and independent terminal readers must pass before publication.",
        ),
        _gate(
            "fresh_process_selector_replay",
            "readiness",
            True,
            selector_ok,
            "A separate process must reproduce the exact selected roster.",
        ),
        _gate(
            "role_and_arm_roster",
            "readiness",
            True,
            role_ok,
            "Fixed counts and complete arms prevent outcome-driven replacement.",
        ),
        _gate(
            "predictor_evaluator_isolation",
            "readiness",
            True,
            isolated,
            "Labels and baseline probabilities cannot enter predictor records.",
        ),
        _gate(
            "delayed_feedback_and_retention",
            "retention",
            True,
            retention_ok,
            "Lag-eight updates and evaluator-only groups prevent future-label training.",
        ),
        _gate(
            "empirical_benefit",
            "benefit",
            "separate_passing_empirical_gate",
            "not_measured_protocol_requalification_only",
            "Protocol and oracle fixtures cannot establish predictive benefit.",
        ),
        _gate(
            "fresh_confirmatory_claim",
            "freshness",
            False,
            protocol.get("fresh_confirmatory_claim_allowed"),
            "Requalification cannot erase historical exposure of selected groups.",
        ),
    ]
    ready = all(
        row["passed"]
        for row in gates
        if row["category"] in {"validity", "readiness", "retention", "freshness"}
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "V664 evidence protocol requalification",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": (
            "complete_null_evidence_requalification_ready"
            if ready
            else "complete_disqualified_evidence_requalification_validation_failed"
        ),
        "verdict_class": "null" if ready else "disqualified",
        "flagged_adversarial": False,
        "positive_claim": False,
        "fixture_control_verdict_class": "circular_positive",
        "empirical_benefit_measured": False,
        "fresh_confirmatory_claim_allowed": False,
        "claim_scope": "descriptive_requalification_of_historically_exposed_groups",
        "verifier_is_oracle": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_id": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "inference_substrate_class": "no_model_load",
        "planned_inference_substrate_class": "no_model_load",
        "random_seed": RANDOM_SEED,
        "random_seeds_used": {"protocol_rows": RANDOM_SEED},
        "selection_salt": v663.SELECTION_SALT,
        "duration_s": float(duration_s),
        "protocol_path": str(bundle["protocol_path_label"]),
        "protocol_sha256": str(bundle["protocol_sha256"]),
        "protocol_bytes": int(bundle["protocol_bytes"]),
        "role_counts": deepcopy(expected_counts),
        "role_learning_contract": deepcopy(schedule),
        "evidence_feature_names": list(v663.EVIDENCE_FEATURE_NAMES),
        "raw_probability_is_separate_offset": True,
        "reader_isolation": deepcopy(protocol["reader_isolation"]),
        "rows": deepcopy(list(bundle["rows"])),
        "independent_row_reduction": reduction,
        "sample_size_budget": {
            "intended_source_groups": sum(v663.SOURCE_ROLE_COUNTS.values()),
            "intended_scored_independent_units": v663.SCORED_GROUPS,
            "intended_pilot_independent_units": v663.PILOT_COUNT,
            "observed_independent_units": v663.SCORED_GROUPS + v663.PILOT_COUNT,
            "excluded_independent_units": sum(v663.SOURCE_ROLE_COUNTS.values())
            - v663.SCORED_GROUPS
            - v663.PILOT_COUNT,
            "censored_independent_units": 0,
            "failed_independent_units": 0,
            "seeds_or_windows_multiply_units": False,
        },
        "acceptance_gate_results": gates,
        "gate_check_summary": _gate_summary(gates),
        "evidence_protocol_ready_score": int(ready),
        "applicable_numbered_e2e": [],
        "capability_e2e": {
            "operations": [
                "authenticate",
                "fresh_process_select",
                "freeze",
                "persist",
                "cold_replay",
                "independent_reduce",
            ],
            "passed": ready,
            "read_only_reporting_numbered_e2e_applicable": False,
        },
        "fresh_process_selector_replay": deepcopy(dict(selector_replay)),
        "historical_failure_reconstruction": deepcopy(dict(historical_failure)),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": deepcopy(dict(bundle["raw_sidecars"])),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "validation_receipts": [deepcopy(dict(row)) for row in validation_receipts],
        "terminal_reader_outcomes": _terminal_outcomes(validation_receipts),
        "field_principles": field_principles(),
        "scope_retirement": {
            "prior_verdict_repeated": False,
            "exact_requalification_scope_retired": False,
            "scientific_hypothesis_retired": False,
            "resource_blocks_do_not_retire_scientific_hypothesis": True,
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "active_research_roadmap_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def build_blocked_artifact(
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Sequence[Mapping[str, Any]],
    *,
    duration_s: float,
    historical_failure: Mapping[str, Any],
) -> JsonDict:
    """Publish an external block as complete work without invented measurements."""

    failed = [deepcopy(dict(row)) for row in checks if row.get("passed") is not True]
    first = (
        failed[0]
        if failed
        else {
            "check": "unknown_precondition",
            "upstream": "unknown",
            "path": str((REPO_ROOT / "unknown").resolve()),
            "field": "unknown",
            "op": "eq",
            "expected": True,
            "observed": False,
            "passed": False,
        }
    )
    reason = re.sub(r"[^a-z0-9]+", "_", str(first["check"]).lower()).strip("_")
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "title": "V664 evidence protocol requalification",
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "complete": True,
        "honest_verdict": f"complete_blocked_{reason}",
        "verdict_class": "blocked",
        "flagged_adversarial": False,
        "positive_claim": False,
        "fixture_control_verdict_class": "circular_positive",
        "empirical_benefit_measured": False,
        "fresh_confirmatory_claim_allowed": False,
        "verifier_is_oracle": True,
        "MODEL_SPECS": [],
        "model_specs": [],
        "no_model_load": True,
        "model_invoked": False,
        "historical_model_id": HISTORICAL_MODEL_ID,
        "invocation_counts": deepcopy(ZERO_INVOCATION_COUNTS),
        "inference_substrate": "blocked_before_cached_aggregation",
        "inference_substrate_class": "blocked_no_run",
        "planned_inference_substrate_class": "no_model_load",
        "random_seed": RANDOM_SEED,
        "selection_salt": v663.SELECTION_SALT,
        "duration_s": float(duration_s),
        "protocol_path": None,
        "protocol_sha256": None,
        "role_counts": {**{role: 0 for role in v663.ROLE_COUNTS}, "pilot": 0},
        "evidence_feature_names": list(v663.EVIDENCE_FEATURE_NAMES),
        "rows": [],
        "independent_row_reduction": None,
        "sample_size_budget": {
            "intended_source_groups": sum(v663.SOURCE_ROLE_COUNTS.values()),
            "intended_scored_independent_units": v663.SCORED_GROUPS,
            "intended_pilot_independent_units": v663.PILOT_COUNT,
            "observed_independent_units": 0,
            "excluded_independent_units": 0,
            "censored_independent_units": 0,
            "failed_independent_units": 0,
            "unstarted_independent_units": v663.SCORED_GROUPS + v663.PILOT_COUNT,
        },
        "acceptance_gate_results": failed,
        "gate_check_summary": {
            "passed": False,
            "failed_count": len(failed),
            "failed_checks": [str(row["check"]) for row in failed],
            "first_failure": first,
        },
        "evidence_protocol_ready_score": 0,
        "applicable_numbered_e2e": [],
        "capability_e2e": {
            "operations": [],
            "passed": False,
            "read_only_reporting_numbered_e2e_applicable": False,
        },
        "fresh_process_selector_replay": None,
        "historical_failure_reconstruction": deepcopy(dict(historical_failure)),
        "source_artifact_hashes": [deepcopy(dict(row)) for row in source_hashes],
        "raw_sidecars": {},
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "validation_receipts": [],
        "terminal_reader_outcomes": [],
        "field_principles": field_principles(),
        "scope_retirement": {
            "prior_verdict_repeated": str(first.get("observed")) == "source_group_incomplete",
            "exact_requalification_scope_retired": str(first.get("observed"))
            == "source_group_incomplete",
            "scientific_hypothesis_retired": False,
            "resource_blocks_do_not_retire_scientific_hypothesis": True,
        },
        "external_publication_authorized": False,
        "submission_authorized": False,
        "purchase_authorized": False,
        "generator_weight_change_authorized": False,
        "default_promotion_authorized": False,
        "research_conductor_modified": False,
        "active_research_roadmap_modified": False,
    }
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _resolve(root: Path, label: str) -> Path:
    path = Path(label)
    return path if path.is_absolute() else root / path


def _receipt_valid(root: Path, receipt: Mapping[str, Any]) -> bool:
    path = _resolve(root, str(receipt.get("path") or ""))
    return (
        path.is_file()
        and sha256_file(path) == receipt.get("sha256")
        and path.stat().st_size == receipt.get("bytes")
    )


def _sidecars_valid(root: Path, sidecars: object) -> bool:
    if not isinstance(sidecars, Mapping) or set(sidecars) != {
        "model_inputs",
        "evaluator_stores",
    }:
        return False
    expected_roles = {*v663.ROLE_COUNTS, "pilot"}
    for store in sidecars.values():
        if not isinstance(store, Mapping) or set(store) != expected_roles:
            return False
        if not all(
            isinstance(receipt, Mapping) and _receipt_valid(root, receipt)
            for receipt in store.values()
        ):
            return False
    return True


def validate_artifact(
    value: object,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Cold-check identity, no-model claims, rows, protocol, and sidecar bytes."""

    if not isinstance(value, Mapping):
        raise ValueError("artifact_object_required")
    artifact = dict(value)
    errors: list[str] = []
    if artifact.get("schema") != SCHEMA or artifact.get("experiment_id") != EXPERIMENT_ID:
        errors.append("identity_invalid")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        errors.append("checksum_invalid")
    if artifact.get("MODEL_SPECS") != [] or artifact.get("model_specs") != []:
        errors.append("model_specs_not_empty")
    if artifact.get("no_model_load") is not True or artifact.get("model_invoked") is not False:
        errors.append("model_invocation_claim_invalid")
    if artifact.get("invocation_counts") != ZERO_INVOCATION_COUNTS:
        errors.append("invocation_counts_nonzero")
    if artifact.get("fresh_confirmatory_claim_allowed") is not False:
        errors.append("freshness_invalid")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or not set(REQUIRED_PRINCIPLE_FIELDS) <= set(principles):
        errors.append("field_principles_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    blocked = artifact.get("verdict_class") == "blocked"
    if blocked:
        failure = artifact.get("gate_check_summary", {}).get("first_failure")
        required = {"check", "upstream", "path", "field", "op", "expected", "observed"}
        if not str(artifact.get("honest_verdict") or "").startswith("complete_blocked_"):
            errors.append("blocked_verdict_invalid")
        if not isinstance(failure, Mapping) or not required <= set(failure):
            errors.append("blocked_gate_summary_invalid")
        if (
            artifact.get("rows") != []
            or artifact.get("inference_substrate_class") != "blocked_no_run"
        ):
            errors.append("blocked_measurement_invalid")
    else:
        if artifact.get("verdict_class") not in {"null", "disqualified"}:
            errors.append("terminal_verdict_invalid")
        if artifact.get("inference_substrate_class") != "no_model_load":
            errors.append("substrate_invalid")
        protocol_path = _resolve(root, str(artifact.get("protocol_path") or ""))
        protocol = _load_object(protocol_path)
        if not protocol_path.is_file() or sha256_file(protocol_path) != artifact.get(
            "protocol_sha256"
        ):
            errors.append("protocol_hash_invalid")
        expected_counts = {**v663.ROLE_COUNTS, "pilot": v663.PILOT_COUNT}
        if (
            artifact.get("role_counts") != expected_counts
            or protocol.get("role_counts") != expected_counts
        ):
            errors.append("role_counts_invalid")
        if protocol.get("selection_salt") != v663.SELECTION_SALT:
            errors.append("selection_salt_invalid")
        if (
            protocol.get("model_records_include_labels") is not False
            or protocol.get("model_records_include_raw_probabilities") is not False
        ):
            errors.append("reader_isolation_invalid")
        if protocol.get("fresh_confirmatory_claim_allowed") is not False:
            errors.append("protocol_freshness_invalid")
        rows = artifact.get("rows")
        if not isinstance(rows, list) or not rows:
            errors.append("rows_missing")
        else:
            try:
                reduced = v663.reduce_protocol_rows(rows)
            except (KeyError, TypeError, ValueError):
                errors.append("row_reduction_invalid")
            else:
                if reduced != artifact.get("independent_row_reduction"):
                    errors.append("row_reduction_mismatch")
                if reduced.get("unique_units") != v663.SCORED_GROUPS + v663.PILOT_COUNT:
                    errors.append("row_unit_count_invalid")
        if not _sidecars_valid(root, artifact.get("raw_sidecars")):
            errors.append("raw_sidecar_invalid")
        if artifact.get("evidence_protocol_ready_score") not in {0, 1}:
            errors.append("ready_score_invalid")
        historical = artifact.get("historical_failure_reconstruction")
        if (
            not isinstance(historical, Mapping)
            or historical.get("terminal_artifact_bytes_preserved") is not True
        ):
            errors.append("historical_failure_not_preserved")
        if isinstance(historical, Mapping) and (
            historical.get("exact_runtime_bytes_reconstructable") is not False
            or historical.get("historical_cause") is not None
        ):
            errors.append("historical_failure_overinterpreted")
        if require_validation and not _validation_passed(artifact.get("validation_receipts", [])):
            errors.append("validation_invalid")
    if errors:
        raise ValueError(";".join(errors))
    return {
        "valid": True,
        "blocked": blocked,
        "ready": artifact.get("evidence_protocol_ready_score") == 1,
        "row_count": len(artifact.get("rows", [])),
    }


def cold_replay(
    path: Path,
    *,
    root: Path = REPO_ROOT,
    require_validation: bool = True,
) -> JsonDict:
    """Reload one exact candidate and validate it without model work."""

    return validate_artifact(_load_object(path), root=root, require_validation=require_validation)


def independent_reduce_artifact(path: Path, *, root: Path = REPO_ROOT) -> JsonDict:
    """Recompute all arm rows and authenticate the exact frozen protocol."""

    artifact = _load_object(path)
    validation = validate_artifact(artifact, root=root, require_validation=False)
    reduction = (
        v663.reduce_protocol_rows(artifact.get("rows", [])) if artifact.get("rows") else None
    )
    return {
        "passed": validation["valid"],
        "row_reduction_sha256": canonical_hash(reduction) if reduction is not None else None,
        "protocol_sha256": artifact.get("protocol_sha256"),
        "row_count": validation["row_count"],
    }


def progress(started: float, phase: str, event: str, **details: Any) -> None:  # pragma: no cover
    """Flush every phase boundary with monotonic elapsed time."""

    payload = {
        "phase": phase,
        "event": event,
        "elapsed_s": round(time.monotonic() - started, 3),
        **details,
    }
    print("[exp7602-progress] " + json.dumps(payload, sort_keys=True), flush=True)


def _run_specs(  # pragma: no cover - streamed subprocess boundary.
    root: Path,
    commands: Sequence[validation_scope.CommandSpec],
    log_dir: Path,
    *,
    category: str,
) -> list[JsonDict]:
    planned = [PlannedCommand(command, category, True) for command in commands]
    rows = run_categorized_commands(root, planned, log_dir=log_dir, heartbeat_s=60.0)
    for row in rows:
        row["worktree"] = str(root.resolve())
    return rows


def _terminal_commands(candidate: Path) -> list[validation_scope.CommandSpec]:  # pragma: no cover
    """Build four independent readers for one unpublished candidate."""

    python = ".venv/bin/python"
    return [
        validation_scope.CommandSpec(
            "declared_entrypoint_cold_replay",
            (python, "-u", WRAPPER_PATH.as_posix(), "--verify-artifact", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "independent_reduction",
            (python, "-u", WRAPPER_PATH.as_posix(), "--independent-reduce", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "adversarial_verify",
            (python, "-u", "scripts/adversarial_verify.py", str(candidate)),
            "exact_candidate",
            300.0,
        ),
        validation_scope.CommandSpec(
            "verdict_row_consistency_strict",
            (python, "-u", "scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)),
            "exact_candidate",
            300.0,
        ),
    ]


def _selector_command(snapshot_path: Path) -> validation_scope.CommandSpec:  # pragma: no cover
    return validation_scope.CommandSpec(
        "fresh_process_selector",
        (
            ".venv/bin/python",
            "-u",
            WRAPPER_PATH.as_posix(),
            "--selector-snapshot-out",
            str(snapshot_path),
        ),
        "authenticated_v662_and_committed_v663",
        300.0,
    )


def _publish_blocked(
    root: Path,
    checks: Sequence[Mapping[str, Any]],
    hashes: Sequence[Mapping[str, Any]],
    historical: Mapping[str, Any],
    started: float,
) -> int:  # pragma: no cover
    artifact = build_blocked_artifact(
        checks,
        hashes,
        duration_s=time.monotonic() - started,
        historical_failure=historical,
    )
    atomic_json(root / RESULT_PATH, artifact)
    progress(
        started,
        "publish",
        "blocked_after",
        path=str(root / RESULT_PATH),
        verdict=artifact["honest_verdict"],
    )
    return 0


def run_experiment(root: Path, run_date: str) -> int:  # pragma: no cover - declared E2E.
    """Authenticate, replay, freeze, independently read, and publish exact bytes."""

    started = time.monotonic()
    root = root.resolve()
    if root != REPO_ROOT.resolve() or run_date != RUN_DATE:
        raise ValueError("root_or_date_invalid")
    progress(started, "preconditions", "before")
    checks, source_hashes, roles, historical = collect_preconditions(root)
    progress(started, "preconditions", "after", completed=len(checks))
    if any(row["passed"] is not True for row in checks):
        return _publish_blocked(root, checks, source_hashes, historical, started)

    progress(started, "selector_parent", "before", source_groups=480)
    try:
        selected = select_requalified_roles(roles)
    except ValueError as exc:
        checks.append(
            precondition(
                "selected_role_roster",
                "exp7575_via_committed_v663",
                root / V662_PATH,
                "salted_selection",
                "240_scored_plus_8_disjoint_pilot",
                str(exc),
            )
        )
        return _publish_blocked(root, checks, source_hashes, historical, started)
    parent_snapshot = build_selector_snapshot(checks, selected)
    progress(started, "selector_parent", "after", selected_unique=248)

    private_root = Path(tempfile.mkdtemp(prefix="carnot-exp7602-"))
    snapshot_path = private_root / "selector_snapshot.json"
    progress(started, "selector_fresh_process", "before")
    selector_receipts = _run_specs(
        root,
        [_selector_command(snapshot_path)],
        private_root / "logs" / "selector",
        category="fresh_process_reproduction",
    )
    progress(
        started,
        "selector_fresh_process",
        "after",
        passed=_validation_passed(selector_receipts),
    )
    if not _validation_passed(selector_receipts):
        raise RuntimeError("fresh_process_selector_failed")
    fresh_snapshot = _load_object(snapshot_path)
    checks.append(
        precondition(
            "fresh_process_selector",
            "committed_v663_module",
            snapshot_path,
            "selected_ids_sha256",
            parent_snapshot["selected_ids_sha256"],
            fresh_snapshot.get("selected_ids_sha256"),
        )
    )
    if checks[-1]["passed"] is not True:
        return _publish_blocked(root, checks, source_hashes, historical, started)

    raw_dir = root / RAW_DIR
    raw_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        raw_dir / "affected_validation_manifest.json",
        {
            "experiment_id": AFFECTED_MANIFEST.experiment_id,
            "test_paths": list(AFFECTED_MANIFEST.test_paths),
            "changed_modules": list(AFFECTED_MANIFEST.changed_modules),
            "static_paths": list(AFFECTED_MANIFEST.static_paths),
        },
    )
    commands = build_command_plan(root, AFFECTED_MANIFEST, private_root)
    plan_errors = validate_command_plan(root, AFFECTED_MANIFEST, commands)
    if plan_errors:
        raise RuntimeError("validation_plan_invalid:" + ",".join(plan_errors))
    progress(started, "affected_validation", "before", commands=len(commands))
    affected = _run_specs(
        root,
        commands,
        private_root / "logs" / "affected",
        category="required_validation",
    )
    progress(started, "affected_validation", "after", passed=_validation_passed(affected))
    if not _validation_passed(affected):
        raise RuntimeError("affected_validation_failed")

    progress(started, "protocol_freeze", "before", units=248)
    bundle = freeze_requalification_files(raw_dir, selected, root=root)
    progress(started, "protocol_freeze", "after", units=248, rows=len(bundle["rows"]))
    candidate = private_root / "terminal_candidate.json"
    initial_receipts = [*selector_receipts, *affected]
    provisional = build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=initial_receipts,
        duration_s=time.monotonic() - started,
        historical_failure=historical,
        selector_replay=fresh_snapshot,
    )
    atomic_json(candidate, provisional)
    progress(started, "terminal_validation", "before", commands=4)
    terminal = _run_specs(
        root,
        _terminal_commands(candidate),
        private_root / "logs" / "terminal",
        category="terminal_independent_reader",
    )
    progress(started, "terminal_validation", "after", passed=_validation_passed(terminal))
    if not _validation_passed(terminal):
        raise RuntimeError("terminal_validation_failed")

    final = build_artifact(
        bundle,
        preconditions=checks,
        source_hashes=source_hashes,
        validation_receipts=[*initial_receipts, *terminal],
        duration_s=time.monotonic() - started,
        historical_failure=historical,
        selector_replay=fresh_snapshot,
    )
    atomic_json(candidate, final)
    progress(started, "exact_terminal_validation", "before", commands=4)
    exact = _run_specs(
        root,
        _terminal_commands(candidate),
        private_root / "logs" / "exact_terminal",
        category="exact_terminal_candidate_reader",
    )
    progress(started, "exact_terminal_validation", "after", passed=_validation_passed(exact))
    if not _validation_passed(exact):
        raise RuntimeError("exact_terminal_validation_failed")
    atomic_json(root / RESULT_PATH, final)
    progress(
        started,
        "publish",
        "after",
        path=str(root / RESULT_PATH),
        verdict=final["honest_verdict"],
    )
    return 0


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:  # pragma: no cover
    """Parse producer and read-only independent reader modes."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=REPO_ROOT)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--verify-artifact", type=Path)
    parser.add_argument("--independent-reduce", type=Path)
    parser.add_argument("--selector-snapshot-out", type=Path)
    return parser.parse_args(argv)


def _argument_path(path: Path, root: Path) -> Path:  # pragma: no cover
    return path if path.is_absolute() else root / path


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - thin CLI.
    """Dispatch production or one bounded read-only verification mode."""

    args = parse_args(argv)
    root = args.root.resolve()
    if args.verify_artifact is not None:
        result = cold_replay(
            _argument_path(args.verify_artifact, root),
            root=root,
            require_validation=True,
        )
        print(json.dumps({"mode": "cold_replay", **result}, sort_keys=True), flush=True)
        return 0
    if args.independent_reduce is not None:
        result = independent_reduce_artifact(
            _argument_path(args.independent_reduce, root), root=root
        )
        print(json.dumps({"mode": "independent_reduce", **result}, sort_keys=True), flush=True)
        return int(result["passed"] is not True)
    if args.selector_snapshot_out is not None:
        checks, _hashes, roles, _historical = collect_preconditions(root)
        if any(row["passed"] is not True for row in checks):
            raise RuntimeError("selector_snapshot_precondition_failed")
        selected = select_requalified_roles(roles)
        snapshot = build_selector_snapshot(checks, selected)
        atomic_json(_argument_path(args.selector_snapshot_out, root), snapshot)
        print(json.dumps({"mode": "selector_snapshot", **snapshot}, sort_keys=True), flush=True)
        return 0
    return run_experiment(root, args.date)
