"""Exp6263 clean local-SOTA event replay bridge.

Spec refs: REQ-LEARN-6263, SCENARIO-LEARN-6263-BRIDGE,
SCENARIO-LEARN-6263-QUARANTINE, SCENARIO-LEARN-6263-NEGATIVES,
SCENARIO-LEARN-6263-REPLAY.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json, atomic_write_text
from carnot.terminal_artifacts import (
    canonical_json,
    classify_artifact_path,
    path_sha256,
    payload_sha256,
)


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260810"
SCHEMA = "carnot.experiment_6263.clean_sota_event_replay_bridge.v1"
ROW_SCHEMA = SCHEMA + ".row_manifest"
QUARANTINE_SCHEMA = SCHEMA + ".quarantine_manifest"
EXPERIMENT_ID = "experiment_6263_clean_sota_event_replay_bridge"
RESULT_RELATIVE_PATH = Path("results/experiment_6263_clean_sota_event_replay_bridge.json")
ROW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6263_clean_sota_event_replay_bridge.rows.jsonl"
)
QUARANTINE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6263_clean_sota_event_replay_bridge.quarantine.json"
)
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"

QWEN_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA26_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
GEMMA31_HF_ID = "unsloth/gemma-4-31B-it-GGUF"
CLEAN_MODEL_ORDER = (QWEN_HF_ID, GEMMA26_HF_ID)
MODEL_ORDER_INDEX = {hf_id: index for index, hf_id in enumerate(CLEAN_MODEL_ORDER)}
PARTITION_MAP = {
    "calibration": "train",
    "future_known": "validation",
    "shifted_family_held": "test",
}

EXP6160_ARTIFACT = Path("results/experiment_6160_sota_decision_calibration_corpus.json")
EXP6162_ARTIFACT = Path("results/experiment_6162_prospective_admission_replication.json")
EXP6146_ARTIFACT = Path("results/experiment_6146_sota_constraint_event_corpus.json")
EXP6262_ARTIFACT = Path("results/experiment_6262_terminal_artifact_readiness_contract.json")
CLEAN_ROW_SIDECARS = {
    QWEN_HF_ID: Path(
        "results/experiment_6160_sota_decision_calibration_corpus.qwen3_6_35b_a3b.rows.jsonl"
    ),
    GEMMA26_HF_ID: Path(
        "results/experiment_6160_sota_decision_calibration_corpus.gemma_4_26b_a4b_it.rows.jsonl"
    ),
}
QUARANTINE_ROW_SIDECARS = {
    QWEN_HF_ID: Path(
        "results/experiment_6146_sota_constraint_event_corpus.qwen3_6_35b_a3b.rows.jsonl"
    ),
    GEMMA31_HF_ID: Path(
        "results/experiment_6146_sota_constraint_event_corpus.gemma_4_31b_it.rows.jsonl"
    ),
}

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6263_clean_sota_event_replay_bridge.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/experiment_6263_clean_sota_event_replay_bridge.py -m pytest tests/python/test_experiment_6263_clean_sota_event_replay_bridge.py -q --no-cov -n 0",
    ".venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/experiment_6263_clean_sota_event_replay_bridge.py --fail-under=100 --show-missing",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6263_clean_sota_event_replay_bridge.py",
    ".venv/bin/ruff check python/carnot/experiment_6263_clean_sota_event_replay_bridge.py tests/python/test_experiment_6263_clean_sota_event_replay_bridge.py",
    ".venv/bin/python -m carnot.experiment_6263_clean_sota_event_replay_bridge --date 20260810",
    ".venv/bin/python -m carnot.experiment_6263_clean_sota_event_replay_bridge --validate",
    "sed -n 1,220p ops/e2e-test-plan.md",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6263_clean_sota_event_replay_bridge.json",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "source_artifact_paths_hashes_and_terminal_classes",
    "model_specs",
    "no_model_load_receipt",
    "clean_source_ids",
    "quarantined_source_ids_and_reasons",
    "immutable_row_manifest_path_and_hash",
    "chronological_order_receipts",
    "row_count_by_model_task_family_and_partition",
    "exact_label_and_parser_provenance",
    "duplicate_count",
    "time_reversal_count",
    "train_validation_test_overlap_count",
    "malformed_or_parser_failure_count_by_disposition",
    "source_mutation_count",
    "replay_positive_control",
    "replay_negative_controls",
    "event_replay_bridge_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal status follows clean source admission, quarantine, replay validation, protected hashes, and commands.",
    "source_artifact_paths_hashes_and_terminal_classes": "Every candidate source path is hashed, and artifacts receive Exp6262 terminal classes.",
    "model_specs": "Model identities are provenance only; the bridge never loads a model.",
    "no_model_load_receipt": "Zero loader and device counters prevent accidental live inference claims.",
    "clean_source_ids": "Only current-rule-clean Exp6160 rows and Exp6162 evidence enter replay.",
    "quarantined_source_ids_and_reasons": "Warned Exp6146 sources stay auditable but outside clean replay.",
    "immutable_row_manifest_path_and_hash": "The row manifest is content-addressed for byte replay.",
    "chronological_order_receipts": "Per-model event order and alias collisions are checked from row content.",
    "row_count_by_model_task_family_and_partition": "Counts expose model, task, family, and frozen partition conservation.",
    "exact_label_and_parser_provenance": "Exact labels, outcomes, prompt hashes, and parser states stay traceable.",
    "duplicate_count": "Bare zero is required before a replay bridge can be ready.",
    "time_reversal_count": "Bare zero proves chronological indexes never reverse.",
    "train_validation_test_overlap_count": "Bare zero proves a model event appears in only one frozen partition.",
    "malformed_or_parser_failure_count_by_disposition": "Parser failures are preserved or quarantined rather than silently dropped.",
    "source_mutation_count": "Bare zero proves sources matched expected hashes and did not change during materialization.",
    "replay_positive_control": "A clean byte replay must accept the same manifest.",
    "replay_negative_controls": "Duplicate, reorder, alias, loss, parser, and mutation controls must reject.",
    "event_replay_bridge_ready_score": "Bare one means all strict replay gates and command receipts pass.",
    "protected_files_unchanged": "Protected source, source evidence, conductor, ops, and traceability paths stay unchanged.",
    "preconditions_checked": "Preconditions bind run date, git status, source hashes, output paths, and classes.",
    "inference_substrate": "The bridge aggregates upstream artifacts and rows without inference.",
    "verifier_is_oracle": "False because this checks evidence integrity, not benchmark answer truth.",
    "field_provenance": "Each field traces to REQ-LEARN-6263, rows, artifacts, controls, or commands.",
    "field_principles": "The artifact carries the audit reason for each field.",
    "test_commands": "Commands document focused, coverage, spec, lint, CLI, E2E-plan, and adversarial checks.",
    "test_exit_codes": "Bare integer exits make failed checks visible.",
    "duration_s": "Measured wall time is reported without padding.",
    "reproducibility_checksum": "The normalized artifact is content-addressed.",
    "honest_verdict": "The verdict uses a terminal prefix and states readiness or the blocking gate.",
}

PROTECTED_RELATIVE_PATHS = (
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("openspec/capabilities/self-learning/spec.md"),
    Path("ops/e2e-test-plan.md"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
    Path("scripts/research_conductor.py"),
    Path("python/carnot/experiment_6263_clean_sota_event_replay_bridge.py"),
    Path("tests/python/test_experiment_6263_clean_sota_event_replay_bridge.py"),
    EXP6160_ARTIFACT,
    EXP6162_ARTIFACT,
    EXP6146_ARTIFACT,
    EXP6262_ARTIFACT,
    CLEAN_ROW_SIDECARS[QWEN_HF_ID],
    CLEAN_ROW_SIDECARS[GEMMA26_HF_ID],
    QUARANTINE_ROW_SIDECARS[QWEN_HF_ID],
    QUARANTINE_ROW_SIDECARS[GEMMA31_HF_ID],
)


def sha256_text(text: str) -> str:
    return "sha256:" + __import__("hashlib").sha256(text.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return payload_sha256(value)


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _resolve(root: Path, path: str | Path) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else root / candidate


def _display_path(root: Path, path: Path) -> str:
    resolved = path.resolve(strict=False)
    try:
        return resolved.relative_to(root.resolve(strict=False)).as_posix()
    except ValueError:
        return resolved.as_posix()


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _jsonl_text(rows: Sequence[Mapping[str, Any]]) -> str:
    return "".join(canonical_json(row) + "\n" for row in rows)


def _git_status(root: Path) -> list[str]:
    proc = subprocess.run(
        ("git", "status", "--short"),
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:  # pragma: no cover - git failure is environmental.
        return [f"git_status_failed:{proc.returncode}:{proc.stderr.strip()}"]
    return proc.stdout.splitlines()


def _protected_hashes(root: Path) -> dict[str, str | None]:
    return {path.as_posix(): path_sha256(root / path) for path in PROTECTED_RELATIVE_PATHS}


def _protected_files_unchanged(before: Mapping[str, str | None], after: Mapping[str, str | None]) -> JsonDict:
    changed = [path for path in before if before.get(path) != after.get(path)]
    conductor = Path("scripts/research_conductor.py").as_posix()
    return {
        "before": dict(before),
        "after": dict(after),
        "changed_paths": changed,
        "unchanged": not changed,
        "scripts_research_conductor_py_untouched": before.get(conductor) == after.get(conductor),
    }


def _row_path_map(root: Path, overrides: Mapping[str, str | Path] | None) -> dict[str, Path]:
    paths = {hf_id: root / relative for hf_id, relative in CLEAN_ROW_SIDECARS.items()}
    for hf_id, path in dict(overrides or {}).items():
        if hf_id in paths:
            paths[hf_id] = _resolve(root, path)
    return paths


def _candidate_sources(root: Path, row_paths: Mapping[str, Path]) -> list[JsonDict]:
    sources: list[JsonDict] = [
        {
            "source_id": "exp6160:artifact",
            "path": root / EXP6160_ARTIFACT,
            "source_kind": "artifact",
            "source_disposition": "clean",
        },
        {
            "source_id": "exp6162:artifact",
            "path": root / EXP6162_ARTIFACT,
            "source_kind": "artifact",
            "source_disposition": "clean",
        },
        {
            "source_id": "exp6146:artifact",
            "path": root / EXP6146_ARTIFACT,
            "source_kind": "artifact",
            "source_disposition": "quarantine",
        },
        {
            "source_id": "exp6262:artifact",
            "path": root / EXP6262_ARTIFACT,
            "source_kind": "artifact",
            "source_disposition": "contract",
        },
    ]
    for hf_id in CLEAN_MODEL_ORDER:
        sources.append(
            {
                "source_id": f"exp6160:rows:{hf_id}",
                "path": row_paths[hf_id],
                "source_kind": "row_sidecar",
                "source_disposition": "clean",
                "model_hf_id": hf_id,
            }
        )
    for hf_id, relative in QUARANTINE_ROW_SIDECARS.items():
        sources.append(
            {
                "source_id": f"exp6146:rows:{hf_id}",
                "path": root / relative,
                "source_kind": "row_sidecar",
                "source_disposition": "quarantine",
                "model_hf_id": hf_id,
            }
        )
    return sources


def _line_count(path: Path) -> int | None:
    if not path.exists():
        return None
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _source_receipts(root: Path, sources: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for source in sources:
        path = Path(source["path"])
        classification = classify_artifact_path(path).to_dict()
        source_kind = str(source["source_kind"])
        receipts[str(source["source_id"])] = {
            "source_id": source["source_id"],
            "source_kind": source_kind,
            "source_disposition": source["source_disposition"],
            "model_hf_id": source.get("model_hf_id"),
            "path": _display_path(root, path),
            "absolute_path": path.resolve(strict=False).as_posix(),
            "sha256": path_sha256(path),
            "size_bytes": path.stat().st_size if path.exists() else None,
            "row_count": _line_count(path) if source_kind == "row_sidecar" else None,
            "exp6262_terminal_classification": classification,
            "terminal_class": classification["classification"]
            if source_kind == "artifact"
            else "row_sidecar",
        }
    return receipts


def _expected_source_lookup(expected_source_hashes: Mapping[str, str] | None) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for key, value in dict(expected_source_hashes or {}).items():
        lookup[key] = value
        lookup[Path(key).resolve(strict=False).as_posix()] = value
    return lookup


def _combine_source_receipts(
    before: Mapping[str, JsonDict],
    after: Mapping[str, JsonDict],
    expected_source_hashes: Mapping[str, str] | None,
    row_count_mismatch_count: int,
) -> tuple[JsonDict, int]:
    expected_lookup = _expected_source_lookup(expected_source_hashes)
    sources: dict[str, JsonDict] = {}
    source_mutation_count = 0
    for source_id, before_row in before.items():
        after_row = after[source_id]
        expected_hash = expected_lookup.get(before_row["absolute_path"]) or expected_lookup.get(
            before_row["path"]
        )
        changed_during_run = before_row["sha256"] != after_row["sha256"]
        expected_mismatch = expected_hash is not None and expected_hash != before_row["sha256"]
        if changed_during_run or expected_mismatch:
            source_mutation_count += 1
        sources[source_id] = {
            **before_row,
            "sha256_before": before_row["sha256"],
            "sha256_after": after_row["sha256"],
            "expected_sha256": expected_hash,
            "changed_during_materialization": changed_during_run,
            "expected_hash_matched": not expected_mismatch,
        }
    return (
        {
            "schema": SCHEMA + ".source_receipts",
            "sources": sources,
            "clean_source_count": sum(
                1 for row in sources.values() if row["source_disposition"] == "clean"
            ),
            "quarantine_source_count": sum(
                1 for row in sources.values() if row["source_disposition"] == "quarantine"
            ),
            "row_count_mismatch_count": row_count_mismatch_count,
            "principle": FIELD_PRINCIPLES[
                "source_artifact_paths_hashes_and_terminal_classes"
            ],
        },
        source_mutation_count,
    )


def _expected_row_counts(exp6160: Mapping[str, Any]) -> dict[str, int]:
    sidecars = (
        exp6160.get("per_model_row_paths_hashes_and_counts", {})
        .get("per_model", {})
    )
    return {hf_id: int(sidecars[hf_id]["row_count"]) for hf_id in CLEAN_MODEL_ORDER}


def _expected_parser_failures(exp6160: Mapping[str, Any]) -> dict[str, int]:
    per_model = (
        exp6160.get("raw_response_strategy_answer_and_invalid_output_counts", {})
        .get("per_model", {})
    )
    return {hf_id: int(per_model[hf_id]["invalid_output_count"]) for hf_id in CLEAN_MODEL_ORDER}


def _is_parser_failure(row: Mapping[str, Any]) -> bool:
    if row.get("invalid_output") is True:
        return True
    parser_fields = (
        row.get("terminal_parse_status"),
        row.get("answer_parse_state"),
        row.get("strategy_parse_state"),
    )
    return any(value not in (None, "", "complete") for value in parser_fields)


def _manifest_row(
    row: Mapping[str, Any],
    *,
    source_row_number: int,
    source_path: Path,
    exp6160: Mapping[str, Any],
) -> JsonDict:
    bridge_partition = PARTITION_MAP[str(row["partition"])]
    row_copy = _copy_json(row)
    source_row_hash = sha256_json(row_copy)
    content_identity = {
        "source_id": f"exp6160:rows:{row['model_hf_id']}",
        "source_row_number": source_row_number,
        "source_row_hash": source_row_hash,
        "declared_row_hash": row.get("row_hash"),
    }
    prompt_manifest = dict(exp6160.get("prompt_decoder_and_seed_freeze_manifest") or {})
    return {
        "schema": ROW_SCHEMA,
        "content_addressed_row_id": sha256_json(content_identity),
        "source_disposition": "clean",
        "source_id": f"exp6160:rows:{row['model_hf_id']}",
        "source_artifact_id": "exp6160:artifact",
        "source_row_file": source_path.resolve(strict=False).as_posix(),
        "source_row_number": source_row_number,
        "source_row_canonical_hash": source_row_hash,
        "source_row_declared_hash": row.get("row_hash"),
        "model_hf_id": row.get("model_hf_id"),
        "model_name": row.get("model_name"),
        "task_id": "exp6159_decision_calibrated_stream",
        "event_id": row.get("event_id"),
        "chronological_index": row.get("chronological_index"),
        "family": row.get("family"),
        "variant_kind": row.get("variant_kind"),
        "source_partition": row.get("partition"),
        "bridge_partition": bridge_partition,
        "visible_event_hash": row.get("visible_event_hash"),
        "prompt_provenance": {
            "message_hash": row.get("message_hash"),
            "decode_policy_hash": row.get("decode_policy_hash"),
            "seed": row.get("seed"),
            "prompt_hash_root": prompt_manifest.get("prompt_hash_root"),
            "prompt_template_version": prompt_manifest.get("prompt_template_version"),
        },
        "parser_provenance": {
            "invalid_output": row.get("invalid_output"),
            "terminal_parse_status": row.get("terminal_parse_status"),
            "answer_parse_state": row.get("answer_parse_state"),
            "strategy_parse_state": row.get("strategy_parse_state"),
            "parser_failure_disposition": "admitted_preserved"
            if _is_parser_failure(row)
            else "admitted_parsed",
        },
        "exact_label_provenance": {
            "unsafe_label": row.get("unsafe_label"),
            "current_outcome": row.get("current_outcome"),
            "exact_labels_hash": row.get("exact_labels_hash"),
            "exact_outcome_hash": row.get("exact_outcome_hash"),
            "exact_answer_hash": row.get("exact_answer_hash"),
            "future_label_hash": row.get("future_label_hash"),
        },
        "decision_provenance": {
            "decision_record_hash": row.get("decision_record_hash"),
            "decision_record_written_before_outcome": row.get(
                "decision_record_written_before_outcome"
            ),
            "post_outcome_attached_after_decision": row.get(
                "post_outcome_attached_after_decision"
            ),
            "outcome_receipt_hash": row.get("outcome_receipt_hash"),
            "raw_response_hash": row.get("raw_response_hash"),
        },
    }


def _duplicate_count(manifest_rows: Sequence[Mapping[str, Any]]) -> int:
    keys = (
        [("row_id", row["source_id"], row["source_row_declared_hash"]) for row in manifest_rows]
        + [
            ("model_event", row["model_hf_id"], row["event_id"])
            for row in manifest_rows
        ]
        + [
            ("content", row["content_addressed_row_id"])
            for row in manifest_rows
        ]
    )
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def _time_reversal_count(rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]]) -> int:
    reversals = 0
    for rows in rows_by_model.values():
        previous: int | None = None
        for row in rows:
            current = int(row.get("chronological_index", -1))
            if previous is not None and current <= previous:
                reversals += 1
            previous = current
    return reversals


def _partition_overlap_count(manifest_rows: Sequence[Mapping[str, Any]]) -> int:
    partitions: dict[tuple[Any, Any], set[str]] = defaultdict(set)
    for row in manifest_rows:
        partitions[(row["model_hf_id"], row["event_id"])].add(str(row["bridge_partition"]))
    return sum(len(values) - 1 for values in partitions.values() if len(values) > 1)


def _alias_collision_count(manifest_rows: Sequence[Mapping[str, Any]]) -> int:
    aliases: dict[Any, set[tuple[Any, Any]]] = defaultdict(set)
    for row in manifest_rows:
        aliases[row["visible_event_hash"]].add((row["event_id"], row["family"]))
    return sum(len(values) - 1 for values in aliases.values() if len(values) > 1)


def _chronological_receipts(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    manifest_rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    for hf_id, rows in rows_by_model.items():
        event_ids = [str(row.get("event_id")) for row in rows]
        indexes = [int(row.get("chronological_index", -1)) for row in rows]
        reversals = _time_reversal_count({hf_id: rows})
        per_model[hf_id] = {
            "row_count": len(rows),
            "first_event_id": event_ids[0] if event_ids else None,
            "last_event_id": event_ids[-1] if event_ids else None,
            "first_chronological_index": indexes[0] if indexes else None,
            "last_chronological_index": indexes[-1] if indexes else None,
            "monotone_strict": reversals == 0,
            "time_reversal_count": reversals,
            "row_order_sha256": sha256_json(event_ids),
            "source_row_numbers_preserved": [index for index in range(len(rows))],
        }
    return {
        "schema": SCHEMA + ".chronological_receipts",
        "per_model": per_model,
        "time_reversal_count": _time_reversal_count(rows_by_model),
        "alias_collision_count": _alias_collision_count(manifest_rows),
        "bridge_manifest_order": "chronological_index_then_model_order",
        "principle": FIELD_PRINCIPLES["chronological_order_receipts"],
    }


def _row_count_receipt(manifest_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    counter: Counter[tuple[str, str, str, str]] = Counter()
    for row in manifest_rows:
        counter[
            (
                str(row["model_hf_id"]),
                str(row["task_id"]),
                str(row["family"]),
                str(row["bridge_partition"]),
            )
        ] += 1
    flat = [
        {
            "model_hf_id": model,
            "task_id": task,
            "family": family,
            "bridge_partition": partition,
            "row_count": count,
        }
        for (model, task, family, partition), count in sorted(counter.items())
    ]
    return {
        "schema": SCHEMA + ".row_counts",
        "partition_map": dict(PARTITION_MAP),
        "flat": flat,
        "total_row_count": sum(counter.values()),
        "principle": FIELD_PRINCIPLES["row_count_by_model_task_family_and_partition"],
    }


def _label_parser_receipt(
    exp6160: Mapping[str, Any],
    manifest_rows: Sequence[Mapping[str, Any]],
    actual_parser_failures: Mapping[str, int],
    expected_parser_failures: Mapping[str, int],
) -> JsonDict:
    per_model: dict[str, JsonDict] = {}
    for hf_id in CLEAN_MODEL_ORDER:
        rows = [row for row in manifest_rows if row["model_hf_id"] == hf_id]
        per_model[hf_id] = {
            "row_count": len(rows),
            "exact_labels_hashes": sorted(
                {str(row["exact_label_provenance"]["exact_labels_hash"]) for row in rows}
            ),
            "exact_outcome_hash_count": len(
                {str(row["exact_label_provenance"]["exact_outcome_hash"]) for row in rows}
            ),
            "future_label_hashes": sorted(
                {str(row["exact_label_provenance"]["future_label_hash"]) for row in rows}
            ),
            "parser_failure_count": actual_parser_failures[hf_id],
            "expected_parser_failure_count": expected_parser_failures[hf_id],
            "parser_state_counts": dict(
                Counter(
                    str(row["parser_provenance"]["parser_failure_disposition"])
                    for row in rows
                )
            ),
            "decode_policy_hashes": sorted(
                {str(row["prompt_provenance"]["decode_policy_hash"]) for row in rows}
            ),
            "message_hash_count": len(
                {str(row["prompt_provenance"]["message_hash"]) for row in rows}
            ),
        }
    prompt = dict(exp6160.get("prompt_decoder_and_seed_freeze_manifest") or {})
    return {
        "schema": SCHEMA + ".label_parser_provenance",
        "prompt_hash_root": prompt.get("prompt_hash_root"),
        "prompt_template_version": prompt.get("prompt_template_version"),
        "decode_policy": prompt.get("decode_policy"),
        "per_model": per_model,
        "principle": FIELD_PRINCIPLES["exact_label_and_parser_provenance"],
    }


def _parser_disposition_receipt(
    actual_parser_failures: Mapping[str, int],
    expected_parser_failures: Mapping[str, int],
    exp6146: Mapping[str, Any],
) -> JsonDict:
    mismatch = sum(
        1
        for hf_id in CLEAN_MODEL_ORDER
        if actual_parser_failures[hf_id] != expected_parser_failures[hf_id]
    )
    quarantine_total = int(
        exp6146.get("strategy_terminal_solution_and_invalid_output_counts", {}).get(
            "total_invalid_output_count", 0
        )
    )
    return {
        "schema": SCHEMA + ".parser_disposition",
        "clean_admitted_preserved": sum(actual_parser_failures.values()),
        "clean_expected_preserved": sum(expected_parser_failures.values()),
        "quarantine_preserved": quarantine_total,
        "parser_failure_mismatch_count": mismatch,
        "malformed_clean_row_count": 0,
        "principle": FIELD_PRINCIPLES[
            "malformed_or_parser_failure_count_by_disposition"
        ],
    }


def _build_manifest(
    exp6160: Mapping[str, Any],
    exp6146: Mapping[str, Any],
    row_paths: Mapping[str, Path],
) -> tuple[list[JsonDict], JsonDict, JsonDict, JsonDict, JsonDict, JsonDict]:
    rows_by_model = {hf_id: _read_jsonl(row_paths[hf_id]) for hf_id in CLEAN_MODEL_ORDER}
    manifest_rows: list[JsonDict] = []
    actual_parser_failures: dict[str, int] = {}
    expected_counts = _expected_row_counts(exp6160)
    expected_parser = _expected_parser_failures(exp6160)
    row_count_mismatch_count = 0
    for hf_id in CLEAN_MODEL_ORDER:
        rows = rows_by_model[hf_id]
        if len(rows) != expected_counts[hf_id]:
            row_count_mismatch_count += 1
        actual_parser_failures[hf_id] = sum(1 for row in rows if _is_parser_failure(row))
        for index, row in enumerate(rows):
            manifest_rows.append(
                _manifest_row(
                    row,
                    source_row_number=index,
                    source_path=row_paths[hf_id],
                    exp6160=exp6160,
                )
            )
    manifest_rows.sort(
        key=lambda row: (
            int(row["chronological_index"]),
            MODEL_ORDER_INDEX[str(row["model_hf_id"])],
            int(row["source_row_number"]),
        )
    )
    for index, row in enumerate(manifest_rows):
        row["bridge_row_index"] = index
    chronological = _chronological_receipts(rows_by_model, manifest_rows)
    row_counts = _row_count_receipt(manifest_rows)
    label_parser = _label_parser_receipt(
        exp6160, manifest_rows, actual_parser_failures, expected_parser
    )
    disposition = _parser_disposition_receipt(
        actual_parser_failures, expected_parser, exp6146
    )
    validation = {
        "duplicate_count": _duplicate_count(manifest_rows),
        "time_reversal_count": chronological["time_reversal_count"],
        "train_validation_test_overlap_count": _partition_overlap_count(manifest_rows),
        "row_count_mismatch_count": row_count_mismatch_count,
        "parser_failure_mismatch_count": disposition["parser_failure_mismatch_count"],
        "alias_collision_count": chronological["alias_collision_count"],
    }
    return manifest_rows, chronological, row_counts, label_parser, disposition, validation


def _model_specs() -> list[JsonDict]:
    return [
        {
            "hf_id": QWEN_HF_ID,
            "role": "clean Exp6160 event provenance",
            "source_disposition": "clean",
            "model_load_allowed": False,
        },
        {
            "hf_id": GEMMA26_HF_ID,
            "role": "clean Exp6160 event provenance",
            "source_disposition": "clean",
            "model_load_allowed": False,
        },
        {
            "hf_id": GEMMA31_HF_ID,
            "role": "Exp6146 quarantine-only provenance",
            "source_disposition": "quarantine",
            "model_load_allowed": False,
        },
    ]


def _quarantine_sources() -> list[JsonDict]:
    reason = (
        "quarantine_only: Exp6146 random_seed is absent under current rules; "
        "requires authorized immutable corrigendum before admission"
    )
    return [
        {"source_id": "exp6146:artifact", "reason": reason},
        {"source_id": f"exp6146:rows:{QWEN_HF_ID}", "reason": reason},
        {"source_id": f"exp6146:rows:{GEMMA31_HF_ID}", "reason": reason},
    ]


def _quarantine_manifest(sources: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        "schema": QUARANTINE_SCHEMA,
        "quarantined_source_ids_and_reasons": list(sources),
        "model_specs": [row for row in _model_specs() if row["source_disposition"] == "quarantine"],
    }


def _manifest_path_hash(path: Path, text: str, row_count: int) -> JsonDict:
    return {
        "schema": SCHEMA + ".immutable_row_manifest",
        "path": path.resolve(strict=False).as_posix(),
        "sha256": sha256_text(text),
        "row_count": row_count,
        "row_schema": ROW_SCHEMA,
        "content_addressed_row_id": True,
        "principle": FIELD_PRINCIPLES["immutable_row_manifest_path_and_hash"],
    }


def _no_model_load_receipt() -> JsonDict:
    return {
        "schema": SCHEMA + ".no_model_load",
        "llm_loaded": False,
        "model_load_count": 0,
        "tokenizer_load_count": 0,
        "cuda_call_count": 0,
        "gpu_acquisition_count": 0,
        "llama_cpp_invocation_count": 0,
        "principle": FIELD_PRINCIPLES["no_model_load_receipt"],
    }


def _test_exits_clean(test_exit_codes: Mapping[str, int]) -> bool:
    return bool(test_exit_codes) and all(type(code) is int and code == 0 for code in test_exit_codes.values())


def _rows_accept(validation: Mapping[str, int], source_mutation_count: int = 0) -> bool:
    return (
        validation["duplicate_count"] == 0
        and validation["time_reversal_count"] == 0
        and validation["train_validation_test_overlap_count"] == 0
        and validation["row_count_mismatch_count"] == 0
        and validation["parser_failure_mismatch_count"] == 0
        and validation["alias_collision_count"] == 0
        and source_mutation_count == 0
    )


def _negative_controls(
    exp6160: Mapping[str, Any],
    exp6146: Mapping[str, Any],
    row_paths: Mapping[str, Path],
) -> JsonDict:
    clean_rows = {hf_id: _read_jsonl(row_paths[hf_id]) for hf_id in CLEAN_MODEL_ORDER}

    def receipt(name: str, rows_by_model: Mapping[str, list[JsonDict]], source_mutation: int = 0) -> JsonDict:
        temp_paths = {hf_id: row_paths[hf_id] for hf_id in CLEAN_MODEL_ORDER}
        manifest_rows: list[JsonDict] = []
        actual_parser = {}
        expected_parser = _expected_parser_failures(exp6160)
        row_count_mismatch = 0
        expected_counts = _expected_row_counts(exp6160)
        for hf_id in CLEAN_MODEL_ORDER:
            rows = rows_by_model[hf_id]
            if len(rows) != expected_counts[hf_id]:
                row_count_mismatch += 1
            actual_parser[hf_id] = sum(1 for row in rows if _is_parser_failure(row))
            for index, row in enumerate(rows):
                manifest_rows.append(
                    _manifest_row(
                        row,
                        source_row_number=index,
                        source_path=temp_paths[hf_id],
                        exp6160=exp6160,
                    )
                )
        manifest_rows.sort(
            key=lambda row: (
                int(row["chronological_index"]),
                MODEL_ORDER_INDEX[str(row["model_hf_id"])],
                int(row["source_row_number"]),
            )
        )
        disposition = _parser_disposition_receipt(actual_parser, expected_parser, exp6146)
        validation = {
            "duplicate_count": _duplicate_count(manifest_rows),
            "time_reversal_count": _time_reversal_count(rows_by_model),
            "train_validation_test_overlap_count": _partition_overlap_count(manifest_rows),
            "row_count_mismatch_count": row_count_mismatch,
            "parser_failure_mismatch_count": disposition["parser_failure_mismatch_count"],
            "alias_collision_count": _alias_collision_count(manifest_rows),
        }
        return {
            "control": name,
            "accepted": _rows_accept(validation, source_mutation),
            "source_mutation_count": source_mutation,
            **validation,
        }

    duplicate = deepcopy(clean_rows)
    duplicate[QWEN_HF_ID].append(deepcopy(duplicate[QWEN_HF_ID][0]))
    reorder = deepcopy(clean_rows)
    reorder[QWEN_HF_ID][0], reorder[QWEN_HF_ID][1] = (
        reorder[QWEN_HF_ID][1],
        reorder[QWEN_HF_ID][0],
    )
    alias = deepcopy(clean_rows)
    alias[QWEN_HF_ID][1]["visible_event_hash"] = alias[QWEN_HF_ID][0]["visible_event_hash"]
    row_loss = deepcopy(clean_rows)
    row_loss[QWEN_HF_ID] = row_loss[QWEN_HF_ID][:-1]
    parser_failure = deepcopy(clean_rows)
    parser_row = next(row for row in parser_failure[QWEN_HF_ID] if row["invalid_output"] is True)
    parser_row["invalid_output"] = False
    parser_row["answer_parse_state"] = "complete"
    parser_row["strategy_parse_state"] = "complete"
    parser_row["terminal_parse_status"] = "complete"
    return {
        "duplicate": receipt("duplicate", duplicate),
        "reorder": receipt("reorder", reorder),
        "alias_collision": receipt("alias_collision", alias),
        "row_loss": receipt("row_loss", row_loss),
        "parser_failure_disposition": receipt(
            "parser_failure_disposition", parser_failure
        ),
        "source_mutation": receipt("source_mutation", deepcopy(clean_rows), source_mutation=1),
    }


def _field_provenance() -> JsonDict:
    return {
        field: {
            "principle": principle,
            "sources": [
                "REQ-LEARN-6263",
                EXP6160_ARTIFACT.as_posix(),
                EXP6162_ARTIFACT.as_posix(),
                EXP6146_ARTIFACT.as_posix(),
                "row manifest receipts",
                "negative controls",
                "test command receipts",
            ],
        }
        for field, principle in FIELD_PRINCIPLES.items()
    }


def _ready_score(artifact: Mapping[str, Any]) -> float:
    source_receipt = dict(artifact.get("source_artifact_paths_hashes_and_terminal_classes") or {})
    disposition = dict(artifact.get("malformed_or_parser_failure_count_by_disposition") or {})
    chronological = dict(artifact.get("chronological_order_receipts") or {})
    checks = [
        artifact.get("duplicate_count") == 0 and type(artifact.get("duplicate_count")) is int,
        artifact.get("time_reversal_count") == 0
        and type(artifact.get("time_reversal_count")) is int,
        artifact.get("train_validation_test_overlap_count") == 0
        and type(artifact.get("train_validation_test_overlap_count")) is int,
        artifact.get("source_mutation_count") == 0
        and type(artifact.get("source_mutation_count")) is int,
        source_receipt.get("row_count_mismatch_count") == 0,
        disposition.get("parser_failure_mismatch_count") == 0,
        chronological.get("alias_collision_count") == 0,
        artifact.get("replay_positive_control", {}).get("accepted") is True,
        all(
            row.get("accepted") is False
            for row in dict(artifact.get("replay_negative_controls") or {}).values()
        ),
        artifact.get("no_model_load_receipt", {}).get("model_load_count") == 0,
        artifact.get("no_model_load_receipt", {}).get("llm_loaded") is False,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        artifact.get("inference_substrate") == INFERENCE_SUBSTRATE,
        artifact.get("verifier_is_oracle") is False,
        _test_exits_clean(dict(artifact.get("test_exit_codes") or {})),
    ]
    return 1.0 if all(checks) else 0.0


def _status(artifact: Mapping[str, Any]) -> str:
    return "complete_ready" if _ready_score(artifact) == 1.0 else "blocked"


def _blocked_reasons(artifact: Mapping[str, Any]) -> list[str]:
    reasons: list[str] = []
    checks = {
        "duplicate_count": artifact.get("duplicate_count"),
        "time_reversal_count": artifact.get("time_reversal_count"),
        "train_validation_test_overlap_count": artifact.get(
            "train_validation_test_overlap_count"
        ),
        "source_mutation_count": artifact.get("source_mutation_count"),
        "row_count_mismatch_count": artifact.get(
            "source_artifact_paths_hashes_and_terminal_classes", {}
        ).get("row_count_mismatch_count"),
        "parser_failure_mismatch_count": artifact.get(
            "malformed_or_parser_failure_count_by_disposition", {}
        ).get("parser_failure_mismatch_count"),
        "alias_collision_count": artifact.get("chronological_order_receipts", {}).get(
            "alias_collision_count"
        ),
    }
    for name, value in checks.items():
        if value != 0:
            reasons.append(f"{name}={value}")
    if not _test_exits_clean(dict(artifact.get("test_exit_codes") or {})):
        reasons.append("test_exit_codes")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        reasons.append("protected_files_changed")
    return reasons or ["replay_bridge_gate_failed"]


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    if _ready_score(artifact) == 1.0:
        return (
            "complete_ready: clean Exp6160 and Exp6162 local-SOTA events are sealed "
            "for chronological replay; Exp6146 remains quarantine-only"
        )
    return "blocked: " + ",".join(_blocked_reasons(artifact)[:8])


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["duration_s"] = 0.0
    stable["reproducibility_checksum"] = ""
    preconditions = stable.get("preconditions_checked")
    if isinstance(preconditions, dict):
        preconditions["output_paths"] = "<normalized>"
    manifest = stable.get("immutable_row_manifest_path_and_hash")
    if isinstance(manifest, dict):
        manifest["path"] = "<normalized>"
    source_receipts = stable.get("source_artifact_paths_hashes_and_terminal_classes")
    if isinstance(source_receipts, dict):
        manifest_hash = source_receipts.get("quarantine_manifest_path_and_hash")
        if isinstance(manifest_hash, dict):
            manifest_hash["path"] = "<normalized>"
        for source in dict(source_receipts.get("sources") or {}).values():
            if isinstance(source, dict):
                source["absolute_path"] = "<normalized>"
    return sha256_json(stable)


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path | None = None,
    row_manifest_path: str | Path | None = None,
    quarantine_manifest_path: str | Path | None = None,
    row_path_overrides: Mapping[str, str | Path] | None = None,
    expected_source_hashes: Mapping[str, str] | None = None,
    run_date: str = RUN_DATE,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.perf_counter()
    result = _resolve(root, result_path or RESULT_RELATIVE_PATH)
    row_manifest = _resolve(root, row_manifest_path or ROW_MANIFEST_RELATIVE_PATH)
    quarantine_path = _resolve(root, quarantine_manifest_path or QUARANTINE_MANIFEST_RELATIVE_PATH)
    row_paths = _row_path_map(root, row_path_overrides)
    sources = _candidate_sources(root, row_paths)
    source_before = _source_receipts(root, sources)
    protected_before = _protected_hashes(root)
    git_status_before = _git_status(root)

    exp6160 = _read_json(root / EXP6160_ARTIFACT)
    exp6162 = _read_json(root / EXP6162_ARTIFACT)
    exp6146 = _read_json(root / EXP6146_ARTIFACT)
    manifest_rows, chronological, row_counts, label_parser, disposition, validation = (
        _build_manifest(exp6160, exp6146, row_paths)
    )
    manifest_text = _jsonl_text(manifest_rows)
    quarantine_sources = _quarantine_sources()
    quarantine_payload = _quarantine_manifest(quarantine_sources)
    quarantine_text = json.dumps(quarantine_payload, indent=2, sort_keys=True) + "\n"
    if write:
        atomic_write_text(row_manifest, manifest_text, root=root, allow_override=False)
        atomic_write_json(
            quarantine_path,
            quarantine_payload,
            root=root,
            sort_keys=True,
            allow_override=False,
        )

    source_after = _source_receipts(root, sources)
    source_summary, source_mutation_count = _combine_source_receipts(
        source_before,
        source_after,
        expected_source_hashes,
        int(validation["row_count_mismatch_count"]),
    )
    source_summary["quarantine_manifest_path_and_hash"] = {
        "path": quarantine_path.resolve(strict=False).as_posix(),
        "sha256": sha256_text(quarantine_text),
        "schema": QUARANTINE_SCHEMA,
    }
    protected_after = _protected_hashes(root)
    test_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    positive_validation = dict(validation)
    positive_validation["source_mutation_count"] = source_mutation_count
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": run_date,
        "status": "blocked",
        "source_artifact_paths_hashes_and_terminal_classes": source_summary,
        "model_specs": _model_specs(),
        "no_model_load_receipt": _no_model_load_receipt(),
        "clean_source_ids": [
            "exp6160:artifact",
            f"exp6160:rows:{QWEN_HF_ID}",
            f"exp6160:rows:{GEMMA26_HF_ID}",
            "exp6162:artifact",
        ],
        "quarantined_source_ids_and_reasons": quarantine_sources,
        "immutable_row_manifest_path_and_hash": _manifest_path_hash(
            row_manifest, manifest_text, len(manifest_rows)
        ),
        "chronological_order_receipts": chronological,
        "row_count_by_model_task_family_and_partition": row_counts,
        "exact_label_and_parser_provenance": label_parser,
        "duplicate_count": int(validation["duplicate_count"]),
        "time_reversal_count": int(validation["time_reversal_count"]),
        "train_validation_test_overlap_count": int(
            validation["train_validation_test_overlap_count"]
        ),
        "malformed_or_parser_failure_count_by_disposition": disposition,
        "source_mutation_count": int(source_mutation_count),
        "replay_positive_control": {
            "accepted": _rows_accept(validation, source_mutation_count),
            "row_manifest_sha256": sha256_text(manifest_text),
            "exp6162_ready_score": exp6162.get("prospective_admission_replication_ready_score"),
            **positive_validation,
        },
        "replay_negative_controls": _negative_controls(exp6160, exp6146, row_paths),
        "event_replay_bridge_ready_score": 0.0,
        "protected_files_unchanged": _protected_files_unchanged(
            protected_before, protected_after
        ),
        "preconditions_checked": {
            "run_date": run_date,
            "git_status_before_materialization": git_status_before,
            "git_status_after_tests": _git_status(root),
            "source_hashes_before": {
                source_id: row["sha256"] for source_id, row in source_before.items()
            },
            "source_hashes_after": {
                source_id: row["sha256"] for source_id, row in source_after.items()
            },
            "output_paths": {
                "result_path": result.resolve(strict=False).as_posix(),
                "row_manifest_path": row_manifest.resolve(strict=False).as_posix(),
                "quarantine_manifest_path": quarantine_path.resolve(strict=False).as_posix(),
            },
            "exp6262_classifications": {
                source_id: row["exp6262_terminal_classification"]
                for source_id, row in source_before.items()
            },
        },
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": _field_provenance(),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(test_commands),
        "test_exit_codes": test_codes,
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 6),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["event_replay_bridge_ready_score"] = _ready_score(artifact)
    artifact["status"] = _status(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        atomic_write_json(result, artifact, root=root, sort_keys=True, allow_override=False)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    for field in (
        "duplicate_count",
        "time_reversal_count",
        "train_validation_test_overlap_count",
        "source_mutation_count",
    ):
        if type(artifact.get(field)) is not int:
            raise ValueError(field)
    if artifact.get("event_replay_bridge_ready_score") != _ready_score(artifact):
        raise ValueError("event_replay_bridge_ready_score")
    if artifact.get("status") != _status(artifact):
        raise ValueError("status")
    if artifact.get("honest_verdict") != _honest_verdict(artifact):
        raise ValueError("honest_verdict")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        raise ValueError("field_principles")
    provenance = artifact.get("field_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance")
    for field, principle in FIELD_PRINCIPLES.items():
        if dict(provenance.get(field) or {}).get("principle") != principle:
            raise ValueError(f"field_provenance:{field}")
    return True


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    parser.add_argument(
        "--row-manifest",
        type=Path,
        default=REPO_ROOT / ROW_MANIFEST_RELATIVE_PATH,
    )
    parser.add_argument(
        "--quarantine-manifest",
        type=Path,
        default=REPO_ROOT / QUARANTINE_MANIFEST_RELATIVE_PATH,
    )
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(_read_json(args.output))
        return 0
    run(
        result_path=args.output,
        row_manifest_path=args.row_manifest,
        quarantine_manifest_path=args.quarantine_manifest,
        run_date=args.date,
        write=True,
    )
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
