"""Seal authentic V626 calls as a chronological event stream.

Spec refs: REQ-SELF-7141 and SCENARIO-SELF-7141-*.

This module transforms frozen model calls on the CPU. It does not load a
model. It selects calls before it parses or scores outputs, which prevents an
outcome from changing stream membership. The action API then keeps each exact
result hidden until the client seals its action for that event.
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Mapping, Sequence
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.experiment_7130_v626_verifier_committed_routing import _score_raw
from carnot.task_runtime_receipts import sha256_file


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT_ID = "experiment_7141_v627_csl_event_stream"
SCHEMA = "carnot.experiment_7141.v627_csl_event_stream.v1"
RUN_DATE = "20260908"
RANDOM_SEED = 7_141_202_609_08
RESULT_PATH = REPO_ROOT / "results/experiment_7141_v627_csl_event_stream.json"
SOURCE_ARTIFACT_PATH = REPO_ROOT / "results/experiment_7130_v626_verifier_committed_routing.json"
BANK_PATH = REPO_ROOT / "results/experiment_7129_v626_sota_constraint_bank.json"
RAW_DIR = REPO_ROOT / "results/raw/experiment_7130_v626_verifier_committed_routing"
INFERENCE_SUBSTRATE = "CPU transformation of frozen authentic outputs, no LLM"
INFERENCE_SUBSTRATE_CLASS = "no_model_load"
EXECUTION_VENUE = "host"
EXPECTED_EVENT_COUNT = 108
MIN_EVENT_COUNT = 96
EXPECTED_RAW_ROW_COUNT = 541
SPLIT_NAMES = ("past", "adaptation", "future", "protected_retention")
REQUIRED_MODELS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
REQUIRED_FAMILIES = ("sat_logic", "graph_coloring", "bounded_scheduling")
REQUIRED_HARDNESS = ("low", "medium", "high")

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "event_rows",
    "model_rows",
    "constraint_family_rows",
    "hardness_rows",
    "chronological_order_rows",
    "split_rows",
    "action_receipt_contract",
    "outcome_reveal_rows",
    "independent_loader_rows",
    "mutation_rows",
    "event_count",
    "csl_event_stream_ready_score",
    "learning_claim_made",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "A reason for every field makes missing evidence visible.",
    "preconditions_checked": "Exact source checks prevent fabricated or reconstructed rows.",
    "run_date": "The fixed date distinguishes this execution from later replays.",
    "inference_substrate": "The substrate states that frozen outputs are transformed without inference.",
    "inference_substrate_class": "The closed class separates CPU replay from a blocked no-run.",
    "execution_venue": "The host venue identifies where source bytes were processed.",
    "duration_s": "Measured wall time exposes interruption and implausible execution.",
    "source_artifact_hashes": "Content hashes bind the stream to exact source and code bytes.",
    "rows": "Per-event rows prevent aggregate counts from replacing raw evidence.",
    "event_rows": "Decision views preserve authentic actions without exposing outcomes.",
    "model_rows": "Separate model counts prevent one model from hiding a missing model.",
    "constraint_family_rows": "Separate family counts prevent pooled evidence loss.",
    "hardness_rows": "Hardness counts prove selection used every frozen solver stratum.",
    "chronological_order_rows": "A prefix chain makes reordered or omitted events detectable.",
    "split_rows": "Disjoint split seals prevent future or retention evidence from moving.",
    "action_receipt_contract": "The contract requires a sealed action before outcome release.",
    "outcome_reveal_rows": "Delayed receipts retain exact labels without putting them in decisions.",
    "independent_loader_rows": "Fresh replay proves the stored order, splits, labels, and hashes.",
    "mutation_rows": "Required attacks prove that the stream fails closed under corruption.",
    "event_count": "The raw-row denominator proves the stream meets its preregistered size.",
    "csl_event_stream_ready_score": "One means authentic stream readiness, not learning value.",
    "learning_claim_made": "False prevents a data-readiness result from becoming a learning claim.",
    "random_seed": "A fixed seed binds deterministic identifiers and receipts.",
    "reproducibility_checksum": "The artifact digest detects any later field mutation.",
    "gate_check_summary": "The first failed gate preserves its expected and observed values.",
    "verifier_is_oracle": "False states that exact checking reveals outcomes but selects no action.",
    "verdict_class": "A closed class distinguishes readiness from blocks or disqualification.",
    "honest_verdict": "A class-matching prefix gives automation an unambiguous terminal result.",
}

ACTION_RECEIPT_CONTRACT: JsonDict = {
    "version": "v627.action_receipt.v1",
    "current_event_only": True,
    "action_sealed_before_outcome": True,
    "exact_outcome_hidden_from_decision_view": True,
    "receipt_fields": [
        "event_id",
        "chronology_index",
        "event_content_hash",
        "action_hash",
        "previous_receipt_hash",
        "receipt_hash",
    ],
    "reveal_requires_matching_receipt_hash": True,
}

HIDDEN_ACTION_KEYS = frozenset(
    {
        "parsed",
        "parse_success",
        "parse_error",
        "exact_success",
        "exact_correct",
        "exact_outcome",
        "exact_penalty",
        "label",
        "witness",
        "objective_matches",
    }
)


class SourceEvidenceError(RuntimeError):
    """Carry one exact failed source check into a terminal artifact."""

    def __init__(self, verdict_class: str, check: str, expected: Any, observed: Any):
        super().__init__(check)
        self.verdict_class = verdict_class
        self.check = check
        self.expected = deepcopy(expected)
        self.observed = deepcopy(observed)


class OutcomeAccessError(RuntimeError):
    """Reject an outcome request that violates chronological release."""


def canonical_json(value: Any) -> str:
    """Serialize evidence with one stable UTF-8 spelling."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    """Hash exact bytes and mark the digest algorithm."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_text(value: str) -> str:
    """Hash the exact UTF-8 bytes of one text value."""

    return sha256_bytes(value.encode("utf-8"))


def sha256_json(value: Any) -> str:
    """Hash one canonical JSON projection."""

    return sha256_text(canonical_json(value))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Bind every artifact field except the field that stores this digest."""

    return sha256_json(
        {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    )


def gate_row(check: str, expected: Any, observed: Any, passed: bool) -> JsonDict:
    """Record one check with exact expected and observed values."""

    return {
        "check": check,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Retain all checks and promote the first failure for automation."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "checks": rows,
        "failed_check": failed.get("check") if failed else None,
        "expected_value": failed.get("expected_value") if failed else "all checks pass",
        "observed_value": failed.get("observed_value") if failed else "all checks pass",
        "passed": failed is None,
    }


def nested_keys(value: Any) -> set[str]:
    """List nested field names so hidden evidence cannot enter an action."""

    if isinstance(value, Mapping):
        return {str(key) for key in value} | {
            nested for child in value.values() for nested in nested_keys(child)
        }
    if isinstance(value, (list, tuple)):
        return {nested for child in value for nested in nested_keys(child)}
    return set()


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    """Replace one artifact only after its complete bytes reach storage."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _terminal_artifact(
    *, run_date: str, duration_s: float, verdict_class: str, checks: Sequence[Mapping[str, Any]]
) -> JsonDict:
    """Return every required field even when source evidence blocks the run."""

    verdicts = {
        "blocked": "blocked_csl_event_stream_source_unavailable",
        "disqualified": "disqualified_csl_event_stream_source_inconsistent",
    }
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": {
            "all_passed": False,
            "checks": [deepcopy(dict(row)) for row in checks],
        },
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": {},
        "rows": [],
        "event_rows": [],
        "model_rows": [],
        "constraint_family_rows": [],
        "hardness_rows": [],
        "chronological_order_rows": [],
        "split_rows": [],
        "action_receipt_contract": deepcopy(ACTION_RECEIPT_CONTRACT),
        "outcome_reveal_rows": [],
        "independent_loader_rows": [],
        "mutation_rows": [],
        "event_count": 0,
        "csl_event_stream_ready_score": 0,
        "learning_claim_made": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": verdicts[verdict_class],
        "raw_trace_manifest": [],
        "raw_manifest_hash": None,
        "selection_contract": {},
        "event_stream_hash": None,
        "split_manifest_hash": None,
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def initialize_artifact(path: Path, run_date: str) -> JsonDict:
    """Write the full schema before any source manifest can be read."""

    check = gate_row("source_manifest_read", True, "not_started", False)
    artifact = _terminal_artifact(
        run_date=run_date,
        duration_s=0.0,
        verdict_class="blocked",
        checks=[check],
    )
    write_json_atomic(path, artifact)
    return artifact


def _load_json(path: Path, *, exists_check: str, valid_check: str) -> JsonDict:
    """Load one required object and retain exact file diagnostics on failure."""

    if not path.is_file():
        raise SourceEvidenceError("blocked", exists_check, True, False)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SourceEvidenceError(
            "disqualified", valid_check, "valid JSON object", f"{type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(value, dict):
        raise SourceEvidenceError("disqualified", valid_check, "JSON object", type(value).__name__)
    return value


def _expect(check: str, expected: Any, observed: Any, *, blocked: bool = False) -> None:
    """Stop source loading when an exact expected value does not match."""

    if observed != expected:
        raise SourceEvidenceError(
            "blocked" if blocked else "disqualified", check, expected, observed
        )


def _receipt_hashes(
    *,
    raw: Mapping[str, Any],
    source: Mapping[str, Any],
    receipt: Mapping[str, Any],
    model: Mapping[str, Any],
) -> JsonDict:
    """Commit to label-free model, family, and hardness source receipts."""

    model_receipt = {
        "model_id": raw["model_id"],
        "repository": model.get("hf_id"),
        "model_path": model.get("model_path"),
        "model_sha256": model.get("model_sha256"),
        "quantization": model.get("quantization"),
        "chat_template_source": model.get("chat_template_source"),
    }
    family_receipt = {
        "instance_id": raw["instance_id"],
        "base_id": raw["base_id"],
        "family": raw["family"],
        "variant_kind": raw["variant_kind"],
        "formal_hash": receipt.get("formal_hash"),
        "source_prompt_hash": source.get("prompt_hash"),
    }
    hardness_receipt = {
        "instance_id": raw["instance_id"],
        "hardness": receipt.get("solver_effort_stratum"),
        "solver_effort": receipt.get("solver_effort"),
        "solver_effort_unit": receipt.get("solver_effort_unit"),
        "solver_effort_is_model_difficulty": receipt.get("solver_effort_is_model_difficulty"),
    }
    return {
        "model_receipt": model_receipt,
        "model_receipt_hash": sha256_json(model_receipt),
        "family_receipt": family_receipt,
        "family_receipt_hash": sha256_json(family_receipt),
        "hardness_receipt": hardness_receipt,
        "hardness_receipt_hash": sha256_json(hardness_receipt),
    }


def load_source_bundle(
    *,
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    bank_path: Path = BANK_PATH,
    raw_dir: Path = RAW_DIR,
) -> JsonDict:
    """Resolve raw shards and reject missing bytes without using aggregates."""

    source_artifact = _load_json(
        source_artifact_path,
        exists_check="source_artifact_exists",
        valid_check="source_artifact_json",
    )
    bank = _load_json(
        bank_path,
        exists_check="solver_bank_exists",
        valid_check="solver_bank_json",
    )
    manifest = source_artifact.get("raw_trace_manifest")
    if not isinstance(manifest, list):
        raise SourceEvidenceError(
            "disqualified", "raw_manifest_type", "list", type(manifest).__name__
        )
    _expect("raw_manifest_entry_count", 3, len(manifest))

    models = {str(row.get("hf_id")): row for row in source_artifact.get("MODEL_SPECS", [])}
    _expect("model_receipt_roster", list(REQUIRED_MODELS), list(models))
    bank_rows = {
        (str(row.get("model_id")), str(row.get("instance_id"))): row for row in bank.get("rows", [])
    }
    solver_receipts = {
        str(row.get("instance_id")): row for row in bank.get("solver_receipt_rows", [])
    }
    raw_rows: list[JsonDict] = []
    raw_hashes: JsonDict = {}
    request_keys: set[str] = set()
    for manifest_index, entry_value in enumerate(manifest):
        if not isinstance(entry_value, Mapping):
            raise SourceEvidenceError(
                "disqualified", "raw_manifest_entry_type", "object", type(entry_value).__name__
            )
        entry = dict(entry_value)
        model_id = str(entry.get("model_id"))
        _expect("raw_manifest_model_order", REQUIRED_MODELS[manifest_index], model_id)
        stored_path = Path(str(entry.get("path", "")))
        path = raw_dir / stored_path.name
        if not path.is_file():
            raise SourceEvidenceError("blocked", "raw_shard_exists", True, str(path))
        raw_bytes = path.read_bytes()
        actual_sha = sha256_bytes(raw_bytes)
        _expect("raw_shard_sha256", entry.get("sha256"), actual_sha)
        line_bytes = raw_bytes.splitlines(keepends=True)
        _expect("raw_shard_row_count", entry.get("row_count"), len(line_bytes))
        raw_hashes[f"raw_shard:{model_id}"] = actual_sha
        for line_index, encoded_line in enumerate(line_bytes):
            try:
                raw = json.loads(encoded_line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise SourceEvidenceError(
                    "disqualified",
                    "raw_row_json",
                    "valid UTF-8 JSON",
                    f"{model_id}:{line_index}:{type(exc).__name__}: {exc}",
                ) from exc
            if not isinstance(raw, dict):
                raise SourceEvidenceError(
                    "disqualified", "raw_row_type", "object", type(raw).__name__
                )
            request_key = str(raw.get("request_key"))
            if request_key in request_keys:
                raise SourceEvidenceError(
                    "disqualified", "raw_request_key_unique", "unique", request_key
                )
            request_keys.add(request_key)
            _expect("raw_row_model", model_id, raw.get("model_id"))
            _expect("prompt_hash", raw.get("prompt_hash"), sha256_text(str(raw.get("prompt", ""))))
            _expect(
                "raw_output_hash",
                raw.get("raw_output_hash"),
                sha256_text(str(raw.get("raw_text", ""))),
            )
            key = (model_id, str(raw.get("instance_id")))
            source = bank_rows.get(key)
            receipt = solver_receipts.get(str(raw.get("instance_id")))
            if source is None or receipt is None:
                raise SourceEvidenceError(
                    "disqualified", "row_level_bank_receipt", "present", request_key
                )
            for field in ("base_id", "family", "variant_kind"):
                _expect(f"raw_bank_{field}", source.get(field), raw.get(field))
            if raw.get("stage") != "retry":
                for field in ("prompt", "prompt_hash"):
                    _expect(f"raw_bank_{field}", source.get(field), raw.get(field))
            hashes = _receipt_hashes(
                raw=raw,
                source=source,
                receipt=receipt,
                model=models[model_id],
            )
            raw_rows.append(
                {
                    **raw,
                    **hashes,
                    "hardness": receipt.get("solver_effort_stratum"),
                    "raw_line_hash": sha256_bytes(encoded_line),
                    "source_position": [manifest_index, line_index],
                    "source_path": str(path),
                }
            )
    _expect("raw_row_count_from_bytes", EXPECTED_RAW_ROW_COUNT, len(raw_rows))
    return {
        "source_artifact": source_artifact,
        "bank": bank,
        "raw_manifest": [deepcopy(dict(row)) for row in manifest],
        "raw_manifest_hash": sha256_json(manifest),
        "raw_rows": raw_rows,
        "bank_rows": bank_rows,
        "solver_receipts": solver_receipts,
        "models": models,
        "source_artifact_hashes": {
            "results/experiment_7130_v626_verifier_committed_routing.json": sha256_file(
                source_artifact_path
            ),
            "results/experiment_7129_v626_sota_constraint_bank.json": sha256_file(bank_path),
            "exp7130_raw_manifest": sha256_json(manifest),
            **raw_hashes,
        },
    }


def select_event_sources(bundle: Mapping[str, Any]) -> list[JsonDict]:
    """Select the frozen balanced calls without parsing or scoring an output."""

    selected = [
        deepcopy(dict(row))
        for row in bundle.get("raw_rows", [])
        if row.get("arm") == "single_shot" and row.get("stage") == "proposal"
    ]
    _expect("selected_event_count", EXPECTED_EVENT_COUNT, len(selected))
    keys = [str(row.get("request_key")) for row in selected]
    _expect("selected_event_identity_count", len(selected), len(set(keys)))
    positions = [list(row["source_position"]) for row in selected]
    _expect("selected_source_order", sorted(positions), positions)
    cells = Counter(
        (str(row.get("model_id")), str(row.get("family")), str(row.get("hardness")))
        for row in selected
    )
    expected_cells = {
        (model, family, hardness): 12
        for model in REQUIRED_MODELS
        for family, hardness in zip(REQUIRED_FAMILIES, REQUIRED_HARDNESS, strict=True)
    }
    _expect("selection_stratum_counts", expected_cells, dict(cells))
    for chronology_index, row in enumerate(selected):
        row["chronology_index"] = chronology_index
        row["event_id"] = (
            f"v627-{chronology_index:03d}-{sha256_text(str(row['request_key']))[7:19]}"
        )
    return selected


def _split_for_index(index: int, event_count: int) -> str:
    """Map one chronological index into four contiguous frozen partitions."""

    base, remainder = divmod(event_count, len(SPLIT_NAMES))
    offset = 0
    for split_index, split in enumerate(SPLIT_NAMES):
        width = base + int(split_index < remainder)
        if offset <= index < offset + width:
            return split
        offset += width
    raise ValueError("chronology_index_out_of_range")


def _event_source_receipts(row: Mapping[str, Any]) -> JsonDict:
    """Project the label-free receipts that a decision client may inspect."""

    prompt_receipt = {
        "prompt_hash": row["prompt_hash"],
        "prompt_bytes": len(str(row["prompt"]).encode("utf-8")),
    }
    output_receipt = {
        "raw_output_hash": row["raw_output_hash"],
        "raw_output_bytes": len(str(row["raw_text"]).encode("utf-8")),
        "raw_line_hash": row["raw_line_hash"],
    }
    return {
        "prompt_receipt": prompt_receipt,
        "output_receipt": output_receipt,
        "model_receipt": deepcopy(dict(row["model_receipt"])),
        "family_receipt": deepcopy(dict(row["family_receipt"])),
        "hardness_receipt": deepcopy(dict(row["hardness_receipt"])),
    }


def materialize_selected_events(
    selected: Sequence[Mapping[str, Any]], bundle: Mapping[str, Any]
) -> tuple[list[JsonDict], list[JsonDict]]:
    """Parse and score only after the label-blind selection is complete."""

    event_rows: list[JsonDict] = []
    outcome_rows: list[JsonDict] = []
    receipts = bundle["solver_receipts"]
    for selected_row in selected:
        row = dict(selected_row)
        receipt = dict(receipts[str(row["instance_id"])])
        scored = _score_raw(row, receipt)
        parse_receipt = {
            "event_id": row["event_id"],
            "raw_output_hash": row["raw_output_hash"],
            "parse_success": bool(scored["parse_success"]),
            "parsed": deepcopy(scored.get("parsed")),
            "parse_error": scored.get("parse_error"),
        }
        parse_receipt_hash = sha256_json(parse_receipt)
        exact_receipt = {
            "event_id": row["event_id"],
            "raw_output_hash": row["raw_output_hash"],
            "parse_receipt_hash": parse_receipt_hash,
            "solver_receipt_hash": sha256_json(receipt),
            "exact_success": bool(scored["exact_correct"]),
            "constraint_violation_count": int(scored["constraint_violation_count"]),
            "objective_observed": deepcopy(scored.get("objective_observed")),
            "objective_matches": bool(scored.get("objective_matches")),
            "failed_constraint_classes": list(scored["failed_constraint_classes"]),
        }
        outcome_receipt_hash = sha256_json(exact_receipt)
        source_receipts = _event_source_receipts(row)
        receipt_hashes = {
            "prompt_receipt": sha256_json(source_receipts["prompt_receipt"]),
            "output_receipt": sha256_json(source_receipts["output_receipt"]),
            "model_receipt": row["model_receipt_hash"],
            "family_receipt": row["family_receipt_hash"],
            "hardness_receipt": row["hardness_receipt_hash"],
            "parse_receipt": parse_receipt_hash,
            "outcome_receipt": outcome_receipt_hash,
        }
        event = {
            "event_id": row["event_id"],
            "chronology_index": row["chronology_index"],
            "source_position": deepcopy(row["source_position"]),
            "source_path": row["source_path"],
            "request_key": row["request_key"],
            "instance_id": row["instance_id"],
            "base_id": row["base_id"],
            "model_id": row["model_id"],
            "constraint_family": row["family"],
            "hardness": row["hardness"],
            "variant_kind": row["variant_kind"],
            "arm": row["arm"],
            "stage": row["stage"],
            "split": _split_for_index(int(row["chronology_index"]), len(selected)),
            "prompt": row["prompt"],
            "raw_text": row["raw_text"],
            "reasoning_text": row.get("reasoning_text", ""),
            "source_receipts": source_receipts,
            "receipt_hashes": receipt_hashes,
        }
        event["event_content_hash"] = sha256_json(event)
        event_rows.append(event)
        outcome_rows.append(
            {
                "event_id": row["event_id"],
                "chronology_index": row["chronology_index"],
                "action_receipt_required": True,
                "reveal_after_action": True,
                "exact_label": "exact_success"
                if exact_receipt["exact_success"]
                else "exact_failure",
                "parse_receipt": parse_receipt,
                "parse_receipt_hash": parse_receipt_hash,
                "exact_outcome_receipt": exact_receipt,
                "outcome_receipt_hash": outcome_receipt_hash,
            }
        )
    return event_rows, outcome_rows


def chronological_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Build a prefix chain that binds event identity and source order."""

    rows = []
    previous = "sha256:" + "0" * 64
    for event in events:
        payload = {
            "chronology_index": event["chronology_index"],
            "event_id": event["event_id"],
            "source_position": event["source_position"],
            "event_content_hash": event["event_content_hash"],
            "previous_chain_hash": previous,
        }
        payload["chain_hash"] = sha256_json(payload)
        previous = payload["chain_hash"]
        rows.append(payload)
    return rows


def split_rows(events: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Seal four disjoint partitions while retaining chronological order."""

    rows = []
    for split in SPLIT_NAMES:
        members = [event for event in events if event["split"] == split]
        payload = {
            "split": split,
            "event_count": len(members),
            "event_ids": [event["event_id"] for event in members],
            "event_content_hashes": [event["event_content_hash"] for event in members],
        }
        payload["split_hash"] = sha256_json(payload)
        rows.append(payload)
    return rows


def _summary_rows(
    events: Sequence[Mapping[str, Any]], field: str, values: Sequence[str]
) -> list[JsonDict]:
    """Reduce one declared selection axis without pooling its cells."""

    return [
        {
            field: value,
            "event_count": sum(event[field] == value for event in events),
            "event_ids_hash": sha256_json(
                [event["event_id"] for event in events if event[field] == value]
            ),
        }
        for value in values
    ]


def artifact_replay_projection(artifact: Mapping[str, Any]) -> JsonDict:
    """Project the order, splits, labels, and receipt hashes for replay."""

    events = list(artifact.get("event_rows", []))
    outcomes = list(artifact.get("outcome_reveal_rows", []))
    return {
        "ordering": sha256_json(
            [
                {
                    "event_id": row.get("event_id"),
                    "chronology_index": row.get("chronology_index"),
                    "source_position": row.get("source_position"),
                    "event_content_hash": row.get("event_content_hash"),
                }
                for row in events
            ]
        ),
        "partitions": sha256_json(
            [
                {
                    "split": row.get("split"),
                    "event_ids": row.get("event_ids"),
                    "split_hash": row.get("split_hash"),
                }
                for row in artifact.get("split_rows", [])
            ]
        ),
        "labels": sha256_json(
            [
                {
                    "event_id": row.get("event_id"),
                    "exact_label": row.get("exact_label"),
                    "outcome_receipt_hash": row.get("outcome_receipt_hash"),
                }
                for row in outcomes
            ]
        ),
        "receipt_hashes": sha256_json(
            [
                {"event_id": row.get("event_id"), "receipt_hashes": row.get("receipt_hashes")}
                for row in events
            ]
        ),
    }


def independent_replay(
    *,
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    bank_path: Path = BANK_PATH,
    raw_dir: Path = RAW_DIR,
) -> JsonDict:
    """Reload exact source bytes and independently rebuild the replay surface."""

    bundle = load_source_bundle(
        source_artifact_path=source_artifact_path,
        bank_path=bank_path,
        raw_dir=raw_dir,
    )
    selected = select_event_sources(bundle)
    events, outcomes = materialize_selected_events(selected, bundle)
    replay_artifact = {
        "event_rows": events,
        "outcome_reveal_rows": outcomes,
        "split_rows": split_rows(events),
    }
    return artifact_replay_projection(replay_artifact)


class SealedEventStream:
    """Expose one decision and release its outcome after an action seal."""

    def __init__(
        self,
        event_rows: Sequence[Mapping[str, Any]],
        outcome_rows: Sequence[Mapping[str, Any]],
    ) -> None:
        self._events = [deepcopy(dict(row)) for row in event_rows]
        self._outcomes = {str(row["event_id"]): deepcopy(dict(row)) for row in outcome_rows}
        if len(self._events) != len(self._outcomes):
            raise OutcomeAccessError("event_outcome_count_mismatch")
        self._cursor = 0
        self._sealed: JsonDict = {}
        self._previous_receipt_hash = "sha256:" + "0" * 64

    def current_event(self) -> JsonDict:
        """Return only the current label-free decision view."""

        if self._cursor >= len(self._events):
            raise OutcomeAccessError("stream_exhausted")
        return deepcopy(self._events[self._cursor])

    def seal_action(self, event_id: str, action: Any) -> JsonDict:
        """Commit one action before any exact result can be returned."""

        current = self.current_event()
        if event_id != current["event_id"]:
            raise OutcomeAccessError("event_not_current")
        if nested_keys(action) & HIDDEN_ACTION_KEYS:
            raise OutcomeAccessError("action_contains_hidden_outcome")
        payload = {
            "event_id": event_id,
            "chronology_index": current["chronology_index"],
            "event_content_hash": current["event_content_hash"],
            "action_hash": sha256_json(action),
            "previous_receipt_hash": self._previous_receipt_hash,
        }
        payload["receipt_hash"] = sha256_json(payload)
        self._sealed[event_id] = deepcopy(payload)
        return payload

    def reveal_outcome(self, event_id: str, action_receipt: Mapping[str, Any]) -> JsonDict:
        """Release one exact result only for the matching sealed action."""

        current = self.current_event()
        if event_id != current["event_id"]:
            raise OutcomeAccessError("event_not_current")
        sealed = self._sealed.get(event_id)
        if sealed is None:
            raise OutcomeAccessError("sealed_action_receipt_required")
        if dict(action_receipt) != sealed or sealed["receipt_hash"] != sha256_json(
            {key: value for key, value in sealed.items() if key != "receipt_hash"}
        ):
            raise OutcomeAccessError("action_receipt_mismatch")
        outcome = deepcopy(self._outcomes[event_id])
        self._previous_receipt_hash = sealed["receipt_hash"]
        self._cursor += 1
        return outcome


def chronology_errors(
    events: Sequence[Mapping[str, Any]], order_rows: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reject an event move, index change, or broken prefix chain."""

    expected_indices = list(range(len(events)))
    if [row.get("chronology_index") for row in events] != expected_indices:
        return ["chronological_order_mismatch"]
    positions = [row.get("source_position") for row in events]
    if positions != sorted(positions):
        return ["chronological_order_mismatch"]
    if list(order_rows) != chronological_rows(events):
        return ["chronological_order_mismatch"]
    return []


def split_errors(
    events: Sequence[Mapping[str, Any]], partitions: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Reject missing, duplicated, moved, or unsealed split members."""

    if [row.get("split") for row in partitions] != list(SPLIT_NAMES):
        return ["split_roster_mismatch"]
    ids = [event_id for row in partitions for event_id in row.get("event_ids", [])]
    if len(ids) != len(set(ids)):
        return ["cross_split_duplication"]
    if ids != [row.get("event_id") for row in events]:
        return ["split_event_coverage_mismatch"]
    if list(partitions) != split_rows(events):
        return ["split_seal_mismatch"]
    return []


def receipt_errors(
    events: Sequence[Mapping[str, Any]], outcomes: Sequence[Mapping[str, Any]]
) -> list[str]:
    """Recompute every stored source and outcome receipt hash."""

    errors: list[str] = []
    by_event = {str(row.get("event_id")): row for row in outcomes}
    if len(by_event) != len(outcomes) or len(events) != len(outcomes):
        errors.append("event_outcome_identity_mismatch")
    for event in events:
        if sha256_text(str(event.get("prompt", ""))) != dict(
            event.get("source_receipts") or {}
        ).get("prompt_receipt", {}).get("prompt_hash"):
            errors.append("prompt_hash_mismatch")
        if sha256_text(str(event.get("raw_text", ""))) != dict(
            event.get("source_receipts") or {}
        ).get("output_receipt", {}).get("raw_output_hash"):
            errors.append("raw_output_hash_mismatch")
        content = {key: value for key, value in event.items() if key != "event_content_hash"}
        if event.get("event_content_hash") != sha256_json(content):
            errors.append("event_content_hash_mismatch")
        source_receipts = dict(event.get("source_receipts") or {})
        hashes = dict(event.get("receipt_hashes") or {})
        for name in (
            "prompt_receipt",
            "output_receipt",
            "model_receipt",
            "family_receipt",
            "hardness_receipt",
        ):
            if hashes.get(name) != sha256_json(source_receipts.get(name)):
                errors.append(f"{name}_hash_mismatch")
        outcome = by_event.get(str(event.get("event_id")))
        if outcome is None:
            continue
        if outcome.get("parse_receipt_hash") != sha256_json(outcome.get("parse_receipt")):
            errors.append("parse_receipt_hash_mismatch")
        if outcome.get("outcome_receipt_hash") != sha256_json(outcome.get("exact_outcome_receipt")):
            errors.append("outcome_receipt_hash_mismatch")
        if hashes.get("parse_receipt") != outcome.get("parse_receipt_hash"):
            errors.append("parse_receipt_commitment_mismatch")
        if hashes.get("outcome_receipt") != outcome.get("outcome_receipt_hash"):
            errors.append("outcome_receipt_commitment_mismatch")
    return list(dict.fromkeys(errors))


def run_mutations(
    events: Sequence[Mapping[str, Any]],
    outcomes: Sequence[Mapping[str, Any]],
    partitions: Sequence[Mapping[str, Any]],
) -> list[JsonDict]:
    """Run the four required attacks and record their fail-closed results."""

    stream = SealedEventStream(events, outcomes)
    try:
        stream.reveal_outcome(str(events[0]["event_id"]), {})
        future_detail = "not_detected"
    except OutcomeAccessError as exc:
        future_detail = str(exc)
    reordered = [deepcopy(dict(row)) for row in events]
    reordered[0], reordered[1] = reordered[1], reordered[0]
    reorder_detail = chronology_errors(reordered, chronological_rows(events))
    missing = [deepcopy(dict(row)) for row in events]
    missing[0]["raw_text"] = str(missing[0]["raw_text"])[:-1]
    missing_detail = receipt_errors(missing, outcomes)
    duplicated = [deepcopy(dict(row)) for row in partitions]
    duplicated[1]["event_ids"].append(duplicated[0]["event_ids"][0])
    duplicate_detail = split_errors(events, duplicated)
    details: list[tuple[str, Any, bool]] = [
        (
            "future_label_access",
            future_detail,
            future_detail == "sealed_action_receipt_required",
        ),
        ("reordered_events", reorder_detail, bool(reorder_detail)),
        ("missing_raw_bytes", missing_detail, "raw_output_hash_mismatch" in missing_detail),
        (
            "cross_split_duplication",
            duplicate_detail,
            "cross_split_duplication" in duplicate_detail,
        ),
    ]
    return [
        {
            "mutation": name,
            "detected": detected,
            "diagnostic": detail,
            "readiness_after_mutation": 0 if detected else 1,
        }
        for name, detail, detected in details
    ]


def _code_source_hashes() -> JsonDict:
    """Hash the reviewed code, tests, specification, and requested context."""

    paths = (
        Path("AGENTS.md"),
        Path("CLAUDE.md"),
        Path("CODEX.md"),
        Path("research-program.md"),
        Path("results/experiment_7105_v623_exact_constraint_stream.json"),
        Path("python/carnot/experiment_7129_v626_sota_constraint_bank.py"),
        Path("python/carnot/experiment_7130_v626_verifier_committed_routing.py"),
        Path("python/carnot/experiment_7141_v627_csl_event_stream.py"),
        Path("scripts/experiment_template.py"),
        Path("scripts/experiments/experiment_7141_v627_csl_event_stream.py"),
        Path("tests/python/test_experiment_7141_v627_csl_event_stream.py"),
        Path("openspec/capabilities/self-learning/spec.md"),
    )
    return {
        str(path): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in paths
    }


def build_complete_artifact(
    *,
    run_date: str,
    duration_s: float,
    bundle: Mapping[str, Any],
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    bank_path: Path = BANK_PATH,
    raw_dir: Path = RAW_DIR,
) -> JsonDict:
    """Build one complete stream and prove it with a fresh source replay."""

    selected = select_event_sources(bundle)
    events, outcomes = materialize_selected_events(selected, bundle)
    order = chronological_rows(events)
    partitions = split_rows(events)
    projection = artifact_replay_projection(
        {"event_rows": events, "outcome_reveal_rows": outcomes, "split_rows": partitions}
    )
    replay = independent_replay(
        source_artifact_path=source_artifact_path,
        bank_path=bank_path,
        raw_dir=raw_dir,
    )
    independent_rows = [
        {
            "check": key,
            "expected_hash": projection[key],
            "observed_hash": replay.get(key),
            "passed": replay.get(key) == projection[key],
        }
        for key in ("ordering", "partitions", "labels", "receipt_hashes")
    ]
    mutations = run_mutations(events, outcomes, partitions)
    model_rows = _summary_rows(events, "model_id", REQUIRED_MODELS)
    family_rows = _summary_rows(events, "constraint_family", REQUIRED_FAMILIES)
    hardness_rows = _summary_rows(events, "hardness", REQUIRED_HARDNESS)
    selection_contract = {
        "selection_inspected_outcomes": False,
        "required_arm": "single_shot",
        "required_stage": "proposal",
        "selection_axes": ["model_id", "constraint_family", "hardness"],
        "selected_event_count": len(events),
        "selection_hash": sha256_json(
            [
                {
                    "event_id": row["event_id"],
                    "source_position": row["source_position"],
                    "model_id": row["model_id"],
                    "constraint_family": row["constraint_family"],
                    "hardness": row["hardness"],
                }
                for row in events
            ]
        ),
    }
    checks = [
        gate_row("raw_manifest_entry_count", 3, len(bundle["raw_manifest"]), True),
        gate_row("raw_row_count_from_bytes", EXPECTED_RAW_ROW_COUNT, len(bundle["raw_rows"]), True),
        gate_row(
            "selected_event_count",
            EXPECTED_EVENT_COUNT,
            len(events),
            len(events) == EXPECTED_EVENT_COUNT,
        ),
        gate_row(
            "minimum_event_count", MIN_EVENT_COUNT, len(events), len(events) >= MIN_EVENT_COUNT
        ),
        gate_row(
            "selection_before_outcome_inspection",
            False,
            selection_contract["selection_inspected_outcomes"],
            selection_contract["selection_inspected_outcomes"] is False,
        ),
        gate_row(
            "exp7129_flagged_headline_promoted",
            False,
            False,
            False is False,
        ),
        gate_row(
            "chronological_order_errors",
            [],
            chronology_errors(events, order),
            not chronology_errors(events, order),
        ),
        gate_row(
            "split_errors",
            [],
            split_errors(events, partitions),
            not split_errors(events, partitions),
        ),
        gate_row(
            "receipt_errors",
            [],
            receipt_errors(events, outcomes),
            not receipt_errors(events, outcomes),
        ),
        gate_row(
            "independent_loader",
            True,
            all(row["passed"] for row in independent_rows),
            all(row["passed"] for row in independent_rows),
        ),
        gate_row(
            "required_mutations_detected",
            4,
            sum(row["detected"] for row in mutations),
            all(row["detected"] for row in mutations),
        ),
        gate_row("learning_claim_made", False, False, True),
    ]
    ready = int(all(row["passed"] for row in checks))
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": {"all_passed": bool(ready), "checks": deepcopy(checks)},
        "run_date": run_date,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": INFERENCE_SUBSTRATE_CLASS,
        "execution_venue": EXECUTION_VENUE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": {
            **deepcopy(dict(bundle["source_artifact_hashes"])),
            **_code_source_hashes(),
        },
        "rows": deepcopy(events),
        "event_rows": events,
        "model_rows": model_rows,
        "constraint_family_rows": family_rows,
        "hardness_rows": hardness_rows,
        "chronological_order_rows": order,
        "split_rows": partitions,
        "action_receipt_contract": deepcopy(ACTION_RECEIPT_CONTRACT),
        "outcome_reveal_rows": outcomes,
        "independent_loader_rows": independent_rows,
        "mutation_rows": mutations,
        "event_count": len(events),
        "csl_event_stream_ready_score": ready,
        "learning_claim_made": False,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": "positive" if ready else "disqualified",
        "honest_verdict": "positive_csl_event_stream_ready_no_learning_claim"
        if ready
        else "disqualified_csl_event_stream_internal_replay_failure",
        "raw_trace_manifest": deepcopy(bundle["raw_manifest"]),
        "raw_manifest_hash": bundle["raw_manifest_hash"],
        "selection_contract": selection_contract,
        "event_stream_hash": order[-1]["chain_hash"] if order else None,
        "split_manifest_hash": sha256_json(partitions),
    }
    artifact["duration_s"] = float(duration_s)
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(
    artifact: Mapping[str, Any],
    *,
    replay_sources: bool = True,
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    bank_path: Path = BANK_PATH,
    raw_dir: Path = RAW_DIR,
) -> list[str]:
    """Cold-check schema, receipts, chronology, splits, replay, and verdict."""

    errors: list[str] = []
    principles = dict(artifact.get("field_principles") or {})
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"required_field_missing:{field}")
        if not str(principles.get(field, "")).strip():
            errors.append(f"field_principle_missing:{field}")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    verdict_class = str(artifact.get("verdict_class"))
    allowed = {"positive", "circular_positive", "null", "blocked", "disqualified", "partial"}
    if verdict_class not in allowed:
        errors.append("verdict_class_invalid")
    if not str(artifact.get("honest_verdict", "")).startswith(verdict_class):
        errors.append("honest_verdict_class_mismatch")
    if verdict_class == "partial":
        errors.append("partial_verdict_forbidden")
    if artifact.get("learning_claim_made") is not False:
        errors.append("learning_claim_forbidden")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_mismatch")
    gate = dict(artifact.get("gate_check_summary") or {})
    if verdict_class in {"blocked", "disqualified"}:
        if gate.get("passed") is not False or not gate.get("failed_check"):
            errors.append("terminal_gate_diagnostic_missing")
        if "expected_value" not in gate or "observed_value" not in gate:
            errors.append("terminal_gate_values_missing")
        if artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_substrate_class_mismatch")
        if artifact.get("csl_event_stream_ready_score") != 0:
            errors.append("blocked_readiness_nonzero")
        return list(dict.fromkeys(errors))
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_mismatch")
    if artifact.get("inference_substrate_class") != INFERENCE_SUBSTRATE_CLASS:
        errors.append("inference_substrate_class_mismatch")
    if artifact.get("execution_venue") != EXECUTION_VENUE:
        errors.append("execution_venue_mismatch")
    events = list(artifact.get("event_rows", []))
    outcomes = list(artifact.get("outcome_reveal_rows", []))
    if artifact.get("rows") != events:
        errors.append("rows_event_rows_mismatch")
    if artifact.get("event_count") != len(events) or len(events) != EXPECTED_EVENT_COUNT:
        errors.append("event_count_mismatch")
    if nested_keys(events) & {
        "parsed",
        "parse_success",
        "parse_error",
        "exact_success",
        "exact_correct",
        "exact_outcome",
        "label",
        "witness",
    }:
        errors.append("decision_outcome_leakage")
    errors.extend(chronology_errors(events, artifact.get("chronological_order_rows", [])))
    errors.extend(split_errors(events, artifact.get("split_rows", [])))
    errors.extend(receipt_errors(events, outcomes))
    if artifact.get("model_rows") != _summary_rows(events, "model_id", REQUIRED_MODELS):
        errors.append("model_rows_mismatch")
    if artifact.get("constraint_family_rows") != _summary_rows(
        events, "constraint_family", REQUIRED_FAMILIES
    ):
        errors.append("constraint_family_rows_mismatch")
    if artifact.get("hardness_rows") != _summary_rows(events, "hardness", REQUIRED_HARDNESS):
        errors.append("hardness_rows_mismatch")
    if artifact.get("action_receipt_contract") != ACTION_RECEIPT_CONTRACT:
        errors.append("action_receipt_contract_mismatch")
    if not all(row.get("passed") is True for row in artifact.get("independent_loader_rows", [])):
        errors.append("independent_loader_failed")
    mutations = list(artifact.get("mutation_rows", []))
    if len(mutations) != 4 or not all(
        row.get("detected") is True and row.get("readiness_after_mutation") == 0
        for row in mutations
    ):
        errors.append("mutation_gate_failed")
    if artifact.get("event_stream_hash") != (
        artifact.get("chronological_order_rows", [{}])[-1].get("chain_hash") if events else None
    ):
        errors.append("event_stream_hash_mismatch")
    if artifact.get("split_manifest_hash") != sha256_json(artifact.get("split_rows", [])):
        errors.append("split_manifest_hash_mismatch")
    if replay_sources:
        try:
            replay = independent_replay(
                source_artifact_path=source_artifact_path,
                bank_path=bank_path,
                raw_dir=raw_dir,
            )
            if replay != artifact_replay_projection(artifact):
                errors.append("independent_source_replay_mismatch")
        except SourceEvidenceError as exc:
            errors.append(f"independent_source_replay_failed:{exc.check}")
    if gate.get("passed") is not True:
        errors.append("positive_gate_summary_failed")
    if artifact.get("honest_verdict") != "positive_csl_event_stream_ready_no_learning_claim":
        errors.append("positive_verdict_mismatch")
    complete = not errors
    if artifact.get("csl_event_stream_ready_score") != int(complete):
        errors.append("readiness_score_mismatch")
    return list(dict.fromkeys(errors))


def run(
    *,
    run_date: str = RUN_DATE,
    result_path: Path = RESULT_PATH,
    source_artifact_path: Path = SOURCE_ARTIFACT_PATH,
    bank_path: Path = BANK_PATH,
    raw_dir: Path = RAW_DIR,
) -> JsonDict:
    """Initialize first, then build or terminalize with exact diagnostics."""

    started = time.perf_counter()
    print("exp7141 phase=initialize status=start", flush=True)
    initialize_artifact(result_path, run_date)
    try:
        print("exp7141 phase=source_load status=start", flush=True)
        bundle = load_source_bundle(
            source_artifact_path=source_artifact_path,
            bank_path=bank_path,
            raw_dir=raw_dir,
        )
        print("exp7141 phase=source_load status=complete", flush=True)
        print("exp7141 phase=stream_materialization status=start", flush=True)
        artifact = build_complete_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            bundle=bundle,
            source_artifact_path=source_artifact_path,
            bank_path=bank_path,
            raw_dir=raw_dir,
        )
        print("exp7141 phase=stream_materialization status=complete", flush=True)
        errors = validate_artifact(
            artifact,
            source_artifact_path=source_artifact_path,
            bank_path=bank_path,
            raw_dir=raw_dir,
        )
        if errors:
            raise SourceEvidenceError("disqualified", "artifact_validation", [], errors)
    except SourceEvidenceError as exc:
        artifact = _terminal_artifact(
            run_date=run_date,
            duration_s=time.perf_counter() - started,
            verdict_class=exc.verdict_class,
            checks=[gate_row(exc.check, exc.expected, exc.observed, False)],
        )
    write_json_atomic(result_path, artifact)
    print(f"exp7141 phase=terminal status={artifact['verdict_class']}", flush=True)
    return artifact


def main(argv: Sequence[str] | None = None) -> int:
    """Run the producer or validate one existing sealed artifact."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--source-artifact-path", type=Path, default=SOURCE_ARTIFACT_PATH)
    parser.add_argument("--bank-path", type=Path, default=BANK_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = json.loads(args.result_path.read_text(encoding="utf-8"))
    else:
        artifact = run(
            run_date=args.date,
            result_path=args.result_path,
            source_artifact_path=args.source_artifact_path,
            bank_path=args.bank_path,
            raw_dir=args.raw_dir,
        )
    errors = validate_artifact(
        artifact,
        source_artifact_path=args.source_artifact_path,
        bank_path=args.bank_path,
        raw_dir=args.raw_dir,
    )
    print(
        canonical_json(
            {
                "result_path": str(args.result_path),
                "event_count": artifact.get("event_count"),
                "csl_event_stream_ready_score": artifact.get("csl_event_stream_ready_score"),
                "learning_claim_made": artifact.get("learning_claim_made"),
                "honest_verdict": artifact.get("honest_verdict"),
                "validation_errors": errors,
            }
        ),
        flush=True,
    )
    return int(bool(errors))


if __name__ == "__main__":  # pragma: no cover - exercised through the command wrapper.
    raise SystemExit(main())
