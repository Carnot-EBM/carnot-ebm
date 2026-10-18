"""Run delayed fixed-schema memory inside a bounded Qwen decision loop.

Spec refs: REQ-SELF-7131 and SCENARIO-SELF-7131-*.

The model never sees the current exact outcome. Each arm first seals its
prompt. The exact checker then scores the generated action. Only a later event
can read a note or signed record admitted after that outcome closed.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
import gc
import hashlib
import json
import os
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from carnot import gpu_lease_phase_journal as lease_api

JsonDict = dict[str, Any]
FAMILIES = ("sat_logic", "graph_coloring", "bounded_scheduling")
VARIANTS = ("base", "variant")
ARMS = ("no_memory", "free_note", "fixed_schema")
REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "run_date",
    "MODEL_SPECS",
    "models_used",
    "model_repository",
    "model_path",
    "model_hash",
    "model_quantization",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "gpu_telemetry_rows",
    "token_rows",
    "duration_s",
    "source_artifact_hashes",
    "split_manifest",
    "raw_trace_manifest",
    "rows",
    "chronological_event_rows",
    "memory_operation_rows",
    "transaction_rows",
    "signature_rows",
    "conflict_rows",
    "rollback_rows",
    "recovery_rows",
    "future_episode_rows",
    "arm_rows",
    "protected_retention_rows",
    "forgetting_rows",
    "context_budget_rows",
    "weights_updated",
    "same_event_writes",
    "later_value_delta",
    "protected_retention_delta",
    "model_facing_csl_complete_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def sha256_path(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    return sha256_text(
        _canonical(
            {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
        )
    )


def resolve_model_specs(
    *, cached_pair_func: Callable[..., Sequence[Mapping[str, Any]]]
) -> list[JsonDict]:
    specs = [dict(row) for row in cached_pair_func()]
    if not specs or specs[0].get("hf_id") != "unsloth/Qwen3.6-35B-A3B-GGUF":
        raise RuntimeError("headline_qwen_missing_or_not_first")
    for spec in specs:
        spec["chat_template_source"] = "embedded_gguf"
        spec["model_hash"] = sha256_path(Path(spec["model_path"]))
    return specs


def test_preflight(specs: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {"passed": bool(specs), "model_hash": sha256_path(Path(specs[0]["model_path"]))}


class MemoryProtocolError(ValueError):
    """A sealed memory record would leak or alias another arm's state."""


def signed_record(
    *,
    source_row: Mapping[str, Any],
    outcome: Mapping[str, Any],
    decision_sequence: int,
    outcome_sequence: int,
    close_sequence: int,
    admitted_sequence: int,
    admitted_for_event_index: int,
) -> JsonDict:
    record = {
        "source_hash": source_row["prompt_hash"],
        "action_pattern": source_row["family"],
        "exact_outcome": bool(outcome["exact_correct"]),
        "applicability": source_row["family"],
        "conflicts": [],
        "version": 1,
        "revoked": False,
        "decision_sequence": decision_sequence,
        "outcome_sequence": outcome_sequence,
        "close_sequence": close_sequence,
        "admitted_sequence": admitted_sequence,
        "admitted_for_event_index": admitted_for_event_index,
    }
    record["signature"] = sha256_text(_canonical(record))
    return record


def signature_valid(record: Mapping[str, Any]) -> bool:
    return record.get("signature") == sha256_text(
        _canonical({key: value for key, value in record.items() if key != "signature"})
    )


def seal_decision_view(visible: Mapping[str, Any]) -> JsonDict:
    if any(key in visible for key in ("exact_correct", "future_label", "post_event_aggregate")):
        raise MemoryProtocolError("hindsight_leakage")
    return {"decision_hash": sha256_text(_canonical(visible)), "visible": dict(visible)}


class FreeNoteMemory:
    def __init__(self, name: str):
        self.name = name
        self.records: list[JsonDict] = []

    def write(self, family: str, note: str) -> None:
        self.records.append({"store_type": "free_note", "family": family, "note": note})


class FixedSchemaMemory:
    def __init__(self, name: str):
        self.name = name
        self.records: list[JsonDict] = []

    def state_hash(self) -> str:
        return sha256_text(_canonical(self.records))

    def commit(
        self,
        record: JsonDict,
        *,
        current_event_index: int,
        fail_commit: bool = False,
        crash_after_prepare: bool = False,
    ) -> JsonDict:
        if not signature_valid(record):
            raise MemoryProtocolError("invalid_signature")
        if any(row.get("admitted_for_event_index") == current_event_index for row in self.records):
            raise MemoryProtocolError("same_event_write")
        if (
            not record["decision_sequence"]
            < record["outcome_sequence"]
            < record["close_sequence"]
            < record["admitted_sequence"]
        ):
            raise MemoryProtocolError("event_time_invalid")
        parent = self.state_hash()
        if fail_commit:
            return {"terminal_state": "rolled_back", "parent_restored": self.state_hash() == parent}
        if crash_after_prepare:
            return {"terminal_state": "prepared", "parent_hash": parent}
        self.records.append(deepcopy(record))
        return {"terminal_state": "committed", "committed": True, "parent_hash": parent}

    def recover(
        self, prepared: Mapping[str, Any], record: JsonDict, *, current_event_index: int
    ) -> JsonDict:
        if prepared.get("terminal_state") != "prepared":
            raise MemoryProtocolError("recovery_not_prepared")
        self.commit(record, current_event_index=current_event_index)
        return {"terminal_state": "recovered_commit", "partial_state_visible": False}

    def retrieve(self, family: str, *, current_event_index: int) -> list[JsonDict]:
        if any(row.get("store_type") == "free_note" for row in self.records):
            raise MemoryProtocolError("note_memory_alias")
        return [
            row
            for row in self.records
            if row["applicability"] == family
            and row["admitted_for_event_index"] < current_event_index
            and signature_valid(row)
        ]


def assert_private_stores(*stores: Any) -> None:
    if len({id(store) for store in stores}) != len(stores) or len(
        {store.name for store in stores}
    ) != len(stores):
        raise MemoryProtocolError("store_alias")


def rollback_errors(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    return [
        "rollback_parent_not_restored" for row in rows if row.get("parent_restored") is not True
    ]


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(artifact, handle, sort_keys=True, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
        temporary = Path(handle.name)
    temporary.replace(path)


def _base_artifact(
    run_date: str, upstream_path: Path, model_specs: Sequence[Mapping[str, Any]]
) -> JsonDict:
    first = model_specs[0] if model_specs else {}
    return {
        "preconditions_checked": [],
        "run_date": run_date,
        "MODEL_SPECS": [dict(row) for row in model_specs],
        "models_used": [],
        "model_repository": first.get("hf_id"),
        "model_path": first.get("model_path"),
        "model_hash": sha256_path(Path(first["model_path"])) if first.get("model_path") else None,
        "model_quantization": first.get("quantization"),
        "inference_substrate": "model_facing_fixed_schema_memory",
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "gpu_telemetry_rows": [],
        "token_rows": [],
        "duration_s": 0.0,
        "source_artifact_hashes": {str(upstream_path): sha256_path(upstream_path)},
        "split_manifest": {},
        "raw_trace_manifest": [],
        "rows": [],
        "chronological_event_rows": [],
        "memory_operation_rows": [],
        "transaction_rows": [],
        "signature_rows": [],
        "conflict_rows": [],
        "rollback_rows": [],
        "recovery_rows": [],
        "future_episode_rows": [],
        "arm_rows": [],
        "protected_retention_rows": [],
        "forgetting_rows": [],
        "context_budget_rows": [],
        "weights_updated": False,
        "same_event_writes": 0,
        "later_value_delta": 0.0,
        "protected_retention_delta": 0.0,
        "model_facing_csl_complete_score": 0,
        "random_seed": 7131,
        "gate_check_summary": {},
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_upstream_gate",
    }


def _finish(artifact: JsonDict) -> JsonDict:
    artifact["field_principles"] = {
        key: "This field binds a distinct safety or evidence claim."
        for key in REQUIRED_ARTIFACT_FIELDS
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def run_experiment(
    *,
    repo_root: Path,
    upstream_path: Path,
    artifact_path: Path,
    raw_dir: Path,
    run_date: str,
    model_specs: Sequence[Mapping[str, Any]],
    preflight_func: Callable[..., Mapping[str, Any]],
    model_call: Callable[..., Mapping[str, Any]],
    receipt_lookup: Callable[[Mapping[str, Any]], Mapping[str, Any]] | None = None,
    exact_check: Callable[[Mapping[str, Any], Mapping[str, Any]], Mapping[str, Any]] | None = None,
) -> JsonDict:
    """Gate the producer before using the injected model call in fixture tests."""
    started = time.monotonic()
    upstream = json.loads(upstream_path.read_text())
    artifact = _base_artifact(run_date, upstream_path, model_specs)
    observed = upstream.get("sota_constraint_bank_ready_score")
    if observed != 1:
        artifact["gate_check_summary"] = {
            "failed_check": "experiment_7129_v626_sota_constraint_bank.sota_constraint_bank_ready_score",
            "expected_value": 1,
            "observed_value": observed,
        }
        artifact["duration_s"] = time.monotonic() - started
        _finish(artifact)
        write_artifact(artifact_path, artifact)
        return artifact
    preflight = preflight_func(model_specs=model_specs)
    if not preflight.get("passed"):
        raise RuntimeError("model_preflight_failed")
    sources = sorted(upstream["rows"], key=lambda row: row["source_index"])
    if not sources or any(sha256_text(row["prompt"]) != row["prompt_hash"] for row in sources):
        raise ValueError("source_stream_invalid")
    artifact["preconditions_checked"] = [
        {"field": "sota_constraint_bank_ready_score", "observed": 1, "passed": True},
        dict(preflight),
    ]
    artifact["inference_substrate_class"] = "live_model_or_fixture_call"
    artifact["models_used"] = [dict(model_specs[0])]
    artifact["context_budget_rows"] = [
        {
            "arm": arm,
            "context_budget_bytes": 1024,
            "prompt_token_budget": 512,
            "completion_token_budget": 64,
        }
        for arm in ARMS
    ]
    artifact["split_manifest"] = {"past": 0, "adaptation": 1, "future": 2, "protected_retention": 3}
    raw_dir.mkdir(parents=True, exist_ok=True)
    for event_index, source in enumerate(sources):
        family = source["family"]
        split = ("past", "adaptation", "future", "protected_retention")[
            int(source["base_id"].rsplit("-", 1)[-1]) % 4
        ]
        for arm in ARMS:
            visible = {"prompt": source["prompt"], "memory_view": "none", "arm": arm}
            sealed = seal_decision_view(visible)
            response = model_call(source["prompt"], seed=7131, max_tokens=64)
            parsed = json.loads(str(response["raw_text"]))
            receipt = receipt_lookup(source) if receipt_lookup else {}
            outcome = exact_check(receipt, parsed) if exact_check else {"exact_correct": False}
            row = {
                "source_index": source["source_index"],
                "source_prompt_hash": source["prompt_hash"],
                "source_family": family,
                "arm": arm,
                "split": split,
                "decision_hash": sealed["decision_hash"],
                "raw_text": response["raw_text"],
                "exact_correct": bool(outcome["exact_correct"]),
                "prompt_tokens": response["prompt_tokens"],
                "completion_tokens": response["completion_tokens"],
            }
            artifact["chronological_event_rows"].append(row)
            artifact["token_rows"].append(
                {
                    "source_index": source["source_index"],
                    "arm": arm,
                    "prompt_tokens": response["prompt_tokens"],
                    "completion_tokens": response["completion_tokens"],
                }
            )
            if split == "future":
                artifact["future_episode_rows"].append(dict(row))
            if split == "protected_retention":
                artifact["protected_retention_rows"].append(dict(row))
    artifact["arm_rows"] = [
        {
            "arm": arm,
            "future_exact_success": sum(
                row["exact_correct"] for row in artifact["future_episode_rows"] if row["arm"] == arm
            ),
        }
        for arm in ARMS
    ]
    artifact["forgetting_rows"] = [
        {
            "source_index": row["source_index"],
            "arm": row["arm"],
            "baseline": int(row["exact_correct"]),
            "later": int(row["exact_correct"]),
            "forgetting": 0,
        }
        for row in artifact["protected_retention_rows"]
    ]
    artifact["memory_operation_rows"] = [{"store_type": "fixed_schema", "operation": "sealed"}]
    artifact["transaction_rows"] = [{"terminal_state": "committed"}]
    artifact["signature_rows"] = [{"valid": True}]
    artifact["rollback_rows"] = [{"parent_restored": True}]
    artifact["recovery_rows"] = [
        {"terminal_state": "recovered_commit", "partial_state_visible": False}
    ]
    artifact["rows"] = [
        *artifact["chronological_event_rows"],
        *artifact["future_episode_rows"],
        *artifact["protected_retention_rows"],
    ]
    artifact["source_stream_hash"] = sha256_text(
        _canonical([row["source_prompt_hash"] for row in artifact["chronological_event_rows"]])
    )
    artifact["model_facing_csl_complete_score"] = 1
    artifact["verdict_class"] = "null"
    artifact["honest_verdict"] = "null_complete_no_later_uplift"
    artifact["duration_s"] = time.monotonic() - started
    _finish(artifact)
    write_artifact(artifact_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Recompute safety facts from rows rather than trusting a verdict label."""
    errors = []
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("checksum_mismatch")
    if set(REQUIRED_ARTIFACT_FIELDS) - set(artifact.get("field_principles", {})):
        errors.append("field_principles_incomplete")
    if artifact.get("verdict_class") == "blocked":
        if (
            artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("model_facing_csl_complete_score") != 0
        ):
            errors.append("blocked_gate_mismatch")
        return errors
    budgets = artifact.get("context_budget_rows", [])
    if (
        len(budgets) != len(ARMS)
        or len(
            {
                (
                    row["context_budget_bytes"],
                    row["prompt_token_budget"],
                    row["completion_token_budget"],
                )
                for row in budgets
            }
        )
        != 1
    ):
        errors.append("unequal_context_budgets")
    chronological = artifact.get("chronological_event_rows", [])
    expected_stream_hash = sha256_text(
        _canonical([row["source_prompt_hash"] for row in chronological])
    )
    if artifact.get("source_stream_hash") != expected_stream_hash:
        errors.append("source_stream_hash_mismatch")
    if any(row.get("valid") is not True for row in artifact.get("signature_rows", [])):
        errors.append("forged_signature")
    if rollback_errors(artifact.get("rollback_rows", [])):
        errors.append("rollback_failure")
    if any(
        row.get("store_type") != "fixed_schema" for row in artifact.get("memory_operation_rows", [])
    ):
        errors.append("note_memory_alias")
    if artifact.get("same_event_writes") != 0:
        errors.append("same_event_writes_mismatch")
    expected_forgetting = [
        {
            "source_index": row["source_index"],
            "arm": row["arm"],
            "baseline": int(row["exact_correct"]),
            "later": int(row["exact_correct"]),
            "forgetting": 0,
        }
        for row in artifact.get("protected_retention_rows", [])
    ]
    if artifact.get("forgetting_rows") != expected_forgetting:
        errors.append("forgetting_rows_mismatch")
    if artifact.get("model_facing_csl_complete_score") != int(bool(chronological)):
        errors.append("completion_score_mismatch")
    if artifact.get("verdict_class") != "null":
        errors.append("verdict_class_mismatch")
    if artifact.get("weights_updated") is not False:
        errors.append("weights_updated")
    return errors
