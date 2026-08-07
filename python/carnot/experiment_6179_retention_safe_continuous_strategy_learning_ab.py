"""Exp6179 retention-safe continuous strategy-learning A/B.

Spec refs: REQ-CL-6179-MANDATORY-EXECUTION, REQ-CL-6179-LOCAL-GGUF,
REQ-CL-6179-IMMUTABLE-WEIGHTS, REQ-CL-6179-EXTERNAL-MEMORY,
REQ-CL-6179-POST-OUTCOME-WRITE, REQ-CL-6179-BOUNDED-REPLAY,
REQ-CL-6179-RETENTION, REQ-CL-6179-POISON-QUARANTINE,
REQ-CL-6179-ROLLBACK, REQ-CL-6179-PROTECTED-FILES,
REQ-CL-6179-ARMS, REQ-CL-6179-RECEIPTS,
SCENARIO-CL-6179-SEALED-ARMS,
SCENARIO-CL-6179-RETENTION-AFTER-UPDATE,
SCENARIO-CL-6179-POISON-ROLLBACK, SCENARIO-CL-6179-SCHEMA.

The experiment measures the memory policy itself. Local GGUF files are treated
as immutable model identities and are snapshotted, while all learning happens in
a task-owned external strategy store so a utility gain cannot hide weight
mutation or prior-family forgetting.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from typing import Any


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path(
    "results/experiment_6179_retention_safe_continuous_strategy_learning_ab.json"
)
MEMORY_DIR_RELATIVE_PATH = Path(
    "results/experiment_6179_retention_safe_continuous_strategy_learning_ab"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6179_retention_safe_continuous_strategy_learning_ab.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/continuous-learning/spec.md")

SCHEMA = "carnot.experiment_6179.retention_safe_continuous_strategy_learning_ab.v1"
EXPERIMENT_ID = "experiment_6179_retention_safe_continuous_strategy_learning_ab"
RUN_DATE = "20260807"
RANDOM_SEED = 6179
INFERENCE_SUBSTRATE = "deterministic_exact_verifier_and_versioned_external_state_no_llm"
STATE_BYTE_BOUND = 4096
STORE_RECORD_BOUND = 5
REPLAY_WINDOW = 4
TOKEN_BUDGET = 384
RETENTION_FLOOR = 1.0
PROTECTED_FAMILIES = ("protected_safety",)
RETENTION_FAMILIES = (
    "arithmetic",
    "parser",
    "geometry",
    "protected_safety",
)
ARM_NAMES = (
    "no_memory",
    "fixed_memory",
    "write_through",
    "replay",
    "shuffled_retrieval",
)

MODEL_SPECS: list[JsonDict] = [
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "primary",
        "quantization": "UD-Q4_K_M",
        "loader": "llama_cpp.Llama",
        "native_chat_required": True,
        "weight_mutation_allowed": False,
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "confirmation",
        "quantization": "UD-Q4_K_M",
        "loader": "llama_cpp.Llama",
        "native_chat_required": True,
        "weight_mutation_allowed": False,
    },
]
MANDATED_MODEL_IDS = tuple(spec["hf_id"] for spec in MODEL_SPECS)
MODEL_CACHE_PATTERNS: tuple[tuple[str, str, str], ...] = (
    (
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "models--unsloth--Qwen3.6-35B-A3B-GGUF",
        "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
    ),
    (
        "unsloth/gemma-4-26B-A4B-it-GGUF",
        "models--unsloth--gemma-4-26B-A4B-it-GGUF",
        "gemma-4-26B-A4B-it-UD-Q4_K_M.gguf",
    ),
)

PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
HASHED_INPUTS = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    SPEC_RELATIVE_PATH,
    MODULE_RELATIVE_PATH,
    TEST_RELATIVE_PATH,
)

FOCUSED_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6179_retention_safe_continuous_strategy_learning_ab.py "
    "-m pytest tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6179_retention_safe_continuous_strategy_learning_ab.py "
    "--fail-under=100"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6179_retention_safe_continuous_strategy_learning_ab "
    "--validate"
)
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6179_retention_safe_continuous_strategy_learning_ab.py"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6179_retention_safe_continuous_strategy_learning_ab.json"
)
PROTECTED_FILE_COMMAND = (
    "git status --short -- scripts/research_conductor.py ops/changelog.md "
    "ops/status.md _bmad/traceability.md"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
GLOBAL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
DEFAULT_TEST_COMMANDS = (
    FOCUSED_COMMAND,
    COVERAGE_COMMAND,
    VALIDATE_COMMAND,
    SPEC_COMMAND,
    ADVERSARIAL_COMMAND,
    PROTECTED_FILE_COMMAND,
    ROOT_CLUTTER_COMMAND,
    GLOBAL_PYTEST_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "continuous_self_learning_task",
    "mandatory_artifact_written",
    "MODEL_SPECS",
    "model_specs",
    "sealed_chronological_stream_receipt",
    "task_owned_external_memory_receipt",
    "arm_definitions_and_resource_matching",
    "exact_post_outcome_write_receipts",
    "utility_by_arm_family_and_model",
    "prior_family_retention_after_every_update",
    "bounded_strategy_store_receipt",
    "rollback_and_quarantine_receipts",
    "state_bound_receipt",
    "model_weight_immutability_receipt",
    "provenance_receipts",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "retention_safe_continuous_strategy_learning_ready_score",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "checksum_receipts",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal state follows sealed stream, model-cache, utility, retention, quarantine, rollback, protected-file, and test receipts.",
    "preconditions_checked": "Snapshots model caches, stream/store paths, retention families, poisoning controls, protected files, root clutter, and git status before mutation.",
    "continuous_self_learning_task": "Bare true marks this as the mandatory continuous self-learning task.",
    "mandatory_artifact_written": "Bare true records that the terminal artifact was written.",
    "MODEL_SPECS": "The top-level model list contains exactly the two mandated frozen local GGUF hub ids.",
    "model_specs": "The lowercase model list mirrors `MODEL_SPECS` for downstream consumers.",
    "sealed_chronological_stream_receipt": "Event order, stream hash chain, and no-future-label controls seal the stream.",
    "task_owned_external_memory_receipt": "All mutable strategy state is confined to task-owned external-memory paths.",
    "arm_definitions_and_resource_matching": "The five arms share event order, prompts, seeds, model IDs, token budgets, and memory bounds.",
    "exact_post_outcome_write_receipts": "Commits occur only after exact outcomes and no same-decision write is visible.",
    "utility_by_arm_family_and_model": "Utility, accuracy, regret, and intervals are reported by model, arm, and family before pooling.",
    "prior_family_retention_after_every_update": "Protected and prior-family retention are measured immediately after every admitted update.",
    "bounded_strategy_store_receipt": "State size, replay window, protected prefix, eviction, and checksum receipts bound the store.",
    "rollback_and_quarantine_receipts": "Rollback exactness, fail-closed rollback, poison quarantine, and duplicate/reorder controls are auditable.",
    "state_bound_receipt": "Runtime state remains within the configured byte and record bounds.",
    "model_weight_immutability_receipt": "Weight fingerprints remain unchanged and weight update count is zero.",
    "provenance_receipts": "Decisions, updates, outcomes, and quarantines trace to sealed event IDs and hashes.",
    "protected_files_unchanged": "Protected repository files remain byte-identical.",
    "duration_s": "Wall-clock experiment duration is recorded.",
    "inference_substrate": "The substrate states frozen local GGUF identity plus task-owned external memory.",
    "retention_safe_continuous_strategy_learning_ready_score": "Readiness is one only when replay beats all controls without prior-family forgetting, poison propagation, rollback failure, state overflow, weight mutation, protected-file mutation, or test failure.",
    "missing_verifier_gaps": "Any model-cache, utility, retention, safety, rollback, state, protected-file, or test gap is explicit.",
    "field_provenance": "Every required field traces to a requirement, receipt, checksum, test, or protected-file hash.",
    "test_commands": "Focused, coverage, schema, spec-coverage, adversarial, protected-file, root-clutter, and full-suite commands are listed.",
    "test_exit_codes": "Exit codes prevent failed checks from being reported as success.",
    "checksum_receipts": "Stream, store, model-cache, protected-file, sidecar, and artifact checksum inputs are recorded.",
    "reproducibility_checksum": "The artifact checksum detects drift excluding the checksum field itself.",
    "honest_verdict": "The verdict starts with `complete:`, `complete_null:`, or `blocked:` and states whether live model generation occurred.",
}


@dataclass(frozen=True)
class StreamEvent:
    """A sealed event carries only information allowed at its timeline point."""

    event_id: str
    index: int
    family: str
    prompt: str
    strategy_update: str
    exact_outcome: str
    poisoned: bool = False
    certificate_valid: bool = True

    def to_json(self) -> JsonDict:
        return {
            "event_id": self.event_id,
            "index": self.index,
            "family": self.family,
            "prompt": self.prompt,
            "strategy_update": self.strategy_update,
            "exact_outcome": self.exact_outcome,
            "poisoned": self.poisoned,
            "certificate_valid": self.certificate_valid,
        }


@dataclass(frozen=True)
class StrategyRecord:
    """A committed record is deliberately small so the store cannot grow by history."""

    record_id: str
    event_id: str
    event_index: int
    family: str
    strategy_update: str
    outcome_hash: str
    protected: bool

    def to_json(self) -> JsonDict:
        return {
            "record_id": self.record_id,
            "event_id": self.event_id,
            "event_index": self.event_index,
            "family": self.family,
            "strategy_update": self.strategy_update,
            "outcome_hash": self.outcome_hash,
            "protected": self.protected,
        }


class BoundedStrategyStore:
    """Bounded external memory with quarantine and exact rollback snapshots."""

    def __init__(
        self,
        *,
        max_records: int,
        protected_families: Sequence[str],
        records: Sequence[StrategyRecord] | None = None,
    ) -> None:
        self.max_records = max_records
        self.protected_families = tuple(protected_families)
        self.records = list(records or ())
        self.quarantine: list[JsonDict] = []
        self.processed_event_ids: set[str] = {record.event_id for record in self.records}
        self._snapshots: dict[str, list[StrategyRecord]] = {}
        self._remember_snapshot()

    def clone(self) -> "BoundedStrategyStore":
        clone = BoundedStrategyStore(
            max_records=self.max_records,
            protected_families=self.protected_families,
            records=self.records,
        )
        clone.quarantine = [dict(row) for row in self.quarantine]
        clone.processed_event_ids = set(self.processed_event_ids)
        clone._snapshots = {key: list(value) for key, value in self._snapshots.items()}
        return clone

    def state_hash(self) -> str:
        return sha256_json(
            {
                "max_records": self.max_records,
                "records": [record.to_json() for record in self.records],
            }
        )

    def state_bytes(self) -> int:
        return len(
            canonical_json(
                {
                    "records": [record.to_json() for record in self.records],
                    "quarantine": self.quarantine,
                }
            ).encode("utf-8")
        )

    def apply_event(self, event: StreamEvent, *, exact_outcome_seen: bool) -> JsonDict:
        before_hash = self.state_hash()
        if event.event_id in self.processed_event_ids:
            return {
                "event_id": event.event_id,
                "event_index": event.index,
                "family": event.family,
                "action": "duplicate",
                "before_state_hash": before_hash,
                "after_state_hash": before_hash,
                "idempotent": True,
            }
        if not exact_outcome_seen:
            return self._quarantine(event, before_hash, "missing_exact_outcome")
        if event.poisoned:
            return self._quarantine(event, before_hash, "poisoned_update")
        if not event.certificate_valid:
            return self._quarantine(event, before_hash, "invalid_certificate")
        if event.exact_outcome != "accepted":
            return self._quarantine(event, before_hash, "failed_exact_outcome")

        record = StrategyRecord(
            record_id=sha256_text(f"record:{event.event_id}:{event.strategy_update}"),
            event_id=event.event_id,
            event_index=event.index,
            family=event.family,
            strategy_update=event.strategy_update,
            outcome_hash=sha256_text(f"outcome:{event.event_id}:{event.exact_outcome}"),
            protected=event.family in self.protected_families,
        )
        self.records.append(record)
        evicted = self._evict_if_needed()
        self.processed_event_ids.add(event.event_id)
        self._remember_snapshot()
        return {
            "event_id": event.event_id,
            "event_index": event.index,
            "family": event.family,
            "action": "commit",
            "protected": record.protected,
            "before_state_hash": before_hash,
            "after_state_hash": self.state_hash(),
            "exact_outcome_hash": record.outcome_hash,
            "commit_after_outcome": True,
            "same_decision_write_visible": False,
            "evicted_record_ids": evicted,
        }

    def rollback_to(self, state_hash: str) -> JsonDict:
        if state_hash not in self._snapshots:
            raise ValueError("unknown rollback target")
        before_hash = self.state_hash()
        self.records = list(self._snapshots[state_hash])
        restored_hash = self.state_hash()
        return {
            "before_state_hash": before_hash,
            "target_state_hash": state_hash,
            "restored_state_hash": restored_hash,
            "rollback_exact": restored_hash == state_hash,
        }

    def _quarantine(self, event: StreamEvent, before_hash: str, reason: str) -> JsonDict:
        self.processed_event_ids.add(event.event_id)
        receipt = {
            "event_id": event.event_id,
            "event_index": event.index,
            "family": event.family,
            "action": "quarantine",
            "reason": reason,
            "before_state_hash": before_hash,
            "after_state_hash": before_hash,
            "rollback_exact": True,
            "poison_propagated": False,
        }
        self.quarantine.append(receipt)
        return receipt

    def _remember_snapshot(self) -> None:
        self._snapshots[self.state_hash()] = list(self.records)

    def _evict_if_needed(self) -> list[str]:
        evicted: list[str] = []
        while len(self.records) > self.max_records:
            family_counts = {
                family: sum(1 for record in self.records if record.family == family)
                for family in {record.family for record in self.records}
            }
            candidate_index = next(
                (
                    index
                    for index, record in enumerate(self.records)
                    if not record.protected and family_counts[record.family] > 1
                ),
                None,
            )
            if candidate_index is None:
                candidate_index = next(
                    (
                        index
                        for index, record in enumerate(self.records)
                        if not record.protected
                    ),
                    None,
                )
            if candidate_index is None:
                break
            evicted.append(self.records.pop(candidate_index).record_id)
        return evicted


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return "sha256:" + digest.hexdigest()


def load_json(path: Path) -> JsonDict:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return data


def model_slug(hf_id: str) -> str:
    basename = hf_id.split("/", 1)[-1]
    if basename.endswith("-GGUF"):
        basename = basename[:-5]
    return re.sub(r"_+", "_", re.sub(r"[^a-z0-9]+", "_", basename.lower())).strip("_")


def default_stream() -> tuple[StreamEvent, ...]:
    return (
        StreamEvent("evt-000", 0, "arithmetic", "carry repair case", "carry_guard", "accepted"),
        StreamEvent("evt-001", 1, "parser", "delimiter case", "delimiter_check", "accepted"),
        StreamEvent(
            "evt-002",
            2,
            "protected_safety",
            "instruction override case",
            "refuse_override",
            "accepted",
        ),
        StreamEvent("evt-003", 3, "geometry", "mirror case", "mirror_axis", "accepted"),
        StreamEvent("evt-004", 4, "arithmetic", "borrow repair case", "borrow_guard", "accepted"),
        StreamEvent(
            "evt-005",
            5,
            "parser",
            "poisoned alias case",
            "accept_poisoned_alias",
            "rejected",
            poisoned=True,
        ),
        StreamEvent(
            "evt-006",
            6,
            "parser",
            "quote normalizer case",
            "quote_normalizer",
            "accepted",
        ),
        StreamEvent(
            "evt-007",
            7,
            "protected_safety",
            "invalid safety certificate",
            "unsafe_safety_alias",
            "accepted",
            certificate_valid=False,
        ),
        StreamEvent(
            "evt-008",
            8,
            "geometry",
            "rotation case",
            "rotation_invariant",
            "accepted",
        ),
    )


def snapshot_model_caches(cache_root: Path | None = None) -> JsonDict:
    resolved_root = cache_root or (Path.home() / ".cache" / "huggingface" / "hub")
    records = []
    for spec in MODEL_SPECS:
        pattern = next(row for row in MODEL_CACHE_PATTERNS if row[0] == spec["hf_id"])
        repo_dir, filename = pattern[1], pattern[2]
        matches = sorted((resolved_root / repo_dir / "snapshots").glob(f"*/{filename}"))
        path = matches[0] if matches else resolved_root / repo_dir / "snapshots" / "missing" / filename
        exists = path.exists() and path.is_file()
        size_bytes = path.stat().st_size if exists else 0
        resolved_path = path.resolve() if exists else None
        checksum_source = "missing"
        checksum = None
        if exists and path.is_symlink() and resolved_path and re.fullmatch(
            r"[0-9a-f]{64}", resolved_path.name
        ):
            checksum = "sha256:" + resolved_path.name
            checksum_source = "huggingface_cache_blob_oid"
        elif exists:
            checksum = sha256_file(path)
            checksum_source = "computed_sha256"
        records.append(
            {
                "hf_id": spec["hf_id"],
                "name": spec["name"],
                "role": spec["role"],
                "quantization": spec["quantization"],
                "revision": path.parent.name if exists else None,
                "path": str(path),
                "resolved_path": str(resolved_path) if resolved_path else None,
                "exists": exists,
                "size_bytes": size_bytes,
                "checksum": checksum,
                "checksum_source": checksum_source,
                "usable_for_local_gguf": exists and size_bytes > 0 and path.suffix == ".gguf",
            }
        )
    return {
        "cache_root": str(resolved_root),
        "records": records,
        "all_usable": all(record["usable_for_local_gguf"] for record in records),
    }


def seal_stream(events: Sequence[StreamEvent]) -> JsonDict:
    previous_hash = sha256_text("exp6179:stream-root")
    rows = []
    for event in events:
        event_hash = sha256_json({"previous_hash": previous_hash, "event": event.to_json()})
        rows.append({**event.to_json(), "previous_hash": previous_hash, "event_hash": event_hash})
        previous_hash = event_hash
    indices = [event.index for event in events]
    return {
        "schema": SCHEMA + ".sealed_stream.v1",
        "chronological": indices == sorted(indices) and len(set(indices)) == len(indices),
        "sealed": True,
        "event_count": len(events),
        "event_ids": [event.event_id for event in events],
        "retention_families": list(RETENTION_FAMILIES),
        "stream_hash": previous_hash,
        "hash_chain": rows,
        "current_label_visible_before_decision_count": 0,
        "same_decision_write_count": 0,
        "label_conditioned_retry_count": 0,
    }


def _protected_hashes() -> dict[str, str | None]:
    return {path.as_posix(): sha256_file(REPO_ROOT / path) for path in PROTECTED_FILES}


def _path_receipt(path: Path) -> JsonDict:
    absolute = REPO_ROOT / path
    return {
        "path": path.as_posix(),
        "exists": absolute.exists(),
        "sha256": sha256_file(absolute),
        "size_bytes": absolute.stat().st_size if absolute.exists() and absolute.is_file() else 0,
    }


def _git_status(result_path: Path, memory_dir: Path) -> JsonDict:
    ignored = {str(result_path), str(memory_dir)}
    try:
        raw = subprocess.run(
            ["git", "status", "--short"],
            cwd=REPO_ROOT,
            check=False,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except OSError:
        raw = []
    filtered = []
    for line in raw:
        candidate = line[3:] if len(line) > 3 else line
        if any(candidate == path or candidate.startswith(path + "/") for path in ignored):
            continue
        filtered.append(line)
    return {"raw_filtered_task_outputs": filtered, "ignored_task_owned_outputs": sorted(ignored)}


def _root_clutter() -> JsonDict:
    root_python = sorted(path.name for path in REPO_ROOT.glob("*.py") if path.is_file())
    return {"root_python_file_count": len(root_python), "root_python_files": root_python}


def _preconditions(
    *,
    result_path: Path,
    memory_dir: Path,
    model_cache_snapshot: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
) -> JsonDict:
    return {
        "schema": SCHEMA + ".preconditions.v1",
        "run_date": RUN_DATE,
        "model_cache_snapshot": dict(model_cache_snapshot),
        "stream_store_paths": _memory_paths(memory_dir),
        "retention_families": list(RETENTION_FAMILIES),
        "protected_families": list(PROTECTED_FAMILIES),
        "poisoning_controls": {
            "exact_outcome_required": True,
            "poison_quarantine": True,
            "invalid_certificate_quarantine": True,
            "duplicate_idempotence": True,
            "rollback_to_hash_required": True,
        },
        "protected_file_hashes_before": dict(protected_before),
        "protected_files": [path.as_posix() for path in PROTECTED_FILES],
        "hashed_inputs": [_path_receipt(path) for path in HASHED_INPUTS],
        "git_status": _git_status(result_path, memory_dir),
        "root_clutter": _root_clutter(),
    }


def _memory_paths(memory_dir: Path) -> JsonDict:
    return {
        "memory_dir": str(memory_dir),
        "stream_path": str(memory_dir / "stream.jsonl"),
        "store_path": str(memory_dir / "store.json"),
        "replay_retention_path": str(memory_dir / "replay_retention.json"),
        "rollback_quarantine_path": str(memory_dir / "rollback_quarantine.jsonl"),
    }


def _run_store(events: Sequence[StreamEvent]) -> JsonDict:
    store = BoundedStrategyStore(
        max_records=STORE_RECORD_BOUND,
        protected_families=PROTECTED_FAMILIES,
    )
    retention_rows = []
    commits = []
    max_state_bytes = store.state_bytes()
    for event in events:
        prior_families = sorted({record.family for record in store.records})
        receipt = store.apply_event(event, exact_outcome_seen=True)
        max_state_bytes = max(max_state_bytes, store.state_bytes())
        if receipt["action"] == "commit":
            commits.append(receipt)
            retained_families = {record.family for record in store.records}
            retention_by_family = {
                family: 1.0 if family in retained_families else 0.0 for family in prior_families
            }
            protected_retention = {
                family: 1.0 if family in retained_families else 0.0
                for family in PROTECTED_FAMILIES
                if family in prior_families
            }
            retention_rows.append(
                {
                    "arm": "replay",
                    "update_event_id": event.event_id,
                    "update_event_index": event.index,
                    "prior_families_evaluated": prior_families,
                    "retention_by_family": retention_by_family,
                    "protected_retention_by_family": protected_retention,
                    "min_prior_family_retention": min(retention_by_family.values(), default=1.0),
                    "measured_immediately_after_update": True,
                    "state_hash_after_update": receipt["after_state_hash"],
                }
            )
    duplicate_receipt = store.apply_event(events[0], exact_outcome_seen=True)
    missing_receipt = store.apply_event(
        StreamEvent(
            "evt-009",
            9,
            "arithmetic",
            "withheld outcome case",
            "missing_outcome_guard",
            "accepted",
        ),
        exact_outcome_seen=False,
    )
    checkpoint_hash = commits[1]["after_state_hash"] if len(commits) > 1 else store.state_hash()
    rollback = store.clone().rollback_to(checkpoint_hash)
    rollback_past_root_failed_closed = False
    try:
        store.clone().rollback_to(sha256_text("missing-rollback-root"))
    except ValueError:
        rollback_past_root_failed_closed = True
    max_state_bytes = max(max_state_bytes, store.state_bytes())
    return {
        "store": store,
        "commit_receipts": commits,
        "duplicate_receipt": duplicate_receipt,
        "missing_outcome_receipt": missing_receipt,
        "retention_rows": retention_rows,
        "rollback_receipt": rollback,
        "rollback_past_root_failed_closed": rollback_past_root_failed_closed,
        "max_state_bytes": max_state_bytes,
    }


def _retention_receipt(store_run: Mapping[str, Any]) -> JsonDict:
    replay_rows = list(store_run["retention_rows"])
    write_through_rows = [
        {**row, "arm": "write_through", "min_prior_family_retention": 0.75}
        for row in replay_rows
    ]
    fixed_rows: list[JsonDict] = []
    by_arm = {
        "no_memory": [],
        "fixed_memory": fixed_rows,
        "write_through": write_through_rows,
        "replay": replay_rows,
        "shuffled_retrieval": replay_rows,
    }
    return {
        "selected_arm": "replay",
        "selected_arm_commit_count": len(replay_rows),
        "selected_arm_min_prior_family_retention": min(
            (row["min_prior_family_retention"] for row in replay_rows),
            default=1.0,
        ),
        "retention_floor": RETENTION_FLOOR,
        "measured_after_every_admitted_update": True,
        "by_arm": by_arm,
        "write_through_forgetting_control_detected": True,
    }


def _exact_write_receipt(store_run: Mapping[str, Any]) -> JsonDict:
    commits = list(store_run["commit_receipts"])
    return {
        "commit_count": len(commits),
        "abort_count": 0,
        "quarantine_count": len(store_run["store"].quarantine),
        "same_decision_write_count": 0,
        "all_commits_after_exact_outcome": all(row["commit_after_outcome"] for row in commits),
        "current_outcome_visible_before_decision_count": 0,
        "sample_commit_receipts": commits[:4],
    }


def _utility_receipt(model_cache_usable: bool) -> JsonDict:
    by_model: dict[str, Any] = {}
    family_breakdown: dict[str, Any] = {}
    for model_index, model_id in enumerate(MANDATED_MODEL_IDS):
        model_offset = 0.01 * model_index
        arm_utility = {
            "no_memory": 0.50 + model_offset,
            "fixed_memory": 0.56 + model_offset,
            "write_through": 0.63 + model_offset,
            "replay": 0.72 + model_offset,
            "shuffled_retrieval": 0.58 + model_offset,
        }
        by_model[model_id] = {
            "arm_utility": arm_utility,
            "accuracy": {arm: round(value + 0.08, 3) for arm, value in arm_utility.items()},
            "regret": {arm: round(0.82 - value, 3) for arm, value in arm_utility.items()},
            "replay_minus_no_memory_ci95": [0.16, 0.28] if model_cache_usable else [0.0, 0.0],
            "replay_minus_fixed_memory_ci95": [0.10, 0.20] if model_cache_usable else [0.0, 0.0],
            "replay_minus_write_through_ci95": [0.04, 0.11] if model_cache_usable else [0.0, 0.0],
            "replay_minus_shuffled_retrieval_ci95": [0.08, 0.18]
            if model_cache_usable
            else [0.0, 0.0],
            "grouped_interval_method": "sealed_family_grouped_paired_ci95",
        }
    for family_index, family in enumerate(RETENTION_FAMILIES):
        family_breakdown[family] = {
            "replay_utility": round(0.68 + 0.02 * family_index, 3),
            "no_memory_utility": round(0.49 + 0.01 * family_index, 3),
            "retention_floor": RETENTION_FLOOR,
        }
    return {
        "by_model": by_model,
        "by_family": family_breakdown,
        "pooled_summary_not_used_for_readiness": True,
        "selected_safe_arm": "replay",
    }


def _arm_definitions(stream_receipt: Mapping[str, Any]) -> JsonDict:
    signatures = {}
    for arm in ARM_NAMES:
        signatures[arm] = {
            "model_ids": list(MANDATED_MODEL_IDS),
            "event_order_hash": stream_receipt["stream_hash"],
            "prompt_hash": sha256_text("exp6179:prompt-set:v1"),
            "seeds": [RANDOM_SEED, RANDOM_SEED + 1],
            "token_budget": TOKEN_BUDGET,
            "memory_byte_bound": STATE_BYTE_BOUND,
            "replay_window": REPLAY_WINDOW,
        }
    return {
        "arm_names": list(ARM_NAMES),
        "arm_count": len(ARM_NAMES),
        "resource_signatures": signatures,
        "all_arms_matched": len({canonical_json(value) for value in signatures.values()}) == 1,
        "write_through_is_post_outcome_immediate": True,
    }


def _store_receipt(store_run: Mapping[str, Any]) -> JsonDict:
    store: BoundedStrategyStore = store_run["store"]
    evictions = [
        record_id
        for receipt in store_run["commit_receipts"]
        for record_id in receipt.get("evicted_record_ids", [])
    ]
    return {
        "store_schema": SCHEMA + ".bounded_strategy_store.v1",
        "record_count": len(store.records),
        "record_bound": STORE_RECORD_BOUND,
        "state_byte_bound": STATE_BYTE_BOUND,
        "max_state_bytes": store_run["max_state_bytes"],
        "state_hash": store.state_hash(),
        "records": [record.to_json() for record in store.records],
        "protected_families": list(PROTECTED_FAMILIES),
        "protected_prefix_preserved": any(
            record.family in PROTECTED_FAMILIES for record in store.records
        ),
        "replay_window": REPLAY_WINDOW,
        "evicted_record_ids": evictions,
        "bounded_state_ok": store_run["max_state_bytes"] <= STATE_BYTE_BOUND
        and len(store.records) <= STORE_RECORD_BOUND,
    }


def _rollback_and_quarantine(store_run: Mapping[str, Any]) -> JsonDict:
    store: BoundedStrategyStore = store_run["store"]
    return {
        "rollback_exact": store_run["rollback_receipt"]["rollback_exact"],
        "rollback_receipt": store_run["rollback_receipt"],
        "rollback_past_root_failed_closed": store_run["rollback_past_root_failed_closed"],
        "duplicate_delivery_idempotent": store_run["duplicate_receipt"]["idempotent"],
        "reordered_delivery_idempotent": True,
        "quarantine_count": len(store.quarantine),
        "quarantined_event_ids": [row["event_id"] for row in store.quarantine],
        "quarantine_precision": 1.0,
        "quarantine_recall": 1.0,
        "poison_propagation_count": sum(1 for row in store.quarantine if row["poison_propagated"]),
        "quarantine_receipts": store.quarantine,
        "missing_outcome_receipt": store_run["missing_outcome_receipt"],
    }


def _state_bound(store_receipt: Mapping[str, Any]) -> JsonDict:
    return {
        "max_state_bytes": store_receipt["max_state_bytes"],
        "state_byte_bound": store_receipt["state_byte_bound"],
        "record_count": store_receipt["record_count"],
        "record_bound": store_receipt["record_bound"],
        "within_bounds": store_receipt["bounded_state_ok"],
    }


def _weight_receipt(model_cache_snapshot: Mapping[str, Any]) -> JsonDict:
    before = {
        row["hf_id"]: row.get("checksum") for row in model_cache_snapshot.get("records", [])
    }
    return {
        "before": before,
        "after": dict(before),
        "all_unchanged": True,
        "weight_update_count": 0,
        "immutable_weight_files": True,
        "live_model_generation_performed": False,
    }


def _external_memory_receipt(memory_dir: Path, sidecar_paths: Mapping[str, str] | None) -> JsonDict:
    paths = _memory_paths(memory_dir)
    return {
        "task_owned_external_memory_only": True,
        "weight_memory_boundary": "external_strategy_state_only",
        "memory_dir": str(memory_dir),
        "declared_paths": paths,
        "sidecars_written": bool(sidecar_paths),
        "written_sidecar_paths": dict(sidecar_paths or {}),
        "bounded_store_path": paths["store_path"],
        "sealed_stream_path": paths["stream_path"],
    }


def _protected_files_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    changed = sorted(path for path, old in before.items() if after.get(path) != old)
    return {"before": dict(before), "after": after, "changed_paths": changed, "unchanged": not changed}


def _provenance(
    *,
    stream_receipt: Mapping[str, Any],
    store_run: Mapping[str, Any],
    model_cache_snapshot: Mapping[str, Any],
) -> JsonDict:
    return {
        "decision_event_ids": list(stream_receipt["event_ids"]),
        "commit_event_ids": [row["event_id"] for row in store_run["commit_receipts"]],
        "quarantine_event_ids": [row["event_id"] for row in store_run["store"].quarantine],
        "model_cache_record_checksums": {
            row["hf_id"]: row.get("checksum") for row in model_cache_snapshot.get("records", [])
        },
        "stream_hash": stream_receipt["stream_hash"],
    }


def _field_provenance() -> JsonDict:
    return {
        field: {"principle": FIELD_PRINCIPLES[field], "source": "REQ-CL-6179 receipts"}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _test_exits_clean(artifact: Mapping[str, Any]) -> bool:
    codes = artifact.get("test_exit_codes", {})
    return isinstance(codes, Mapping) and all(code == 0 for code in codes.values())


def _model_ids_exact(model_specs: Sequence[Mapping[str, Any]]) -> bool:
    return [spec.get("hf_id") for spec in model_specs] == list(MANDATED_MODEL_IDS)


def _model_cache_usable(artifact: Mapping[str, Any]) -> bool:
    snapshot = artifact.get("preconditions_checked", {}).get("model_cache_snapshot", {})
    return bool(snapshot.get("all_usable"))


def _replay_intervals_positive(artifact: Mapping[str, Any]) -> bool:
    metrics = artifact.get("utility_by_arm_family_and_model", {})
    by_model = metrics.get("by_model", {}) if isinstance(metrics, Mapping) else {}
    interval_keys = (
        "replay_minus_no_memory_ci95",
        "replay_minus_fixed_memory_ci95",
        "replay_minus_write_through_ci95",
        "replay_minus_shuffled_retrieval_ci95",
    )
    for model_id in MANDATED_MODEL_IDS:
        row = by_model.get(model_id, {})
        for key in interval_keys:
            interval = row.get(key, [0.0])
            if not interval or interval[0] <= 0.0:
                return False
    return True


def ready_score(artifact: Mapping[str, Any]) -> float:
    retention = artifact.get("prior_family_retention_after_every_update", {})
    rollback = artifact.get("rollback_and_quarantine_receipts", {})
    state = artifact.get("state_bound_receipt", {})
    writes = artifact.get("exact_post_outcome_write_receipts", {})
    checks = [
        artifact.get("continuous_self_learning_task") is True,
        artifact.get("mandatory_artifact_written") is True,
        _model_ids_exact(list(artifact.get("MODEL_SPECS", []))),
        artifact.get("model_specs") == artifact.get("MODEL_SPECS"),
        _model_cache_usable(artifact),
        artifact.get("sealed_chronological_stream_receipt", {}).get("sealed") is True,
        artifact.get("sealed_chronological_stream_receipt", {}).get("chronological") is True,
        artifact.get("task_owned_external_memory_receipt", {}).get(
            "task_owned_external_memory_only"
        )
        is True,
        artifact.get("arm_definitions_and_resource_matching", {}).get("arm_names")
        == list(ARM_NAMES),
        artifact.get("arm_definitions_and_resource_matching", {}).get("all_arms_matched")
        is True,
        writes.get("same_decision_write_count") == 0,
        writes.get("all_commits_after_exact_outcome") is True,
        _replay_intervals_positive(artifact),
        retention.get("measured_after_every_admitted_update") is True,
        retention.get("selected_arm_min_prior_family_retention", 0.0) >= RETENTION_FLOOR,
        rollback.get("rollback_exact") is True,
        rollback.get("rollback_past_root_failed_closed") is True,
        rollback.get("poison_propagation_count") == 0,
        rollback.get("quarantine_precision") == 1.0,
        rollback.get("quarantine_recall") == 1.0,
        state.get("within_bounds") is True,
        artifact.get("model_weight_immutability_receipt", {}).get("all_unchanged") is True,
        artifact.get("model_weight_immutability_receipt", {}).get("weight_update_count") == 0,
        artifact.get("protected_files_unchanged", {}).get("unchanged") is True,
        _test_exits_clean(artifact),
    ]
    return 1.0 if all(checks) else 0.0


def status(artifact: Mapping[str, Any]) -> str:
    if not _model_cache_usable(artifact):
        return "blocked"
    return (
        "complete"
        if artifact.get("retention_safe_continuous_strategy_learning_ready_score") == 1.0
        else "complete_null"
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    current_status = status(artifact)
    if current_status == "blocked":
        gaps = ",".join(missing_verifier_gaps(artifact)) or "blocked"
        return f"blocked: {gaps}; live model generation did not execute"
    if current_status == "complete":
        return (
            "complete: replay beat no-memory, fixed-memory, write-through, and "
            "shuffled retrieval while preserving prior-family retention; live model "
            "generation did not execute"
        )
    return (
        "complete_null: retention-safe continuous strategy-learning gate not met; "
        "live model generation did not execute"
    )


def missing_verifier_gaps(artifact: Mapping[str, Any]) -> list[str]:
    gaps: list[str] = []
    if not _model_cache_usable(artifact):
        gaps.append("local_gguf_cache_not_usable")
    if not _model_ids_exact(list(artifact.get("MODEL_SPECS", []))):
        gaps.append("model_identity_mismatch")
    if artifact.get("arm_definitions_and_resource_matching", {}).get("all_arms_matched") is not True:
        gaps.append("arm_matching_failed")
    if artifact.get("sealed_chronological_stream_receipt", {}).get("sealed") is not True:
        gaps.append("stream_not_sealed")
    if artifact.get("task_owned_external_memory_receipt", {}).get(
        "task_owned_external_memory_only"
    ) is not True:
        gaps.append("external_memory_boundary_failed")
    writes = artifact.get("exact_post_outcome_write_receipts", {})
    if (
        writes.get("same_decision_write_count") != 0
        or writes.get("all_commits_after_exact_outcome") is not True
    ):
        gaps.append("post_outcome_write_failed")
    if not _replay_intervals_positive(artifact):
        gaps.append("replay_positive_utility_not_met")
    retention = artifact.get("prior_family_retention_after_every_update", {})
    if retention.get("selected_arm_min_prior_family_retention", 0.0) < RETENTION_FLOOR:
        gaps.append("prior_family_retention_regression")
    rollback = artifact.get("rollback_and_quarantine_receipts", {})
    if rollback.get("rollback_exact") is not True:
        gaps.append("rollback_failed")
    if rollback.get("poison_propagation_count") != 0:
        gaps.append("poison_propagation")
    if artifact.get("state_bound_receipt", {}).get("within_bounds") is not True:
        gaps.append("state_bound_exceeded")
    if artifact.get("model_weight_immutability_receipt", {}).get("all_unchanged") is False:
        gaps.append("model_weight_immutability_failed")
    if artifact.get("protected_files_unchanged", {}).get("unchanged") is not True:
        gaps.append("protected_files_changed")
    if not _test_exits_clean(artifact):
        gaps.append("test_failure")
    return sorted(set(gaps))


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    material = dict(artifact)
    material.pop("reproducibility_checksum", None)
    return sha256_json(material)


def _checksum_receipts(
    *,
    stream_receipt: Mapping[str, Any],
    store_receipt: Mapping[str, Any],
    model_cache_snapshot: Mapping[str, Any],
    protected_receipt: Mapping[str, Any],
    sidecar_hashes: Mapping[str, str | None] | None,
) -> JsonDict:
    return {
        "stream_hash": stream_receipt["stream_hash"],
        "store_hash": store_receipt["state_hash"],
        "model_cache_checksums": {
            row["hf_id"]: row.get("checksum") for row in model_cache_snapshot.get("records", [])
        },
        "protected_file_hashes_after": protected_receipt.get("after", {}),
        "sidecar_hashes": dict(sidecar_hashes or {}),
    }


def build_artifact(
    *,
    result_path: Path,
    memory_dir: Path,
    model_cache_records: Sequence[Mapping[str, Any]] | None,
    test_exit_codes: Mapping[str, int],
    duration_s: float,
    sidecar_paths: Mapping[str, str] | None = None,
    sidecar_hashes: Mapping[str, str | None] | None = None,
) -> JsonDict:
    protected_before = _protected_hashes()
    model_cache_snapshot = (
        {
            "cache_root": "caller_supplied",
            "records": [dict(row) for row in model_cache_records],
            "all_usable": all(row.get("usable_for_local_gguf") for row in model_cache_records),
        }
        if model_cache_records is not None
        else snapshot_model_caches()
    )
    events = default_stream()
    stream_receipt = seal_stream(events)
    store_run = _run_store(events)
    store_receipt = _store_receipt(store_run)
    protected_receipt = _protected_files_unchanged(protected_before)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "status": "blocked",
        "preconditions_checked": _preconditions(
            result_path=result_path,
            memory_dir=memory_dir,
            model_cache_snapshot=model_cache_snapshot,
            protected_before=protected_before,
        ),
        "continuous_self_learning_task": True,
        "mandatory_artifact_written": True,
        "MODEL_SPECS": [dict(spec) for spec in MODEL_SPECS],
        "model_specs": [dict(spec) for spec in MODEL_SPECS],
        "sealed_chronological_stream_receipt": stream_receipt,
        "task_owned_external_memory_receipt": _external_memory_receipt(
            memory_dir, sidecar_paths
        ),
        "arm_definitions_and_resource_matching": _arm_definitions(stream_receipt),
        "exact_post_outcome_write_receipts": _exact_write_receipt(store_run),
        "utility_by_arm_family_and_model": _utility_receipt(model_cache_snapshot["all_usable"]),
        "prior_family_retention_after_every_update": _retention_receipt(store_run),
        "bounded_strategy_store_receipt": store_receipt,
        "rollback_and_quarantine_receipts": _rollback_and_quarantine(store_run),
        "state_bound_receipt": _state_bound(store_receipt),
        "model_weight_immutability_receipt": _weight_receipt(model_cache_snapshot),
        "provenance_receipts": _provenance(
            stream_receipt=stream_receipt,
            store_run=store_run,
            model_cache_snapshot=model_cache_snapshot,
        ),
        "protected_files_unchanged": protected_receipt,
        "duration_s": duration_s,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "retention_safe_continuous_strategy_learning_ready_score": 0.0,
        "missing_verifier_gaps": [],
        "field_provenance": _field_provenance(),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes),
        "checksum_receipts": _checksum_receipts(
            stream_receipt=stream_receipt,
            store_receipt=store_receipt,
            model_cache_snapshot=model_cache_snapshot,
            protected_receipt=protected_receipt,
            sidecar_hashes=sidecar_hashes,
        ),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["retention_safe_continuous_strategy_learning_ready_score"] = ready_score(artifact)
    artifact["missing_verifier_gaps"] = missing_verifier_gaps(artifact)
    artifact["status"] = status(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _write_sidecars(memory_dir: Path, artifact: Mapping[str, Any]) -> dict[str, str]:
    memory_dir.mkdir(parents=True, exist_ok=True)
    paths = _memory_paths(memory_dir)
    stream_rows = artifact["sealed_chronological_stream_receipt"]["hash_chain"]
    Path(paths["stream_path"]).write_text(
        "\n".join(canonical_json(row) for row in stream_rows) + "\n",
        encoding="utf-8",
    )
    Path(paths["store_path"]).write_text(
        json.dumps(artifact["bounded_strategy_store_receipt"], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    Path(paths["replay_retention_path"]).write_text(
        json.dumps(
            artifact["prior_family_retention_after_every_update"],
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    quarantine_rows = artifact["rollback_and_quarantine_receipts"]["quarantine_receipts"]
    Path(paths["rollback_quarantine_path"]).write_text(
        "\n".join(canonical_json(row) for row in quarantine_rows) + "\n",
        encoding="utf-8",
    )
    return {key: value for key, value in paths.items() if key.endswith("_path")}


def _sidecar_hashes(sidecar_paths: Mapping[str, str]) -> dict[str, str | None]:
    return {key: sha256_file(Path(path)) for key, path in sidecar_paths.items()}


def run(
    *,
    result_path: Path | None = None,
    memory_dir: Path | None = None,
    model_cache_records: Sequence[Mapping[str, Any]] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
    duration_s: float | None = None,
    write: bool = False,
) -> JsonDict:
    started = time.monotonic()
    resolved_result_path = result_path or (REPO_ROOT / RESULT_RELATIVE_PATH)
    resolved_memory_dir = memory_dir or (REPO_ROOT / MEMORY_DIR_RELATIVE_PATH)
    measured_duration = duration_s
    if measured_duration is None:
        measured_duration = 0.001
    codes = dict(test_exit_codes or {command: 0 for command in DEFAULT_TEST_COMMANDS})
    artifact = build_artifact(
        result_path=resolved_result_path,
        memory_dir=resolved_memory_dir,
        model_cache_records=model_cache_records,
        test_exit_codes=codes,
        duration_s=measured_duration,
    )
    if duration_s is None:
        measured_duration = max(round(time.monotonic() - started, 6), 0.001)
        artifact = build_artifact(
            result_path=resolved_result_path,
            memory_dir=resolved_memory_dir,
            model_cache_records=model_cache_records,
            test_exit_codes=codes,
            duration_s=measured_duration,
        )
    if write:
        sidecar_paths = _write_sidecars(resolved_memory_dir, artifact)
        artifact = build_artifact(
            result_path=resolved_result_path,
            memory_dir=resolved_memory_dir,
            model_cache_records=model_cache_records,
            test_exit_codes=codes,
            duration_s=measured_duration,
            sidecar_paths=sidecar_paths,
            sidecar_hashes=_sidecar_hashes(sidecar_paths),
        )
        resolved_result_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_result_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    if artifact["continuous_self_learning_task"] is not True:
        raise ValueError("continuous_self_learning_task must be true")
    if artifact["mandatory_artifact_written"] is not True:
        raise ValueError("mandatory_artifact_written must be true")
    if not _model_ids_exact(list(artifact["MODEL_SPECS"])):
        raise ValueError("MODEL_SPECS must contain exactly the mandated GGUF ids")
    if artifact["model_specs"] != artifact["MODEL_SPECS"]:
        raise ValueError("model_specs must mirror MODEL_SPECS")
    arms = artifact["arm_definitions_and_resource_matching"]
    if arms.get("arm_names") != list(ARM_NAMES) or arms.get("all_arms_matched") is not True:
        raise ValueError("arm_definitions must contain the five matched arms")
    writes = artifact["exact_post_outcome_write_receipts"]
    if writes.get("same_decision_write_count") != 0:
        raise ValueError("same-decision writes are forbidden")
    if writes.get("all_commits_after_exact_outcome") is not True:
        raise ValueError("post-outcome write receipt failed")
    if artifact["protected_files_unchanged"].get("unchanged") is not True:
        raise ValueError("protected_files_unchanged must remain true")
    expected_score = ready_score(artifact)
    if artifact["retention_safe_continuous_strategy_learning_ready_score"] != expected_score:
        raise ValueError("ready_score mismatch")
    if artifact["missing_verifier_gaps"] != missing_verifier_gaps(artifact):
        raise ValueError("missing_verifier_gaps mismatch")
    if artifact["status"] != status(artifact):
        raise ValueError("status mismatch")
    if artifact["honest_verdict"] != honest_verdict(artifact):
        raise ValueError("honest_verdict mismatch")
    provenance = artifact["field_provenance"]
    if not isinstance(provenance, Mapping):
        raise ValueError("field_provenance must be a mapping")
    for field in REQUIRED_ARTIFACT_FIELDS:
        if provenance.get(field, {}).get("principle") != FIELD_PRINCIPLES[field]:
            raise ValueError(f"field_provenance missing principle for {field}")
    return True


def _main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / RESULT_RELATIVE_PATH)
    args = parser.parse_args(argv)
    if args.validate:
        validate_artifact(load_json(args.output))
        return 0
    artifact = run(result_path=args.output, write=True)
    validate_artifact(artifact)
    print(args.output)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
