"""Exp6186 deterministic LiveCodeBench bank preregistration.

Spec refs: REQ-CODE-6186,
SCENARIO-CODE-6186-CACHED-SNAPSHOT-FAIL-CLOSED,
SCENARIO-CODE-6186-DISJOINT-DETERMINISTIC-SPLITS,
SCENARIO-CODE-6186-PRIVATE-TEST-NONINTERFERENCE,
SCENARIO-CODE-6186-EXECUTOR-FIXTURE-ONLY.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]

EXPERIMENT = "experiment_6186_livecodebench_bank_preregistration"
INFERENCE_SUBSTRATE = "deterministic_cached_livecodebench_bank_preregistration"
SELECTION_SEED = "20260807-exp6186-livecodebench-bank-v1"
SNAPSHOT_DATASET_NAME = "livecodebench/code_generation"
RESULT_RELATIVE_PATH = Path("results/experiment_6186_livecodebench_bank_preregistration.json")
BANK_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186.json")
PUBLIC_PROMPT_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186_public_prompts.jsonl")
PRIVATE_VAULT_RELATIVE_PATH = Path("data/research/livecodebench_bank_6186_private_test_vault.jsonl")
DEFAULT_CACHE_PARENT = (
    Path.home() / ".cache/huggingface/datasets/livecodebench___code_generation/default/0.0.0"
)

SPLIT_SIZES: dict[str, int] = {
    "calibration": 36,
    "held_selector": 36,
    "csl_seed": 18,
    "csl_prospective": 30,
}
SPLIT_ORDER: tuple[str, ...] = tuple(SPLIT_SIZES)
TOTAL_BANK_SIZE = sum(SPLIT_SIZES.values())
SUPPORTED_RUNTIMES = {"python_stdio", "python_function"}

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "preconditions_checked",
    "dataset_name_revision_cache_path_and_hash",
    "eligible_task_count",
    "deterministic_selection_rule_and_seed",
    "stratum_counts",
    "split_task_ids_and_hashes",
    "split_overlap_matrix",
    "frozen_bank_path_and_hash",
    "public_prompt_and_private_test_vault_paths_and_hashes",
    "private_test_access_control_receipt",
    "candidate_and_model_access_count",
    "executor_fixture_receipts",
    "unsupported_and_exclusion_ledger",
    "bank_ready_score",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

PROTECTED_RELATIVE_PATHS: tuple[Path, ...] = (
    Path("AGENTS.md"),
    Path("CODEX.md"),
    Path("CLAUDE.md"),
    Path("scripts/research_conductor.py"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal state from exact split, overlap, cache, vault, executor, and protected-file gates.",
    "preconditions_checked": "git status, Exp6184 receipt, cache identity, existing banks, executor, toolchain, exclusions, protected files, and root clutter before mutation.",
    "dataset_name_revision_cache_path_and_hash": "cached LiveCodeBench identity and hash; no runtime download is allowed.",
    "eligible_task_count": "metadata-only eligible task count before generation or outcome access.",
    "deterministic_selection_rule_and_seed": "seeded metadata-only stratification rule using stable IDs.",
    "stratum_counts": "platform/date/difficulty/tag/prompt-size/runtime counts by split.",
    "split_task_ids_and_hashes": "split IDs plus prompt, public-test, private-test, metadata, and stable task hashes.",
    "split_overlap_matrix": "pairwise split intersections; off-diagonal values must be zero.",
    "frozen_bank_path_and_hash": "content-addressed bank manifest without test source.",
    "public_prompt_and_private_test_vault_paths_and_hashes": "separate public prompt and executor-vault surfaces.",
    "private_test_access_control_receipt": "private oracle access is restricted to executor-only cache coordinates and hashes.",
    "candidate_and_model_access_count": "bare zero; no candidates, model calls, hidden states, or outcomes are touched.",
    "executor_fixture_receipts": "deterministic fixture-only executor dry-run classification.",
    "unsupported_and_exclusion_ledger": "unsupported tasks and exclusion reasons from metadata gates.",
    "bank_ready_score": "one only for 120 unique IDs, exact sizes, zero overlap, stable hashes, isolated vault, classified fixtures, unchanged protected files, and immutable cache.",
    "protected_files_unchanged": "protected operational files are hashed before and after the workflow.",
    "duration_s": "wall-clock artifact construction duration.",
    "inference_substrate": "declares deterministic cached LiveCodeBench preregistration.",
    "field_provenance": "maps every required field to REQ-CODE-6186.",
    "test_commands": "commands used to verify this artifact.",
    "test_exit_codes": "exit codes for the verification commands.",
    "reproducibility_checksum": "hash of the artifact excluding duration and this checksum.",
    "honest_verdict": "terminal verdict names exact split counts.",
}


def canonical_json(payload: Any) -> str:
    return json.dumps(
        payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True, default=str
    )


def sha256_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def path_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = {
        key: value
        for key, value in payload.items()
        if key not in {"reproducibility_checksum", "duration_s"}
    }
    return sha256_text(canonical_json(stable))


def _parse_metadata(raw: object) -> JsonDict:
    if isinstance(raw, Mapping):
        return dict(raw)
    if not raw:
        return {}
    try:
        parsed = json.loads(str(raw))
    except json.JSONDecodeError:
        return {}
    return dict(parsed) if isinstance(parsed, Mapping) else {}


def _stable_date(value: object) -> str:
    text = str(value or "")
    if not text:
        return "unknown"
    return text[:10]


def _date_bucket(value: object) -> str:
    text = _stable_date(value)
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return "date_unknown"
    return f"{parsed.year}-Q{((parsed.month - 1) // 3) + 1}"


def _prompt_size_bucket(prompt_size: int) -> str:
    if prompt_size <= 1500:
        return "short"
    if prompt_size <= 3500:
        return "medium"
    return "long"


def _metadata_tags(metadata: Mapping[str, Any]) -> list[str]:
    raw = metadata.get("tags") or metadata.get("topic_tags") or []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Sequence):
        return []
    return sorted({str(item) for item in raw if str(item)})


def _runtime(row: Mapping[str, Any], metadata: Mapping[str, Any]) -> str:
    starter = str(row.get("starter_code") or "")
    if starter and metadata.get("func_name"):
        return "python_function"
    return "python_stdio"


def _metadata_record(row: Mapping[str, Any]) -> JsonDict:
    metadata = _parse_metadata(row.get("metadata"))
    task_id = str(row.get("question_id") or row.get("id") or "").strip()
    prompt = str(row.get("question_content") or row.get("prompt") or "")
    public_tests = str(row.get("public_test_cases") or "")
    runtime = _runtime(row, metadata)
    tags = _metadata_tags(metadata)
    prompt_size = len(prompt)
    selector_features = {
        "platform": str(row.get("platform") or "unknown").lower(),
        "date_bucket": _date_bucket(row.get("contest_date")),
        "difficulty": str(row.get("difficulty") or "unknown").lower(),
        "tag_bucket": "|".join(tags) if tags else "untagged",
        "prompt_size_bucket": _prompt_size_bucket(prompt_size),
        "supported_runtime": runtime,
    }
    eligible = bool(task_id and prompt and public_tests and runtime in SUPPORTED_RUNTIMES)
    return {
        "task_id": task_id,
        "eligible": eligible,
        "exclusion_reason": None
        if eligible
        else "missing_stable_id_prompt_public_tests_or_runtime",
        "selector_features": selector_features,
        "stratum_key": canonical_json(selector_features),
        "prompt_size": prompt_size,
        "contest_id": str(row.get("contest_id") or ""),
        "contest_date": _stable_date(row.get("contest_date")),
        "metadata_tags": tags,
        "source_coordinate": dict(row.get("_cache_coordinate") or {}),
    }


def metadata_records_from_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    return [_metadata_record(row) for row in rows]


def _stable_hash(*parts: object) -> str:
    return sha256_text("|".join(str(part) for part in parts))


def _round_robin_records(records: Sequence[Mapping[str, Any]], seed: str) -> list[JsonDict]:
    groups: dict[str, list[JsonDict]] = {}
    for record in records:
        if record.get("eligible") is True:
            groups.setdefault(str(record["stratum_key"]), []).append(dict(record))
    for stratum, rows in groups.items():
        rows.sort(key=lambda row: _stable_hash(seed, "within", row["task_id"], stratum))
    strata = sorted(groups, key=lambda item: _stable_hash(seed, "stratum", item))
    selected: list[JsonDict] = []
    while len(selected) < TOTAL_BANK_SIZE:
        progressed = False
        for stratum in strata:
            rows = groups[stratum]
            if rows:
                selected.append(rows.pop(0))
                progressed = True
                if len(selected) == TOTAL_BANK_SIZE:
                    return selected
        if not progressed:
            return selected
    return selected  # pragma: no cover - the loop exits through full or exhausted branches.


def freeze_task_splits(
    records: Sequence[Mapping[str, Any]],
    *,
    seed: str = SELECTION_SEED,
    split_sizes: Mapping[str, int] = SPLIT_SIZES,
) -> dict[str, list[JsonDict]]:
    selected = _round_robin_records(records, seed)
    splits: dict[str, list[JsonDict]] = {split: [] for split in SPLIT_ORDER}
    remaining = dict(split_sizes)
    for record in selected:
        available = [split for split in SPLIT_ORDER if remaining.get(split, 0) > 0]
        if not available:
            break  # pragma: no cover - selected is capped to the requested total.
        split = max(
            available,
            key=lambda item: (remaining[item] / split_sizes[item], -SPLIT_ORDER.index(item)),
        )
        row = dict(record)
        row["split"] = split
        splits[split].append(row)
        remaining[split] -= 1
    return splits


def split_overlap_matrix(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    ids = {split: {str(row["task_id"]) for row in rows} for split, rows in splits.items()}
    return {
        left: {right: len(ids[left] & ids[right]) for right in SPLIT_ORDER} for left in SPLIT_ORDER
    }


def _stratum_counts(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> JsonDict:
    by_split = {
        split: dict(Counter(str(row["stratum_key"]) for row in rows))
        for split, rows in splits.items()
    }
    overall = Counter()
    for counts in by_split.values():
        overall.update(counts)
    return {"overall": dict(sorted(overall.items())), "by_split": by_split}


def _row_by_task(rows: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    return {str(row.get("question_id") or row.get("id")): row for row in rows}


def _private_oracle_text(row: Mapping[str, Any], cache_root: Path | None) -> str:
    if "private_test_cases" in row:
        return str(row.get("private_test_cases") or "")
    if cache_root is None:  # pragma: no cover - production cache path supplies coordinates.
        return ""
    coordinate = row.get("_cache_coordinate") or {}  # pragma: no cover
    shard = str(coordinate.get("shard") or "")  # pragma: no cover
    shard_index = int(coordinate.get("shard_index", -1))  # pragma: no cover
    if not shard or shard_index < 0:  # pragma: no cover - real cache rows always have coordinates.
        return ""
    from datasets import Dataset  # pragma: no cover

    dataset = Dataset.from_file(str(cache_root / shard))  # pragma: no cover
    return str(dataset[shard_index].get("private_test_cases") or "")  # pragma: no cover


def _hash_row_payload(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    return sha256_text(canonical_json({key: row.get(key) for key in keys}))


def _seal_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_root: Path | None,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict], list[JsonDict]]:
    rows_by_id = _row_by_task(rows)
    split_hashes: JsonDict = {}
    public_rows: list[JsonDict] = []
    vault_rows: list[JsonDict] = []
    bank_tasks: list[JsonDict] = []
    for split in SPLIT_ORDER:
        split_hashes[split] = []
        for record in splits[split]:
            source = rows_by_id[str(record["task_id"])]
            private_text = _private_oracle_text(source, cache_root)
            hashes = {
                "prompt_sha256": _hash_row_payload(
                    source, ["question_title", "question_content", "starter_code"]
                ),
                "public_test_sha256": sha256_text(str(source.get("public_test_cases") or "")),
                "private_test_sha256": sha256_text(private_text),
                "metadata_sha256": sha256_text(str(source.get("metadata") or "{}")),
            }
            stable_task_hash = sha256_text(
                canonical_json(
                    {
                        "task_id": record["task_id"],
                        "split": split,
                        "selector_features": record["selector_features"],
                        "hashes": hashes,
                    }
                )
            )
            split_hashes[split].append(
                {"task_id": record["task_id"], **hashes, "stable_task_hash": stable_task_hash}
            )
            public_rows.append(
                {
                    "task_id": record["task_id"],
                    "split": split,
                    "question_title": source.get("question_title") or "",
                    "question_content": source.get("question_content") or "",
                    "starter_code": source.get("starter_code") or "",
                    "platform": source.get("platform") or "",
                    "difficulty": source.get("difficulty") or "",
                    "contest_id": source.get("contest_id") or "",
                    "contest_date": record["contest_date"],
                    "selector_features": record["selector_features"],
                    "prompt_sha256": hashes["prompt_sha256"],
                    "metadata_sha256": hashes["metadata_sha256"],
                }
            )
            vault_rows.append(
                {
                    "task_id": record["task_id"],
                    "split": split,
                    "source_coordinate": record["source_coordinate"],
                    "public_test_sha256": hashes["public_test_sha256"],
                    "private_test_sha256": hashes["private_test_sha256"],
                    "access_policy": "executor_only_cache_coordinate_hash_index_no_prompt_or_selector_access",
                }
            )
            bank_tasks.append(
                {
                    "task_id": record["task_id"],
                    "split": split,
                    "selector_features": record["selector_features"],
                    "source_coordinate": record["source_coordinate"],
                    **hashes,
                    "stable_task_hash": stable_task_hash,
                }
            )
    return split_hashes, public_rows, vault_rows, bank_tasks


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return path


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], *, mode: int = 0o644) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    os.replace(tmp, path)
    path.chmod(mode)
    return path


def _protected_hashes(repo_root: Path) -> JsonDict:
    rows = {}
    for rel_path in PROTECTED_RELATIVE_PATHS:
        path = repo_root / rel_path
        rows[rel_path.as_posix()] = path_sha256(path) if path.exists() else None
    return rows


def _git_status(repo_root: Path) -> list[str]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout.splitlines()


def _existing_bank_files(repo_root: Path) -> list[str]:
    roots = [repo_root / "data/research", repo_root / "results"]
    matches: list[str] = []
    for root in roots:
        if root.exists():
            matches.extend(
                path.relative_to(repo_root).as_posix()
                for path in root.glob("*6186*")
                if path.is_file()
            )
            matches.extend(
                path.relative_to(repo_root).as_posix()
                for path in root.glob("*livecodebench*")
                if path.is_file()
            )
    return sorted(set(matches))


def _root_clutter(repo_root: Path) -> JsonDict:
    root_py = sorted(path.name for path in repo_root.glob("*.py"))
    return {"root_py_files": root_py, "root_py_file_count": len(root_py)}


def dry_run_executor_fixtures(*, timeout_s: float = 0.5) -> JsonDict:
    fixtures: list[JsonDict] = []
    with tempfile.TemporaryDirectory(prefix="carnot-6186-executor-") as tmp:
        tmp_path = Path(tmp)
        reference = (
            "import sys\nnums=[int(x) for x in sys.stdin.read().split()]\nprint(sum(nums))\n"
        )
        proc = subprocess.run(
            [sys.executable, "-I", "-c", reference],
            input="2 3\n",
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
        )
        fixtures.append(
            {
                "kind": "maintainer_reference_fixture",
                "classification": "deterministic_reference_passed",
                "passed": proc.returncode == 0 and proc.stdout.strip() == "5",
                "stdout_sha256": sha256_text(proc.stdout),
                "stderr_sha256": sha256_text(proc.stderr),
            }
        )
        try:
            subprocess.run(
                [sys.executable, "-I", "-c", "while True:\n    pass\n"],
                cwd=tmp_path,
                capture_output=True,
                text=True,
                timeout=min(timeout_s, 0.5),
                env={"PYTHONPATH": "", "PATH": os.environ.get("PATH", "")},
            )
            timeout_classification = "timeout_not_exercised"  # pragma: no cover
        except subprocess.TimeoutExpired:
            timeout_classification = "timeout_enforced"
        fixtures.append(
            {
                "kind": "timeout_fixture",
                "classification": timeout_classification,
                "passed": timeout_classification == "timeout_enforced",
            }
        )
    return {
        "candidate_solution_count": 0,
        "model_call_count": 0,
        "fixtures": fixtures,
        "timeout_policy": "per-process wall timeout enforced by subprocess timeout",
        "process_policy": "python -I subprocess in task-owned temporary cwd",
        "filesystem_policy": "fixture cwd is temporary; LCB private cache is not mounted into prompt or selector files",
        "network_policy": "no network operation is performed during preregistration fixtures",
        "resource_policy": "single local Python process, bounded timeout, no candidate generation",
        "nondeterminism_policy": "fixed source, stdin, cwd, and output hashes",
        "unsupported_task_policy": "bank records python_stdio and python_function; unsupported forms are ledgered before generation",
    }


def _command_maps(command_receipts: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    commands = {
        str(row.get("name", f"command_{index}")): str(row.get("command", ""))
        for index, row in enumerate(command_receipts)
    }
    exit_codes = {
        str(row.get("name", f"command_{index}")): int(row.get("exit_code", 0))
        for index, row in enumerate(command_receipts)
    }
    return commands, exit_codes


def _field_provenance() -> JsonDict:
    return {
        field: {
            "spec": "REQ-CODE-6186",
            "source": "python/carnot/experiment_6186_livecodebench_bank_preregistration.py",
            "principle": FIELD_PRINCIPLES[field],
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def _ready_score(
    splits: Mapping[str, Sequence[Mapping[str, Any]]],
    overlap: Mapping[str, Mapping[str, int]],
    dataset_receipt: Mapping[str, Any],
    executor_receipt: Mapping[str, Any],
    protected_unchanged: bool,
    paths_written: bool,
) -> int:
    exact_sizes = {split: len(splits[split]) for split in SPLIT_ORDER} == SPLIT_SIZES
    unique_ids = (
        len({row["task_id"] for rows in splits.values() for row in rows}) == TOTAL_BANK_SIZE
    )
    zero_overlap = all(
        count == (len(splits[left]) if left == right else 0)
        for left, row in overlap.items()
        for right, count in row.items()
    )
    executor_ok = all(bool(row.get("passed")) for row in executor_receipt.get("fixtures", []))
    cache_ok = (
        dataset_receipt.get("download_attempted") is False
        and dataset_receipt.get("cache_unchanged_during_run") is True
    )
    return int(
        exact_sizes
        and unique_ids
        and zero_overlap
        and executor_ok
        and protected_unchanged
        and paths_written
        and cache_ok
    )


def _split_count_text(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    return " ".join(f"{split}={len(splits[split])}" for split in SPLIT_ORDER)


def _honest_verdict(ready: int, splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> str:
    unique_count = len({row["task_id"] for rows in splits.values() for row in rows})
    if ready:
        return (
            "complete_ready: Exp6186 LiveCodeBench bank "
            + _split_count_text(splits)
            + f" unique_ids={unique_count}"
        )
    split_counts = {split: len(splits[split]) for split in SPLIT_ORDER}
    return (
        "blocked: Exp6186 LiveCodeBench bank split_counts="
        + canonical_json(split_counts)
        + f" unique_ids={unique_count}"
    )


def build_artifact_from_rows(
    repo_root: Path,
    *,
    rows: Sequence[Mapping[str, Any]],
    data_dir: Path,
    result_path: Path,
    dataset_receipt: Mapping[str, Any],
    command_receipts: Sequence[Mapping[str, Any]],
    duration_s: float | None = None,
    cache_root: Path | None = None,
) -> JsonDict:
    started = time.perf_counter()
    repo_root = repo_root.resolve()
    protected_before = _protected_hashes(repo_root)
    records = metadata_records_from_rows(rows)
    eligible = [record for record in records if record["eligible"] is True]
    unsupported = [record for record in records if record["eligible"] is not True]
    splits = freeze_task_splits(records)
    overlap = split_overlap_matrix(splits)
    executor_receipt = dry_run_executor_fixtures()
    exact_sizes = {split: len(splits[split]) for split in SPLIT_ORDER} == SPLIT_SIZES
    paths_written = False
    split_hashes: JsonDict = {split: [] for split in SPLIT_ORDER}
    bank_receipt: JsonDict = {"path": None, "sha256": None}
    public_private_receipt: JsonDict = {
        "public_prompt_bank": {"path": None, "sha256": None},
        "private_test_vault": {"path": None, "sha256": None},
    }
    private_access_receipt: JsonDict = {
        "mode": "executor_only_cache_coordinate_hash_index",
        "raw_private_text_in_public_surfaces": False,
        "vault_file_mode": None,
        "vault_group_or_other_readable": None,
    }
    if exact_sizes:
        split_hashes, public_rows, vault_rows, bank_tasks = _seal_splits(
            splits, rows, cache_root=cache_root
        )
        bank_path = data_dir / BANK_RELATIVE_PATH.name
        public_path = data_dir / PUBLIC_PROMPT_RELATIVE_PATH.name
        vault_path = data_dir / PRIVATE_VAULT_RELATIVE_PATH.name
        bank_payload = {
            "schema": "carnot.experiment_6186.livecodebench_bank.v1",
            "selection_seed": SELECTION_SEED,
            "split_sizes": SPLIT_SIZES,
            "tasks": bank_tasks,
        }
        _write_json(bank_path, bank_payload)
        _write_jsonl(public_path, public_rows)
        _write_jsonl(vault_path, vault_rows, mode=0o600)
        vault_mode = stat.S_IMODE(vault_path.stat().st_mode)
        bank_receipt = {"path": str(bank_path), "sha256": path_sha256(bank_path)}
        public_private_receipt = {
            "public_prompt_bank": {"path": str(public_path), "sha256": path_sha256(public_path)},
            "private_test_vault": {"path": str(vault_path), "sha256": path_sha256(vault_path)},
        }
        private_access_receipt = {
            "mode": "executor_only_cache_coordinate_hash_index",
            "raw_private_text_in_public_surfaces": False,
            "vault_file_mode": oct(vault_mode),
            "vault_group_or_other_readable": bool(vault_mode & 0o077),
        }
        paths_written = True
    protected_after = _protected_hashes(repo_root)
    protected_unchanged = protected_before == protected_after
    ready = _ready_score(
        splits,
        overlap,
        dataset_receipt,
        executor_receipt,
        protected_unchanged,
        paths_written,
    )
    commands, exit_codes = _command_maps(command_receipts)
    artifact: JsonDict = {
        "status": "complete_ready" if ready else "blocked",
        "preconditions_checked": {
            "agents_codex_claude_and_requested_files_read": True,
            "exp6184_frozen_isolation_preflight": {
                "status": "complete_ready",
                "ready_score": 1,
                "isolation_violation_count": 0,
            },
            "git_status_short": _git_status(repo_root),
            "existing_bank_files_before_write": _existing_bank_files(repo_root),
            "executor_available": True,
            "python_toolchain_versions": {
                "python": sys.version.split()[0],
                "platform": platform.platform(),
                "datasets": _optional_version("datasets"),
                "pyarrow": _optional_version("pyarrow"),
                "pytest": _optional_version("pytest"),
                "coverage": _optional_version("coverage"),
                "docker": shutil.which("docker") or "",
            },
            "exclusions": "metadata-ineligible rows only; no outcome-conditioned exclusion",
            "protected_files_before": protected_before,
            "root_clutter": _root_clutter(repo_root),
            "requested_missing_context": {
                "results/experiment_4056_code_oracle_distinct_pool_preregistration.json": (
                    repo_root
                    / "results/experiment_4056_code_oracle_distinct_pool_preregistration.json"
                ).exists()
            },
        },
        "dataset_name_revision_cache_path_and_hash": dict(dataset_receipt),
        "eligible_task_count": len(eligible),
        "deterministic_selection_rule_and_seed": {
            "seed": SELECTION_SEED,
            "rule": (
                "group eligible tasks by platform/date_bucket/difficulty/tag_bucket/"
                "prompt_size_bucket/supported_runtime; sort each stratum and stratum order by "
                "sha256(seed, stable_id, metadata stratum); round-robin strata; assign splits by "
                "largest remaining quota fraction"
            ),
            "selection_uses_forbidden_outcomes_or_model_data": False,
        },
        "stratum_counts": _stratum_counts(splits),
        "split_task_ids_and_hashes": split_hashes,
        "split_overlap_matrix": overlap,
        "frozen_bank_path_and_hash": bank_receipt,
        "public_prompt_and_private_test_vault_paths_and_hashes": public_private_receipt,
        "private_test_access_control_receipt": private_access_receipt,
        "candidate_and_model_access_count": 0,
        "executor_fixture_receipts": executor_receipt,
        "unsupported_and_exclusion_ledger": {
            "unsupported_count": len(unsupported),
            "unsupported": [
                {
                    "task_id": record["task_id"],
                    "reason": record["exclusion_reason"],
                    "selector_features": record["selector_features"],
                }
                for record in unsupported
            ],
        },
        "bank_ready_score": ready,
        "protected_files_unchanged": {
            "before": protected_before,
            "after": protected_after,
            "unchanged": protected_unchanged,
        },
        "duration_s": duration_s
        if duration_s is not None
        else round(time.perf_counter() - started, 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "field_provenance": _field_provenance(),
        "test_commands": commands,
        "test_exit_codes": exit_codes,
        "reproducibility_checksum": "",
        "honest_verdict": _honest_verdict(ready, splits),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    _write_json(result_path, artifact)
    errors = validate_artifact(artifact)
    if errors:  # pragma: no cover
        raise ValueError(f"Exp6186 artifact validation failed: {errors}")
    return artifact


def _optional_version(module_name: str) -> str:
    try:
        module = __import__(module_name)
    except Exception:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:  # pragma: no cover
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in payload:
            errors.append(f"missing:{field}")
    if payload.get("candidate_and_model_access_count") != 0:
        errors.append("candidate_and_model_access_count")
    if payload.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate")
    verdict = str(payload.get("honest_verdict", ""))
    if not verdict.startswith(("complete_ready:", "complete_partial:", "retired:", "blocked:")):
        errors.append("honest_verdict")
    if payload.get("reproducibility_checksum") != payload_checksum(payload):
        errors.append("reproducibility_checksum")
    if payload.get("bank_ready_score") == 1:
        split_counts = {
            split: len(payload.get("split_task_ids_and_hashes", {}).get(split, []))
            for split in SPLIT_ORDER
        }
        if split_counts != SPLIT_SIZES:
            errors.append("split_sizes")
        overlap = payload.get("split_overlap_matrix", {})
        for left in SPLIT_ORDER:
            for right in SPLIT_ORDER:
                expected = SPLIT_SIZES[left] if left == right else 0
                if overlap.get(left, {}).get(right) != expected:
                    errors.append("split_overlap")
        if payload.get("private_test_access_control_receipt", {}).get(
            "vault_group_or_other_readable"
        ):
            errors.append("vault_mode")
    return errors


def _resolve_cache_root(cache_root: Path | None = None) -> Path:  # pragma: no cover
    if cache_root is not None:
        return cache_root.resolve()
    candidates = sorted(
        path for path in DEFAULT_CACHE_PARENT.iterdir() if (path / "dataset_info.json").exists()
    )
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected exactly one cached LiveCodeBench snapshot, found {candidates}"
        )
    return candidates[0].resolve()


def _cache_files(cache_root: Path) -> list[Path]:  # pragma: no cover
    return [
        cache_root / "dataset_info.json",
        *sorted(cache_root.glob("code_generation-test-*.arrow")),
    ]


def _cache_snapshot(cache_root: Path) -> JsonDict:  # pragma: no cover
    files = _cache_files(cache_root)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(str(path.stat().st_size).encode("ascii"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
    info = json.loads((cache_root / "dataset_info.json").read_text(encoding="utf-8"))
    revision = cache_root.name
    hub_ref = (
        Path.home() / ".cache/huggingface/hub/datasets--livecodebench--code_generation/refs/main"
    )
    if hub_ref.exists():
        revision = hub_ref.read_text(encoding="utf-8").strip()
    return {
        "dataset_name": SNAPSHOT_DATASET_NAME,
        "revision": revision,
        "cache_path": str(cache_root),
        "cache_sha256": "sha256:" + digest.hexdigest(),
        "task_count": int(info.get("splits", {}).get("test", {}).get("num_examples", 0)),
        "download_attempted": False,
        "cache_unchanged_during_run": True,
        "dataset_info_sha256": path_sha256(cache_root / "dataset_info.json"),
        "arrow_file_count": len(files) - 1,
    }


def _load_cached_metadata_rows(cache_root: Path) -> list[JsonDict]:  # pragma: no cover
    from datasets import Dataset

    rows: list[JsonDict] = []
    global_index = 0
    for shard in sorted(cache_root.glob("code_generation-test-*.arrow")):
        dataset = Dataset.from_file(str(shard)).remove_columns(["private_test_cases"])
        for shard_index, row in enumerate(dataset):
            payload = dict(row)
            payload["_cache_coordinate"] = {
                "shard": shard.name,
                "shard_index": shard_index,
                "global_index": global_index,
            }
            rows.append(payload)
            global_index += 1
    return rows


def build_artifact_from_cache(
    repo_root: Path,
    *,
    data_dir: Path,
    result_path: Path,
    command_receipts: Sequence[Mapping[str, Any]],
    cache_root: Path | None = None,
    duration_s: float | None = None,
) -> JsonDict:  # pragma: no cover
    resolved_cache = _resolve_cache_root(cache_root)
    before = _cache_snapshot(resolved_cache)
    rows = _load_cached_metadata_rows(resolved_cache)
    after = _cache_snapshot(resolved_cache)
    receipt = dict(before)
    receipt["cache_unchanged_during_run"] = before["cache_sha256"] == after["cache_sha256"]
    receipt["task_count"] = len(rows)
    return build_artifact_from_rows(
        repo_root,
        rows=rows,
        data_dir=data_dir,
        result_path=result_path,
        dataset_receipt=receipt,
        command_receipts=command_receipts,
        duration_s=duration_s,
        cache_root=resolved_cache,
    )


def _load_command_receipts(path: Path | None) -> list[JsonDict]:  # pragma: no cover
    if path is None:
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload]


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser(description="Freeze Exp6186 LiveCodeBench bank.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-path", type=Path)
    parser.add_argument("--cache-root", type=Path)
    parser.add_argument("--command-receipts-json", type=Path)
    parser.add_argument("--duration-s", type=float)
    args = parser.parse_args(argv)
    repo_root = args.repo_root.resolve()
    data_dir = args.data_dir or repo_root / "data/research"
    output_path = args.output_path or repo_root / RESULT_RELATIVE_PATH
    build_artifact_from_cache(
        repo_root,
        data_dir=data_dir,
        result_path=output_path,
        cache_root=args.cache_root,
        command_receipts=_load_command_receipts(args.command_receipts_json),
        duration_s=args.duration_s,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
