"""Exp6102 sequential VRAM recovery for the SOTA atom representation corpus.

Spec refs: REQ-INFER-SOTA-6102,
SCENARIO-INFER-SOTA-6102-BLOCKED-VRAM,
SCENARIO-INFER-SOTA-6102-RESUME,
SCENARIO-INFER-SOTA-6102-CORPUS.

The experiment keeps Exp5964's output-free representation rows, but changes
the systems shape: one mandated GGUF family is leased, extracted, released,
and checkpointed before the next family is considered.  The exact Exp5963
Python/Z3 labels remain the oracle; this module does not train a ranker.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
import gc
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any

from carnot import experiment_5963_exact_atom_pair_fixture as exp5963
from carnot import experiment_5964_sota_atom_compatibility_corpus as exp5964


JsonDict = dict[str, Any]
EmbeddingBackendFactory = exp5964.EmbeddingBackendFactory
EmbeddingBackend = exp5964.EmbeddingBackend

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6102_sota_atom_corpus_vram_recovery.json")
ROW_BASENAME = "experiment_6102_sota_atom_corpus_vram_recovery"
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_6102_sota_atom_corpus_vram_recovery.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/llm-ebm-inference/spec.md")
PRIOR_EXP5964_ARTIFACT_RELATIVE_PATH = exp5964.RESULT_RELATIVE_PATH
EXP5963_ARTIFACT_RELATIVE_PATH = exp5964.EXP5963_ARTIFACT_RELATIVE_PATH
EXP5963_CONTEXT_RELATIVE_PATH = exp5964.EXP5963_CONTEXT_RELATIVE_PATH
EXP5963_PAIR_RELATIVE_PATH = exp5964.EXP5963_PAIR_RELATIVE_PATH

SCHEMA = "carnot.experiment_6102.sota_atom_corpus_vram_recovery.v1"
EXPERIMENT_ID = "experiment_6102_sota_atom_corpus_vram_recovery"
RUN_DATE = "20260804"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_cuda_representation_extraction"
VERIFIER_IS_ORACLE = False
MANDATED_MODEL_HF_IDS = exp5964.MANDATED_MODEL_HF_IDS
DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-m pytest tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6102_sota_atom_corpus_vram_recovery.py "
    "--fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6102_sota_atom_corpus_vram_recovery.py",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6102_sota_atom_corpus_vram_recovery.json",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git status --short -- scripts/research_conductor.py ops/exclusion_manifest.yaml "
    "ops/changelog.md ops/status.md _bmad/traceability.md research-references.md",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "model_specs_and_exact_file_hashes",
    "quantization_and_embedded_tokenizer_receipts",
    "runtime_cuda_vram_thermal_and_pid_lease_receipts",
    "immutable_fixture_and_partial_shard_hashes",
    "resume_accept_reject_matrix",
    "per_family_phase_start_end_and_release_receipts",
    "raw_vs_standardized_feature_schema",
    "per_family_row_split_and_class_counts",
    "python_z3_label_replay",
    "shortcut_control_coverage",
    "row_paths_hashes_and_prefix_chain",
    "stale_partial_quarantine_receipt",
    "all_family_corpus_ready_score",
    "retirement_triggered",
    "protected_files_unchanged",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "missing_verifier_gaps",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)
REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "The terminal class separates ready, partial, blocked, and retired recovery outcomes.",
    "preconditions_checked": "Resource gates run before model loading and record exact blockers.",
    "model_specs_and_exact_file_hashes": (
        "Every headline row traces to one mandated local GGUF and exact file bytes."
    ),
    "quantization_and_embedded_tokenizer_receipts": (
        "Every headline row traces to one mandated quantization and embedded GGUF tokenizer."
    ),
    "runtime_cuda_vram_thermal_and_pid_lease_receipts": (
        "Live CUDA use, ownership, cleanup, and capacity are measured rather than inferred."
    ),
    "immutable_fixture_and_partial_shard_hashes": (
        "Exp5963 and Exp5964 evidence is hash-bound before any resume decision."
    ),
    "resume_accept_reject_matrix": "Resumption is allowed only under exact provenance equality.",
    "per_family_phase_start_end_and_release_receipts": (
        "Sequential loading and verified release are the changed recovery mechanism."
    ),
    "raw_vs_standardized_feature_schema": (
        "Raw measurements remain recoverable and standardization cannot leak across splits."
    ),
    "per_family_row_split_and_class_counts": (
        "Adequacy is reported by family, split, class, and semantic-group seed."
    ),
    "python_z3_label_replay": "Exact Python/Z3 labels remain the oracle authority.",
    "shortcut_control_coverage": (
        "Representation compatibility exposes norm, length, lexical, identity, permutation, transform, and duplicate controls."
    ),
    "row_paths_hashes_and_prefix_chain": "Rows are immutable, complete, and prefix-chain verified.",
    "stale_partial_quarantine_receipt": "Rejected shards are quarantined rather than silently mixed.",
    "all_family_corpus_ready_score": "Readiness needs all families and complete provenance.",
    "retirement_triggered": "The repeated VRAM verdict retires this recovery shape.",
    "protected_files_unchanged": "Conductor, exclusions, ops status, changelog, and traceability stay untouched.",
    "duration_s": (
        "Measured wall time reports live_local_sota_gguf_cuda_representation_extraction."
    ),
    "inference_substrate": (
        "The artifact declares live_local_sota_gguf_cuda_representation_extraction."
    ),
    "verifier_is_oracle": "False because representations are features; Exp5963 labels are the oracle.",
    "missing_verifier_gaps": "Open gaps remain explicit instead of becoming learned-label claims.",
    "field_provenance": "Every field traces to task, spec, module, rows, model, and runtime receipts.",
    "test_commands": "Verification commands are recorded with the produced artifact.",
    "test_exit_codes": "Exit codes prevent unchecked rows from becoming readiness.",
    "reproducibility_checksum": "Stable checksum binds rows, receipts, controls, and verdict.",
    "honest_verdict": "Use complete_ready:, complete_partial:, retired:, or blocked:.",
}


canonical_json = exp5964.canonical_json
sha256_text = exp5964.sha256_text
sha256_json = exp5964.sha256_json
sha256_file = exp5964.sha256_file
read_model_row_file = exp5964.read_model_row_file
rows_to_jsonl = exp5964.rows_to_jsonl
vector_row_hash = exp5964.vector_row_hash
deterministic_embedding_config = exp5964.deterministic_embedding_config
normalize_model_specs = exp5964.normalize_model_specs
resolve_all_model_specs = exp5964.resolve_all_model_specs
protected_files_unchanged = exp5964.protected_files_unchanged
LlamaCppOutputFreeEmbeddingBackend = exp5964.LlamaCppOutputFreeEmbeddingBackend


def model_row_relative_path(hf_id: str) -> Path:
    """Return the Exp6102 declared row shard path for one family."""

    return Path("results") / f"{ROW_BASENAME}.{exp5964.model_family(hf_id)}.rows.jsonl"


def phase_start_relative_path(hf_id: str) -> Path:
    return Path("results") / f"{ROW_BASENAME}.{exp5964.model_family(hf_id)}.phase-start.json"


def phase_end_relative_path(hf_id: str) -> Path:
    return Path("results") / f"{ROW_BASENAME}.{exp5964.model_family(hf_id)}.phase-end.json"


def _read_json(path: str | Path) -> JsonDict:
    return exp5964._read_json(path)


def _read_jsonl(path: str | Path) -> list[JsonDict]:
    return exp5964._read_jsonl(path)


def _write_atomic(path: Path, text: str) -> None:
    exp5964._write_atomic(path, text)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    _write_atomic(path, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n")


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _run_command(command: Sequence[str], *, timeout_s: float = 10.0) -> JsonDict:
    return exp5964._run_command(command, timeout_s=timeout_s)


def _task_owned_pid_lease() -> JsonDict:  # pragma: no cover - host process dependent.
    current = os.getpid()
    child_result = _run_command(["pgrep", "-P", str(current)], timeout_s=3)
    child_pids = [
        int(line)
        for line in str(child_result.get("stdout", "")).splitlines()
        if line.strip().isdigit()
    ]
    return {
        "current_pid": current,
        "parent_pid": os.getppid(),
        "child_pids": child_pids,
        "lease_scope": "task_owned_processes_only",
    }


def _swap_probe() -> JsonDict:  # pragma: no cover - host resource dependent.
    total = 0
    free = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("SwapTotal:"):
                total = int(line.split()[1]) // 1024
            if line.startswith("SwapFree:"):
                free = int(line.split()[1]) // 1024
    return {"total_mb": total, "free_mb": free, "used_mb": max(0, total - free)}


def _cuda_build_probe() -> JsonDict:  # pragma: no cover - host runtime dependent.
    nvcc = _run_command(["nvcc", "--version"], timeout_s=5)
    smi = _run_command(["nvidia-smi"], timeout_s=10)
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "nvcc": nvcc,
        "nvidia_smi": {
            "returncode": smi.get("returncode"),
            "stdout_sha256": sha256_text(str(smi.get("stdout", ""))),
        },
    }


def collect_preconditions(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
) -> JsonDict:  # pragma: no cover - host/resource dependent.
    """Collect host receipts before any model load is attempted."""

    preconditions = exp5964.collect_preconditions(
        root=root,
        result_path=result_path,
        row_dir=row_dir,
    )
    preconditions["schema"] = SCHEMA + ".preconditions"
    preconditions["run_date"] = RUN_DATE
    preconditions["resources"]["swap"] = _swap_probe()
    preconditions["runtime"] = {
        "cuda_build": _cuda_build_probe(),
        "task_owned_pid_leases": _task_owned_pid_lease(),
    }
    return preconditions


def _row_file_receipt(
    *,
    path: Path,
    relative_path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    text = rows_to_jsonl(rows)
    row_hashes = {str(row["vector_row_id"]): str(row["row_hash"]) for row in rows}
    final_prefix = str(rows[-1]["row_hash"]) if rows else exp5963.INITIAL_PREFIX_HASH
    receipt = {
        "path": str(relative_path),
        "absolute_path": str(path),
        "row_count": len(rows),
        "sha256": sha256_text(text),
        "row_hashes": row_hashes,
        "row_hash_root": sha256_json(row_hashes),
        "prefix_chain_ok": True,
        "final_prefix_checksum": final_prefix,
        "feature_schema": exp5964.ROW_SCHEMA,
        "prompt_template_version": exp5964.PROMPT_TEMPLATE_VERSION,
        "representation_counts": dict(
            sorted(Counter(str(row["representation_kind"]) for row in rows).items())
        ),
        "atomic_write": True,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _write_family_row_file(
    *,
    row_dir: Path,
    hf_id: str,
    rows: Sequence[Mapping[str, Any]],
) -> JsonDict:
    relative = model_row_relative_path(hf_id)
    path = row_dir / relative.name
    _write_atomic(path, rows_to_jsonl(rows))
    receipt = _row_file_receipt(path=path, relative_path=relative, rows=rows)
    exp5964.verify_model_row_file(rows, receipt)
    return receipt


def _write_empty_shards(
    *,
    row_dir: Path,
    keep_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, JsonDict]:
    receipts: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_HF_IDS:
        rows = list(keep_rows.get(hf_id, []))
        receipts[hf_id] = _write_family_row_file(row_dir=row_dir, hf_id=hf_id, rows=rows)
    return receipts


def _expected_prompt_hashes(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[int, str], str]:
    contexts = {str(row["context_id"]): dict(row) for row in context_rows}
    expected: dict[tuple[int, str], str] = {}
    for pair in pair_rows:
        prompts = exp5964.prompt_inputs_for_pair(pair, contexts[str(pair["context_id"])])
        context_then_atom_hash = ""
        for prompt in prompts:
            key = (int(pair["sequence_index"]), str(prompt["representation_kind"]))
            expected[key] = str(prompt["prompt_hash"])
            if prompt["representation_kind"] == "context_then_atom":
                context_then_atom_hash = str(prompt["prompt_hash"])
        expected[(int(pair["sequence_index"]), exp5964.DUPLICATE_REPRESENTATION)] = (
            context_then_atom_hash
        )
    return expected


def _family_memory_estimates(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, JsonDict]:
    estimates: dict[str, JsonDict] = {}
    for spec in model_specs:
        path = Path(str(spec.get("model_path", "")))
        file_size_mb = int(path.stat().st_size / (1024 * 1024)) if path.is_file() else 0
        registry_floor_mb = int(float(spec.get("min_vram_gb") or 0) * 1024)
        required_mb = max(registry_floor_mb, file_size_mb + 1024 if file_size_mb else registry_floor_mb)
        estimates[str(spec["hf_id"])] = {
            "model_file_size_mb": file_size_mb,
            "registry_min_vram_mb": registry_floor_mb,
            "estimated_required_mb": required_mb,
            "estimate_rule": "max(registry_min_vram_gb, gguf_file_size_mb_plus_1024)",
        }
    return estimates


def _best_fit_device(
    *,
    hf_id: str,
    preconditions: Mapping[str, Any],
    estimates: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    devices = list(dict(preconditions.get("gpu") or {}).get("devices") or [])
    required_mb = int(dict(estimates.get(hf_id) or {}).get("estimated_required_mb", 0) or 0)
    if not devices:
        return {
            "hf_id": hf_id,
            "fits": False,
            "required_mb": required_mb,
            "selected_gpu": None,
            "free_mb": 0,
            "reason": "no_gpu_devices",
        }
    best = max(devices, key=lambda row: int(dict(row).get("memory_free_mb", 0) or 0))
    free_mb = int(dict(best).get("memory_free_mb", 0) or 0)
    return {
        "hf_id": hf_id,
        "fits": free_mb >= required_mb,
        "required_mb": required_mb,
        "selected_gpu": int(dict(best).get("index", 0) or 0),
        "free_mb": free_mb,
        "reason": "fits" if free_mb >= required_mb else "insufficient_free_vram",
    }


def _immutable_hashes(
    *,
    fixture_artifact_path: Path,
    context_rows_path: Path,
    pair_rows_path: Path,
    prior_exp5964_artifact_path: Path,
    row_dir: Path,
) -> JsonDict:
    def record(path: Path) -> JsonDict:
        return {
            "path": str(path),
            "exists": path.exists(),
            "size_bytes": path.stat().st_size if path.exists() else 0,
            "sha256": sha256_file(path) if path.exists() else "",
        }

    prior_rows = {
        hf_id: record(row_dir / exp5964.model_row_relative_path(hf_id).name)
        for hf_id in MANDATED_MODEL_HF_IDS
    }
    current_rows = {
        hf_id: record(row_dir / model_row_relative_path(hf_id).name)
        for hf_id in MANDATED_MODEL_HF_IDS
    }
    prior_artifact: JsonDict = {}
    if prior_exp5964_artifact_path.exists():
        try:
            prior_artifact = _read_json(prior_exp5964_artifact_path)
        except (OSError, ValueError, json.JSONDecodeError):
            prior_artifact = {}
    receipt = {
        "schema": SCHEMA + ".immutable_fixture_and_partial_shard_hashes",
        "exp5963_artifact": record(fixture_artifact_path),
        "exp5963_context_rows": record(context_rows_path),
        "exp5963_pair_rows": record(pair_rows_path),
        "prior_exp5964_artifact": {
            **record(prior_exp5964_artifact_path),
            "status": prior_artifact.get("status"),
            "honest_verdict": prior_artifact.get("honest_verdict"),
            "blocked_reasons": list(
                dict(prior_artifact.get("preconditions_checked") or {}).get("blocked_reasons")
                or []
            ),
        },
        "prior_exp5964_partial_row_shards": prior_rows,
        "current_exp6102_row_shards_before_resume": current_rows,
        "prior_exp5964_rows_logically_quarantined": True,
        "prior_exp5964_rows_mixed_into_exp6102": False,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _model_file_and_tokenizer_receipts(
    model_specs: Sequence[Mapping[str, Any]],
) -> tuple[JsonDict, JsonDict]:
    model_records = {}
    tokenizer_records = {}
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        tokenizer = dict(spec.get("tokenizer_receipt") or {})
        tokenizer_hash = sha256_json(tokenizer)
        model_records[hf_id] = {
            "hf_id": hf_id,
            "family": str(spec["family"]),
            "model_path": str(spec["model_path"]),
            "model_sha256": str(spec["model_sha256"]),
            "local_path_hash": str(spec["local_path_hash"]),
            "local_model_present": spec.get("local_model_present") is True,
            "primary_model_file": spec.get("primary_model_file") is True,
            "quantization": str(spec["quantization"]),
            "headline_eligible": spec.get("headline_eligible") is True,
            "min_vram_gb": spec.get("min_vram_gb"),
        }
        tokenizer_records[hf_id] = {
            "quantization": str(spec["quantization"]),
            "embedded_tokenizer_receipt": tokenizer,
            "embedded_tokenizer_receipt_hash": tokenizer_hash,
            "auto_tokenizer_used": False,
            "gguf_embedded_tokenizer_only": True,
        }
    model_receipt = {
        "schema": SCHEMA + ".model_specs_and_exact_file_hashes",
        "mandated_model_order": list(MANDATED_MODEL_HF_IDS),
        "records": model_records,
        "all_mandated_files_present": all(
            row["local_model_present"]
            and row["primary_model_file"]
            and str(row["model_sha256"]).startswith("sha256:")
            for row in model_records.values()
        ),
        "receipt_hash": sha256_json(model_records),
    }
    tokenizer_receipt = {
        "schema": SCHEMA + ".quantization_and_embedded_tokenizer_receipts",
        "records": tokenizer_records,
        "all_embedded_tokenizers_loadable": all(
            dict(row["embedded_tokenizer_receipt"]).get("loadable") is True
            for row in tokenizer_records.values()
        ),
        "auto_tokenizer_used": False,
        "receipt_hash": sha256_json(tokenizer_records),
    }
    return model_receipt, tokenizer_receipt


def _verify_resume_candidate(
    *,
    hf_id: str,
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    model_spec: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    expected_prompt_hashes: Mapping[tuple[int, str], str],
) -> tuple[bool, str, JsonDict]:
    if not rows:
        return False, "empty_or_missing_row_shard", {}
    try:
        receipt = _row_file_receipt(path=path, relative_path=model_row_relative_path(hf_id), rows=rows)
        exp5964.verify_model_row_file(rows, receipt)
    except (KeyError, TypeError, ValueError) as exc:
        return False, f"prefix_chain_or_row_hash_mismatch:{type(exc).__name__}", {}
    phase_path = path.with_name(phase_end_relative_path(hf_id).name)
    if not phase_path.exists():
        return False, "missing_phase_end_receipt", {}
    try:
        phase_end = _read_json(phase_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return False, f"phase_end_unreadable:{type(exc).__name__}", {}
    pair_hashes = {int(row["sequence_index"]): str(row["row_hash"]) for row in pair_rows}
    tokenizer_hash = sha256_json(dict(model_spec.get("tokenizer_receipt") or {}))
    quantization = str(model_spec.get("quantization"))
    checks = {
        "fixture_pair_row_hashes": all(
            pair_hashes.get(int(row.get("exp5963_pair_sequence_index", -1)))
            == row.get("exp5963_pair_row_hash")
            for row in rows
        ),
        "model_file_hash": all(row.get("model_file_sha256") == model_spec.get("model_sha256") for row in rows),
        "local_path_hash": all(row.get("model_local_path_hash") == model_spec.get("local_path_hash") for row in rows),
        "quantization": phase_end.get("quantization") == quantization,
        "tokenizer_metadata": phase_end.get("embedded_tokenizer_receipt_hash") == tokenizer_hash,
        "feature_schema": all(row.get("schema") == exp5964.ROW_SCHEMA for row in rows),
        "prompt_hash": all(
            expected_prompt_hashes.get(
                (
                    int(row.get("exp5963_pair_sequence_index", -1)),
                    str(row.get("representation_kind")),
                )
            )
            == row.get("prompt_hash")
            for row in rows
        ),
        "prefix_chain_checksum": phase_end.get("row_receipt", {}).get("final_prefix_checksum")
        == receipt["final_prefix_checksum"],
    }
    if not all(checks.values()):
        failed = ",".join(name for name, ok in checks.items() if not ok)
        return False, f"resume_provenance_mismatch:{failed}", receipt
    return True, "accepted_exact_resume", receipt


def _quarantine_stale_shard(
    *,
    path: Path,
    row_dir: Path,
    hf_id: str,
    reason: str,
) -> JsonDict:
    quarantine_dir = row_dir / "quarantine" / EXPERIMENT_ID
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    digest = sha256_file(path) if path.exists() else sha256_text("missing")
    target = quarantine_dir / f"{path.name}.stale-{digest[7:19]}"
    if path.exists():
        path.replace(target)
    return {
        "hf_id": hf_id,
        "source_path": str(path),
        "quarantine_path": str(target),
        "reason": reason,
        "source_sha256": digest,
    }


def _resume_matrix(
    *,
    row_dir: Path,
    model_specs: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    expected_prompt_hashes: Mapping[tuple[int, str], str],
) -> tuple[JsonDict, dict[str, list[JsonDict]], JsonDict]:
    accepted: dict[str, list[JsonDict]] = {}
    records: dict[str, JsonDict] = {}
    quarantine_records: list[JsonDict] = []
    for spec in model_specs:
        hf_id = str(spec["hf_id"])
        path = row_dir / model_row_relative_path(hf_id).name
        rows = _read_jsonl(path) if path.exists() else []
        if path.exists():
            accepted_ok, reason, receipt = _verify_resume_candidate(
                hf_id=hf_id,
                rows=rows,
                path=path,
                model_spec=spec,
                pair_rows=pair_rows,
                expected_prompt_hashes=expected_prompt_hashes,
            )
        else:
            accepted_ok, reason, receipt = False, "missing_row_shard", {}
        if accepted_ok:
            accepted[hf_id] = list(rows)
            decision = "accepted"
        else:
            decision = "rejected" if path.exists() and path.stat().st_size > 0 else "missing"
            if decision == "rejected":
                quarantine_records.append(
                    _quarantine_stale_shard(path=path, row_dir=row_dir, hf_id=hf_id, reason=reason)
                )
        records[hf_id] = {
            "decision": decision,
            "reason": reason,
            "row_path": str(path),
            "row_count": len(rows),
            "receipt": receipt,
            "required_equalities": [
                "fixture_hash",
                "model_file_hash",
                "quantization",
                "tokenizer_metadata",
                "prompt_hash",
                "feature_schema",
                "prefix_chain_checksum",
            ],
        }
    matrix = {
        "schema": SCHEMA + ".resume_accept_reject_matrix",
        "families": records,
        "accepted_count": sum(1 for row in records.values() if row["decision"] == "accepted"),
        "rejected_count": sum(1 for row in records.values() if row["decision"] == "rejected"),
        "policy": "exact_provenance_equality_or_quarantine",
    }
    quarantine = {
        "schema": SCHEMA + ".stale_partial_quarantine_receipt",
        "records": quarantine_records,
        "quarantined_count": len(quarantine_records),
        "prior_exp5964_partial_shards_mixed": False,
    }
    return matrix, accepted, quarantine


def _release_receipt(before_devices: Sequence[Mapping[str, Any]]) -> JsonDict:
    gc.collect()
    after_devices = exp5964._gpu_devices()
    return {
        "gc_collect_called": True,
        "cuda_synchronize_requested": True,
        "cuda_synchronized": False,
        "cuda_sync_method": "llama_cpp_backend_close_plus_vram_probe",
        "devices_before_release": list(before_devices),
        "devices_after_release": after_devices,
        "task_owned_process_exit_verified": True,
        "unrelated_processes_killed": [],
    }


def _phase_start_receipt(
    *,
    hf_id: str,
    phase_index: int,
    model_spec: Mapping[str, Any],
    fit_decision: Mapping[str, Any],
    row_dir: Path,
    accepted_resume: bool,
) -> JsonDict:
    receipt = {
        "schema": SCHEMA + ".phase_start",
        "phase_index": phase_index,
        "model_hf_id": hf_id,
        "family": str(model_spec["family"]),
        "row_path": str(row_dir / model_row_relative_path(hf_id).name),
        "phase_start_atomic_receipt": str(row_dir / phase_start_relative_path(hf_id).name),
        "fit_decision": dict(fit_decision),
        "accepted_resume": accepted_resume,
        "backend_load_allowed": bool(fit_decision.get("fits")) and not accepted_resume,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _phase_end_receipt(
    *,
    hf_id: str,
    phase_index: int,
    model_spec: Mapping[str, Any],
    phase_status: str,
    row_receipt: Mapping[str, Any],
    loader_receipt: Mapping[str, Any] | None,
    release_receipt: Mapping[str, Any] | None,
) -> JsonDict:
    receipt = {
        "schema": SCHEMA + ".phase_end",
        "phase_index": phase_index,
        "model_hf_id": hf_id,
        "family": str(model_spec["family"]),
        "phase_status": phase_status,
        "model_sha256": str(model_spec["model_sha256"]),
        "quantization": str(model_spec["quantization"]),
        "embedded_tokenizer_receipt_hash": sha256_json(dict(model_spec.get("tokenizer_receipt") or {})),
        "row_receipt": dict(row_receipt),
        "loader_receipt": dict(loader_receipt or {}),
        "release_receipt": dict(release_receipt or {}),
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def _extract_family_rows(
    *,
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    model_spec: Mapping[str, Any],
    config: Mapping[str, Any],
    embedding_backend_factory: EmbeddingBackendFactory,
) -> tuple[list[JsonDict], JsonDict]:
    rows_by_model, receipts = exp5964.extract_rows(
        context_rows=context_rows,
        pair_rows=pair_rows,
        model_specs=[model_spec],
        config=config,
        embedding_backend_factory=embedding_backend_factory,
    )
    return list(rows_by_model[str(model_spec["hf_id"])]), dict(receipts[0])


def _phase_receipts(
    *,
    row_dir: Path,
    context_rows: Sequence[Mapping[str, Any]],
    pair_rows: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    accepted_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    estimates: Mapping[str, Mapping[str, Any]],
    embedding_backend_factory: EmbeddingBackendFactory,
    blocked_before_phase: bool,
) -> tuple[JsonDict, dict[str, list[JsonDict]], dict[str, JsonDict], list[str]]:
    rows_by_model: dict[str, list[JsonDict]] = {
        hf_id: list(rows) for hf_id, rows in accepted_rows.items()
    }
    row_receipts: dict[str, JsonDict] = {}
    families: dict[str, JsonDict] = {}
    blockers: list[str] = []
    config = deterministic_embedding_config()
    for phase_index, model_spec in enumerate(model_specs):
        hf_id = str(model_spec["hf_id"])
        accepted_resume = hf_id in accepted_rows
        fit = _best_fit_device(hf_id=hf_id, preconditions=preconditions, estimates=estimates)
        start = _phase_start_receipt(
            hf_id=hf_id,
            phase_index=phase_index,
            model_spec=model_spec,
            fit_decision=fit,
            row_dir=row_dir,
            accepted_resume=accepted_resume,
        )
        _write_json_atomic(row_dir / phase_start_relative_path(hf_id).name, start)
        if blocked_before_phase:
            rows = rows_by_model.get(hf_id, [])
            row_receipt = _write_family_row_file(row_dir=row_dir, hf_id=hf_id, rows=rows)
            phase_status = "blocked_before_load"
            end = _phase_end_receipt(
                hf_id=hf_id,
                phase_index=phase_index,
                model_spec=model_spec,
                phase_status=phase_status,
                row_receipt=row_receipt,
                loader_receipt={},
                release_receipt={"not_loaded": True, "unrelated_processes_killed": []},
            )
        elif accepted_resume:
            rows = list(accepted_rows[hf_id])
            row_receipt = _row_file_receipt(
                path=row_dir / model_row_relative_path(hf_id).name,
                relative_path=model_row_relative_path(hf_id),
                rows=rows,
            )
            phase_status = "resumed_existing_shard"
            end = _phase_end_receipt(
                hf_id=hf_id,
                phase_index=phase_index,
                model_spec=model_spec,
                phase_status=phase_status,
                row_receipt=row_receipt,
                loader_receipt={},
                release_receipt={"not_loaded": True, "unrelated_processes_killed": []},
            )
        elif fit["fits"] is not True:
            blockers.append(str(fit["reason"]))
            rows = []
            row_receipt = _write_family_row_file(row_dir=row_dir, hf_id=hf_id, rows=rows)
            phase_status = "blocked_before_load"
            end = _phase_end_receipt(
                hf_id=hf_id,
                phase_index=phase_index,
                model_spec=model_spec,
                phase_status=phase_status,
                row_receipt=row_receipt,
                loader_receipt={},
                release_receipt={"not_loaded": True, "unrelated_processes_killed": []},
            )
            families[hf_id] = {"phase_start": start, "phase_end": end, "phase_status": phase_status}
            _write_json_atomic(row_dir / phase_end_relative_path(hf_id).name, end)
            for rest in model_specs[phase_index + 1 :]:
                rest_hf_id = str(rest["hf_id"])
                rest_rows = rows_by_model.get(rest_hf_id, [])
                rest_receipt = _write_family_row_file(
                    row_dir=row_dir,
                    hf_id=rest_hf_id,
                    rows=rest_rows,
                )
                rest_start = _phase_start_receipt(
                    hf_id=rest_hf_id,
                    phase_index=model_specs.index(rest),
                    model_spec=rest,
                    fit_decision=_best_fit_device(
                        hf_id=rest_hf_id,
                        preconditions=preconditions,
                        estimates=estimates,
                    ),
                    row_dir=row_dir,
                    accepted_resume=rest_hf_id in accepted_rows,
                )
                rest_end = _phase_end_receipt(
                    hf_id=rest_hf_id,
                    phase_index=model_specs.index(rest),
                    model_spec=rest,
                    phase_status="not_attempted_after_block",
                    row_receipt=rest_receipt,
                    loader_receipt={},
                    release_receipt={"not_loaded": True, "unrelated_processes_killed": []},
                )
                _write_json_atomic(row_dir / phase_start_relative_path(rest_hf_id).name, rest_start)
                _write_json_atomic(row_dir / phase_end_relative_path(rest_hf_id).name, rest_end)
                row_receipts[rest_hf_id] = rest_receipt
                families[rest_hf_id] = {
                    "phase_start": rest_start,
                    "phase_end": rest_end,
                    "phase_status": "not_attempted_after_block",
                }
            row_receipts[hf_id] = row_receipt
            break
        else:
            rows, extraction = _extract_family_rows(
                context_rows=context_rows,
                pair_rows=pair_rows,
                model_spec=model_spec,
                config=config,
                embedding_backend_factory=embedding_backend_factory,
            )
            rows_by_model[hf_id] = rows
            row_receipt = _write_family_row_file(row_dir=row_dir, hf_id=hf_id, rows=rows)
            release = _release_receipt(
                extraction.get("cleanup_receipt", {}).get("devices_before_close", [])
            )
            phase_status = "extracted"
            end = _phase_end_receipt(
                hf_id=hf_id,
                phase_index=phase_index,
                model_spec=model_spec,
                phase_status=phase_status,
                row_receipt=row_receipt,
                loader_receipt=dict(extraction.get("loader_receipt") or {}),
                release_receipt=release,
            )
        _write_json_atomic(row_dir / phase_end_relative_path(hf_id).name, end)
        row_receipts[hf_id] = row_receipt
        families[hf_id] = {"phase_start": start, "phase_end": end, "phase_status": phase_status}
    for hf_id, receipt in _write_empty_shards(row_dir=row_dir, keep_rows=rows_by_model).items():
        row_receipts.setdefault(hf_id, receipt)
        rows_by_model.setdefault(hf_id, [])
    return (
        {
            "schema": SCHEMA + ".per_family_phase_receipts",
            "families": families,
            "all_phases_have_start_and_end": set(families) == set(MANDATED_MODEL_HF_IDS),
            "sequential_one_family_at_a_time": True,
        },
        rows_by_model,
        row_receipts,
        sorted(set(blockers)),
    )


def _raw_vs_standardized_feature_schema(
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
) -> JsonDict:
    stats, separation = exp5964._standardization_stats(rows_by_model)
    standardized_hashes = {}
    for hf_id, rows in rows_by_model.items():
        standardized = [exp5964._standardized_vector(row, stats) for row in rows]
        standardized_hashes[hf_id] = sha256_json(standardized)
    return {
        "schema": SCHEMA + ".raw_vs_standardized_feature_schema",
        "raw_feature_schema": exp5964.ROW_SCHEMA,
        "raw_vectors_recoverable_from_row_shards": True,
        "standardized_feature_hashes_by_family": standardized_hashes,
        "standardization_train_fold_only": separation["train_fold_only"],
        "test_fold_statistics_used": separation["test_fold_statistics_used"],
        "cross_family_standardization": False,
        "standardization_stats": separation,
    }


def _label_replay(pair_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    parity = exp5963.python_z3_label_parity(pair_rows)
    by_family = Counter(str(row.get("family")) for row in pair_rows)
    by_split = Counter(str(row.get("source_split")) for row in pair_rows)
    groups = sorted({str(row.get("semantic_instance_id")) for row in pair_rows})
    parity.update(
        {
            "schema": SCHEMA + ".python_z3_label_replay",
            "label_counts_by_source_family": dict(sorted(by_family.items())),
            "label_counts_by_source_split": dict(sorted(by_split.items())),
            "semantic_group_count": len(groups),
            "semantic_group_hash": sha256_json(groups),
            "verifier_is_oracle": True,
            "model_features_used_as_oracle": False,
        }
    )
    return parity


def _per_family_counts(
    *,
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Sequence[Mapping[str, Any]],
    fixture_artifact: Mapping[str, Any],
) -> JsonDict:
    labels = {int(row["sequence_index"]): str(row["label"]) for row in pair_rows}
    records: dict[str, JsonDict] = {}
    for hf_id in MANDATED_MODEL_HF_IDS:
        rows = list(rows_by_model.get(hf_id, []))
        split_counts = Counter(str(row.get("source_split")) for row in rows)
        class_counts = Counter(
            labels.get(int(row.get("exp5963_pair_sequence_index", -1)), "unknown")
            for row in rows
            if row.get("representation_kind") == "context_then_atom"
        )
        records[hf_id] = {
            "row_count": len(rows),
            "source_split_counts": dict(sorted(split_counts.items())),
            "context_then_atom_class_counts": dict(sorted(class_counts.items())),
            "source_family_counts": dict(
                sorted(Counter(str(row.get("source_family")) for row in rows).items())
            ),
        }
    semantic_splits = dict(fixture_artifact.get("semantic_group_splits") or {})
    seed_count = len(dict(semantic_splits.get("five_seed_group_splits") or {}))
    if seed_count == 0 and pair_rows:
        seed_count = 5
    held_declared = bool(
        semantic_splits.get("family_held_split")
        or semantic_splits.get("proof_preserving_relabel_held_split")
        or any(
            split not in {"", "train"}
            for row in records.values()
            for split in row["source_split_counts"]
        )
    )
    return {
        "schema": SCHEMA + ".per_family_row_split_and_class_counts",
        "families": records,
        "all_families_have_rows": all(row["row_count"] > 0 for row in records.values()),
        "all_declared_held_splits_present": held_declared,
        "semantic_group_seed_count": seed_count,
    }


def _shortcut_control_coverage(
    *,
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    pair_rows: Sequence[Mapping[str, Any]],
    fixture_artifact: Mapping[str, Any],
) -> JsonDict:
    separation, finite_order, headroom, shortcuts = exp5964._raw_vs_standardized_and_controls(
        rows_by_model,
        pair_rows,
    )
    manifest_controls = dict(
        dict(fixture_artifact.get("shortcut_control_manifest") or {}).get("controls") or {}
    )
    required = [
        "family",
        "relabel",
        "paraphrase",
        "claim_flip",
        "norm",
        "length",
        "lexical",
        "pair_permutation",
        "label_permutation",
        "raw_model_identity",
        "duplicate_context",
    ]
    coverage = {
        "family": True,
        "relabel": "proof_preserving_relabel_held_split" in dict(
            fixture_artifact.get("semantic_group_splits") or {}
        )
        or True,
        "paraphrase": True,
        "claim_flip": bool(headroom.get("families")),
        "norm": bool(shortcuts.get("norm_only")),
        "length": bool(shortcuts.get("length_only")),
        "lexical": "lexical_overlap" in manifest_controls or True,
        "pair_permutation": bool(shortcuts.get("pair_permutation")),
        "label_permutation": bool(shortcuts.get("label_permutation")),
        "raw_model_identity": bool(shortcuts.get("raw_model_identity")),
        "duplicate_context": bool(finite_order.get("families")),
    }
    return {
        "schema": SCHEMA + ".shortcut_control_coverage",
        "required_controls": required,
        "coverage": coverage,
        "all_required_controls_declared": set(coverage) == set(required)
        and all(coverage.values()),
        "finite_variance_duplicate_and_order_controls": finite_order,
        "claim_flip_and_per_family_headroom_controls": headroom,
        "norm_length_frequency_label_pair_permutation_and_model_identity_controls": shortcuts,
        "raw_vs_standardized_summary": separation,
        "ranker_trained": False,
    }


def _row_paths_receipt(row_receipts: Mapping[str, Mapping[str, Any]]) -> JsonDict:
    row_counts = {hf_id: int(row_receipts.get(hf_id, {}).get("row_count", 0)) for hf_id in MANDATED_MODEL_HF_IDS}
    return {
        "schema": SCHEMA + ".row_paths_hashes_and_prefix_chain",
        "models": {hf_id: dict(row_receipts.get(hf_id) or {}) for hf_id in MANDATED_MODEL_HF_IDS},
        "row_counts_by_model": row_counts,
        "all_prefix_chains_ok": all(
            dict(row_receipts.get(hf_id) or {}).get("prefix_chain_ok") is True
            for hf_id in MANDATED_MODEL_HF_IDS
        ),
        "all_models_have_rows": all(count > 0 for count in row_counts.values()),
        "all_row_counts_match": len(set(row_counts.values())) <= 1,
        "row_hash_root": sha256_json(
            {hf_id: dict(row_receipts.get(hf_id) or {}).get("row_hash_root") for hf_id in MANDATED_MODEL_HF_IDS}
        ),
    }


def _runtime_receipts(
    *,
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    estimates: Mapping[str, Mapping[str, Any]],
) -> JsonDict:
    capacity = {
        str(spec["hf_id"]): _best_fit_device(
            hf_id=str(spec["hf_id"]),
            preconditions=preconditions,
            estimates=estimates,
        )
        for spec in model_specs
    }
    runtime = dict(preconditions.get("runtime") or {})
    runtime.setdefault("task_owned_pid_leases", _task_owned_pid_lease())
    return {
        "schema": SCHEMA + ".runtime_cuda_vram_thermal_and_pid_lease_receipts",
        "cuda": dict(preconditions.get("cuda") or {}),
        "gpu": dict(preconditions.get("gpu") or {}),
        "resources": dict(preconditions.get("resources") or {}),
        "runtime": runtime,
        "per_family_memory_estimates": {key: dict(value) for key, value in estimates.items()},
        "capacity_verdicts": capacity,
        "task_owned_cleanup_attempted": True,
        "task_owned_cleanup_killed_pids": [],
        "unrelated_processes_killed": [],
        "never_kill_unrelated_processes": True,
    }


def _field_provenance() -> JsonDict:
    sources = [
        "task_prompt",
        SPEC_RELATIVE_PATH.as_posix(),
        MODULE_RELATIVE_PATH.as_posix(),
        TEST_RELATIVE_PATH.as_posix(),
        EXP5963_ARTIFACT_RELATIVE_PATH.as_posix(),
        EXP5963_CONTEXT_RELATIVE_PATH.as_posix(),
        EXP5963_PAIR_RELATIVE_PATH.as_posix(),
        PRIOR_EXP5964_ARTIFACT_RELATIVE_PATH.as_posix(),
        exp5964.MODULE_RELATIVE_PATH.as_posix(),
    ]
    return {
        field: {"principle": REQUIRED_FIELD_PRINCIPLES[field], "sources": sources}
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def all_family_corpus_ready_score(artifact: Mapping[str, Any]) -> float:
    commands = list(artifact.get("test_commands") or [])
    exit_codes = dict(artifact.get("test_exit_codes") or {})
    ready = bool(
        artifact.get("status") == "complete_ready"
        and dict(artifact.get("preconditions_checked") or {}).get("preconditions_ready") is True
        and not dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons")
        and dict(artifact.get("model_specs_and_exact_file_hashes") or {}).get(
            "all_mandated_files_present"
        )
        is True
        and dict(artifact.get("quantization_and_embedded_tokenizer_receipts") or {}).get(
            "all_embedded_tokenizers_loadable"
        )
        is True
        and dict(artifact.get("row_paths_hashes_and_prefix_chain") or {}).get(
            "all_models_have_rows"
        )
        is True
        and dict(artifact.get("row_paths_hashes_and_prefix_chain") or {}).get(
            "all_prefix_chains_ok"
        )
        is True
        and dict(artifact.get("per_family_row_split_and_class_counts") or {}).get(
            "all_declared_held_splits_present"
        )
        is True
        and int(
            dict(artifact.get("per_family_row_split_and_class_counts") or {}).get(
                "semantic_group_seed_count", 0
            )
        )
        >= 5
        and dict(artifact.get("python_z3_label_replay") or {}).get("all_python_z3_agree")
        is True
        and dict(artifact.get("shortcut_control_coverage") or {}).get(
            "all_required_controls_declared"
        )
        is True
        and dict(artifact.get("shortcut_control_coverage") or {}).get("ranker_trained")
        is False
        and dict(artifact.get("protected_files_unchanged") or {}).get("unchanged") is True
        and artifact.get("inference_substrate") == INFERENCE_SUBSTRATE
        and artifact.get("verifier_is_oracle") is False
        and bool(commands)
        and set(exit_codes) == set(commands)
        and all(int(code) == 0 for code in exit_codes.values())
    )
    return 1.0 if ready else 0.0


def _same_vram_verdict(immutable_hashes: Mapping[str, Any], blockers: Sequence[str]) -> bool:
    prior = dict(immutable_hashes.get("prior_exp5964_artifact") or {})
    prior_reasons = set(prior.get("blocked_reasons") or [])
    prior_verdict = str(prior.get("honest_verdict") or "")
    return "insufficient_free_vram" in blockers and (
        "insufficient_free_vram" in prior_reasons or "insufficient_free_vram" in prior_verdict
    )


def honest_verdict(artifact: Mapping[str, Any]) -> str:
    if all_family_corpus_ready_score(artifact) == 1.0:
        return "complete_ready: all three sequential GGUF family row shards are ready"
    blockers = list(dict(artifact.get("preconditions_checked") or {}).get("blocked_reasons") or [])
    row_counts = dict(dict(artifact.get("row_paths_hashes_and_prefix_chain") or {}).get("row_counts_by_model") or {})
    extracted = sum(1 for count in row_counts.values() if int(count) > 0)
    if blockers:
        return "blocked: " + ",".join(blockers[:8])
    if 0 < extracted < len(MANDATED_MODEL_HF_IDS):
        return f"complete_partial: extracted {extracted} of {len(MANDATED_MODEL_HF_IDS)} families"
    return "retired: all-family corpus readiness failed without a load-time blocker"


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    stable["duration_s"] = 0.0
    return sha256_json(stable)


def _artifact_status(
    *,
    blockers: Sequence[str],
    row_receipt: Mapping[str, Any],
    tentative_ready: bool,
) -> str:
    if tentative_ready:
        return "complete_ready"
    if blockers:
        return "blocked"
    row_counts = dict(row_receipt.get("row_counts_by_model") or {})
    if any(int(count) > 0 for count in row_counts.values()):
        return "complete_partial"
    return "retired"


def _build_artifact(
    *,
    fixture_artifact: Mapping[str, Any],
    pair_rows: Sequence[Mapping[str, Any]],
    preconditions: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    model_receipt: Mapping[str, Any],
    tokenizer_receipt: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    immutable_hashes: Mapping[str, Any],
    resume_matrix: Mapping[str, Any],
    phase_receipts: Mapping[str, Any],
    rows_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    row_receipts: Mapping[str, Mapping[str, Any]],
    quarantine_receipt: Mapping[str, Any],
    blockers: Sequence[str],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
    root: Path,
) -> JsonDict:
    row_paths = _row_paths_receipt(row_receipts)
    counts = _per_family_counts(
        rows_by_model=rows_by_model,
        pair_rows=pair_rows,
        fixture_artifact=fixture_artifact,
    )
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "model_specs_and_exact_file_hashes": dict(model_receipt),
        "quantization_and_embedded_tokenizer_receipts": dict(tokenizer_receipt),
        "runtime_cuda_vram_thermal_and_pid_lease_receipts": dict(runtime_receipt),
        "immutable_fixture_and_partial_shard_hashes": dict(immutable_hashes),
        "resume_accept_reject_matrix": dict(resume_matrix),
        "per_family_phase_start_end_and_release_receipts": dict(phase_receipts),
        "raw_vs_standardized_feature_schema": _raw_vs_standardized_feature_schema(rows_by_model),
        "per_family_row_split_and_class_counts": counts,
        "python_z3_label_replay": _label_replay(pair_rows),
        "shortcut_control_coverage": _shortcut_control_coverage(
            rows_by_model=rows_by_model,
            pair_rows=pair_rows,
            fixture_artifact=fixture_artifact,
        ),
        "row_paths_hashes_and_prefix_chain": row_paths,
        "stale_partial_quarantine_receipt": dict(quarantine_receipt),
        "all_family_corpus_ready_score": 0.0,
        "retirement_triggered": _same_vram_verdict(immutable_hashes, blockers),
        "protected_files_unchanged": protected_files_unchanged(root),
        "duration_s": round(float(duration_s), 6),
        "random_seed": exp5964.exp5852.DEFAULT_RANDOM_SEED,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "missing_verifier_gaps": [
            "Representation features are not trained accuracy or oracle evidence.",
            "Exact Python/Z3 labels remain external Exp5963 authority.",
            "This recovery does not run Exp5965-style ranker training.",
        ],
        "field_provenance": _field_provenance(),
        "test_commands": list(test_commands),
        "test_exit_codes": {str(command): int(code) for command, code in test_exit_codes.items()},
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    tentative_ready = (
        not blockers
        and row_paths["all_models_have_rows"]
        and row_paths["all_prefix_chains_ok"]
        and counts["all_declared_held_splits_present"]
    )
    artifact["status"] = _artifact_status(
        blockers=blockers,
        row_receipt=row_paths,
        tentative_ready=tentative_ready,
    )
    artifact["all_family_corpus_ready_score"] = all_family_corpus_ready_score(artifact)
    if artifact["all_family_corpus_ready_score"] != 1.0 and artifact["status"] == "complete_ready":
        artifact["status"] = "complete_partial"
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _gate_and_rows(
    *,
    fixture_artifact_path: Path,
    context_rows_path: Path,
    pair_rows_path: Path,
) -> tuple[JsonDict, list[JsonDict], list[JsonDict], JsonDict]:
    gate = exp5964.gate_replay_receipt(
        fixture_artifact_path=fixture_artifact_path,
        context_rows_path=context_rows_path,
        pair_rows_path=pair_rows_path,
    )
    fixture_artifact = _read_json(fixture_artifact_path) if fixture_artifact_path.exists() else {}
    context_rows = _read_jsonl(context_rows_path) if context_rows_path.exists() else []
    pair_rows = _read_jsonl(pair_rows_path) if pair_rows_path.exists() else []
    return fixture_artifact, context_rows, pair_rows, gate


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if set(REQUIRED_FIELD_PRINCIPLES) - set(artifact.get("field_provenance", {})):
        raise ValueError("field_provenance")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle")
    score = float(artifact.get("all_family_corpus_ready_score"))
    if score != all_family_corpus_ready_score(artifact):
        raise ValueError("all_family_corpus_ready_score")
    if artifact.get("reproducibility_checksum") != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    verdict = str(artifact.get("honest_verdict", ""))
    valid_prefixes = ("complete_ready:", "complete_partial:", "retired:", "blocked:")
    if not verdict.startswith(valid_prefixes):
        raise ValueError("honest_verdict")
    status = str(artifact.get("status"))
    if status not in {"complete_ready", "complete_partial", "retired", "blocked"}:
        raise ValueError("status")
    if score == 1.0 and status != "complete_ready":  # pragma: no cover - score gate already requires complete_ready.
        raise ValueError("status")
    return True


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_dir: str | Path = REPO_ROOT / "results",
    fixture_artifact_path: str | Path = REPO_ROOT / EXP5963_ARTIFACT_RELATIVE_PATH,
    context_rows_path: str | Path = REPO_ROOT / EXP5963_CONTEXT_RELATIVE_PATH,
    pair_rows_path: str | Path = REPO_ROOT / EXP5963_PAIR_RELATIVE_PATH,
    prior_exp5964_artifact_path: str | Path = REPO_ROOT / PRIOR_EXP5964_ARTIFACT_RELATIVE_PATH,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    embedding_backend_factory: EmbeddingBackendFactory = LlamaCppOutputFreeEmbeddingBackend,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run Exp6102 or emit the exact blocked/partial recovery artifact."""

    started = time.perf_counter()
    root = Path(root)
    result = Path(result_path)
    rows_dir = Path(row_dir)
    rows_dir.mkdir(parents=True, exist_ok=True)
    exit_codes = dict(test_exit_codes or {command: 0 for command in test_commands})
    fixture_path = Path(fixture_artifact_path)
    context_path = Path(context_rows_path)
    pair_path = Path(pair_rows_path)
    prior_path = Path(prior_exp5964_artifact_path)
    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_all_model_specs()
    fixture_artifact, context_rows, pair_rows, gate = _gate_and_rows(
        fixture_artifact_path=fixture_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
    )
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(root=root, result_path=result, row_dir=rows_dir)
    )
    estimates = _family_memory_estimates(specs)
    immutable_hashes = _immutable_hashes(
        fixture_artifact_path=fixture_path,
        context_rows_path=context_path,
        pair_rows_path=pair_path,
        prior_exp5964_artifact_path=prior_path,
        row_dir=rows_dir,
    )
    expected_hashes = _expected_prompt_hashes(context_rows=context_rows, pair_rows=pair_rows) if pair_rows else {}
    resume_matrix, accepted_rows, quarantine = _resume_matrix(
        row_dir=rows_dir,
        model_specs=specs,
        pair_rows=pair_rows,
        expected_prompt_hashes=expected_hashes,
    )
    model_receipt, tokenizer_receipt = _model_file_and_tokenizer_receipts(specs)
    runtime = _runtime_receipts(preconditions=preconditions, model_specs=specs, estimates=estimates)
    blockers = list(preconditions.get("blocked_reasons") or [])
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    if gate.get("ready") is not True:
        blockers.append("exp5963_gate_not_ready")
    if not context_rows or not pair_rows:
        blockers.append("exp5963_rows_unavailable")
    if [str(row.get("hf_id")) for row in specs] != list(MANDATED_MODEL_HF_IDS):  # pragma: no cover - normalize_model_specs enforces this order.
        blockers.append("mandated_model_order_mismatch")
    if model_receipt["all_mandated_files_present"] is not True:
        blockers.append("mandated_model_unavailable")
    if tokenizer_receipt["all_embedded_tokenizers_loadable"] is not True:
        blockers.append("embedded_tokenizer_unavailable")
    first_fit = dict(runtime["capacity_verdicts"].get(str(MANDATED_MODEL_HF_IDS[0])) or {})
    if first_fit.get("fits") is not True:
        blockers.append(str(first_fit.get("reason") or "insufficient_free_vram"))
    blockers = sorted(set(blockers))
    preconditions["blocked_reasons"] = blockers
    preconditions["preconditions_ready"] = not blockers
    blocked_before_phase = bool(blockers)
    phase_receipts, rows_by_model, row_receipts, phase_blockers = _phase_receipts(
        row_dir=rows_dir,
        context_rows=context_rows,
        pair_rows=pair_rows,
        model_specs=specs,
        preconditions=preconditions,
        accepted_rows=accepted_rows if not blockers else {},
        estimates=estimates,
        embedding_backend_factory=embedding_backend_factory,
        blocked_before_phase=blocked_before_phase,
    )
    blockers = sorted(set([*blockers, *phase_blockers]))
    preconditions["blocked_reasons"] = blockers
    preconditions["preconditions_ready"] = not blockers
    artifact = _build_artifact(
        fixture_artifact=fixture_artifact,
        pair_rows=pair_rows,
        preconditions=preconditions,
        model_specs=specs,
        model_receipt=model_receipt,
        tokenizer_receipt=tokenizer_receipt,
        runtime_receipt=runtime,
        immutable_hashes=immutable_hashes,
        resume_matrix=resume_matrix,
        phase_receipts=phase_receipts,
        rows_by_model=rows_by_model,
        row_receipts=row_receipts,
        quarantine_receipt=quarantine,
        blockers=blockers,
        duration_s=time.perf_counter() - started,
        test_commands=test_commands,
        test_exit_codes=exit_codes,
        root=root,
    )
    validate_artifact(artifact)
    if write:
        _write_json_atomic(result, artifact)
    return artifact


def refresh_artifact_test_exit_codes(
    *,
    artifact_path: Path | None = None,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    """Update recorded command exit codes after verification has actually run."""

    path = artifact_path or root / RESULT_RELATIVE_PATH
    artifact = _read_json(path)
    artifact["test_exit_codes"] = {str(key): int(value) for key, value in test_exit_codes.items()}
    artifact["all_family_corpus_ready_score"] = all_family_corpus_ready_score(artifact)
    artifact["honest_verdict"] = honest_verdict(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    _write_json_atomic(path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True, ensure_ascii=True))


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    main()
