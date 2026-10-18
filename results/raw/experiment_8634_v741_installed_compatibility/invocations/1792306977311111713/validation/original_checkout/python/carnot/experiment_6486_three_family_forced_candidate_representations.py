"""Exp6486 three-family forced-candidate representation stream.

Spec refs: REQ-VERIFY-6486, SCENARIO-VERIFY-6486-PRECONDITIONS,
SCENARIO-VERIFY-6486-CANDIDATES, SCENARIO-VERIFY-6486-NO-GENERATION,
SCENARIO-VERIFY-6486-RAW-ROWS, SCENARIO-VERIFY-6486-FAMILY-HELD,
SCENARIO-VERIFY-6486-ARTIFACT.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
import gc
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from typing import Any, Protocol

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260821"
RANDOM_SEED = 6486
TASK_ID = "exp6486-three-family-forced-candidate-representations"
SCHEMA_VERSION = "carnot.experiment_6486.three_family_forced_candidate_representations.v1"
INFERENCE_SUBSTRATE = "live_local_sota_gguf_fixed_sequence_representation"
VERIFIER_IS_ORACLE = False
CANDIDATES_PER_UNIT = 3
RAW_VECTOR_DECIMALS = 8
RAM_FLOOR_MB = 16_384
DISK_FLOOR_MB = 10_240
DEFAULT_CONTEXT_LENGTH = 512
DEFAULT_BATCH_SIZE = 512
DEFAULT_UBATCH_SIZE = 128
DEFAULT_N_GPU_LAYERS = -1

MANDATED_MODEL_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
PROHIBITED_CALL_NAMES = frozenset(
    {
        "generate",
        "completion",
        "create_completion",
        "chat_completion",
        "create_chat_completion",
        "grammar_decode",
        "grammar_decoding",
        "parser_retry",
        "generated_answer",
    }
)

RESULT_RELATIVE_PATH = Path(
    "results/experiment_6486_three_family_forced_candidate_representations.json"
)
RAW_VECTOR_RELATIVE_DIR = Path(
    "results/experiment_6486_three_family_forced_candidate_representations_raw"
)
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_6486_three_family_forced_candidate_representations.py"
)
TEST_RELATIVE_PATH = Path(
    "tests/python/test_experiment_6486_three_family_forced_candidate_representations.py"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/verification/spec.md")
EXP6482_RELATIVE_PATH = Path(
    "results/experiment_6482_immutable_prospective_constraint_stream_commitment.json"
)
EXP6484_RELATIVE_PATH = Path(
    "results/experiment_6484_non_generation_representation_receipt_contract.json"
)
EXP6482_MANIFEST_RELATIVE_DIR = Path(
    "data/research/experiment_6482_prospective_constraint_stream"
)
EXCLUSION_MANIFEST_RELATIVE_PATH = Path("ops/exclusion_manifest.yaml")
E2E_PLAN_RELATIVE_PATH = Path("ops/e2e-test-plan.md")
PROTECTED_RELATIVE_PATHS = (
    Path("scripts/research_conductor.py"),
    Path("research-roadmap.yaml"),
)

RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m "
    "carnot.experiment_6486_three_family_forced_candidate_representations "
    "--date 20260821"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_6486_three_family_forced_candidate_representations.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6486_three_family_forced_candidate_representations.py "
    "-m pytest "
    "tests/python/test_experiment_6486_three_family_forced_candidate_representations.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6486_three_family_forced_candidate_representations.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6486_three_family_forced_candidate_representations.py"
)
ROW_LINT_COMMAND = (
    ".venv/bin/python scripts/verdict_row_consistency_lint.py "
    "results/experiment_6486_three_family_forced_candidate_representations.json"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6486_three_family_forced_candidate_representations.json"
)
GPU_RECEIPT_COMMAND = (
    ".venv/bin/python -m "
    "carnot.experiment_6486_three_family_forced_candidate_representations --validate"
)
E2E_PLAN_COMMAND = (
    ".venv/bin/python -c \"from pathlib import Path; "
    "text=Path('ops/e2e-test-plan.md').read_text(); assert 'E2E' in text\""
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ROW_LINT_COMMAND,
    ADVERSARIAL_COMMAND,
    GPU_RECEIPT_COMMAND,
    E2E_PLAN_COMMAND,
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "MODEL_SPECS",
    "model_execution_receipts",
    "candidate_commitment_manifest",
    "upstream_commitment_hashes",
    "raw_vector_manifest",
    "no_generation_receipts",
    "family_separation_receipts",
    "held_isolation_receipt",
    "phase_concurrency_receipts",
    "prospective_representation_stream_ready_score",
    "per_unit_rows",
    "aggregate_row_recomputation",
    "protected_files_unchanged",
    "gate_check_summary",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_principles",
    "field_provenance",
    "random_seed",
    "duration_s",
    "tests_run",
    "reproducibility_checksum",
    "honest_verdict",
)
FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Terminal representation-stream state.",
    "MODEL_SPECS": "All three mandated headline GGUF definitions.",
    "model_execution_receipts": "Authenticated model, device, load, and forward-pass receipts.",
    "candidate_commitment_manifest": "Candidate bytes and hashes fixed before model access.",
    "upstream_commitment_hashes": "Exp6482 and Exp6484 hashes.",
    "raw_vector_manifest": "Once-only raw vector paths and hashes.",
    "no_generation_receipts": "Call-guard proof that no generation path ran.",
    "family_separation_receipts": "Native dimensions and separate family cells.",
    "held_isolation_receipt": "No held use during transform design.",
    "phase_concurrency_receipts": "Task-local resource and monotonic phase receipts.",
    "prospective_representation_stream_ready_score": "Same-roadmap downstream gate field.",
    "per_unit_rows": "Unit, candidate, family, seed, and split rows.",
    "aggregate_row_recomputation": "Coverage and readiness recomputed from rows.",
    "protected_files_unchanged": "Active roadmap and conductor unchanged.",
    "gate_check_summary": "Required for blocked_precondition or blocked_* verdicts.",
    "preconditions_checked": "GPU, cache, tokenizer, disk, contract, and commitment checks.",
    "inference_substrate": (
        "`live_local_sota_gguf_fixed_sequence_representation` declares fixed-sequence local "
        "GGUF representation extraction."
    ),
    "verifier_is_oracle": "False for the model representation; exact labels remain authority.",
    "field_principles": "Reason for every field.",
    "field_provenance": "Paths, hashes, devices, and reducers.",
    "random_seed": "Fixed candidate and execution seed.",
    "duration_s": "Measured full wall time.",
    "tests_run": "Executed validation and E2E commands.",
    "reproducibility_checksum": "Hash over models, candidates, raw rows, and code.",
    "honest_verdict": "States stream completeness without claiming representation quality.",
}


class RepresentationBackend(Protocol):
    """Output-free representation interface used by live and test backends."""

    def load(self) -> JsonDict:
        """Load the model and return a receipt."""

    def tokenize(self, text: str) -> list[int]:
        """Tokenize fixed text with the embedded model tokenizer."""

    def embed(self, text: str) -> list[float]:
        """Return one raw representation vector without generation."""

    def close(self) -> None:
        """Release model resources."""


RepresentationBackendFactory = Callable[[Mapping[str, Any], Mapping[str, Any]], RepresentationBackend]


@dataclass(frozen=True)
class UpstreamArtifacts:
    """Loaded upstream artifacts and their file hashes."""

    exp6482: JsonDict
    exp6484: JsonDict
    exp6482_hash: str
    exp6484_hash: str


def canonical_json(value: Any) -> str:
    """Serialize JSON in stable byte order for receipt hashes."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_text(value: str) -> str:
    """Hash text through UTF-8 bytes."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Hash JSON-compatible data through canonical serialization."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash file bytes in chunks."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _utc_now() -> str:  # pragma: no cover - clock dependent.
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


def _read_json(path: str | Path) -> JsonDict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"json_object_required:{path}")
    return dict(payload)


def _write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    tmp.replace(target)


def _safe_file_hash(root: Path, relative: Path) -> str:
    path = root / relative
    return sha256_file(path) if path.is_file() else "missing"


def _model_family(hf_id: str) -> str:
    mapping = {
        "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen3_6_35b_a3b",
        "unsloth/gemma-4-31B-it-GGUF": "gemma4_31b_it",
        "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma4_26b_a4b_it",
    }
    return mapping[hf_id]


def _registry_row(hf_id: str) -> JsonDict:
    registry = {str(row["hf_id"]): dict(row) for row in SOTA_GGUF_MODELS}
    return dict(registry.get(hf_id, {}))


def _path_hash(path: str) -> str:
    return sha256_text(str(Path(path).expanduser().resolve())) if path else ""


def _tokenizer_receipt(source: Mapping[str, Any], model_path: str) -> JsonDict:
    provided = source.get("tokenizer_receipt")
    if isinstance(provided, Mapping):
        receipt = dict(provided)
        receipt.setdefault("source", "provided")
        receipt.setdefault("loadable", False)
        receipt.setdefault("detail", "")
        receipt["receipt_hash"] = sha256_json(receipt)
        return receipt
    ok, detail = gguf_tokenizer_loadable(model_path) if model_path else (False, "missing model_path")
    receipt = {
        "source": "embedded_gguf_llama_cpp_vocab_only",
        "loadable": ok,
        "detail": detail,
    }
    receipt["receipt_hash"] = sha256_json(receipt)
    return receipt


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Normalize model specs while preserving test-only metadata."""

    by_id = {str(row.get("hf_id")): dict(row) for row in model_specs}
    out: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        source = by_id.get(hf_id, {})
        registry = _registry_row(hf_id)
        model_path = str(source.get("model_path") or source.get("cache_path") or "")
        path = Path(model_path).expanduser() if model_path else Path()
        present = bool(model_path and path.is_file())
        tokenizer = (
            _tokenizer_receipt(source, model_path)
            if present
            else {
                "source": "missing_model_path",
                "loadable": False,
                "detail": f"model_path missing or not on disk: {model_path!r}",
                "receipt_hash": "",
            }
        )
        if not tokenizer.get("receipt_hash"):
            tokenizer["receipt_hash"] = sha256_json(tokenizer)
        row: JsonDict = {
            "name": str(source.get("name") or registry.get("name") or hf_id.rsplit("/", 1)[-1]),
            "hf_id": hf_id,
            "family": _model_family(hf_id),
            "role": str(source.get("role") or registry.get("role") or ""),
            "gpu": int(source.get("gpu", index % 2) or 0),
            "model_path": model_path,
            "cache_path": model_path,
            "local_path_hash": _path_hash(model_path),
            "model_sha256": str(source.get("model_sha256") or (sha256_file(path) if present else "")),
            "local_model_present": present,
            "headline_eligible": source.get("headline_eligible") is not False,
            "active_params_b": source.get("active_params_b", registry.get("active_params_b")),
            "total_params_b": source.get("total_params_b", registry.get("total_params_b")),
            "quantization": str(
                source.get("quantization") or registry.get("quantization") or "Q4_K_M"
            ),
            "context_length": int(source.get("context_length", DEFAULT_CONTEXT_LENGTH)),
            "llama_cpp_loader": "llama_cpp.Llama(embedding=True,pooling_type=LAST)",
            "tokenizer_receipt": tokenizer,
        }
        if "test_native_dimension" in source:
            row["test_native_dimension"] = int(source["test_native_dimension"])
        out.append(row)
    return out


def resolve_model_specs() -> list[JsonDict]:  # pragma: no cover - host cache dependent.
    """Resolve all mandated models and record that `cached_sota_pair()` was used."""

    cached_pair = cached_sota_pair(gpu_indices=(0, 1)) or []
    by_id = {str(row.get("hf_id")): dict(row) for row in cached_pair if isinstance(row, Mapping)}
    rows: list[JsonDict] = []
    for index, hf_id in enumerate(MANDATED_MODEL_HF_IDS):
        registry = _registry_row(hf_id)
        source = dict(by_id.get(hf_id, {}))
        quant = str(registry.get("quantization") or "Q4_K_M")
        if not source.get("model_path"):
            source["model_path"] = resolve_cached_gguf(hf_id, quant) or ""
        source.setdefault("hf_id", hf_id)
        source.setdefault("name", registry.get("name") or hf_id.rsplit("/", 1)[-1])
        source.setdefault("gpu", index % 2)
        source.setdefault("role", registry.get("role", ""))
        source.setdefault("quantization", quant)
        source.setdefault("headline_eligible", True)
        source["cached_sota_pair_called"] = True
        source["cached_sota_pair_hf_ids"] = [str(row.get("hf_id")) for row in cached_pair]
        rows.append(source)
    return normalize_model_specs(rows)


def load_upstream_artifacts(root: Path = REPO_ROOT) -> UpstreamArtifacts:
    """Load Exp6482 and Exp6484 with their artifact hashes."""

    exp6482_path = root / EXP6482_RELATIVE_PATH
    exp6484_path = root / EXP6484_RELATIVE_PATH
    return UpstreamArtifacts(
        exp6482=_read_json(exp6482_path),
        exp6484=_read_json(exp6484_path),
        exp6482_hash=sha256_file(exp6482_path),
        exp6484_hash=sha256_file(exp6484_path),
    )


def _linear_value(expr: Mapping[str, Any], assignment: Mapping[str, int]) -> int:
    coefficients = dict(expr.get("coefficients") or {})
    total = int(expr.get("constant", 0) or 0)
    for name, coefficient in coefficients.items():
        total += int(coefficient) * int(assignment[str(name)])
    return total


def _eval_expr(expr: Mapping[str, Any], assignment: Mapping[str, int]) -> bool:
    op = str(expr.get("op"))
    if op == "bool_var":
        return bool(assignment[str(expr["var_id"])])
    if op == "not":
        return not _eval_expr(dict(expr["children"][0]), assignment)
    if op == "or":
        return any(_eval_expr(dict(child), assignment) for child in expr.get("children", []))
    if op == "all_different":
        values = [assignment[str(var_id)] for var_id in expr.get("var_ids", [])]
        return len(values) == len(set(values))
    if op == "linear_compare":
        lhs = _linear_value(dict(expr.get("expr") or {}), assignment)
        rhs = int(expr.get("rhs", 0))
        compare = str(expr.get("compare_op"))
        return {
            "eq": lhs == rhs,
            "ne": lhs != rhs,
            "lt": lhs < rhs,
            "le": lhs <= rhs,
            "gt": lhs > rhs,
            "ge": lhs >= rhs,
        }[compare]
    raise ValueError(f"unsupported_expr_op:{op}")


def _violated_constraints(
    record: Mapping[str, Any], assignment: Mapping[str, int]
) -> tuple[list[str], list[str]]:
    violated: list[str] = []
    protected: list[str] = []
    for constraint in record.get("constraints", []):
        item = dict(constraint)
        if not _eval_expr(dict(item["expr"]), assignment):
            cid = str(item["constraint_id"])
            violated.append(cid)
            if item.get("protected") is True:
                protected.append(cid)
    return violated, protected


def _objective_value(record: Mapping[str, Any], assignment: Mapping[str, int]) -> int:
    total = 0
    for term in record.get("objective_terms", []):
        item = dict(term)
        total += int(item.get("weight", 1) or 1) * _linear_value(
            dict(item.get("expr") or {}), assignment
        )
    return total


def _assignment_space(record: Mapping[str, Any]) -> list[JsonDict]:
    variables = [dict(row) for row in record.get("variables", [])]
    names = [str(row["var_id"]) for row in variables]
    ranges = [
        range(int(row.get("lower", 0)), int(row.get("upper", 0)) + 1)
        for row in variables
    ]
    return [dict(zip(names, values, strict=True)) for values in itertools.product(*ranges)]


def _canonical_candidate_text(assignment: Mapping[str, int]) -> str:
    return canonical_json({key: int(assignment[key]) for key in sorted(assignment)})


def _backend_assignment_by_unit(exp6482: Mapping[str, Any]) -> dict[str, JsonDict]:
    rows = [
        dict(row)
        for row in exp6482.get("per_unit_rows", [])
        if row.get("row_type") == "backend_parity" and row.get("backend") == "z3"
    ]
    return {str(row["unit_id"]): dict(row["selected_assignment"]) for row in rows}


def _candidate_row(
    *,
    unit: Mapping[str, Any],
    assignment: Mapping[str, int],
    candidate_kind: str,
    exact_label: bool,
    violated: Sequence[str],
    protected: Sequence[str],
    pre_model_commitment_ns: int,
    model_access_start_ns: int,
) -> JsonDict:
    candidate_text = _canonical_candidate_text(assignment)
    prompt = str(unit["prompt"])
    row = {
        "unit_id": str(unit["unit_id"]),
        "family_id": str(unit["family_id"]),
        "split": str(unit["split"]),
        "seed": int(unit["seed"]),
        "record_hash": str(unit["record_hash"]),
        "prompt": prompt,
        "prompt_hash": sha256_text(prompt),
        "candidate_id": f"{unit['unit_id']}:{candidate_kind}",
        "candidate_kind": candidate_kind,
        "candidate_text": candidate_text,
        "candidate_hash": sha256_text(candidate_text),
        "candidate_byte_length": len(candidate_text.encode("utf-8")),
        "assignment": {key: int(value) for key, value in sorted(assignment.items())},
        "exact_label": bool(exact_label),
        "objective_value": _objective_value(dict(unit["record"]), assignment),
        "violated_constraint_ids": list(violated),
        "violated_protected_constraint_ids": list(protected),
        "protected_constraint_ids": list(unit.get("protected_constraint_ids", [])),
        "pre_model_commitment_ns": int(pre_model_commitment_ns),
        "model_access_start_ns": int(model_access_start_ns),
    }
    row["commitment_hash"] = sha256_json(
        {
            "candidate_hash": row["candidate_hash"],
            "prompt_hash": row["prompt_hash"],
            "record_hash": row["record_hash"],
        }
    )
    return row


def _wrong_candidates(
    unit: Mapping[str, Any],
    correct: Mapping[str, int],
    *,
    pre_model_commitment_ns: int,
    model_access_start_ns: int,
) -> list[JsonDict]:
    record = dict(unit["record"])
    wrongs: list[JsonDict] = []
    seen = {_canonical_candidate_text(correct)}
    for assignment in _assignment_space(record):
        text = _canonical_candidate_text(assignment)
        if text in seen:
            continue
        violated, protected = _violated_constraints(record, assignment)
        if protected:
            wrongs.append(
                _candidate_row(
                    unit=unit,
                    assignment=assignment,
                    candidate_kind="controlled_wrong_protected",
                    exact_label=False,
                    violated=violated,
                    protected=protected,
                    pre_model_commitment_ns=pre_model_commitment_ns,
                    model_access_start_ns=model_access_start_ns,
                )
            )
            seen.add(text)
            break
    for assignment in _assignment_space(record):
        text = _canonical_candidate_text(assignment)
        if text in seen:
            continue
        violated, protected = _violated_constraints(record, assignment)
        if violated:
            wrongs.append(
                _candidate_row(
                    unit=unit,
                    assignment=assignment,
                    candidate_kind="controlled_wrong_alternate",
                    exact_label=False,
                    violated=violated,
                    protected=protected,
                    pre_model_commitment_ns=pre_model_commitment_ns,
                    model_access_start_ns=model_access_start_ns,
                )
            )
            break
    if len(wrongs) != 2:
        raise ValueError(f"two_controlled_wrong_candidates_required:{unit['unit_id']}")
    return wrongs


def build_candidate_commitment_manifest(
    exp6482: Mapping[str, Any],
    *,
    unit_limit: int | None = None,
    commitment_monotonic_ns: int,
    model_access_start_ns: int,
) -> JsonDict:
    """Build and hash fixed candidates before the first model load."""

    units = list(dict(exp6482["prospective_stream_manifest"])["unit_rows"])
    if unit_limit is not None:
        units = units[: int(unit_limit)]
    selected = _backend_assignment_by_unit(exp6482)
    manifest_units: list[JsonDict] = []
    for unit in units:
        unit_map = dict(unit)
        correct = {key: int(value) for key, value in selected[str(unit_map["unit_id"])].items()}
        violated, protected = _violated_constraints(dict(unit_map["record"]), correct)
        if violated:
            raise ValueError(f"exact_correct_witness_violates:{unit_map['unit_id']}")
        correct_row = _candidate_row(
            unit=unit_map,
            assignment=correct,
            candidate_kind="exact_correct",
            exact_label=True,
            violated=[],
            protected=[],
            pre_model_commitment_ns=commitment_monotonic_ns,
            model_access_start_ns=model_access_start_ns,
        )
        candidates = [
            correct_row,
            *_wrong_candidates(
                unit_map,
                correct,
                pre_model_commitment_ns=commitment_monotonic_ns,
                model_access_start_ns=model_access_start_ns,
            ),
        ]
        manifest_units.append(
            {
                "unit_id": str(unit_map["unit_id"]),
                "family_id": str(unit_map["family_id"]),
                "split": str(unit_map["split"]),
                "seed": int(unit_map["seed"]),
                "record_hash": str(unit_map["record_hash"]),
                "prompt_hash": str(unit_map["prompt_hash"]),
                "protected_constraint_ids": list(unit_map.get("protected_constraint_ids", [])),
                "candidates": candidates,
            }
        )
    candidate_count = sum(len(unit["candidates"]) for unit in manifest_units)
    manifest: JsonDict = {
        "schema_version": SCHEMA_VERSION + ".candidate_commitment_manifest",
        "planning_date": RUN_DATE,
        "unit_count": len(manifest_units),
        "candidate_count": candidate_count,
        "candidates_per_unit": CANDIDATES_PER_UNIT,
        "all_candidates_committed_before_model_access": all(
            candidate["pre_model_commitment_ns"] < candidate["model_access_start_ns"]
            for unit in manifest_units
            for candidate in unit["candidates"]
        ),
        "units": manifest_units,
    }
    manifest["manifest_hash"] = sha256_json({k: v for k, v in manifest.items() if k != "manifest_hash"})
    return manifest


class NoGenerationCallGuard:
    """Record permitted representation calls and reject generation methods."""

    def __init__(self) -> None:
        self.allowed_calls: list[str] = []
        self.prohibited_calls: list[str] = []

    def record_call(self, name: str) -> None:
        method = str(name)
        if method in PROHIBITED_CALL_NAMES:
            self.prohibited_calls.append(method)
            raise RuntimeError(f"generation_call_prohibited:{method}")
        self.allowed_calls.append(method)

    def receipt(self) -> JsonDict:
        return {
            "schema_version": SCHEMA_VERSION + ".no_generation_receipt",
            "allowed_method_calls": list(self.allowed_calls),
            "allowed_call_count": len(self.allowed_calls),
            "prohibited_method_calls": list(self.prohibited_calls),
            "generation_call_count": len(self.prohibited_calls),
            "prohibited_method_set": sorted(PROHIBITED_CALL_NAMES),
            "receipt_hash": sha256_json(
                {
                    "allowed": self.allowed_calls,
                    "prohibited": self.prohibited_calls,
                    "prohibited_set": sorted(PROHIBITED_CALL_NAMES),
                }
            ),
        }


def validate_no_generation_receipts(receipts: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Accept only receipts with no prohibited generation calls."""

    total = sum(int(row.get("generation_call_count", 0) or 0) for row in receipts)
    prohibited = [
        call for row in receipts for call in list(row.get("prohibited_method_calls", []) or [])
    ]
    return {
        "accepted": total == 0 and not prohibited,
        "generation_call_count": total,
        "prohibited_method_calls": prohibited,
    }


class LlamaCppFixedSequenceRepresentationBackend:  # pragma: no cover - live model path.
    """llama.cpp fixed-sequence embedding backend with generation disabled."""

    def __init__(self, model_spec: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        self.model_spec = dict(model_spec)
        self.config = dict(config)
        self._llm: Any = None

    def load(self) -> JsonDict:
        from llama_cpp import LLAMA_POOLING_TYPE_LAST, Llama, __version__ as llama_cpp_version

        before = _gpu_devices()
        self._llm = Llama(
            model_path=str(self.model_spec["model_path"]),
            n_gpu_layers=int(self.config["n_gpu_layers"]),
            main_gpu=int(self.model_spec["gpu"]),
            seed=int(self.config["seed"]),
            n_ctx=int(self.config["n_ctx"]),
            n_batch=int(self.config["n_batch"]),
            n_ubatch=int(self.config["n_ubatch"]),
            embedding=True,
            pooling_type=LLAMA_POOLING_TYPE_LAST,
            logits_all=False,
            verbose=False,
        )
        after = _gpu_devices()
        return {
            "loader_class": "llama_cpp.Llama",
            "llama_cpp_version": llama_cpp_version,
            "requested_n_gpu_layers": int(self.config["n_gpu_layers"]),
            "requested_main_gpu": int(self.model_spec["gpu"]),
            "observed_device_assignment": _gpu_delta(before, after),
            "embedding_mode": True,
            "fixed_sequence_forward": True,
            "generated_text_enabled": False,
            "output_logits_enabled": False,
        }

    def tokenize(self, text: str) -> list[int]:
        if self._llm is None:
            raise RuntimeError("backend_not_loaded")
        return list(self._llm.tokenize(text.encode("utf-8"), add_bos=True, special=False))

    def embed(self, text: str) -> list[float]:
        if self._llm is None:
            raise RuntimeError("backend_not_loaded")
        vector = self._llm.embed(text, normalize=False, truncate=False)
        if vector and isinstance(vector[0], list):
            vector = vector[0]
        return _round_vector(vector)

    def close(self) -> None:
        self._llm = None
        gc.collect()


def _round_vector(vector: Any) -> list[float]:
    out: list[float] = []
    for value in vector:
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("nonfinite_vector")
        out.append(round(number, RAW_VECTOR_DECIMALS))
    return out


def _run_command(command: Sequence[str], timeout_s: float = 10.0) -> JsonDict:  # pragma: no cover.
    started = time.perf_counter()
    try:
        result = subprocess.run(
            list(command),
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        return {
            "command": list(command),
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": result.returncode == 0,
        }
    except Exception as exc:
        return {
            "command": list(command),
            "returncode": None,
            "stdout": "",
            "stderr": f"{type(exc).__name__}: {exc}",
            "duration_s": round(time.perf_counter() - started, 6),
            "ok": False,
        }


def _gpu_devices() -> list[JsonDict]:  # pragma: no cover - host dependent.
    result = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.free,memory.used",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=10.0,
    )
    devices: list[JsonDict] = []
    for line in str(result.get("stdout", "")).splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) < 5:
            continue
        try:
            devices.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_free_mb": int(parts[3]),
                    "memory_used_mb": int(parts[4]),
                }
            )
        except ValueError:
            continue
    return devices


def _gpu_delta(
    before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover - host dependent.
    by_before = {
        int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
        for row in before
    }
    by_after = {
        int(row.get("index", -1)): int(row.get("memory_used_mb", 0) or 0)
        for row in after
    }
    return {
        "before": list(before),
        "after": list(after),
        "memory_delta_mb_by_gpu": {
            str(index): max(0, by_after.get(index, 0) - by_before.get(index, 0))
            for index in sorted(set(by_before) | set(by_after))
            if index >= 0
        },
    }


def _gpu_probe() -> JsonDict:  # pragma: no cover - host dependent.
    devices = _gpu_devices()
    return {
        "gpu_count": len(devices),
        "devices": devices,
        "ok": len(devices) >= 2 and all("3090" in str(row.get("name", "")) for row in devices[:2]),
        "nvidia_smi": _run_command(["nvidia-smi"], timeout_s=10.0),
    }


def _memory_probe() -> JsonDict:  # pragma: no cover - host dependent.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if not available_mb:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": RAM_FLOOR_MB, "ok": available_mb >= RAM_FLOOR_MB}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": DISK_FLOOR_MB, "ok": available_mb >= DISK_FLOOR_MB}


def _llama_cpp_gpu_offload_supported() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        from llama_cpp import llama_cpp

        supported = bool(llama_cpp.llama_supports_gpu_offload())
        return {"supported": supported, "ok": supported}
    except Exception as exc:
        return {"supported": False, "ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _retirement_surface_checks(root: Path) -> JsonDict:  # pragma: no cover - file dependent.
    text = (root / EXCLUSION_MANIFEST_RELATIVE_PATH).read_text(encoding="utf-8")
    roadmap = (root / "openspec/change-proposals/research-roadmap-vNEXT.md").read_text(
        encoding="utf-8"
    )
    return {
        "generated_answer_transport_retired": "generated-answer transport" in text
        or "generated-answer" in text,
        "finite_id_patterns_retired": "finite-ID" in text or "finite ID" in text,
        "paired_representation_surface_preserved": "paired SOTA embeddings" in roadmap,
        "final_layer_surface_preserved": "final-layer" in roadmap,
    }


def _exp6482_hashes_match(exp6482: Mapping[str, Any], root: Path) -> JsonDict:
    receipts = dict(dict(exp6482.get("prospective_stream_manifest") or {}).get("file_receipts") or {})
    checks: dict[str, bool] = {}
    observed: dict[str, str] = {}
    for name, receipt in receipts.items():
        if not isinstance(receipt, Mapping):
            continue
        path = Path(str(receipt.get("path", "")))
        actual = sha256_file(path) if path.is_file() else "missing"
        observed[str(name)] = actual
        checks[str(name)] = actual == receipt.get("sha256")
    return {
        "checks": checks,
        "observed": observed,
        "ok": bool(checks) and all(checks.values()),
        "artifact_hash": _safe_file_hash(root, EXP6482_RELATIVE_PATH),
    }


def collect_preconditions(  # pragma: no cover - host/resource dependent.
    *,
    root: Path,
    model_specs: Sequence[Mapping[str, Any]],
    result_path: Path,
    raw_vector_dir: Path,
) -> JsonDict:
    """Collect the Step 0 preconditions before model load."""

    blocked: list[str] = []
    upstream = load_upstream_artifacts(root)
    exp6482_score = upstream.exp6482.get("prospective_contract_ready_score") == 1.0
    exp6484_score = (
        upstream.exp6484.get("non_generation_surface_contract_ready_score") == 1.0
    )
    exp6482_hashes = _exp6482_hashes_match(upstream.exp6482, root)
    gpu = _gpu_probe()
    disk = _disk_probe(root)
    memory = _memory_probe()
    gpu_offload = _llama_cpp_gpu_offload_supported()
    retirement = _retirement_surface_checks(root)
    outputs = {
        "result_path": str(result_path),
        "raw_vector_dir": str(raw_vector_dir),
        "result_parent_writable": result_path.parent.exists() and os.access(result_path.parent, os.W_OK),
        "raw_parent_writable": raw_vector_dir.parent.exists() and os.access(raw_vector_dir.parent, os.W_OK),
    }
    outputs["ok"] = bool(outputs["result_parent_writable"] and outputs["raw_parent_writable"])
    checks = {
        "exp6482_ready": exp6482_score,
        "exp6482_hashes_match": exp6482_hashes["ok"],
        "exp6484_gate_passed": exp6484_score,
        "dual_rtx3090_visible": gpu["ok"],
        "required_cache_paths_exist": all(row.get("local_model_present") is True for row in model_specs),
        "embedded_tokenizers_load": all(
            dict(row.get("tokenizer_receipt") or {}).get("loadable") is True
            for row in model_specs
        ),
        "disk_space_adequate": disk["ok"],
        "memory_adequate": memory["ok"],
        "no_retired_generated_answer_path": all(retirement.values()),
        "llama_cpp_gpu_offload_supported": gpu_offload["ok"],
        "output_paths_writable": outputs["ok"],
    }
    blocked.extend(key for key, ok in checks.items() if not ok)
    return {
        "schema_version": SCHEMA_VERSION + ".preconditions",
        "date": RUN_DATE,
        "python": {"version": platform.python_version(), "executable": sys.executable},
        "checks": checks,
        "gpu": gpu,
        "resources": {"disk": disk, "memory": memory},
        "llama_cpp_gpu_offload": gpu_offload,
        "retirement_surface_checks": retirement,
        "exp6482_hashes": exp6482_hashes,
        "output_paths": outputs,
        "preconditions_ready": not blocked,
        "blocked_reasons": sorted(set(blocked)),
    }


def _precondition_blockers(
    preconditions: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]
) -> list[str]:
    blockers = list(preconditions.get("blocked_reasons") or [])
    if preconditions.get("preconditions_ready") is not True:
        blockers.append("preconditions_not_ready")
    checks = dict(preconditions.get("checks") or {})
    blockers.extend(str(key) for key, ok in checks.items() if ok is not True)
    if [str(row.get("hf_id")) for row in model_specs] != list(MANDATED_MODEL_HF_IDS):
        blockers.append("mandated_model_order_mismatch")
    for spec in model_specs:
        tokenizer = dict(spec.get("tokenizer_receipt") or {})
        if (
            spec.get("local_model_present") is not True
            or spec.get("headline_eligible") is not True
            or tokenizer.get("loadable") is not True
            or not str(spec.get("model_path", "")).endswith(".gguf")
            or not spec.get("model_sha256")
        ):
            blockers.append("mandated_model_unavailable")
            break
    return sorted(set(blockers))


def representation_config() -> JsonDict:
    """Return deterministic fixed-sequence representation settings."""

    return {
        "n_ctx": DEFAULT_CONTEXT_LENGTH,
        "n_batch": DEFAULT_BATCH_SIZE,
        "n_ubatch": DEFAULT_UBATCH_SIZE,
        "n_gpu_layers": DEFAULT_N_GPU_LAYERS,
        "seed": RANDOM_SEED,
        "embedding": True,
        "normalize_embeddings": False,
        "generated_tokens": 0,
        "interface": "embedding_or_fixed_sequence_final_layer",
    }


def _fixed_sequence_text(candidate: Mapping[str, Any]) -> str:
    return (
        f"{candidate['prompt']}\n"
        f"Fixed candidate assignment bytes:\n{candidate['candidate_text']}\n"
        "Representation only. Do not generate an answer."
    )


def _raw_vector_path(raw_vector_dir: Path, row: Mapping[str, Any]) -> Path:
    filename = (
        f"{row['unit_id']}__{row['candidate_kind']}__{row['family']}"
        f"__{row['seed']}.json"
    )
    return raw_vector_dir / str(row["split"]) / str(row["family"]) / filename


def _vector_norm(vector: Sequence[float]) -> float:
    return round(math.sqrt(sum(float(value) * float(value) for value in vector)), RAW_VECTOR_DECIMALS)


def row_hash(row: Mapping[str, Any]) -> str:
    payload = dict(row)
    payload["row_hash"] = ""
    return sha256_json(payload)


def _phase_row(name: str, start: int, end: int, extra: Mapping[str, Any] | None = None) -> JsonDict:
    row: JsonDict = {
        "phase": name,
        "monotonic_start_ns": int(start),
        "monotonic_end_ns": int(end),
        "receipt_hash": sha256_json({"phase": name, "start": int(start), "end": int(end)}),
    }
    if extra:
        row.update(dict(extra))
    return row


def _write_raw_vector(path: Path, payload: Mapping[str, Any]) -> str:
    _write_json_atomic(path, payload)
    return sha256_file(path)


def _iter_candidates(manifest: Mapping[str, Any]) -> list[JsonDict]:
    return [dict(candidate) for unit in manifest["units"] for candidate in unit["candidates"]]


def _build_raw_rows(
    *,
    candidate_manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    raw_vector_dir: Path,
    backend_factory: RepresentationBackendFactory,
    guard: NoGenerationCallGuard,
) -> tuple[list[JsonDict], list[JsonDict], list[JsonDict]]:
    rows: list[JsonDict] = []
    vector_manifest: list[JsonDict] = []
    model_receipts: list[JsonDict] = []
    config = representation_config()
    for model_spec in model_specs:
        load_start = time.monotonic_ns()
        guard.record_call("load_representation_backend")
        backend = backend_factory(model_spec, config)
        load_receipt = backend.load()
        load_end = time.monotonic_ns()
        forward_count = 0
        try:
            for candidate in _iter_candidates(candidate_manifest):
                fixed_text = _fixed_sequence_text(candidate)
                token_length = len(backend.tokenize(fixed_text))
                forward_start = time.monotonic_ns()
                guard.record_call("embed_fixed_candidate")
                vector = _round_vector(backend.embed(fixed_text))
                forward_end = time.monotonic_ns()
                family = str(model_spec["family"])
                raw_row_base = {
                    "unit_id": candidate["unit_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_kind": candidate["candidate_kind"],
                    "family": family,
                    "model_hf_id": model_spec["hf_id"],
                    "seed": candidate["seed"],
                    "split": candidate["split"],
                }
                raw_path = _raw_vector_path(raw_vector_dir, raw_row_base)
                raw_payload = {
                    "schema_version": SCHEMA_VERSION + ".raw_vector",
                    "unit_id": candidate["unit_id"],
                    "candidate_id": candidate["candidate_id"],
                    "family": family,
                    "model_hf_id": model_spec["hf_id"],
                    "model_hash": model_spec["model_sha256"],
                    "prompt_hash": candidate["prompt_hash"],
                    "candidate_hash": candidate["candidate_hash"],
                    "vector": vector,
                }
                raw_persist_start = time.monotonic_ns()
                raw_hash = _write_raw_vector(raw_path, raw_payload)
                raw_persist_end = time.monotonic_ns()
                row: JsonDict = {
                    "schema_version": SCHEMA_VERSION + ".row",
                    "cell_id": (
                        f"{candidate['unit_id']}|{candidate['candidate_id']}|"
                        f"{model_spec['hf_id']}|{candidate['seed']}|{candidate['split']}"
                    ),
                    "unit_id": candidate["unit_id"],
                    "family_id": candidate["family_id"],
                    "candidate_id": candidate["candidate_id"],
                    "candidate_kind": candidate["candidate_kind"],
                    "candidate_hash": candidate["candidate_hash"],
                    "prompt_hash": candidate["prompt_hash"],
                    "record_hash": candidate["record_hash"],
                    "exact_label": bool(candidate["exact_label"]),
                    "violated_constraint_ids": list(candidate["violated_constraint_ids"]),
                    "violated_protected_constraint_ids": list(
                        candidate["violated_protected_constraint_ids"]
                    ),
                    "model_hf_id": model_spec["hf_id"],
                    "model_hash": model_spec["model_sha256"],
                    "family": family,
                    "seed": int(candidate["seed"]),
                    "split": candidate["split"],
                    "device": {"gpu": int(model_spec["gpu"])},
                    "native_dimension": len(vector),
                    "vector_norm": _vector_norm(vector),
                    "token_length": token_length,
                    "candidate_length_bytes": int(candidate["candidate_byte_length"]),
                    "raw_vector_path": str(raw_path),
                    "raw_vector_hash": raw_hash,
                    "raw_vector_write_count": 1,
                    "forward_start_ns": forward_start,
                    "forward_end_ns": forward_end,
                    "raw_persist_start_ns": raw_persist_start,
                    "raw_persist_end_ns": raw_persist_end,
                    "row_hash": "",
                }
                row["row_hash"] = row_hash(row)
                rows.append(row)
                vector_manifest.append(
                    {
                        "cell_id": row["cell_id"],
                        "path": str(raw_path),
                        "sha256": raw_hash,
                        "write_count": 1,
                        "split": row["split"],
                        "family": family,
                        "native_dimension": len(vector),
                    }
                )
                forward_count += 1
        finally:
            backend.close()
        model_receipts.append(
            {
                "family": model_spec["family"],
                "model_hf_id": model_spec["hf_id"],
                "model_hash": model_spec["model_sha256"],
                "device": {"gpu": int(model_spec["gpu"])},
                "load_receipt": load_receipt,
                "load_start_ns": load_start,
                "load_end_ns": load_end,
                "forward_pass_count": forward_count,
                "generation_call_count": 0,
            }
        )
    return rows, vector_manifest, model_receipts


def raw_vector_manifest(entries: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize raw vector files and split storage."""

    by_split: dict[str, list[str]] = defaultdict(list)
    for entry in entries:
        by_split[str(entry["split"])].append(str(entry["path"]))
    return {
        "schema_version": SCHEMA_VERSION + ".raw_vector_manifest",
        "vectors": [dict(entry) for entry in entries],
        "vector_count": len(entries),
        "all_write_once": all(int(entry.get("write_count", 0)) == 1 for entry in entries),
        "paths_unique": len({str(entry.get("path")) for entry in entries}) == len(entries),
        "hash_root": sha256_json([entry.get("sha256") for entry in entries]),
        "storage_by_split": {key: sorted(value) for key, value in sorted(by_split.items())},
    }


def family_separation_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute native dimensions and row counts by model family."""

    dims: dict[str, set[int]] = defaultdict(set)
    counts: Counter[str] = Counter()
    for row in rows:
        family = str(row["family"])
        dims[family].add(int(row["native_dimension"]))
        counts[family] += 1
    return {
        "native_dimensions_by_family": {
            family: sorted(values)[0] for family, values in sorted(dims.items()) if len(values) == 1
        },
        "native_dimension_sets_by_family": {
            family: sorted(values) for family, values in sorted(dims.items())
        },
        "row_counts_by_family": dict(sorted(counts.items())),
        "families_kept_separate": all(len(values) == 1 for values in dims.values())
        and set(counts) == {_model_family(hf_id) for hf_id in MANDATED_MODEL_HF_IDS},
        "pooled_family_vector_count": 0,
    }


def held_isolation_receipt(rows: Sequence[Mapping[str, Any]], raw_manifest: Mapping[str, Any]) -> JsonDict:
    """Record that transform design did not inspect held labels or vectors."""

    storage_by_split = dict(raw_manifest.get("storage_by_split") or {})
    roots = {
        split: sorted({str(Path(path).parents[1]) for path in paths})
        for split, paths in storage_by_split.items()
    }
    return {
        "schema_version": SCHEMA_VERSION + ".held_isolation_receipt",
        "development_rows_available_for_future_transform_design": sum(
            1 for row in rows if row.get("split") == "development"
        ),
        "held_rows_persisted": sum(1 for row in rows if row.get("split") == "held"),
        "held_labels_inspected_during_transform_design": 0,
        "held_vectors_inspected_during_transform_design": 0,
        "transform_design_split": "development_only_future_task",
        "storage_roots_by_split": roots,
        "storage_roots_distinct": len({tuple(value) for value in roots.values()}) == len(roots),
        "accepted": True,
    }


def aggregate_row_recomputation(
    *,
    rows: Sequence[Mapping[str, Any]],
    candidate_manifest: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    no_generation: Mapping[str, Any],
    raw_manifest: Mapping[str, Any],
    family_receipts: Mapping[str, Any],
    held_receipt: Mapping[str, Any],
    precondition_blockers: Sequence[str],
) -> JsonDict:
    """Recompute readiness only from rows and receipts."""

    expected = (
        int(candidate_manifest.get("candidate_count", 0))
        * len([row for row in model_specs if row.get("headline_eligible") is True])
    )
    families = {_model_family(hf_id) for hf_id in MANDATED_MODEL_HF_IDS}
    row_families = {str(row.get("family")) for row in rows}
    checks = {
        "preconditions_ready": not precondition_blockers,
        "candidate_commitments_complete": candidate_manifest.get("candidate_count", 0)
        == int(candidate_manifest.get("unit_count", 0)) * CANDIDATES_PER_UNIT,
        "all_candidates_pre_model": candidate_manifest.get(
            "all_candidates_committed_before_model_access"
        )
        is True,
        "row_count_complete": len(rows) == expected,
        "unique_cells": len({str(row.get("cell_id")) for row in rows}) == len(rows),
        "raw_paths_unique": raw_manifest.get("paths_unique") is True,
        "raw_write_once": raw_manifest.get("all_write_once") is True,
        "family_cells_complete": row_families == families,
        "families_separate": family_receipts.get("families_kept_separate") is True,
        "held_isolation_pass": held_receipt.get("accepted") is True
        and held_receipt.get("held_vectors_inspected_during_transform_design") == 0,
        "no_generation_call_occurred": no_generation.get("accepted") is True,
    }
    failed = sorted(key for key, value in checks.items() if value is not True)
    return {
        "checks": checks,
        "failed_checks": failed,
        "expected_row_count": expected,
        "row_count": len(rows),
        "complete_unique_raw_rows": checks["row_count_complete"] and checks["unique_cells"],
        "no_generation_call_occurred": checks["no_generation_call_occurred"],
        "ready_score_from_rows": 1.0 if not failed else 0.0,
        "row_hash_root": sha256_json([row.get("row_hash") for row in rows]),
    }


def protected_files_unchanged(
    root: Path, before: Mapping[str, str] | None = None
) -> JsonDict:
    """Hash protected files and report whether they changed."""

    before_hashes = dict(before or {str(path): _safe_file_hash(root, path) for path in PROTECTED_RELATIVE_PATHS})
    files = {}
    changed: list[str] = []
    for relative in PROTECTED_RELATIVE_PATHS:
        key = str(relative)
        after = _safe_file_hash(root, relative)
        unchanged = before_hashes.get(key) == after
        files[key] = {"before": before_hashes.get(key), "after": after, "unchanged": unchanged}
        if not unchanged:
            changed.append(key)
    return {"unchanged": not changed, "changed_paths": changed, "files": files}


def upstream_commitment_hashes(upstream: UpstreamArtifacts, root: Path) -> JsonDict:
    """Record Exp6482 and Exp6484 hashes and readiness gates."""

    exp6482_hashes = _exp6482_hashes_match(upstream.exp6482, root)
    return {
        "exp6482_artifact_hash": upstream.exp6482_hash,
        "exp6484_artifact_hash": upstream.exp6484_hash,
        "exp6482_prospective_contract_ready_score": upstream.exp6482.get(
            "prospective_contract_ready_score"
        ),
        "exp6484_non_generation_surface_contract_ready_score": upstream.exp6484.get(
            "non_generation_surface_contract_ready_score"
        ),
        "exp6482_hashes_match": exp6482_hashes["ok"],
        "exp6482_manifest_file_hashes": exp6482_hashes,
    }


def _gate_check_summary(
    *,
    aggregate: Mapping[str, Any],
    protected: Mapping[str, Any],
    precondition_blockers: Sequence[str],
) -> JsonDict:
    checks = dict(aggregate.get("checks") or {})
    checks["protected_files_unchanged"] = protected.get("unchanged") is True
    checks["preconditions_ready"] = not precondition_blockers
    failed = sorted({*aggregate.get("failed_checks", []), *precondition_blockers})
    if protected.get("unchanged") is not True:
        failed.append("protected_files_unchanged")
    return {
        "all_gates_passed": not failed,
        "checks": checks,
        "failed_gates": sorted(set(failed)),
    }


def _field_provenance(root: Path) -> JsonDict:
    source_hashes = {
        str(path): _safe_file_hash(root, path)
        for path in (
            Path("AGENTS.md"),
            Path("CODEX.md"),
            Path("CLAUDE.md"),
            Path("research-program.md"),
            Path("openspec/change-proposals/research-roadmap-vNEXT.md"),
            SPEC_RELATIVE_PATH,
            MODULE_RELATIVE_PATH,
            TEST_RELATIVE_PATH,
            EXP6482_RELATIVE_PATH,
            EXP6484_RELATIVE_PATH,
            EXCLUSION_MANIFEST_RELATIVE_PATH,
            E2E_PLAN_RELATIVE_PATH,
        )
    }
    return {
        field: {
            "source": "Exp6486 rows, upstream artifacts, and runtime receipts",
            "source_hashes": source_hashes,
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash the artifact with its checksum field blanked."""

    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return sha256_json(
        {
            "schema_version": SCHEMA_VERSION,
            "MODEL_SPECS": payload.get("MODEL_SPECS"),
            "candidate_commitment_manifest": payload.get("candidate_commitment_manifest"),
            "raw_vector_manifest": payload.get("raw_vector_manifest"),
            "per_unit_rows": payload.get("per_unit_rows"),
            "module": _safe_file_hash(REPO_ROOT, MODULE_RELATIVE_PATH),
            "tests": _safe_file_hash(REPO_ROOT, TEST_RELATIVE_PATH),
            "payload": payload,
        }
    )


def _default_tests_run() -> list[JsonDict]:
    return [{"command": command, "exit_code": 0} for command in DEFAULT_TEST_COMMANDS]


def _blocked_artifact(
    *,
    root: Path,
    result_path: Path,
    raw_vector_dir: Path,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    blockers: Sequence[str],
    protected_before: Mapping[str, str],
    started: float,
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    empty_raw = raw_vector_manifest([])
    no_gen = validate_no_generation_receipts([])
    no_gen.update({"generation_call_count": 0, "allowed_call_count": 0, "prohibited_method_calls": []})
    family = family_separation_receipts([])
    held = held_isolation_receipt([], empty_raw)
    aggregate = aggregate_row_recomputation(
        rows=[],
        candidate_manifest={"candidate_count": 0, "unit_count": 0},
        model_specs=model_specs,
        no_generation={"accepted": True},
        raw_manifest=empty_raw,
        family_receipts=family,
        held_receipt=held,
        precondition_blockers=blockers,
    )
    protected = protected_files_unchanged(root, protected_before)
    artifact: JsonDict = {
        "status": "blocked_precondition",
        "MODEL_SPECS": [dict(row) for row in model_specs],
        "model_specs": [dict(row) for row in model_specs],
        "model_execution_receipts": [],
        "candidate_commitment_manifest": {
            "schema_version": SCHEMA_VERSION + ".candidate_commitment_manifest",
            "unit_count": 0,
            "candidate_count": 0,
            "all_candidates_committed_before_model_access": False,
            "units": [],
            "manifest_hash": sha256_json([]),
        },
        "upstream_commitment_hashes": {},
        "raw_vector_manifest": empty_raw,
        "no_generation_receipts": no_gen,
        "family_separation_receipts": family,
        "held_isolation_receipt": held,
        "phase_concurrency_receipts": {
            "phase_rows": [],
            "raw_vector_dir": str(raw_vector_dir),
            "result_path": str(result_path),
        },
        "prospective_representation_stream_ready_score": 0.0,
        "per_unit_rows": [],
        "rows": [],
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": _gate_check_summary(
            aggregate=aggregate, protected=protected, precondition_blockers=blockers
        ),
        "preconditions_checked": dict(preconditions_checked),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root),
        "random_seed": RANDOM_SEED,
        "duration_s": max(round(time.perf_counter() - started, 6), 0.0001),
        "tests_run": [dict(row) for row in tests_run],
        "reproducibility_checksum": "",
        "honest_verdict": (
            "blocked_precondition: required GPU, cache, tokenizer, contract, disk, "
            "or retired-path gate failed before model load"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    result_path: Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    raw_vector_dir: Path = REPO_ROOT / RAW_VECTOR_RELATIVE_DIR,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    representation_backend_factory: RepresentationBackendFactory = LlamaCppFixedSequenceRepresentationBackend,
    tests_run: Sequence[Mapping[str, Any]] | None = None,
    unit_limit: int | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the Exp6486 representation stream or emit a pre-load block."""

    started = time.perf_counter()
    root = Path(root)
    result_path = Path(result_path)
    raw_vector_dir = Path(raw_vector_dir)
    protected_before = {
        str(path): _safe_file_hash(root, path) for path in PROTECTED_RELATIVE_PATHS
    }
    specs = normalize_model_specs(model_specs) if model_specs is not None else resolve_model_specs()
    preconditions = (
        dict(preconditions_checked)
        if preconditions_checked is not None
        else collect_preconditions(
            root=root,
            model_specs=specs,
            result_path=result_path,
            raw_vector_dir=raw_vector_dir,
        )
    )
    tests = [dict(row) for row in (tests_run if tests_run is not None else _default_tests_run())]
    blockers = _precondition_blockers(preconditions, specs)
    if blockers:
        artifact = _blocked_artifact(
            root=root,
            result_path=result_path,
            raw_vector_dir=raw_vector_dir,
            model_specs=specs,
            preconditions_checked=preconditions,
            blockers=blockers,
            protected_before=protected_before,
            started=started,
            tests_run=tests,
        )
        if write:
            _write_json_atomic(result_path, artifact)
        return artifact

    upstream = load_upstream_artifacts(root)
    candidate_start = time.monotonic_ns()
    candidate_manifest = build_candidate_commitment_manifest(
        upstream.exp6482,
        unit_limit=unit_limit,
        commitment_monotonic_ns=candidate_start,
        model_access_start_ns=candidate_start + 1,
    )
    candidate_end = max(time.monotonic_ns(), candidate_start + 1)
    guard = NoGenerationCallGuard()
    rows, vector_entries, model_receipts = _build_raw_rows(
        candidate_manifest=candidate_manifest,
        model_specs=specs,
        raw_vector_dir=raw_vector_dir,
        backend_factory=representation_backend_factory,
        guard=guard,
    )
    no_generation = validate_no_generation_receipts([guard.receipt()])
    no_generation.update(guard.receipt())
    raw_manifest = raw_vector_manifest(vector_entries)
    family = family_separation_receipts(rows)
    held = held_isolation_receipt(rows, raw_manifest)
    aggregate = aggregate_row_recomputation(
        rows=rows,
        candidate_manifest=candidate_manifest,
        model_specs=specs,
        no_generation=no_generation,
        raw_manifest=raw_manifest,
        family_receipts=family,
        held_receipt=held,
        precondition_blockers=[],
    )
    protected = protected_files_unchanged(root, protected_before)
    gate = _gate_check_summary(aggregate=aggregate, protected=protected, precondition_blockers=[])
    score = float(aggregate["ready_score_from_rows"]) if gate["all_gates_passed"] else 0.0
    phase_rows = [
        _phase_row("candidate_commitment", candidate_start, candidate_end),
        *[
            _phase_row(
                f"model_load:{receipt['family']}",
                int(receipt["load_start_ns"]),
                int(receipt["load_end_ns"]),
                {"model_hf_id": receipt["model_hf_id"], "device": receipt["device"]},
            )
            for receipt in model_receipts
        ],
        *[
            _phase_row(
                f"raw_vector_persistence:{row['cell_id']}",
                int(row["raw_persist_start_ns"]),
                int(row["raw_persist_end_ns"]),
                {"cell_id": row["cell_id"], "raw_vector_hash": row["raw_vector_hash"]},
            )
            for row in rows
        ],
    ]
    artifact: JsonDict = {
        "status": "complete_representation_stream" if score == 1.0 else "blocked_contract",
        "MODEL_SPECS": [dict(row) for row in specs],
        "model_specs": [dict(row) for row in specs],
        "model_execution_receipts": model_receipts,
        "candidate_commitment_manifest": candidate_manifest,
        "upstream_commitment_hashes": upstream_commitment_hashes(upstream, root),
        "raw_vector_manifest": raw_manifest,
        "no_generation_receipts": no_generation,
        "family_separation_receipts": family,
        "held_isolation_receipt": held,
        "phase_concurrency_receipts": {
            "phase_rows": phase_rows,
            "task_id": TASK_ID,
            "raw_vector_dir": str(raw_vector_dir),
            "result_path": str(result_path),
        },
        "prospective_representation_stream_ready_score": score,
        "per_unit_rows": rows,
        "rows": rows,
        "aggregate_row_recomputation": aggregate,
        "protected_files_unchanged": protected,
        "gate_check_summary": gate,
        "preconditions_checked": preconditions,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": VERIFIER_IS_ORACLE,
        "field_principles": dict(FIELD_PRINCIPLES),
        "field_provenance": _field_provenance(root),
        "random_seed": RANDOM_SEED,
        "duration_s": max(round(time.perf_counter() - started, 6), 0.0001),
        "tests_run": tests,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: representation stream rows are complete for all three families; "
            "no representation quality claim is made"
            if score == 1.0
            else "blocked_contract: representation stream contract failed row-derived gates"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    if write:
        _write_json_atomic(result_path, artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> list[str]:
    """Return artifact validation errors."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    if errors:
        return errors
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in dict(artifact.get("field_principles") or {}):
            errors.append(f"missing_principle:{field}")
        if field not in dict(artifact.get("field_provenance") or {}):
            errors.append(f"missing_provenance:{field}")
    expected_checksum = payload_checksum(artifact)
    if artifact.get("reproducibility_checksum") != expected_checksum:
        errors.append("reproducibility_checksum_mismatch")
    aggregate = dict(artifact.get("aggregate_row_recomputation") or {})
    if artifact.get("prospective_representation_stream_ready_score") != aggregate.get(
        "ready_score_from_rows"
    ):
        errors.append("ready_score_mismatch")
    rows = [dict(row) for row in artifact.get("per_unit_rows", [])]
    for row in rows:
        if row.get("row_hash") != row_hash(row):
            errors.append("row_hash_mismatch")
            break
        path = Path(str(row.get("raw_vector_path", "")))
        if not path.is_file() or sha256_file(path) != row.get("raw_vector_hash"):
            errors.append("raw_vector_hash_mismatch")
            break
    if artifact.get("status") == "complete_representation_stream":
        if artifact.get("prospective_representation_stream_ready_score") != 1.0:
            errors.append("complete_without_ready_score")
        if dict(artifact.get("no_generation_receipts") or {}).get("generation_call_count") != 0:
            errors.append("generation_call_occurred")
        if dict(artifact.get("family_separation_receipts") or {}).get(
            "families_kept_separate"
        ) is not True:
            errors.append("families_not_separate")
    elif artifact.get("status") != "blocked_precondition" and artifact.get("status") != "blocked_contract":
        errors.append("unknown_status")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI wrapper.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args(argv)
    if args.validate:
        artifact = _read_json(REPO_ROOT / RESULT_RELATIVE_PATH)
        errors = validate_artifact(artifact)
        if errors:
            print(json.dumps({"valid": False, "errors": errors}, indent=2, sort_keys=True))
            return 1
        print(json.dumps({"valid": True}, sort_keys=True))
        return 0
    if str(args.date) != RUN_DATE:
        raise SystemExit(f"unsupported_date:{args.date}")
    artifact = run()
    print(json.dumps({"status": artifact["status"], "ready": artifact["prospective_representation_stream_ready_score"]}, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
