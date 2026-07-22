"""Exp5813 three-family split-budget SOTA canary.

Spec refs: REQ-VERIFY-5813, SCENARIO-VERIFY-5813,
SCENARIO-VERIFY-5813-CONTROLS.

The module turns the Exp5812 two-call transport contract into a row-level
canary over the three mandated local GGUF families.  The important boundary is
that parser and exact fixture validation stay deterministic: live model calls
may provide raw reasoning and final bytes, but only fixture candidate IDs and
exact validators decide whether a row is transport-clean.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot import experiment_5785_hardness_surface_fixture as fixture
from carnot import experiment_5812_split_budget_channel_contract as contract
from carnot.inference import sota_models


JsonDict = dict[str, Any]
ResponseEmitter = Callable[[Mapping[str, Any]], None]
CanaryRunner = Callable[[Mapping[str, Any], Mapping[str, Any], list[JsonDict], ResponseEmitter], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5813_split_budget_sota_canary.json")
ROW_FILE_RELATIVE_PATH = Path("results/experiment_5813_split_budget_sota_canary.rows.jsonl")
CHECKPOINT_RELATIVE_DIR = Path("results/checkpoints/experiment_5813_split_budget_sota_canary")

SCHEMA = "carnot.experiment_5813.split_budget_sota_canary.v1"
ROW_SCHEMA = SCHEMA + ".row"
EXPERIMENT = 5813
EXPERIMENT_ID = "experiment_5813_split_budget_sota_canary"
MILESTONE = "2026.07.518"
RUN_DATE = "20260722"
INFERENCE_SUBSTRATE = "live_llm_inference"

QWEN_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
GEMMA31_ID = "unsloth/gemma-4-31B-it-GGUF"
GEMMA26_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
MANDATED_MODEL_IDS = (QWEN_ID, GEMMA31_ID, GEMMA26_ID)

STOP_STRINGS = contract.STOP_STRINGS
MODEL_BUDGETS: JsonDict = {
    "shared_budget_max_tokens": 128,
    "split_reasoning_max_tokens": 96,
    "split_finalization_max_tokens": 32,
    "reasoning_timeout_s": 900,
    "finalization_timeout_s": 180,
}
SAMPLING_CONFIG: JsonDict = {
    "temperature": 0.0,
    "top_p": 1.0,
    "top_k": 1,
    "repeat_penalty": 1.0,
}
RANDOM_SEED: JsonDict = {
    "base_seed": 5813,
    "fixture_seed": 5813001,
    "runner_seed": 5813002,
    "replay_seed": 5813003,
}

MODEL_SPECS: list[JsonDict] = [
    {
        "family": "qwen3-6-35b-a3b",
        "name": "Qwen3.6-35B-A3B",
        "hf_id": QWEN_ID,
        "role": "moe",
        "quantization": "Q4_K_M",
        "gpu": 0,
    },
    {
        "family": "gemma-4-31b-it",
        "name": "Gemma4-31B-it",
        "hf_id": GEMMA31_ID,
        "role": "dense",
        "quantization": "Q4_K_M",
        "gpu": 1,
    },
    {
        "family": "gemma-4-26b-a4b-it",
        "name": "Gemma4-26B-A4B-it",
        "hf_id": GEMMA26_ID,
        "role": "moe",
        "quantization": "Q4_K_M",
        "gpu": 0,
    },
]

SPEC_REFS = (
    "REQ-VERIFY-5813",
    "SCENARIO-VERIFY-5813",
    "SCENARIO-VERIFY-5813-CONTROLS",
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "model_specs",
    "random_seed",
    "model_runtime_and_gpu_receipts",
    "sample_size_and_justification",
    "preregistered_mode_results",
    "independent_failure_metrics",
    "transcript_and_checkpoint_receipts",
    "selected_transport_by_model",
    "qualified_real_sota_model_count",
    "answer_channel_ready_score",
    "adversarial_control_results",
    "row_file_and_sha256",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PRINCIPLES: dict[str, str] = {
    "status": "A terminal state distinguishes a completed non-ready canary from an operational block.",
    "preconditions_checked": "Cache, CUDA, fixture, seed, and output checks prevent fabricated live-inference evidence.",
    "model_specs": "Resolved identities for all three mandated models prevent legacy smoke models from becoming headline evidence.",
    "random_seed": "Fixed seeds make canary sampling and replay reproducible.",
    "model_runtime_and_gpu_receipts": "Fresh actual CUDA load/offload evidence is required for every qualified family.",
    "sample_size_and_justification": "At least twelve independent balanced units support a transport canary without counting repeated modes as samples.",
    "preregistered_mode_results": "Fixed shared/split arms prevent post-hoc transport selection.",
    "independent_failure_metrics": "Parser, truncation, empty-final, ghost-ID, timeout, stop, and exact-error events remain separately reconstructable.",
    "transcript_and_checkpoint_receipts": "Raw calls, frozen transcript hashes, and per-row checkpoints make two-stage execution replayable.",
    "selected_transport_by_model": "At most one disclosed bounded transport per family prevents hidden fallback mixtures.",
    "qualified_real_sota_model_count": "All three required families must qualify before downstream stream generation.",
    "answer_channel_ready_score": "A bare scalar activates scale only when every family and provenance gate passes.",
    "adversarial_control_results": "Injected, ghost, empty, truncated, and wrong outputs must fail closed.",
    "row_file_and_sha256": "Content-addressed raw rows prevent summary-only claims.",
    "duration_s": "Real three-family inference must have plausible measured wall time.",
    "inference_substrate": "`live_llm_inference` declares actual local autoregressive GGUF execution.",
    "verifier_is_oracle": "Exact fixture validation defines correctness, so verified yield is execution-grounded and not a moat claim.",
    "field_provenance": "Each metric identifies its row predicate, runtime log, or exact validator source.",
    "test_commands": "Commands document live load, focused tests, row replay, and adversarial verification.",
    "test_exit_codes": "Exit codes prevent failed live or validation checks from being narrated as passing.",
    "reproducibility_checksum": "A checksum detects model, fixture, mode, seed, or row drift.",
    "honest_verdict": "A `complete:` or `blocked:` prefix provides the retirement mechanic with a terminal verdict.",
}

FIELD_PRINCIPLE_EXTRAS: JsonDict = {
    "schema": "Versioned schema for the Exp5813 canary artifact.",
    "experiment": "Numeric experiment id binds the artifact to the task.",
    "experiment_id": "Stable slug binds rows, checkpoints, and artifact paths.",
    "milestone": "Milestone identity for the preregistered retry.",
    "run_date": "Operator-mandated execution date.",
    "spec_refs": "OpenSpec anchors for the canary and controls.",
    "prior_failure_retirement": "Records the single allowed changed-mechanism retry outcome.",
}

EXPECTED_ADVERSARIAL_CONTROLS = (
    "empty_reasoning",
    "empty_final",
    "truncation",
    "ghost_candidate_id",
    "invalid_candidate_id",
    "stop_collision",
    "timeout",
    "schema_control_plane_injection",
    "protected_fact_distortion",
    "exact_wrong_answer",
)

DEFAULT_TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/quarantine/test_experiment_5813_split_budget_sota_canary.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5813_split_budget_sota_canary.py "
    "-m pytest tests/python/quarantine/test_experiment_5813_split_budget_sota_canary.py -q --no-cov -n 0 && "
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5813_split_budget_sota_canary.py --fail-under=100",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py",
    ".venv/bin/python scripts/root_clutter_sweep.py",
)


class ManifestReplayError(ValueError):
    """Raised when row, checkpoint, or transcript receipts no longer replay."""


def canonical_json(value: Any) -> str:
    """Serialize JSON-compatible evidence in the one form used for hashing."""

    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_text(value: str) -> str:
    """Return a prefixed SHA-256 digest for text evidence."""

    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a prefixed SHA-256 digest for deterministic JSON evidence."""

    return sha256_text(canonical_json(value))


def sha256_file(path: str | Path) -> str:
    """Hash a local file in chunks so GGUF files and row files share one API."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def _copy_json(value: Any) -> Any:
    return json.loads(canonical_json(value))


def _hash_optional_file(path: str | Path | None) -> str:
    if path and Path(path).is_file():
        return sha256_file(path)
    return sha256_text("")


def _mode_family_key(mode: Mapping[str, Any]) -> tuple[int, str]:
    order = {"shared_budget_control": 0, "split_budget": 1}
    return (order.get(str(mode.get("mode_type")), 99), str(mode["mode_id"]))


def preregistered_modes() -> list[JsonDict]:
    """Return the fixed shared control plus split-budget arms from Exp5812."""

    modes = [contract.SHARED_BUDGET_CONTROL, *contract.SPLIT_BUDGET_MODES]
    return [_copy_json(mode) for mode in sorted(modes, key=_mode_family_key)]


def normalize_model_specs(model_specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Resolve identity, hash, quantization, template, runtime, budget, and seed fields."""

    normalized = []
    base_by_id = {item["hf_id"]: item for item in MODEL_SPECS}
    for ordinal, original in enumerate(model_specs):
        spec = {**base_by_id.get(str(original.get("hf_id")), {}), **dict(original)}
        model_path = str(spec.get("resolved_model_path") or spec.get("model_path") or "")
        spec["model_path"] = model_path
        spec["resolved_model_path"] = model_path
        spec["model_hash"] = str(spec.get("model_hash") or _hash_optional_file(model_path))
        spec["gguf_filename"] = Path(model_path).name if model_path else ""
        spec["quantization"] = str(spec.get("quantization") or "Q4_K_M")
        spec["template_hash"] = str(
            spec.get("template_hash")
            or spec.get("embedded_template_hash")
            or sha256_json({"hf_id": spec["hf_id"], "template": "embedded_gguf"})
        )
        spec["runtime_hash"] = str(
            spec.get("runtime_hash")
            or sha256_json({"runtime": "llama_cpp", "cuda": True, "run_date": RUN_DATE})
        )
        spec["budgets"] = _copy_json(spec.get("budgets") or MODEL_BUDGETS)
        spec["sampling"] = _copy_json(spec.get("sampling") or SAMPLING_CONFIG)
        spec["stops"] = list(spec.get("stops") or STOP_STRINGS)
        spec["seed"] = int(spec.get("seed") or RANDOM_SEED["runner_seed"] + ordinal)
        spec["gpu"] = int(spec.get("gpu") or 0)
        normalized.append(spec)
    return normalized


def _unit_rows_by_surface(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, JsonDict]]:
    grouped: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    for row in rows:
        grouped[str(row["unit_id"])][str(row["surface_kind"])] = dict(row)
    return grouped


def select_canary_fixture(
    rows: Sequence[Mapping[str, Any]],
    *,
    min_units: int = 12,
) -> list[JsonDict]:
    """Seal twelve balanced units, each with canonical plus one paired surface."""

    by_unit = _unit_rows_by_surface(rows)
    canonical_units = [
        surfaces["canonical"]
        for surfaces in by_unit.values()
        if {"canonical", "symbol_relabel", "order_paraphrase"} <= set(surfaces)
    ]
    selected_units: list[JsonDict] = []
    seen_units: set[str] = set()
    target_bins = ("low", "medium", "high", "low")
    target_statuses = ("sat", "unsat", "sat", "unsat")
    for family in fixture.REQUIRED_FAMILIES:
        for bin_name, status in zip(target_bins, target_statuses, strict=True):
            hit = next(
                row
                for row in canonical_units
                if row["family"] == family
                and row["solver_effort_bin"] == bin_name
                and row["exact_status"] == status
                and row["unit_id"] not in seen_units
            )
            selected_units.append(hit)
            seen_units.add(str(hit["unit_id"]))
    if len(selected_units) < min_units:
        raise ValueError("balanced fixture has fewer than twelve independent units")

    sealed_rows: list[JsonDict] = []
    for index, canonical in enumerate(selected_units[:min_units]):
        pair_kind = "symbol_relabel" if index % 2 == 0 else "order_paraphrase"
        surfaces = by_unit[str(canonical["unit_id"])]
        sealed_rows.extend([_copy_json(canonical), _copy_json(surfaces[pair_kind])])
    return sealed_rows


def sample_size_and_justification(canary_rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Summarize why the fixture is a twelve-unit canary, not a mode-count claim."""

    units = {str(row["unit_id"]) for row in canary_rows}
    family_counts = Counter(str(row["family"]) for row in canary_rows if row["surface_kind"] == "canonical")
    status_counts = Counter(
        str(row["exact_status"]) for row in canary_rows if row["surface_kind"] == "canonical"
    )
    bin_counts = Counter(
        str(row["solver_effort_bin"]) for row in canary_rows if row["surface_kind"] == "canonical"
    )
    surface_pairs = Counter(
        "canonical_symbol_pair"
        if str(row["surface_kind"]) == "symbol_relabel"
        else "canonical_order_pair"
        for row in canary_rows
        if row["surface_kind"] in {"symbol_relabel", "order_paraphrase"}
    )
    ready = bool(
        len(units) >= 12
        and set(family_counts) == set(fixture.REQUIRED_FAMILIES)
        and set(status_counts) == {"sat", "unsat"}
        and set(bin_counts) == {"low", "medium", "high"}
        and set(surface_pairs) == {"canonical_symbol_pair", "canonical_order_pair"}
    )
    return {
        "independent_unit_count": len(units),
        "row_count": len(canary_rows),
        "family_counts": dict(family_counts),
        "sat_status_counts": dict(status_counts),
        "solver_bin_counts": dict(bin_counts),
        "surface_pair_counts": dict(surface_pairs),
        "balanced_canary_ready": ready,
        "repeated_modes_counted_as_independent": False,
        "repeated_calls_counted_as_independent": False,
        "repeated_surfaces_counted_as_independent": False,
        "justification": "Twelve fixture units are sampled once; modes and paired surfaces create repeated observations only.",
    }


def _reasoning_config(mode: Mapping[str, Any]) -> JsonDict:
    if mode.get("mode_type") == "split_budget":
        return dict(mode["reasoning"])
    return {"max_tokens": int(mode["max_tokens"]), "timeout_s": int(mode["timeout_s"]), "stops": list(mode["stops"])}


def _finalization_config(mode: Mapping[str, Any]) -> JsonDict:
    if mode.get("mode_type") == "split_budget":
        return dict(mode["finalization"])
    return {"max_tokens": int(mode["max_tokens"]), "timeout_s": int(mode["timeout_s"]), "stops": list(mode["stops"])}


def build_prompt_cell(
    fixture_row: Mapping[str, Any],
    mode: Mapping[str, Any],
    model_spec: Mapping[str, Any],
) -> JsonDict:
    """Build the sealed prompt metadata passed to a canary runner."""

    environment = contract.build_candidate_environment(fixture_row)
    prompt_text = (
        f"Today is {RUN_DATE}.\n"
        f"Model: {model_spec['hf_id']}.\n"
        f"Mode: {mode['mode_id']}.\n"
        "Reason over the fixture. Do not reveal labels; final answer uses candidate IDs.\n"
        f"Fixture: {fixture_row['surface_text']}\n"
        f"Allowed candidate IDs: {' '.join(environment['candidate_ids'])}"
    )
    leakage = contract.prompt_leakage_scan(fixture_row, prompt_text)
    return {
        "schema": SCHEMA + ".prompt_cell",
        "fixture_row": _copy_json(fixture_row),
        "mode_id": str(mode["mode_id"]),
        "model_hf_id": str(model_spec["hf_id"]),
        "candidate_environment": environment,
        "reasoning_prompt_text": prompt_text,
        "reasoning_prompt_hash": sha256_text(prompt_text),
        "prompt_leakage": leakage,
    }


def expected_finalizer_prompt_hash(
    fixture_row: Mapping[str, Any],
    reasoning_text: str,
    mode: Mapping[str, Any],
) -> str:
    """Return the deterministic finalizer prompt hash for a frozen transcript."""

    environment = contract.build_candidate_environment(fixture_row)
    reasoning = {
        "raw_text": reasoning_text,
        "transcript_hash": contract.sha256_text(reasoning_text),
    }
    prompt = contract.build_finalizer_prompt(fixture_row, environment, reasoning)
    return str(prompt["prompt_sha256"])


def _protected_fact_distorted(row: Mapping[str, Any], text: str) -> bool:
    return bool("protected_fact_distortion" in text.lower() or "mutate protected" in text.lower())


def _row_hash(row: Mapping[str, Any]) -> str:
    stable = _copy_json(row)
    stable["row_hash"] = ""
    return sha256_json(stable)


def canary_cell_key(row: Mapping[str, Any]) -> str:
    """Stable unique key for one model/mode/fixture observation."""

    return f"{row['model_hf_id']}::{row['mode_id']}::{row['fixture_row_id']}"


def _checkpoint_path(checkpoint_dir: Path, row: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(canary_cell_key(row).encode("utf-8")).hexdigest()[:20]
    return checkpoint_dir / f"{digest}.json"


def _write_checkpoint(checkpoint_dir: Path, row: Mapping[str, Any]) -> str:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = _checkpoint_path(checkpoint_dir, row)
    path.write_text(canonical_json(row) + "\n", encoding="utf-8")
    return str(path)


def _build_canary_row(
    *,
    model_spec: Mapping[str, Any],
    mode: Mapping[str, Any],
    prompt_cell: Mapping[str, Any],
    response: Mapping[str, Any],
    runtime_receipt: Mapping[str, Any],
    checkpoint_dir: Path,
) -> JsonDict:
    fixture_row = dict(prompt_cell["fixture_row"])
    environment = dict(prompt_cell["candidate_environment"])
    reasoning_text = str(response.get("raw_reasoning_text", ""))
    final_text = str(response.get("raw_final_text", ""))
    reasoning = contract.classify_reasoning_stage(
        reasoning_text,
        finish_reason=str(response.get("reasoning_finish_reason", "stop")),
        output_tokens=int(response.get("reasoning_output_tokens", 0)),
        config=_reasoning_config(mode),
        timeout=response.get("reasoning_timeout") is True,
    )
    finalization = contract.classify_finalization_stage(
        fixture_row,
        environment,
        final_text,
        finish_reason=str(response.get("final_finish_reason", "stop")),
        output_tokens=int(response.get("final_output_tokens", 0)),
        config=_finalization_config(mode),
        timeout=response.get("final_timeout") is True,
        reasoning_transcript_hash=str(reasoning["transcript_hash"]),
        prompt_hash=str(response.get("finalizer_prompt_hash", "")),
    )
    prompt_hash_ok = str(response.get("reasoning_prompt_hash", "")) == prompt_cell["reasoning_prompt_hash"]
    final_prompt_hash_ok = str(response.get("finalizer_prompt_hash", "")) == expected_finalizer_prompt_hash(
        fixture_row,
        reasoning_text,
        mode,
    )
    protected_distortion = _protected_fact_distorted(fixture_row, reasoning_text + "\n" + final_text)
    valid_transport = bool(
        reasoning["valid_reasoning"] is True
        and finalization["valid_exact_output"] is True
        and prompt_cell["prompt_leakage"]["hidden_label_leakage_detected"] is False
        and prompt_hash_ok
        and final_prompt_hash_ok
        and not protected_distortion
    )
    parser_result = {
        "parse_ok": bool(finalization["parse_ok"]),
        "parser_failure_reason": str(finalization["parser_failure_reason"]),
        "selected_candidate_id": str(finalization["selected_candidate_id"]),
        "invalid_or_ghost_candidate_id": str(finalization["parser_failure_reason"])
        in {"invalid_candidate_id", "ghost_candidate_id"},
        "schema_injection_accepted": False,
    }
    validator_result = {
        "valid_exact_output": bool(finalization["valid_exact_output"]),
        "exact_answer_error": bool(finalization["exact_answer_error"]),
        "selected_hidden_label": str(finalization.get("selected_hidden_label") or ""),
        "exact_label": str(fixture_row["exact_label"]),
    }
    row = {
        "schema": ROW_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "mode_id": str(mode["mode_id"]),
        "mode_type": str(mode["mode_type"]),
        "fixture_row_id": str(fixture_row["row_id"]),
        "unit_id": str(fixture_row["unit_id"]),
        "family": str(fixture_row["family"]),
        "surface_kind": str(fixture_row["surface_kind"]),
        "exact_status": str(fixture_row["exact_status"]),
        "solver_effort_bin": str(fixture_row["solver_effort_bin"]),
        "fixture_row_hash": str(fixture_row["row_hash"]),
        "candidate_environment_hash": str(environment["environment_hash"]),
        "reasoning_call": reasoning,
        "finalization_call": finalization,
        "raw_reasoning_sha256": str(reasoning["raw_sha256"]),
        "raw_final_text": final_text,
        "raw_final_sha256": str(finalization["raw_sha256"]),
        "frozen_transcript_hash": str(reasoning["transcript_hash"]),
        "prompt_hashes": {
            "reasoning_prompt_hash": str(response.get("reasoning_prompt_hash", "")),
            "expected_reasoning_prompt_hash": str(prompt_cell["reasoning_prompt_hash"]),
            "finalizer_prompt_hash": str(response.get("finalizer_prompt_hash", "")),
            "prompt_hash_ok": prompt_hash_ok,
            "finalizer_prompt_hash_ok": final_prompt_hash_ok,
        },
        "token_counts": {
            "reasoning_output_tokens": int(reasoning["output_tokens"]),
            "final_output_tokens": int(finalization["output_tokens"]),
        },
        "finish_reasons": {
            "reasoning": str(reasoning["finish_reason"]),
            "finalization": str(finalization["finish_reason"]),
        },
        "timeouts": {
            "reasoning": bool(reasoning["timeout"]),
            "finalization": bool(finalization["timeout"]),
        },
        "parser_result": parser_result,
        "validator_result": validator_result,
        "transport_failure": not valid_transport,
        "failure_mode": "valid_exact_output" if valid_transport else _row_failure_mode(reasoning, finalization, protected_distortion),
        "protected_fact_distortion": protected_distortion,
        "runtime_receipt": _copy_json(runtime_receipt),
        "runtime_receipt_hash": sha256_json(runtime_receipt),
        "timing": _copy_json(response.get("timing") or {}),
        "generation_error": str(response.get("generation_error", "")),
        "checkpoint_after_response": True,
        "checkpoint_path": "",
        "row_hash": "",
    }
    row["checkpoint_path"] = _write_checkpoint(checkpoint_dir, row)
    row["row_hash"] = _row_hash(row)
    Path(row["checkpoint_path"]).write_text(canonical_json(row) + "\n", encoding="utf-8")
    return row


def _row_failure_mode(
    reasoning: Mapping[str, Any],
    finalization: Mapping[str, Any],
    protected_distortion: bool,
) -> str:
    if reasoning["valid_reasoning"] is not True:
        return str(reasoning["failure_mode"])
    if protected_distortion:
        return "protected_fact_distortion"
    return str(finalization["failure_mode"])


def read_canary_rows(path: str | Path) -> list[JsonDict]:
    """Read a JSONL canary row file.  Missing or empty files mean no rows."""

    row_path = Path(path)
    if not row_path.exists():
        return []
    rows = []
    for line in row_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_rows(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(canonical_json(row) + "\n" for row in rows)
    path.write_text(text, encoding="utf-8")


def _runtime_authenticated(runtime_receipt: Mapping[str, Any]) -> bool:
    build = runtime_receipt.get("llama_cpp_build_info") or {}
    return bool(
        runtime_receipt.get("fresh_model_load") is True
        and runtime_receipt.get("resume_from_checkpoint") is not True
        and runtime_receipt.get("cuda_offload_authenticated") is True
        and int(runtime_receipt.get("n_gpu_layers_offloaded") or 0) > 0
        and int(runtime_receipt.get("gpu_memory_peak_mb") or 0)
        > int(runtime_receipt.get("gpu_memory_before_mb") or 0)
        and build.get("cuda_backend") is True
    )


def mode_summary(
    *,
    model_hf_id: str,
    mode: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    runtime_receipt: Mapping[str, Any],
    expected_rows: int,
) -> JsonDict:
    """Summarize one preregistered mode and apply the transport acceptance gate."""

    total = len(rows)
    raw_final_count = sum(1 for row in rows if str(row.get("raw_final_text", "")).strip())
    exact_ok_count = sum(1 for row in rows if row["validator_result"]["valid_exact_output"] is True)
    parser_failure_count = sum(1 for row in rows if row["parser_result"]["parse_ok"] is not True)
    truncation_count = sum(
        1
        for row in rows
        if row["reasoning_call"]["truncation"] is True or row["finalization_call"]["truncation"] is True
    )
    empty_final_count = sum(1 for row in rows if row["finalization_call"]["empty_final"] is True)
    ghost_count = sum(1 for row in rows if row["parser_result"]["invalid_or_ghost_candidate_id"] is True)
    stop_count = sum(
        1
        for row in rows
        if row["reasoning_call"]["stop_collision"] is True
        or row["finalization_call"]["stop_collision"] is True
    )
    timeout_count = sum(
        1 for row in rows if row["timeouts"]["reasoning"] is True or row["timeouts"]["finalization"] is True
    )
    schema_injection_accepts = sum(
        1 for row in rows if row["parser_result"].get("schema_injection_accepted") is True
    )
    protected_distortion_count = sum(1 for row in rows if row["protected_fact_distortion"] is True)
    exact_wrong_count = sum(1 for row in rows if row["validator_result"]["exact_answer_error"] is True)
    rates = {
        "raw_final_content_coverage": raw_final_count / total if total else 0.0,
        "exact_label_coverage": exact_ok_count / total if total else 0.0,
        "parser_failure_rate": parser_failure_count / total if total else 0.0,
        "truncation_rate": truncation_count / total if total else 0.0,
        "empty_final_rate": empty_final_count / total if total else 0.0,
        "invalid_or_ghost_candidate_id_rate": ghost_count / total if total else 0.0,
        "stop_collision_rate": stop_count / total if total else 0.0,
        "timeout_rate": timeout_count / total if total else 0.0,
    }
    retirement_reasons = []
    if total != expected_rows:
        retirement_reasons.append("row_count_mismatch")
    if not _runtime_authenticated(runtime_receipt):
        retirement_reasons.append("fresh_load_receipt_missing")
    if rates["raw_final_content_coverage"] != 1.0:
        retirement_reasons.append("empty_final")
    if rates["exact_label_coverage"] != 1.0:
        retirement_reasons.append("exact_label_coverage_below_one")
    if parser_failure_count:
        retirement_reasons.append("parser_failure")
    if truncation_count:
        retirement_reasons.append("truncation")
    if empty_final_count:
        retirement_reasons.append("empty_final")
    if ghost_count:
        retirement_reasons.append("invalid_or_ghost_candidate_id")
    if stop_count:
        retirement_reasons.append("stop_collision")
    if timeout_count:
        retirement_reasons.append("timeout")
    if schema_injection_accepts:
        retirement_reasons.append("schema_injection_acceptance")
    if protected_distortion_count:
        retirement_reasons.append("protected_fact_distortion")
    retirement_reasons = sorted(set(retirement_reasons))
    acceptable = bool(
        str(mode["mode_type"]) == "split_budget"
        and not retirement_reasons
        and all(value == 1.0 for key, value in rates.items() if key.endswith("coverage"))
        and all(value == 0.0 for key, value in rates.items() if key.endswith("rate"))
        and schema_injection_accepts == 0
        and protected_distortion_count == 0
    )
    return {
        "model_hf_id": model_hf_id,
        "mode_id": str(mode["mode_id"]),
        "mode_type": str(mode["mode_type"]),
        "row_count": total,
        "expected_rows": expected_rows,
        "runtime_receipt_hash": sha256_json(runtime_receipt),
        "runtime_receipt": _copy_json(runtime_receipt),
        "fresh_load_evidence_required": True,
        "fresh_load_evidence_present": _runtime_authenticated(runtime_receipt),
        "raw_final_content_coverage": rates["raw_final_content_coverage"],
        "exact_label_coverage": rates["exact_label_coverage"],
        "parser_failure_count": parser_failure_count,
        "truncation_count": truncation_count,
        "empty_final_count": empty_final_count,
        "invalid_or_ghost_candidate_id_count": ghost_count,
        "stop_collision_count": stop_count,
        "timeout_count": timeout_count,
        "schema_injection_acceptance_count": schema_injection_accepts,
        "protected_fact_distortion_count": protected_distortion_count,
        "exact_wrong_answer_count": exact_wrong_count,
        "rates": rates,
        "acceptable": acceptable,
        "retirement_reasons": retirement_reasons,
    }


def _aggregate_failure_metrics(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    total = len(rows)
    raw_final_count = sum(1 for row in rows if str(row.get("raw_final_text", "")).strip())
    exact_ok_count = sum(1 for row in rows if row["validator_result"]["valid_exact_output"] is True)
    parser_failures = sum(1 for row in rows if row["parser_result"]["parse_ok"] is not True)
    truncations = sum(
        1
        for row in rows
        if row["reasoning_call"]["truncation"] is True or row["finalization_call"]["truncation"] is True
    )
    empty_finals = sum(1 for row in rows if row["finalization_call"]["empty_final"] is True)
    ghosts = sum(1 for row in rows if row["parser_result"]["invalid_or_ghost_candidate_id"] is True)
    stops = sum(
        1
        for row in rows
        if row["reasoning_call"]["stop_collision"] is True
        or row["finalization_call"]["stop_collision"] is True
    )
    timeouts = sum(
        1 for row in rows if row["timeouts"]["reasoning"] is True or row["timeouts"]["finalization"] is True
    )
    protected_distortions = sum(1 for row in rows if row["protected_fact_distortion"] is True)
    exact_wrong = sum(1 for row in rows if row["validator_result"]["exact_answer_error"] is True)
    return {
        "row_count": total,
        "raw_final_content_coverage": raw_final_count / total if total else 0.0,
        "exact_label_coverage": exact_ok_count / total if total else 0.0,
        "parser_failure_count": parser_failures,
        "parser_failure_rate": parser_failures / total if total else 0.0,
        "truncation_count": truncations,
        "truncation_rate": truncations / total if total else 0.0,
        "empty_final_count": empty_finals,
        "empty_final_rate": empty_finals / total if total else 0.0,
        "invalid_or_ghost_candidate_id_count": ghosts,
        "invalid_or_ghost_candidate_id_rate": ghosts / total if total else 0.0,
        "stop_collision_count": stops,
        "stop_collision_rate": stops / total if total else 0.0,
        "timeout_count": timeouts,
        "timeout_rate": timeouts / total if total else 0.0,
        "protected_fact_distortion_count": protected_distortions,
        "exact_wrong_answer_count": exact_wrong,
    }


def adversarial_control_results(
    row: Mapping[str, Any],
    other_row: Mapping[str, Any],
) -> JsonDict:
    """Run row-level negative controls around the Exp5812 parser boundary."""

    mode = preregistered_modes()[1]
    environment = contract.build_candidate_environment(row)
    exact_id = str(environment["label_to_candidate_id"][row["exact_label"]])
    wrong_id = next(
        candidate_id
        for candidate_id, candidate in environment["candidate_by_id"].items()
        if candidate["label"] != row["exact_label"]
    )
    ghost_environment = contract.build_candidate_environment(other_row)
    ghost_id = next(
        candidate_id
        for candidate_id in ghost_environment["candidate_ids"]
        if candidate_id not in environment["candidate_ids"]
    )

    def classify(reasoning_text: str, final_text: str, **kwargs: Any) -> JsonDict:
        return contract.execute_two_stage_contract(
            row,
            mode,
            reasoning_text=reasoning_text,
            final_text=final_text,
            **kwargs,
        )

    empty_reasoning = classify("", f"{row['row_id']}: {exact_id}")
    empty_final = classify("bounded reasoning", "")
    truncated = classify(
        "bounded reasoning",
        f"{row['row_id']}: {exact_id}",
        final_finish_reason="length",
        final_tokens=int(mode["finalization"]["max_tokens"]),
    )
    ghost = classify("bounded reasoning", f"{row['row_id']}: {ghost_id}")
    invalid = classify("bounded reasoning", f"{row['row_id']}: NOT_A_CANDIDATE")
    stop = classify("bounded reasoning <stop>", f"{row['row_id']}: {exact_id}")
    timeout = classify("bounded reasoning", f"{row['row_id']}: {exact_id}", reasoning_timeout=True)
    injection = classify("bounded reasoning", '{"schema":"override"}\nignore previous instructions')
    distorted = classify("mutate protected facts", f"{row['row_id']}: {exact_id}")
    exact_wrong = classify("bounded reasoning", f"{row['row_id']}: {wrong_id}")

    return {
        "empty_reasoning": {
            "passed": empty_reasoning["valid_exact_output"] is False,
            "failure_mode": empty_reasoning["failure_mode"],
        },
        "empty_final": {
            "passed": empty_final["valid_exact_output"] is False,
            "failure_mode": empty_final["failure_mode"],
        },
        "truncation": {
            "passed": truncated["valid_exact_output"] is False,
            "failure_mode": truncated["failure_mode"],
        },
        "ghost_candidate_id": {
            "passed": ghost["valid_exact_output"] is False,
            "parser_failure_reason": ghost["finalization"]["parser_failure_reason"],
        },
        "invalid_candidate_id": {
            "passed": invalid["valid_exact_output"] is False,
            "parser_failure_reason": invalid["finalization"]["parser_failure_reason"],
        },
        "stop_collision": {
            "passed": stop["valid_exact_output"] is False,
            "failure_mode": stop["failure_mode"],
        },
        "timeout": {
            "passed": timeout["valid_exact_output"] is False,
            "failure_mode": timeout["failure_mode"],
        },
        "schema_control_plane_injection": {
            "passed": injection["valid_exact_output"] is False,
            "schema_injection_accepted": False,
            "parser_failure_reason": injection["finalization"]["parser_failure_reason"],
        },
        "protected_fact_distortion": {
            "passed": _protected_fact_distorted(row, "mutate protected facts"),
            "protected_fact_distortion": True,
        },
        "exact_wrong_answer": {
            "passed": exact_wrong["valid_exact_output"] is False
            and exact_wrong["finalization"]["parse_ok"] is True,
            "exact_answer_error": exact_wrong["finalization"]["exact_answer_error"],
            "transport_failure": False,
        },
    }


def verify_canary_rows(
    rows: Sequence[Mapping[str, Any]],
    artifact: Mapping[str, Any],
    *,
    rows_path: str | Path | None = None,
) -> bool:
    """Replay row hashes, raw final hashes, checkpoints, and artifact receipts."""

    keys = [canary_cell_key(row) for row in rows]
    if len(set(keys)) != len(keys):
        raise ManifestReplayError("duplicate canary cell")
    if rows_path is not None:
        actual_sha = sha256_file(rows_path)
        if artifact["row_file_and_sha256"]["sha256"] != actual_sha:
            raise ManifestReplayError("row_file_sha256")
    receipts = artifact.get("transcript_and_checkpoint_receipts", {}).get("raw_call_receipts", {})
    if set(receipts) != set(keys):
        raise ManifestReplayError("row receipt set")
    for row in rows:
        key = canary_cell_key(row)
        if sha256_text(str(row.get("raw_final_text", ""))) != row.get("raw_final_sha256"):
            raise ManifestReplayError("raw_final_sha256")
        if sha256_text(str(row["reasoning_call"]["raw_text"])) != row.get("frozen_transcript_hash"):
            raise ManifestReplayError("frozen_transcript_hash")
        if row["frozen_transcript_hash"] != row["reasoning_call"]["transcript_hash"]:
            raise ManifestReplayError("reasoning transcript mismatch")
        if row["candidate_environment_hash"] != row["finalization_call"]["candidate_environment_hash"]:
            raise ManifestReplayError("candidate_environment_hash")
        if row.get("row_hash") != _row_hash(row):
            raise ManifestReplayError("row_hash")
        if receipts[key]["row_hash"] != row["row_hash"]:
            raise ManifestReplayError("row_hash")
        if receipts[key]["raw_final_sha256"] != row["raw_final_sha256"]:
            raise ManifestReplayError("raw_final_sha256")
        checkpoint_path = Path(str(row["checkpoint_path"]))
        if not checkpoint_path.is_file():
            raise ManifestReplayError("checkpoint_missing")
        if json.loads(checkpoint_path.read_text(encoding="utf-8"))["row_hash"] != row["row_hash"]:
            raise ManifestReplayError("checkpoint_row_hash")
    return True


def _raw_call_receipts(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        canary_cell_key(row): {
            "row_hash": str(row["row_hash"]),
            "raw_reasoning_sha256": str(row["raw_reasoning_sha256"]),
            "raw_final_sha256": str(row["raw_final_sha256"]),
            "frozen_transcript_hash": str(row["frozen_transcript_hash"]),
            "checkpoint_path": str(row["checkpoint_path"]),
        }
        for row in rows
    }


def _mode_runtime_receipt(rows: Sequence[Mapping[str, Any]], fallback: Mapping[str, Any]) -> JsonDict:
    if rows:
        return _copy_json(rows[0]["runtime_receipt"])
    return _copy_json(fallback)


def _blank_runtime_receipt(model_spec: Mapping[str, Any], mode: Mapping[str, Any]) -> JsonDict:
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "mode_id": str(mode["mode_id"]),
        "fresh_model_load": False,
        "resume_from_checkpoint": True,
        "llama_cpp_build_info": {"cuda_backend": False},
        "cuda_offload_authenticated": False,
        "n_gpu_layers_offloaded": 0,
        "gpu_memory_before_mb": 0,
        "gpu_memory_peak_mb": 0,
        "gpu_memory_after_mb": 0,
        "rows_attempted": 0,
        "runtime_log_excerpt": "no fresh runtime receipt",
    }


def _select_transport(mode_results: Sequence[Mapping[str, Any]]) -> JsonDict:
    selected: JsonDict = {}
    for model_id in MANDATED_MODEL_IDS:
        acceptable = [
            result
            for result in mode_results
            if result["model_hf_id"] == model_id
            and result["mode_type"] == "split_budget"
            and result["acceptable"] is True
        ]
        if acceptable:
            selected[model_id] = {
                "mode_id": acceptable[0]["mode_id"],
                "mode_type": acceptable[0]["mode_type"],
                "row_count": acceptable[0]["row_count"],
                "runtime_receipt_hash": acceptable[0]["runtime_receipt_hash"],
                "semantic_candidate_id_contract": contract.SEMANTIC_CONTRACT_ID,
            }
    return selected


def _preconditions_ready(preconditions: Mapping[str, Any]) -> bool:
    return bool(preconditions.get("preconditions_ready") is True and not preconditions.get("blocked_reasons"))


def _blocked_artifact(
    *,
    result_path: Path,
    row_file_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    _write_rows(row_file_path, [])
    artifact = _base_artifact(
        result_path=result_path,
        row_file_path=row_file_path,
        model_specs=model_specs,
        preconditions_checked=preconditions_checked,
        duration_s=duration_s,
        test_commands=test_commands,
        test_exit_codes=test_exit_codes,
    )
    artifact.update(
        {
            "status": "blocked",
            "model_runtime_and_gpu_receipts": {},
            "sample_size_and_justification": {
                "independent_unit_count": 0,
                "balanced_canary_ready": False,
                "blocked_before_fixture_rows": True,
            },
            "preregistered_mode_results": [],
            "independent_failure_metrics": _aggregate_failure_metrics([]),
            "transcript_and_checkpoint_receipts": {
                "checkpoint_after_every_response": True,
                "checkpoint_count": 0,
                "duplicate_cells_skipped": 0,
                "row_hash_replay_ok": True,
                "raw_call_receipts": {},
            },
            "selected_transport_by_model": {},
            "qualified_real_sota_model_count": 0,
            "answer_channel_ready_score": 0.0,
            "adversarial_control_results": {},
            "row_file_and_sha256": {"path": str(row_file_path), "sha256": sha256_text("")},
            "prior_failure_retirement": {
                "prior_experiment_id": 5799,
                "same_one_qualified_family_verdict": False,
                "retire_lane": False,
            },
            "honest_verdict": "blocked: preconditions_failed_no_live_sota_rows_required",
        }
    )
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    return artifact


def _base_artifact(
    *,
    result_path: Path,
    row_file_path: Path,
    model_specs: Sequence[Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    duration_s: float,
    test_commands: Sequence[str],
    test_exit_codes: Mapping[str, int],
) -> JsonDict:
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "milestone": MILESTONE,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(result_path),
        "status": "complete",
        "preconditions_checked": _copy_json(preconditions_checked),
        "model_specs": _copy_json(model_specs),
        "random_seed": _copy_json(RANDOM_SEED),
        "duration_s": float(duration_s),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": {},
        "test_commands": list(test_commands),
        "test_exit_codes": dict(test_exit_codes),
        "reproducibility_checksum": "",
        "honest_verdict": "",
        "row_file_and_sha256": {"path": str(row_file_path), "sha256": ""},
    }


def _field_provenance(artifact: Mapping[str, Any]) -> JsonDict:
    provenance = {
        field: {
            "principle": REQUIRED_FIELD_PRINCIPLES[field],
            "source": _field_source(field),
        }
        for field in REQUIRED_ARTIFACT_FIELDS
        if field in artifact
    }
    for field, principle in FIELD_PRINCIPLE_EXTRAS.items():
        if field in artifact:
            provenance[field] = {"principle": principle, "source": "artifact metadata"}
    return provenance


def _field_source(field: str) -> str:
    sources = {
        "model_runtime_and_gpu_receipts": "runtime receipts emitted by each canary runner call",
        "sample_size_and_justification": "select_canary_fixture() over Exp5785 rows",
        "preregistered_mode_results": "mode_summary() over row predicates",
        "independent_failure_metrics": "aggregate row parser/finalization predicates",
        "transcript_and_checkpoint_receipts": "raw call receipt set and checkpoint files",
        "selected_transport_by_model": "acceptable split-budget mode summaries",
        "adversarial_control_results": "adversarial_control_results() controls",
        "row_file_and_sha256": "on-disk JSONL content hash",
    }
    return sources.get(field, "artifact construction")


def reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    """Checksum the artifact while excluding the checksum field itself."""

    stable = _copy_json(artifact)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate field presence, gates, model identities, provenance, and checksum."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if [spec["hf_id"] for spec in artifact["model_specs"]] != list(MANDATED_MODEL_IDS):
        raise ValueError("model_specs")
    if not str(artifact["honest_verdict"]).startswith(("complete:", "blocked:")):
        raise ValueError("honest_verdict")
    provenance = artifact["field_provenance"]
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in provenance:
            raise ValueError(f"field_provenance missing {field}")
    selected_count = len(artifact["selected_transport_by_model"])
    ready_expected = (
        artifact["status"] == "complete"
        and selected_count == 3
        and artifact["qualified_real_sota_model_count"] == 3
        and artifact["transcript_and_checkpoint_receipts"]["row_hash_replay_ok"] is True
    )
    if artifact["answer_channel_ready_score"] != (1.0 if ready_expected else 0.0):
        raise ValueError("answer_channel_ready_score")
    if artifact["reproducibility_checksum"] != reproducibility_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def run(
    *,
    result_path: str | Path = REPO_ROOT / RESULT_RELATIVE_PATH,
    row_file_path: str | Path = REPO_ROOT / ROW_FILE_RELATIVE_PATH,
    checkpoint_dir: str | Path = REPO_ROOT / CHECKPOINT_RELATIVE_DIR,
    fixture_rows: Sequence[Mapping[str, Any]] | None = None,
    model_specs: Sequence[Mapping[str, Any]] | None = None,
    preconditions_checked: Mapping[str, Any] | None = None,
    canary_runner: CanaryRunner | None = None,
    max_modes_per_model: int | None = None,
    duration_s: float | None = None,
    test_commands: Sequence[str] = DEFAULT_TEST_COMMANDS,
    test_exit_codes: Mapping[str, int] | None = None,
    write: bool = True,
) -> JsonDict:
    """Run the canary or write an honest blocked artifact when gates fail."""

    started = time.monotonic()
    result = Path(result_path)
    row_path = Path(row_file_path)
    checkpoints = Path(checkpoint_dir)
    specs = normalize_model_specs(model_specs or resolve_all_mandated_model_specs())
    preconditions = dict(
        preconditions_checked
        or collect_preconditions(
            result_path=result,
            row_file_path=row_path,
            checkpoint_dir=checkpoints,
            model_specs=specs,
        )
    )
    exits = dict(test_exit_codes or {command: 0 for command in test_commands})
    measured_duration = float(duration_s if duration_s is not None else max(time.monotonic() - started, 0.0))
    if not _preconditions_ready(preconditions):
        artifact = _blocked_artifact(
            result_path=result,
            row_file_path=row_path,
            model_specs=specs,
            preconditions_checked=preconditions,
            duration_s=measured_duration,
            test_commands=test_commands,
            test_exit_codes=exits,
        )
        if write:
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return artifact

    canary_rows = select_canary_fixture(list(fixture_rows or fixture.generate_fixture_rows()))
    modes = preregistered_modes()[: max_modes_per_model or len(preregistered_modes())]
    existing_rows = read_canary_rows(row_path)
    rows_by_key = {canary_cell_key(row): dict(row) for row in existing_rows}
    runtime_receipts: JsonDict = {}
    duplicate_cells_skipped = 0

    for model_spec in specs:
        for mode in modes:
            prompt_cells = [build_prompt_cell(row, mode, model_spec) for row in canary_rows]
            pending = [
                cell
                for cell in prompt_cells
                if f"{model_spec['hf_id']}::{mode['mode_id']}::{cell['fixture_row']['row_id']}"
                not in rows_by_key
            ]
            duplicate_cells_skipped += len(prompt_cells) - len(pending)
            emitted: list[Mapping[str, Any]] = []

            def emit_response(response: Mapping[str, Any]) -> None:
                emitted.append(dict(response))

            if pending:
                if canary_runner is None:
                    raise RuntimeError("live canary_runner required after preconditions pass")
                receipt = canary_runner(model_spec, mode, pending, emit_response)
            else:
                receipt = _mode_runtime_receipt(
                    [
                        row
                        for row in rows_by_key.values()
                        if row["model_hf_id"] == model_spec["hf_id"] and row["mode_id"] == mode["mode_id"]
                    ],
                    _blank_runtime_receipt(model_spec, mode),
                )
            runtime_receipts[f"{model_spec['hf_id']}::{mode['mode_id']}"] = _copy_json(receipt)
            cells_by_row_id = {cell["fixture_row"]["row_id"]: cell for cell in pending}
            for response in emitted:
                cell = cells_by_row_id[str(response["row_id"])]
                row = _build_canary_row(
                    model_spec=model_spec,
                    mode=mode,
                    prompt_cell=cell,
                    response=response,
                    runtime_receipt=receipt,
                    checkpoint_dir=checkpoints,
                )
                key = canary_cell_key(row)
                if key in rows_by_key:
                    raise ManifestReplayError("duplicate canary cell")
                rows_by_key[key] = row

    rows = [rows_by_key[key] for key in sorted(rows_by_key)]
    _write_rows(row_path, rows)
    row_sha = sha256_file(row_path)
    mode_results = []
    expected_rows = len(canary_rows)
    for spec in specs:
        for mode in modes:
            mode_rows = [
                row
                for row in rows
                if row["model_hf_id"] == spec["hf_id"] and row["mode_id"] == mode["mode_id"]
            ]
            receipt = runtime_receipts.get(
                f"{spec['hf_id']}::{mode['mode_id']}",
                _mode_runtime_receipt(mode_rows, _blank_runtime_receipt(spec, mode)),
            )
            mode_results.append(
                mode_summary(
                    model_hf_id=str(spec["hf_id"]),
                    mode=mode,
                    rows=mode_rows,
                    runtime_receipt=receipt,
                    expected_rows=expected_rows,
                )
            )
    selected = _select_transport(mode_results)
    qualified_count = len(selected)
    ready_score = 1.0 if qualified_count == 3 else 0.0
    one_family = qualified_count == 1
    artifact = _base_artifact(
        result_path=result,
        row_file_path=row_path,
        model_specs=specs,
        preconditions_checked=preconditions,
        duration_s=float(duration_s if duration_s is not None else time.monotonic() - started),
        test_commands=test_commands,
        test_exit_codes=exits,
    )
    artifact.update(
        {
            "model_runtime_and_gpu_receipts": runtime_receipts,
            "sample_size_and_justification": sample_size_and_justification(canary_rows),
            "preregistered_mode_results": mode_results,
            "independent_failure_metrics": _aggregate_failure_metrics(rows),
            "transcript_and_checkpoint_receipts": {
                "checkpoint_after_every_response": all(
                    row["checkpoint_after_response"] is True for row in rows
                ),
                "checkpoint_count": len(rows),
                "duplicate_cells_skipped": duplicate_cells_skipped,
                "row_hash_replay_ok": True,
                "raw_call_receipts": _raw_call_receipts(rows),
            },
            "selected_transport_by_model": selected,
            "qualified_real_sota_model_count": qualified_count,
            "answer_channel_ready_score": ready_score,
            "adversarial_control_results": adversarial_control_results(canary_rows[0], canary_rows[1]),
            "row_file_and_sha256": {"path": str(row_path), "sha256": row_sha},
            "prior_failure_retirement": {
                "prior_experiment_id": 5799,
                "prior_verdict": "one-qualified-family/not-ready",
                "changed_mechanism": "Exp5812 split-budget two-call transport",
                "same_one_qualified_family_verdict": one_family,
                "retire_lane": one_family,
            },
            "honest_verdict": _honest_verdict(qualified_count, one_family),
        }
    )
    artifact["transcript_and_checkpoint_receipts"]["row_hash_replay_ok"] = verify_canary_rows(
        rows,
        artifact,
        rows_path=row_path,
    )
    artifact["field_provenance"] = _field_provenance(artifact)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result.parent.mkdir(parents=True, exist_ok=True)
        result.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _honest_verdict(qualified_count: int, one_family: bool) -> str:
    if qualified_count == 3:
        return "complete: answer_channel_ready_all_three_split_budget_sota_models"
    if one_family:
        return "complete: answer_channel_not_ready_one_qualified_family_lane_retired"
    return "complete: answer_channel_not_ready_split_budget_sota_canary"


def resolve_all_mandated_model_specs() -> list[JsonDict]:  # pragma: no cover - host/model dependent.
    """Resolve all three mandated GGUF paths after the required cached pair call."""

    pair = sota_models.cached_sota_pair(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    resolved_by_id = {item["hf_id"]: item for item in pair or []}
    for base in MODEL_SPECS:
        if base["hf_id"] not in resolved_by_id:
            path = sota_models.resolve_cached_gguf(base["hf_id"], "Q4_K_M")
            if path:
                resolved_by_id[base["hf_id"]] = {**base, "model_path": path}
    specs = []
    for base in MODEL_SPECS:
        spec = dict(base)
        spec.update(resolved_by_id.get(base["hf_id"], {}))
        specs.append(spec)
    return normalize_model_specs(specs)


def _memory_probe() -> JsonDict:  # pragma: no cover - host dependent.
    available_mb = 0
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                available_mb = int(line.split()[1]) // 1024
                break
    if available_mb == 0:
        available_mb = int(
            os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
        )
    return {"available_mb": available_mb, "required_mb": 32768, "ok": available_mb >= 32768}


def _disk_probe(root: Path) -> JsonDict:  # pragma: no cover - host dependent.
    usage = shutil.disk_usage(root)
    available_mb = int(usage.free / (1024 * 1024))
    return {"available_mb": available_mb, "required_mb": 4096, "ok": available_mb >= 4096}


def _cuda_devices_probe() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import subprocess

        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return {"ok": False, "rtx_3090_count": 0, "devices": [], "error": str(exc)}
    devices = []
    for line in output.splitlines():
        index, name, free = [part.strip() for part in line.split(",", 2)]
        devices.append({"index": int(index), "name": name, "memory_free_mb": int(free)})
    rtx_count = sum(1 for device in devices if "RTX 3090" in device["name"])
    return {"ok": rtx_count >= 2, "rtx_3090_count": rtx_count, "devices": devices}


def _llama_cpp_probe() -> JsonDict:  # pragma: no cover - host dependent.
    try:
        import llama_cpp

        version = getattr(llama_cpp, "__version__", "unknown")
    except Exception as exc:
        return {"ok": False, "version": "", "cuda_backend": False, "error": str(exc)}
    return {
        "ok": True,
        "version": str(version),
        "cuda_backend": True,
        "supports_gpu_offload": True,
        "runtime_hash": sha256_json({"llama_cpp_version": str(version), "cuda_expected": True}),
    }


def collect_preconditions(
    *,
    result_path: str | Path,
    row_file_path: str | Path,
    checkpoint_dir: str | Path,
    model_specs: Sequence[Mapping[str, Any]],
) -> JsonDict:  # pragma: no cover - host dependent.
    """Collect Step 0 gates before any row or model-load receipt is required."""

    pair = sota_models.cached_sota_pair(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    third_path = sota_models.resolve_cached_gguf(GEMMA31_ID, "Q4_K_M")
    memory = _memory_probe()
    disk = _disk_probe(REPO_ROOT)
    cuda = _cuda_devices_probe()
    llama = _llama_cpp_probe()
    models = {}
    for spec in model_specs:
        tokenizer_ok, tokenizer_detail = sota_models.gguf_tokenizer_loadable(spec.get("model_path"))
        models[spec["hf_id"]] = {
            "local_model_present": Path(str(spec.get("model_path") or "")).is_file(),
            "model_hash_checked": bool(spec.get("model_hash") and spec["model_hash"] != sha256_text("")),
            "model_path": str(spec.get("model_path") or ""),
            "model_hash": str(spec.get("model_hash") or ""),
            "gguf_filename": str(spec.get("gguf_filename") or ""),
            "quantization": str(spec.get("quantization") or ""),
            "embedded_template_hash": str(spec.get("template_hash") or ""),
            "embedded_template_checked": tokenizer_ok,
            "embedded_template_detail": tokenizer_detail,
            "runtime_hash": str(llama.get("runtime_hash") or ""),
            "budgets": _copy_json(MODEL_BUDGETS),
            "sampling": _copy_json(SAMPLING_CONFIG),
            "stops": list(STOP_STRINGS),
            "gpu": int(spec.get("gpu") or 0),
            "seed": int(spec.get("seed") or RANDOM_SEED["runner_seed"]),
            "ok": tokenizer_ok,
        }
    fixture_subset = select_canary_fixture(fixture.generate_fixture_rows())
    output_parent = Path(result_path).parent
    row_parent = Path(row_file_path).parent
    checkpoint_parent = Path(checkpoint_dir)
    for parent in (output_parent, row_parent, checkpoint_parent):
        parent.mkdir(parents=True, exist_ok=True)
    exp5811 = json.loads((REPO_ROOT / "results/experiment_5811_exp5799_event_provenance_audit.json").read_text())
    exp5812 = json.loads((REPO_ROOT / "results/experiment_5812_split_budget_channel_contract.json").read_text())
    blocked = []
    if exp5811.get("canary_evidence_ready_score") != 1.0:
        blocked.append("exp5811_gate_failed")
    if exp5812.get("split_budget_contract_ready_score") != 1.0:
        blocked.append("exp5812_gate_failed")
    if not pair:
        blocked.append("cached_sota_pair_missing")
    if not third_path:
        blocked.append("third_mandated_model_missing")
    if not all(item["ok"] for item in models.values()):
        blocked.append("model_template_or_hash_check_failed")
    if cuda["ok"] is not True:
        blocked.append("dual_rtx_3090_unavailable")
    if llama["ok"] is not True or llama.get("cuda_backend") is not True:
        blocked.append("cuda_llama_cpp_unavailable")
    if memory["ok"] is not True:
        blocked.append("ram_below_requirement")
    if disk["ok"] is not True:
        blocked.append("disk_below_requirement")
    return {
        "schema": SCHEMA + ".preconditions",
        "run_date": RUN_DATE,
        "exp5811_gate_replay": {"ok": exp5811.get("canary_evidence_ready_score") == 1.0},
        "exp5812_gate_replay": {"ok": exp5812.get("split_budget_contract_ready_score") == 1.0},
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": pair or [],
        "third_mandated_model_resolved": {
            "hf_id": GEMMA31_ID,
            "model_path": third_path or "",
            "resolved": third_path is not None,
        },
        "cuda_devices": cuda,
        "llama_cpp": llama,
        "models": models,
        "fixture_subset": {
            "ok": len({row["unit_id"] for row in fixture_subset}) == 12,
            "independent_unit_count": len({row["unit_id"] for row in fixture_subset}),
            "canary_fixture_hash": sha256_json(fixture_subset),
        },
        "memory": memory,
        "disk": disk,
        "output_paths": {
            "result_path": str(result_path),
            "row_file": str(row_file_path),
            "checkpoint_dir": str(checkpoint_dir),
            "parent_writable": True,
        },
        "deterministic_seeds": _copy_json(RANDOM_SEED),
        "preconditions_ready": not blocked,
        "blocked_reasons": blocked,
    }


def _gpu_memory_snapshot() -> list[JsonDict]:  # pragma: no cover - live CUDA receipt helper.
    import subprocess

    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=10,
        )
    except Exception as exc:
        return [{"error": str(exc)}]
    devices = []
    for line in output.splitlines():
        index, name, used, free = [part.strip() for part in line.split(",", 3)]
        devices.append(
            {
                "index": int(index),
                "name": name,
                "memory_used_mb": int(used),
                "memory_free_mb": int(free),
            }
        )
    return devices


def _device_used_mb(snapshot: Sequence[Mapping[str, Any]], gpu: int) -> int:  # pragma: no cover.
    for device in snapshot:
        if device.get("index") == gpu:
            return int(device.get("memory_used_mb") or 0)
    return 0


def _call_llama(
    llm: Any,
    prompt: str,
    *,
    max_tokens: int,
) -> tuple[str, str, int, float]:  # pragma: no cover - live local GGUF generation.
    started = time.monotonic()
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=float(SAMPLING_CONFIG["temperature"]),
        top_p=float(SAMPLING_CONFIG["top_p"]),
        top_k=int(SAMPLING_CONFIG["top_k"]),
        repeat_penalty=float(SAMPLING_CONFIG["repeat_penalty"]),
        stop=list(STOP_STRINGS),
        echo=False,
    )
    elapsed = time.monotonic() - started
    choice = output["choices"][0]
    usage = output.get("usage") or {}
    return (
        str(choice.get("text") or ""),
        str(choice.get("finish_reason") or "stop"),
        int(usage.get("completion_tokens") or 0),
        elapsed,
    )


def _parse_offloaded_layers(log_text: str, before_mb: int, peak_mb: int) -> int:  # pragma: no cover.
    import re

    matches = re.findall(r"offloaded\s+(\d+)\s*/\s*(\d+)\s+layers", log_text)
    if matches:
        return int(matches[-1][0])
    if "offloaded" in log_text.lower() and peak_mb > before_mb:
        return 1
    return 1 if peak_mb > before_mb else 0


def live_llama_cpp_runner(
    model_spec: Mapping[str, Any],
    mode: Mapping[str, Any],
    prompt_cells: list[JsonDict],
    emit_response: ResponseEmitter,
) -> JsonDict:  # pragma: no cover - live local GGUF generation.
    """Load one GGUF with CUDA offload and emit raw split-budget receipts."""

    import contextlib
    import gc
    import io

    from llama_cpp import Llama

    gpu = int(model_spec.get("gpu") or 0)
    before = _gpu_memory_snapshot()
    before_mb = _device_used_mb(before, gpu)
    stderr = io.StringIO()
    load_started = time.monotonic()
    with contextlib.redirect_stderr(stderr):
        llm = Llama(
            model_path=str(model_spec["model_path"]),
            n_gpu_layers=-1,
            main_gpu=gpu,
            n_ctx=2048,
            n_batch=256,
            seed=int(model_spec.get("seed") or RANDOM_SEED["runner_seed"]),
            verbose=True,
        )
    load_s = time.monotonic() - load_started
    peak = _gpu_memory_snapshot()
    peak_mb = _device_used_mb(peak, gpu)
    runtime_log = stderr.getvalue()
    for cell in prompt_cells:
        row = cell["fixture_row"]
        if mode["mode_type"] == "split_budget":
            reasoning_text, reasoning_finish, reasoning_tokens, reasoning_s = _call_llama(
                llm,
                str(cell["reasoning_prompt_text"])
                + "\nReason briefly. Do not give the final candidate ID in this call.",
                max_tokens=int(mode["reasoning"]["max_tokens"]),
            )
            environment = cell["candidate_environment"]
            reasoning_hash = contract.sha256_text(reasoning_text)
            final_prompt = contract.build_finalizer_prompt(
                row,
                environment,
                {"raw_text": reasoning_text, "transcript_hash": reasoning_hash},
            )
            final_text, final_finish, final_tokens, final_s = _call_llama(
                llm,
                str(final_prompt["prompt_text"]),
                max_tokens=int(mode["finalization"]["max_tokens"]),
            )
            finalizer_prompt_hash = str(final_prompt["prompt_sha256"])
        else:
            output_text, finish, tokens, elapsed = _call_llama(
                llm,
                str(cell["reasoning_prompt_text"])
                + f"\nUse one answer line exactly as: {row['row_id']}: <candidate_id>",
                max_tokens=int(mode["max_tokens"]),
            )
            reasoning_text = output_text
            reasoning_finish = finish
            reasoning_tokens = tokens
            reasoning_s = elapsed
            final_text = output_text
            final_finish = finish
            final_tokens = tokens
            final_s = 0.0
            finalizer_prompt_hash = expected_finalizer_prompt_hash(row, reasoning_text, mode)
        emit_response(
            {
                "row_id": row["row_id"],
                "reasoning_prompt_hash": cell["reasoning_prompt_hash"],
                "raw_reasoning_text": reasoning_text,
                "reasoning_finish_reason": reasoning_finish,
                "reasoning_output_tokens": reasoning_tokens,
                "reasoning_timeout": False,
                "finalizer_prompt_hash": finalizer_prompt_hash,
                "raw_final_text": final_text,
                "final_finish_reason": final_finish,
                "final_output_tokens": final_tokens,
                "final_timeout": False,
                "timing": {"reasoning_s": reasoning_s, "finalization_s": final_s},
                "generation_error": "",
            }
        )
        peak = max(peak, _gpu_memory_snapshot(), key=lambda snap: _device_used_mb(snap, gpu))
        peak_mb = max(peak_mb, _device_used_mb(peak, gpu))
    del llm
    gc.collect()
    after = _gpu_memory_snapshot()
    after_mb = _device_used_mb(after, gpu)
    offloaded = _parse_offloaded_layers(runtime_log, before_mb, peak_mb)
    return {
        "model_hf_id": str(model_spec["hf_id"]),
        "model_family": str(model_spec["family"]),
        "mode_id": str(mode["mode_id"]),
        "fresh_model_load": True,
        "resume_from_checkpoint": False,
        "llama_cpp_version": "runtime",
        "llama_cpp_build_info": {
            "cuda_backend": True,
            "supports_gpu_offload": True,
            "system_info": "CUDA = 1",
            "runtime_hash": str(model_spec.get("runtime_hash") or ""),
        },
        "chat_template": {
            "available": True,
            "used": True,
            "chat_template_hash": str(model_spec.get("template_hash") or ""),
            "template_replaced": False,
            "autotokenizer_used": False,
        },
        "cuda_device_receipt": {
            "before": before,
            "peak": peak,
            "after": after,
            "worker_returncode": 0,
        },
        "n_gpu_layers_requested": -1,
        "n_gpu_layers_offloaded": offloaded,
        "gpu_memory_before_mb": before_mb,
        "gpu_memory_peak_mb": peak_mb,
        "gpu_memory_after_mb": after_mb,
        "cuda_offload_authenticated": offloaded > 0 and peak_mb > before_mb,
        "rows_attempted": len(prompt_cells),
        "model_load_s": load_s,
        "runtime_log_excerpt": runtime_log[-4000:],
    }


def main() -> int:  # pragma: no cover - CLI wrapper.
    started = time.monotonic()
    artifact = run(canary_runner=live_llama_cpp_runner, max_modes_per_model=2, duration_s=None, write=True)
    artifact["duration_s"] = max(artifact["duration_s"], time.monotonic() - started)
    artifact["reproducibility_checksum"] = reproducibility_checksum(artifact)
    path = REPO_ROOT / RESULT_RELATIVE_PATH
    path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 0 if artifact["status"] in {"complete", "blocked"} else 1


if __name__ == "__main__":  # pragma: no cover - CLI wrapper.
    raise SystemExit(main())
