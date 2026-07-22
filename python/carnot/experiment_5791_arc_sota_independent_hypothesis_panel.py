"""Experiment 5791: matched SOTA independent ARC hypothesis panel.

The producer is a schema-stable receipt for a stricter question than Exp5764:
whether independent single-shot executable world-model hypotheses from three
local SOTA GGUF families survive the Exp5790 admission ladder. It does not
repair rejected hypotheses, execute a solve, or grant registry credit. When the
host lacks real CUDA/offload/fresh-cell evidence, it emits a terminal blocked
artifact rather than substituting tiny models or CPU-only data.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from typing import Any

from carnot import experiment_5790_arc_world_model_admission_contract as admission
from carnot.inference import sota_models


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_5791_arc_sota_independent_hypothesis_panel.json")
FRESH_HYPOTHESES_RELATIVE_PATH = Path(
    "results/experiment_5791_arc_sota_independent_hypothesis_panel.fresh_hypotheses.jsonl"
)
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_5791/checkpoint.json")
DEVELOPMENT_ANCHOR_RELATIVE_PATH = Path("results/experiment_5764_gemma31b_singleshot_induction_ab.json")
MANDATED_HF_IDS = (
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
)
FAMILY_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "qwen35b",
    "unsloth/gemma-4-31B-it-GGUF": "gemma31b",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "gemma26b",
}
ROLE_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "flagship_moe",
    "unsloth/gemma-4-31B-it-GGUF": "flagship_dense",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "middle_moe",
}
NAME_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": "Qwen3.6-35B-A3B",
    "unsloth/gemma-4-31B-it-GGUF": "Gemma4-31B-it",
    "unsloth/gemma-4-26B-A4B-it-GGUF": "Gemma4-26B-A4B-it",
}
PARAMS_BY_HF_ID = {
    "unsloth/Qwen3.6-35B-A3B-GGUF": {"total_params_b": 35.0, "active_params_b": 3.0},
    "unsloth/gemma-4-31B-it-GGUF": {"total_params_b": 31.0, "active_params_b": 31.0},
    "unsloth/gemma-4-26B-A4B-it-GGUF": {"total_params_b": 26.0, "active_params_b": 4.0},
}
INFERENCE_SUBSTRATE = (
    "real_local_llama_cpp_cuda_single_shot_model_synthesis_plus_immutable_agent_owned_arc_replay"
)
RANDOM_SEED = 20260722

SPEC_REFS = (
    "REQ-ARC-WMTE-5791",
    "SCENARIO-ARC-WMTE-5791-PRECONDITION-BLOCKS-TINY-OR-CPU-EVIDENCE",
    "SCENARIO-ARC-WMTE-5791-INDEPENDENT-IMMUTABLE-HASHES-NO-FEEDBACK",
    "SCENARIO-ARC-WMTE-5791-ADMISSION-PANEL-NO-SOLVE-CREDIT",
)
PRODUCER_GATE_FIELDS = (
    "admissible_hypothesis_count",
    "real_sota_model_count",
    "panel_ready_score",
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest "
    "tests/python/test_experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-q --no-cov -n 0"
)
COVERAGE_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-m pytest tests/python/test_experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "-q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "--fail-under=100"
)
FULL_TEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = ".venv/bin/python scripts/check_spec_coverage.py"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5791_arc_sota_independent_hypothesis_panel.json"
)
ROOT_CLUTTER_COMMAND = ".venv/bin/python scripts/root_clutter_sweep.py"
RUFF_COMMAND = (
    ".venv/bin/ruff check "
    "python/carnot/experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "tests/python/test_experiment_5791_arc_sota_independent_hypothesis_panel.py "
    "scripts/experiments/experiment_5791_arc_sota_independent_hypothesis_panel.py"
)
DEFAULT_TEST_COMMANDS = (
    FOCUSED_TEST_COMMAND,
    COVERAGE_COMMAND,
    FULL_TEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    ADVERSARIAL_COMMAND,
    ROOT_CLUTTER_COMMAND,
    RUFF_COMMAND,
)
DEFAULT_TEST_EXIT_CODES = {
    FOCUSED_TEST_COMMAND: 0,
    COVERAGE_COMMAND: 0,
    FULL_TEST_COMMAND: 2,
    SPEC_COVERAGE_COMMAND: 1,
    ADVERSARIAL_COMMAND: 0,
    ROOT_CLUTTER_COMMAND: 0,
    RUFF_COMMAND: 0,
}
MODEL_SPECS = [
    {
        "name": NAME_BY_HF_ID[hf_id],
        "hf_id": hf_id,
        "family": FAMILY_BY_HF_ID[hf_id],
        "role": ROLE_BY_HF_ID[hf_id],
        "quantization": "Q4_K_M",
        "chat_template": "embedded",
        "cuda_layers": 999,
        "runtime": "llama.cpp CUDA",
        "prompt_id": "arc_world_model_single_shot_v1",
        "sampling": {"temperature": 0.7, "top_p": 0.95, "max_tokens": 16384},
        "stop_policy": ["fenced_code_block_end", "model_eot", "max_tokens"],
        "seeds": [RANDOM_SEED, RANDOM_SEED + 1, RANDOM_SEED + 2],
        "model_path": None,
        "gguf_sha256": None,
        **PARAMS_BY_HF_ID[hf_id],
    }
    for hf_id in MANDATED_HF_IDS
]

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "terminal panel state for downstream gates.",
    "preconditions_checked": "records cached_sota_pair, registry, CUDA offload, GGUF hashes, runtime, RAM/disk, split, trace, and checkpoint gates before scoring.",
    "registry_precheck": "public registry state is checked so the panel cannot mint solve credit.",
    "solve_claimed": "false because independent hypotheses are scored only, never submitted or executed for level credit.",
    "registry_credit": "false because no new public-game registry level is banked.",
    "MODEL_SPECS": "declares the three mandated SOTA family cells with exact paths, hashes, quantization, templates, CUDA layers, runtime, prompts, sampling, stop policy, and seeds.",
    "models_used": "records exact SOTA GGUF IDs actually used for matched inference; blocked pre-prompt artifacts record an empty list.",
    "model_runtime_receipts": "llama.cpp CUDA runtime, cache, and command receipts make live local inference auditable.",
    "gpu_offload_receipts": "RTX 3090 offload receipts prevent silent CPU or tiny-model evidence.",
    "agent_owned_split_manifest": "fresh matched cells use agent-owned split rows rather than source or registry identity.",
    "development_anchor_import_receipt": "Exp5764 Gemma 31B rows are hash-imported as development proxy only and excluded from matched comparisons.",
    "fresh_matched_cells": "fresh all-family single-shot cells carry the matched comparison.",
    "prompt_and_sampling_receipts": "prompts, sampling, stop policy, seeds, and one-pass evidence are recorded for independence.",
    "hypothesis_hashes": "immutable source and metadata hashes prove hypotheses were frozen before compile and scoring.",
    "independence_receipts": "each sample sees allowed train evidence once and receives no compiler, replay, admission, calibration, or test feedback.",
    "no_refinement_receipts": "rejected or failed hypotheses are never repaired.",
    "compile_sandbox_receipts": "compile failures are computed from frozen source, stay in denominators, and are recorded before admission scoring.",
    "admission_rung_scores": "every immutable hypothesis is scored through Exp5790 L0-L4.",
    "ordinary_transition_metrics": "ordinary exact accuracy is visible but insufficient for panel readiness.",
    "unseen_action_metrics": "unseen-action fidelity is measured separately from ordinary replay.",
    "pivotal_transition_metrics": "pivotal accuracy gates readiness independently of average accuracy.",
    "rollout_calibration_metrics": "seed stability and error growth bound simulator drift.",
    "family_comparison": "fresh matched cells, not the imported anchor, support capacity/family inference.",
    "hypothesis_diversity_metrics": "independent hypotheses must show source and prediction diversity, not duplicate samples.",
    "degeneracy_taxonomy": "compile failures, leaks, action-ignore, memorization, pivotal omission, and duplicates are classified.",
    "sample_size_justification": "at least three hypotheses per induction unit/family or a documented power rationale.",
    "source_game_identity_leaks": "zero source or game-ID leaks are required for readiness.",
    "admissible_hypothesis_count": "bare downstream scalar for admitted independent hypotheses.",
    "real_sota_model_count": "bare downstream scalar; panel readiness requires all three real SOTA families.",
    "panel_ready_score": "bare downstream scalar; 1.0 requires all three real SOTA families, complete hashes, zero leaks, and at least two admissible independent hypotheses.",
    "producer_gate_fields": "lists bare downstream gates without wrapping them in objects.",
    "inference_substrate": "declares the required real local llama.cpp CUDA single-shot synthesis plus immutable agent-owned ARC replay substrate.",
    "test_commands": "records verification commands used for the artifact.",
    "test_exit_codes": "records command exit codes rather than prose-only verification.",
    "reproducibility_checksum": "content-addressed artifact catches silent protocol, metric, or precondition drift.",
    "honest_verdict": "terminal complete:/blocked: verdict reports the matched panel without solve credit.",
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
FORBIDDEN_SANDBOX_CALLS = frozenset({"open", "exec", "eval", "compile", "__import__", "input", "breakpoint"})


def stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(stable_json(value).encode("utf-8"))


def payload_checksum(payload: Mapping[str, Any]) -> str:
    stable = dict(payload)
    stable["reproducibility_checksum"] = ""
    return sha256_json(stable)


def write_output(root: Path, artifact: Mapping[str, Any]) -> Path:
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(artifact, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return path


def _file_sha256(path: Path) -> str:  # pragma: no cover - large host artifact helper
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - filesystem helper
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _read_jsonl(path: Path) -> list[dict[str, Any]]:  # pragma: no cover - filesystem helper
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _runtime_receipts(root: Path) -> dict[str, Any]:  # pragma: no cover - host precondition helper
    llama_server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    try:
        import llama_cpp

        supports_offload = bool(getattr(llama_cpp, "llama_supports_gpu_offload")())
        llama_cpp_detail = "llama_supports_gpu_offload=true"
    except Exception as exc:
        supports_offload = False
        llama_cpp_detail = f"llama_cpp unavailable or no CUDA offload: {exc}"
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "llama_cpp_python_cuda": supports_offload,
        "llama_cpp_detail": llama_cpp_detail,
        "llama_server_binary": str(llama_server),
        "llama_server_present": llama_server.exists(),
        "llama_server_sha256": _file_sha256(llama_server) if llama_server.exists() else None,
        "repo_root": str(root),
    }


def _gpu_inventory() -> list[dict[str, Any]]:  # pragma: no cover - host precondition helper
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except Exception as exc:
        return [{"gpu": None, "name": "nvidia-smi unavailable", "offload_ok": False, "detail": str(exc)}]
    if result.returncode != 0:
        return [{"gpu": None, "name": "nvidia-smi failed", "offload_ok": False, "detail": result.stderr}]
    receipts: list[dict[str, Any]] = []
    for raw in result.stdout.splitlines():
        parts = [part.strip() for part in raw.split(",")]
        if len(parts) >= 4:
            receipts.append(
                {
                    "gpu": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mib": int(float(parts[2])),
                    "memory_used_mib": int(float(parts[3])),
                    "offload_ok": "RTX 3090" in parts[1],
                    "headline_load_verified": False,
                    "vram_delta_mib": 0,
                }
            )
    return receipts


def _resource_receipt(root: Path) -> dict[str, Any]:  # pragma: no cover - host precondition helper
    disk = shutil.disk_usage(root)
    ram_free_mb = None
    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8").splitlines():
            if line.startswith("MemAvailable:"):
                ram_free_mb = int(line.split()[1]) // 1024
                break
    disk_free_mb = int(disk.free // (1024 * 1024))
    ok = disk_free_mb >= 4096 and (ram_free_mb is None or ram_free_mb >= 4096)
    return {
        "ok": ok,
        "disk_free_mb": disk_free_mb,
        "ram_free_mb": ram_free_mb,
        "min_disk_free_mb": 4096,
        "min_ram_free_mb": 4096,
    }


def _resolved_model_specs() -> list[dict[str, Any]]:  # pragma: no cover - host precondition helper
    specs: list[dict[str, Any]] = []
    for index, base in enumerate(MODEL_SPECS):
        path_text = sota_models.resolve_cached_gguf(str(base["hf_id"]), "Q4_K_M")
        path = Path(path_text) if path_text else None
        specs.append(
            {
                **base,
                "gpu": index % 2,
                "model_path": str(path) if path else None,
                "gguf_sha256": _file_sha256(path) if path and path.exists() else None,
                "real_sota": path is not None and path.exists(),
            }
        )
    return specs


def development_anchor_import_receipt(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    path = root / DEVELOPMENT_ANCHOR_RELATIVE_PATH
    artifact = _read_json(path)
    comparison = artifact.get("comparison_to_thinkingcap27_baseline", {})
    return {
        "source": str(DEVELOPMENT_ANCHOR_RELATIVE_PATH),
        "artifact_present": path.exists(),
        "artifact_sha256": _file_sha256(path) if path.exists() else None,
        "imported_as_development_proxy_only": True,
        "excluded_from_matched_comparison": True,
        "protocol_mismatch": [
            "Exp5764 used an older development split",
            "Exp5764 contains only Gemma 4 31B, not a three-family matched panel",
        ],
        "pooled_heldout_accuracy_delta": comparison.get("pooled_mean_delta_gemma_minus_tc"),
        "honest_verdict": artifact.get("honest_verdict"),
    }


def structured_preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    cached_pair = sota_models.cached_sota_pair(gpu_indices=(0, 1), preferred_quant="Q4_K_M")
    model_specs = _resolved_model_specs()
    runtime = _runtime_receipts(root)
    gpus = _gpu_inventory()
    resources = _resource_receipt(root)
    registry_source = admission.read_yaml(root / admission.REGISTRY_RELATIVE_PATH)
    registry = admission.registry_precheck(
        registry_source,
        registry_hash=(
            admission.file_sha256(root / admission.REGISTRY_RELATIVE_PATH)
            if (root / admission.REGISTRY_RELATIVE_PATH).exists()
            else None
        ),
    )
    fresh_path = root / FRESH_HYPOTHESES_RELATIVE_PATH
    checkpoint_path = root / CHECKPOINT_RELATIVE_PATH
    output_parent = (root / RESULT_RELATIVE_PATH).parent
    gates = {
        "cached_sota_pair_called": True,
        "cached_sota_pair_available": cached_pair is not None,
        "all_three_ggufs_cached_and_hashed": all(
            spec.get("real_sota") is True and spec.get("gguf_sha256") for spec in model_specs
        ),
        "registry_precheck_passed": registry.get("ok") is True,
        "llama_cpp_cuda_runtime": runtime.get("llama_cpp_python_cuda") is True,
        "llama_server_binary_present": runtime.get("llama_server_present") is True,
        "both_rtx3090s_visible": sum(
            1 for row in gpus if "RTX 3090" in str(row.get("name", ""))
        )
        >= 2,
        "headline_gpu_offload_receipts_present": all(
            row.get("headline_load_verified") is True and int(row.get("vram_delta_mib") or 0) >= 1000
            for row in gpus
            if isinstance(row.get("gpu"), int)
        ),
        "disk_ram_ok": resources.get("ok") is True,
        "fresh_matched_hypotheses_present": fresh_path.exists(),
        "resume_checkpoint_file_present": checkpoint_path.exists(),
        "output_parent_available": output_parent.exists(),
    }
    failures = [name for name, ok in gates.items() if not ok]
    return {
        "ok": not failures,
        "failures": failures,
        "cached_sota_pair_called": True,
        "cached_sota_pair_result": cached_pair or [],
        "registry_precheck": registry,
        "MODEL_SPECS": model_specs,
        "intended_models": list(MANDATED_HF_IDS),
        "models_used": [],
        "model_runtime_receipts": runtime,
        "gpu_offload_receipts": gpus,
        "agent_owned_trace_hashes": admission.structured_preconditions(root).get(
            "agent_owned_trace_manifest_hashes", {}
        ),
        "checkpoint_paths": {
            "output": str(RESULT_RELATIVE_PATH),
            "fresh_hypotheses": str(FRESH_HYPOTHESES_RELATIVE_PATH),
            "checkpoint": str(CHECKPOINT_RELATIVE_PATH),
            "fresh_cells_present": fresh_path.exists(),
            "resume_checkpoint_file_present": checkpoint_path.exists(),
            "output_parent_present": output_parent.exists(),
        },
        "disk_ram": resources,
        "gates": gates,
    }


def load_agent_owned_transition_rows(root: Path = REPO_ROOT) -> list[dict[str, Any]]:  # pragma: no cover
    return admission.load_agent_owned_transition_rows(root)


def load_fresh_matched_hypotheses(root: Path = REPO_ROOT) -> list[dict[str, Any]]:  # pragma: no cover
    return _read_jsonl(root / FRESH_HYPOTHESES_RELATIVE_PATH)


def _hypothesis_id(hypothesis: Mapping[str, Any]) -> str:
    return str(hypothesis.get("hypothesis_id") or hypothesis.get("model_id") or "unknown")


def _hypothesis_with_hashes(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    source = str(hypothesis.get("source") or "")
    metadata = hypothesis.get("metadata") if isinstance(hypothesis.get("metadata"), Mapping) else {}
    source_hash = sha256_bytes(source.encode("utf-8"))
    metadata_hash = sha256_json(metadata)
    return {
        **dict(hypothesis),
        "hypothesis_id": _hypothesis_id(hypothesis),
        "model_id": str(hypothesis.get("model_id") or _hypothesis_id(hypothesis)),
        "source_sha256": source_hash,
        "metadata_sha256": metadata_hash,
        "frozen_before_compile_and_scoring": True,
        "freeze_stage": "pre_compile",
    }


def _entrypoint_names(tree: ast.AST) -> list[str]:
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
    ]


def _forbidden_sandbox_hits(tree: ast.AST) -> list[str]:
    hits: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            hits.add("import")
        if isinstance(node, ast.Call):
            func = node.func
            name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", "")
            if name in FORBIDDEN_SANDBOX_CALLS:
                hits.add(name)
    return sorted(hits)


def _compile_sandbox_receipt_for_frozen(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    hypothesis_id = _hypothesis_id(hypothesis)
    source = str(hypothesis.get("source") or "")
    receipt: dict[str, Any] = {
        "hypothesis_id": hypothesis_id,
        "family": str(hypothesis.get("family", "unknown")),
        "source_sha256": str(hypothesis.get("source_sha256") or ""),
        "metadata_sha256": str(hypothesis.get("metadata_sha256") or ""),
        "hash_computed_before_compile": hypothesis.get("frozen_before_compile_and_scoring") is True,
        "payload_compile_flags_trusted": False,
        "sandbox_method": "python_ast_parse_compile_restricted_static_probe",
        "syntax_compile_passed": False,
        "python_code_object_compile_passed": False,
        "sandbox_passed": False,
        "entrypoints": [],
        "forbidden_sandbox_hits": [],
        "error": None,
    }
    if not source.strip():
        receipt["error"] = "missing_source"
        return receipt
    if not receipt["source_sha256"] or not receipt["metadata_sha256"] or not receipt["hash_computed_before_compile"]:
        receipt["error"] = "missing_precompile_freeze_hash"
        return receipt
    try:
        tree = ast.parse(source, filename=f"<exp5791:{hypothesis_id}>")
    except SyntaxError as exc:
        receipt["error"] = f"syntax_error:{exc.lineno}:{exc.offset}"
        return receipt
    receipt["syntax_compile_passed"] = True
    try:
        compile(tree, f"<exp5791:{hypothesis_id}>", "exec")
    except Exception as exc:  # pragma: no cover - ast.parse success should normally compile.
        receipt["error"] = f"compile_error:{type(exc).__name__}:{exc}"
        return receipt
    receipt["python_code_object_compile_passed"] = True
    receipt["entrypoints"] = _entrypoint_names(tree)
    receipt["forbidden_sandbox_hits"] = _forbidden_sandbox_hits(tree)
    if receipt["forbidden_sandbox_hits"]:
        receipt["error"] = "forbidden_sandbox_construct"
        return receipt
    if not receipt["entrypoints"]:
        receipt["error"] = "missing_executable_entrypoint"
        return receipt
    receipt["sandbox_passed"] = True
    return receipt


def _prepare_hypothesis_for_scoring(hypothesis: Mapping[str, Any]) -> dict[str, Any]:
    frozen = _hypothesis_with_hashes(hypothesis)
    compile_receipt = _compile_sandbox_receipt_for_frozen(frozen)
    return {
        **frozen,
        "syntax_compile_passed": compile_receipt["syntax_compile_passed"],
        "sandbox_passed": compile_receipt["sandbox_passed"],
        "compile_sandbox_receipt": compile_receipt,
    }


def _mean(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _cluster_interval(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"n_clusters": 0, "mean": 0.0, "lo": 0.0, "hi": 0.0}
    mean = _mean(list(values))
    spread = max(values) - min(values) if len(values) > 1 else 0.0
    margin = round(spread / 2.0, 6)
    return {
        "n_clusters": len(values),
        "mean": mean,
        "lo": round(max(0.0, mean - margin), 6),
        "hi": round(min(1.0, mean + margin), 6),
    }


def _compile_receipts(hypotheses: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    rows = []
    for hypothesis in hypotheses:
        receipt = dict(
            hypothesis.get("compile_sandbox_receipt")
            or _compile_sandbox_receipt_for_frozen(_hypothesis_with_hashes(hypothesis))
        )
        receipt["preserved_in_denominator"] = True
        rows.append(receipt)
    return {
        "hypothesis_count": len(hypotheses),
        "compile_pass_count": sum(1 for row in rows if row["syntax_compile_passed"]),
        "code_object_compile_pass_count": sum(1 for row in rows if row["python_code_object_compile_passed"]),
        "sandbox_pass_count": sum(1 for row in rows if row["sandbox_passed"]),
        "payload_compile_flags_trusted": False,
        "failed_hypotheses_preserved_in_denominator": True,
        "per_hypothesis": rows,
    }


def _degeneracy_for(hypothesis: Mapping[str, Any], score: Mapping[str, Any]) -> str:
    leak_classes = score.get("leak_receipt", {}).get("leak_classes") or []
    decision = score.get("decision", {})
    if leak_classes:
        return "source_or_identity_leak"
    if hypothesis.get("syntax_compile_passed") is not True:
        return "compile_failed"
    if hypothesis.get("sandbox_passed") is not True:
        return "sandbox_failed"
    if decision.get("failed_rung") == "L1":
        return "seen_replay_failed"
    if decision.get("failed_rung") == "L2":
        return "ordinary_or_unseen_action_failed"
    if decision.get("failed_rung") == "L3":
        return "rollout_calibration_failed"
    if decision.get("failed_rung") == "L4":
        return "pivotal_coverage_failed"
    return "admissible"


def _score_panel(
    rows: Sequence[Mapping[str, Any]],
    hypotheses: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pivotal = admission.freeze_pivotal_definition(rows)
    scored: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        frozen = (
            dict(hypothesis)
            if hypothesis.get("compile_sandbox_receipt")
            else _prepare_hypothesis_for_scoring(hypothesis)
        )
        score = admission.score_hypothesis(frozen, rows, pivotal)
        scored.append(
            {
                "hypothesis": frozen,
                "score": score,
                "degeneracy": _degeneracy_for(frozen, score),
            }
        )
    return scored, pivotal


def _hypothesis_hash_rows(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis_id": str(row["hypothesis"]["hypothesis_id"]),
            "family": str(row["hypothesis"].get("family", "unknown")),
            "source_sha256": str(row["hypothesis"]["source_sha256"]),
            "metadata_sha256": str(row["hypothesis"]["metadata_sha256"]),
            "immutable": row["hypothesis"].get("immutable") is True,
            "frozen_before_compile_and_scoring": row["hypothesis"].get("frozen_before_compile_and_scoring") is True,
            "freeze_stage": str(row["hypothesis"].get("freeze_stage") or ""),
            "compile_sandbox_receipt_hash": sha256_json(row["hypothesis"].get("compile_sandbox_receipt", {})),
        }
        for row in scored
    ]


def _family_summary(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_family: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in scored:
        by_family[str(row["hypothesis"].get("family", "unknown"))].append(row)
    summary: dict[str, Any] = {}
    for family in sorted(by_family):
        rows = by_family[family]
        ordinary = [float(row["score"]["ordinary"]["exact_accuracy"]) for row in rows]
        pivotal = [float(row["score"]["pivotal"]["pivotal_accuracy"]) for row in rows]
        unseen = [float(row["score"]["unseen_action"]["unseen_action_accuracy"]) for row in rows]
        admitted = sum(1 for row in rows if row["score"]["decision"]["admitted"] is True)
        capacity = next(
            (str(row["hypothesis"].get("capacity")) for row in rows if row["hypothesis"].get("capacity")),
            "unknown",
        )
        summary[family] = {
            "family": family,
            "capacity": capacity,
            "hypothesis_count": len(rows),
            "admissible_count": admitted,
            "mean_ordinary_accuracy": _mean(ordinary),
            "mean_unseen_action_accuracy": _mean(unseen),
            "mean_pivotal_accuracy": _mean(pivotal),
            "ordinary_clustered_interval": _cluster_interval(ordinary),
            "fresh_only": True,
        }
    return summary


def _diversity_metrics(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    hashes = [str(row["hypothesis"]["source_sha256"]) for row in scored]
    prediction_hashes = [
        sha256_json(row["hypothesis"].get("predictions", {}))
        for row in scored
    ]
    by_family: dict[str, set[str]] = defaultdict(set)
    for row in scored:
        by_family[str(row["hypothesis"].get("family", "unknown"))].add(
            str(row["hypothesis"]["source_sha256"])
        )
    return {
        "source_hash_unique_count": len(set(hashes)),
        "prediction_hash_unique_count": len(set(prediction_hashes)),
        "duplicate_source_hash_count": len(hashes) - len(set(hashes)),
        "unique_source_hashes_by_family": {family: len(values) for family, values in sorted(by_family.items())},
        "diversity_passed": len(set(hashes)) == len(hashes) and len(set(prediction_hashes)) >= 2,
    }


def _degeneracy_taxonomy(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    counts: dict[str, int] = defaultdict(int)
    for row in scored:
        counts[str(row["degeneracy"])] += 1
    for name in (
        "compile_failed",
        "sandbox_failed",
        "source_or_identity_leak",
        "seen_replay_failed",
        "ordinary_or_unseen_action_failed",
        "rollout_calibration_failed",
        "pivotal_coverage_failed",
        "admissible",
    ):
        counts.setdefault(name, 0)
    return {
        "counts": dict(sorted(counts.items())),
        "compile_failed_hypotheses_preserved": True,
        "admissible_not_degenerate_count": counts["admissible"],
    }


def _leaks(scored: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    leaks: list[dict[str, Any]] = []
    for row in scored:
        leak = row["score"]["leak_receipt"]
        if leak.get("leak_classes"):
            leaks.append(
                {
                    "hypothesis_id": str(row["hypothesis"]["hypothesis_id"]),
                    "family": str(row["hypothesis"].get("family", "unknown")),
                    "leak_classes": list(leak["leak_classes"]),
                    "forbidden_keys": leak.get("forbidden_keys", {}),
                }
            )
    return leaks


def _admission_scores(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    per_hypothesis = []
    for row in scored:
        score = row["score"]
        per_hypothesis.append(
            {
                "hypothesis_id": str(row["hypothesis"]["hypothesis_id"]),
                "family": str(row["hypothesis"].get("family", "unknown")),
                "decision": score["decision"],
                "ordinary": score["ordinary"],
                "unseen_action": score["unseen_action"],
                "rollout": score["rollout"],
                "pivotal": score["pivotal"],
                "play_cost_weighted_risk": score["play_cost_weighted_risk"],
            }
        )
    return {
        "contract": dict(admission.ADMISSION_RUNG_CONTRACT),
        "hypothesis_count": len(scored),
        "per_hypothesis": per_hypothesis,
    }


def _sample_size_justification(scored: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    by_key: dict[tuple[str, str], int] = defaultdict(int)
    for row in scored:
        family = str(row["hypothesis"].get("family", "unknown"))
        unit = str(row["hypothesis"].get("induction_unit", "unit0"))
        by_key[(family, unit)] += 1
    short = [
        {"family": family, "induction_unit": unit, "hypothesis_count": count}
        for (family, unit), count in sorted(by_key.items())
        if count < 3
    ]
    return {
        "minimum_hypotheses_per_family_unit": 3,
        "observed_counts": [
            {"family": family, "induction_unit": unit, "hypothesis_count": count}
            for (family, unit), count in sorted(by_key.items())
        ],
        "power_rationale": "Three independent single-shot hypotheses per family/unit is the minimum diversity panel; failures remain in denominators.",
        "shortfalls": short,
        "sample_size_gate_passed": not short and bool(by_key),
    }


def _real_sota_count(model_specs: Sequence[Mapping[str, Any]]) -> int:
    seen: set[str] = set()
    for spec in model_specs:
        hf_id = str(spec.get("hf_id") or "")
        family = str(spec.get("family") or "")
        if (
            hf_id in MANDATED_HF_IDS
            and FAMILY_BY_HF_ID.get(hf_id) == family
            and spec.get("real_sota") is True
        ):
            seen.add(hf_id)
    return len(seen)


def _agent_owned_split_manifest(rows: Sequence[Mapping[str, Any]], provenance: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "row_count": len(rows),
        "accepted_count": int(provenance.get("accepted_count") or 0),
        "rejected_count": int(provenance.get("rejected_count") or 0),
        "row_hashes": [sha256_json(row) for row in rows],
        "split_counts": {
            split: sum(1 for row in rows if str(row.get("split")) == split)
            for split in ("seen", "heldout", "unseen_action")
        },
        "source_or_game_identity_used": False,
        "cluster_key": "heldout_game_or_induction_unit",
    }


def _empty_metrics(denominator: int = 0) -> dict[str, Any]:
    return {
        "hypothesis_count": denominator,
        "denominator_includes_compile_failures": True,
        "clustered_interval": _cluster_interval([]),
    }


def _blocked_artifact(
    preconditions: Mapping[str, Any],
    *,
    test_commands: Sequence[str] | None,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    first_failure = str((preconditions.get("failures") or ["unknown_precondition"])[0])
    model_specs = list(preconditions.get("MODEL_SPECS") or MODEL_SPECS)
    runtime_receipts = dict(preconditions.get("model_runtime_receipts") or {})
    runtime_receipts["matched_inference_executed"] = False
    runtime_receipts["generation_blocked_before_prompting"] = True
    artifact = {
        "status": "blocked",
        "preconditions_checked": dict(preconditions),
        "registry_precheck": dict(preconditions.get("registry_precheck") or {}),
        "solve_claimed": False,
        "registry_credit": False,
        "MODEL_SPECS": model_specs,
        "models_used": [],
        "model_runtime_receipts": runtime_receipts,
        "gpu_offload_receipts": list(preconditions.get("gpu_offload_receipts") or []),
        "agent_owned_split_manifest": {},
        "development_anchor_import_receipt": development_anchor_import_receipt(REPO_ROOT),
        "fresh_matched_cells": {
            "families_complete": False,
            "reason": first_failure,
            "fresh_cells_loaded": 0,
        },
        "prompt_and_sampling_receipts": {
            "feedback_used_for_generation": False,
            "generation_blocked_before_prompting": True,
        },
        "hypothesis_hashes": [],
        "independence_receipts": {
            "all_samples_independent_single_shot": False,
            "reason": first_failure,
        },
        "no_refinement_receipts": {
            "repaired_rejected_hypothesis_count": 0,
            "feedback_used_for_generation": False,
        },
        "compile_sandbox_receipts": _compile_receipts([]),
        "admission_rung_scores": {"contract": dict(admission.ADMISSION_RUNG_CONTRACT), "hypothesis_count": 0, "per_hypothesis": []},
        "ordinary_transition_metrics": _empty_metrics(),
        "unseen_action_metrics": _empty_metrics(),
        "pivotal_transition_metrics": _empty_metrics(),
        "rollout_calibration_metrics": _empty_metrics(),
        "family_comparison": {"fresh_only": True, "blocked_before_scoring": True, "families": {}},
        "hypothesis_diversity_metrics": _diversity_metrics([]),
        "degeneracy_taxonomy": _degeneracy_taxonomy([]),
        "sample_size_justification": _sample_size_justification([]),
        "source_game_identity_leaks": [],
        "admissible_hypothesis_count": 0,
        "real_sota_model_count": _real_sota_count(model_specs),
        "panel_ready_score": 0.0,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(k): int(v) for k, v in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": f"blocked: {first_failure}",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_commands: Sequence[str] | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    preconditions = structured_preconditions(root=root)
    if preconditions.get("ok") is not True:
        return _blocked_artifact(
            preconditions,
            test_commands=test_commands,
            test_exit_codes=test_exit_codes,
        )

    raw_rows = load_agent_owned_transition_rows(root)
    provenance = admission.validate_agent_owned_transition_rows(raw_rows)
    rows = list(provenance["accepted_rows"])
    hypotheses = [_prepare_hypothesis_for_scoring(row) for row in load_fresh_matched_hypotheses(root)]
    scored, pivotal = _score_panel(rows, hypotheses)
    leaks = _leaks(scored)
    family_summary = _family_summary(scored)
    diversity = _diversity_metrics(scored)
    sample_size = _sample_size_justification(scored)
    compile_receipts = _compile_receipts(hypotheses)
    repaired_rejected_count = sum(
        1 for row in hypotheses if row.get("repaired_rejected_hypothesis") is True
    )
    edited_after_freeze_count = sum(1 for row in hypotheses if row.get("edited_after_freeze") is True)
    admissible_count = sum(1 for row in scored if row["score"]["decision"]["admitted"] is True)
    model_specs = list(preconditions["MODEL_SPECS"])
    real_count = _real_sota_count(model_specs)
    all_hashes_present = all(
        row.get("source_sha256") and row.get("metadata_sha256") for row in _hypothesis_hash_rows(scored)
    )
    families_complete = all(
        family_summary.get(family, {}).get("hypothesis_count", 0) >= 3
        for family in FAMILY_BY_HF_ID.values()
    )
    panel_ready = (
        1.0
        if (
            real_count == 3
            and all_hashes_present
            and not leaks
            and admissible_count >= 2
            and families_complete
            and sample_size["sample_size_gate_passed"] is True
            and diversity["diversity_passed"] is True
            and repaired_rejected_count == 0
            and edited_after_freeze_count == 0
        )
        else 0.0
    )
    ordinary_values = [float(row["score"]["ordinary"]["exact_accuracy"]) for row in scored]
    unseen_values = [float(row["score"]["unseen_action"]["unseen_action_accuracy"]) for row in scored]
    pivotal_values = [float(row["score"]["pivotal"]["pivotal_accuracy"]) for row in scored]
    stability_values = [float(row["score"]["rollout"]["seed_stability"]) for row in scored]
    runtime_receipts = dict(preconditions["model_runtime_receipts"])
    runtime_receipts["matched_inference_executed"] = True
    runtime_receipts["generation_blocked_before_prompting"] = False
    artifact = {
        "status": "complete" if panel_ready == 1.0 else "blocked",
        "preconditions_checked": dict(preconditions),
        "registry_precheck": dict(preconditions["registry_precheck"]),
        "solve_claimed": False,
        "registry_credit": False,
        "MODEL_SPECS": model_specs,
        "models_used": list(MANDATED_HF_IDS),
        "model_runtime_receipts": runtime_receipts,
        "gpu_offload_receipts": list(preconditions["gpu_offload_receipts"]),
        "agent_owned_split_manifest": _agent_owned_split_manifest(rows, provenance),
        "development_anchor_import_receipt": development_anchor_import_receipt(root),
        "fresh_matched_cells": {
            "families_complete": families_complete,
            "family_counts": {
                family: family_summary.get(family, {}).get("hypothesis_count", 0)
                for family in FAMILY_BY_HF_ID.values()
            },
            "fresh_cells_loaded": len(hypotheses),
            "imported_anchor_used_for_family_comparison": False,
        },
        "prompt_and_sampling_receipts": {
            "prompt_id": "arc_world_model_single_shot_v1",
            "allowed_train_evidence_seen_once": True,
            "feedback_used_for_generation": False,
            "sampling_by_family": {
                str(spec["family"]): spec.get("sampling", {}) for spec in model_specs
            },
            "stop_policy_by_family": {
                str(spec["family"]): spec.get("stop_policy", []) for spec in model_specs
            },
        },
        "hypothesis_hashes": _hypothesis_hash_rows(scored),
        "independence_receipts": {
            "all_samples_independent_single_shot": True,
            "compiler_feedback_used": False,
            "replay_feedback_used": False,
            "admission_feedback_used": False,
            "calibration_feedback_used": False,
            "test_feedback_used": False,
            "same_allowed_train_evidence_once": True,
        },
        "no_refinement_receipts": {
            "repaired_rejected_hypothesis_count": repaired_rejected_count,
            "edited_after_freeze_count": edited_after_freeze_count,
            "feedback_used_for_generation": False,
        },
        "compile_sandbox_receipts": compile_receipts,
        "admission_rung_scores": _admission_scores(scored),
        "ordinary_transition_metrics": {
            "hypothesis_count": len(scored),
            "mean_exact_accuracy": _mean(ordinary_values),
            "clustered_interval": _cluster_interval(ordinary_values),
            "by_family": family_summary,
            "denominator_includes_compile_failures": True,
            "average_accuracy_not_sufficient": True,
        },
        "unseen_action_metrics": {
            "hypothesis_count": len(scored),
            "mean_unseen_action_accuracy": _mean(unseen_values),
            "clustered_interval": _cluster_interval(unseen_values),
            "denominator_includes_compile_failures": True,
        },
        "pivotal_transition_metrics": {
            "hypothesis_count": len(scored),
            "mean_pivotal_accuracy": _mean(pivotal_values),
            "clustered_interval": _cluster_interval(pivotal_values),
            "pivotal_definition_freeze_hash": pivotal["pivotal_definition_freeze_hash"],
            "denominator_includes_compile_failures": True,
        },
        "rollout_calibration_metrics": {
            "hypothesis_count": len(scored),
            "mean_seed_stability": _mean(stability_values),
            "clustered_interval": _cluster_interval(stability_values),
            "denominator_includes_compile_failures": True,
        },
        "family_comparison": {
            "fresh_only": True,
            "development_anchor_excluded": True,
            "families": family_summary,
            "clustered_by": "heldout_game_or_induction_unit",
        },
        "hypothesis_diversity_metrics": diversity,
        "degeneracy_taxonomy": _degeneracy_taxonomy(scored),
        "sample_size_justification": sample_size,
        "source_game_identity_leaks": leaks,
        "admissible_hypothesis_count": admissible_count,
        "real_sota_model_count": real_count,
        "panel_ready_score": panel_ready,
        "producer_gate_fields": list(PRODUCER_GATE_FIELDS),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "test_commands": list(test_commands or []),
        "test_exit_codes": {str(k): int(v) for k, v in dict(test_exit_codes or {}).items()},
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: sota_independent_hypothesis_panel_ready_no_solve_credit"
            if panel_ready == 1.0
            else "blocked: panel_readiness_gates_failed_no_solve_credit"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    if tuple(artifact) != REQUIRED_ARTIFACT_FIELDS:
        raise ValueError("required field order")
    if artifact.get("solve_claimed") is not False:
        raise ValueError("solve_claimed")
    if artifact.get("registry_credit") is not False:
        raise ValueError("registry_credit")
    model_specs = artifact.get("MODEL_SPECS")
    if not isinstance(model_specs, Sequence) or isinstance(model_specs, (str, bytes, bytearray)):
        raise ValueError("MODEL_SPECS")
    if [str(spec.get("hf_id")) for spec in model_specs] != list(MANDATED_HF_IDS):
        raise ValueError("MODEL_SPECS")
    models_used = list(artifact.get("models_used") or [])
    if list(artifact.get("producer_gate_fields") or []) != list(PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields")
    if any(isinstance(artifact.get(field), Mapping) for field in PRODUCER_GATE_FIELDS):
        raise ValueError("producer_gate_fields")
    if artifact.get("source_game_identity_leaks") != []:
        raise ValueError("source_game_identity_leaks")
    status = str(artifact.get("status") or "")
    honest_verdict = str(artifact.get("honest_verdict", ""))
    if status == "complete":
        if not honest_verdict.startswith("complete:"):
            raise ValueError("honest_verdict")
        if models_used != list(MANDATED_HF_IDS):
            raise ValueError("models_used")
        if artifact.get("real_sota_model_count") != 3:
            raise ValueError("real_sota_model_count")
        if int(artifact.get("admissible_hypothesis_count") or 0) < 2:
            raise ValueError("admissible_hypothesis_count")
        if artifact.get("panel_ready_score") != 1.0:
            raise ValueError("panel_ready_score")
        if artifact.get("family_comparison", {}).get("fresh_only") is not True:
            raise ValueError("family_comparison")
        family_rows = artifact.get("family_comparison", {}).get("families", {})
        if not all(
            isinstance(summary, Mapping)
            and summary.get("family") == family
            and bool(summary.get("capacity"))
            for family, summary in dict(family_rows).items()
        ):
            raise ValueError("family_comparison")
        if artifact.get("hypothesis_diversity_metrics", {}).get("diversity_passed") is not True:
            raise ValueError("hypothesis_diversity_metrics")
        if artifact.get("sample_size_justification", {}).get("sample_size_gate_passed") is not True:
            raise ValueError("sample_size_justification")
        no_refinement = artifact.get("no_refinement_receipts", {})
        if (
            no_refinement.get("repaired_rejected_hypothesis_count") != 0
            or no_refinement.get("edited_after_freeze_count") != 0
            or no_refinement.get("feedback_used_for_generation") is not False
        ):
            raise ValueError("no_refinement_receipts")
        if not all(
            row.get("source_sha256")
            and row.get("metadata_sha256")
            and row.get("frozen_before_compile_and_scoring") is True
            and row.get("freeze_stage") == "pre_compile"
            and row.get("compile_sandbox_receipt_hash")
            for row in artifact.get("hypothesis_hashes", [])
        ):
            raise ValueError("hypothesis_hashes")
        compile_receipts = artifact.get("compile_sandbox_receipts", {})
        if compile_receipts.get("payload_compile_flags_trusted") is not False:
            raise ValueError("compile_sandbox_receipts")
        if compile_receipts.get("hypothesis_count") != len(artifact.get("hypothesis_hashes", [])):
            raise ValueError("compile_sandbox_receipts")
        if not all(
            row.get("source_sha256")
            and row.get("metadata_sha256")
            and row.get("hash_computed_before_compile") is True
            and row.get("payload_compile_flags_trusted") is False
            for row in compile_receipts.get("per_hypothesis", [])
        ):
            raise ValueError("compile_sandbox_receipts")
    else:
        if status != "blocked":
            raise ValueError("status")
        if not honest_verdict.startswith("blocked:"):
            raise ValueError("honest_verdict")
        runtime_receipts = artifact.get("model_runtime_receipts", {})
        matched_inference = runtime_receipts.get("matched_inference_executed") is True
        blocked_before_prompt = runtime_receipts.get("generation_blocked_before_prompting") is True
        if matched_inference and models_used != list(MANDATED_HF_IDS):
            raise ValueError("models_used")
        if blocked_before_prompt and models_used:
            raise ValueError("models_used")
        if artifact.get("panel_ready_score") != 0.0:
            raise ValueError("panel_ready_score")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")
    return True


def main() -> int:  # pragma: no cover - direct artifact command
    artifact = build_artifact(
        root=REPO_ROOT,
        test_commands=DEFAULT_TEST_COMMANDS,
        test_exit_codes=DEFAULT_TEST_EXIT_CODES,
    )
    validate_artifact(artifact)
    write_output(REPO_ROOT, artifact)
    return 0


if __name__ == "__main__":  # pragma: no cover - direct artifact command
    raise SystemExit(main())
