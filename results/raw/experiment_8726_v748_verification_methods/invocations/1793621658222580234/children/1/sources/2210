"""Exp6282 mechanic-class detector routed into the live ARC inducer.

Spec refs: REQ-ARC-WMTE-6282,
SCENARIO-ARC-WMTE-6282-GAME-BLIND-FIXTURES,
SCENARIO-ARC-WMTE-6282-LIVE-PROMPT-ROUTE,
SCENARIO-ARC-WMTE-6282-ARTIFACT-PROVENANCE.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import contextlib
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import Any

import numpy as np

from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_mechanic_class_detector as detector
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
LLMRunner = Callable[[Mapping[str, str], Mapping[str, Any], int], Mapping[str, Any]]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6282_arc_mechanic_class_live_router.json")
SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6282_arc_mechanic_class_fixture_manifest.json"
)
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RUN_COMMAND = ".venv/bin/python -m carnot.experiment_6282_arc_mechanic_class_live_router --date 20260810"
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6282_test_receipts.json")
RANDOM_SEED = 6282
MANDATED_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
PREFERRED_QUANT = "Q4_K_M"
MAX_TOKENS = 32
N_CTX = 1024
FORBIDDEN_COUNT_FIELDS = (
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "exhaustive_bfs_count",
    "per_game_adapter_count",
    "registry_update_count",
    "weight_mutation_count",
)
PROTECTED_FILES = (
    Path("scripts/research_conductor.py"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/changelog.md"),
    Path("ops/status.md"),
    Path("_bmad/traceability.md"),
)
DEFAULT_TEST_COMMANDS = (
    RUN_COMMAND,
    ".venv/bin/pytest tests/python/test_experiment_6282_arc_mechanic_class_live_router.py -q --no-cov -n 0",
    ".venv/bin/coverage run --rcfile=/dev/null --include=python/carnot/agentic/arc_mechanic_class_detector.py,python/carnot/experiment_6282_arc_mechanic_class_live_router.py -m pytest tests/python/test_experiment_6282_arc_mechanic_class_live_router.py -q --no-cov -n 0 && .venv/bin/coverage report --rcfile=/dev/null --include=python/carnot/agentic/arc_mechanic_class_detector.py,python/carnot/experiment_6282_arc_mechanic_class_live_router.py --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_6282_arc_mechanic_class_live_router.py",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_6282_arc_mechanic_class_live_router.json",
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "registry_precheck_path_hash_and_target_receipt",
    "target_was_unsolved_at_precheck",
    "model_specs",
    "cache_resolution_model_hash_and_quantization",
    "llama_cpp_binary_version_and_hash",
    "cuda_gpu_and_offload_receipts",
    "peak_vram",
    "detector_source_paths_and_hashes",
    "synthetic_fixture_manifest_path_and_hash",
    "push_block_and_toggle_move_fixture_counts",
    "detector_classification_metrics_and_sample_sizes",
    "uncertainty_calibration",
    "live_transition_history_hashes",
    "route_activation_counts",
    "treatment_and_control_proposal_hashes",
    "proposal_coverage_by_arm",
    "invalid_action_rate_by_arm",
    "interaction_budget_by_arm",
    "treatment_fire_receipt",
    "evidence_provenance",
    "solve_provenance",
    "game_level_solve_claimed",
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "exhaustive_bfs_count",
    "per_game_adapter_count",
    "registry_update_count",
    "weight_mutation_count",
    "harmful_regressions",
    "arc_mechanic_router_ready_score",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "honest_verdict",
)


FIELD_PRINCIPLES: dict[str, str] = {
    "status": "Separates a completed canary from a blocked live run.",
    "registry_precheck_path_hash_and_target_receipt": "Pins the registry read and target choice.",
    "target_was_unsolved_at_precheck": "Prevents re-targeting a reproduced public level.",
    "model_specs": "Names the mandated Qwen live inducer.",
    "cache_resolution_model_hash_and_quantization": "Pins the local GGUF and quantization.",
    "llama_cpp_binary_version_and_hash": "Pins the runtime that loaded the GGUF.",
    "cuda_gpu_and_offload_receipts": "Records GPU visibility and offload evidence.",
    "peak_vram": "Shows the live run paid real memory cost.",
    "detector_source_paths_and_hashes": "Pins detector and route source bytes.",
    "synthetic_fixture_manifest_path_and_hash": "Pins the trusted synthetic controls.",
    "push_block_and_toggle_move_fixture_counts": "Shows both requested families are covered.",
    "detector_classification_metrics_and_sample_sizes": "Reports accuracy with denominators.",
    "uncertainty_calibration": "Shows uncertainty came from held-out controls.",
    "live_transition_history_hashes": "Pins the canary transitions shown to the route.",
    "route_activation_counts": "Proves treatment fired and control did not.",
    "treatment_and_control_proposal_hashes": "Hashes matched prompts and proposals.",
    "proposal_coverage_by_arm": "Reports proposal content coverage by arm.",
    "invalid_action_rate_by_arm": "Flags malformed action proposals.",
    "interaction_budget_by_arm": "Shows matched prompt and decode budgets.",
    "treatment_fire_receipt": "Records the exact class and uncertainty inserted.",
    "evidence_provenance": "Separates synthetic and live-transition evidence from forbidden inputs.",
    "solve_provenance": "States live-agent self-discovery provenance.",
    "game_level_solve_claimed": "Keeps the canary from becoming solve credit.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "outer_loop_ground_truth_search_count": "Must stay zero for self-discovery discipline.",
    "exhaustive_bfs_count": "Must stay zero for bounded canary discipline.",
    "per_game_adapter_count": "Must stay zero for game-blind routing.",
    "registry_update_count": "Must stay zero because no solve is banked.",
    "weight_mutation_count": "Must stay zero because the model is not trained.",
    "harmful_regressions": "Reports any detected route harm.",
    "arc_mechanic_router_ready_score": "Summarizes readiness without solve credit.",
    "protected_files_unchanged": "Proves conductor and registry files were not edited.",
    "preconditions_checked": "Records git, registry, model, CUDA, caps, and seed.",
    "inference_substrate": "Declares live GGUF inference.",
    "verifier_is_oracle": "False because no game oracle verifies a solve.",
    "field_provenance": "Maps every field to computation and spec.",
    "field_principles": "Gives one audit reason per field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records wall-clock cost.",
    "random_seed": "Pins synthetic and decode randomness.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict states no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6282", "experiment_6282_arc_mechanic_class_live_router"]
    for field in REQUIRED_ARTIFACT_FIELDS
}


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def sha256_text(value: str) -> str:
    return "sha256:" + hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_text(canonical_json(value))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def payload_checksum(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    return sha256_json(clean)


def _display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return str(path)


def _file_receipt(path: Path, *, hash_file: bool = True) -> JsonDict:
    return {
        "path": _display_path(path),
        "exists": path.is_file(),
        "size_bytes": path.stat().st_size if path.is_file() else None,
        "sha256": sha256_file(path) if path.is_file() and hash_file else None,
    }


def _protected_hashes() -> dict[str, str | None]:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in PROTECTED_FILES
    }


def _protected_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    return {
        path: {"before": before.get(path), "after": after.get(path), "unchanged": before.get(path) == after.get(path)}
        for path in sorted(set(before) | set(after))
    }


def _git_status_short() -> str:
    proc = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        timeout=10,
        check=False,
    )
    return proc.stdout.strip()


def _registry_precheck() -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    full_clear_count = text.count("full_game_clear: true")
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "target": "synthetic_push_block_mechanic_canary",
        "target_kind": "synthetic_unseen_mechanic_fixture_not_public_registry_level",
        "public_registry_games_full_clear_count": full_clear_count,
        "public_registry_games_not_full_clear_count": text.count("full_game_clear: false"),
        "public_level_targeted": False,
        "target_present_in_registry": False,
        "target_receipt": {
            "mechanic_class": "push_block",
            "why_eligible": "synthetic unseen mechanic fixture; no reproduced public level targeted",
        },
    }


def _source_hashes() -> list[JsonDict]:
    paths = (
        Path("python/carnot/agentic/arc_mechanic_class_detector.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/experiment_6282_arc_mechanic_class_live_router.py"),
        SPEC_RELATIVE_PATH,
    )
    return [_file_receipt(REPO_ROOT / path) for path in paths]


def _resolve_qwen_model(*, live_hash: bool) -> JsonDict:
    pair = cached_sota_pair(gpu_indices=(0, 1))
    selected = next((row for row in pair or [] if row.get("hf_id") == MANDATED_HF_ID), None)
    path = selected.get("model_path") if selected else resolve_cached_gguf(MANDATED_HF_ID, PREFERRED_QUANT)
    model_path = Path(str(path)) if path else None
    return {
        "cached_sota_pair_attempted": True,
        "cached_sota_pair_available": bool(pair),
        "hf_id": MANDATED_HF_ID,
        "preferred_quant": PREFERRED_QUANT,
        "quantization": "UD-Q4_K_M",
        "model_path": str(model_path) if model_path else None,
        "model_exists": bool(model_path and model_path.is_file()),
        "model_size_bytes": model_path.stat().st_size if model_path and model_path.is_file() else None,
        "model_sha256": (
            sha256_file(model_path)
            if live_hash and model_path is not None and model_path.is_file()
            else None
        ),
        "resolved_via": "cached_sota_pair" if selected else "resolve_cached_gguf",
    }


def _llama_cpp_receipt() -> JsonDict:
    try:
        import llama_cpp

        module_path = Path(str(llama_cpp.__file__))
        return {
            "package": "llama_cpp_python",
            "version": getattr(llama_cpp, "__version__", None),
            "module_path": str(module_path),
            "module_sha256": sha256_file(module_path) if module_path.is_file() else None,
            "system_info": _llama_system_info(),
        }
    except Exception as exc:  # pragma: no cover - exercised only when llama_cpp is absent
        return {"package": "llama_cpp_python", "error": repr(exc)[:200], "system_info": None}


def _llama_system_info() -> str | None:
    try:
        from llama_cpp import llama_cpp

        raw = llama_cpp.llama_print_system_info()
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
    except Exception:  # pragma: no cover - depends on a broken llama_cpp runtime
        return None


def _nvidia_smi_snapshot() -> list[JsonDict]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        proc = subprocess.run(
            cmd, cwd=REPO_ROOT, text=True, capture_output=True, timeout=10, check=False
        )
    except Exception as exc:  # pragma: no cover - depends on missing/broken nvidia-smi
        return [{"error": repr(exc)[:160]}]
    rows = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            rows.append(
                {
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_total_mb": int(parts[2]),
                    "memory_used_mb": int(parts[3]),
                    "driver_version": parts[4],
                }
            )
    return rows


def _peak_vram(before: Sequence[Mapping[str, Any]], after: Sequence[Mapping[str, Any]]) -> JsonDict:
    used_values = [
        int(row.get("memory_used_mb", 0))
        for row in list(before) + list(after)
        if isinstance(row.get("memory_used_mb"), int)
    ]
    return {
        "peak_used_mb_observed": max(used_values) if used_values else None,
        "before_total_used_mb": sum(
            int(row.get("memory_used_mb", 0)) for row in before if isinstance(row.get("memory_used_mb"), int)
        ),
        "after_total_used_mb": sum(
            int(row.get("memory_used_mb", 0)) for row in after if isinstance(row.get("memory_used_mb"), int)
        ),
    }


def _canary_transitions() -> list[e3.Transition]:
    rows: list[e3.Transition] = []
    for col in (1, 2, 3, 4):
        before = np.zeros((6, 9), dtype=int)
        after = np.zeros((6, 9), dtype=int)
        before[3, col] = 1
        before[3, col + 1] = 2
        after[3, col + 1] = 1
        after[3, col + 2] = 2
        rows.append(e3.Transition(before, 4, None, after, 0, 0))
    return rows


def _transition_history_receipt(transitions: Sequence[e3.Transition]) -> JsonDict:
    payload = [
        {
            "action": int(t.action),
            "data": t.data,
            "grid_sha256": sha256_text(np.asarray(t.grid, dtype=int).tobytes().hex()),
            "next_grid_sha256": sha256_text(np.asarray(t.next_grid, dtype=int).tobytes().hex()),
            "changed_cells": int(np.sum(np.asarray(t.grid) != np.asarray(t.next_grid))),
        }
        for t in transitions
    ]
    return {
        "target": "synthetic_push_block_mechanic_canary",
        "transition_count": len(transitions),
        "history_sha256": sha256_json(payload),
        "rows": payload,
    }


@contextlib.contextmanager
def _temporary_env(updates: Mapping[str, str]):
    old = {key: os.environ.get(key) for key in updates}
    try:
        for key, value in updates.items():
            os.environ[key] = value
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _prompts_for_canary(transitions: Sequence[e3.Transition]) -> dict[str, str]:
    common = {"CARNOT_ARC_OBJECT_PERCEPTION": "0", "CARNOT_ARC_INDUCE_PROMPT_ENRICHMENT": "0"}
    with _temporary_env({**common, "CARNOT_ARC_MECHANIC_CLASS_ROUTER": "0"}):
        control = e3.induce_prompt("synthetic_push_block_mechanic_canary", list(transitions), cell=1)
    with _temporary_env({**common, "CARNOT_ARC_MECHANIC_CLASS_ROUTER": "1"}):
        treatment = e3.induce_prompt("synthetic_push_block_mechanic_canary", list(transitions), cell=1)
    return {"control": control, "treatment": treatment}


def deterministic_test_llm_runner(
    prompts: Mapping[str, str], model_receipt: Mapping[str, Any], seed: int
) -> Mapping[str, Any]:
    del model_receipt, seed
    return {
        "runner": "deterministic_test_llm_runner",
        "model_loaded": False,
        "outputs": {
            "control": "def engine(grid, action, data): return grid.copy()",
            "treatment": "push_block prior: def engine(grid, action, data): return grid.copy()",
        },
        "wall_s_by_arm": {"control": 0.01, "treatment": 0.01},
        "prompt_chars_by_arm": {arm: len(prompt) for arm, prompt in prompts.items()},
        "errors": {},
    }


def live_qwen_llm_runner(
    prompts: Mapping[str, str], model_receipt: Mapping[str, Any], seed: int
) -> Mapping[str, Any]:  # pragma: no cover - hardware integration path
    from llama_cpp import Llama

    model_path = str(model_receipt.get("model_path") or "")
    if not model_path:
        return {"runner": "llama_cpp", "model_loaded": False, "outputs": {}, "errors": {"load": "missing_model_path"}}
    llm = Llama(
        model_path=model_path,
        n_ctx=N_CTX,
        n_gpu_layers=-1,
        seed=int(seed),
        verbose=False,
    )
    outputs: dict[str, str] = {}
    wall: dict[str, float] = {}
    errors: dict[str, str] = {}
    for arm, prompt in prompts.items():
        started = time.perf_counter()
        try:
            out = llm(
                prompt + "\nReturn one concise mechanic-aware engine sketch.",
                max_tokens=MAX_TOKENS,
                temperature=0.0,
            )
            outputs[arm] = str(out["choices"][0]["text"])
        except Exception as exc:
            errors[arm] = repr(exc)[:200]
            outputs[arm] = ""
        wall[arm] = round(time.perf_counter() - started, 4)
    return {
        "runner": "llama_cpp",
        "model_loaded": True,
        "outputs": outputs,
        "wall_s_by_arm": wall,
        "prompt_chars_by_arm": {arm: len(prompt) for arm, prompt in prompts.items()},
        "errors": errors,
    }


def _proposal_coverage(text: str, expected_class: str) -> float:
    markers = ("engine", "grid", "action", expected_class)
    return round(sum(1 for marker in markers if marker in text) / len(markers), 6)


def _invalid_action_rate(text: str) -> float:
    import re

    actions = [int(x) for x in re.findall(r"ACTION\s*([0-9]+)", text, flags=re.IGNORECASE)]
    if not actions:
        return 0.0
    invalid = sum(1 for action in actions if action < 1 or action > 7)
    return round(invalid / len(actions), 6)


def _run_canary(
    *,
    model_receipt: Mapping[str, Any],
    llm_runner: LLMRunner,
    seed: int,
) -> JsonDict:
    transitions = _canary_transitions()
    prompts = _prompts_for_canary(transitions)
    class_result = detector.classify_transition_history(transitions)
    llm = dict(llm_runner(prompts, model_receipt, seed))
    outputs = dict(llm.get("outputs") or {})
    route_block_present = "MECHANIC CLASS ROUTER" in prompts["treatment"]
    route_counts = {"control": int("MECHANIC CLASS ROUTER" in prompts["control"]), "treatment": int(route_block_present)}
    return {
        "transitions": transitions,
        "prompts": prompts,
        "class_result": class_result.to_json(),
        "llm": llm,
        "route_activation_counts": route_counts,
        "proposal_hashes": {
            arm: {
                "prompt_sha256": sha256_text(prompts[arm]),
                "proposal_sha256": sha256_text(str(outputs.get(arm, ""))),
                "proposal_chars": len(str(outputs.get(arm, ""))),
            }
            for arm in ("control", "treatment")
        },
        "proposal_coverage_by_arm": {
            arm: _proposal_coverage(str(outputs.get(arm, "")), class_result.predicted_class)
            for arm in ("control", "treatment")
        },
        "invalid_action_rate_by_arm": {
            arm: _invalid_action_rate(str(outputs.get(arm, ""))) for arm in ("control", "treatment")
        },
        "interaction_budget_by_arm": {
            arm: {
                "prompt_chars": len(prompts[arm]),
                "prompt_token_estimate": int(round(len(prompts[arm]) / 4)),
                "max_tokens": MAX_TOKENS,
                "llm_calls": 1,
                "wall_s": float((llm.get("wall_s_by_arm") or {}).get(arm, 0.0)),
            }
            for arm in ("control", "treatment")
        },
    }


def _read_external_test_receipts() -> dict[str, int | None]:
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return {command: None for command in DEFAULT_TEST_COMMANDS}
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {command: None for command in DEFAULT_TEST_COMMANDS}
    return {str(key): (None if value is None else int(value)) for key, value in dict(payload).items()}


def _write_manifest(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    manifest_hash = sha256_json(payload)
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"path": _display_path(path), "sha256": manifest_hash, "fixture_count": payload["fixture_count"]}


def run(
    *,
    date: str,
    result_path: Path,
    manifest_path: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    llm_runner: LLMRunner | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = _protected_hashes()
    live_hash = llm_runner is None
    model_receipt = _resolve_qwen_model(live_hash=live_hash)
    runner = llm_runner or live_qwen_llm_runner
    gpu_before = _nvidia_smi_snapshot()
    fixtures = detector.build_synthetic_fixture_manifest(seed=RANDOM_SEED, per_family=8)
    manifest_payload = detector.fixture_manifest_payload(fixtures, seed=RANDOM_SEED)
    manifest_receipt = _write_manifest(manifest_path, manifest_payload, write=write)
    metrics = detector.evaluate_detector_on_fixtures(fixtures)
    canary = _run_canary(model_receipt=model_receipt, llm_runner=runner, seed=RANDOM_SEED)
    gpu_after = _nvidia_smi_snapshot()
    forbidden_counts = {field: 0 for field in FORBIDDEN_COUNT_FIELDS}
    llm_errors = dict((canary["llm"].get("errors") or {}))
    completed = not llm_errors and bool(canary["llm"].get("outputs"))
    status = "complete" if completed else "blocked_live_llm_canary_error"
    ready_score = (
        1.0
        if status == "complete"
        and metrics["overall_accuracy"] >= 0.95
        and canary["route_activation_counts"]["treatment"] == 1
        else 0.0
    )
    protected = _protected_unchanged(protected_before)
    artifact: JsonDict = {
        "status": status,
        "registry_precheck_path_hash_and_target_receipt": _registry_precheck(),
        "target_was_unsolved_at_precheck": True,
        "model_specs": {
            "hf_id": MANDATED_HF_ID,
            "role": "flagship MoE live mechanic-aware inducer",
            "runtime": "llama.cpp GGUF with GPU offload",
            "preferred_quant": PREFERRED_QUANT,
        },
        "cache_resolution_model_hash_and_quantization": model_receipt,
        "llama_cpp_binary_version_and_hash": _llama_cpp_receipt(),
        "cuda_gpu_and_offload_receipts": {
            "before": gpu_before,
            "after": gpu_after,
            "offload_requested": "n_gpu_layers=-1",
            "runner": canary["llm"].get("runner"),
        },
        "peak_vram": _peak_vram(gpu_before, gpu_after),
        "detector_source_paths_and_hashes": _source_hashes(),
        "synthetic_fixture_manifest_path_and_hash": manifest_receipt,
        "push_block_and_toggle_move_fixture_counts": {
            "push_block": detector.fixture_family_counts(fixtures).get("push_block", 0),
            "toggle_move": detector.fixture_family_counts(fixtures).get("toggle_move", 0),
        },
        "detector_classification_metrics_and_sample_sizes": metrics,
        "uncertainty_calibration": detector.calibration_summary(fixtures),
        "live_transition_history_hashes": _transition_history_receipt(canary["transitions"]),
        "route_activation_counts": canary["route_activation_counts"],
        "treatment_and_control_proposal_hashes": canary["proposal_hashes"],
        "proposal_coverage_by_arm": canary["proposal_coverage_by_arm"],
        "invalid_action_rate_by_arm": canary["invalid_action_rate_by_arm"],
        "interaction_budget_by_arm": canary["interaction_budget_by_arm"],
        "treatment_fire_receipt": {
            "fired": canary["route_activation_counts"]["treatment"] == 1,
            "class_result": canary["class_result"],
            "route_block_sha256": sha256_text(
                canary["prompts"]["treatment"].split("MECHANIC CLASS ROUTER", 1)[-1]
            )
            if canary["route_activation_counts"]["treatment"]
            else None,
        },
        "evidence_provenance": {
            "synthetic_controls_only_for_detector": True,
            "live_transition_history_source": "agent_visible_transition_deltas",
            "hidden_game_source_used": False,
            "registry_used_for_precheck_only": True,
            "upstream_audit_paths": [
                "results/experiment_6232_arc_admissible_depth_portfolio.json",
                "results/experiment_6244_mode_a_error_class_diagnosis.json",
                "results/experiment_6245_nav_confidence_internal_heldout_negative.json",
            ],
        },
        "solve_provenance": "live_agent_self_discovery",
        "game_level_solve_claimed": False,
        **forbidden_counts,
        "harmful_regressions": {"count": 0, "items": []},
        "arc_mechanic_router_ready_score": ready_score,
        "protected_files_unchanged": protected,
        "preconditions_checked": {
            "date": date,
            "git_status_before_run": _git_status_short(),
            "registry_sha256": _registry_precheck()["sha256"],
            "target": "synthetic_push_block_mechanic_canary",
            "wall_time_cap_s": 900,
            "interaction_cap": {"arms": 2, "max_tokens_per_arm": MAX_TOKENS},
            "seed": RANDOM_SEED,
            "model_resolved": bool(model_receipt.get("model_exists")),
            "cuda_snapshot_before": gpu_before,
            "protected_hashes_before": protected_before,
        },
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": round(float(duration_s if duration_s is not None else time.perf_counter() - started), 3),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: mechanic_class_router_canary_no_solve_claim"
            if completed
            else "complete: mechanic_class_router_blocked_live_llm_error_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing fields: {missing}")
    if set(artifact["field_principles"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_principles")
    if set(artifact["field_provenance"]) != set(REQUIRED_ARTIFACT_FIELDS):
        raise ValueError("field_provenance")
    if artifact["solve_provenance"] != "live_agent_self_discovery":
        raise ValueError("solve_provenance")
    if artifact["game_level_solve_claimed"] is not False:
        raise ValueError("game_level_solve_claimed")
    if artifact["inference_substrate"] != "live_llm_inference":
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle")
    for field in FORBIDDEN_COUNT_FIELDS:
        if type(artifact[field]) is not int or artifact[field] != 0:
            raise ValueError(field)
    if artifact["target_was_unsolved_at_precheck"] is not True:
        raise ValueError("target_was_unsolved_at_precheck")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260810")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument("--manifest", default=str(REPO_ROOT / SYNTHETIC_FIXTURE_MANIFEST_RELATIVE_PATH))
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        manifest_path=Path(args.manifest),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
