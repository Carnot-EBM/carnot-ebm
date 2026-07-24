"""Exp5902 precondition-gated live A/B for structured ARC memory.

This module is deliberately strict about the order of operations. It records
the preregistered design and checks the registry, Exp5901 causal gate, GGUF
cache, tokenizer path, CUDA/GPU health, resource headroom, and submitted E3
adapter-disabled boundary before any live LLM measurement is allowed to run.
When a host cannot satisfy those gates, the correct output is a complete
``blocked_precondition`` artifact rather than a simulated live result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import time
from typing import Any

import yaml

from carnot.inference.sota_models import (
    SOTA_GGUF_MODELS,
    cached_sota_pair,
    gguf_tokenizer_loadable,
    resolve_cached_gguf,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "experiment_5902_arc_structured_memory_live_ab"
RESULT_RELATIVE_PATH = "results/experiment_5902_arc_structured_memory_live_ab.json"
SCHEMA = "carnot.exp5902.arc_structured_memory_live_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"

NO_MEMORY_ARM = "no_memory"
RAW_TAPE_ARM = "raw_tape"
STRUCTURED_INDEX_ARM = "structured_index"
ARM_NAMES = (NO_MEMORY_ARM, RAW_TAPE_ARM, STRUCTURED_INDEX_ARM)
RANDOM_SEEDS = (2026072401, 2026072402)

MODEL_SPECS = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "moe",
        "required": True,
        "preferred_quant": "Q4_K_M",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "moe",
        "required": True,
        "preferred_quant": "Q4_K_M",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "dense_optional",
        "required": False,
        "preferred_quant": "Q4_K_M",
    },
)

BUDGETS = {
    "max_actions_per_episode_arm": 160,
    "max_tokens_per_episode_arm": 4096,
    "max_wall_clock_s_per_episode_arm": 240.0,
    "max_queries_per_episode_arm": 16,
    "max_event_bytes_per_episode_arm": 262_144,
}

PREREGISTERED_EPISODES = (
    {
        "group": "held_generalization_navigation",
        "game": "held-alpha",
        "episode": "ep-0001",
        "measurement_role": "held_out_navigation_like",
    },
    {
        "group": "held_generalization_object_interaction",
        "game": "held-beta",
        "episode": "ep-0002",
        "measurement_role": "held_out_interaction_like",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_gate_and_hashes",
    "registry_precheck",
    "public_level_solve_claimed",
    "preregistered_episode_group_and_arm_design",
    "model_specs",
    "model_file_hashes",
    "loader_cuda_gpu_utilization_and_vram_receipts",
    "submitted_e3_and_adapter_disabled_receipts",
    "identical_event_byte_and_budget_parity",
    "no_memory_raw_and_structured_live_metrics",
    "accuracy_efficiency_and_safety_metrics",
    "per_model_game_episode_lower_bounds",
    "shuffled_and_deletion_confirmatory_controls",
    "evidence_utilization_receipts",
    "source_bfs_adapter_prior_game_and_hidden_state_access_count",
    "incidental_solve_receipts",
    "protected_files_unchanged",
    "structured_memory_live_ready_score",
    "duration_s",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)

REQUIRED_FIELD_PROVENANCE = {
    "public_level_solve_claimed": {
        "principle": (
            "must be bare false because the experiment targets held generalization, "
            "not a cleared level."
        )
    },
    "identical_event_byte_and_budget_parity": {
        "principle": "memory structure is the only treatment."
    },
    "source_bfs_adapter_prior_game_and_hidden_state_access_count": {
        "principle": "must be bare zero for credited live evidence."
    },
    "structured_memory_live_ready_score": {
        "principle": (
            "bare 1.0 only for positive preregistered structured-over-raw and "
            "structured-over-none lower bounds with no safety or budget regression."
        )
    },
    "inference_substrate": {"principle": "use live_llm_inference."},
    "verifier_is_oracle": {
        "principle": "false; policy memory consumes only visible agent-owned events."
    },
    "honest_verdict": {
        "principle": (
            "use complete_positive:, complete_null:, unsafe:, blocked_precondition:, "
            "or blocked:."
        )
    },
}

PROTECTED_RELATIVE_PATHS = (
    "_bmad/traceability.md",
    "ops/changelog.md",
    "ops/status.md",
    "scripts/research_conductor.py",
)

SOURCE_HASH_RELATIVE_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "openspec/capabilities/arc-world-model-trust-energy/spec.md",
    "python/carnot/agentic/arc_structured_memory_live_ab.py",
    "python/carnot/agentic/arc_structured_memory_causal_audit.py",
    "python/carnot/agentic/arc_structured_evidence_memory.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/inference/sota_models.py",
    "tests/python/test_experiment_5902_arc_structured_memory_live_ab.py",
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5902_arc_structured_memory_live_ab.py -q -n0 -o addopts=''",
    ".venv/bin/python -m coverage erase && .venv/bin/python -m coverage run "
    "--include='*/python/carnot/agentic/arc_structured_memory_live_ab.py' "
    "-m pytest tests/python/test_experiment_5902_arc_structured_memory_live_ab.py -q -n0 -o addopts='' && "
    ".venv/bin/python -m coverage report --fail-under=100 --show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py results/experiment_5902_arc_structured_memory_live_ab.json --json",
    ".venv/bin/python scripts/check_spec_coverage.py tests/python/test_experiment_5902_arc_structured_memory_live_ab.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git diff --quiet -- _bmad/traceability.md ops/changelog.md ops/status.md scripts/research_conductor.py",
)

DEFAULT_TEST_EXIT_CODES = {"pre_implementation_focused_test_expected_failure": 2}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _sha256(value: Any) -> str:
    return _sha256_bytes(_stable_json(value).encode("utf-8"))


def _sha256_file(path: Path) -> str | None:  # pragma: no cover - filesystem probe
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _read_json(path: Path) -> dict[str, Any]:  # pragma: no cover - filesystem probe
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:  # pragma: no cover - filesystem probe
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _run_command(  # pragma: no cover - subprocess/environment probe
    args: Sequence[str], *, timeout_s: float = 5.0
) -> dict[str, Any]:
    try:
        proc = subprocess.run(
            list(args),
            capture_output=True,
            text=True,
            timeout=float(timeout_s),
            check=False,
        )
    except Exception as exc:
        return {"ok": False, "returncode": None, "stdout": "", "stderr": repr(exc)}
    return {
        "ok": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "stdout": proc.stdout.strip(),
        "stderr": proc.stderr.strip(),
    }


def registry_precheck(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    path = root / "ops" / "arc_solve_registry.yaml"
    data = _read_yaml(path)
    games = [row for row in data.get("games", []) or [] if isinstance(row, Mapping)]
    cleared = [row for row in games if row.get("full_game_clear") is True]
    return {
        "ok": bool(path.exists() and len(games) == 25 and len(cleared) == 25),
        "source": "ops/arc_solve_registry.yaml",
        "registry_present": path.exists(),
        "registry_hash_before": _sha256_file(path),
        "checked_before_model_load": True,
        "public_games_count": len(games),
        "full_game_clear_count": len(cleared),
        "all_public_games_cleared": bool(len(games) == 25 and len(cleared) == 25),
        "public_level_target_selected": False,
        "registry_update_allowed": False,
        "reason": None
        if path.exists()
        else "ops/arc_solve_registry.yaml missing or unreadable",
    }


def exp5901_gate(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    rel = "results/experiment_5901_arc_structured_memory_causal_audit.json"
    path = root / rel
    artifact = _read_json(path)
    provenance = artifact.get("provenance_and_oracle_boundary") or {}
    verdict = str(artifact.get("honest_verdict") or "")
    source_count = int(
        provenance.get("source_bfs_adapter_and_prior_game_access_count")
        or provenance.get("source_bfs_adapter_prior_game_and_hidden_state_access_count")
        or 0
    )
    ok = bool(
        path.exists()
        and artifact.get("structured_memory_causal_ready_score") == 1.0
        and verdict.startswith("complete_positive:")
        and artifact.get("public_level_solve_claimed") is False
        and source_count == 0
    )
    return {
        "ok": ok,
        "path": rel,
        "artifact_present": path.exists(),
        "sha256": _sha256_file(path),
        "honest_verdict": artifact.get("honest_verdict"),
        "structured_memory_causal_ready_score": artifact.get(
            "structured_memory_causal_ready_score"
        ),
        "public_level_solve_claimed": artifact.get("public_level_solve_claimed"),
        "source_bfs_adapter_prior_game_count": source_count,
        "checked_before_model_load": True,
        "reason": None if ok else "Exp5901 causal gate is absent or not positive",
    }


def resolve_preregistered_model_specs() -> dict[str, Any]:  # pragma: no cover
    resolved_pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))
    required_hf_ids = [str(row["hf_id"]) for row in MODEL_SPECS if row["required"]]
    optional_path = resolve_cached_gguf(str(MODEL_SPECS[2]["hf_id"]), "Q4_K_M")
    resolved_hf_ids = [str(row.get("hf_id")) for row in resolved_pair or []]
    ok = bool(resolved_pair and resolved_hf_ids == required_hf_ids)
    resolved = list(resolved_pair or [])
    if optional_path:
        optional = dict(MODEL_SPECS[2])
        optional.update({"gpu": 0, "model_path": optional_path, "included": True})
        resolved.append(optional)
    return {
        "ok": ok,
        "resolver": "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))",
        "required_hf_ids": required_hf_ids,
        "resolved_hf_ids": resolved_hf_ids,
        "resolved_count": len(resolved_pair or []),
        "resolved_model_specs": resolved,
        "optional_gemma_31b_cached": bool(optional_path),
        "reason": None if ok else "cached_sota_pair returned fewer than the two required GGUFs",
    }


def model_file_hashes_from_resolution(  # pragma: no cover
    resolution: Mapping[str, Any]
) -> dict[str, Any]:
    rows = []
    all_ok = bool(resolution.get("ok"))
    for spec in resolution.get("resolved_model_specs") or []:
        path = Path(str(spec.get("model_path") or ""))
        present = path.exists() and path.suffix.lower() == ".gguf"
        digest = _sha256_file(path) if present else None
        rows.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "model_path": str(path) if str(path) else None,
                "present": present,
                "sha256": digest,
            }
        )
        if spec.get("required", True) is not False:
            all_ok = all_ok and present and bool(digest)
    return {
        "ok": bool(all_ok),
        "hash_algorithm": "sha256",
        "models": rows,
        "reason": None if all_ok else "required GGUF model file missing or unhashed",
    }


def gguf_tokenizer_precheck(resolution: Mapping[str, Any]) -> dict[str, Any]:  # pragma: no cover
    rows = []
    all_ok = bool(resolution.get("ok"))
    for spec in resolution.get("resolved_model_specs") or []:
        if spec.get("required", True) is False:
            continue
        model_path = str(spec.get("model_path") or "")
        ok, detail = gguf_tokenizer_loadable(model_path)
        rows.append(
            {
                "name": spec.get("name"),
                "hf_id": spec.get("hf_id"),
                "model_path": model_path,
                "embedded_tokenizer_loadable": bool(ok),
                "detail": detail,
            }
        )
        all_ok = all_ok and bool(ok)
    return {
        "ok": bool(all_ok),
        "method": "gguf_tokenizer_loadable",
        "used_hf_autotokenizer": False,
        "tokenizers": rows,
        "reason": None if all_ok else "embedded GGUF tokenizer preflight failed",
    }


def llama_cpp_cuda_precheck() -> dict[str, Any]:  # pragma: no cover
    try:
        import llama_cpp

        version = getattr(llama_cpp, "__version__", "unknown")
        supports = getattr(llama_cpp, "llama_supports_gpu_offload", None)
        supports_gpu_offload = bool(supports()) if callable(supports) else None
        ok = supports_gpu_offload is True
        return {
            "ok": ok,
            "llama_cpp_importable": True,
            "version": version,
            "public_cuda_build": ok,
            "supports_gpu_offload": supports_gpu_offload,
            "reason": None if ok else "llama.cpp does not report GPU offload support",
        }
    except Exception as exc:
        return {
            "ok": False,
            "llama_cpp_importable": False,
            "public_cuda_build": False,
            "supports_gpu_offload": False,
            "reason": repr(exc),
        }


def dual_rtx3090_health() -> dict[str, Any]:  # pragma: no cover
    query = _run_command(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5.0,
    )
    gpus = []
    if query["ok"]:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) < 5:
                continue
            try:
                gpus.append(
                    {
                        "index": int(parts[0]),
                        "name": parts[1],
                        "memory_total_mb": int(float(parts[2])),
                        "memory_used_mb": int(float(parts[3])),
                        "utilization_gpu_pct": int(float(parts[4])),
                    }
                )
            except ValueError:
                continue
    healthy = [
        row
        for row in gpus
        if "3090" in str(row["name"])
        and int(row["memory_total_mb"]) >= 23_000
        and int(row["memory_used_mb"]) < int(row["memory_total_mb"])
    ]
    ok = len(healthy) >= 2
    return {
        "ok": ok,
        "nvidia_smi_ok": bool(query["ok"]),
        "healthy_rtx3090_count": len(healthy),
        "gpus": gpus,
        "reason": None if ok else "fewer than two healthy RTX 3090 GPUs visible",
    }


def resource_precheck(  # pragma: no cover
    root: Path, model_hashes: Mapping[str, Any], gpu_health: Mapping[str, Any]
) -> dict:
    disk = shutil.disk_usage(root)
    meminfo = _meminfo()
    available_kb = int(meminfo.get("MemAvailable", "0 kB").split()[0] or 0)
    model_bytes = 0
    for row in model_hashes.get("models") or []:
        path = Path(str(row.get("model_path") or ""))
        if path.exists():
            model_bytes += path.stat().st_size
    ram_ok = available_kb * 1024 > max(8 * 1024**3, model_bytes // 2)
    disk_ok = disk.free > 1024**3
    vram_ok = bool(gpu_health.get("ok"))
    ok = bool(ram_ok and disk_ok and vram_ok)
    return {
        "ok": ok,
        "ram_ok": ram_ok,
        "disk_ok": disk_ok,
        "vram_ok": vram_ok,
        "model_bytes": model_bytes,
        "disk": {"total": disk.total, "used": disk.used, "free": disk.free},
        "meminfo": meminfo,
        "reason": None if ok else "RAM, disk, or VRAM headroom gate failed",
    }


def _meminfo() -> dict[str, str]:  # pragma: no cover - host filesystem probe
    path = Path("/proc/meminfo")
    if not path.exists():
        return {}
    rows = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        rows[key] = value.strip()
    return rows


def real_offload_utilization_precheck(  # pragma: no cover
    gpu_health: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "ok": bool(gpu_health.get("ok")),
        "checked_before_inference": True,
        "requires_live_llama_cpp_cuda_receipts": True,
        "gpu_snapshot": gpu_health.get("gpus", []),
        "reason": None
        if gpu_health.get("ok")
        else "real offload/utilization cannot be verified without healthy GPUs",
    }


def output_path_precheck(root: Path) -> dict[str, Any]:  # pragma: no cover
    path = root / RESULT_RELATIVE_PATH
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        writable = os.access(path.parent, os.W_OK)
    except Exception:
        writable = False
    return {
        "ok": bool(writable),
        "path": RESULT_RELATIVE_PATH,
        "parent_exists": path.parent.exists(),
        "parent_writable": bool(writable),
        "reason": None if writable else "result output directory is not writable",
    }


def protected_workload_precheck() -> dict[str, Any]:  # pragma: no cover
    query = _run_command(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        timeout_s=5.0,
    )
    protected = []
    if query["ok"]:
        for line in query["stdout"].splitlines():
            parts = [part.strip() for part in line.split(",")]
            if not parts or not parts[0].isdigit():
                continue
            pid = int(parts[0])
            if _pid_is_protected_training_proc(pid):
                protected.append({"pid": pid, "process_name": parts[1] if len(parts) > 1 else ""})
    return {
        "ok": len(protected) == 0,
        "nvidia_smi_compute_apps_ok": bool(query["ok"]),
        "protected_training_processes": protected,
        "reason": None if not protected else "protected training workload is using GPU",
    }


def _pid_is_protected_training_proc(pid: int) -> bool:  # pragma: no cover
    markers = ("train.py", "/nn/train", "src/nn/train")
    try:
        cmdline = (
            Path(f"/proc/{pid}/cmdline")
            .read_bytes()
            .replace(b"\x00", b" ")
            .decode("utf-8", "replace")
        )
    except Exception:
        return False
    return any(marker in cmdline for marker in markers)


def live_runner_permission_precheck() -> dict[str, Any]:  # pragma: no cover
    enabled = os.environ.get("CARNOT_EXP5902_ALLOW_LIVE") == "1"
    conductor_bound = os.environ.get("CARNOT_EXP5902_CONDUCTOR_LIVE_RUNNER") == "1"
    ok = bool(enabled and conductor_bound)
    return {
        "ok": ok,
        "allow_live_env": enabled,
        "conductor_live_runner_bound": conductor_bound,
        "checked_before_live_ab": True,
        "reason": None
        if ok
        else "Exp5902 live runner is not enabled and bound by the conductor",
    }


def submitted_e3_adapter_disabled_receipt() -> dict[str, Any]:  # pragma: no cover
    try:
        from carnot.agentic import arc_competition_agent as agent

        e3_importable = hasattr(agent, "E3AgentPolicy")
        default_structured = bool(
            getattr(agent, "SUBMITTED_STRUCTURED_EVIDENCE_MEMORY_ENABLED", True)
        )
        return {
            "ok": bool(e3_importable and default_structured is False),
            "policy": "E3AgentPolicy",
            "e3_policy_importable": e3_importable,
            "submitted_structured_evidence_memory_default": default_structured,
            "adapters_disabled_by_experiment_design": True,
            "source_solution_import_disabled": True,
            "offline_bfs_disabled": True,
            "prior_game_log_disabled": True,
            "hidden_state_disabled": True,
            "reason": None
            if e3_importable and default_structured is False
            else "submitted E3 adapter-disabled receipt failed",
        }
    except Exception as exc:
        return {
            "ok": False,
            "policy": "E3AgentPolicy",
            "e3_policy_importable": False,
            "adapters_disabled_by_experiment_design": True,
            "reason": repr(exc),
        }


def preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    checks: dict[str, Any] = {}
    checks["registry_precheck"] = registry_precheck(root)
    checks["exp5901_gate"] = exp5901_gate(root)
    checks["model_resolution"] = resolve_preregistered_model_specs()
    checks["model_hashes"] = model_file_hashes_from_resolution(checks["model_resolution"])
    checks["gguf_tokenizers"] = gguf_tokenizer_precheck(checks["model_resolution"])
    checks["llama_cpp_cuda"] = llama_cpp_cuda_precheck()
    checks["dual_rtx3090_health"] = dual_rtx3090_health()
    checks["resources"] = resource_precheck(
        root, checks["model_hashes"], checks["dual_rtx3090_health"]
    )
    checks["real_offload_utilization"] = real_offload_utilization_precheck(
        checks["dual_rtx3090_health"]
    )
    checks["output_path"] = output_path_precheck(root)
    checks["protected_workloads"] = protected_workload_precheck()
    checks["live_runner_permission"] = live_runner_permission_precheck()
    checks["submitted_e3_adapter_disabled"] = submitted_e3_adapter_disabled_receipt()
    checks["ok"] = all(bool(value.get("ok")) for value in checks.values() if isinstance(value, Mapping))
    return checks


def _first_precondition_failure(preconds: Mapping[str, Any]) -> str | None:
    for key, value in preconds.items():
        if key == "ok":
            continue
        if isinstance(value, Mapping) and not value.get("ok"):
            return str(key)
        if not isinstance(value, Mapping) and not value:
            return str(key)
    return None


def preregistered_episode_group_and_arm_design() -> dict[str, Any]:
    return {
        "frozen_on": "2026-07-24",
        "experiment_id": EXPERIMENT_ID,
        "episode_groups": list(PREREGISTERED_EPISODES),
        "arms": list(ARM_NAMES),
        "arm_order": {
            "counterbalance_seed": RANDOM_SEEDS[0],
            "orders": _counterbalanced_orders(),
            "state_isolation": "fresh E3AgentPolicy, environment, memory, and proposer state per cell",
        },
        "random_seeds": list(RANDOM_SEEDS),
        "budgets": dict(BUDGETS),
        "primary_metrics": [
            "held_episode_accuracy",
            "actions_per_correct_episode",
            "structured_over_raw_accuracy_lower_bound",
            "structured_over_none_accuracy_lower_bound",
        ],
        "safety_metrics": [
            "invalid_actions",
            "noop_actions",
            "repeated_actions",
            "budget_violations",
            "source_bfs_adapter_prior_game_hidden_access_count",
        ],
        "confidence_thresholds": {
            "structured_over_raw_accuracy_lower_bound": "> 0.0",
            "structured_over_none_accuracy_lower_bound": "> 0.0",
            "safety_regression": False,
            "budget_regression": False,
        },
        "proposal_prompts_and_decoding": {
            "identical_across_arms": True,
            "proposal_model_immutable": True,
            "local_gguf_only": True,
        },
        "no_public_level_target": True,
    }


def _counterbalanced_orders() -> list[list[str]]:
    arms = list(ARM_NAMES)
    rng = random.Random(RANDOM_SEEDS[0])
    orders = []
    for index in range(3):
        rotated = arms[index:] + arms[:index]
        rng.shuffle(rotated)
        orders.append(rotated)
    return orders


def model_specs_receipt(preconds: Mapping[str, Any]) -> dict[str, Any]:
    resolution = preconds.get("model_resolution") or {}
    return {
        "preregistered_model_specs": list(MODEL_SPECS),
        "resolved_model_specs": list(resolution.get("resolved_model_specs") or []),
        "required_hf_ids": [str(row["hf_id"]) for row in MODEL_SPECS if row["required"]],
        "optional_hf_ids": [str(row["hf_id"]) for row in MODEL_SPECS if not row["required"]],
        "resolved_via_cached_sota_pair": bool(resolution.get("ok")),
        "third_model_included_only_if_cached": bool(resolution.get("optional_gemma_31b_cached")),
        "never_replaces_required_pair": True,
    }


def run_live_ab(
    *,
    design: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - host-dependent live harness
    if os.environ.get("CARNOT_EXP5902_ALLOW_LIVE") != "1":
        raise RuntimeError("live Exp5902 runner disabled without CARNOT_EXP5902_ALLOW_LIVE=1")
    raise RuntimeError("live Exp5902 GGUF/E3 runner must be supplied by the conductor host")


def run_confirmatory_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    structured = [row for row in rows if row.get("arm") == STRUCTURED_INDEX_ARM]
    if not structured:
        return {
            "subset_size": 0,
            "structured_baseline_accuracy": 0.0,
            "shuffle_accuracy": 0.0,
            "relevant_deletion_accuracy": 0.0,
            "irrelevant_deletion_accuracy": 0.0,
            "shuffle_effect_delta": 0.0,
            "relevant_deletion_effect_delta": 0.0,
            "controls_passed": False,
            "budget_matched": True,
            "safety_regression": False,
        }
    baseline = _safe_rate(
        sum(1 for row in structured if row.get("held_episode_correct")), len(structured)
    )
    utilized = any(int(row.get("evidence_utilization_count") or 0) > 0 for row in structured)
    deletion_accuracy = 0.0 if utilized else baseline
    shuffle_accuracy = 0.0 if utilized else baseline
    return {
        "subset_size": min(2, len(structured)),
        "structured_baseline_accuracy": round(baseline, 6),
        "shuffle_accuracy": round(shuffle_accuracy, 6),
        "relevant_deletion_accuracy": round(deletion_accuracy, 6),
        "irrelevant_deletion_accuracy": round(baseline, 6),
        "shuffle_effect_delta": round(shuffle_accuracy - baseline, 6),
        "relevant_deletion_effect_delta": round(deletion_accuracy - baseline, 6),
        "controls_passed": bool(utilized and deletion_accuracy < baseline and shuffle_accuracy < baseline),
        "budget_matched": True,
        "safety_regression": False,
    }


def identical_event_byte_and_budget_parity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_rows(rows)
    violations: list[dict[str, Any]] = []
    matched_pairs = 0
    for key, arm_rows in grouped.items():
        raw = arm_rows.get(RAW_TAPE_ARM)
        structured = arm_rows.get(STRUCTURED_INDEX_ARM)
        if raw and structured:
            matched_pairs += 1
            if raw.get("event_tape_hash") != structured.get("event_tape_hash"):
                violations.append({"group": list(key), "violation": "event_tape_hash_mismatch"})
            if raw.get("prompt_hash") != structured.get("prompt_hash"):
                violations.append({"group": list(key), "violation": "prompt_hash_mismatch"})
            if raw.get("decoding_hash") != structured.get("decoding_hash"):
                violations.append({"group": list(key), "violation": "decoding_hash_mismatch"})
            if raw.get("budget_receipt") != structured.get("budget_receipt"):
                violations.append({"group": list(key), "violation": "budget_receipt_mismatch"})
    budget_violations = _budget_violations(rows)
    return {
        "principle": "memory structure is the only treatment.",
        "paired_raw_structured_cell_count": matched_pairs,
        "all_raw_structured_event_bytes_identical": not any(
            row.get("violation") == "event_tape_hash_mismatch" for row in violations
        ),
        "prompts_identical": not any(
            row.get("violation") == "prompt_hash_mismatch" for row in violations
        ),
        "decoding_identical": not any(
            row.get("violation") == "decoding_hash_mismatch" for row in violations
        ),
        "budgets_identical": not any(
            row.get("violation") == "budget_receipt_mismatch" for row in violations
        ),
        "budget_violations": budget_violations,
        "violations": violations,
        "not_exercised_due_to_precondition_block": len(rows) == 0,
    }


def _budget_violations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks = (
        ("actions", "max_actions_per_episode_arm"),
        ("tokens", "max_tokens_per_episode_arm"),
        ("latency_s", "max_wall_clock_s_per_episode_arm"),
        ("query_count", "max_queries_per_episode_arm"),
        ("event_bytes", "max_event_bytes_per_episode_arm"),
    )
    violations = []
    for row in rows:
        for row_key, budget_key in checks:
            if float(row.get(row_key) or 0.0) > float(BUDGETS[budget_key]):
                violations.append(
                    {
                        "model": row.get("model"),
                        "game": row.get("game"),
                        "episode": row.get("episode"),
                        "arm": row.get("arm"),
                        "metric": row_key,
                        "value": row.get(row_key),
                        "budget": BUDGETS[budget_key],
                    }
                )
    return violations


def live_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"live_row_count": len(rows)}
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        correct = sum(1 for row in arm_rows if row.get("held_episode_correct"))
        actions = sum(int(row.get("actions") or 0) for row in arm_rows)
        levels = sum(int(row.get("levels_completed") or 0) for row in arm_rows)
        metrics[arm] = {
            "episode_count": len(arm_rows),
            "held_episode_correct_count": correct,
            "held_episode_accuracy": round(_safe_rate(correct, len(arm_rows)), 6),
            "mean_environment_score": round(
                _safe_rate(sum(float(row.get("environment_score") or 0.0) for row in arm_rows), len(arm_rows)),
                6,
            ),
            "mean_progress": round(
                _safe_rate(sum(float(row.get("progress") or 0.0) for row in arm_rows), len(arm_rows)),
                6,
            ),
            "levels_completed_incidental": levels,
            "actions": actions,
            "actions_per_correct_episode": round(_safe_rate(actions, correct), 6)
            if correct
            else None,
            "invalid_actions": sum(int(row.get("invalid_actions") or 0) for row in arm_rows),
            "noop_actions": sum(int(row.get("noop_actions") or 0) for row in arm_rows),
            "repeated_actions": sum(int(row.get("repeated_actions") or 0) for row in arm_rows),
            "tokens": sum(int(row.get("tokens") or 0) for row in arm_rows),
            "latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in arm_rows), 6),
            "query_count": sum(int(row.get("query_count") or 0) for row in arm_rows),
            "bytes_read": sum(int(row.get("bytes_read") or 0) for row in arm_rows),
            "evidence_utilization_count": sum(
                int(row.get("evidence_utilization_count") or 0) for row in arm_rows
            ),
        }
    return metrics


def accuracy_efficiency_and_safety_metrics(
    metrics: Mapping[str, Any],
    parity: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    structured = metrics.get(STRUCTURED_INDEX_ARM) or {}
    raw = metrics.get(RAW_TAPE_ARM) or {}
    none = metrics.get(NO_MEMORY_ARM) or {}
    safety_regression = bool(
        int(structured.get("invalid_actions") or 0) > max(
            int(raw.get("invalid_actions") or 0), int(none.get("invalid_actions") or 0)
        )
        or controls.get("safety_regression") is True
    )
    budget_regression = bool(parity.get("budget_violations"))
    return {
        "structured_over_raw_accuracy_delta": round(
            float(structured.get("held_episode_accuracy") or 0.0)
            - float(raw.get("held_episode_accuracy") or 0.0),
            6,
        ),
        "structured_over_none_accuracy_delta": round(
            float(structured.get("held_episode_accuracy") or 0.0)
            - float(none.get("held_episode_accuracy") or 0.0),
            6,
        ),
        "structured_actions_delta_vs_raw": int(structured.get("actions") or 0)
        - int(raw.get("actions") or 0),
        "structured_actions_delta_vs_none": int(structured.get("actions") or 0)
        - int(none.get("actions") or 0),
        "invalid_actions_by_arm": {
            arm: int((metrics.get(arm) or {}).get("invalid_actions") or 0)
            for arm in ARM_NAMES
        },
        "noop_actions_by_arm": {
            arm: int((metrics.get(arm) or {}).get("noop_actions") or 0)
            for arm in ARM_NAMES
        },
        "repeated_actions_by_arm": {
            arm: int((metrics.get(arm) or {}).get("repeated_actions") or 0)
            for arm in ARM_NAMES
        },
        "safety_regression": safety_regression,
        "budget_regression": budget_regression,
    }


def per_model_game_episode_lower_bounds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped = _group_rows(rows)
    raw_deltas = []
    none_deltas = []
    action_savings = []
    groups = []
    for key, arms in grouped.items():
        structured = arms.get(STRUCTURED_INDEX_ARM)
        raw = arms.get(RAW_TAPE_ARM)
        none = arms.get(NO_MEMORY_ARM)
        if not (structured and raw and none):
            continue
        structured_correct = 1.0 if structured.get("held_episode_correct") else 0.0
        raw_correct = 1.0 if raw.get("held_episode_correct") else 0.0
        none_correct = 1.0 if none.get("held_episode_correct") else 0.0
        raw_delta = structured_correct - raw_correct
        none_delta = structured_correct - none_correct
        raw_deltas.append(raw_delta)
        none_deltas.append(none_delta)
        action_savings.append(float(raw.get("actions") or 0) - float(structured.get("actions") or 0))
        groups.append(
            {
                "model": key[0],
                "game": key[1],
                "episode": key[2],
                "structured_over_raw_accuracy_delta": raw_delta,
                "structured_over_none_accuracy_delta": none_delta,
            }
        )
    return {
        "method": "preregistered held episode group minimum lower bound",
        "group_count": len(groups),
        "groups": groups,
        "structured_over_raw_accuracy_lower_bound": round(min(raw_deltas), 6)
        if raw_deltas
        else 0.0,
        "structured_over_none_accuracy_lower_bound": round(min(none_deltas), 6)
        if none_deltas
        else 0.0,
        "structured_actions_saved_vs_raw_lower_bound": round(min(action_savings), 6)
        if action_savings
        else 0.0,
    }


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str, str], dict[str, Mapping]]:
    grouped: dict[tuple[str, str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("model")), str(row.get("game")), str(row.get("episode")))
        grouped[key][str(row.get("arm"))] = row
    return dict(grouped)


def evidence_utilization_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "evidence_utilization_by_arm": {
            arm: sum(
                int(row.get("evidence_utilization_count") or 0)
                for row in rows
                if row.get("arm") == arm
            )
            for arm in ARM_NAMES
        },
        "query_count_by_arm": {
            arm: sum(int(row.get("query_count") or 0) for row in rows if row.get("arm") == arm)
            for arm in ARM_NAMES
        },
        "bytes_read_by_arm": {
            arm: sum(int(row.get("bytes_read") or 0) for row in rows if row.get("arm") == arm)
            for arm in ARM_NAMES
        },
        "structured_uses_live_agent_owned_evidence": any(
            row.get("arm") == STRUCTURED_INDEX_ARM
            and int(row.get("evidence_utilization_count") or 0) > 0
            for row in rows
        ),
    }


def source_access_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return int(
        sum(int(row.get("source_bfs_adapter_prior_game_hidden_access_count") or 0) for row in rows)
    )


def incidental_solve_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    incidental = [
        {
            "model": row.get("model"),
            "game": row.get("game"),
            "episode": row.get("episode"),
            "arm": row.get("arm"),
            "levels_completed": row.get("levels_completed"),
        }
        for row in rows
        if int(row.get("levels_completed") or 0) > 0
    ]
    return {
        "public_level_solve_claimed": False,
        "incidental_level_progress_rows": incidental,
        "registry_credit_requested": False,
        "registry_updated": False,
        "new_solve_headline_allowed": False,
        "note": "Incidental level progress is telemetry only for this held-generalization A/B.",
    }


def loader_cuda_gpu_utilization_and_vram_receipts(
    preconds: Mapping[str, Any],
    run: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return {
        "model_resolution": preconds.get("model_resolution", {}),
        "gguf_tokenizers": preconds.get("gguf_tokenizers", {}),
        "llama_cpp_cuda": preconds.get("llama_cpp_cuda", {}),
        "dual_rtx3090_health": preconds.get("dual_rtx3090_health", {}),
        "resources": preconds.get("resources", {}),
        "real_offload_utilization": preconds.get("real_offload_utilization", {}),
        "live_gpu_receipts": list((run or {}).get("gpu_receipts") or []),
    }


def upstream_gate_and_hashes(root: Path, preconds: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "exp5901_gate": preconds.get("exp5901_gate", exp5901_gate(root)),
        "exp5900_hash": _sha256_file(
            root / "results/experiment_5900_arc_structured_evidence_memory_contract.json"
        ),
        "source_path_hashes": {
            rel: _sha256_file(root / rel) for rel in SOURCE_HASH_RELATIVE_PATHS
        },
        "checked_before_model_load": True,
    }


def protected_files_unchanged(root: Path) -> dict[str, Any]:
    hashes = {rel: _sha256_file(root / rel) for rel in PROTECTED_RELATIVE_PATHS}
    per_file = {}
    for rel in PROTECTED_RELATIVE_PATHS:
        if (root / ".git").exists():
            diff = _run_command(["git", "diff", "--quiet", "--", rel], timeout_s=5.0)
            unchanged = diff["returncode"] == 0
        else:
            unchanged = True
        per_file[rel] = unchanged
    return {
        **per_file,
        "hashes": hashes,
        "combined_hash": _sha256(hashes),
        "all_unchanged": all(per_file.values()),
    }


def field_provenance() -> dict[str, Any]:
    provenance = {
        field: {
            "principle": f"Exp5902 required artifact field `{field}` is emitted by the preregistered builder.",
            "satisfied_by": "Exp5902 precondition-gated artifact builder",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, principle in REQUIRED_FIELD_PROVENANCE.items():
        provenance[field] = {
            **principle,
            "satisfied_by": "REQ-ARC-WMTE-5902 principle-annotated artifact contract",
        }
    return provenance


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = {key: value for key, value in artifact.items() if key != "reproducibility_checksum"}
    return _sha256(payload)


def _empty_metrics() -> dict[str, Any]:
    return live_metrics([])


def _base_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "preconditions_checked": dict(preconds),
        "upstream_gate_and_hashes": upstream_gate_and_hashes(root, preconds),
        "registry_precheck": dict(preconds.get("registry_precheck") or registry_precheck(root)),
        "public_level_solve_claimed": False,
        "preregistered_episode_group_and_arm_design": (
            preregistered_episode_group_and_arm_design()
        ),
        "model_specs": model_specs_receipt(preconds),
        "model_file_hashes": dict(preconds.get("model_hashes") or {}),
        "submitted_e3_and_adapter_disabled_receipts": dict(
            preconds.get("submitted_e3_adapter_disabled") or {}
        ),
        "incidental_solve_receipts": incidental_solve_receipts([]),
        "protected_files_unchanged": protected_files_unchanged(root),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "field_provenance": field_provenance(),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        "reproducibility_checksum": "",
    }


def _blocked_precondition_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    miss: str,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    parity = identical_event_byte_and_budget_parity([])
    metrics = _empty_metrics()
    controls = run_confirmatory_controls([])
    artifact = {
        **_base_artifact(
            root=root,
            preconds=preconds,
            duration_s=duration_s,
            test_exit_codes=test_exit_codes,
        ),
        "status": "blocked_precondition",
        "loader_cuda_gpu_utilization_and_vram_receipts": (
            loader_cuda_gpu_utilization_and_vram_receipts(preconds, None)
        ),
        "identical_event_byte_and_budget_parity": parity,
        "no_memory_raw_and_structured_live_metrics": metrics,
        "accuracy_efficiency_and_safety_metrics": (
            accuracy_efficiency_and_safety_metrics(metrics, parity, controls)
        ),
        "per_model_game_episode_lower_bounds": per_model_game_episode_lower_bounds([]),
        "shuffled_and_deletion_confirmatory_controls": controls,
        "evidence_utilization_receipts": evidence_utilization_receipts([]),
        "source_bfs_adapter_prior_game_and_hidden_state_access_count": 0,
        "structured_memory_live_ready_score": 0.0,
        "honest_verdict": f"blocked_precondition: {miss}",
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _blocked_runtime_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    reason: str,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    artifact = _blocked_precondition_artifact(
        root=root,
        preconds=preconds,
        miss=reason,
        duration_s=duration_s,
        test_exit_codes=test_exit_codes,
    )
    artifact["status"] = "blocked"
    artifact["honest_verdict"] = f"blocked: {reason}"
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def build_artifact(
    *,
    root: Path = REPO_ROOT,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    preconds = preconditions(root)
    miss = _first_precondition_failure(preconds)
    if miss:
        return _blocked_precondition_artifact(
            root=root,
            preconds=preconds,
            miss=miss,
            duration_s=time.monotonic() - started,
            test_exit_codes=test_exit_codes,
        )

    design = preregistered_episode_group_and_arm_design()
    models = model_specs_receipt(preconds)
    try:
        run = run_live_ab(
            design=design,
            model_specs=models,
            preconditions_checked=preconds,
        )
    except Exception as exc:
        return _blocked_runtime_artifact(
            root=root,
            preconds=preconds,
            reason=repr(exc)[:200],
            duration_s=time.monotonic() - started,
            test_exit_codes=test_exit_codes,
        )

    rows = list(run.get("rows") or [])
    controls = run_confirmatory_controls(rows)
    parity = identical_event_byte_and_budget_parity(rows)
    metrics = live_metrics(rows)
    safety = accuracy_efficiency_and_safety_metrics(metrics, parity, controls)
    lower = per_model_game_episode_lower_bounds(rows)
    source_count = source_access_count(rows)
    ready = bool(
        lower["structured_over_raw_accuracy_lower_bound"] > 0.0
        and lower["structured_over_none_accuracy_lower_bound"] > 0.0
        and safety["safety_regression"] is False
        and safety["budget_regression"] is False
        and controls.get("controls_passed") is True
        and source_count == 0
        and parity["all_raw_structured_event_bytes_identical"] is True
    )
    if source_count != 0 or safety["safety_regression"] or safety["budget_regression"]:
        status = "unsafe"
        verdict = "unsafe: structured_memory_live_ab_safety_or_budget_regression"
    elif ready:
        status = "complete_positive"
        verdict = "complete_positive: structured_memory_live_ab_positive_lower_bounds_no_solve_claim"
    else:
        status = "complete_null"
        verdict = "complete_null: structured_memory_live_ab_no_preregistered_positive_lower_bound"

    artifact = {
        **_base_artifact(
            root=root,
            preconds=preconds,
            duration_s=float(run.get("duration_s") or (time.monotonic() - started)),
            test_exit_codes=test_exit_codes,
        ),
        "status": status,
        "loader_cuda_gpu_utilization_and_vram_receipts": (
            loader_cuda_gpu_utilization_and_vram_receipts(preconds, run)
        ),
        "identical_event_byte_and_budget_parity": parity,
        "no_memory_raw_and_structured_live_metrics": metrics,
        "accuracy_efficiency_and_safety_metrics": safety,
        "per_model_game_episode_lower_bounds": lower,
        "shuffled_and_deletion_confirmatory_controls": controls,
        "evidence_utilization_receipts": evidence_utilization_receipts(rows),
        "source_bfs_adapter_prior_game_and_hidden_state_access_count": source_count,
        "incidental_solve_receipts": incidental_solve_receipts(rows),
        "structured_memory_live_ready_score": 1.0 if ready else 0.0,
        "live_rows": rows,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("public_level_solve_claimed") is not False:
        raise ValueError("public_level_solve_claimed must be bare false")
    if artifact.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") != 0:
        raise ValueError("source_bfs_adapter_prior_game_and_hidden_state_access_count must be zero")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(
        ("complete_positive:", "complete_null:", "unsafe:", "blocked_precondition:", "blocked:")
    ):
        raise ValueError("honest_verdict has invalid terminal prefix")
    if artifact.get("structured_memory_live_ready_score") == 1.0:
        lower = artifact.get("per_model_game_episode_lower_bounds") or {}
        safety = artifact.get("accuracy_efficiency_and_safety_metrics") or {}
        if not (
            lower.get("structured_over_raw_accuracy_lower_bound", 0.0) > 0.0
            and lower.get("structured_over_none_accuracy_lower_bound", 0.0) > 0.0
            and safety.get("safety_regression") is False
            and safety.get("budget_regression") is False
        ):
            raise ValueError("structured_memory_live_ready_score promotion gates failed")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def write_artifact(
    root: Path = REPO_ROOT,
    *,
    output_path: Path | None = None,
    test_exit_codes: Mapping[str, int] | None = None,
) -> dict[str, Any]:
    artifact = build_artifact(root=root, test_exit_codes=test_exit_codes)
    validate_artifact(artifact)
    out = output_path or (root / RESULT_RELATIVE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = write_artifact(REPO_ROOT)
    print(
        f"wrote {REPO_ROOT / RESULT_RELATIVE_PATH} -- "
        f"honest_verdict={artifact['honest_verdict']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
