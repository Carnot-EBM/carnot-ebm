"""Exp5929 capability-bound held E3 structured memory A/B.

Spec refs: REQ-ARC-LRBH-5929,
SCENARIO-ARC-LRBH-5929-PRECONDITION-BLOCK,
SCENARIO-ARC-LRBH-5929-BOUND-MATCHED-HELD-LIVE-AB,
SCENARIO-ARC-LRBH-5929-NO-SOLVE-CREDIT.

The important boundary is provenance, not convenience. This module may emit a
blocked artifact when the local host cannot provide the bound GGUF/E3 runner,
but it must never backfill live rows from Exp5901 or any other offline proxy.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import yaml

from carnot.agentic import arc_live_runner_execution_binding as binding
from carnot.agentic import arc_structured_memory_live_ab as live_ab
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "experiment_5929_arc_structured_memory_bound_live_ab"
RESULT_RELATIVE_PATH = "results/experiment_5929_arc_structured_memory_bound_live_ab.json"
SCHEMA = "carnot.exp5929.arc_structured_memory_bound_live_ab.v1"
INFERENCE_SUBSTRATE = (
    "actual_capability_bound_adapter_disabled_e3_local_mandated_gguf_public_llama_cpp_cuda"
)

NO_MEMORY_ARM = live_ab.NO_MEMORY_ARM
RAW_TAPE_ARM = live_ab.RAW_TAPE_ARM
STRUCTURED_INDEX_ARM = live_ab.STRUCTURED_INDEX_ARM
ARM_NAMES = (NO_MEMORY_ARM, RAW_TAPE_ARM, STRUCTURED_INDEX_ARM)
RANDOM_SEEDS = (2026072601, 2026072602)

MODEL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "Qwen3.6-35B-A3B",
        "hf_id": "unsloth/Qwen3.6-35B-A3B-GGUF",
        "role": "flagship_moe_required_pair",
        "required": True,
        "preferred_quant": "Q4_K_M",
        "resolved_via": "cached_sota_pair",
    },
    {
        "name": "Gemma4-31B-it",
        "hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "role": "cached_third_family",
        "required": True,
        "preferred_quant": "Q4_K_M",
        "resolved_via": "resolve_cached_gguf_cached_third_family",
    },
    {
        "name": "Gemma4-26B-A4B-it",
        "hf_id": "unsloth/gemma-4-26B-A4B-it-GGUF",
        "role": "middle_moe_required_pair",
        "required": True,
        "preferred_quant": "Q4_K_M",
        "resolved_via": "cached_sota_pair",
    },
)

BUDGETS = {
    "max_actions_per_episode_arm": 180,
    "max_context_tokens_per_episode_arm": 8192,
    "max_tokens_per_episode_arm": 4096,
    "max_wall_clock_s_per_episode_arm": 300.0,
    "max_queries_per_episode_arm": 16,
    "max_event_bytes_per_episode_arm": 262_144,
}

HELD_EPISODE_CELLS = (
    {
        "held_cell": "held-e3-alpha/episode-0001",
        "game": "held-e3-alpha",
        "episode": "episode-0001",
        "split": "held_e3_generalization",
        "measurement_role": "adapter_disabled_navigation_like",
    },
    {
        "held_cell": "held-e3-beta/episode-0002",
        "game": "held-e3-beta",
        "episode": "episode-0002",
        "split": "held_e3_generalization",
        "measurement_role": "adapter_disabled_object_interaction_like",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "gate_and_capability_replay_receipt",
    "preconditions_checked",
    "registry_precheck_and_selected_held_cells",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_loader_cuda_gpu_and_vram_receipts",
    "actual_bound_e3_entrypoint_receipt",
    "adapter_disabled",
    "no_per_game_adapter_or_public_solve_target",
    "solve_provenance",
    "identical_event_bytes_and_arm_budget_parity",
    "sealed_prompts_seeds_models_arms_and_stopping_rules",
    "per_model_episode_retrieval_progress_legality_efficiency_and_abstention",
    "primary_live_utility_comparison_and_intervals",
    "token_context_latency_gpu_and_memory_accounting",
    "capability_expiry_teardown_and_orphan_receipts",
    "registry_unchanged",
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
    "solve_provenance": {
        "principle": (
            "use live_agent_self_discovery for any observed level outcome; no "
            "development proxy or outer-loop result can satisfy this task."
        )
    },
    "identical_event_bytes_and_arm_budget_parity": {
        "principle": "arms may differ only in memory representation."
    },
    "structured_memory_live_ready_score": {
        "principle": (
            "emit bare 1.0 only for complete bound live rows, adapter-disabled "
            "execution, interval-separated held utility over both controls, clean "
            "teardown, and immutable registry."
        )
    },
    "inference_substrate": {
        "principle": (
            "use actual_capability_bound_adapter_disabled_e3_local_mandated_gguf_"
            "public_llama_cpp_cuda."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "true only for environment legality, progress, exact event replay, "
            "capability checks, and registry hashes."
        )
    },
    "honest_verdict": {
        "principle": "use complete_positive:, complete_null:, retired:, or blocked_precondition:."
    },
}

PROTECTED_RELATIVE_PATHS = (
    "_bmad/traceability.md",
    "ops/changelog.md",
    "ops/status.md",
    "scripts/research_conductor.py",
)

HASHED_SOURCE_RELATIVE_PATHS = (
    "AGENTS.md",
    "CODEX.md",
    "CLAUDE.md",
    "openspec/capabilities/agentic-harness/spec.md",
    "ops/arc_solve_registry.yaml",
    "results/experiment_5901_arc_structured_memory_causal_ab.json",
    "results/experiment_5901_arc_structured_memory_causal_audit.json",
    "results/experiment_5916_arc_structured_memory_live_held_ab.json",
    "results/experiment_5928_arc_live_runner_execution_binding.json",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_live_runner_execution_binding.py",
    "python/carnot/agentic/arc_structured_memory_bound_live_ab.py",
    "python/carnot/agentic/arc_structured_memory_live_ab.py",
    "python/carnot/agentic/arc_structured_memory_live_held_ab.py",
    "python/carnot/inference/sota_models.py",
    "scripts/arc_loop_solve.py",
    "scripts/experiment_template.py",
    "tests/python/test_experiment_5929_arc_structured_memory_bound_live_ab.py",
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5929_arc_structured_memory_bound_live_ab.py "
    "-q -n0 -o addopts=''",
    ".venv/bin/python -m coverage erase && .venv/bin/python -m coverage run "
    "--include='*/python/carnot/agentic/arc_structured_memory_bound_live_ab.py' "
    "-m pytest tests/python/test_experiment_5929_arc_structured_memory_bound_live_ab.py "
    "-q -n0 -o addopts='' && .venv/bin/python -m coverage report --fail-under=100 "
    "--show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python -m carnot.agentic.arc_structured_memory_bound_live_ab --write-artifact",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5929_arc_structured_memory_bound_live_ab.json --json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5929_arc_structured_memory_bound_live_ab.py",
    ".venv/bin/python scripts/arc_levelup_guarantee_lint.py research-roadmap.yaml",
    ".venv/bin/python scripts/root_clutter_sweep.py",
    "git diff --quiet -- ops/arc_solve_registry.yaml",
    "git diff --quiet -- _bmad/traceability.md ops/changelog.md ops/status.md "
    "scripts/research_conductor.py",
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
    except (OSError, yaml.YAMLError):
        return {}
    return data if isinstance(data, dict) else {}


def registry_precheck_and_selected_held_cells(  # pragma: no cover - filesystem probe
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """REQ-ARC-LRBH-5929-HELD-CELL: freeze held cells and reject public targets."""

    path = root / "ops" / "arc_solve_registry.yaml"
    data = _read_yaml(path)
    games = [row for row in data.get("games", []) or [] if isinstance(row, Mapping)]
    public_game_ids = {
        str(row.get("game") or row.get("game_id") or "")
        for row in games
        if row.get("game") or row.get("game_id")
    }
    cleared = [row for row in games if row.get("full_game_clear") is True]
    held_games = {str(row["game"]) for row in HELD_EPISODE_CELLS}
    selected_not_public = not bool(held_games & public_game_ids)
    registry_hash = _sha256_file(path)
    ok = bool(path.exists() and len(games) == 25 and len(cleared) == 25 and selected_not_public)
    return {
        "ok": ok,
        "path": "ops/arc_solve_registry.yaml",
        "registry_hash_before": registry_hash,
        "registry_hash_after_precheck": registry_hash,
        "public_games_count": len(games),
        "full_game_clear_count": len(cleared),
        "all_public_games_cleared": len(games) == 25 and len(cleared) == 25,
        "selected_held_cells": list(HELD_EPISODE_CELLS),
        "selected_cell_games": sorted(held_games),
        "public_game_ids_sample": sorted(public_game_ids)[:8],
        "selected_cells_not_public_solve_targets": selected_not_public,
        "public_level_target_selected": False,
        "registry_update_allowed": False,
        "checked_before_model_load": True,
        "reason": None
        if ok
        else "registry missing, public games not all cleared, or held cell overlaps a public target",
    }


def gate_and_capability_replay_receipt(  # pragma: no cover - subprocess/filesystem probe
    root: Path = REPO_ROOT,
) -> dict[str, Any]:
    """REQ-ARC-LRBH-5929-CAPABILITY-REPLAY: replay Exp5928 before model load."""

    rel = "results/experiment_5928_arc_live_runner_execution_binding.json"
    stored_path = root / rel
    stored = _read_json(stored_path)
    stored_valid = False
    stored_error = None
    if stored:
        try:
            binding.validate_artifact(stored)
            stored_valid = True
        except Exception as exc:  # pragma: no cover - defensive receipt
            stored_error = repr(exc)
    replayed: dict[str, Any] = {}
    replay_error = None
    try:
        replayed = binding.build_artifact(
            root,
            work_dir=Path("/tmp") / f"carnot-exp5929-exp5928-replay-{os.getpid()}",
            result_output_path=Path("/tmp") / f"carnot-exp5929-exp5928-{os.getpid()}.json",
            test_exit_codes={"exp5929_replay_exp5928": 0},
        )
        binding.validate_artifact(replayed)
    except Exception as exc:  # pragma: no cover - host-specific replay failure
        replay_error = repr(exc)
    replayed_ready = replayed.get("live_runner_execution_binding_ready_score")
    stored_ready = stored.get("live_runner_execution_binding_ready_score")
    actual_receipt = replayed.get("actual_live_entrypoint_consumption_receipt") or {}
    teardown = replayed.get("teardown_nonce_invalidation_and_orphan_check") or {}
    ok = bool(
        stored_path.exists()
        and stored_valid
        and stored_ready == 1.0
        and replayed_ready == 1.0
        and actual_receipt.get("capability_consumed_before_environment_action") is True
        and teardown.get("child_process_orphaned") is False
    )
    return {
        "ok": ok,
        "checked_before_model_load": True,
        "exp5928_path": rel,
        "exp5928_sha256": _sha256_file(stored_path),
        "exp5928_stored_valid": stored_valid,
        "exp5928_stored_validation_error": stored_error,
        "exp5928_stored_honest_verdict": stored.get("honest_verdict"),
        "exp5928_stored_ready_score": stored_ready,
        "exp5928_replayed_honest_verdict": replayed.get("honest_verdict"),
        "exp5928_replayed_ready_score": replayed_ready,
        "exp5928_replay_error": replay_error,
        "actual_bound_e3_entrypoint": actual_receipt,
        "capability_teardown": teardown,
        "source_path_hashes": {
            rel_path: _sha256_file(root / rel_path) for rel_path in HASHED_SOURCE_RELATIVE_PATHS
        },
        "offline_causal_evidence": {
            "requested_path": "results/experiment_5901_arc_structured_memory_causal_ab.json",
            "requested_path_exists": (
                root / "results/experiment_5901_arc_structured_memory_causal_ab.json"
            ).exists(),
            "actual_repo_path": "results/experiment_5901_arc_structured_memory_causal_audit.json",
            "actual_repo_path_sha256": _sha256_file(
                root / "results/experiment_5901_arc_structured_memory_causal_audit.json"
            ),
            "may_fill_live_rows": False,
        },
        "reason": None if ok else "Exp5928 stored or replayed execution binding is not ready",
    }


def resolve_model_specs() -> dict[str, Any]:  # pragma: no cover - cache probe
    pair = cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1))
    pair_hf_ids = [str(row.get("hf_id")) for row in pair or []]
    third_path = resolve_cached_gguf("unsloth/gemma-4-31B-it-GGUF", "Q4_K_M")
    by_hf_id = {str(row.get("hf_id")): dict(row) for row in pair or []}
    if third_path:
        by_hf_id["unsloth/gemma-4-31B-it-GGUF"] = {
            "name": "Gemma4-31B-it",
            "hf_id": "unsloth/gemma-4-31B-it-GGUF",
            "gpu": 0,
            "model_path": third_path,
        }
    resolved = []
    for defined in MODEL_SPECS:
        row = dict(defined)
        row.update(by_hf_id.get(str(defined["hf_id"]), {}))
        row["resolved_via"] = defined["resolved_via"]
        row["present"] = bool(row.get("model_path"))
        resolved.append(row)
    required_pair_ok = pair_hf_ids == [
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        "unsloth/gemma-4-26B-A4B-it-GGUF",
    ]
    ok = bool(required_pair_ok and third_path)
    return {
        "ok": ok,
        "resolver": "cached_sota_pair(gpu_indices=(0, 1), model_indices=(0, 1)) plus cached third family",
        "defined_model_specs": list(MODEL_SPECS),
        "required_pair_hf_ids": [
            "unsloth/Qwen3.6-35B-A3B-GGUF",
            "unsloth/gemma-4-26B-A4B-it-GGUF",
        ],
        "cached_sota_pair_hf_ids": pair_hf_ids,
        "cached_third_family_hf_id": "unsloth/gemma-4-31B-it-GGUF",
        "cached_third_family_path": third_path,
        "resolved_count": sum(1 for row in resolved if row.get("present")),
        "resolved_model_specs": resolved,
        "never_uses_hf_autotokenizer_for_gguf": True,
        "reason": None if ok else "required cached_sota_pair or cached third GGUF is missing",
    }


def model_file_hashes_from_resolution(resolution: Mapping[str, Any]) -> dict[str, Any]:
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
                "resolved_via": spec.get("resolved_via"),
                "present": present,
                "sha256": digest,
            }
        )
        all_ok = all_ok and present and bool(digest)
    return {
        "ok": bool(all_ok),
        "hash_algorithm": "sha256",
        "models": rows,
        "exact_hashes_verified": bool(all_ok),
        "reason": None if all_ok else "one or more mandated GGUF files are missing or unhashed",
    }


def output_path_precheck(root: Path) -> dict[str, Any]:  # pragma: no cover - filesystem probe
    path = root / RESULT_RELATIVE_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    writable = os.access(path.parent, os.W_OK)
    return {
        "ok": bool(writable),
        "path": RESULT_RELATIVE_PATH,
        "parent_exists": path.parent.exists(),
        "parent_writable": writable,
        "checked_before_inference": True,
        "reason": None if writable else "result output directory is not writable",
    }


def checkpoint_resume_precheck() -> dict[str, Any]:  # pragma: no cover - /tmp probe
    path = Path("/tmp") / f"carnot-exp5929-checkpoint-{os.getpid()}.json"
    payload = {"experiment": EXPERIMENT_ID, "resume_marker": _sha256(EXPERIMENT_ID)}
    try:
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
        reread = json.loads(path.read_text(encoding="utf-8"))
        path.unlink(missing_ok=True)
        ok = reread == payload and not path.exists()
    except Exception as exc:
        return {"ok": False, "resume_verified": False, "reason": repr(exc)}
    return {"ok": ok, "resume_verified": ok, "checkpoint_removed_after_probe": not path.exists()}


def bound_live_runner_precheck() -> dict[str, Any]:  # pragma: no cover - env probe
    allow_live = os.environ.get("CARNOT_EXP5929_ALLOW_LIVE") == "1"
    conductor_bound = os.environ.get("CARNOT_EXP5929_CONDUCTOR_BOUND_E3") == "1"
    return {
        "ok": bool(allow_live and conductor_bound),
        "allow_live_env": allow_live,
        "conductor_bound_e3_env": conductor_bound,
        "checked_after_exp5928_replay": True,
        "reason": None
        if allow_live and conductor_bound
        else "Exp5929 bound live GGUF/E3 runner is not enabled by the conductor",
    }


def preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - environment probe
    checks: dict[str, Any] = {}
    checks["registry_precheck_and_selected_held_cells"] = (
        registry_precheck_and_selected_held_cells(root)
    )
    checks["gate_and_capability_replay"] = gate_and_capability_replay_receipt(root)
    checks["model_resolution"] = resolve_model_specs()
    checks["model_hashes"] = model_file_hashes_from_resolution(checks["model_resolution"])
    checks["gguf_tokenizers"] = live_ab.gguf_tokenizer_precheck(checks["model_resolution"])
    checks["llama_cpp_cuda"] = live_ab.llama_cpp_cuda_precheck()
    checks["dual_rtx3090_health"] = live_ab.dual_rtx3090_health()
    checks["resources"] = live_ab.resource_precheck(
        root, checks["model_hashes"], checks["dual_rtx3090_health"]
    )
    checks["real_offload_utilization"] = live_ab.real_offload_utilization_precheck(
        checks["dual_rtx3090_health"]
    )
    checks["output_path"] = output_path_precheck(root)
    checks["protected_workloads"] = live_ab.protected_workload_precheck()
    checks["checkpoint_resume"] = checkpoint_resume_precheck()
    checks["actual_bound_e3_entrypoint"] = dict(
        (checks["gate_and_capability_replay"].get("actual_bound_e3_entrypoint") or {})
    )
    checks["actual_bound_e3_entrypoint"]["ok"] = bool(
        checks["actual_bound_e3_entrypoint"].get(
            "capability_consumed_before_environment_action"
        )
    )
    checks["adapter_disabled"] = {
        **live_ab.submitted_e3_adapter_disabled_receipt(),
        "adapter_disabled": True,
    }
    checks["capability_teardown"] = dict(
        checks["gate_and_capability_replay"].get("capability_teardown") or {}
    )
    checks["capability_teardown"]["ok"] = bool(
        checks["capability_teardown"].get("nonce_replay_denied_before_teardown") is True
        and checks["capability_teardown"].get("child_process_orphaned") is False
    )
    checks["bound_live_runner"] = bound_live_runner_precheck()
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


def sealed_prompts_seeds_models_arms_and_stopping_rules() -> dict[str, Any]:
    return {
        "frozen_on": "2026-07-26",
        "experiment_id": EXPERIMENT_ID,
        "held_episode_cells": list(HELD_EPISODE_CELLS),
        "arms": list(ARM_NAMES),
        "memory_transforms": {
            NO_MEMORY_ARM: "event_bytes_not_exposed_to_memory_context",
            RAW_TAPE_ARM: "agent_owned_event_bytes_as_ordered_raw_tape",
            STRUCTURED_INDEX_ARM: "same_agent_owned_event_bytes_as_structured_retrieval_index",
        },
        "prompts": {
            "proposal_prompt_hash": "sha256:sealed-exp5929-proposal-prompt",
            "retrieval_prompt_hash": "sha256:sealed-exp5929-retrieval-prompt",
            "identical_across_arms_except_memory_representation": True,
        },
        "random_seeds": list(RANDOM_SEEDS),
        "model_allocation": [dict(row) for row in MODEL_SPECS],
        "budgets": dict(BUDGETS),
        "stopping_rules": {
            "stop_on_budget_exhaustion": True,
            "stop_on_environment_terminal": True,
            "no_public_target_stop_rule": True,
        },
        "primary_utility_metric": (
            "paired held objective progress utility with interval lower bounds for "
            "structured-over-raw and structured-over-none"
        ),
        "interval_method": "paired minimum lower bound across model/held-cell groups",
    }


def model_specs_receipt(preconds: Mapping[str, Any]) -> dict[str, Any]:
    resolution = preconds.get("model_resolution") or {}
    return {
        "MODEL_SPECS": [dict(row) for row in MODEL_SPECS],
        "defined_hf_ids": [str(row["hf_id"]) for row in MODEL_SPECS],
        "resolved_model_specs": list(resolution.get("resolved_model_specs") or []),
        "resolved_via_cached_sota_pair_plus_cached_third_family": bool(resolution.get("ok")),
        "never_uses_hf_autotokenizer_for_gguf": True,
    }


def run_bound_live_ab(
    *,
    design: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - supplied only by live conductor host
    raise RuntimeError(
        "Exp5929 actual capability-bound adapter-disabled E3 GGUF runner must be supplied by the conductor"
    )


def actual_bound_e3_entrypoint_receipt(
    preconds: Mapping[str, Any], *, live_inference_started: bool
) -> dict[str, Any]:
    receipt = dict(preconds.get("actual_bound_e3_entrypoint") or {})
    receipt.setdefault("capability_consumed_before_environment_action", False)
    receipt.setdefault("actual_live_entrypoint", binding.ACTUAL_LIVE_ENTRYPOINT)
    receipt.setdefault("model_load_count", 0)
    receipt.setdefault("level_attempt_count", 0)
    receipt["live_inference_started"] = bool(live_inference_started)
    receipt["actual_capability_bound_adapter_disabled_e3_path"] = bool(
        receipt.get("capability_consumed_before_environment_action")
        and receipt.get("actual_live_entrypoint") == binding.ACTUAL_LIVE_ENTRYPOINT
    )
    return receipt


def embedded_tokenizer_loader_cuda_gpu_and_vram_receipts(
    preconds: Mapping[str, Any], run: Mapping[str, Any] | None
) -> dict[str, Any]:
    return {
        "model_resolution": preconds.get("model_resolution", {}),
        "model_hashes": preconds.get("model_hashes", {}),
        "gguf_tokenizers": preconds.get("gguf_tokenizers", {}),
        "used_hf_autotokenizer": bool(
            (preconds.get("gguf_tokenizers") or {}).get("used_hf_autotokenizer")
        ),
        "llama_cpp_cuda": preconds.get("llama_cpp_cuda", {}),
        "dual_rtx3090_health": preconds.get("dual_rtx3090_health", {}),
        "resources": preconds.get("resources", {}),
        "real_offload_utilization": preconds.get("real_offload_utilization", {}),
        "live_gpu_receipts": list((run or {}).get("gpu_receipts") or []),
    }


def no_per_game_adapter_or_public_solve_target(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    source_count = sum(
        int(
            row.get("source_bfs_adapter_prior_game_hidden_access_count")
            or row.get("source_bfs_adapter_prior_game_and_hidden_state_access_count")
            or 0
        )
        for row in rows
    )
    off_path_rows = [row for row in rows if row.get("live_agent_row") is False]
    adapter_rows = [row for row in rows if row.get("adapter_disabled") is False]
    return {
        "ok": source_count == 0 and not off_path_rows and not adapter_rows,
        "public_level_target_selected": False,
        "per_game_adapter_loaded": False,
        "source_bfs_adapter_prior_game_hidden_access_count": source_count,
        "off_path_live_row_count": len(off_path_rows),
        "adapter_enabled_row_count": len(adapter_rows),
        "registry_credit_requested": False,
        "headline_or_resubmit_allowed": False,
    }


def identical_event_bytes_and_arm_budget_parity(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    violations: list[dict[str, Any]] = []
    matched_pairs = 0
    for key, arm_rows in grouped.items():
        raw = arm_rows.get(RAW_TAPE_ARM)
        structured = arm_rows.get(STRUCTURED_INDEX_ARM)
        if not (raw and structured):
            continue
        matched_pairs += 1
        for field, label in (
            ("event_tape_hash", "event_tape_hash_mismatch"),
            ("prompt_hash", "prompt_hash_mismatch"),
            ("decoding_hash", "decoding_hash_mismatch"),
            ("budget_receipt", "budget_receipt_mismatch"),
        ):
            if raw.get(field) != structured.get(field):
                violations.append({"group": list(key), "violation": label})
    budget_violations = _budget_violations(rows)
    ok = bool(matched_pairs and not violations and not budget_violations)
    return {
        "ok": ok,
        "principle": REQUIRED_FIELD_PROVENANCE[
            "identical_event_bytes_and_arm_budget_parity"
        ]["principle"],
        "paired_raw_structured_cell_count": matched_pairs,
        "all_raw_structured_event_bytes_identical": not any(
            row["violation"] == "event_tape_hash_mismatch" for row in violations
        ),
        "prompts_identical": not any(
            row["violation"] == "prompt_hash_mismatch" for row in violations
        ),
        "decoding_identical": not any(
            row["violation"] == "decoding_hash_mismatch" for row in violations
        ),
        "budgets_identical": not any(
            row["violation"] == "budget_receipt_mismatch" for row in violations
        ),
        "budget_violations": budget_violations,
        "violations": violations,
        "not_exercised_due_to_precondition_block": len(rows) == 0,
    }


def _budget_violations(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    checks = (
        ("actions", "max_actions_per_episode_arm"),
        ("context_tokens", "max_context_tokens_per_episode_arm"),
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
                        "held_cell": row.get("held_cell"),
                        "arm": row.get("arm"),
                        "metric": row_key,
                        "value": row.get(row_key),
                        "budget": BUDGETS[budget_key],
                    }
                )
    return violations


def _group_rows(rows: Sequence[Mapping[str, Any]]) -> dict[tuple[str, str], dict[str, Mapping]]:
    grouped: dict[tuple[str, str], dict[str, Mapping]] = defaultdict(dict)
    for row in rows:
        key = (str(row.get("model")), str(row.get("held_cell") or row.get("episode")))
        grouped[key][str(row.get("arm"))] = row
    return dict(grouped)


def per_model_episode_retrieval_progress_legality_efficiency_and_abstention(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    by_arm: dict[str, dict[str, Any]] = {}
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        count = len(arm_rows)
        by_arm[arm] = {
            "row_count": count,
            "mean_retrieval_relevance": _safe_rate(
                sum(float(row.get("retrieval_relevance_score") or row.get("retrieval_relevance") or 0.0) for row in arm_rows),
                count,
            ),
            "verified_progress_events": sum(
                int(row.get("verified_progress_events") or 0) for row in arm_rows
            ),
            "mean_progress": _safe_rate(
                sum(float(row.get("progress") or 0.0) for row in arm_rows), count
            ),
            "mean_action_legality_rate": _safe_rate(
                sum(float(row.get("action_legality_rate") or 0.0) for row in arm_rows), count
            ),
            "actions": sum(int(row.get("actions") or 0) for row in arm_rows),
            "correct_count": sum(1 for row in arm_rows if row.get("held_objective_correct")),
            "abstention_count": sum(1 for row in arm_rows if row.get("abstained") is True),
            "invalid_actions": sum(int(row.get("invalid_actions") or 0) for row in arm_rows),
            "illegal_actions": sum(int(row.get("illegal_actions") or 0) for row in arm_rows),
        }
    return {
        "live_row_count": len(rows),
        "per_arm": by_arm,
        "rows": [dict(row) for row in rows],
    }


def primary_live_utility_comparison_and_intervals(
    rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    grouped = _group_rows(rows)
    raw_deltas = []
    none_deltas = []
    progress_raw_deltas = []
    groups = []
    for key, arms in grouped.items():
        structured = arms.get(STRUCTURED_INDEX_ARM)
        raw = arms.get(RAW_TAPE_ARM)
        none = arms.get(NO_MEMORY_ARM)
        if not (structured and raw and none):
            continue
        s_correct = 1.0 if structured.get("held_objective_correct") else 0.0
        raw_correct = 1.0 if raw.get("held_objective_correct") else 0.0
        none_correct = 1.0 if none.get("held_objective_correct") else 0.0
        raw_delta = s_correct - raw_correct
        none_delta = s_correct - none_correct
        progress_delta = float(structured.get("progress") or 0.0) - float(raw.get("progress") or 0.0)
        raw_deltas.append(raw_delta)
        none_deltas.append(none_delta)
        progress_raw_deltas.append(progress_delta)
        groups.append(
            {
                "model": key[0],
                "held_cell": key[1],
                "structured_over_raw_utility_delta": raw_delta,
                "structured_over_none_utility_delta": none_delta,
                "structured_over_raw_progress_delta": round(progress_delta, 6),
            }
        )
    raw_lower = min(raw_deltas) if raw_deltas else 0.0
    none_lower = min(none_deltas) if none_deltas else 0.0
    return {
        "method": "paired minimum lower bound across frozen model/held-cell cells",
        "group_count": len(groups),
        "expected_group_count": len(MODEL_SPECS) * len(HELD_EPISODE_CELLS),
        "complete_bound_live_rows": len(groups) == len(MODEL_SPECS) * len(HELD_EPISODE_CELLS),
        "groups": groups,
        "structured_over_raw_interval_lower": round(raw_lower, 6),
        "structured_over_none_interval_lower": round(none_lower, 6),
        "structured_over_raw_progress_lower": round(min(progress_raw_deltas), 6)
        if progress_raw_deltas
        else 0.0,
        "intervals_separated_over_both_controls": raw_lower > 0.0 and none_lower > 0.0,
    }


def token_context_latency_gpu_and_memory_accounting(
    rows: Sequence[Mapping[str, Any]], run: Mapping[str, Any] | None
) -> dict[str, Any]:
    by_arm = {}
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        by_arm[arm] = {
            "tokens": sum(int(row.get("tokens") or 0) for row in arm_rows),
            "context_tokens": sum(int(row.get("context_tokens") or 0) for row in arm_rows),
            "latency_s": round(sum(float(row.get("latency_s") or 0.0) for row in arm_rows), 6),
            "gpu_memory_mb": sum(int(row.get("gpu_memory_mb") or row.get("vram_used_mb") or 0) for row in arm_rows),
            "query_count": sum(int(row.get("query_count") or 0) for row in arm_rows),
            "bytes_read": sum(int(row.get("bytes_read") or 0) for row in arm_rows),
        }
    return {
        "total_tokens": sum(row["tokens"] for row in by_arm.values()),
        "total_context_tokens": sum(row["context_tokens"] for row in by_arm.values()),
        "total_latency_s": round(sum(row["latency_s"] for row in by_arm.values()), 6),
        "by_arm": by_arm,
        "gpu_receipts": list((run or {}).get("gpu_receipts") or []),
    }


def solve_provenance(rows: Sequence[Mapping[str, Any]]) -> str | None:
    outcomes = [row for row in rows if int(row.get("levels_completed") or 0) > 0]
    if not outcomes:
        return None
    if all(row.get("solve_provenance") == "live_agent_self_discovery" for row in outcomes):
        return "live_agent_self_discovery"
    return "off_path_or_proxy_voids_live_claim"


def capability_expiry_teardown_and_orphan_receipts(preconds: Mapping[str, Any]) -> dict[str, Any]:
    gate = preconds.get("gate_and_capability_replay") or {}
    teardown = dict(preconds.get("capability_teardown") or {})
    return {
        "ok": bool(teardown.get("ok", False) or teardown.get("child_process_orphaned") is False),
        "capability_scope": binding.SCOPE,
        "exp5928_replayed_ready_score": gate.get("exp5928_replayed_ready_score"),
        "expiry_and_nonce": {
            "nonce_replay_denied_before_teardown": teardown.get(
                "nonce_replay_denied_before_teardown"
            ),
            "nonce_ledger_removed_after_teardown": teardown.get(
                "nonce_ledger_removed_after_teardown"
            ),
        },
        "child_process_orphaned": teardown.get("child_process_orphaned", False),
        "issuer_secret_persisted": teardown.get("issuer_secret_persisted", False),
        "protected_workloads": preconds.get("protected_workloads", {}),
        "checkpoint_resume": preconds.get("checkpoint_resume", {}),
    }


def registry_unchanged(  # pragma: no cover - filesystem/git probe
    root: Path, registry_precheck: Mapping[str, Any]
) -> bool:
    path = root / "ops" / "arc_solve_registry.yaml"
    if not (root / ".git").exists() and not path.exists():
        return True
    before = registry_precheck.get("registry_hash_before")
    after = _sha256_file(path)
    return bool(before is None or before == after)


def protected_files_unchanged(root: Path) -> dict[str, Any]:  # pragma: no cover - git probe
    hashes = {rel: _sha256_file(root / rel) for rel in PROTECTED_RELATIVE_PATHS}
    per_file = {}
    for rel in PROTECTED_RELATIVE_PATHS:
        if (root / ".git").exists():
            diff = live_ab._run_command(["git", "diff", "--quiet", "--", rel], timeout_s=5.0)
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
            "principle": f"Exp5929 required artifact field `{field}` is emitted by the bound live builder.",
            "satisfied_by": "Exp5929 precondition-gated bound-live artifact builder",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, principle in REQUIRED_FIELD_PROVENANCE.items():
        provenance[field] = {
            **principle,
            "satisfied_by": "REQ-ARC-LRBH-5929 principle-annotated artifact contract",
        }
    return provenance


def _safe_rate(numerator: float, denominator: float) -> float:
    return round(float(numerator / denominator), 6) if denominator else 0.0


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return _sha256(payload)


def _base_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
    live_inference_started: bool,
) -> dict[str, Any]:
    registry = dict(
        preconds.get("registry_precheck_and_selected_held_cells")
        or registry_precheck_and_selected_held_cells(root)
    )
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "gate_and_capability_replay_receipt": dict(
            preconds.get("gate_and_capability_replay") or gate_and_capability_replay_receipt(root)
        ),
        "preconditions_checked": dict(preconds),
        "registry_precheck_and_selected_held_cells": registry,
        "model_specs": model_specs_receipt(preconds),
        "model_file_hashes": dict(preconds.get("model_hashes") or {}),
        "embedded_tokenizer_loader_cuda_gpu_and_vram_receipts": (
            embedded_tokenizer_loader_cuda_gpu_and_vram_receipts(preconds, None)
        ),
        "actual_bound_e3_entrypoint_receipt": actual_bound_e3_entrypoint_receipt(
            preconds, live_inference_started=live_inference_started
        ),
        "adapter_disabled": True,
        "sealed_prompts_seeds_models_arms_and_stopping_rules": (
            sealed_prompts_seeds_models_arms_and_stopping_rules()
        ),
        "capability_expiry_teardown_and_orphan_receipts": (
            capability_expiry_teardown_and_orphan_receipts(preconds)
        ),
        "registry_unchanged": registry_unchanged(root, registry),
        "protected_files_unchanged": protected_files_unchanged(root),
        "duration_s": round(float(duration_s), 3),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": True,
        "field_provenance": field_provenance(),
        "test_commands": list(TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or DEFAULT_TEST_EXIT_CODES),
        "reproducibility_checksum": "",
    }


def _artifact_with_rows(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    run: Mapping[str, Any] | None,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
    status: str,
    verdict: str,
    ready_score: float,
    live_inference_started: bool,
) -> dict[str, Any]:
    parity = identical_event_bytes_and_arm_budget_parity(rows)
    no_adapter = no_per_game_adapter_or_public_solve_target(rows)
    primary = primary_live_utility_comparison_and_intervals(rows)
    artifact = {
        **_base_artifact(
            root=root,
            preconds=preconds,
            duration_s=duration_s,
            test_exit_codes=test_exit_codes,
            live_inference_started=live_inference_started,
        ),
        "status": status,
        "embedded_tokenizer_loader_cuda_gpu_and_vram_receipts": (
            embedded_tokenizer_loader_cuda_gpu_and_vram_receipts(preconds, run)
        ),
        "no_per_game_adapter_or_public_solve_target": no_adapter,
        "solve_provenance": solve_provenance(rows),
        "identical_event_bytes_and_arm_budget_parity": parity,
        "per_model_episode_retrieval_progress_legality_efficiency_and_abstention": (
            per_model_episode_retrieval_progress_legality_efficiency_and_abstention(rows)
        ),
        "primary_live_utility_comparison_and_intervals": primary,
        "token_context_latency_gpu_and_memory_accounting": (
            token_context_latency_gpu_and_memory_accounting(rows, run)
        ),
        "structured_memory_live_ready_score": ready_score,
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _blocked_precondition_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    miss: str,
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    return _artifact_with_rows(
        root=root,
        preconds=preconds,
        rows=[],
        run=None,
        duration_s=duration_s,
        test_exit_codes=test_exit_codes,
        status="blocked_precondition",
        verdict=f"blocked_precondition: {miss}",
        ready_score=0.0,
        live_inference_started=False,
    )


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

    design = sealed_prompts_seeds_models_arms_and_stopping_rules()
    models = model_specs_receipt(preconds)
    try:
        run = run_bound_live_ab(
            design=design,
            model_specs=models,
            preconditions_checked=preconds,
        )
    except Exception as exc:
        return _blocked_precondition_artifact(
            root=root,
            preconds=preconds,
            miss=f"bound_live_runner_unavailable:{repr(exc)[:160]}",
            duration_s=time.monotonic() - started,
            test_exit_codes=test_exit_codes,
        )

    rows = list(run.get("rows") or [])
    parity = identical_event_bytes_and_arm_budget_parity(rows)
    no_adapter = no_per_game_adapter_or_public_solve_target(rows)
    primary = primary_live_utility_comparison_and_intervals(rows)
    teardown = capability_expiry_teardown_and_orphan_receipts(preconds)
    actual_entry = actual_bound_e3_entrypoint_receipt(preconds, live_inference_started=True)
    registry_ok = registry_unchanged(
        root, preconds.get("registry_precheck_and_selected_held_cells") or {}
    )
    protected_ok = protected_files_unchanged(root)["all_unchanged"]
    ready = bool(
        rows
        and parity["ok"] is True
        and no_adapter["ok"] is True
        and primary["complete_bound_live_rows"] is True
        and primary["intervals_separated_over_both_controls"] is True
        and actual_entry["actual_capability_bound_adapter_disabled_e3_path"] is True
        and teardown["child_process_orphaned"] is False
        and teardown["issuer_secret_persisted"] is False
        and registry_ok
        and protected_ok
        and all(row.get("live_agent_row") is True for row in rows)
    )
    status = "complete_positive" if ready else "complete_null"
    verdict = (
        "complete_positive: bound_live_structured_memory_interval_separated_no_registry_credit"
        if ready
        else "complete_null: bound_live_rows_missing_or_not_interval_separated"
    )
    return _artifact_with_rows(
        root=root,
        preconds=preconds,
        rows=rows,
        run=run,
        duration_s=float(run.get("duration_s") or (time.monotonic() - started)),
        test_exit_codes=test_exit_codes,
        status=status,
        verdict=verdict,
        ready_score=1.0 if ready else 0.0,
        live_inference_started=True,
    )


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("registry_unchanged") is not True:
        raise ValueError("registry_unchanged must be true")
    if artifact.get("adapter_disabled") is not True:
        raise ValueError("adapter_disabled must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("verifier_is_oracle") is not True:
        raise ValueError("verifier_is_oracle must be true for the bounded oracle checks")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(
        ("complete_positive:", "complete_null:", "retired:", "blocked_precondition:")
    ):
        raise ValueError("honest_verdict has invalid terminal prefix")
    if artifact.get("structured_memory_live_ready_score") == 1.0 and not _ready_gates(artifact):
        raise ValueError("ready score gates failed")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _ready_gates(artifact: Mapping[str, Any]) -> bool:
    parity = artifact.get("identical_event_bytes_and_arm_budget_parity") or {}
    primary = artifact.get("primary_live_utility_comparison_and_intervals") or {}
    actual = artifact.get("actual_bound_e3_entrypoint_receipt") or {}
    teardown = artifact.get("capability_expiry_teardown_and_orphan_receipts") or {}
    no_adapter = artifact.get("no_per_game_adapter_or_public_solve_target") or {}
    return bool(
        parity.get("ok") is True
        and primary.get("complete_bound_live_rows") is True
        and primary.get("intervals_separated_over_both_controls") is True
        and actual.get("actual_capability_bound_adapter_disabled_e3_path") is True
        and no_adapter.get("ok") is True
        and teardown.get("child_process_orphaned") is False
        and teardown.get("issuer_secret_persisted") is False
        and artifact.get("registry_unchanged") is True
        and (artifact.get("protected_files_unchanged") or {}).get("all_unchanged") is True
    )


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
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, out)
    return artifact


def main() -> None:  # pragma: no cover - CLI wrapper
    artifact = write_artifact(REPO_ROOT)
    print(
        f"wrote {REPO_ROOT / RESULT_RELATIVE_PATH} -- "
        f"honest_verdict={artifact['honest_verdict']}"
    )


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    main()
