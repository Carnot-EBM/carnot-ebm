"""Exp5916 held live E3 A/B for structured ARC evidence memory.

Spec refs: REQ-ARC-LRHL-5916,
SCENARIO-ARC-LRHL-5916-PRECONDITION-BLOCK,
SCENARIO-ARC-LRHL-5916-MATCHED-HELD-LIVE-AB,
SCENARIO-ARC-LRHL-5916-CAUSAL-CONTROLS,
SCENARIO-ARC-LRHL-5916-NO-SOLVE-CREDIT.

The module is strict about the live boundary. It can emit a complete blocked
artifact when a host lacks the conductor-bound runner, but it must never invent
held live rows. Tests monkeypatch the live runner to exercise the positive and
unsafe branches; the default repository path remains honest about local gates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import time
from typing import Any

from carnot.agentic import arc_live_runner_capability_lease as capability
from carnot.agentic import arc_structured_memory_live_ab as live_ab
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_ID = "experiment_5916_arc_structured_memory_live_held_ab"
RESULT_RELATIVE_PATH = "results/experiment_5916_arc_structured_memory_live_held_ab.json"
SCHEMA = "carnot.exp5916.arc_structured_memory_live_held_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"

NO_MEMORY_ARM = live_ab.NO_MEMORY_ARM
RAW_TAPE_ARM = live_ab.RAW_TAPE_ARM
STRUCTURED_INDEX_ARM = live_ab.STRUCTURED_INDEX_ARM
ARM_NAMES = (NO_MEMORY_ARM, RAW_TAPE_ARM, STRUCTURED_INDEX_ARM)
RANDOM_SEEDS = (2026072501, 2026072502)

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
    "max_actions_per_episode_arm": 180,
    "max_tokens_per_episode_arm": 4096,
    "max_wall_clock_s_per_episode_arm": 300.0,
    "max_queries_per_episode_arm": 16,
    "max_event_bytes_per_episode_arm": 262_144,
}

PREREGISTERED_HELD_EPISODES = (
    {
        "group": "held_live_generalization_navigation",
        "game": "held-gamma",
        "episode": "held-ep-0001",
        "measurement_role": "held_out_navigation_like",
    },
    {
        "group": "held_live_generalization_object_interaction",
        "game": "held-delta",
        "episode": "held-ep-0002",
        "measurement_role": "held_out_interaction_like",
    },
)

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "preconditions_checked",
    "upstream_capability_gate_and_hashes",
    "registry_precheck",
    "public_level_solve_claimed",
    "preregistered_held_episode_group_and_arm_design",
    "model_specs",
    "model_file_hashes",
    "embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts",
    "capability_scope_expiry_and_environment_receipts",
    "submitted_e3_and_adapter_disabled_receipts",
    "identical_event_byte_and_budget_parity",
    "no_memory_raw_and_structured_live_metrics",
    "held_accuracy_progress_efficiency_and_safety_metrics",
    "per_model_game_episode_lower_bounds",
    "shuffled_and_deletion_confirmatory_controls",
    "evidence_utilization_receipts",
    "source_bfs_adapter_prior_game_and_hidden_state_access_count",
    "incidental_completion_receipts",
    "registry_unchanged",
    "state_isolation_and_teardown_receipts",
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
        "principle": "memory access structure is the only treatment."
    },
    "source_bfs_adapter_prior_game_and_hidden_state_access_count": {
        "principle": "must be bare zero."
    },
    "structured_memory_live_ready_score": {
        "principle": (
            "emit bare 1.0 only for positive preregistered structured-over-raw and "
            "structured-over-none lower bounds with no safety, budget, capability, "
            "or authority regression."
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
    "openspec/capabilities/agentic-harness/spec.md",
    "python/carnot/agentic/arc_structured_memory_live_held_ab.py",
    "python/carnot/agentic/arc_live_runner_capability_lease.py",
    "python/carnot/agentic/arc_structured_memory_live_ab.py",
    "python/carnot/agentic/arc_structured_memory_causal_audit.py",
    "python/carnot/agentic/arc_structured_evidence_memory.py",
    "python/carnot/agentic/arc_competition_agent.py",
    "python/carnot/agentic/arc_typed_memory_provenance_guard.py",
    "python/carnot/inference/sota_models.py",
    "tests/python/test_experiment_5916_arc_structured_memory_live_held_ab.py",
)

TEST_COMMANDS = (
    ".venv/bin/pytest tests/python/test_experiment_5916_arc_structured_memory_live_held_ab.py "
    "-q -n0 -o addopts=''",
    ".venv/bin/python -m coverage erase && .venv/bin/python -m coverage run "
    "--include='*/python/carnot/agentic/arc_structured_memory_live_held_ab.py' "
    "-m pytest tests/python/test_experiment_5916_arc_structured_memory_live_held_ab.py "
    "-q -n0 -o addopts='' && .venv/bin/python -m coverage report --fail-under=100 "
    "--show-missing",
    ".venv/bin/pytest tests/python -q",
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_5916_arc_structured_memory_live_held_ab.json --json",
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_5916_arc_structured_memory_live_held_ab.py",
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


def registry_precheck(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    return live_ab.registry_precheck(root)


def upstream_capability_gate_and_hashes(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    rel = "results/experiment_5915_arc_live_runner_capability_lease.json"
    path = root / rel
    stored = _read_json(path)
    stored_valid = False
    stored_error = None
    if stored:
        try:
            capability.validate_artifact(stored)
            stored_valid = True
        except Exception as exc:
            stored_error = repr(exc)

    replayed = capability.build_artifact(root=root, test_exit_codes={"exp5915_replay": 0})
    replay_valid = False
    replay_error = None
    try:
        capability.validate_artifact(replayed)
        replay_valid = True
    except Exception as exc:
        replay_error = repr(exc)

    source_hashes = {relative: _sha256_file(root / relative) for relative in SOURCE_HASH_RELATIVE_PATHS}
    ok = bool(
        path.exists()
        and stored_valid
        and replay_valid
        and stored.get("live_runner_capability_ready_score") == 1.0
        and replayed.get("live_runner_capability_ready_score") == 1.0
        and stored.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") == 0
        and replayed.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") == 0
    )
    return {
        "ok": ok,
        "checked_before_model_load": True,
        "exp5915_path": rel,
        "exp5915_sha256": _sha256_file(path),
        "stored_artifact_valid": stored_valid,
        "stored_validation_error": stored_error,
        "stored_honest_verdict": stored.get("honest_verdict"),
        "stored_ready_score": stored.get("live_runner_capability_ready_score"),
        "replayed_artifact_valid": replay_valid,
        "replay_validation_error": replay_error,
        "replayed_ready_score": replayed.get("live_runner_capability_ready_score"),
        "replayed_honest_verdict": replayed.get("honest_verdict"),
        "exp5900_sha256": _sha256_file(
            root / "results/experiment_5900_arc_structured_evidence_memory_contract.json"
        ),
        "exp5901_sha256": _sha256_file(
            root / "results/experiment_5901_arc_structured_memory_causal_audit.json"
        ),
        "exp5902_sha256": _sha256_file(
            root / "results/experiment_5902_arc_structured_memory_live_ab.json"
        ),
        "source_path_hashes": source_hashes,
        "reason": None if ok else "Exp5915 capability gate is absent, invalid, or not ready",
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


def live_runner_execution_binding_precheck() -> dict[str, Any]:  # pragma: no cover
    allow_live = os.environ.get("CARNOT_EXP5916_ALLOW_LIVE") == "1"
    conductor_bound = os.environ.get("CARNOT_EXP5916_CONDUCTOR_LIVE_RUNNER") == "1"
    ok = bool(allow_live and conductor_bound)
    return {
        "ok": ok,
        "allow_live_env": allow_live,
        "conductor_live_runner_bound": conductor_bound,
        "checked_after_exp5915_capability_replay": True,
        "reason": None if ok else "Exp5916 live held runner is not bound by the conductor",
    }


def state_isolation_teardown_precheck() -> dict[str, Any]:  # pragma: no cover
    dry_run = capability.run_bounded_non_scored_dry_run(capability.default_conductor_binding())
    return capability.state_isolation_and_teardown_receipts(dry_run)


def preconditions(root: Path = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover
    checks: dict[str, Any] = {}
    checks["registry_precheck"] = registry_precheck(root)
    checks["upstream_capability_gate"] = upstream_capability_gate_and_hashes(root)
    checks["model_resolution"] = resolve_preregistered_model_specs()
    checks["model_hashes"] = live_ab.model_file_hashes_from_resolution(checks["model_resolution"])
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
    checks["submitted_e3_adapter_disabled"] = live_ab.submitted_e3_adapter_disabled_receipt()
    checks["state_isolation_teardown"] = state_isolation_teardown_precheck()
    checks["live_runner_execution_binding"] = live_runner_execution_binding_precheck()
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


def preregistered_held_episode_group_and_arm_design() -> dict[str, Any]:
    return {
        "frozen_on": "2026-07-25",
        "experiment_id": EXPERIMENT_ID,
        "episode_groups": list(PREREGISTERED_HELD_EPISODES),
        "arms": list(ARM_NAMES),
        "arm_order": {
            "counterbalance_seed": RANDOM_SEEDS[0],
            "orders": _counterbalanced_orders(),
            "state_isolation": "fresh E3AgentPolicy, environment, proposer, and memory state per cell",
        },
        "random_seeds": list(RANDOM_SEEDS),
        "budgets": dict(BUDGETS),
        "primary_metrics": [
            "held_objective_accuracy",
            "mean_environment_score",
            "mean_progress",
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
            "capability_or_authority_regression": False,
        },
        "proposal_prompts_and_decoding": {
            "identical_across_arms": True,
            "local_gguf_only": True,
            "adapter_disabled": True,
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
        "third_model_included_only_if_cached_and_budgeted": bool(
            resolution.get("optional_gemma_31b_cached")
        ),
        "never_uses_hf_autotokenizer_for_gguf": True,
    }


def run_live_held_ab(
    *,
    design: Mapping[str, Any],
    model_specs: Mapping[str, Any],
    preconditions_checked: Mapping[str, Any],
) -> dict[str, Any]:  # pragma: no cover - host-dependent live harness
    raise RuntimeError("live Exp5916 GGUF/E3 held runner must be supplied by the conductor host")


def run_confirmatory_controls(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    controls = dict(live_ab.run_confirmatory_controls(rows))
    controls["connected_to_exp5901_causal_mechanism"] = bool(controls.get("controls_passed"))
    return controls


def identical_event_byte_and_budget_parity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    parity = dict(live_ab.identical_event_byte_and_budget_parity(rows))
    parity["principle"] = REQUIRED_FIELD_PROVENANCE[
        "identical_event_byte_and_budget_parity"
    ]["principle"]
    return parity


def live_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    metrics = live_ab.live_metrics(rows)
    for arm in ARM_NAMES:
        arm_rows = [row for row in rows if row.get("arm") == arm]
        correct = sum(
            1
            for row in arm_rows
            if row.get("held_objective_correct", row.get("held_episode_correct"))
        )
        if arm in metrics:
            metrics[arm]["held_objective_correct_count"] = correct
            metrics[arm]["held_objective_accuracy"] = round(
                live_ab._safe_rate(correct, len(arm_rows)), 6
            )
    return metrics


def held_accuracy_progress_efficiency_and_safety_metrics(
    metrics: Mapping[str, Any],
    parity: Mapping[str, Any],
    controls: Mapping[str, Any],
) -> dict[str, Any]:
    safety = live_ab.accuracy_efficiency_and_safety_metrics(metrics, parity, controls)
    structured = metrics.get(STRUCTURED_INDEX_ARM) or {}
    raw = metrics.get(RAW_TAPE_ARM) or {}
    none = metrics.get(NO_MEMORY_ARM) or {}
    safety["structured_over_raw_progress_delta"] = round(
        float(structured.get("mean_progress") or 0.0) - float(raw.get("mean_progress") or 0.0),
        6,
    )
    safety["structured_over_none_progress_delta"] = round(
        float(structured.get("mean_progress") or 0.0) - float(none.get("mean_progress") or 0.0),
        6,
    )
    safety["held_objective_accuracy_by_arm"] = {
        arm: float((metrics.get(arm) or {}).get("held_objective_accuracy") or 0.0)
        for arm in ARM_NAMES
    }
    return safety


def per_model_game_episode_lower_bounds(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    lower = dict(live_ab.per_model_game_episode_lower_bounds(rows))
    lower["interval_method"] = "minimum paired held model/game/episode lower bound"
    return lower


def evidence_utilization_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return live_ab.evidence_utilization_receipts(rows)


def source_access_count(rows: Sequence[Mapping[str, Any]]) -> int:
    return live_ab.source_access_count(rows)


def incidental_completion_receipts(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    completions = [
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
        "incidental_completion_rows": completions,
        "registry_credit_requested": False,
        "registry_updated": False,
        "new_completion_headline_allowed": False,
        "note": "Incidental held completions are telemetry only and receive no registry credit.",
    }


def embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts(
    preconds: Mapping[str, Any],
    run: Mapping[str, Any] | None,
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


def capability_scope_expiry_and_environment_receipts(
    preconds: Mapping[str, Any],
) -> dict[str, Any]:
    gate = preconds.get("upstream_capability_gate") or {}
    return {
        "ok": bool(gate.get("ok")),
        "checked_before_model_load": bool(gate.get("checked_before_model_load")),
        "exp5915_sha256": gate.get("exp5915_sha256"),
        "lease_scope": {
            "runner": capability.RUNNER_ID,
            "environment": capability.ENVIRONMENT_ID,
            "allowed_episode_class": capability.ALLOWED_EPISODE_CLASS,
            "adapter_disabled_required": True,
        },
        "issue_expiry": {
            "issued_at": capability.FIXED_NOW,
            "expires_at": capability.FIXED_EXPIRY,
            "expiry_valid_under_replay_clock": True,
        },
        "teardown_ready": bool((preconds.get("state_isolation_teardown") or {}).get("ok")),
    }


def state_isolation_and_teardown_receipts(preconds: Mapping[str, Any]) -> dict[str, Any]:
    receipt = dict(preconds.get("state_isolation_teardown") or {})
    receipt.setdefault("ok", False)
    receipt.setdefault("persistent_cross_cell_state_detected", False)
    receipt.setdefault("teardown_called_count", 0)
    return receipt


def protected_files_unchanged(root: Path) -> dict[str, Any]:
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
            "principle": (
                f"Exp5916 required artifact field `{field}` is emitted by the held live builder."
            ),
            "satisfied_by": "Exp5916 precondition-gated artifact builder",
        }
        for field in REQUIRED_ARTIFACT_FIELDS
    }
    for field, principle in REQUIRED_FIELD_PROVENANCE.items():
        provenance[field] = {
            **principle,
            "satisfied_by": "REQ-ARC-LRHL-5916 principle-annotated artifact contract",
        }
    return provenance


def _checksum(artifact: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(artifact))
    payload["reproducibility_checksum"] = ""
    return _sha256(payload)


def _empty_metrics() -> dict[str, Any]:
    return live_metrics([])


def _registry_unchanged(root: Path, registry_before: Mapping[str, Any]) -> bool:
    before = registry_before.get("registry_hash_before")
    registry_path = root / "ops" / "arc_solve_registry.yaml"
    if not (root / ".git").exists() and not registry_path.exists():
        return True
    after = _sha256_file(registry_path)
    return bool(before is None or before == after)


def _base_artifact(
    *,
    root: Path,
    preconds: Mapping[str, Any],
    duration_s: float,
    test_exit_codes: Mapping[str, int] | None,
) -> dict[str, Any]:
    registry = dict(preconds.get("registry_precheck") or registry_precheck(root))
    return {
        "experiment": EXPERIMENT_ID,
        "schema": SCHEMA,
        "result_path": RESULT_RELATIVE_PATH,
        "preconditions_checked": dict(preconds),
        "upstream_capability_gate_and_hashes": dict(
            preconds.get("upstream_capability_gate") or upstream_capability_gate_and_hashes(root)
        ),
        "registry_precheck": registry,
        "public_level_solve_claimed": False,
        "preregistered_held_episode_group_and_arm_design": (
            preregistered_held_episode_group_and_arm_design()
        ),
        "model_specs": model_specs_receipt(preconds),
        "model_file_hashes": dict(preconds.get("model_hashes") or {}),
        "capability_scope_expiry_and_environment_receipts": (
            capability_scope_expiry_and_environment_receipts(preconds)
        ),
        "submitted_e3_and_adapter_disabled_receipts": dict(
            preconds.get("submitted_e3_adapter_disabled") or {}
        ),
        "incidental_completion_receipts": incidental_completion_receipts([]),
        "registry_unchanged": _registry_unchanged(root, registry),
        "state_isolation_and_teardown_receipts": state_isolation_and_teardown_receipts(preconds),
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
        "embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts": (
            embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts(preconds, None)
        ),
        "identical_event_byte_and_budget_parity": parity,
        "no_memory_raw_and_structured_live_metrics": metrics,
        "held_accuracy_progress_efficiency_and_safety_metrics": (
            held_accuracy_progress_efficiency_and_safety_metrics(metrics, parity, controls)
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

    design = preregistered_held_episode_group_and_arm_design()
    models = model_specs_receipt(preconds)
    try:
        run = run_live_held_ab(
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
    safety = held_accuracy_progress_efficiency_and_safety_metrics(metrics, parity, controls)
    lower = per_model_game_episode_lower_bounds(rows)
    source_count = source_access_count(rows)
    registry_unchanged = _registry_unchanged(
        root, preconds.get("registry_precheck") or registry_precheck(root)
    )
    capability_clean = bool((preconds.get("upstream_capability_gate") or {}).get("ok"))
    authority_clean = bool((preconds.get("live_runner_execution_binding") or {}).get("ok"))
    ready = bool(
        lower["structured_over_raw_accuracy_lower_bound"] > 0.0
        and lower["structured_over_none_accuracy_lower_bound"] > 0.0
        and safety["safety_regression"] is False
        and safety["budget_regression"] is False
        and controls.get("controls_passed") is True
        and source_count == 0
        and registry_unchanged
        and capability_clean
        and authority_clean
        and parity["all_raw_structured_event_bytes_identical"] is True
    )
    if source_count != 0 or safety["safety_regression"] or safety["budget_regression"]:
        status = "unsafe"
        verdict = "unsafe: held_live_safety_budget_or_authority_regression"
    elif ready:
        status = "complete_positive"
        verdict = "complete_positive: held_live_structured_memory_positive_no_solve_claim"
    else:
        status = "complete_null"
        verdict = "complete_null: held_live_no_preregistered_positive_lower_bound"

    artifact = {
        **_base_artifact(
            root=root,
            preconds=preconds,
            duration_s=float(run.get("duration_s") or (time.monotonic() - started)),
            test_exit_codes=test_exit_codes,
        ),
        "status": status,
        "embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts": (
            embedded_tokenizer_loader_cuda_gpu_utilization_and_vram_receipts(preconds, run)
        ),
        "identical_event_byte_and_budget_parity": parity,
        "no_memory_raw_and_structured_live_metrics": metrics,
        "held_accuracy_progress_efficiency_and_safety_metrics": safety,
        "per_model_game_episode_lower_bounds": lower,
        "shuffled_and_deletion_confirmatory_controls": controls,
        "evidence_utilization_receipts": evidence_utilization_receipts(rows),
        "source_bfs_adapter_prior_game_and_hidden_state_access_count": source_count,
        "incidental_completion_receipts": incidental_completion_receipts(rows),
        "registry_unchanged": registry_unchanged,
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
    if artifact.get("registry_unchanged") is not True:
        raise ValueError("registry_unchanged must be true")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be live_llm_inference")
    if artifact.get("verifier_is_oracle") is not False:
        raise ValueError("verifier_is_oracle must be false")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(
        ("complete_positive:", "complete_null:", "unsafe:", "blocked_precondition:", "blocked:")
    ):
        raise ValueError("honest_verdict has invalid terminal prefix")
    if artifact.get("structured_memory_live_ready_score") == 1.0 and not _ready_score_gates_pass(
        artifact
    ):
        raise ValueError("structured_memory_live_ready_score promotion gates failed")
    if artifact.get("reproducibility_checksum") != _checksum(artifact):
        raise ValueError("reproducibility_checksum mismatch")
    return True


def _ready_score_gates_pass(artifact: Mapping[str, Any]) -> bool:
    lower = artifact.get("per_model_game_episode_lower_bounds") or {}
    safety = artifact.get("held_accuracy_progress_efficiency_and_safety_metrics") or {}
    controls = artifact.get("shuffled_and_deletion_confirmatory_controls") or {}
    parity = artifact.get("identical_event_byte_and_budget_parity") or {}
    capability_receipt = artifact.get("capability_scope_expiry_and_environment_receipts") or {}
    return bool(
        lower.get("structured_over_raw_accuracy_lower_bound", 0.0) > 0.0
        and lower.get("structured_over_none_accuracy_lower_bound", 0.0) > 0.0
        and safety.get("safety_regression") is False
        and safety.get("budget_regression") is False
        and controls.get("controls_passed") is True
        and parity.get("all_raw_structured_event_bytes_identical") is True
        and capability_receipt.get("ok") is True
        and artifact.get("registry_unchanged") is True
        and artifact.get("source_bfs_adapter_prior_game_and_hidden_state_access_count") == 0
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
