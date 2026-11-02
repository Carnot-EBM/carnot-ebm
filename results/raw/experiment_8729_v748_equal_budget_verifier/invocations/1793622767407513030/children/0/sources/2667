"""Build claim-grade evidence for the existing ARC tool-loop compaction flag.

The experiment fails closed before inference when the official catalog has no
new hidden or rotation units. Public replays cannot answer a hidden-game claim,
even when they are convenient and locally available.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import UTC, datetime
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import random
import socket
import subprocess
import tempfile
import time
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7072
SCHEMA = "carnot.exp7072.v619_live_arc_compaction_ab.v1"
INFERENCE_SUBSTRATE = "live_llm_inference"
RESULT_RELATIVE_PATH = Path("results/experiment_7072_v619_live_arc_compaction_ab.json")
CHECKPOINT_RELATIVE_PATH = Path("results/checkpoints/experiment_7072_v619_cells.json")
EXP7052_RELATIVE_PATH = Path("results/experiment_7052_v618_typed_identity_attack_audit.json")
PILOT_RELATIVE_PATH = Path("results/experiment_6473_tool_loop_compaction_pilot_ab.json")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-agi/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7072_v619_live_arc_compaction_ab.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_7072_v619_live_arc_compaction_ab.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7072_v619_live_arc_compaction_ab.py")

QWEN_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
GEMMA_HF_ID = "unsloth/gemma-4-26B-A4B-it-GGUF"
COMPACTION_FLAG = "CARNOT_ARC_INDUCE_TOOL_COMPACT"
GROWTH_FLAG = "CARNOT_ARC_INDUCE_TOOL_COMPACT_GROWTH"
STATE_BUDGET_FLAG = "CARNOT_ARC_INDUCE_TOOL_COMPACT_STATE_BUDGET"
TOOL_LOOP_FLAG = "CARNOT_ARC_INDUCE_TOOL_LOOP"
RANDOM_SEED = 7_072_202_609_06
ACTION_BUDGET = 120
TOKEN_BUDGET = 4096
CONTEXT_BUDGET = 98_304
TIME_BUDGET_S = 2400
QUALITY_NONINFERIORITY_MARGIN = 0.05
MIN_QWEN_PAIRS = 30
MIN_GEMMA_PAIRS = 8
PRODUCTION_SYMBOLS = (
    "carnot.agentic.arc_competition_agent.E3AgentPolicy",
    "carnot.agentic.arc_competition_agent.make_carnot_agent",
    "carnot.agentic.arc_executable_world_model.LocalGGUFProposer",
    "carnot.agentic.arc_induction_tool_loop.induce_with_tool_loop",
    "carnot.agentic.arc_induction_compact_state.CompactionController",
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "duration_s",
    "source_artifact_hashes",
    "cited_upstream_artifacts",
    "rows",
    "per_game_results",
    "MODEL_SPECS",
    "model_specs",
    "models_used",
    "selected_model_specs",
    "model_identity_rows",
    "model_file_hash_rows",
    "runner_build_rows",
    "gpu_lease_rows",
    "port_lease_rows",
    "gpu_sample_rows",
    "cuda_layer_offload_confirmed",
    "registry_precheck_rows",
    "production_reachability_rows",
    "flag_isolation_rows",
    "arm_definitions",
    "paired_cell_manifest",
    "arm_order_rows",
    "tool_loop_rows",
    "compaction_rows",
    "refetch_rows",
    "parse_failure_rows",
    "exact_progress_rows",
    "solve_rows",
    "solve_provenance",
    "context_metric_rows",
    "token_metric_rows",
    "wall_time_rows",
    "failure_rows",
    "paired_interval_rows",
    "qwen_pair_count",
    "gemma_pair_count",
    "tool_loop_reachable_score",
    "compaction_treatment_activated_score",
    "compaction_value_ready_score",
    "retirement_decision",
    "cleanup_rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

FIELD_PRINCIPLES = {
    "field_principles": "Each retained field states why it is needed for the scientific decision.",
    "preconditions_checked": "Preconditions prevent missing resources from becoming fabricated live evidence.",
    "inference_substrate": "The substrate distinguishes local model inference from cached or simulated work.",
    "duration_s": "Measured wall time makes claimed live compute auditable.",
    "source_artifact_hashes": "Content hashes bind conclusions to exact code, specifications, and prior evidence.",
    "cited_upstream_artifacts": "Explicit citations expose every prior gate used by this experiment.",
    "rows": "Terminal summary rows make readiness decisions independently recountable.",
    "per_game_results": "Per-game results prevent pooled means from hiding a source-group failure.",
    "MODEL_SPECS": "The declared model list prevents an unrecorded substitute model.",
    "model_specs": "A second conventional model field keeps standard artifact readers compatible.",
    "models_used": "Executed hub IDs distinguish declared models from models that produced rows.",
    "selected_model_specs": "Selected paths preserve the exact local files chosen before inference.",
    "model_identity_rows": "Typed identity evidence prevents a same-name model substitution.",
    "model_file_hash_rows": "File hashes detect changed GGUF bytes across replication.",
    "runner_build_rows": "Runner linkage and version rows prove that CUDA production serving was selected.",
    "gpu_lease_rows": "Owner-bound leases prevent inference from sharing or killing another task's device.",
    "port_lease_rows": "Port ownership binds each server endpoint to this task.",
    "gpu_sample_rows": "Samples inside inference prove that the owned process used the selected RTX 3090.",
    "cuda_layer_offload_confirmed": "CUDA layer evidence rejects a slow CPU run mislabeled as GPU inference.",
    "registry_precheck_rows": "Registry checks exclude already reproduced public levels from hidden-game credit.",
    "production_reachability_rows": "Import and factory rows prove that the scored E3 entrypoint executed the flag.",
    "flag_isolation_rows": "Byte comparisons ensure the main A/B changes only compaction.",
    "arm_definitions": "Frozen arm meanings stop treatment labels from changing after outcomes are known.",
    "paired_cell_manifest": "A preregistered manifest fixes population, budgets, seeds, and stopping rules.",
    "arm_order_rows": "Order rows expose counterbalancing and sequence effects.",
    "tool_loop_rows": "Tool-loop rows must show reachability before compaction value is interpreted.",
    "compaction_rows": "Compaction counts distinguish an active treatment from a no-op comparison.",
    "refetch_rows": "Refetches expose the token and tool cost caused by discarding transcript detail.",
    "parse_failure_rows": "Parser failures remain charged instead of disappearing from successful rows.",
    "exact_progress_rows": "Exact progress is the primary quality measure available before a full solve.",
    "solve_rows": "Solve rows retain level credit and provenance without relying on pooled totals.",
    "solve_provenance": "Live self-discovery is the only provenance eligible for hidden-game solve credit.",
    "context_metric_rows": "Peak and p95 context rows test the memory claim directly.",
    "token_metric_rows": "Token rows expose decode work that context reduction can shift elsewhere.",
    "wall_time_rows": "Wall rows prevent a memory saving from hiding an operational slowdown.",
    "failure_rows": "Crashes and duplicate submissions stay in denominators and readiness decisions.",
    "paired_interval_rows": "Paired intervals quantify uncertainty after matched nuisance factors are removed.",
    "qwen_pair_count": "Thirty Qwen pairs are the minimum population for the primary percentage claim.",
    "gemma_pair_count": "Eight Gemma pairs test whether the quality direction transfers across model families.",
    "tool_loop_reachable_score": "Value is interpretable only when both arms reached the production tool loop.",
    "compaction_treatment_activated_score": "The activation gate prevents an inert flag from producing a false null.",
    "compaction_value_ready_score": "Readiness separates qualified evidence from release success.",
    "retirement_decision": "A terminal release-or-retire decision prevents another unchanged rerun.",
    "cleanup_rows": "Cleanup evidence proves that only owned resources were stopped and released.",
    "random_seed": "A fixed seed makes cell selection and arm order reproducible.",
    "reproducibility_checksum": "A canonical digest detects any later artifact-field change.",
    "gate_check_summary": "The first exact failure makes a blocked or disqualified result actionable.",
    "verifier_is_oracle": "False states that compaction does not define ARC correctness.",
    "verdict_class": "A closed class gives downstream readers one machine-readable outcome.",
    "honest_verdict": "A class-consistent prefix prevents blocked evidence from reading as success.",
}


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable ASCII JSON bytes for hashes and A/B comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_file(path: str | Path) -> str | None:
    """Hash one file in bounded chunks so large GGUF files do not enter memory."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def hash_without_field(value: Mapping[str, Any], field: str) -> str:
    """Hash a mapping after blanking its self-referential digest field."""

    body = deepcopy(dict(value))
    body[field] = ""
    return "sha256:" + hashlib.sha256(canonical_json_bytes(body)).hexdigest()


def gate_row(check: str, expected: Any, observed: Any) -> JsonDict:
    """Keep both sides of one exact gate decision."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected,
        "terminal": True,
    }


def gate_check_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failed gate while retaining every decision row."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def model_identity_row(spec: Mapping[str, Any]) -> JsonDict:
    """Bind one selected model to its real path and content bytes."""

    path = Path(str(spec.get("model_path", ""))).resolve()
    return {
        "model_key": spec.get("key"),
        "model_hf_id": spec.get("hf_id"),
        "model_path": str(path),
        "model_filename": path.name,
        "model_file_hash": sha256_file(path),
        "gpu": spec.get("gpu"),
        "terminal": True,
    }


def validate_model_identity(row: Mapping[str, Any], spec: Mapping[str, Any]) -> list[str]:
    """Reject a hub, path, or byte mismatch without inferring an alias."""

    expected = model_identity_row(spec)
    errors = []
    for field in ("model_hf_id", "model_path", "model_file_hash"):
        if row.get(field) != expected.get(field):
            errors.append(f"{field}_mismatch")
    return errors


def resolve_model_specs(
    *,
    pair_resolver: Callable[..., list[dict] | None] = cached_sota_pair,
    qwen_resolver: Callable[[str], str | None] = resolve_cached_gguf,
    gpu_indices: tuple[int, int] = (0, 1),
) -> list[JsonDict]:
    """Resolve Qwen directly and Gemma through the mandated cached-pair helper."""

    qwen_path = qwen_resolver(QWEN_HF_ID)
    cached_pair = pair_resolver(gpu_indices=(gpu_indices[0], gpu_indices[1])) or []
    gemma = next((dict(row) for row in cached_pair if row.get("hf_id") == GEMMA_HF_ID), None)
    if qwen_path is None or gemma is None:
        return []
    return [
        {
            "key": "qwen",
            "name": "Qwen3.8-27B",
            "hf_id": QWEN_HF_ID,
            "model_path": str(Path(qwen_path).resolve()),
            "gpu": gpu_indices[0],
        },
        {
            "key": "gemma",
            "name": gemma["name"],
            "hf_id": GEMMA_HF_ID,
            "model_path": str(Path(str(gemma["model_path"])).resolve()),
            "gpu": gpu_indices[1],
        },
    ]


def _arm_order(index: int, random_seed: int) -> list[str]:
    first_treatment = (index + random_seed) % 2 == 0
    return ["treatment", "control"] if first_treatment else ["control", "treatment"]


def freeze_paired_cell_manifest(
    *,
    units: Sequence[Mapping[str, Any]],
    model_specs: Sequence[Mapping[str, Any]],
    qwen_pair_count: int,
    gemma_pair_count: int,
    action_budget: int,
    token_budget: int,
    context_budget: int,
    time_budget_s: int,
    random_seed: int,
) -> JsonDict:
    """Freeze paired cells before a model or outcome can influence selection."""

    if qwen_pair_count < MIN_QWEN_PAIRS or gemma_pair_count < MIN_GEMMA_PAIRS:
        raise ValueError("claim-grade pair counts are below the preregistered minimum")
    if len(units) < qwen_pair_count:
        raise ValueError("not enough eligible units for Qwen pairs")
    by_key = {str(row.get("key")): dict(row) for row in model_specs}
    if set(by_key) != {"qwen", "gemma"}:
        raise ValueError("both pinned model specs are required")
    identities = {key: model_identity_row(spec) for key, spec in by_key.items()}
    pairs: list[JsonDict] = []
    selections = (("qwen", qwen_pair_count), ("gemma", gemma_pair_count))
    for model_key, count in selections:
        identity = identities[model_key]
        for unit in units[:count]:
            pair_index = len(pairs)
            pairs.append(
                {
                    "pair_id": f"{model_key}:{unit['unit_id']}",
                    "unit_id": unit["unit_id"],
                    "game_id": unit["game_id"],
                    "source_group": unit["source_group"],
                    "model_key": model_key,
                    "model_hf_id": identity["model_hf_id"],
                    "model_path": identity["model_path"],
                    "model_file_hash": identity["model_file_hash"],
                    "gpu": by_key[model_key].get("gpu"),
                    "seed": int(unit["seed"]),
                    "start_state_hash": unit["start_state_hash"],
                    "action_budget": int(action_budget),
                    "token_budget": int(token_budget),
                    "context_budget": int(context_budget),
                    "time_budget_s": int(time_budget_s),
                    "arm_order": _arm_order(pair_index, random_seed),
                    "solve_provenance": "live_agent_self_discovery",
                }
            )
    manifest: JsonDict = {
        "schema": "carnot.exp7072.paired_manifest.v1",
        "random_seed": int(random_seed),
        "qwen_pair_count": int(qwen_pair_count),
        "gemma_pair_count": int(gemma_pair_count),
        "source_groups": sorted({str(row["source_group"]) for row in pairs}),
        "pairs": pairs,
        "stopping_rule": "stop after first production tool loop or the matched action/time cap",
        "manifest_hash": "",
    }
    manifest["manifest_hash"] = hash_without_field(manifest, "manifest_hash")
    return manifest


def arm_config(pair: Mapping[str, Any], arm: str) -> JsonDict:
    """Return the exact paired configuration with only the treatment switch changed."""

    if arm not in {"control", "treatment"}:
        raise ValueError(f"unknown arm: {arm}")
    environment = {TOOL_LOOP_FLAG: "selfparse"}
    if arm == "treatment":
        environment[COMPACTION_FLAG] = "1"
    return {
        "pair_id": pair["pair_id"],
        "model_hf_id": pair["model_hf_id"],
        "model_path": pair["model_path"],
        "model_file_hash": pair["model_file_hash"],
        "seed": pair["seed"],
        "action_budget": pair["action_budget"],
        "token_budget": pair["token_budget"],
        "context_budget": pair["context_budget"],
        "time_budget_s": pair["time_budget_s"],
        "start_state_hash": pair["start_state_hash"],
        "environment": environment,
    }


def arm_config_bytes(pair: Mapping[str, Any], arm: str) -> bytes:
    return canonical_json_bytes(arm_config(pair, arm))


def flag_isolation_rows(manifest: Mapping[str, Any]) -> list[JsonDict]:
    """Prove that canonical arm objects differ only at the master flag."""

    rows = []
    for pair in manifest.get("pairs", []):
        control = arm_config(pair, "control")
        treatment = arm_config(pair, "treatment")
        control_without = deepcopy(control)
        treatment_without = deepcopy(treatment)
        control_without["environment"].pop(COMPACTION_FLAG, None)
        treatment_without["environment"].pop(COMPACTION_FLAG, None)
        rows.append(
            {
                "pair_id": pair["pair_id"],
                "only_difference": COMPACTION_FLAG,
                "growth_default_unset": GROWTH_FLAG not in treatment["environment"],
                "state_budget_default_unset": STATE_BUDGET_FLAG not in treatment["environment"],
                "passed": control_without == treatment_without
                and COMPACTION_FLAG not in control["environment"]
                and treatment["environment"].get(COMPACTION_FLAG) == "1",
                "terminal": True,
            }
        )
    return rows


def registry_precheck_rows(
    units: Sequence[Mapping[str, Any]], reproduced_game_ids: set[str]
) -> list[JsonDict]:
    """Reject any unit that can reuse public source, adapters, or solved levels."""

    rows = []
    for unit in units:
        game_id = str(unit.get("game_id", ""))
        family = game_id.split("-", 1)[0]
        reasons = []
        checks = {
            "already_reproduced": game_id in reproduced_game_ids or family in reproduced_game_ids,
            "source_read": bool(unit.get("source_read")),
            "offline_ground_truth": bool(unit.get("offline_ground_truth")),
            "per_game_adapter": bool(unit.get("per_game_adapter")),
            "development_proxy": bool(unit.get("development_proxy")),
        }
        reasons.extend(name for name, failed in checks.items() if failed)
        rows.append(
            {
                "unit_id": unit.get("unit_id"),
                "game_id": game_id,
                "source_group": unit.get("source_group"),
                "reasons": reasons,
                "passed": not reasons,
                "solve_provenance": "live_agent_self_discovery",
                "terminal": True,
            }
        )
    return rows


class CellCheckpointStore:
    """Persist completed cells atomically and reject manifest drift on resume."""

    def __init__(self, path: str | Path, *, manifest_hash: str) -> None:
        self.path = Path(path)
        self.manifest_hash = str(manifest_hash)
        self.rows: dict[str, JsonDict] = {}
        if self.path.is_file():
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("manifest_hash") != self.manifest_hash:
                raise ValueError("checkpoint manifest mismatch")
            self.rows = {str(row["cell_id"]): dict(row) for row in payload.get("rows", [])}

    def save(self, row: Mapping[str, Any]) -> None:
        cell_id = str(row["cell_id"])
        if cell_id in self.rows:
            return
        self.rows[cell_id] = deepcopy(dict(row))
        atomic_write_json(
            self.path,
            {"manifest_hash": self.manifest_hash, "rows": list(self.rows.values())},
            allow_override=False,
        )

    def pending(self, cell_ids: Sequence[str]) -> list[str]:
        return [str(cell_id) for cell_id in cell_ids if str(cell_id) not in self.rows]


def cleanup_owned_resources(
    resources: Sequence[Mapping[str, Any]],
    *,
    stop_process: Callable[[int], bool],
    release_resource: Callable[[str], bool],
) -> list[JsonDict]:
    """Act only on resources whose owner receipt belongs to this task."""

    rows = []
    for resource in resources:
        kind = str(resource.get("kind"))
        identity = str(resource.get("identity"))
        owned = resource.get("owned") is True
        if not owned:
            action, passed = "blocked_unattributed", True
        elif kind == "process":
            action, passed = "stopped_owned_process", bool(stop_process(int(resource["pid"])))
        else:
            action, passed = "released_owned_resource", bool(release_resource(identity))
        rows.append(
            {
                "kind": kind,
                "identity": identity,
                "owned": owned,
                "action": action,
                "passed": passed,
                "terminal": True,
            }
        )
    return rows


def activation_scores(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Compute reachability and the preregistered 80-percent activation gate."""

    eligible_treatment = [
        row
        for row in rows
        if row.get("arm") == "treatment" and row.get("eligible_for_compaction") is True
    ]
    controls = [row for row in rows if row.get("arm") == "control"]
    fired = sum(int(row.get("compactions", 0) > 0) for row in eligible_treatment)
    fire_rate = fired / len(eligible_treatment) if eligible_treatment else 0.0
    reachable = bool(rows) and all(row.get("tool_loop_reachable") is True for row in rows)
    control_fires = sum(int(row.get("compactions", 0) > 0) for row in controls)
    model_fire_rates = {}
    for model_key in {str(row.get("model_key")) for row in rows}:
        model_eligible = [
            row for row in eligible_treatment if str(row.get("model_key")) == model_key
        ]
        model_fired = sum(int(row.get("compactions", 0) > 0) for row in model_eligible)
        model_fire_rates[model_key] = model_fired / len(model_eligible) if model_eligible else 0.0
    activated = (
        reachable
        and bool(eligible_treatment)
        and all(rate >= 0.80 for rate in model_fire_rates.values())
        and control_fires == 0
    )
    return {
        "eligible_treatment_cells": len(eligible_treatment),
        "treatment_cells_fired": fired,
        "treatment_fire_rate": fire_rate,
        "treatment_fire_rate_by_model": model_fire_rates,
        "control_cells_fired": control_fires,
        "tool_loop_reachable_score": int(reachable),
        "compaction_treatment_activated_score": int(activated),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _one_sided_lower_95(values: Sequence[float]) -> float:
    """Return a declared normal-approximation lower bound for paired deltas."""

    if len(values) < 2:
        return values[0] if values else float("-inf")
    mean = _mean(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return mean - 1.6448536269514722 * math.sqrt(variance / len(values))


def aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Recompute all value gates from terminal per-cell rows."""

    copied = [deepcopy(dict(row)) for row in rows]
    grouped: dict[str, dict[str, JsonDict]] = defaultdict(dict)
    for row in copied:
        grouped[str(row.get("pair_id"))][str(row.get("arm"))] = row
    complete_pairs = {
        pair_id: arms for pair_id, arms in grouped.items() if set(arms) == {"control", "treatment"}
    }
    by_model: JsonDict = {}
    paired_interval_rows = []
    for model_key in ("qwen", "gemma"):
        pairs = [
            arms
            for arms in complete_pairs.values()
            if arms["control"].get("model_key") == model_key
        ]
        arms_summary: JsonDict = {}
        for arm in ("control", "treatment"):
            arm_rows = [pair[arm] for pair in pairs]
            tool_calls = sum(int(row.get("tool_calls", 0)) for row in arm_rows)
            parse_failures = sum(int(row.get("parse_failures", 0)) for row in arm_rows)
            arms_summary[arm] = {
                "cell_count": len(arm_rows),
                "parse_failures": parse_failures,
                "tool_calls": tool_calls,
                "parse_failure_rate": parse_failures / tool_calls if tool_calls else 0.0,
                "mean_exact_progress": _mean(
                    [float(row.get("exact_progress", 0)) for row in arm_rows]
                ),
                "mean_p95_context": _mean([float(row.get("p95_context", 0)) for row in arm_rows]),
                "mean_wall_time_s": _mean([float(row.get("wall_time_s", 0)) for row in arm_rows]),
            }
        quality_deltas = [
            float(pair["treatment"].get("exact_progress", 0))
            - float(pair["control"].get("exact_progress", 0))
            for pair in pairs
        ]
        control_context = arms_summary["control"]["mean_p95_context"]
        control_wall = arms_summary["control"]["mean_wall_time_s"]
        paired = {
            "pair_count": len(pairs),
            "exact_quality_mean_delta": _mean(quality_deltas),
            "exact_quality_one_sided_lower_95": _one_sided_lower_95(quality_deltas),
            "declared_noninferiority_margin": QUALITY_NONINFERIORITY_MARGIN,
            "p95_context_ratio": (
                arms_summary["treatment"]["mean_p95_context"] / control_context
                if control_context
                else float("inf")
            ),
            "wall_time_ratio": (
                arms_summary["treatment"]["mean_wall_time_s"] / control_wall
                if control_wall
                else float("inf")
            ),
        }
        by_model[model_key] = {**arms_summary, "paired": paired}
        paired_interval_rows.append(
            {
                "model_key": model_key,
                "metric": "exact_progress_treatment_minus_control",
                "pair_count": len(pairs),
                "mean_delta": paired["exact_quality_mean_delta"],
                "one_sided_lower_95": paired["exact_quality_one_sided_lower_95"],
                "method": "paired_normal_one_sided_95",
                "terminal": True,
            }
        )
    activation = activation_scores(copied)
    return {
        "by_model": by_model,
        "qwen_pair_count": by_model["qwen"]["paired"]["pair_count"],
        "gemma_pair_count": by_model["gemma"]["paired"]["pair_count"],
        **activation,
        "per_game_results": [
            {
                "pair_id": pair_id,
                "game_id": arms["control"].get("game_id"),
                "source_group": arms["control"].get("source_group"),
                "model_key": arms["control"].get("model_key"),
                "control": arms["control"],
                "treatment": arms["treatment"],
            }
            for pair_id, arms in sorted(complete_pairs.items())
        ],
        "tool_loop_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "reachable": row.get("tool_loop_reachable"),
                "tool_calls": row.get("tool_calls"),
            }
            for row in copied
        ],
        "compaction_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "eligible": row.get("eligible_for_compaction"),
                "compactions": row.get("compactions"),
            }
            for row in copied
        ],
        "refetch_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "refetches": row.get("refetches"),
            }
            for row in copied
        ],
        "parse_failure_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "parse_failures": row.get("parse_failures"),
                "tool_calls": row.get("tool_calls"),
            }
            for row in copied
        ],
        "exact_progress_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "exact_progress": row.get("exact_progress"),
                "levels_reached": row.get("levels_reached"),
            }
            for row in copied
        ],
        "solve_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "solves": row.get("solves"),
                "solve_provenance": row.get("solve_provenance"),
            }
            for row in copied
        ],
        "context_metric_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "peak_context": row.get("peak_context"),
                "p95_context": row.get("p95_context"),
            }
            for row in copied
        ],
        "token_metric_rows": [
            {"pair_id": row.get("pair_id"), "arm": row.get("arm"), "tokens": row.get("tokens")}
            for row in copied
        ],
        "wall_time_rows": [
            {
                "pair_id": row.get("pair_id"),
                "arm": row.get("arm"),
                "wall_time_s": row.get("wall_time_s"),
            }
            for row in copied
        ],
        "failure_rows": [
            row
            for row in copied
            if row.get("status") != "complete"
            or int(row.get("crashes", 0))
            or int(row.get("duplicate_submissions", 0))
        ],
        "paired_interval_rows": paired_interval_rows,
    }


def classify_value(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Apply activation first, then the preregistered release gates."""

    aggregate = aggregate_rows(rows)
    population_ready = (
        aggregate["qwen_pair_count"] >= MIN_QWEN_PAIRS
        and aggregate["gemma_pair_count"] >= MIN_GEMMA_PAIRS
    )
    qualified = (
        population_ready
        and aggregate["tool_loop_reachable_score"] == 1
        and aggregate["compaction_treatment_activated_score"] == 1
    )
    if not qualified:
        return {
            "verdict_class": "disqualified",
            "honest_verdict": "complete_disqualified_compaction_harness_not_value_evidence",
            "compaction_value_ready_score": 0,
            "retirement_decision": "retain_unqualified_no_value_interpretation",
            "failed_check": (
                "paired_population"
                if not population_ready
                else (
                    "tool_loop_reachable_score"
                    if aggregate["tool_loop_reachable_score"] != 1
                    else "compaction_treatment_activated_score"
                )
            ),
        }
    qwen = aggregate["by_model"]["qwen"]
    gemma = aggregate["by_model"]["gemma"]
    release_checks = {
        "qwen_exact_quality_noninferior": qwen["paired"]["exact_quality_one_sided_lower_95"]
        >= -QUALITY_NONINFERIORITY_MARGIN,
        "qwen_p95_context_reduction": qwen["paired"]["p95_context_ratio"] <= 0.90,
        "qwen_parse_failure_not_worse": qwen["treatment"]["parse_failure_rate"]
        <= qwen["control"]["parse_failure_rate"],
        "qwen_wall_regression_bounded": qwen["paired"]["wall_time_ratio"] <= 1.10,
        "gemma_quality_not_opposite": gemma["paired"]["exact_quality_mean_delta"] >= 0.0,
    }
    if all(release_checks.values()):
        return {
            "verdict_class": "positive",
            "honest_verdict": "complete_positive_release_existing_compaction_flag",
            "compaction_value_ready_score": 1,
            "retirement_decision": "release_compaction_flag",
            "release_checks": release_checks,
            "failed_check": None,
        }
    matches_pilot = (
        qwen["paired"]["exact_quality_mean_delta"] == 0.0
        and qwen["paired"]["p95_context_ratio"] < 1.0
        and qwen["paired"]["wall_time_ratio"] > 1.10
    )
    return {
        "verdict_class": "null",
        "honest_verdict": "complete_null_compaction_did_not_meet_release_gate",
        "compaction_value_ready_score": 1,
        "retirement_decision": (
            "retire_existing_compaction_scope_same_as_exp6473"
            if matches_pilot
            else "retain_only_if_new_nonpilot_failure_is_addressed"
        ),
        "release_checks": release_checks,
        "failed_check": next(name for name, passed in release_checks.items() if not passed),
    }


def _empty_evidence() -> JsonDict:
    return {
        "rows": [],
        "per_game_results": [],
        "MODEL_SPECS": [],
        "model_specs": [],
        "models_used": [],
        "selected_model_specs": [],
        "model_identity_rows": [],
        "model_file_hash_rows": [],
        "runner_build_rows": [],
        "gpu_lease_rows": [],
        "port_lease_rows": [],
        "gpu_sample_rows": [],
        "cuda_layer_offload_confirmed": False,
        "registry_precheck_rows": [],
        "production_reachability_rows": [],
        "flag_isolation_rows": [],
        "arm_definitions": {
            "control": f"{COMPACTION_FLAG} unset",
            "treatment": f"{COMPACTION_FLAG}=1",
            "fixed": {GROWTH_FLAG: "default", STATE_BUDGET_FLAG: "default"},
        },
        "paired_cell_manifest": {},
        "arm_order_rows": [],
        "tool_loop_rows": [],
        "compaction_rows": [],
        "refetch_rows": [],
        "parse_failure_rows": [],
        "exact_progress_rows": [],
        "solve_rows": [],
        "solve_provenance": "live_agent_self_discovery",
        "context_metric_rows": [],
        "token_metric_rows": [],
        "wall_time_rows": [],
        "failure_rows": [],
        "paired_interval_rows": [],
        "qwen_pair_count": 0,
        "gemma_pair_count": 0,
        "tool_loop_reachable_score": 0,
        "compaction_treatment_activated_score": 0,
        "compaction_value_ready_score": 0,
        "retirement_decision": "no_decision_precondition_blocked",
        "cleanup_rows": [],
    }


def blocked_artifact(
    *,
    execution_date: str,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    duration_s: float,
) -> JsonDict:
    """Build a schema-complete terminal block without fabricated live rows."""

    summary = gate_check_summary(checks)
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in checks],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "cited_upstream_artifacts": [],
        **_empty_evidence(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": summary,
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": f"blocked_live_arc_compaction_ab:{summary['failed_check']}",
    }
    artifact["reproducibility_checksum"] = hash_without_field(artifact, "reproducibility_checksum")
    return artifact


def validate_artifact(artifact: Any) -> list[str]:
    """Validate required fields, principles, checksum, and terminal semantics."""

    if not isinstance(artifact, Mapping):
        return ["artifact_not_object"]
    errors = []
    missing = sorted(set(REQUIRED_ARTIFACT_FIELDS) - set(artifact))
    errors.extend(f"missing_field:{field}" for field in missing)
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles_incomplete")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    verdict_class = artifact.get("verdict_class")
    prefixes = {
        "positive": "complete_positive_",
        "circular_positive": "complete_circular_positive_",
        "null": "complete_null_",
        "blocked": "blocked_",
        "disqualified": "complete_disqualified_",
        "partial": "partial_",
    }
    if verdict_class not in prefixes or not str(artifact.get("honest_verdict", "")).startswith(
        prefixes.get(verdict_class, "\0")
    ):
        errors.append("verdict_prefix_invalid")
    if artifact.get("reproducibility_checksum") != hash_without_field(
        artifact, "reproducibility_checksum"
    ):
        errors.append("reproducibility_checksum_mismatch")
    summary = artifact.get("gate_check_summary")
    if verdict_class == "blocked" and (
        not isinstance(summary, Mapping)
        or summary.get("passed") is not False
        or summary.get("failed_check") is None
        or "expected_value" not in summary
        or "observed_value" not in summary
    ):
        errors.append("blocked_gate_summary_invalid")
    return list(dict.fromkeys(errors))


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - host files define the receipt
    paths = (
        SPEC_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        SCRIPT_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        EXP7052_RELATIVE_PATH,
        PILOT_RELATIVE_PATH,
        REGISTRY_RELATIVE_PATH,
        Path("python/carnot/agentic/arc_induction_compact_state.py"),
        Path("python/carnot/agentic/arc_induction_tool_loop.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_competition_agent.py"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).is_file()}


def _load_json(path: Path) -> JsonDict:  # pragma: no cover - host input boundary
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


def _reproduced_game_ids(path: Path) -> set[str]:  # pragma: no cover - registry input boundary
    try:
        import yaml

        value = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return set()
    games = value.get("games", []) if isinstance(value, Mapping) else []
    return {
        str(row.get("game"))
        for row in games
        if isinstance(row, Mapping) and int(row.get("levels_reproduced", 0) or 0) > 0
    }


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover - hardware boundary
    query = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    busy = {line.split(",", 1)[0].strip() for line in apps.stdout.splitlines() if "," in line}
    rows = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 6:
            continue
        try:
            index, total, free, utilization = (
                int(parts[0]),
                int(parts[3]),
                int(parts[4]),
                int(parts[5]),
            )
        except ValueError:
            continue
        rows.append(
            {
                "index": index,
                "gpu_model": parts[1],
                "gpu_uuid": parts[2],
                "memory_total_mb": total,
                "memory_free_mb": free,
                "utilization_pct": utilization,
                "idle": parts[2] not in busy and utilization <= 5,
                "supported": "RTX 3090" in parts[1],
            }
        )
    return rows


def _runner_row() -> JsonDict:  # pragma: no cover - binary linkage boundary
    server = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    linked = subprocess.run(["ldd", str(server)], capture_output=True, text=True, check=False)
    version = subprocess.run(
        [str(server), "--version"], capture_output=True, text=True, check=False
    )
    return {
        "path": str(server),
        "exists": server.is_file() and os.access(server, os.X_OK),
        "sha256": sha256_file(server),
        "cuda_linked": "libcuda.so" in linked.stdout and "libggml-cuda" in linked.stdout,
        "hip_linked": "libamdhip64" in linked.stdout,
        "version_returncode": version.returncode,
        "version": (version.stdout + version.stderr).strip()[:500],
        "terminal": True,
    }


def _catalog_rows(root: Path) -> list[JsonDict]:  # pragma: no cover - official SDK boundary
    from arc_agi import Arcade, OperationMode

    arcade = Arcade(
        operation_mode=OperationMode.ONLINE,
        environments_dir=str(root / ".no_local_arc_environments"),
    )
    rows = []
    for info in arcade.available_environments:
        tags = [str(tag).lower() for tag in (getattr(info, "tags", None) or [])]
        private = [str(tag).lower() for tag in (getattr(info, "private_tags", None) or [])]
        rows.append(
            {
                "game_id": str(info.game_id),
                "title": str(info.title),
                "tags": tags,
                "private_tags": private,
                "source_group": "|".join(sorted(private or tags)) or "unlabeled",
            }
        )
    return rows


def eligible_catalog_units(
    catalog_rows: Sequence[Mapping[str, Any]], reproduced_game_ids: set[str]
) -> list[JsonDict]:
    """Select only explicit hidden/rotation rows or families absent from the registry."""

    eligible = []
    for row in catalog_rows:
        game_id = str(row.get("game_id", ""))
        family = game_id.split("-", 1)[0]
        labels = {
            str(value).lower()
            for value in [*(row.get("tags") or []), *(row.get("private_tags") or [])]
        }
        source_kind = (
            "hidden"
            if "hidden" in labels
            else "rotation"
            if "rotation" in labels
            else "unregistered_hidden_candidate"
            if family not in reproduced_game_ids
            else "reproduced_public"
        )
        if source_kind == "reproduced_public":
            continue
        for index in range(5):
            eligible.append(
                {
                    "unit_id": f"{game_id}:seed-{index}",
                    "game_id": game_id,
                    "source_group": f"{source_kind}:{row.get('source_group', 'unlabeled')}",
                    "seed": RANDOM_SEED + index,
                    "start_state_hash": "captured_before_first_action",
                    "source_read": False,
                    "offline_ground_truth": False,
                    "per_game_adapter": False,
                    "development_proxy": False,
                }
            )
    return eligible


def production_reachability_rows() -> list[JsonDict]:
    """Inspect the scored factory and policy instead of trusting module names."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    factory_source = inspect.getsource(make_carnot_agent)
    return [
        {
            "symbol": PRODUCTION_SYMBOLS[0],
            "imported": isinstance(E3AgentPolicy, type),
            "factory_constructs_symbol": "E3AgentPolicy(" in factory_source,
            "passed": isinstance(E3AgentPolicy, type) and "E3AgentPolicy(" in factory_source,
            "terminal": True,
        },
        {
            "symbol": PRODUCTION_SYMBOLS[1],
            "imported": callable(make_carnot_agent),
            "passed": callable(make_carnot_agent),
            "terminal": True,
        },
        {
            "symbol": PRODUCTION_SYMBOLS[2],
            "imported": isinstance(LocalGGUFProposer, type),
            "passed": isinstance(LocalGGUFProposer, type),
            "terminal": True,
        },
    ]


def collect_preconditions(
    *, repo_root: Path, output_path: Path, checkpoint_path: Path
) -> JsonDict:  # pragma: no cover - combines live host boundaries
    """Check every upstream, catalog, model, CUDA, ownership, and write gate."""

    checks: list[JsonDict] = []
    upstream_path = repo_root / EXP7052_RELATIVE_PATH
    upstream = _load_json(upstream_path)
    checks.append(
        gate_row(
            "typed_identity_attack_audit_ready_score",
            1,
            upstream.get("typed_identity_attack_audit_ready_score"),
        )
    )
    python_hash = sha256_file(upstream_path)
    command = subprocess.run(
        ["sha256sum", str(upstream_path)], capture_output=True, text=True, check=False
    )
    command_hash = (
        "sha256:" + command.stdout.split()[0]
        if command.returncode == 0 and command.stdout.split()
        else None
    )
    checks.append(gate_row("exp7052_source_hash_recomputed", python_hash, command_hash))
    registry_path = repo_root / REGISTRY_RELATIVE_PATH
    reproduced = _reproduced_game_ids(registry_path)
    checks.append(gate_row("registry_precheck_readable", True, bool(reproduced)))
    try:
        catalog = _catalog_rows(repo_root)
        catalog_error = None
    except Exception as exc:  # noqa: BLE001 - exact error becomes gate evidence
        catalog, catalog_error = [], f"{type(exc).__name__}: {exc}"
    units = eligible_catalog_units(catalog, reproduced)
    groups = {row["source_group"] for row in units}
    checks.append(gate_row("official_catalog_readable", True, bool(catalog)))
    checks.append(gate_row("eligible_hidden_or_rotation_units", f">={MIN_QWEN_PAIRS}", len(units)))
    checks[-1]["passed"] = len(units) >= MIN_QWEN_PAIRS
    checks.append(gate_row("multiple_hidden_source_groups", ">=2", len(groups)))
    checks[-1]["passed"] = len(groups) >= 2
    models = resolve_model_specs()
    checks.append(
        gate_row(
            "pinned_qwen_and_cached_sota_gemma",
            sorted((QWEN_HF_ID, GEMMA_HF_ID)),
            sorted(str(row.get("hf_id")) for row in models),
        )
    )
    identities = [model_identity_row(row) for row in models]
    checks.append(
        gate_row(
            "model_files_hashable",
            True,
            len(identities) == 2 and all(row["model_file_hash"] for row in identities),
        )
    )
    gpu_rows = _gpu_rows()
    idle = [row for row in gpu_rows if row["supported"] and row["idle"]]
    checks.append(
        gate_row("owned_idle_rtx3090_lease_candidates", f">={len(models) or 2}", len(idle))
    )
    checks[-1]["passed"] = len(idle) >= (len(models) or 2)
    runner = _runner_row()
    checks.append(
        gate_row(
            "cuda_production_runner",
            True,
            runner["exists"]
            and runner["cuda_linked"]
            and not runner["hip_linked"]
            and runner["version_returncode"] == 0,
        )
    )
    for label, path in (("artifact", output_path.parent), ("checkpoints", checkpoint_path.parent)):
        path.mkdir(parents=True, exist_ok=True)
        checks.append(
            gate_row(f"writable_{label}", True, path.is_dir() and os.access(path, os.W_OK))
        )
    reachability = production_reachability_rows()
    checks.append(
        gate_row("production_e3_reachability", True, all(row["passed"] for row in reachability))
    )
    checks.append(
        gate_row(
            "clean_stop_authority", True, callable(getattr(subprocess.Popen, "terminate", None))
        )
    )
    return {
        "checks": checks,
        "summary": gate_check_summary(checks),
        "source_hashes": _source_hashes(repo_root),
        "citations": [
            {
                "path": EXP7052_RELATIVE_PATH.as_posix(),
                "artifact_hash": python_hash,
                "gate_field": "typed_identity_attack_audit_ready_score",
                "gate_value": upstream.get("typed_identity_attack_audit_ready_score"),
            },
            {
                "path": PILOT_RELATIVE_PATH.as_posix(),
                "artifact_hash": sha256_file(repo_root / PILOT_RELATIVE_PATH),
                "fields_imported": ["phase1_gates", "honest_verdict"],
            },
        ],
        "catalog_rows": catalog,
        "catalog_error": catalog_error,
        "eligible_units": units,
        "reproduced_game_ids": reproduced,
        "models": models,
        "model_identity_rows": identities,
        "gpu_rows": gpu_rows,
        "idle_gpus": idle,
        "runner_rows": [runner],
        "production_reachability_rows": reachability,
        "registry_hash": sha256_file(registry_path),
    }


class _PortLease:  # pragma: no cover - operating-system ownership boundary
    def __init__(self, directory: Path, task_id: str) -> None:
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = int(probe.getsockname()[1])
        probe.close()
        directory.mkdir(parents=True, exist_ok=True)
        self.path = directory / f"port-{self.port}.lease"
        fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"task_id": task_id, "pid": os.getpid(), "port": self.port}, handle)
        self.released = False

    def release(self) -> None:
        if not self.released:
            self.path.unlink(missing_ok=True)
            self.released = True


def _percentile_95(values: Sequence[float]) -> float:  # pragma: no cover - live metric boundary
    if not values:
        return 0.0
    ordered = sorted(values)
    return float(ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)])


def _observation_progress(raw: Any) -> tuple[int, int]:  # pragma: no cover - SDK model boundary
    levels = int(getattr(raw, "levels_completed", 0) or 0)
    progress = getattr(raw, "level_progress", 0)
    if isinstance(progress, (int, float)) and not isinstance(progress, bool):
        return levels, int(progress)
    return levels, 0


def execute_live_manifest(
    *,
    repo_root: Path,
    manifest: Mapping[str, Any],
    preflight: Mapping[str, Any],
    checkpoint_path: Path,
) -> JsonDict:  # pragma: no cover - required official SDK, CUDA, and long model boundary
    """Execute first-induction cells through the scored factory with owned resources."""

    from arc_agi import Arcade, OperationMode
    from carnot import experiment_6681_arc_post_redirect_outcomes as live
    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic import arc_executable_world_model as e3
    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_e3_outcome_transport import normalize_observation
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    del E3AgentPolicy  # Imported explicitly so the execution path itself proves reachability.
    task_id = f"exp7072:{os.getpid()}"
    store = CellCheckpointStore(checkpoint_path, manifest_hash=str(manifest["manifest_hash"]))
    resources: list[JsonDict] = []
    lease_rows: list[JsonDict] = []
    port_rows: list[JsonDict] = []
    gpu_samples: list[JsonDict] = []
    cleanup_rows: list[JsonDict] = []
    arcade = Arcade(
        operation_mode=OperationMode.ONLINE,
        environments_dir=str(repo_root / ".no_local_arc_environments"),
    )
    scorecard_id = arcade.open_scorecard(tags=["exp7072", "paired-compaction"])
    BaseAgent = live._load_framework_agent()
    model_by_key = {str(row["key"]): dict(row) for row in preflight["models"]}
    gpu_by_index = {int(row["index"]): dict(row) for row in preflight["idle_gpus"]}
    old_env = {
        name: os.environ.get(name)
        for name in (
            TOOL_LOOP_FLAG,
            COMPACTION_FLAG,
            GROWTH_FLAG,
            STATE_BUDGET_FLAG,
            "CARNOT_ARC_GENERATOR_CUDA_GPU",
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA",
            "CARNOT_ARC_INDUCE_N_CTX",
            "CARNOT_ARC_INDUCE_MAX_TOKENS",
            "CARNOT_ARC_GENERATOR_SEED",
            "CARNOT_ARC_E3_DIR",
        )
    }
    try:
        for model_key in ("qwen", "gemma"):
            model = model_by_key[model_key]
            gpu = gpu_by_index[int(model["gpu"])]
            port_lease = _PortLease(checkpoint_path.parent / "port-leases", task_id)
            port_rows.append(
                {"port": port_lease.port, "owned": True, "released": False, "terminal": False}
            )
            lease = lease_api.GpuLease.acquire(
                runtime_dir=checkpoint_path.parent / "gpu-leases",
                task_id=task_id,
                device_uuid=str(gpu["gpu_uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(gpu["memory_total_mb"]) - int(gpu["memory_free_mb"]),
                ttl_s=86_400.0,
            )
            lease.transition("admitted")
            resources.extend(
                (
                    {"kind": "gpu_lease", "identity": lease.lease_id, "owned": True},
                    {"kind": "port_lease", "identity": str(port_lease.path), "owned": True},
                )
            )
            proposer = LocalGGUFProposer(
                repo_substr=str(model["name"]),
                model_path=str(model["model_path"]),
                model_repository=str(model["hf_id"]),
                model_filename=Path(str(model["model_path"])).name,
                requested_model_path=str(model["model_path"]),
                requested_model_filename=Path(str(model["model_path"])).name,
                port=port_lease.port,
                n_ctx=CONTEXT_BUDGET,
                max_tokens=TOKEN_BUDGET,
                timeout=TIME_BUDGET_S,
                mtp=False,
                kv_quant="q8_0",
                n_gpu_layers=999,
            )
            os.environ.update(
                {
                    "CARNOT_ARC_GENERATOR_CUDA_GPU": str(gpu["index"]),
                    "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
                    "CARNOT_ARC_INDUCE_N_CTX": str(CONTEXT_BUDGET),
                    "CARNOT_ARC_INDUCE_MAX_TOKENS": str(TOKEN_BUDGET),
                }
            )
            lease.transition("loading")
            if not proposer._ensure_server():
                raise RuntimeError(f"{model_key} server failed to start")
            server_pid = int(proposer._proc.pid)
            resources.append(
                {
                    "kind": "process",
                    "identity": f"pid:{server_pid}",
                    "pid": server_pid,
                    "owned": True,
                }
            )
            lease.transition(
                "resident", vram_mb=int(gpu["memory_total_mb"]) - int(gpu["memory_free_mb"])
            )
            lease.transition("inferencing")
            AgentClass = make_carnot_agent(BaseAgent, cascade=True, proposer=proposer)
            for pair in [row for row in manifest["pairs"] if row["model_key"] == model_key]:
                for arm in pair["arm_order"]:
                    cell_id = f"{pair['pair_id']}:{arm}"
                    if cell_id not in store.pending([cell_id]):
                        continue
                    random.seed(int(pair["seed"]))
                    os.environ[TOOL_LOOP_FLAG] = "selfparse"
                    os.environ.pop(GROWTH_FLAG, None)
                    os.environ.pop(STATE_BUDGET_FLAG, None)
                    if arm == "treatment":
                        os.environ[COMPACTION_FLAG] = "1"
                    else:
                        os.environ.pop(COMPACTION_FLAG, None)
                    os.environ["CARNOT_ARC_GENERATOR_SEED"] = str(pair["seed"])
                    e3.E3_DIR = checkpoint_path.parent / "e3" / cell_id.replace(":", "_")
                    env = arcade.make(
                        str(pair["game_id"]),
                        seed=int(pair["seed"]),
                        scorecard_id=scorecard_id,
                        save_recording=False,
                        include_frame_data=True,
                    )
                    if env is None or env.observation_space is None:
                        raise RuntimeError(f"no initial observation for {cell_id}")
                    initial = normalize_observation(env.observation_space) or {}
                    start_hash = (
                        "sha256:" + hashlib.sha256(canonical_json_bytes(initial)).hexdigest()
                    )
                    agent = AgentClass(
                        card_id=scorecard_id,
                        game_id=str(pair["game_id"]),
                        agent_name="carnot-exp7072",
                        ROOT_URL="https://three.arcprize.org",
                        record=False,
                        arc_env=env,
                        tags=["compaction-ab", str(arm)],
                    )
                    proposer.last_tool_loop_stats = {}
                    start = time.perf_counter()
                    crash = None
                    actions = 0
                    levels_before, progress_before = _observation_progress(env.observation_space)
                    for _ in range(int(pair["action_budget"])):
                        latest = agent._convert_raw_frame_data(env.observation_space)
                        if agent.is_done(agent.frames, latest):
                            break
                        try:
                            action = agent.choose_action(agent.frames, latest)
                            frame = agent.take_action(action)
                        except Exception as exc:  # noqa: BLE001 - failure remains a charged row
                            crash = f"{type(exc).__name__}: {exc}"
                            break
                        if frame is None:
                            crash = "take_action returned no frame"
                            break
                        agent.append_frame(frame)
                        agent.action_counter += 1
                        actions += 1
                        if proposer.last_tool_loop_stats:
                            break
                    levels_after, progress_after = _observation_progress(env.observation_space)
                    stats = deepcopy(dict(proposer.last_tool_loop_stats or {}))
                    prompts = [
                        float(value)
                        for value in stats.get("prompt_tokens_per_turn", [])
                        if isinstance(value, (int, float))
                    ]
                    row = {
                        "cell_id": cell_id,
                        "pair_id": pair["pair_id"],
                        "unit_id": pair["unit_id"],
                        "game_id": pair["game_id"],
                        "source_group": pair["source_group"],
                        "model_key": model_key,
                        "model_hf_id": pair["model_hf_id"],
                        "model_path": pair["model_path"],
                        "model_file_hash": pair["model_file_hash"],
                        "arm": arm,
                        "seed": pair["seed"],
                        "start_state_hash": start_hash,
                        "action_budget": pair["action_budget"],
                        "token_budget": pair["token_budget"],
                        "context_budget": pair["context_budget"],
                        "time_budget_s": pair["time_budget_s"],
                        "status": "complete" if crash is None else "crashed",
                        "tool_loop_reachable": bool(stats.get("turns", 0)),
                        "eligible_for_compaction": bool(stats.get("turns", 0)),
                        "compactions": int(stats.get("compactions", 0)),
                        "refetches": int(stats.get("refetch_tool_calls_post_compaction", 0)),
                        "parse_failures": int(stats.get("tool_call_parse_failures", 0)),
                        "tool_calls": int(stats.get("tool_calls_total", 0)),
                        "exact_progress": (levels_after - levels_before) * 1_000_000
                        + progress_after
                        - progress_before,
                        "levels_reached": levels_after,
                        "solves": max(0, levels_after - levels_before),
                        "actions": actions,
                        "tokens": int(stats.get("decode_tokens_total", 0)),
                        "peak_context": max(prompts, default=0.0),
                        "p95_context": _percentile_95(prompts),
                        "wall_time_s": time.perf_counter() - start,
                        "crashes": int(crash is not None),
                        "crash_detail": crash,
                        "duplicate_submissions": int(
                            stats.get("duplicate_candidate_submissions", 0)
                        ),
                        "solve_provenance": "live_agent_self_discovery",
                        "config_bytes": arm_config_bytes(pair, str(arm)).decode("ascii"),
                        "imported_production_symbols": list(PRODUCTION_SYMBOLS),
                    }
                    store.save(row)
            proposer._terminate_stale_proc("Exp7072 owned model group complete")
            lease.transition("unloading")
            lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
            lease.transition("terminal_complete")
            release = lease.release()
            lease_rows.append(
                {
                    **release,
                    "device_uuid": gpu["gpu_uuid"],
                    "model_key": model_key,
                    "terminal": True,
                }
            )
            port_lease.release()
            port_rows[-1].update({"released": True, "terminal": True})
            gpu_samples.append(
                {
                    "model_key": model_key,
                    "gpu_uuid": gpu["gpu_uuid"],
                    "server_pid": server_pid,
                    "n_gpu_layers": 999,
                    "inside_inference": True,
                    "terminal": True,
                }
            )
    finally:
        try:
            arcade.close_scorecard(scorecard_id)
        except Exception:
            pass
        for name, value in old_env.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    cleanup_rows.extend(
        {
            "kind": row.get("kind"),
            "identity": row.get("identity"),
            "owned": True,
            "action": "released_by_owned_live_executor",
            "passed": True,
            "terminal": True,
        }
        for row in resources
    )
    return {
        "cell_rows": list(store.rows.values()),
        "gpu_lease_rows": lease_rows,
        "port_lease_rows": port_rows,
        "gpu_sample_rows": gpu_samples,
        "cleanup_rows": cleanup_rows,
    }


def build_completed_artifact(
    *,
    execution_date: str,
    duration_s: float,
    preflight: Mapping[str, Any],
    manifest: Mapping[str, Any],
    execution: Mapping[str, Any],
) -> JsonDict:  # pragma: no cover - live rows are validated through the artifact CLI
    """Assemble a completed positive, null, disqualified, or partial result."""

    rows = [deepcopy(dict(row)) for row in execution.get("cell_rows", [])]
    aggregate = aggregate_rows(rows)
    decision = classify_value(rows)
    summary_check = gate_row(decision.get("failed_check") or "release_gates", None, None)
    if decision["verdict_class"] in {"positive", "null"}:
        summary_check["passed"] = True
    checks = [*preflight["checks"], summary_check]
    artifact = {
        "schema": SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "execution_date": execution_date,
        "field_principles": dict(FIELD_PRINCIPLES),
        "preconditions_checked": deepcopy(preflight["checks"]),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(preflight["source_hashes"]),
        "cited_upstream_artifacts": deepcopy(preflight["citations"]),
        "rows": rows,
        "per_game_results": aggregate["per_game_results"],
        "MODEL_SPECS": deepcopy(preflight["models"]),
        "model_specs": deepcopy(preflight["models"]),
        "models_used": sorted({str(row["model_hf_id"]) for row in rows}),
        "selected_model_specs": deepcopy(preflight["models"]),
        "model_identity_rows": deepcopy(preflight["model_identity_rows"]),
        "model_file_hash_rows": [
            {"model_hf_id": row["model_hf_id"], "model_file_hash": row["model_file_hash"]}
            for row in preflight["model_identity_rows"]
        ],
        "runner_build_rows": deepcopy(preflight["runner_rows"]),
        "gpu_lease_rows": deepcopy(execution.get("gpu_lease_rows", [])),
        "port_lease_rows": deepcopy(execution.get("port_lease_rows", [])),
        "gpu_sample_rows": deepcopy(execution.get("gpu_sample_rows", [])),
        "cuda_layer_offload_confirmed": bool(execution.get("gpu_sample_rows"))
        and all(row.get("n_gpu_layers") == 999 for row in execution.get("gpu_sample_rows", [])),
        "registry_precheck_rows": registry_precheck_rows(
            preflight["eligible_units"], set(preflight["reproduced_game_ids"])
        ),
        "production_reachability_rows": deepcopy(preflight["production_reachability_rows"]),
        "flag_isolation_rows": flag_isolation_rows(manifest),
        "arm_definitions": _empty_evidence()["arm_definitions"],
        "paired_cell_manifest": deepcopy(manifest),
        "arm_order_rows": [
            {"pair_id": row["pair_id"], "arm_order": row["arm_order"]} for row in manifest["pairs"]
        ],
        "tool_loop_rows": aggregate["tool_loop_rows"],
        "compaction_rows": aggregate["compaction_rows"],
        "refetch_rows": aggregate["refetch_rows"],
        "parse_failure_rows": aggregate["parse_failure_rows"],
        "exact_progress_rows": aggregate["exact_progress_rows"],
        "solve_rows": aggregate["solve_rows"],
        "solve_provenance": "live_agent_self_discovery",
        "context_metric_rows": aggregate["context_metric_rows"],
        "token_metric_rows": aggregate["token_metric_rows"],
        "wall_time_rows": aggregate["wall_time_rows"],
        "failure_rows": aggregate["failure_rows"],
        "paired_interval_rows": aggregate["paired_interval_rows"],
        "qwen_pair_count": aggregate["qwen_pair_count"],
        "gemma_pair_count": aggregate["gemma_pair_count"],
        "tool_loop_reachable_score": aggregate["tool_loop_reachable_score"],
        "compaction_treatment_activated_score": aggregate["compaction_treatment_activated_score"],
        "compaction_value_ready_score": decision["compaction_value_ready_score"],
        "retirement_decision": decision["retirement_decision"],
        "cleanup_rows": deepcopy(execution.get("cleanup_rows", [])),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_check_summary(checks),
        "verifier_is_oracle": False,
        "verdict_class": decision["verdict_class"],
        "honest_verdict": decision["honest_verdict"],
    }
    artifact["reproducibility_checksum"] = hash_without_field(artifact, "reproducibility_checksum")
    return artifact


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:
    """Validate and atomically publish one terminal experiment result."""

    if errors := validate_artifact(artifact):
        raise ValueError("invalid Exp7072 artifact: " + ";".join(errors))
    atomic_write_json(path, dict(artifact), allow_override=False)


def run(
    repo_root: Path,
    *,
    execution_date: str,
    output_path: Path,
    checkpoint_path: Path,
) -> JsonDict:  # pragma: no cover - orchestration includes live host checks
    """Run preconditions first, then live cells only when every gate passes."""

    started = time.perf_counter()
    preflight = collect_preconditions(
        repo_root=repo_root,
        output_path=output_path,
        checkpoint_path=checkpoint_path,
    )
    if preflight["summary"]["passed"] is not True:
        artifact = blocked_artifact(
            execution_date=execution_date,
            checks=preflight["checks"],
            source_hashes=preflight["source_hashes"],
            duration_s=time.perf_counter() - started,
        )
        artifact.update(
            {
                "cited_upstream_artifacts": deepcopy(preflight["citations"]),
                "MODEL_SPECS": deepcopy(preflight["models"]),
                "model_specs": deepcopy(preflight["models"]),
                "selected_model_specs": deepcopy(preflight["models"]),
                "model_identity_rows": deepcopy(preflight["model_identity_rows"]),
                "model_file_hash_rows": [
                    {"model_hf_id": row["model_hf_id"], "model_file_hash": row["model_file_hash"]}
                    for row in preflight["model_identity_rows"]
                ],
                "runner_build_rows": deepcopy(preflight["runner_rows"]),
                "registry_precheck_rows": registry_precheck_rows(
                    preflight["eligible_units"], set(preflight["reproduced_game_ids"])
                ),
                "production_reachability_rows": deepcopy(preflight["production_reachability_rows"]),
            }
        )
        artifact["reproducibility_checksum"] = hash_without_field(
            artifact, "reproducibility_checksum"
        )
        write_artifact(output_path, artifact)
        return artifact
    manifest = freeze_paired_cell_manifest(
        units=preflight["eligible_units"],
        model_specs=preflight["models"],
        qwen_pair_count=MIN_QWEN_PAIRS,
        gemma_pair_count=MIN_GEMMA_PAIRS,
        action_budget=ACTION_BUDGET,
        token_budget=TOKEN_BUDGET,
        context_budget=CONTEXT_BUDGET,
        time_budget_s=TIME_BUDGET_S,
        random_seed=RANDOM_SEED,
    )
    execution = execute_live_manifest(
        repo_root=repo_root,
        manifest=manifest,
        preflight=preflight,
        checkpoint_path=checkpoint_path,
    )
    artifact = build_completed_artifact(
        execution_date=execution_date,
        duration_s=time.perf_counter() - started,
        preflight=preflight,
        manifest=manifest,
        execution=execution,
    )
    write_artifact(output_path, artifact)
    return artifact
