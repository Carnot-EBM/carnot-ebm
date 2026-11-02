"""Prepare and gate the matched Exp7186 ARC transfer comparison.

The current roadmap names one source file that is absent from the checkout.
This module records that fact before any model load. Its pure comparison helpers
also prevent a future runner from repeating Exp7144's mixed-policy comparison.

Spec refs: REQ-ARC-WMTE-7186 and SCENARIO-ARC-WMTE-7186-*.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence, Set
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any

JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[2]
RUN_DATE = "20260910"
MILESTONE = "2026.09.633"
TASK_ID = "exp7186-arc-withheld-transfer"
RANDOM_SEED = 7_186_202_609_10
SEEDS = (71_860_011, 71_860_019)
MODEL_ID = "unsloth/Qwen3.8-27B-GGUF"
QUANTIZATION = "Q4_K_M"
MODEL_SPECS: list[JsonDict] = [{"hf_id": MODEL_ID, "quantization": QUANTIZATION}]
ARMS = ("adapter_withheld", "adapter_visible_control")
REAL_ENTRYPOINT = "make_carnot_agent:E3AgentPolicy"
ACTION_BUDGET = 120
CELL_TIMEOUT_S = 600
TOTAL_TIMEOUT_S = 2_800
CANARY_MAX_TOKENS = 32
PREVIOUS_GAME = "r11l"
EXPECTED_GAME = "ls20"
ROTATION_RULE = (
    "registry_file_order_after_r11l_with_one_wrap:reproduced+full_game_clear+"
    "levels_reproduced_positive+shipped_adapter"
)
INFERENCE_SUBSTRATE = "preflight_only_no_model_load"
INFERENCE_SUBSTRATE_CLASS = "blocked_no_run"
RESULT_PATH = Path("results/experiment_7186_v633_arc_withheld_transfer.json")
CHECKPOINT_PATH = Path(
    "results/checkpoints/experiment_7186_v633_arc_withheld_transfer/running.json"
)
RAW_DIR = Path("results/raw/experiment_7186_v633_arc_withheld_transfer")
SPEC_PATH = Path("openspec/capabilities/arc-world-model-trust-energy/spec.md")
MISSING_PLANNER_SOURCE_PATH = Path("python/carnot/agentic/arc_eval_runner.py")
MODULE_PATH = Path("python/carnot/experiment_7186_v633_arc_withheld_transfer.py")
WRAPPER_PATH = Path("scripts/experiments/experiment_7186_v633_arc_withheld_transfer.py")
TEST_PATH = Path("tests/python/test_experiment_7186_v633_arc_withheld_transfer.py")

REQUIRED_SOURCE_PATHS = (
    Path("AGENTS.md"),
    Path("CLAUDE.md"),
    Path("CODEX.md"),
    Path("research-program.md"),
    Path("research-references.md"),
    Path("research-roadmap.yaml"),
    Path("ops/exclusion_manifest.yaml"),
    Path("ops/e2e-test-plan.md"),
    Path("python/carnot/agentic/arc_competition_agent.py"),
    Path("python/carnot/agentic/arc_solver_kit.py"),
    MISSING_PLANNER_SOURCE_PATH,
    Path("results/experiment_7144_v627_rebudgeted_arc_loo.json"),
    Path("results/experiment_7128_v626_arc_loo_causal_audit.json"),
    Path("ops/arc_solve_registry.yaml"),
    Path("ops/north-star.md"),
    SPEC_PATH,
    Path("scripts/experiment_template.py"),
    MODULE_PATH,
    WRAPPER_PATH,
    TEST_PATH,
)

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "status",
    "preconditions_checked",
    "run_date",
    "inference_substrate",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "rows",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
    "inference_substrate_class",
    "arc_transfer_complete_score",
    "MODEL_SPECS",
    "model_specs",
    "configuration_diff_rows",
    "adapter_access_rows",
    "per_game_results",
    "solve_provenance",
    "gpu_receipts",
    "runner_receipt",
    "raw_manifest",
)

FIELD_PRINCIPLES: JsonDict = {
    "field_principles": "Echo each field reason so the artifact explains its evidence contract.",
    "status": "Use a terminal state only after the task work is complete or externally blocked.",
    "preconditions_checked": "Name each resource and record its actual availability before measurement.",
    "run_date": "Use 20260910; never copy a historical run date.",
    "inference_substrate": "Describe the computation actually executed, not merely planned.",
    "execution_venue": "Host or device identity limits where the evidence applies.",
    "duration_s": "Measure elapsed work with a monotonic clock; never pad or invent runtime.",
    "source_artifact_hashes": "Hashes bind inputs, code, and frozen contracts to the result.",
    "rows": "Emit one row per unit and arm or condition, including errors and abstentions.",
    "random_seed": "Freeze randomness so another process can reconstruct the study.",
    "reproducibility_checksum": "Hash input contracts, code, seeds, and raw rows to expose drift.",
    "gate_check_summary": "Every blocked verdict names the exact failed check, upstream, field, expected value, and observed value.",
    "verifier_is_oracle": "Declare whether the scored verifier uses the same authority that labels the outcome.",
    "verdict_class": "Use positive | circular_positive | null | blocked | disqualified | partial. Only unfinished own work is partial.",
    "honest_verdict": "A terminal description distinguishes useful evidence, null findings, disqualification, and external blocks.",
    "inference_substrate_class": "Use model_full_generation when the declared work runs; use blocked_no_run only before any qualifying work.",
    "arc_transfer_complete_score": "One requires real distinct arm behavior and matched common settings.",
    "MODEL_SPECS": "The same Qwen3.8 model must serve both live arms.",
    "model_specs": "Resolve model path, quantization, revision, and hash.",
    "configuration_diff_rows": "Exactly one access-policy difference is permitted.",
    "adapter_access_rows": "Executed access receipts establish the intervention.",
    "per_game_results": "Each game, seed, and arm exposes levels and cost.",
    "solve_provenance": "Use live_agent_self_discovery for any reached level; no new solve is claimed.",
    "gpu_receipts": "Device and process evidence authenticate each live cell.",
    "runner_receipt": "Record one generator and its serving strategy.",
    "raw_manifest": "Raw actions and generator receipts preserve reproducibility.",
}

_HASH_PATTERN = re.compile(r"^sha256:[0-9a-f]{64}$")


def canonical_json(value: Any) -> str:
    """Use one JSON spelling for configuration and evidence comparisons."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(value: bytes) -> str:
    """Label a digest so callers cannot confuse it with source text."""

    return "sha256:" + hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hash ordinary source bytes without normalizing line endings."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash all terminal evidence except the checksum that contains the hash."""

    projected = deepcopy(dict(artifact))
    projected.pop("reproducibility_checksum", None)
    return sha256_bytes(canonical_json(projected).encode("utf-8"))


def atomic_write(path: str | Path, value: Mapping[str, Any]) -> None:
    """Replace one JSON artifact only after its complete bytes are durable."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(dict(value), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, target)


def gate_check(
    check: str,
    upstream: str,
    field: str,
    expected: Any,
    observed: Any,
    passed: bool | None = None,
) -> JsonDict:
    """Record both sides and the authority for one falsifiable check."""

    return {
        "check": check,
        "upstream": upstream,
        "field": field,
        "expected_value": deepcopy(expected),
        "observed_value": deepcopy(observed),
        "passed": bool(expected == observed if passed is None else passed),
    }


def gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Expose the first failure while retaining every gate observation."""

    copied = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in copied if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "checks": copied,
        "failed_check": failed.get("check") if failed else None,
        "upstream": failed.get("upstream") if failed else None,
        "field": failed.get("field") if failed else None,
        "expected_value": failed.get("expected_value") if failed else None,
        "observed_value": failed.get("observed_value") if failed else None,
    }


def select_rotated_game(
    registry_rows: Sequence[Mapping[str, Any]],
    *,
    adapter_games: Set[str],
    previous_game: str = PREVIOUS_GAME,
) -> JsonDict:
    """Select the next eligible row after the previous target, with one wrap."""

    eligible: list[tuple[int, Mapping[str, Any]]] = []
    for index, row in enumerate(registry_rows):
        game = str(row.get("game", ""))
        if (
            row.get("reproducibility") == "reproduced"
            and row.get("full_game_clear") is True
            and int(row.get("levels_reproduced", 0) or 0) > 0
            and game in adapter_games
        ):
            eligible.append((index, row))
    if not eligible:
        raise ValueError("no_registry_eligible_adaptered_public_game")
    previous_position = next(
        (
            position
            for position, (_, row) in enumerate(eligible)
            if row.get("game") == previous_game
        ),
        None,
    )
    if previous_position is None:
        raise ValueError(f"previous_rotation_game_not_eligible:{previous_game}")
    selected_index, selected = eligible[(previous_position + 1) % len(eligible)]
    return {
        "game": str(selected["game"]),
        "eligibility_rank": (previous_position + 1) % len(eligible) + 1,
        "registry_index": selected_index,
        "levels_reproduced_precheck": int(selected["levels_reproduced"]),
        "selection_rule": ROTATION_RULE,
    }


def serialize_common_configuration(*, game: str, effective_flags: Mapping[str, Any]) -> str:
    """Freeze every shared policy input before either arm is constructed."""

    configuration = {
        "schema": "carnot.exp7186.common_policy_configuration.v1",
        "game": game,
        "policy_factory": "make_carnot_agent",
        "policy_class": "E3AgentPolicy",
        "policy_entrypoint": REAL_ENTRYPOINT,
        "environment_entrypoint": "carnot.agentic.arc_solver_kit:offline_arcade",
        "model_specs": deepcopy(MODEL_SPECS),
        "generator_settings": {
            "backend": "native_llama.cpp_server",
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "induction_max_tokens": 131_072,
            "canary_max_tokens": CANARY_MAX_TOKENS,
        },
        "budgets": {
            "actions_per_cell": ACTION_BUDGET,
            "cell_timeout_s": CELL_TIMEOUT_S,
            "total_timeout_s": TOTAL_TIMEOUT_S,
        },
        "seed_schedule": list(SEEDS),
        "effective_flags": deepcopy(dict(effective_flags)),
    }
    return canonical_json(configuration)


def construct_policy_configurations(serialized_common: str) -> tuple[JsonDict, JsonDict]:
    """Deserialize twice, then add only each arm's adapter access policy."""

    withheld = json.loads(serialized_common)
    control = json.loads(serialized_common)
    if not isinstance(withheld, dict) or not isinstance(control, dict):
        raise ValueError("serialized_common_configuration_must_be_an_object")
    withheld["adapter_access_policy"] = "selected_adapter_denied"
    control["adapter_access_policy"] = "selected_adapter_allowed"
    return withheld, control


def _leaf_values(value: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten mappings so nested configuration drift has a precise field name."""

    if isinstance(value, Mapping):
        flattened: dict[str, Any] = {}
        for key in sorted(value):
            path = f"{prefix}.{key}" if prefix else str(key)
            flattened.update(_leaf_values(value[key], path))
        return flattened
    return {prefix: deepcopy(value)}


def configuration_diff_rows(
    withheld: Mapping[str, Any], control: Mapping[str, Any]
) -> list[JsonDict]:
    """Return every leaf difference and mark only adapter policy as permitted."""

    left = _leaf_values(withheld)
    right = _leaf_values(control)
    rows: list[JsonDict] = []
    for field in sorted(set(left) | set(right)):
        if left.get(field) != right.get(field):
            rows.append(
                {
                    "field": field,
                    "adapter_withheld": deepcopy(left.get(field)),
                    "adapter_visible_control": deepcopy(right.get(field)),
                    "permitted": field == "adapter_access_policy",
                }
            )
    return rows


def matched_configuration_gate(withheld: Mapping[str, Any], control: Mapping[str, Any]) -> JsonDict:
    """Pass only the single access-policy difference required by the study."""

    rows = configuration_diff_rows(withheld, control)
    observed = [str(row["field"]) for row in rows]
    return gate_check(
        "matched_common_configuration",
        "serialized_common_configuration",
        "canonical_leaf_differences",
        ["adapter_access_policy"],
        observed,
        observed == ["adapter_access_policy"] and rows[0]["permitted"] is True,
    )


def adapter_access_gate(rows: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Require two real denials and two real control accesses, one for each seed."""

    withheld = [row for row in rows if row.get("arm") == "adapter_withheld"]
    control = [row for row in rows if row.get("arm") == "adapter_visible_control"]
    denied = sum(
        1
        for row in withheld
        if row.get("policy") == "selected_adapter_denied"
        and row.get("selected_adapter") == EXPECTED_GAME
        and row.get("recipe_module_loaded_before_denial") is False
        and int(row.get("denied_attempt_count", 0) or 0) > 0
        and int(row.get("allowed_access_count", 0) or 0) == 0
        and row.get("proved") is True
    )
    allowed = sum(
        1
        for row in control
        if row.get("policy") == "selected_adapter_allowed"
        and row.get("selected_adapter") == EXPECTED_GAME
        and int(row.get("allowed_access_count", 0) or 0) > 0
        and int(row.get("denied_attempt_count", 0) or 0) == 0
        and row.get("proved") is True
    )
    observed = {
        "withheld_denial_receipt_count": denied,
        "control_access_receipt_count": allowed,
    }
    expected = {
        "withheld_denial_receipt_count": len(SEEDS),
        "control_access_receipt_count": len(SEEDS),
    }
    return gate_check(
        "executed_adapter_access_intervention",
        "per_cell_adapter_access_receipts",
        "proved_access_counts",
        expected,
        observed,
    )


def completion_gate(
    rows: Sequence[Mapping[str, Any]],
    *,
    adapter_gate_passed: bool,
    configuration_gate_passed: bool,
) -> JsonDict:
    """Require four bounded live E3 cells and both causal construction gates."""

    expected_units = {(seed, arm) for seed in SEEDS for arm in ARMS}
    observed_units = {
        (int(row.get("seed", -1)), str(row.get("arm", "")))
        for row in rows
        if row.get("status") == "complete"
        and row.get("game") == EXPECTED_GAME
        and row.get("entrypoint") == REAL_ENTRYPOINT
        and row.get("policy_class") == "E3AgentPolicy"
        and int(row.get("actions", ACTION_BUDGET + 1) or 0) <= ACTION_BUDGET
        and float(row.get("elapsed_s", CELL_TIMEOUT_S + 1) or 0) <= CELL_TIMEOUT_S
        and int(row.get("generator_invocation_count", 0) or 0) > 0
        and row.get("live_qwen_invoked") is True
        and row.get("supervisor_outcome") == "completed"
        and row.get("gpu_receipt_id")
        and row.get("raw_receipt_id")
    }
    observed = {
        "complete_cell_count": len(observed_units),
        "expected_cells_present": observed_units == expected_units,
        "adapter_gate_passed": bool(adapter_gate_passed),
        "configuration_gate_passed": bool(configuration_gate_passed),
    }
    expected = {
        "complete_cell_count": 4,
        "expected_cells_present": True,
        "adapter_gate_passed": True,
        "configuration_gate_passed": True,
    }
    return gate_check(
        "complete_matched_live_cells",
        "Exp7186 cell schedule",
        "cell_receipts",
        expected,
        observed,
    )


def paired_transfer_rows(rows: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Pair levels by seed and retain zero loss instead of dropping ties."""

    indexed = {
        (int(row["seed"]), str(row["arm"])): row for row in rows if row.get("game") == EXPECTED_GAME
    }
    pairs: list[JsonDict] = []
    for seed in SEEDS:
        withheld = indexed.get((seed, "adapter_withheld"))
        control = indexed.get((seed, "adapter_visible_control"))
        if withheld is None or control is None:
            continue
        withheld_levels = int(withheld.get("levels", 0) or 0)
        control_levels = int(control.get("levels", 0) or 0)
        pairs.append(
            {
                "game": EXPECTED_GAME,
                "seed": seed,
                "adapter_withheld_levels": withheld_levels,
                "adapter_visible_control_levels": control_levels,
                "transfer_loss_levels": control_levels - withheld_levels,
            }
        )
    return pairs


def build_terminal_artifact(
    *,
    run_date: str,
    duration_s: float,
    checks: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    selected_game: Mapping[str, Any] | None = None,
    configuration_rows: Sequence[Mapping[str, Any]] = (),
    cell_rows: Sequence[Mapping[str, Any]] = (),
    model_specs: Sequence[Mapping[str, Any]] = (),
    gpu_receipts: Mapping[str, Any] | None = None,
    runner_receipt: Mapping[str, Any] | None = None,
    raw_manifest: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build one complete blocked, disqualified, null, positive, or partial result."""

    cells = [deepcopy(dict(row)) for row in cell_rows]
    configuration = [deepcopy(dict(row)) for row in configuration_rows]
    config_passed = bool(
        len(configuration) == 1
        and configuration[0].get("field") == "adapter_access_policy"
        and configuration[0].get("permitted") is True
    )
    access_rows = [
        {"seed": row.get("seed"), "arm": row.get("arm"), **deepcopy(dict(row["adapter_access"]))}
        for row in cells
        if isinstance(row.get("adapter_access"), Mapping)
    ]
    access = adapter_access_gate(access_rows)
    completion = completion_gate(
        cells,
        adapter_gate_passed=access["passed"],
        configuration_gate_passed=config_passed,
    )
    all_checks = [deepcopy(dict(row)) for row in checks]
    first_preflight_failure = next(
        (row for row in all_checks if row.get("passed") is not True), None
    )
    if first_preflight_failure is None:
        all_checks.extend(
            [
                gate_check(
                    "matched_common_configuration",
                    "serialized_common_configuration",
                    "configuration_diff_rows",
                    True,
                    config_passed,
                ),
                access,
                completion,
            ]
        )
    complete = bool(
        first_preflight_failure is None
        and config_passed
        and access["passed"]
        and completion["passed"]
        and (gpu_receipts or {}).get("provenance_ok") is True
        and (runner_receipt or {}).get("model_count") == 1
    )
    pairs = paired_transfer_rows(cells)
    if first_preflight_failure is not None:
        verdict_class = "blocked"
        status = "blocked"
        honest_verdict = f"blocked_{first_preflight_failure['check']}"
        substrate = INFERENCE_SUBSTRATE
        substrate_class = INFERENCE_SUBSTRATE_CLASS
    elif not config_passed:
        verdict_class = "disqualified"
        status = "complete"
        honest_verdict = "complete_disqualified_unmatched_common_arm_configuration"
        substrate = INFERENCE_SUBSTRATE
        substrate_class = INFERENCE_SUBSTRATE_CLASS
    elif not complete:
        verdict_class = "partial"
        status = "partial"
        honest_verdict = "partial_adapter_withheld_transfer_incomplete_cells_or_receipts"
        substrate = "bounded_qwen_generation_and_incomplete_live_e3_cells"
        substrate_class = "model_bounded_generation" if cells else "blocked_no_run"
    elif any(int(row["transfer_loss_levels"]) != 0 for row in pairs):
        verdict_class = "positive"
        status = "complete"
        honest_verdict = "complete_positive_descriptive_transfer_loss_no_generalization_claim"
        substrate = "native_qwen38_generation_through_scored_e3_offline_arcade"
        substrate_class = "model_full_generation"
    else:
        verdict_class = "null"
        status = "complete"
        honest_verdict = "complete_null_no_adapter_withheld_transfer_loss"
        substrate = "native_qwen38_generation_through_scored_e3_offline_arcade"
        substrate_class = "model_full_generation"
    artifact: JsonDict = {
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "status": status,
        "preconditions_checked": all_checks,
        "run_date": str(run_date),
        "inference_substrate": substrate,
        "execution_venue": "host",
        "execution_host": os.uname().nodename,
        "duration_s": float(duration_s),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "rows": cells,
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": gate_summary(all_checks),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": honest_verdict,
        "inference_substrate_class": substrate_class,
        "arc_transfer_complete_score": int(complete),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        "model_specs": [deepcopy(dict(row)) for row in model_specs],
        "configuration_diff_rows": configuration,
        "adapter_access_rows": access_rows,
        "per_game_results": pairs,
        "solve_provenance": "live_agent_self_discovery",
        "gpu_receipts": deepcopy(dict(gpu_receipts or {})),
        "runner_receipt": deepcopy(dict(runner_receipt or {})),
        "raw_manifest": [deepcopy(dict(row)) for row in raw_manifest],
        "selected_game": deepcopy(dict(selected_game or {})),
        "new_solve_claimed": False,
        "registered_level_increment": 0,
        "generalization_established": False,
        "sample_scope": "two seeds on one known public game",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(value: Mapping[str, Any] | str | Path) -> list[str]:
    """Validate the closed schema and its terminal consistency."""

    if isinstance(value, (str, Path)):
        artifact = json.loads(Path(value).read_text(encoding="utf-8"))
    else:
        artifact = dict(value)
    errors: list[str] = []
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        errors.append("missing_fields:" + ",".join(missing))
    if artifact.get("field_principles") != FIELD_PRINCIPLES:
        errors.append("field_principles_mismatch")
    if artifact.get("run_date") != RUN_DATE:
        errors.append("run_date_mismatch")
    if artifact.get("MODEL_SPECS") != MODEL_SPECS:
        errors.append("model_specs_declaration_mismatch")
    if artifact.get("execution_venue") != "host":
        errors.append("execution_venue_invalid")
    if artifact.get("verdict_class") not in {
        "positive",
        "circular_positive",
        "null",
        "blocked",
        "disqualified",
        "partial",
    }:
        errors.append("verdict_class_invalid")
    status = artifact.get("status")
    if status not in {"complete", "blocked", "partial"}:
        errors.append("status_invalid")
    if artifact.get("verdict_class") == "blocked":
        if status != "blocked" or artifact.get("inference_substrate_class") != "blocked_no_run":
            errors.append("blocked_terminal_inconsistent")
        if artifact.get("rows"):
            errors.append("blocked_rows_not_empty")
    if artifact.get("arc_transfer_complete_score") == 1:
        if status != "complete" or len(artifact.get("rows", [])) != 4:
            errors.append("complete_score_inconsistent")
    summary = artifact.get("gate_check_summary")
    if not isinstance(summary, Mapping):
        errors.append("gate_summary_missing")
    elif summary != gate_summary(artifact.get("preconditions_checked", [])):
        errors.append("gate_summary_mismatch")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not _HASH_PATTERN.fullmatch(checksum):
        errors.append("reproducibility_checksum_invalid")
    elif checksum != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    if artifact.get("new_solve_claimed") is not False:
        errors.append("new_solve_claim_forbidden")
    if artifact.get("registered_level_increment") != 0:
        errors.append("registry_increment_forbidden")
    return errors


def _snapshot_sources(root: Path) -> tuple[JsonDict, JsonDict]:
    """Read and hash every required source before any measurement starts."""

    sizes: JsonDict = {}
    hashes: JsonDict = {}
    for relative in REQUIRED_SOURCE_PATHS:
        path = root / relative
        try:
            size = path.stat().st_size if path.is_file() else 0
            sizes[str(relative)] = size
            hashes[str(relative)] = sha256_file(path) if size > 0 else "missing"
        except OSError:
            sizes[str(relative)] = 0
            hashes[str(relative)] = "missing"
    return sizes, hashes


def _task_identity(roadmap_text: str) -> JsonDict:
    """Read only the frozen Exp7186 identity fields from the active roadmap."""

    match = re.search(
        r"(?ms)^- id: exp7186-arc-withheld-transfer\n(.*?)(?=^- id:|\Z)", roadmap_text
    )
    task = match.group(0) if match else ""
    return {
        "id": TASK_ID if task else None,
        "milestone": MILESTONE if f"milestone: {MILESTONE}" in task else None,
        "deliverable": (
            RESULT_PATH.as_posix() if f"deliverable: {RESULT_PATH.as_posix()}" in task else None
        ),
    }


def collect_static_preconditions(
    *, root: Path, result_path: Path, checkpoint_path: Path, raw_dir: Path
) -> tuple[list[JsonDict], JsonDict]:
    """Check source, contract, tools, and output paths before model work."""

    sizes, hashes = _snapshot_sources(root)
    spec_path = root / SPEC_PATH
    roadmap_path = root / "research-roadmap.yaml"
    spec_text = spec_path.read_text(encoding="utf-8") if spec_path.is_file() else ""
    roadmap_text = roadmap_path.read_text(encoding="utf-8") if roadmap_path.is_file() else ""
    expected_identity = {
        "id": TASK_ID,
        "milestone": MILESTONE,
        "deliverable": RESULT_PATH.as_posix(),
    }
    observed_identity = _task_identity(roadmap_text)
    tools = {
        "python": Path(sys.executable).is_file(),
        "sha256sum": shutil.which("sha256sum") is not None,
        "nvidia-smi": shutil.which("nvidia-smi") is not None,
    }
    storage = {
        "result_parent_writable": result_path.parent.is_dir()
        and os.access(result_path.parent, os.W_OK),
        "checkpoint_parent_writable": checkpoint_path.parent.is_dir()
        and os.access(checkpoint_path.parent, os.W_OK),
        "raw_dir_writable": raw_dir.is_dir() and os.access(raw_dir, os.W_OK),
    }
    checks = [
        gate_check(
            "driving_capability_spec",
            SPEC_PATH.as_posix(),
            "REQ-ARC-WMTE-7186",
            True,
            "## REQ-ARC-WMTE-7186:" in spec_text,
        ),
        gate_check(
            "required_source_bytes",
            "repository",
            "REQUIRED_SOURCE_PATHS",
            {str(path): "nonempty" for path in REQUIRED_SOURCE_PATHS},
            sizes,
            all(size > 0 for size in sizes.values()),
        ),
        gate_check(
            "required_source_hashes",
            "repository",
            "source_artifact_hashes",
            "sha256:<64 hex> for every required source",
            hashes,
            all(
                isinstance(value, str) and _HASH_PATTERN.fullmatch(value)
                for value in hashes.values()
            ),
        ),
        gate_check(
            "same_milestone_gate_fields",
            "research-roadmap.yaml",
            "id,milestone,deliverable",
            expected_identity,
            observed_identity,
        ),
        gate_check(
            "required_tools",
            "host",
            "python,sha256sum,nvidia-smi",
            {key: True for key in tools},
            tools,
        ),
        gate_check(
            "output_directories",
            "host_filesystem",
            "result,checkpoint,raw",
            {key: True for key in storage},
            storage,
        ),
    ]
    return checks, hashes


def _progress(phase: int, event: str, **fields: Any) -> None:
    """Emit one flushed progress row for the outer task monitor."""

    print(canonical_json({"phase": phase, "event": event, **fields}), flush=True)


def run_experiment(
    *,
    root: Path,
    run_date: str,
    result_path: Path,
    checkpoint_path: Path,
    raw_dir: Path,
    measurement_runner: Callable[..., Any] | None = None,
) -> JsonDict:
    """Run preflight and stop safely when the frozen source contract is absent."""

    started = time.monotonic()
    _progress(0, "phase_start", name="checkpoint_shell_before_checks")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    shell = build_terminal_artifact(
        run_date=run_date,
        duration_s=0.0,
        checks=[gate_check("preflight_started", "experiment_7186", "terminal", True, False)],
        source_hashes={},
    )
    atomic_write(checkpoint_path, shell)
    _progress(0, "phase_end", name="checkpoint_shell_before_checks")

    _progress(1, "phase_start", name="source_contract_tools_and_storage")
    _progress(1, "benchmark_start", operation="static_preconditions")
    checks, source_hashes = collect_static_preconditions(
        root=root,
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    static_ready = all(row.get("passed") is True for row in checks)
    _progress(1, "benchmark_end", operation="static_preconditions", passed=static_ready)
    _progress(1, "phase_end", name="source_contract_tools_and_storage", passed=static_ready)

    if static_ready:  # pragma: no cover - the frozen checkout lacks the required runner source.
        _progress(2, "phase_start", name="matched_configuration_construction")
        if measurement_runner is None:
            checks.append(
                gate_check(
                    "measurement_runner_available",
                    "experiment_7186",
                    "measurement_runner",
                    True,
                    False,
                )
            )
            cell_rows: list[JsonDict] = []
        else:
            cell_rows = list(measurement_runner(root=root, run_date=run_date))
        _progress(2, "phase_end", name="matched_configuration_construction")
    else:
        cell_rows = []
    artifact = build_terminal_artifact(
        run_date=run_date,
        duration_s=time.monotonic() - started,
        checks=checks,
        source_hashes=source_hashes,
        cell_rows=cell_rows,
    )
    _progress(9, "phase_start", name="terminal_validation_and_atomic_write")
    _progress(9, "validation_start", operation="cold_artifact_validation")
    errors = validate_artifact(artifact)
    _progress(9, "validation_end", operation="cold_artifact_validation", errors=errors)
    if errors:
        raise ValueError("terminal_artifact_invalid:" + ",".join(errors))
    _progress(9, "artifact_write_start", path=str(result_path), status=artifact["status"])
    atomic_write(result_path, artifact)
    _progress(9, "artifact_write_end", path=str(result_path), status=artifact["status"])
    _progress(9, "phase_end", name="terminal_validation_and_atomic_write")
    return artifact


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse the fixed run date and explicit storage overrides."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default=RUN_DATE)
    parser.add_argument("--result-path", type=Path, default=RESULT_PATH)
    parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT_PATH)
    parser.add_argument("--raw-dir", type=Path, default=RAW_DIR)
    parser.add_argument("--validate", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - exercised by command E2E.
    """Execute one bounded attempt or validate an existing terminal artifact."""

    _progress(0, "phase_start", name="entrypoint_before_checks")
    args = parse_args(argv)
    root = Path(os.environ.get("CARNOT_REPO_ROOT", REPO_ROOT)).resolve()
    if args.validate is not None:
        _progress(9, "validation_start", path=str(args.validate))
        errors = validate_artifact(args.validate)
        _progress(9, "validation_end", path=str(args.validate), errors=errors)
        return int(bool(errors))
    result_path = args.result_path if args.result_path.is_absolute() else root / args.result_path
    checkpoint_path = (
        args.checkpoint_path if args.checkpoint_path.is_absolute() else root / args.checkpoint_path
    )
    raw_dir = args.raw_dir if args.raw_dir.is_absolute() else root / args.raw_dir
    artifact = run_experiment(
        root=root,
        run_date=str(args.date),
        result_path=result_path,
        checkpoint_path=checkpoint_path,
        raw_dir=raw_dir,
    )
    print(
        canonical_json(
            {
                "artifact": str(result_path),
                "status": artifact["status"],
                "verdict": artifact["honest_verdict"],
            }
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
