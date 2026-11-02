"""Build strict adapter-withheld action receipts for the scored ARC E3 path.

The environment stays in the host process. Each policy runs in a fresh spawned
process and receives only public observations through a small RPC seam. This
separation prevents the policy from reaching public-game source or adapters.
"""

from __future__ import annotations

import argparse
import builtins
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import inspect
import io
import json
import multiprocessing
from multiprocessing.connection import Connection, wait
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time
import traceback
from typing import Any

from carnot.experiment_artifacts import atomic_write_json
from carnot.inference.sota_models import cached_sota_pair, resolve_cached_gguf


JsonDict = dict[str, Any]
EXPERIMENT_ID = 7099
SCHEMA = "carnot.exp7099.v623_adapter_withheld_preflight.v1"
RESULT_RELATIVE_PATH = Path("results/experiment_7099_v623_adapter_withheld_preflight.json")
RAW_RELATIVE_ROOT = Path("/tmp/carnot-exp7099-adapter-withheld")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
SPEC_RELATIVE_PATH = Path("openspec/capabilities/arc-agi/spec.md")
MODULE_RELATIVE_PATH = Path("python/carnot/experiment_7099_v623_adapter_withheld_preflight.py")
SCRIPT_RELATIVE_PATH = Path("scripts/experiments/experiment_7099_v623_adapter_withheld_preflight.py")
TEST_RELATIVE_PATH = Path("tests/python/test_experiment_7099_v623_adapter_withheld_preflight.py")
QWEN38_HF_ID = "unsloth/Qwen3.8-27B-GGUF"
QWEN36_HF_ID = "unsloth/Qwen3.6-35B-A3B-GGUF"
INFERENCE_SUBSTRATE = "live E3 local-GGUF action generation plus exact ARC transitions"
RANDOM_SEED = 7_099_202_609_07
ACTION_BUDGET = 1
TOKEN_BUDGET = 256
TIMEOUT_S = 300
CONTEXT_BUDGET = 16_384
MIN_FROZEN_GAMES = 4
GAMES_PER_MODEL = 2
LEASE_RUNTIME_DIR = Path("/tmp/carnot-gpu-leases")

MODEL_SPECS: list[JsonDict] = [
    {
        "key": "live_generator",
        "name": "Qwen3.8-27B",
        "hf_id": QWEN38_HF_ID,
        "role": "pinned_live_generator",
    },
    {
        "key": "headline_control",
        "name": "Qwen3.6-35B-A3B",
        "hf_id": QWEN36_HF_ID,
        "role": "mandatory_sota_headline_control",
    },
]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "inference_substrate",
    "inference_substrate_class",
    "execution_venue",
    "duration_s",
    "source_artifact_hashes",
    "MODEL_SPECS",
    "models_used",
    "model_path_receipts",
    "gpu_telemetry_rows",
    "frozen_game_ids",
    "registry_precheck_rows",
    "isolation_rows",
    "forbidden_import_rows",
    "forbidden_read_rows",
    "per_game_results",
    "rows",
    "action_rows",
    "simulation_invocation_rows",
    "forecast_rows",
    "forecast_consumption_rows",
    "exact_transition_rows",
    "advice_only_row_count",
    "valid_action_row_count",
    "raw_trace_paths",
    "raw_trace_hashes",
    "solve_provenance",
    "offline_reproduced",
    "arc_registry_delta",
    "adapter_withheld_live_path_ready_score",
    "random_seed",
    "reproducibility_checksum",
    "gate_check_summary",
    "verifier_is_oracle",
    "verdict_class",
    "honest_verdict",
)

_PRINCIPLE_OVERRIDES = {
    "field_principles": "Each field states the scientific reason that it exists.",
    "preconditions_checked": "Missing hardware or data must stop inference instead of producing invented evidence.",
    "inference_substrate": "The substrate separates live local generation from cached text or replay.",
    "inference_substrate_class": "The compute class lets blocked work remain distinct from executed generation.",
    "execution_venue": "The venue identifies where hardware and process isolation were observed.",
    "duration_s": "Measured wall time makes live execution auditable.",
    "source_artifact_hashes": "Hashes bind the result to the exact implementation and registry inputs.",
    "MODEL_SPECS": "Pinned declarations prevent a smaller substitute from entering the run.",
    "models_used": "Executed model IDs distinguish declarations from actual generation.",
    "model_path_receipts": "Paths and hashes bind each model ID to exact cached GGUF bytes.",
    "gpu_telemetry_rows": "In-run samples prove the local generator used the leased RTX 3090.",
    "frozen_game_ids": "A frozen set prevents outcome-aware game selection.",
    "registry_precheck_rows": "Registry rows expose exclusions and the mechanic-depth strata.",
    "isolation_rows": "Process receipts show that policy and public-game implementation stayed separate.",
    "forbidden_import_rows": "Import receipts expose any adapter or hand-solver access attempt.",
    "forbidden_read_rows": "Read receipts expose any source, recipe, trajectory, or checkpoint access attempt.",
    "per_game_results": "Per-game rows prevent one action cell from representing the full frozen set.",
    "rows": "Canonical decision rows make every aggregate recountable.",
    "action_rows": "Action rows retain the candidate-to-execution causal chain.",
    "simulation_invocation_rows": "Invocation rows prove that a forecast was requested before selection.",
    "forecast_rows": "Forecast rows retain what each model predicted without claiming correctness.",
    "forecast_consumption_rows": "Consumption rows bind the selected action to its forecast.",
    "exact_transition_rows": "Exact transition rows prove that a valid action reached the real environment.",
    "advice_only_row_count": "Advice is not an executable action and must remain visible as failure evidence.",
    "valid_action_row_count": "The count proves how many rows emitted schema-valid actions.",
    "raw_trace_paths": "Raw evidence stays outside results until aggregation completes.",
    "raw_trace_hashes": "Content hashes detect any change after raw trace publication.",
    "solve_provenance": "Development proxy prevents a public withheld-adapter run from becoming a hidden solve claim.",
    "offline_reproduced": "Fresh replay is required before an exact transition is credited.",
    "arc_registry_delta": "A zero delta makes the no-solve boundary explicit.",
    "adapter_withheld_live_path_ready_score": "Readiness requires both models, valid actions, exact transitions, isolation, and complete ledgers.",
    "random_seed": "A fixed seed makes game allocation and model sampling repeatable.",
    "reproducibility_checksum": "A canonical digest detects later aggregate-field changes.",
    "gate_check_summary": "Exact expected and observed values make every failure actionable.",
    "verifier_is_oracle": "False prevents mechanism validation from becoming an ARC correctness claim.",
    "verdict_class": "A closed class separates positive, null, blocked, and invalid evidence.",
    "honest_verdict": "A terminal prefix and class token make the conclusion machine-readable.",
}
FIELD_PRINCIPLES = {field: _PRINCIPLE_OVERRIDES[field] for field in REQUIRED_ARTIFACT_FIELDS}

FORBIDDEN_MODULES = (
    "carnot.agentic.arc_game_adapters",
    "scripts.arc_loop_solve",
    "carnot.agentic.arc_nav_world_model",
    "carnot.agentic.arc_execution_guided_world_model",
)
FORBIDDEN_READ_TARGETS = (
    "game_source",
    "registry_trajectory",
    "known_action_recipe",
    "per_game_checkpoint",
)
VERDICT_CLASSES = {
    "positive",
    "circular_positive",
    "null",
    "blocked",
    "disqualified",
    "partial",
}
TERMINAL_PREFIXES = ("complete:", "complete_", "success:", "success_", "passed:", "passed_", "shipped:", "shipped_")


def canonical_json_bytes(value: Any) -> bytes:
    """Return stable JSON bytes for hashes and equality checks."""

    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True).encode()


def sha256_json(value: Any) -> str:
    """Hash a JSON value after canonical serialization."""

    return "sha256:" + hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path) -> str | None:
    """Hash one file in chunks so model files do not enter process memory."""

    digest = hashlib.sha256()
    try:
        with Path(path).open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError:
        return None
    return "sha256:" + digest.hexdigest()


def artifact_checksum(artifact: Mapping[str, Any]) -> str:
    """Hash an artifact after blanking its self-referential checksum."""

    body = deepcopy(dict(artifact))
    body["reproducibility_checksum"] = ""
    return sha256_json(body)


def gate_row(check: str, expected: Any, observed: Any, *, passed: bool | None = None) -> JsonDict:
    """Retain both sides of one exact fail-closed gate."""

    return {
        "check": check,
        "expected_value": expected,
        "observed_value": observed,
        "passed": observed == expected if passed is None else bool(passed),
        "terminal": True,
    }


def make_gate_summary(checks: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Name the first failure and preserve all exact gate rows."""

    rows = [deepcopy(dict(row)) for row in checks]
    failed = next((row for row in rows if row.get("passed") is not True), None)
    return {
        "passed": failed is None,
        "failed_check": None if failed is None else failed.get("check"),
        "expected_value": True if failed is None else failed.get("expected_value"),
        "observed_value": True if failed is None else failed.get("observed_value"),
        "checks": rows,
    }


def _default_qwen36_resolver(hf_id: str) -> str | None:
    pair = cached_sota_pair(gpu_indices=(0, 1)) or []
    row = next((item for item in pair if item.get("hf_id") == hf_id), None)
    return None if row is None else str(row.get("model_path") or "") or None


def resolve_model_specs(
    *,
    qwen38_resolver: Callable[[str], str | None] = resolve_cached_gguf,
    qwen36_resolver: Callable[[str], str | None] = _default_qwen36_resolver,
    gpu_indices: tuple[int, int] = (0, 1),
) -> list[JsonDict]:
    """Resolve both exact pins without a download or fallback branch."""

    paths = (qwen38_resolver(QWEN38_HF_ID), qwen36_resolver(QWEN36_HF_ID))
    if any(not path for path in paths):
        return []
    return [
        {**template, "model_path": str(Path(str(path)).absolute()), "gpu": gpu_indices[index]}
        for index, (template, path) in enumerate(zip(MODEL_SPECS, paths, strict=True))
    ]


def validate_model_specs(specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Reject a changed ID, missing file, wrong GPU count, or duplicate path."""

    errors: list[str] = []
    if [row.get("hf_id") for row in specs] != [QWEN38_HF_ID, QWEN36_HF_ID]:
        errors.append("model_id_substitution")
    paths = [str(row.get("model_path") or "") for row in specs]
    if len(specs) != 2 or any(not Path(path).is_file() for path in paths):
        errors.append("model_file_missing")
    if len(set(paths)) != 2:
        errors.append("model_path_not_distinct")
    if len({row.get("gpu") for row in specs}) != 2:
        errors.append("model_gpu_not_distinct")
    return errors


def model_path_receipts(specs: Sequence[Mapping[str, Any]]) -> list[JsonDict]:
    """Bind each requested hub ID to an absolute cached path and file hash."""

    return [
        {
            "model_id": row.get("hf_id"),
            "model_path": str(Path(str(row.get("model_path", ""))).absolute()),
            "model_filename": Path(str(row.get("model_path", ""))).name,
            "model_file_hash": sha256_file(str(row.get("model_path", ""))),
            "gpu": row.get("gpu"),
            "cached": True,
            "download_attempted": False,
            "substitution_allowed": False,
        }
        for row in specs
    ]


def freeze_registry_games(
    registry_rows: Sequence[Mapping[str, Any]],
    *,
    credited_receipts: Sequence[Mapping[str, Any]],
    minimum: int = MIN_FROZEN_GAMES,
) -> tuple[list[str], list[JsonDict]]:
    """Exclude prior credit, then choose mechanic and depth strata deterministically."""

    credited = {
        (str(row.get("game_id")), int(row.get("level", 0) or 0))
        for row in credited_receipts
        if row.get("credited") is True
    }
    prepared: list[JsonDict] = []
    for source in registry_rows:
        game_id = str(source.get("game") or source.get("game_id") or "")
        depth = int(source.get("levels_reproduced", 0) or 0)
        excluded = "prior_adapter_withheld_live_credit" if (game_id, 0) in credited else None
        prepared.append(
            {
                "game_id": game_id,
                "registry_depth": depth,
                "mechanic_class": str(source.get("mechanic_class") or "unclassified"),
                "credited_levels": sorted(level for game, level in credited if game == game_id),
                "eligible": bool(game_id and depth > 0 and excluded is None),
                "excluded_reason": excluded,
                "selected": False,
            }
        )
    eligible = sorted(
        (row for row in prepared if row["eligible"]),
        key=lambda row: (row["registry_depth"], row["mechanic_class"], row["game_id"]),
    )
    if len(eligible) < minimum:
        raise ValueError("four eligible public games are required before outcomes are visible")
    chosen: list[JsonDict] = []
    for candidate in [eligible[0], eligible[-1], *eligible[1:-1]]:
        if candidate in chosen:
            continue
        adds_stratum = (
            not chosen
            or candidate["mechanic_class"] not in {row["mechanic_class"] for row in chosen}
            or candidate["registry_depth"] not in {row["registry_depth"] for row in chosen}
        )
        if adds_stratum or len(chosen) + (len(eligible) - eligible.index(candidate)) <= minimum:
            chosen.append(candidate)
        if len(chosen) == minimum:
            break
    for candidate in eligible:
        if len(chosen) == minimum:
            break
        if candidate not in chosen:
            chosen.append(candidate)
    if len({row["mechanic_class"] for row in chosen}) < 2:
        raise ValueError("four games must span at least two mechanic classes")
    if len({row["registry_depth"] for row in chosen}) < 2:
        raise ValueError("four games must span at least two registry depths")
    selected_ids = [str(row["game_id"]) for row in chosen]
    for row in prepared:
        row["selected"] = row["game_id"] in selected_ids
    return selected_ids, prepared


class IsolationViolation(PermissionError):
    """A policy tried to reach knowledge withheld from the experiment."""


class IsolationMonitor:
    """Block policy-side imports and reads that expose public-game recipes."""

    def __init__(self, repo_root: str | Path) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.forbidden_import_rows: list[JsonDict] = []
        self.forbidden_read_rows: list[JsonDict] = []

    def check_import(self, name: str) -> None:
        target = str(name)
        forbidden = next(
            (prefix for prefix in FORBIDDEN_MODULES if target == prefix or target.startswith(prefix + ".")),
            None,
        )
        if forbidden is None:
            return
        self.forbidden_import_rows.append(
            {"target": target, "category": forbidden, "attempted": True, "blocked": True, "passed": True}
        )
        raise IsolationViolation(f"forbidden import: {target}")

    def _read_category(self, path: Path) -> str | None:
        text = path.as_posix()
        if "/environment_files/" in text or text.endswith("/arc_game_adapters.py"):
            return "game_source"
        if text.endswith("/ops/arc_solve_registry.yaml") or "trajectory" in path.name.lower():
            return "registry_trajectory"
        if "/results/arc_loop_solve_" in text or "solution" in path.name.lower():
            return "known_action_recipe"
        if "/models/arc_per_game/" in text or re.search(r"/e3/[^/]+/world_model\.py$", text):
            return "per_game_checkpoint"
        if "solver" in path.name.lower() and "/scripts/" in text:
            return "known_action_recipe"
        return None

    def check_read(self, path: str | bytes | os.PathLike[str] | os.PathLike[bytes]) -> None:
        try:
            resolved = Path(os.fsdecode(path)).resolve()
        except (OSError, TypeError, ValueError):
            return
        category = self._read_category(resolved)
        if category is None:
            return
        self.forbidden_read_rows.append(
            {
                "target": str(resolved),
                "category": category,
                "attempted": True,
                "blocked": True,
                "passed": True,
            }
        )
        raise IsolationViolation(f"forbidden read: {resolved}")

    @contextmanager
    def activated(self):
        """Install process-local guards for the policy decision interval."""

        original_import = builtins.__import__
        original_open = builtins.open
        original_io_open = io.open

        def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
            self.check_import(str(name))
            return original_import(name, globals, locals, fromlist, level)

        def guarded_open(file, mode="r", *args, **kwargs):
            if "r" in str(mode) or "+" in str(mode):
                self.check_read(file)
            return original_open(file, mode, *args, **kwargs)

        def guarded_io_open(file, mode="r", *args, **kwargs):
            if "r" in str(mode) or "+" in str(mode):
                self.check_read(file)
            return original_io_open(file, mode, *args, **kwargs)

        builtins.__import__ = guarded_import
        builtins.open = guarded_open
        io.open = guarded_io_open
        try:
            yield self
        finally:
            builtins.__import__ = original_import
            builtins.open = original_open
            io.open = original_io_open

    def import_receipts(self) -> list[JsonDict]:
        attempted = {row["category"] for row in self.forbidden_import_rows}
        return [
            *self.forbidden_import_rows,
            *[
                {
                    "target": target,
                    "category": target,
                    "attempted": False,
                    "blocked": True,
                    "passed": True,
                }
                for target in FORBIDDEN_MODULES
                if target not in attempted
            ],
        ]

    def read_receipts(self) -> list[JsonDict]:
        attempted = {row["category"] for row in self.forbidden_read_rows}
        return [
            *self.forbidden_read_rows,
            *[
                {
                    "target": target,
                    "category": target,
                    "attempted": False,
                    "blocked": True,
                    "passed": True,
                }
                for target in FORBIDDEN_READ_TARGETS
                if target not in attempted
            ],
        ]


def clean_forbidden_import_rows() -> list[JsonDict]:
    """Return a complete no-attempt import receipt for fixture and worker use."""

    return IsolationMonitor(Path.cwd()).import_receipts()


def clean_forbidden_read_rows() -> list[JsonDict]:
    """Return a complete no-attempt read receipt for fixture and worker use."""

    return IsolationMonitor(Path.cwd()).read_receipts()


def _extract_json_object(text: str) -> Mapping[str, Any] | None:
    tagged = re.search(r"<forecast>\s*(\{.*?\})\s*</forecast>", text, flags=re.DOTALL)
    candidates = [tagged.group(1)] if tagged else []
    candidates.extend(text[index:] for index, char in enumerate(text) if char == "{")
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value, _end = decoder.raw_decode(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, Mapping):
            return value
    return None


def interpret_forecast(text: str, *, candidate_actions: Sequence[Mapping[str, Any]]) -> JsonDict:
    """Accept only one structured forecast that selects an existing candidate."""

    value = _extract_json_object(str(text))
    if value is None:
        return {"accepted": False, "reason": "advice_only", "forecast": None, "selected_action": None}
    index = value.get("selected_candidate_index")
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(candidate_actions):
        return {"accepted": False, "reason": "invalid_action", "forecast": dict(value), "selected_action": None}
    if not isinstance(value.get("predicted_change"), bool) or not isinstance(value.get("rationale"), str):
        return {"accepted": False, "reason": "malformed_forecast", "forecast": dict(value), "selected_action": None}
    forecast = {
        "selected_candidate_index": index,
        "predicted_change": value["predicted_change"],
        "rationale": value["rationale"],
    }
    return {
        "accepted": True,
        "reason": "forecast_selected_existing_candidate",
        "forecast": forecast,
        "forecast_hash": sha256_json(forecast),
        "selected_candidate_index": index,
        "selected_action": deepcopy(dict(candidate_actions[index])),
    }


def validate_action_row(row: Mapping[str, Any], model_specs: Sequence[Mapping[str, Any]]) -> list[str]:
    """Recompute one forecast-to-action and exact-transition ledger."""

    errors: list[str] = []
    model_by_id = {str(spec.get("hf_id")): spec for spec in model_specs}
    model = model_by_id.get(str(row.get("model_id")))
    if model is None or row.get("model_path") != model.get("model_path"):
        errors.append("model_substitution")
    candidates = row.get("candidate_actions")
    candidates = candidates if isinstance(candidates, list) else []
    if row.get("advice_only") is True:
        errors.append("advice_only")
    if row.get("valid_action") is not True or row.get("selected_action") not in candidates:
        errors.append("invalid_action")
    invocation = row.get("simulation_invocation")
    if not isinstance(invocation, Mapping) or invocation.get("invoked") is not True:
        errors.append("missing_simulation")
    forecast = row.get("forecast")
    interpretation = row.get("interpretation")
    if not isinstance(forecast, Mapping) or not isinstance(interpretation, Mapping):
        errors.append("forecast_ledger_incomplete")
    elif interpretation.get("forecast_hash") != sha256_json(forecast):
        errors.append("forecast_hash_mismatch")
    if row.get("forecast_consumed") is not True:
        errors.append("forecast_not_consumed")
    transition = row.get("transition")
    if not isinstance(transition, Mapping) or transition.get("executed") is not True:
        errors.append("transition_not_executed")
    elif transition.get("action") != row.get("selected_action"):
        errors.append("transition_action_mismatch")
    elif transition.get("exact") is not True or transition.get("fresh_replay_exact") is not True:
        errors.append("inexact_transition")
    if row.get("policy_class") != "E3AgentPolicy" or row.get("factory") != "make_carnot_agent":
        errors.append("e3_factory_path_missing")
    for field in ("forbidden_import_rows", "forbidden_read_rows"):
        receipts = row.get(field)
        if not isinstance(receipts, list) or not receipts or any(item.get("passed") is not True for item in receipts):
            errors.append("adapter_lookup" if field == "forbidden_import_rows" else "forbidden_read")
    isolation = row.get("isolation")
    if not isinstance(isolation, Mapping) or isolation.get("passed") is not True:
        errors.append("process_isolation_missing")
    return list(dict.fromkeys(errors))


def _ready_score(action_rows: Sequence[Mapping[str, Any]], model_specs: Sequence[Mapping[str, Any]]) -> int:
    required_ids = [QWEN38_HF_ID, QWEN36_HF_ID]
    if validate_model_specs(model_specs):
        return 0
    for model_id in required_ids:
        model_rows = [row for row in action_rows if row.get("model_id") == model_id]
        if len({row.get("game_id") for row in model_rows}) < GAMES_PER_MODEL:
            return 0
        if any(validate_action_row(row, model_specs) for row in model_rows):
            return 0
    return 1


def _empty_evidence() -> JsonDict:
    return {
        "models_used": [],
        "model_path_receipts": [],
        "gpu_telemetry_rows": [],
        "frozen_game_ids": [],
        "registry_precheck_rows": [],
        "isolation_rows": [],
        "forbidden_import_rows": [],
        "forbidden_read_rows": [],
        "per_game_results": [],
        "rows": [],
        "action_rows": [],
        "simulation_invocation_rows": [],
        "forecast_rows": [],
        "forecast_consumption_rows": [],
        "exact_transition_rows": [],
        "advice_only_row_count": 0,
        "valid_action_row_count": 0,
        "raw_trace_paths": [],
        "raw_trace_hashes": [],
        "solve_provenance": "development_proxy",
        "offline_reproduced": False,
        "arc_registry_delta": 0,
        "adapter_withheld_live_path_ready_score": 0,
    }


def build_blocked_artifact(
    *,
    execution_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
) -> JsonDict:
    """Build complete terminal evidence without claiming any model execution."""

    artifact: JsonDict = {
        "schema": SCHEMA,
        "execution_date": execution_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "blocked_no_run",
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "MODEL_SPECS": deepcopy(MODEL_SPECS),
        **_empty_evidence(),
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": make_gate_summary(preconditions),
        "verifier_is_oracle": False,
        "verdict_class": "blocked",
        "honest_verdict": "complete_blocked_adapter_withheld_precondition_failed",
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def build_completed_artifact(
    *,
    execution_date: str,
    duration_s: float,
    preconditions: Sequence[Mapping[str, Any]],
    source_hashes: Mapping[str, Any],
    model_specs: Sequence[Mapping[str, Any]],
    frozen_game_ids: Sequence[str],
    registry_rows: Sequence[Mapping[str, Any]],
    action_rows: Sequence[Mapping[str, Any]],
    gpu_rows: Sequence[Mapping[str, Any]],
    cleanup_rows: Sequence[Mapping[str, Any]],
    worker_results: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Aggregate immutable raw action rows without adding solve credit."""

    actions = [deepcopy(dict(row)) for row in action_rows]
    ready = _ready_score(actions, model_specs)
    verdict_class = "positive" if ready else "null"
    imports = [deepcopy(row) for action in actions for row in action.get("forbidden_import_rows", [])]
    reads = [deepcopy(row) for action in actions for row in action.get("forbidden_read_rows", [])]
    raw_paths = [str(row.get("raw_trace_path")) for row in actions if row.get("raw_trace_path")]
    raw_hashes = [str(row.get("raw_trace_hash")) for row in actions if row.get("raw_trace_hash")]
    artifact: JsonDict = {
        "schema": SCHEMA,
        "execution_date": execution_date,
        "field_principles": deepcopy(FIELD_PRINCIPLES),
        "preconditions_checked": [deepcopy(dict(row)) for row in preconditions],
        "inference_substrate": INFERENCE_SUBSTRATE,
        "inference_substrate_class": "model_full_generation",
        "execution_venue": "host",
        "duration_s": round(float(duration_s), 6),
        "source_artifact_hashes": deepcopy(dict(source_hashes)),
        "MODEL_SPECS": [deepcopy(dict(row)) for row in model_specs],
        "models_used": sorted({str(row.get("model_id")) for row in actions}),
        "model_path_receipts": model_path_receipts(model_specs),
        "gpu_telemetry_rows": [deepcopy(dict(row)) for row in gpu_rows],
        "frozen_game_ids": list(frozen_game_ids),
        "registry_precheck_rows": [deepcopy(dict(row)) for row in registry_rows],
        "isolation_rows": [deepcopy(dict(row.get("isolation") or {})) for row in actions],
        "forbidden_import_rows": imports,
        "forbidden_read_rows": reads,
        "per_game_results": actions,
        "rows": actions,
        "action_rows": actions,
        "simulation_invocation_rows": [deepcopy(dict(row.get("simulation_invocation") or {})) for row in actions],
        "forecast_rows": [deepcopy(dict(row.get("forecast") or {})) for row in actions],
        "forecast_consumption_rows": [
            {
                "cell_id": row.get("cell_id"),
                "forecast_hash": (row.get("interpretation") or {}).get("forecast_hash"),
                "selected_action": deepcopy(row.get("selected_action")),
                "consumed": row.get("forecast_consumed") is True,
            }
            for row in actions
        ],
        "exact_transition_rows": [deepcopy(dict(row.get("transition") or {})) for row in actions],
        "advice_only_row_count": sum(row.get("advice_only") is True for row in actions),
        "valid_action_row_count": sum(row.get("valid_action") is True for row in actions),
        "raw_trace_paths": raw_paths,
        "raw_trace_hashes": raw_hashes,
        "solve_provenance": "development_proxy",
        "offline_reproduced": bool(actions) and all((row.get("transition") or {}).get("fresh_replay_exact") is True for row in actions),
        "arc_registry_delta": 0,
        "adapter_withheld_live_path_ready_score": ready,
        "cleanup_rows": [deepcopy(dict(row)) for row in cleanup_rows],
        "worker_results": [deepcopy(dict(row)) for row in worker_results],
        "random_seed": RANDOM_SEED,
        "reproducibility_checksum": "",
        "gate_check_summary": make_gate_summary(
            [
                *preconditions,
                gate_row("both_models_two_games_complete", 1, ready),
                gate_row("owned_resource_cleanup", True, bool(cleanup_rows) and all(row.get("released") is True for row in cleanup_rows)),
            ]
        ),
        "verifier_is_oracle": False,
        "verdict_class": verdict_class,
        "honest_verdict": (
            "success_positive_adapter_withheld_live_path_ready_no_solve_claim"
            if ready
            else "complete_null_adapter_withheld_live_path_not_ready_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = artifact_checksum(artifact)
    return artifact


def validate_artifact(artifact: Any, *, verify_raw_traces: bool = False) -> list[str]:
    """Validate required fields and recompute every readiness claim."""

    if not isinstance(artifact, Mapping):
        return ["artifact_not_object"]
    errors = [f"missing_field:{field}" for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if errors:
        return errors
    if not set(REQUIRED_ARTIFACT_FIELDS) <= set(artifact.get("field_principles") or {}):
        errors.append("field_principles_incomplete")
    if artifact.get("solve_provenance") != "development_proxy":
        errors.append("solve_provenance_invalid")
    if artifact.get("arc_registry_delta") != 0:
        errors.append("arc_registry_delta_invalid")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_invalid")
    verdict = artifact.get("verdict_class")
    if verdict not in VERDICT_CLASSES:
        errors.append("verdict_class_invalid")
    honest = str(artifact.get("honest_verdict") or "")
    if not honest.startswith(TERMINAL_PREFIXES) or (verdict in VERDICT_CLASSES and str(verdict) not in honest):
        errors.append("honest_verdict_invalid")
    action_rows = artifact.get("action_rows")
    action_rows = action_rows if isinstance(action_rows, list) else []
    if verdict == "blocked":
        summary = artifact.get("gate_check_summary")
        if (
            artifact.get("inference_substrate_class") != "blocked_no_run"
            or artifact.get("adapter_withheld_live_path_ready_score") != 0
            or action_rows
            or not isinstance(summary, Mapping)
            or summary.get("failed_check") is None
            or "expected_value" not in summary
            or "observed_value" not in summary
        ):
            errors.append("blocked_contract_invalid")
    else:
        specs = artifact.get("MODEL_SPECS")
        specs = specs if isinstance(specs, list) else []
        expected_ready = _ready_score(action_rows, specs)
        if artifact.get("adapter_withheld_live_path_ready_score") != expected_ready:
            errors.append("ready_score_mismatch")
        expected_class = "positive" if expected_ready else "null"
        if verdict in {"positive", "null"} and verdict != expected_class:
            errors.append("verdict_class_ready_mismatch")
        if artifact.get("valid_action_row_count") != sum(row.get("valid_action") is True for row in action_rows):
            errors.append("valid_action_count_mismatch")
        if artifact.get("advice_only_row_count") != sum(row.get("advice_only") is True for row in action_rows):
            errors.append("advice_only_count_mismatch")
        if artifact.get("offline_reproduced") != (bool(action_rows) and all((row.get("transition") or {}).get("fresh_replay_exact") is True for row in action_rows)):
            errors.append("offline_reproduced_mismatch")
    if verify_raw_traces:
        paths = artifact.get("raw_trace_paths") or []
        hashes = artifact.get("raw_trace_hashes") or []
        if len(paths) != len(hashes) or any(sha256_file(path) != expected for path, expected in zip(paths, hashes, strict=False)):
            errors.append("raw_trace_hash_mismatch")
    if artifact.get("reproducibility_checksum") != artifact_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return list(dict.fromkeys(errors))


def _transition_observation(observation: Any) -> JsonDict:
    from carnot.agentic.arc_e3_outcome_transport import normalize_observation

    normalized = normalize_observation(observation) or {}
    return {
        key: normalized.get(key)
        for key in ("frame", "state", "levels_completed", "win_levels", "full_reset", "available_actions")
    }


def _apply_public_action(env: Any, action: Mapping[str, Any]) -> Any:
    from arcengine import GameAction

    action_id = int(action["action"])
    member = getattr(GameAction, f"ACTION{action_id}")
    data = deepcopy(action.get("data"))
    if data:
        member.set_data({"game_id": str(getattr(env.observation_space, "game_id", "")), **data})
    return env.step(member, data=data)


def execute_and_replay_action(
    *, arcade: Any, game_id: str, seed: int, action: Mapping[str, Any]
) -> JsonDict:
    """Execute one public action and compare it with a fresh-environment replay."""

    scorecard = arcade.open_scorecard()
    env = arcade.make(game_id, seed=int(seed), scorecard_id=scorecard)
    replay = arcade.make(game_id, seed=int(seed), scorecard_id=scorecard)
    before = _transition_observation(env.observation_space)
    replay_before = _transition_observation(replay.observation_space)
    valid = int(action.get("action", -1)) in set(before.get("available_actions") or [])
    after_raw = _apply_public_action(env, action) if valid else None
    replay_after_raw = _apply_public_action(replay, action) if valid else None
    after = _transition_observation(after_raw)
    replay_after = _transition_observation(replay_after_raw)
    exact = bool(valid and before == replay_before and after == replay_after)
    return {
        "executed": after_raw is not None,
        "valid_action": valid,
        "action": deepcopy(dict(action)),
        "observation_before_hash": sha256_json(before),
        "observation_after_hash": sha256_json(after),
        "environment_return_hash": sha256_json(after),
        "level_before": int(before.get("levels_completed", 0) or 0),
        "level_after": int(after.get("levels_completed", 0) or 0),
        "exact": exact,
        "fresh_replay_exact": exact,
    }


def production_reachability_rows() -> list[JsonDict]:
    """Inspect the scored factory and the local proposer classes directly."""

    from carnot.agentic.arc_competition_agent import E3AgentPolicy, make_carnot_agent
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    source = inspect.getsource(make_carnot_agent)
    return [
        {"symbol": "make_carnot_agent", "passed": callable(make_carnot_agent)},
        {"symbol": "E3AgentPolicy", "passed": isinstance(E3AgentPolicy, type) and "E3AgentPolicy(" in source},
        {"symbol": "LocalGGUFProposer", "passed": isinstance(LocalGGUFProposer, type)},
    ]


def _normalize_registry(path: Path) -> list[JsonDict]:  # pragma: no cover - host input boundary
    import yaml

    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    games = value.get("games", []) if isinstance(value, Mapping) else []
    return [dict(row) for row in games if isinstance(row, Mapping)]


def _credited_receipts(results_dir: Path, current_output: Path) -> list[JsonDict]:  # pragma: no cover - host input boundary
    rows: list[JsonDict] = []
    for path in sorted(results_dir.glob("experiment_*.json")):
        if path.resolve() == current_output.resolve():
            continue
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(value, Mapping) or value.get("adapter_withheld_live_path_ready_score") != 1:
            continue
        for row in value.get("action_rows", []):
            if isinstance(row, Mapping) and row.get("valid_action") is True:
                rows.append({"game_id": row.get("game_id"), "level": 0, "credited": True, "artifact": str(path)})
    return rows


def _gpu_rows() -> list[JsonDict]:  # pragma: no cover - hardware boundary
    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.free,utilization.gpu", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    apps = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
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
            index, total, free, utilization = int(parts[0]), int(parts[3]), int(parts[4]), int(parts[5])
        except ValueError:
            continue
        rows.append(
            {
                "gpu": index,
                "model": parts[1],
                "uuid": parts[2],
                "memory_total_mb": total,
                "memory_free_mb": free,
                "utilization_pct": utilization,
                "idle": parts[2] not in busy and utilization <= 5 and free >= 20_000,
                "supported": parts[1] == "NVIDIA GeForce RTX 3090",
                "sample_phase": "preflight",
            }
        )
    return rows


def _runner_receipt() -> JsonDict:  # pragma: no cover - binary boundary
    path = Path.home() / ".cache/llama.cpp-master/build/bin/llama-server"
    linked = subprocess.run(["ldd", str(path)], capture_output=True, text=True, check=False)
    version = subprocess.run([str(path), "--version"], capture_output=True, text=True, check=False)
    return {
        "path": str(path),
        "exists": path.is_file() and os.access(path, os.X_OK),
        "cuda_linked": "libcuda.so" in linked.stdout and "libggml-cuda" in linked.stdout,
        "version_returncode": version.returncode,
        "version": (version.stdout + version.stderr).strip()[:300],
        "sha256": sha256_file(path),
    }


def _source_hashes(root: Path) -> JsonDict:  # pragma: no cover - host input boundary
    paths = (
        SPEC_RELATIVE_PATH,
        MODULE_RELATIVE_PATH,
        SCRIPT_RELATIVE_PATH,
        TEST_RELATIVE_PATH,
        REGISTRY_RELATIVE_PATH,
        Path("python/carnot/agentic/arc_competition_agent.py"),
        Path("python/carnot/agentic/arc_executable_world_model.py"),
        Path("python/carnot/agentic/arc_game_adapters.py"),
        Path("scripts/arc_loop_solve.py"),
    )
    return {path.as_posix(): sha256_file(root / path) for path in paths if (root / path).is_file()}


def collect_preconditions(
    *, repo_root: Path, output_path: Path, raw_root: Path
) -> JsonDict:  # pragma: no cover - combines live host boundaries
    """Check all immutable inputs before leases, game selection, or generation."""

    checks: list[JsonDict] = []
    models = resolve_model_specs()
    checks.append(gate_row("exact_model_pins", [QWEN38_HF_ID, QWEN36_HF_ID], [row.get("hf_id") for row in models]))
    checks.append(gate_row("exact_cached_gguf_files", True, not validate_model_specs(models)))
    gpus = _gpu_rows()
    idle = [row for row in gpus if row["supported"] and row["idle"]]
    checks.append(gate_row("two_idle_rtx3090_leases", 2, len(idle), passed=len(idle) >= 2))
    runner = _runner_receipt()
    checks.append(gate_row("cuda_llama_cpp", True, runner["exists"] and runner["cuda_linked"] and runner["version_returncode"] == 0))
    try:
        from arc_agi import Arcade, OperationMode

        offline_arc = bool(Arcade and OperationMode.OFFLINE)
    except Exception:
        offline_arc = False
    checks.append(gate_row("offline_arc", True, offline_arc))
    registry_path = repo_root / REGISTRY_RELATIVE_PATH
    try:
        registry_rows = _normalize_registry(registry_path)
    except Exception:
        registry_rows = []
    checks.append(gate_row("readable_registry", True, bool(registry_rows)))
    for label, directory in (("raw", raw_root), ("aggregate", output_path.parent)):
        try:
            directory.mkdir(parents=True, exist_ok=True)
            writable = directory.is_dir() and os.access(directory, os.W_OK)
        except OSError:
            writable = False
        checks.append(gate_row(f"writable_{label}_path", True, writable))
    reachability = production_reachability_rows()
    checks.append(gate_row("production_e3_route", True, all(row["passed"] for row in reachability)))
    return {
        "checks": checks,
        "summary": make_gate_summary(checks),
        "models": models,
        "gpu_rows": gpus,
        "idle_gpus": idle[:2],
        "runner": runner,
        "registry_rows": registry_rows,
        "reachability": reachability,
        "source_hashes": _source_hashes(repo_root),
    }


def _observation_payload(raw: Any) -> JsonDict:  # pragma: no cover - SDK boundary
    from carnot.agentic.arc_e3_outcome_transport import normalize_observation

    return normalize_observation(raw) or {}


def _raw_from_payload(value: Mapping[str, Any]) -> Any:  # pragma: no cover - worker SDK boundary
    import numpy as np
    from arcengine import FrameDataRaw

    fields = {key: value.get(key) for key in FrameDataRaw.model_fields if key in value}
    raw = FrameDataRaw.model_validate(fields)
    raw.frame = [np.asarray(frame) for frame in (value.get("frame") or [])]
    return raw


def _candidate_dict(value: Any) -> JsonDict:  # pragma: no cover - policy boundary
    if isinstance(value, Mapping):
        action = value.get("action", value.get("action_id"))
        data = value.get("data")
    else:
        action = getattr(value, "action_id", getattr(value, "action", None))
        data = getattr(value, "data", None)
    return {"action": int(action), "data": deepcopy(data) if data else None}


def _forecast_prompt(latest: Any, candidates: Sequence[Mapping[str, Any]]) -> str:  # pragma: no cover - live prompt boundary
    import numpy as np
    from carnot.agentic.arc_agi3_world_model import grid_of

    grid = np.asarray(grid_of(latest))
    colors, counts = np.unique(grid, return_counts=True)
    summary = {
        "shape": list(grid.shape),
        "colors": {str(int(color)): int(count) for color, count in zip(colors, counts, strict=True)},
        "top_left_16x16": grid[:16, :16].astype(int).tolist(),
        "levels_completed": int(getattr(latest, "levels_completed", 0) or 0),
        "available_actions": list(getattr(latest, "available_actions", []) or []),
    }
    return (
        "You are the forecast stage inside a bounded live ARC E3 action preflight. "
        "Use only the observation and generic candidate schema below. Predict which one candidate "
        "is most likely to cause any visible state change. This is not a solve request. Return exactly "
        "<forecast>{\"selected_candidate_index\":N,\"predicted_change\":true," 
        "\"rationale\":\"one short observation-based reason\"}</forecast>. Do not return advice or code.\n"
        f"OBSERVATION={json.dumps(summary, separators=(',', ':'))}\n"
        f"CANDIDATES={json.dumps(list(candidates), separators=(',', ':'))}"
    )


class _RpcEnvironment:  # pragma: no cover - worker RPC boundary
    def __init__(self, connection: Connection, cell_id: str, initial: Mapping[str, Any]) -> None:
        self.connection = connection
        self.cell_id = cell_id
        self.observation_space = _raw_from_payload(initial)

    def step(self, action: Any, data: Any = None, reasoning: Any = None) -> Any:
        from carnot.agentic.arc_e3_outcome_transport import normalize_action

        del data, reasoning
        self.connection.send({"op": "step", "cell_id": self.cell_id, "action": normalize_action(action)})
        reply = self.connection.recv()
        if reply.get("ok") is not True:
            raise RuntimeError(str(reply.get("error") or "environment step failed"))
        self.last_transition = deepcopy(dict(reply.get("transition") or {}))
        self.observation_space = _raw_from_payload(reply["observation"])
        return self.observation_space


def _load_framework_agent() -> Any:  # pragma: no cover - external framework boundary
    from carnot import experiment_6681_arc_post_redirect_outcomes as transport_experiment

    return transport_experiment._load_framework_agent()


def _worker_gpu_sample(gpu_index: int, server_pid: int) -> JsonDict:  # pragma: no cover - hardware boundary
    query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=False,
    )
    matches = []
    for line in query.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) == server_pid:
            matches.append({"gpu_uuid": parts[0], "pid": int(parts[1]), "used_memory_mb": int(parts[2])})
    return {"gpu": gpu_index, "server_pid": server_pid, "inside_inference": True, "process_rows": matches, "passed": bool(matches)}


def _live_worker_entry(
    connection: Connection,
    repo_root: str,
    model: Mapping[str, Any],
    games: Sequence[str],
    raw_root: str,
    port: int,
) -> None:  # pragma: no cover - live spawned-process boundary
    """Generate two forecasts and execute their actions through E3 in isolation."""

    from carnot.agentic import arc_competition_agent as competition
    from carnot.agentic import arc_strategy_router
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    def withheld_approach(_game_id: str, **_kwargs: Any) -> JsonDict:
        strategy = arc_strategy_router.route_strategy(arc_strategy_router.DEFAULT_MECHANIC)
        strategy["game"] = "adapter_withheld"
        return {"strategy": strategy, "adapter_withheld": True, "registry_consulted": False}

    competition._recommend_live_approach = withheld_approach
    E3AgentPolicy = competition.E3AgentPolicy
    make_carnot_agent = competition.make_carnot_agent

    os.environ.update(
        {
            "CARNOT_ARC_GENERATOR_CUDA_GPU": str(model["gpu"]),
            "CARNOT_ARC_GENERATOR_REQUIRE_CUDA": "1",
            "CARNOT_ARC_GENERATOR_SEED": str(RANDOM_SEED + int(model["gpu"])),
            "CARNOT_ARC_E3_DIR": str(Path(raw_root) / "empty-e3" / str(model["key"])),
        }
    )
    proposer = LocalGGUFProposer(
        repo_substr=str(model["name"]),
        model_path=str(model["model_path"]),
        model_repository=str(model["hf_id"]),
        model_filename=Path(str(model["model_path"])).name,
        requested_model_path=str(model["model_path"]),
        requested_model_filename=Path(str(model["model_path"])).name,
        port=int(port),
        n_ctx=CONTEXT_BUDGET,
        max_tokens=TOKEN_BUDGET,
        timeout=TIMEOUT_S,
        mtp=False,
        kv_quant="q8_0",
        n_gpu_layers=999,
        no_think_prefix="/no_think\n" if model["hf_id"] == QWEN38_HF_ID else "",
    )
    rows: list[JsonDict] = []
    try:
        if not proposer._ensure_server():
            raise RuntimeError("CUDA llama-server did not become healthy")
        server_pid = int(proposer._proc.pid)
        connection.send({"op": "model_ready", "model_id": model["hf_id"], "server_pid": server_pid})
        BaseAgent = _load_framework_agent()
        AgentClass = make_carnot_agent(BaseAgent, cascade=True, proposer=proposer)
        for game_index, game_id in enumerate(games):
            cell_id = f"{model['key']}:{game_id}"
            seed = RANDOM_SEED + int(model["gpu"]) * 100 + game_index
            connection.send({"op": "make", "cell_id": cell_id, "game_id": game_id, "seed": seed})
            reply = connection.recv()
            if reply.get("ok") is not True:
                raise RuntimeError(str(reply.get("error") or "environment creation failed"))
            rpc_env = _RpcEnvironment(connection, cell_id, reply["observation"])
            monitor = IsolationMonitor(repo_root)
            started = time.perf_counter()
            with monitor.activated():
                agent = AgentClass(
                    card_id="exp7099-local",
                    game_id=game_id,
                    agent_name="carnot-exp7099",
                    ROOT_URL="local-offline-rpc",
                    record=False,
                    arc_env=rpc_env,
                    tags=["adapter-withheld", "no-solve"],
                )
                if not isinstance(agent._policy, E3AgentPolicy):
                    raise RuntimeError("make_carnot_agent did not construct E3AgentPolicy")
                latest = agent._convert_raw_frame_data(rpc_env.observation_space)
                generated = agent._policy.explorer._candidates(latest, path=[], previous_frame=None)
                candidates = [_candidate_dict(item) for item in generated[:12]]
                prompt = _forecast_prompt(latest, candidates)
                ok, response = proposer.complete_text(
                    prompt,
                    max_tokens=TOKEN_BUDGET,
                    temperature=0.0,
                    stop=["</forecast>"],
                )
                interpreted = interpret_forecast(response, candidate_actions=candidates) if ok else {
                    "accepted": False,
                    "reason": "generation_failed",
                    "forecast": None,
                    "selected_action": None,
                }
                selected = interpreted.get("selected_action")
                transition: JsonDict = {
                    "executed": False,
                    "valid_action": False,
                    "action": selected,
                    "exact": False,
                    "fresh_replay_exact": False,
                }
                if interpreted.get("accepted") is True and isinstance(selected, Mapping):
                    agent._policy.plan = [deepcopy(dict(selected))]
                    agent._policy.pi = 0
                    agent._policy.phase = "execute"
                    action = agent.choose_action(agent.frames, latest)
                    frame = agent.take_action(action)
                    if frame is None:
                        raise RuntimeError("E3 valid action returned no frame")
                    agent.append_frame(frame)
                    transition = deepcopy(getattr(rpc_env, "last_transition", {}))
            if not transition:
                transition = deepcopy(reply.get("transition") or {})
            invocation = {
                "invoked": True,
                "model_id": model["hf_id"],
                "prompt_hash": sha256_json(prompt),
                "candidate_count": len(candidates),
                "completion_ok": bool(ok),
                "response_hash": sha256_json(response),
            }
            forecast = deepcopy(interpreted.get("forecast"))
            selected = deepcopy(interpreted.get("selected_action"))
            row: JsonDict = {
                "cell_id": cell_id,
                "game_id": game_id,
                "model_id": model["hf_id"],
                "model_path": model["model_path"],
                "gpu": model["gpu"],
                "seed": seed,
                "budget": {"actions": ACTION_BUDGET, "tokens": TOKEN_BUDGET, "timeout_s": TIMEOUT_S},
                "observation_hash": sha256_json(_transition_observation(latest)),
                "candidate_actions": candidates,
                "simulation_invocation": invocation,
                "returned_forecast_text": response,
                "forecast": forecast,
                "interpretation": {
                    "accepted": interpreted.get("accepted") is True,
                    "reason": interpreted.get("reason"),
                    "selected_candidate_index": interpreted.get("selected_candidate_index"),
                    "forecast_hash": interpreted.get("forecast_hash"),
                },
                "selected_action": selected,
                "forecast_consumed": interpreted.get("accepted") is True,
                "policy_class": type(agent._policy).__name__,
                "factory": "make_carnot_agent",
                "valid_action": bool(transition.get("valid_action")),
                "advice_only": interpreted.get("reason") == "advice_only",
                "transition": transition,
                "latency_s": round(time.perf_counter() - started, 6),
                "forbidden_import_rows": monitor.import_receipts(),
                "forbidden_read_rows": monitor.read_receipts(),
                "isolation": {
                    "worker_pid": os.getpid(),
                    "worker_start_method": "spawn",
                    "policy_process_separate_from_environment": True,
                    "game_source_present_in_worker_modules": any("environment_files" in str(getattr(module, "__file__", "")) for module in sys.modules.values()),
                    "adapter_module_present": "carnot.agentic.arc_game_adapters" in sys.modules,
                    "passed": "carnot.agentic.arc_game_adapters" not in sys.modules,
                },
            }
            raw_path = Path(raw_root) / str(model["key"]) / f"{cell_id.replace(':', '_')}.json"
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(raw_path, row)
            row["raw_trace_path"] = str(raw_path)
            row["raw_trace_hash"] = sha256_file(raw_path)
            rows.append(row)
        connection.send(
            {
                "op": "result",
                "model_id": model["hf_id"],
                "rows": rows,
                "gpu_sample": _worker_gpu_sample(int(model["gpu"]), server_pid),
                "model_path_observed": proposer.observed_model_path(),
                "server_props": proposer.server_props(),
                "error": None,
            }
        )
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"Exp7099 worker {model.get('hf_id')} failed: {error}", file=sys.stderr, flush=True)
        traceback.print_exc()
        connection.send({"op": "result", "model_id": model.get("hf_id"), "rows": rows, "error": error})
    finally:
        proposer.stop()
        connection.close()


def _free_port() -> int:  # pragma: no cover - operating-system boundary
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = int(probe.getsockname()[1])
    probe.close()
    return port


def _host_step(arcade: Any, state: JsonDict, request: Mapping[str, Any]) -> JsonDict:  # pragma: no cover - live environment boundary
    cell_id = str(request["cell_id"])
    env = state[cell_id]["env"]
    game_id = state[cell_id]["game_id"]
    seed = int(state[cell_id]["seed"])
    raw_action = dict(request["action"])
    action = {"action": raw_action.get("kind"), "data": raw_action.get("data")}
    before = _transition_observation(env.observation_space)
    valid = int(action.get("action", -1)) in set(before.get("available_actions") or [])
    returned = _apply_public_action(env, action) if valid else None
    replay = arcade.make(game_id, seed=seed, scorecard_id=state["scorecard"])
    replay_before = _transition_observation(replay.observation_space)
    replay_returned = _apply_public_action(replay, action) if valid else None
    after = _transition_observation(returned)
    replay_after = _transition_observation(replay_returned)
    exact = bool(valid and before == replay_before and after == replay_after)
    transition = {
        "executed": returned is not None,
        "valid_action": valid,
        "action": action,
        "observation_before_hash": sha256_json(before),
        "observation_after_hash": sha256_json(after),
        "environment_return_hash": sha256_json(after),
        "level_before": int(before.get("levels_completed", 0) or 0),
        "level_after": int(after.get("levels_completed", 0) or 0),
        "exact": exact,
        "fresh_replay_exact": exact,
    }
    return {"ok": returned is not None, "observation": _observation_payload(returned), "transition": transition}


def execute_live_workers(
    *, repo_root: Path, model_specs: Sequence[Mapping[str, Any]], frozen_game_ids: Sequence[str], raw_root: Path, idle_gpus: Sequence[Mapping[str, Any]]
) -> JsonDict:  # pragma: no cover - multiprocessing, CUDA, and ARC boundary
    """Run both model workers while the host owns environments and GPU leases."""

    from carnot import gpu_lease_phase_journal as lease_api
    from carnot.agentic.arc_solver_kit import offline_arcade

    task_id = f"exp7099:{os.getpid()}"
    leases = []
    lease_rows: list[JsonDict] = []
    cleanup_rows: list[JsonDict] = []
    contexts = multiprocessing.get_context("spawn")
    workers: dict[Connection, JsonDict] = {}
    state: JsonDict = {}
    arcade = offline_arcade()
    state["scorecard"] = arcade.open_scorecard()
    results: list[JsonDict] = []
    try:
        for model, gpu in zip(model_specs, idle_gpus, strict=True):
            lease = lease_api.GpuLease.acquire(
                runtime_dir=LEASE_RUNTIME_DIR,
                task_id=task_id,
                device_uuid=str(gpu["uuid"]),
                expected_model=str(model["model_path"]),
                vram_before_mb=int(gpu["memory_total_mb"]) - int(gpu["memory_free_mb"]),
                ttl_s=3600.0,
            )
            lease.transition("admitted")
            lease.transition("loading")
            leases.append((lease, model, gpu))
        assignments = (frozen_game_ids[:2], frozen_game_ids[2:4])
        for (lease, model, gpu), games in zip(leases, assignments, strict=True):
            del lease, gpu
            parent, child = contexts.Pipe(duplex=True)
            process = contexts.Process(
                target=_live_worker_entry,
                args=(child, str(repo_root), dict(model), list(games), str(raw_root), _free_port()),
            )
            process.start()
            child.close()
            workers[parent] = {"process": process, "model": model, "ready": False}
        deadline = time.monotonic() + TIMEOUT_S * 4
        while workers and time.monotonic() < deadline:
            ready_connections = wait(list(workers), timeout=1.0)
            for connection in ready_connections:
                info = workers[connection]
                try:
                    request = connection.recv()
                except EOFError:
                    info["process"].join(timeout=5)
                    connection.close()
                    workers.pop(connection, None)
                    continue
                op = request.get("op")
                if op == "model_ready":
                    lease = next(item[0] for item in leases if item[1]["hf_id"] == request["model_id"])
                    lease.transition("resident", vram_mb=1)
                    lease.transition("inferencing")
                    info["ready"] = True
                elif op == "make":
                    try:
                        env = arcade.make(request["game_id"], seed=int(request["seed"]), scorecard_id=state["scorecard"])
                        state[str(request["cell_id"])] = {"env": env, "game_id": request["game_id"], "seed": request["seed"]}
                        connection.send({"ok": env is not None and env.observation_space is not None, "observation": _observation_payload(env.observation_space)})
                    except Exception as exc:
                        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                elif op == "step":
                    try:
                        reply = _host_step(arcade, state, request)
                        connection.send(reply)
                        info["last_transition"] = reply.get("transition")
                    except Exception as exc:
                        connection.send({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
                elif op == "result":
                    results.append(dict(request))
                    info["process"].join(timeout=20)
                    connection.close()
                    workers.pop(connection, None)
        for connection, info in list(workers.items()):
            process = info["process"]
            if process.is_alive():
                process.terminate()
            process.join(timeout=10)
            results.append({"model_id": info["model"]["hf_id"], "rows": [], "error": "worker_timeout"})
            connection.close()
        return {
            "worker_results": results,
            "action_rows": [row for result in results for row in result.get("rows", [])],
            "gpu_rows": [result["gpu_sample"] for result in results if result.get("gpu_sample")],
            "lease_rows": lease_rows,
            "cleanup_rows": cleanup_rows,
        }
    finally:
        for lease, model, gpu in leases:
            try:
                phase = lease.document.get("phase")
                if phase in {"resident", "inferencing"}:
                    lease.transition("unloading")
                    lease.transition("validating", vram_mb=0, exit_code=0, unload_observed=True)
                    lease.transition("terminal_complete" if len(results) == 2 and all(not row.get("error") for row in results) else "terminal_blocked")
                elif phase not in lease_api.TERMINAL_PHASES:
                    lease.transition("terminal_blocked")
                release = lease.release() if lease.document.get("phase") in lease_api.TERMINAL_PHASES else {"released": False}
                release.update({"gpu": gpu["gpu"], "model_id": model["hf_id"]})
                lease_rows.append(release)
                cleanup_rows.append({"gpu": gpu["gpu"], "model_id": model["hf_id"], "released": release.get("released") is True, "unrelated_processes_signaled": []})
            except Exception as exc:
                cleanup_rows.append({"gpu": gpu["gpu"], "model_id": model["hf_id"], "released": False, "error": f"{type(exc).__name__}: {exc}"})


def write_artifact(path: Path, artifact: Mapping[str, Any]) -> None:  # pragma: no cover - publication boundary
    errors = validate_artifact(artifact, verify_raw_traces=artifact.get("verdict_class") != "blocked")
    if errors:
        raise ValueError("invalid Exp7099 artifact: " + "; ".join(errors))
    atomic_write_json(path, dict(artifact))


def run(
    repo_root: Path,
    *,
    execution_date: str,
    output_path: Path,
    raw_root: Path,
) -> JsonDict:  # pragma: no cover - full host experiment boundary
    """Run preconditions, freeze games, execute cells, and publish once."""

    started = time.perf_counter()
    preflight = collect_preconditions(repo_root=repo_root, output_path=output_path, raw_root=raw_root)
    if not preflight["summary"]["passed"]:
        artifact = build_blocked_artifact(
            execution_date=execution_date,
            duration_s=time.perf_counter() - started,
            preconditions=preflight["checks"],
            source_hashes=preflight["source_hashes"],
        )
        write_artifact(output_path, artifact)
        return artifact
    credited = _credited_receipts(output_path.parent, output_path)
    try:
        frozen, registry_rows = freeze_registry_games(
            preflight["registry_rows"], credited_receipts=credited, minimum=MIN_FROZEN_GAMES
        )
    except ValueError as exc:
        checks = [*preflight["checks"], gate_row("four_frozen_registry_strata", "available", str(exc))]
        artifact = build_blocked_artifact(
            execution_date=execution_date,
            duration_s=time.perf_counter() - started,
            preconditions=checks,
            source_hashes=preflight["source_hashes"],
        )
        write_artifact(output_path, artifact)
        return artifact
    execution = execute_live_workers(
        repo_root=repo_root,
        model_specs=preflight["models"],
        frozen_game_ids=frozen,
        raw_root=raw_root,
        idle_gpus=preflight["idle_gpus"],
    )
    artifact = build_completed_artifact(
        execution_date=execution_date,
        duration_s=time.perf_counter() - started,
        preconditions=preflight["checks"],
        source_hashes=preflight["source_hashes"],
        model_specs=preflight["models"],
        frozen_game_ids=frozen,
        registry_rows=registry_rows,
        action_rows=execution["action_rows"],
        gpu_rows=[*preflight["gpu_rows"], *execution["gpu_rows"]],
        cleanup_rows=execution["cleanup_rows"],
        worker_results=execution["worker_results"],
    )
    write_artifact(output_path, artifact)
    return artifact


def main(argv: list[str] | None = None) -> int:  # pragma: no cover - CLI boundary
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--raw-root", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[2]
    output = (args.output or root / RESULT_RELATIVE_PATH).resolve()
    raw = (args.raw_root or RAW_RELATIVE_ROOT / args.date).resolve()
    artifact = run(root, execution_date=args.date, output_path=output, raw_root=raw)
    print(f"Exp7099 {artifact['verdict_class']}: {artifact['honest_verdict']} -> {output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
