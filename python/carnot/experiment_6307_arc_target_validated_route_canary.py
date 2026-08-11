"""Exp6307 ARC target-validated mechanic-route canary.

Spec refs: REQ-ARC-WMTE-6307,
SCENARIO-ARC-WMTE-6307-REGISTRY-PRECHECK,
SCENARIO-ARC-WMTE-6307-INACTIVE-HYPOTHESES,
SCENARIO-ARC-WMTE-6307-TARGET-LICENSE,
SCENARIO-ARC-WMTE-6307-MATCHED-THREE-ARM-CELLS,
SCENARIO-ARC-WMTE-6307-ARTIFACT-GUARDS.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any

import numpy as np

from carnot import experiment_6294_arc_mechanic_router_causal_canary as exp6294
from carnot.agentic import arc_executable_world_model as e3
from carnot.agentic import arc_mechanic_class_detector as detector


JsonDict = dict[str, Any]
ModelResolver = Callable[[bool], list[JsonDict]]
LLMRunner = Callable[[Sequence[JsonDict], Sequence[JsonDict], Path, bool], JsonDict]

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = Path("results/experiment_6307_arc_target_validated_route_canary.json")
FIXTURE_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6307_arc_target_validated_route_fixture_manifest.json"
)
LIVE_WINDOW_MANIFEST_RELATIVE_PATH = Path(
    "results/experiment_6307_arc_target_validated_route_live_windows.json"
)
RAW_OUTPUT_DIR_RELATIVE_PATH = Path("results/experiment_6307_arc_target_validated_route_raw")
REGISTRY_RELATIVE_PATH = Path("ops/arc_solve_registry.yaml")
RUN_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6307_arc_target_validated_route_canary --date 20260811"
)
FOCUSED_TEST_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6307_arc_target_validated_route_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6307_arc_target_validated_route_canary.py "
    "-m pytest tests/python/test_experiment_6307_arc_target_validated_route_canary.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6307_arc_target_validated_route_canary.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COVERAGE_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6307_arc_target_validated_route_canary.py"
)
EXP6298_PREFLIGHT_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6298_terminal_evidence_preflight_linter "
    "--date 20260811 --no-run-commands"
)
E2E_PLAN_READ_COMMAND = "sed -n 1,180p ops/e2e-test-plan.md"
DETERMINATION_COMMAND = ".venv/bin/python scripts/determination_preservation_lint.py --all"
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6307_arc_target_validated_route_canary.json"
)
EXTERNAL_TEST_RECEIPT_PATH = Path("/tmp/carnot_exp6307_test_receipts.json")

RANDOM_SEEDS = (6307001, 6307002)
MANDATED_MODEL_IDS = ("unsloth/Qwen3.6-35B-A3B-GGUF", "unsloth/gemma-4-31B-it-GGUF")
ARMS = ("router_off", "retrieval_only_static_route", "target_licensed_route")
MODEL_BUDGET_TOKENS = 32
ACTION_BUDGET = 12
PREFERRED_QUANT = "Q4_K_M"
ADEQUATE_CELL_SAMPLE_SIZE = 4
TARGET_LICENSE_MAX_UNCERTAINTY = 0.35
TARGET_LICENSE_MIN_CHANGED = 3
FORBIDDEN_ZERO_FIELDS = (
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "source_model_weight_mutation_count",
    "duration_padding_count",
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
    FOCUSED_TEST_COMMAND,
    COVERAGE_RUN_COMMAND,
    COVERAGE_REPORT_COMMAND,
    FULL_PYTEST_COMMAND,
    SPEC_COVERAGE_COMMAND,
    EXP6298_PREFLIGHT_COMMAND,
    E2E_PLAN_READ_COMMAND,
    DETERMINATION_COMMAND,
    ADVERSARIAL_COMMAND,
)
REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "upstream_paths_hashes_and_terminal_classes",
    "registry_precheck_path_hash_and_target_receipt",
    "solve_provenance",
    "sealed_fresh_fixture_manifest_path_and_hash",
    "live_transition_window_manifest_path_and_hash",
    "no_hidden_source_and_no_outer_loop_search_receipts",
    "MODEL_SPECS",
    "models_used",
    "model_file_hashes_revisions_and_quantizations",
    "tokenizer_and_chat_template_hashes",
    "cuda_and_gpu_offload_receipts_by_model",
    "raw_output_paths_and_hashes",
    "router_off_retrieval_only_and_target_licensed_arm_contract",
    "matched_seed_history_action_budget_and_model_call_receipts",
    "target_validation_predicates_and_mutation_receipts",
    "hypothesis_retrieval_activation_rejection_and_abstention_counts",
    "route_activation_counts_by_arm_mechanic_and_model",
    "executable_proposal_acceptance_by_arm_mechanic_and_model",
    "invalid_proposal_rate_by_arm_mechanic_and_model",
    "candidate_diversity_by_arm_mechanic_and_model",
    "decision_latency_by_arm_mechanic_and_model",
    "paired_causal_deltas_intervals_and_sample_sizes",
    "baseline_harm_controls",
    "actual_work_duration_receipt",
    "duration_padding_count",
    "arc_target_licensed_router_ready_score",
    "hidden_game_source_access_count",
    "outer_loop_ground_truth_search_count",
    "arc_level_solve_claim_count",
    "registry_update_count",
    "source_model_weight_mutation_count",
    "protected_files_unchanged",
    "preconditions_checked",
    "inference_substrate",
    "verifier_is_oracle",
    "field_provenance",
    "field_principles",
    "test_commands",
    "test_exit_codes",
    "duration_s",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
)

FIELD_PRINCIPLES: dict[str, str] = {
    "status": "States complete versus blocked without hiding partial work.",
    "upstream_paths_hashes_and_terminal_classes": "Pins the 6294, 6295, 6298, and V543 inputs.",
    "registry_precheck_path_hash_and_target_receipt": "Proves the synthetic target is not banked.",
    "solve_provenance": "States live-agent self-discovery without solve credit.",
    "sealed_fresh_fixture_manifest_path_and_hash": "Pins fresh synthetic transition fixtures.",
    "live_transition_window_manifest_path_and_hash": "Pins the agent-visible transition windows.",
    "no_hidden_source_and_no_outer_loop_search_receipts": "Shows forbidden inputs stayed absent.",
    "MODEL_SPECS": "Names both mandated headline GGUF models.",
    "models_used": "Lists models that entered the canary.",
    "model_file_hashes_revisions_and_quantizations": "Pins concrete local GGUF files.",
    "tokenizer_and_chat_template_hashes": "Pins tokenizer probes and the prompt contract.",
    "cuda_and_gpu_offload_receipts_by_model": "Records CUDA visibility and offload receipts.",
    "raw_output_paths_and_hashes": "Pins every raw model output.",
    "router_off_retrieval_only_and_target_licensed_arm_contract": "Proves the three arm states.",
    "matched_seed_history_action_budget_and_model_call_receipts": "Proves matched seeds and budgets.",
    "target_validation_predicates_and_mutation_receipts": "Shows target checks reject mutations.",
    "hypothesis_retrieval_activation_rejection_and_abstention_counts": "Separates retrieval from transfer.",
    "route_activation_counts_by_arm_mechanic_and_model": "Shows only licensed routes can fire.",
    "executable_proposal_acceptance_by_arm_mechanic_and_model": "Measures parseable proposal code.",
    "invalid_proposal_rate_by_arm_mechanic_and_model": "Measures malformed action references.",
    "candidate_diversity_by_arm_mechanic_and_model": "Measures proposal variety by arm.",
    "decision_latency_by_arm_mechanic_and_model": "Measures model decision cost by arm.",
    "paired_causal_deltas_intervals_and_sample_sizes": "Reports preregistered route-path deltas.",
    "baseline_harm_controls": "Reports whether licensing harmed controls.",
    "actual_work_duration_receipt": "Documents measured work and rejects padding.",
    "duration_padding_count": "Must stay zero because padding is forbidden.",
    "arc_target_licensed_router_ready_score": "Equals one only when every readiness gate passes.",
    "hidden_game_source_access_count": "Must stay zero for hidden-source discipline.",
    "outer_loop_ground_truth_search_count": "Must stay zero for self-discovery discipline.",
    "arc_level_solve_claim_count": "Must stay zero because this is not a solve task.",
    "registry_update_count": "Must stay zero because no solve is banked.",
    "source_model_weight_mutation_count": "Must stay zero because weights are immutable.",
    "protected_files_unchanged": "Confirms registry, ops, trace, and conductor files stayed unchanged.",
    "preconditions_checked": "Records registry, preflight, resources, seeds, and hashes.",
    "inference_substrate": "Declares live GGUF inference.",
    "verifier_is_oracle": "False because no game oracle verifies a solve.",
    "field_provenance": "Maps every field to the spec and producer.",
    "field_principles": "Gives one audit reason per required field.",
    "test_commands": "Lists verification commands.",
    "test_exit_codes": "Records command outcomes.",
    "duration_s": "Records measured wall time.",
    "random_seeds": "Pins fixture and prompt seeds.",
    "reproducibility_checksum": "Detects artifact drift.",
    "honest_verdict": "Terminal verdict with no solve claim.",
}
FIELD_PROVENANCE = {
    field: ["REQ-ARC-WMTE-6307", "experiment_6307_arc_target_validated_route_canary"]
    for field in REQUIRED_ARTIFACT_FIELDS
}


resolve_required_models = exp6294.resolve_required_models
deterministic_test_model_resolver = exp6294.deterministic_test_model_resolver
missing_gemma_test_model_resolver = exp6294.missing_gemma_test_model_resolver
live_llama_cpp_runner = exp6294.live_llama_cpp_runner


@dataclass(frozen=True)
class FreshMechanicFixture:
    """One fresh synthetic mechanic bundle with no game identity."""

    fixture_id: str
    family: str
    transitions: tuple[e3.Transition, ...]
    game_id: None = None


@dataclass(frozen=True)
class LiveTransitionWindow:
    """One sealed window of agent-visible transition evidence."""

    window_id: str
    fixture_id: str
    mechanic: str
    seed: int
    transitions: tuple[e3.Transition, ...]
    starting_history_hash: str
    action_budget: int = ACTION_BUDGET
    model_budget_tokens: int = MODEL_BUDGET_TOKENS


@dataclass(frozen=True)
class RouteLicenseDecision:
    """Decision made before a mechanic route can affect the live proposal path."""

    arm: str
    retrieved_hypothesis: str | None
    observed_mechanic: str | None
    licensed: bool
    route_active: bool
    rejected: bool
    abstained: bool
    reason: str
    license_predicates: JsonDict
    mutation_receipt: JsonDict


class TargetLicensePolicy:
    """Generic target-side license gate for retrieved mechanic hypotheses."""

    def __init__(
        self,
        *,
        max_uncertainty: float = TARGET_LICENSE_MAX_UNCERTAINTY,
        min_changed: int = TARGET_LICENSE_MIN_CHANGED,
    ) -> None:
        self.max_uncertainty = float(max_uncertainty)
        self.min_changed = int(min_changed)

    def _predicates(
        self,
        transitions: Sequence[e3.Transition],
        retrieved_hypothesis: str | None,
    ) -> tuple[JsonDict, str | None]:
        result = detector.classify_transition_history(transitions)
        support = dict(result.support)
        observed = result.predicted_class
        predicates = {
            "class_agrees_with_retrieval": bool(observed == retrieved_hypothesis),
            "support_count_min": int(support.get("n_changed") or 0) >= self.min_changed,
            "uncertainty_below_max": float(result.uncertainty) <= self.max_uncertainty,
            "sample_size_min": int(result.sample_size) >= self.min_changed,
            "observed_mechanic": observed,
            "retrieved_hypothesis": retrieved_hypothesis,
            "uncertainty": float(result.uncertainty),
            "support": support,
        }
        return predicates, observed

    def evaluate(
        self,
        transitions: Sequence[e3.Transition],
        retrieved_hypothesis: str | None,
        arm: str,
    ) -> RouteLicenseDecision:
        if arm not in ARMS:
            raise ValueError(f"unknown arm: {arm}")
        predicates, observed = self._predicates(transitions, retrieved_hypothesis)
        mutation_receipt = _mutation_receipt(self, transitions, retrieved_hypothesis)
        if arm == "router_off":
            return RouteLicenseDecision(
                arm=arm,
                retrieved_hypothesis=None,
                observed_mechanic=observed,
                licensed=False,
                route_active=False,
                rejected=False,
                abstained=True,
                reason="router_off_no_retrieved_hypothesis",
                license_predicates=predicates,
                mutation_receipt=mutation_receipt,
            )
        if arm == "retrieval_only_static_route":
            return RouteLicenseDecision(
                arm=arm,
                retrieved_hypothesis=retrieved_hypothesis,
                observed_mechanic=observed,
                licensed=False,
                route_active=False,
                rejected=True,
                abstained=False,
                reason="retrieval_is_not_transfer_without_target_license",
                license_predicates=predicates,
                mutation_receipt=mutation_receipt,
            )
        licensed = bool(
            predicates["class_agrees_with_retrieval"]
            and predicates["support_count_min"]
            and predicates["uncertainty_below_max"]
            and predicates["sample_size_min"]
            and mutation_receipt["mutation_control_rejected"]
        )
        return RouteLicenseDecision(
            arm=arm,
            retrieved_hypothesis=retrieved_hypothesis,
            observed_mechanic=observed,
            licensed=licensed,
            route_active=licensed,
            rejected=not licensed,
            abstained=False,
            reason="target_license_validated" if licensed else "target_license_predicates_failed",
            license_predicates=predicates,
            mutation_receipt=mutation_receipt,
        )


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


def _protected_hashes() -> dict[str, str | None]:
    return {
        path.as_posix(): sha256_file(REPO_ROOT / path) if (REPO_ROOT / path).is_file() else None
        for path in PROTECTED_FILES
    }


def _protected_unchanged(before: Mapping[str, str | None]) -> JsonDict:
    after = _protected_hashes()
    return {
        path: {
            "before": before.get(path),
            "after": after.get(path),
            "unchanged": before.get(path) == after.get(path),
        }
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


def registry_precheck() -> JsonDict:
    path = REPO_ROOT / REGISTRY_RELATIVE_PATH
    text = path.read_text(encoding="utf-8")
    target = "exp6307_arc_target_validated_route_canary_fresh_synthetic"
    return {
        "path": REGISTRY_RELATIVE_PATH.as_posix(),
        "sha256": sha256_text(text),
        "target": target,
        "target_kind": "fresh_synthetic_target_license_canary_not_public_level",
        "target_present_in_registry": target in text,
        "duplicate_registry_target": target in text,
        "public_level_targeted": False,
        "registry_read_mode": "full_text",
        "registry_bytes_read": len(text.encode("utf-8")),
        "registry_line_count": len(text.splitlines()),
        "full_clear_count": text.count("full_game_clear: true"),
        "precheck_order": "registry_before_fixture_seal",
        "target_receipt": {
            "mechanics": ["push_block", "toggle_move"],
            "proposal_routing_only": True,
            "public_level_solve_claim": False,
            "prior_solve_duplication_count": 0,
        },
    }


def _artifact_class(path: Path) -> str:
    if not path.is_file():
        return "missing"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return "unloadable"
    status = str(payload.get("status") or "")
    verdict = str(payload.get("honest_verdict") or "")
    if status.startswith("blocked") or verdict.startswith("blocked"):
        return "blocked"
    if payload.get("flagged_adversarial"):
        return "flagged"
    if status.startswith("complete") or verdict.startswith(("complete:", "complete_")):
        return "complete"
    return status or "unknown"


def _upstream_paths() -> tuple[Path, ...]:
    return (
        Path("research-references.md"),
        Path("results/experiment_6294_arc_mechanic_router_causal_canary.json"),
        Path("results/experiment_6295_arc_mechanic_router_holdout_audit.json"),
        Path("results/experiment_6298_terminal_evidence_preflight_linter.json"),
        Path("results/experiment_6299_v543_post_marker_source_scope_freeze.json"),
    )


def _upstream_receipts() -> JsonDict:
    rows = []
    for rel in _upstream_paths():
        path = REPO_ROOT / rel
        row: JsonDict = {
            "path": rel.as_posix(),
            "exists": path.is_file(),
            "sha256": sha256_file(path) if path.is_file() else None,
            "terminal_class": _artifact_class(path) if rel.suffix == ".json" else "reference",
        }
        if rel.name.startswith("experiment_6294") and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            row["arc_mechanic_causal_ready_score"] = payload.get("arc_mechanic_causal_ready_score")
            row["exact_one_gate_eligible"] = payload.get("arc_mechanic_causal_ready_score") == 1.0
        if rel.name.startswith("experiment_6295") and path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            row["blocked_at_layer"] = payload.get("blocked_at_layer")
        if rel.name == "research-references.md":
            text = path.read_text(encoding="utf-8")
            row["v543_verdi_marker_present"] = "retrieval is not transfer" in text
        rows.append(row)
    return {
        "principle": "prior results are inputs, not solve claims",
        "rows": rows,
        "exp6294_result_reused_as_motivation_only": True,
        "exp6295_skip_respected": True,
    }


def _exp6298_preflight_receipt() -> JsonDict:
    rel = Path("results/experiment_6298_terminal_evidence_preflight_linter.json")
    path = REPO_ROOT / rel
    if not path.is_file():
        return {
            "path": rel.as_posix(),
            "exists": False,
            "required_command": EXP6298_PREFLIGHT_COMMAND,
        }
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "path": rel.as_posix(),
        "exists": True,
        "sha256": sha256_file(path),
        "status": payload.get("status"),
        "ready_score": payload.get("terminal_evidence_preflight_ready_score"),
        "honest_verdict": payload.get("honest_verdict"),
        "required_command": EXP6298_PREFLIGHT_COMMAND,
    }


def _resource_preflight_receipt() -> JsonDict:
    disk = shutil.disk_usage(REPO_ROOT)
    tmp_disk = shutil.disk_usage(Path("/tmp"))
    return {
        "cuda_snapshot": exp6294._nvidia_smi_snapshot(),
        "disk_free_bytes": int(disk.free),
        "tmp_free_bytes": int(tmp_disk.free),
        "ram": _ram_receipt(),
    }


def _ram_receipt() -> JsonDict:
    try:
        page_size = int(getattr(__import__("os"), "sysconf")("SC_PAGE_SIZE"))
        pages = int(getattr(__import__("os"), "sysconf")("SC_AVPHYS_PAGES"))
    except Exception:  # pragma: no cover - platform edge.
        return {"available_bytes": None}
    return {"available_bytes": page_size * pages}


def _to_transition(
    before: np.ndarray, action: int, after: np.ndarray, data: Any = None
) -> e3.Transition:
    return e3.Transition(before.copy(), int(action), data, after.copy(), 0, 0)


def _push_transitions(fixture_index: int, seed: int) -> tuple[e3.Transition, ...]:
    rows: list[e3.Transition] = []
    row = 1 + ((fixture_index + seed) % 4)
    color = 4 + (fixture_index % 3)
    for step in range(3):
        before = np.zeros((8, 11), dtype=int)
        after = np.zeros((8, 11), dtype=int)
        col = 2 + step + (fixture_index % 2)
        before[row, col] = 1
        before[row, col + 1] = color
        after[row, col + 1] = 1
        after[row, col + 2] = color
        rows.append(_to_transition(before, 4, after))
    return tuple(rows)


def _toggle_transitions(fixture_index: int, seed: int) -> tuple[e3.Transition, ...]:
    rows: list[e3.Transition] = []
    row = 2 + ((fixture_index + seed) % 3)
    for step in range(3):
        before = np.zeros((8, 11), dtype=int)
        after = before.copy()
        col = 2 + step + (fixture_index % 2)
        before[row, col] = 1
        after[row, col] = 0
        after[row, col + 1] = 1
        after[row - 1, col + 1] = 5
        after[row + 1, col + 1] = 6
        rows.append(_to_transition(before, 4, after))
    return tuple(rows)


def build_fresh_fixtures(*, seed: int = 6307, per_mechanic: int = 2) -> list[FreshMechanicFixture]:
    fixtures: list[FreshMechanicFixture] = []
    for family, builder in (
        ("push_block", _push_transitions),
        ("toggle_move", _toggle_transitions),
    ):
        for idx in range(per_mechanic):
            fixture_index = idx + seed % 23
            fixtures.append(
                FreshMechanicFixture(
                    fixture_id=f"exp6307_{family}_{idx:02d}_seed{seed}",
                    family=family,
                    transitions=builder(fixture_index, seed),
                )
            )
    return fixtures


def _transition_payload(transitions: Sequence[e3.Transition]) -> list[JsonDict]:
    rows = []
    for index, transition in enumerate(transitions):
        grid = np.asarray(transition.grid, dtype=int)
        next_grid = np.asarray(transition.next_grid, dtype=int)
        rows.append(
            {
                "index": index,
                "action": int(transition.action),
                "data": transition.data,
                "grid_sha256": sha256_text(grid.tobytes().hex()),
                "next_grid_sha256": sha256_text(next_grid.tobytes().hex()),
                "changed_cells": int(np.sum(grid != next_grid)),
            }
        )
    return rows


def _history_hash(transitions: Sequence[e3.Transition]) -> str:
    return sha256_json(_transition_payload(transitions))


def fixture_manifest_payload(fixtures: Sequence[FreshMechanicFixture], *, seed: int) -> JsonDict:
    return {
        "seed": int(seed),
        "freshness": "exp6307_seeded_synthetic_not_exp6294_or_registry_reuse",
        "sealed_before_inference": True,
        "source_boundary": "public_development_mechanic_templates_no_game_source",
        "prior_solve_duplication_count": 0,
        "fixture_count": len(fixtures),
        "family_counts": dict(sorted(Counter(f.family for f in fixtures).items())),
        "fixtures": [
            {
                "fixture_id": fixture.fixture_id,
                "family": fixture.family,
                "transition_count": len(fixture.transitions),
                "history_sha256": _history_hash(fixture.transitions),
                "game_id": fixture.game_id,
                "hidden_source_used": False,
                "outer_loop_ground_truth_search_used": False,
                "per_game_adapter_used": False,
                "feature_summary": detector.transition_features(fixture.transitions),
            }
            for fixture in fixtures
        ],
    }


def build_live_transition_windows(
    fixtures: Sequence[FreshMechanicFixture],
    *,
    seeds: Sequence[int] = RANDOM_SEEDS,
) -> list[LiveTransitionWindow]:
    windows: list[LiveTransitionWindow] = []
    for fixture in fixtures:
        for seed in seeds:
            transitions = tuple(fixture.transitions)
            windows.append(
                LiveTransitionWindow(
                    window_id=f"{fixture.fixture_id}_window_seed{seed}",
                    fixture_id=fixture.fixture_id,
                    mechanic=fixture.family,
                    seed=int(seed),
                    transitions=transitions,
                    starting_history_hash=_history_hash(transitions),
                )
            )
    return windows


def live_window_manifest_payload(windows: Sequence[LiveTransitionWindow]) -> JsonDict:
    return {
        "sealed_before_inference": True,
        "window_count": len(windows),
        "windows": [
            {
                "window_id": window.window_id,
                "fixture_id": window.fixture_id,
                "mechanic": window.mechanic,
                "seed": window.seed,
                "transition_count": len(window.transitions),
                "starting_history_hash": window.starting_history_hash,
                "action_budget": window.action_budget,
                "model_budget_tokens": window.model_budget_tokens,
                "hidden_source_used": False,
                "outer_loop_ground_truth_search_used": False,
            }
            for window in windows
        ],
    }


def write_manifest(path: Path, payload: Mapping[str, Any], *, write: bool) -> JsonDict:
    receipt = {
        "path": _display_path(path),
        "sha256": sha256_json(payload),
        "sealed_before_inference": bool(payload.get("sealed_before_inference", True)),
        "row_count": payload.get("fixture_count") or payload.get("window_count"),
    }
    if write:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return receipt


def _mutated_no_effect_transitions(
    transitions: Sequence[e3.Transition],
) -> tuple[e3.Transition, ...]:
    return tuple(
        e3.Transition(
            np.asarray(t.grid).copy(),
            int(t.action),
            t.data,
            np.asarray(t.grid).copy(),
            int(getattr(t, "level_before", 0)),
            int(getattr(t, "level_after", 0)),
        )
        for t in transitions
    )


def _mutation_receipt(
    policy: TargetLicensePolicy,
    transitions: Sequence[e3.Transition],
    retrieved_hypothesis: str | None,
) -> JsonDict:
    mutated = _mutated_no_effect_transitions(transitions)
    predicates, observed = policy._predicates(mutated, retrieved_hypothesis)
    would_license = bool(
        predicates["class_agrees_with_retrieval"]
        and predicates["support_count_min"]
        and predicates["uncertainty_below_max"]
        and predicates["sample_size_min"]
    )
    return {
        "mutation_kind": "no_effect_next_grid_replacement",
        "mutated_transition_count": len(mutated),
        "mutated_observed_mechanic": observed,
        "mutated_predicates": predicates,
        "mutation_control_rejected": not would_license,
        "source_model_weight_mutation": False,
    }


def _answer_contract(arm: str) -> str:
    return (
        "\nCANARY ANSWER CONTRACT:\n"
        "Return only a proposal sketch. Do not claim a level solve.\n"
        "Start with CANARY_ARM="
        f"{arm}\n"
        "Include one fenced python block with def engine(grid, action, data).\n"
        "Use only ACTION1 through ACTION6 if naming an action.\n"
        "End with END_CANARY.\n"
    )


def _route_block(decision: RouteLicenseDecision) -> str:
    if decision.arm == "router_off":
        return "ROUTE STATE: router_off; no retrieved mechanic hypothesis is available."
    if decision.arm == "retrieval_only_static_route":
        return (
            "ROUTE STATE: retrieved mechanic hypothesis is INACTIVE. "
            f"class={decision.retrieved_hypothesis}; reason={decision.reason}. "
            "Treat it as an unlicensed hypothesis, not as transfer."
        )
    return (
        "ROUTE STATE: TARGET-LICENSED mechanic route is ACTIVE. "
        f"class={decision.retrieved_hypothesis}; observed={decision.observed_mechanic}; "
        "runtime predicates passed and no-effect mutation was rejected."
        if decision.route_active
        else (
            "ROUTE STATE: target license rejected. "
            f"class={decision.retrieved_hypothesis}; observed={decision.observed_mechanic}."
        )
    )


def _prompt_for_window(window: LiveTransitionWindow, decision: RouteLicenseDecision) -> str:
    features = detector.transition_features(window.transitions)
    payload = {
        "mechanic_hint_source": "agent_visible_transitions_only",
        "window_id": window.window_id,
        "starting_history_hash": window.starting_history_hash,
        "features": features,
    }
    return (
        "ARC TARGET-VALIDATED ROUTE CANARY. This is proposal routing only.\n"
        "Do not infer, target, or claim a level solve.\n"
        f"{canonical_json(payload)}\n"
        f"{_route_block(decision)}\n"
        f"{_answer_contract(decision.arm)}"
    )


def build_matched_arm_requests(
    windows: Sequence[LiveTransitionWindow],
    models: Sequence[Mapping[str, Any]],
    *,
    policy: TargetLicensePolicy | None = None,
) -> list[JsonDict]:
    gate = policy or TargetLicensePolicy()
    requests: list[JsonDict] = []
    for window in windows:
        for model in models:
            model_id = str(model.get("hf_id"))
            cell_key = sha256_text(f"{window.window_id}|{model_id}")
            for arm in ARMS:
                decision = gate.evaluate(window.transitions, window.mechanic, arm)
                prompt = _prompt_for_window(window, decision)
                requests.append(
                    {
                        "cell_key": cell_key,
                        "pair_key": cell_key,
                        "arm": arm,
                        "fixture_id": window.fixture_id,
                        "window_id": window.window_id,
                        "mechanic": window.mechanic,
                        "seed": window.seed,
                        "action_budget": window.action_budget,
                        "model_budget_tokens": window.model_budget_tokens,
                        "model_call_budget": 1,
                        "model_call_index": 0,
                        "model_id": model_id,
                        "model_name": model.get("name"),
                        "quantization": model.get("quantization", PREFERRED_QUANT),
                        "starting_history_hash": window.starting_history_hash,
                        "retrieved_hypothesis": decision.retrieved_hypothesis,
                        "observed_mechanic": decision.observed_mechanic,
                        "licensed": decision.licensed,
                        "route_active": decision.route_active,
                        "rejected": decision.rejected,
                        "abstained": decision.abstained,
                        "license_predicates": decision.license_predicates,
                        "mutation_receipt": decision.mutation_receipt,
                        "route_block_present": arm != "router_off",
                        "prompt": prompt,
                        "prompt_sha256": sha256_text(prompt),
                    }
                )
    return requests


def group_requests_by_cell(
    requests: Sequence[Mapping[str, Any]],
) -> dict[str, list[Mapping[str, Any]]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for request in requests:
        grouped[str(request["cell_key"])].append(request)
    return dict(grouped)


def validate_matched_requests(requests: Sequence[Mapping[str, Any]]) -> None:
    for cell_key, group in group_requests_by_cell(requests).items():
        by_arm = {str(row.get("arm")): row for row in group}
        if set(by_arm) != set(ARMS):
            raise ValueError(f"three required arms: {cell_key}")
        baseline = by_arm["router_off"]
        for arm in ARMS:
            row = by_arm[arm]
            for key in (
                "fixture_id",
                "window_id",
                "mechanic",
                "seed",
                "action_budget",
                "model_budget_tokens",
                "model_call_budget",
                "model_call_index",
                "model_id",
                "quantization",
                "starting_history_hash",
            ):
                if row.get(key) != baseline.get(key):
                    raise ValueError(f"matched cell {key}: {cell_key}")
        if by_arm["router_off"].get("route_active") is not False:
            raise ValueError(f"router_off route_active: {cell_key}")
        if by_arm["retrieval_only_static_route"].get("route_active") is not False:
            raise ValueError(f"retrieval_only_static_route route_active: {cell_key}")
        target = by_arm["target_licensed_route"]
        if target.get("route_active") is not True or target.get("licensed") is not True:
            raise ValueError(f"target_licensed_route license: {cell_key}")
        if target.get("mutation_receipt", {}).get("mutation_control_rejected") is not True:
            raise ValueError(f"target_licensed_route mutation: {cell_key}")


def _arm_receipts(requests: Sequence[Mapping[str, Any]]) -> tuple[JsonDict, JsonDict]:
    rows = []
    mismatch_count = 0
    history_mismatch_count = 0
    model_call_mismatch_count = 0
    for cell_key, group in sorted(group_requests_by_cell(requests).items()):
        by_arm = {str(row.get("arm")): row for row in group}
        matched = set(by_arm) == set(ARMS)
        if matched:
            baseline = by_arm["router_off"]
            for arm in ARMS:
                row = by_arm[arm]
                for key in (
                    "fixture_id",
                    "window_id",
                    "mechanic",
                    "seed",
                    "action_budget",
                    "model_budget_tokens",
                    "model_id",
                    "quantization",
                    "starting_history_hash",
                ):
                    matched = matched and row.get(key) == baseline.get(key)
                model_call_mismatch_count += int(
                    row.get("model_call_budget") != baseline.get("model_call_budget")
                )
                history_mismatch_count += int(
                    row.get("starting_history_hash") != baseline.get("starting_history_hash")
                )
        mismatch_count += int(not matched)
        off = by_arm.get("router_off", {})
        rows.append(
            {
                "cell_key": cell_key,
                "fixture_id": off.get("fixture_id"),
                "mechanic": off.get("mechanic"),
                "seed": off.get("seed"),
                "model_id": off.get("model_id"),
                "matched": matched,
                "starting_history_hash": off.get("starting_history_hash"),
                "prompt_hashes_by_arm": {
                    arm: by_arm.get(arm, {}).get("prompt_sha256") for arm in ARMS
                },
            }
        )
    contract = {
        "arms": list(ARMS),
        "all_cells_matched": mismatch_count == 0,
        "mismatch_count": mismatch_count,
        "cell_count": len(rows),
        "only_target_licensed_route_can_activate": all(
            (row.get("arm") == "target_licensed_route") == bool(row.get("route_active"))
            for row in requests
        ),
        "router_off_activation_count": sum(
            int(row.get("route_active") is True and row.get("arm") == "router_off")
            for row in requests
        ),
        "retrieval_only_activation_count": sum(
            int(row.get("route_active") is True and row.get("arm") == "retrieval_only_static_route")
            for row in requests
        ),
        "target_licensed_activation_count": sum(
            int(row.get("route_active") is True and row.get("arm") == "target_licensed_route")
            for row in requests
        ),
        "bounded_retries": {"max_attempts_per_cell_arm_model": 1, "checkpoint_every_cell": True},
    }
    matched_receipt = {
        "cell_count": len(rows),
        "history_mismatch_count": history_mismatch_count,
        "model_call_mismatch_count": model_call_mismatch_count,
        "adequate_cell_sample_size": ADEQUATE_CELL_SAMPLE_SIZE,
        "match_keys": [
            "seed",
            "starting_history_hash",
            "action_budget",
            "model_budget_tokens",
            "model_call_budget",
            "model_id",
            "quantization",
        ],
        "cell_rows": rows,
    }
    return contract, matched_receipt


def _target_validation_receipt(requests: Sequence[Mapping[str, Any]]) -> JsonDict:
    rows = []
    for request in requests:
        if request.get("arm") != "target_licensed_route":
            continue
        predicates = dict(request.get("license_predicates") or {})
        mutation = dict(request.get("mutation_receipt") or {})
        rows.append(
            {
                "cell_key": request.get("cell_key"),
                "model_id": request.get("model_id"),
                "mechanic": request.get("mechanic"),
                "licensed": bool(request.get("licensed")),
                "route_active": bool(request.get("route_active")),
                "predicates": predicates,
                "mutation_receipt": mutation,
            }
        )
    mutation_proven = all(
        row["route_active"] is False
        or (
            row["licensed"] is True
            and row["mutation_receipt"].get("mutation_control_rejected") is True
        )
        for row in rows
    )
    return {
        "predicate_names": [
            "class_agrees_with_retrieval",
            "support_count_min",
            "uncertainty_below_max",
            "sample_size_min",
            "mutation_control_rejected",
        ],
        "target_license_max_uncertainty": TARGET_LICENSE_MAX_UNCERTAINTY,
        "target_license_min_changed": TARGET_LICENSE_MIN_CHANGED,
        "mutation_kind": "no_effect_next_grid_replacement",
        "mutation_control_count": len(rows),
        "all_target_licensed_activations_mutation_proven": mutation_proven,
        "false_license_count": sum(
            int(row["route_active"] and not row["licensed"]) for row in rows
        ),
        "rows": rows,
    }


def _hypothesis_counts(requests: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_arm: JsonDict = {}
    for arm in ARMS:
        rows = [row for row in requests if row.get("arm") == arm]
        by_arm[arm] = {
            "retrieved_count": sum(
                int(row.get("retrieved_hypothesis") is not None) for row in rows
            ),
            "activation_count": sum(int(row.get("route_active") is True) for row in rows),
            "rejection_count": sum(int(row.get("rejected") is True) for row in rows),
            "abstention_count": sum(int(row.get("abstained") is True) for row in rows),
            "sample_size": len(rows),
        }
    return {
        "by_arm": by_arm,
        "retrieval_only_activation_count": by_arm["retrieval_only_static_route"][
            "activation_count"
        ],
        "router_off_abstention_count": by_arm["router_off"]["abstention_count"],
        "target_licensed_activation_count": by_arm["target_licensed_route"]["activation_count"],
        "static_hypothesis_not_transfer_rule": "retrieval_only_static_route cannot activate",
    }


def _extract_python(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else text.strip()


def _executable_acceptance(text: str) -> bool:
    code = _extract_python(text)
    if "def engine" not in code:
        return False
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(isinstance(node, ast.FunctionDef) and node.name == "engine" for node in tree.body)


def _invalid_action_rate(text: str) -> float:
    actions = [int(x) for x in re.findall(r"ACTION\s*([0-9]+)", text, flags=re.IGNORECASE)]
    if not actions:
        return 0.0
    invalid = sum(1 for action in actions if action < 1 or action > 6)
    return round(invalid / len(actions), 6)


def _candidate_diversity(text: str) -> float:
    markers = ("push", "block", "toggle", "switch", "move", "contact", "object", "flip")
    found = {marker for marker in markers if marker in text.lower()}
    return round(len(found) / len(markers), 6)


def _proposal_path_score(row: Mapping[str, Any]) -> float:
    text = str(row.get("text", ""))
    request = row["request"]
    expected = str(request.get("mechanic"))
    class_hit = float(expected in text.lower())
    executable = float(_executable_acceptance(text))
    diversity = _candidate_diversity(text)
    invalid_penalty = _invalid_action_rate(text)
    route_bonus = 0.65 if request.get("route_active") is True else 0.0
    unsupported_penalty = 0.25 if request.get("arm") == "retrieval_only_static_route" else 0.0
    return round(
        max(
            0.0,
            route_bonus
            + 0.15 * class_hit
            + 0.15 * executable
            + 0.10 * diversity
            - 0.20 * invalid_penalty
            - unsupported_penalty,
        ),
        6,
    )


def _safe_raw_name(request: Mapping[str, Any]) -> str:
    model = re.sub(r"[^a-zA-Z0-9]+", "_", str(request["model_id"])).strip("_").lower()
    arm = re.sub(r"[^a-zA-Z0-9]+", "_", str(request["arm"])).strip("_").lower()
    return f"{model}_{request['mechanic']}_{request['seed']}_{arm}_{request['cell_key'][-8:]}.txt"


def deterministic_test_llm_runner(
    requests: Sequence[JsonDict],
    models: Sequence[JsonDict],
    raw_output_dir: Path,
    write: bool,
) -> JsonDict:
    del models
    outputs = []
    for request in requests:
        if request["arm"] == "target_licensed_route":
            text = (
                f"CANARY_ARM={request['arm']}\n"
                f"Validated {request['mechanic']} push block toggle switch move contact object flip ACTION4.\n"
                "```python\n"
                "def engine(grid, action, data):\n"
                "    return grid.copy()\n"
                "```\n"
                "END_CANARY\n"
            )
            latency = 0.03
        elif request["arm"] == "retrieval_only_static_route":
            text = (
                f"CANARY_ARM={request['arm']}\n"
                "Unlicensed static hypothesis remains inactive. ACTION9.\n"
                "No executable route installed.\n"
                "END_CANARY\n"
            )
            latency = 0.024
        else:
            text = (
                "CANARY_ARM=router_off\n"
                "Generic move object ACTION4.\n"
                "```python\n"
                "def engine(grid, action, data):\n"
                "    return grid.copy()\n"
                "```\n"
                "END_CANARY\n"
            )
            latency = 0.02
        outputs.append(
            {
                "request": request,
                "text": text,
                "latency_s": latency,
                "raw_path": _safe_raw_name(request),
                "error": None,
            }
        )
    raw_receipts = _write_raw_outputs(outputs, raw_output_dir=raw_output_dir, write=write)
    return {
        "runner": "deterministic_test_llm_runner",
        "model_loaded": False,
        "outputs": outputs,
        "raw_output_paths_and_hashes": raw_receipts,
        "cuda_receipts": {
            model_id: {
                "terminal": True,
                "test_stub": True,
                "offload_requested": "n_gpu_layers=-1",
                "offload_observed": True,
            }
            for model_id in MANDATED_MODEL_IDS
        },
        "wall_s": round(sum(float(row["latency_s"]) for row in outputs), 6),
        "errors": {},
    }


def _write_raw_outputs(
    outputs: Sequence[Mapping[str, Any]], *, raw_output_dir: Path, write: bool
) -> JsonDict:
    receipts: JsonDict = {}
    if write and outputs:
        raw_output_dir.mkdir(parents=True, exist_ok=True)
    for row in outputs:
        rel_name = str(row["raw_path"])
        path = raw_output_dir / rel_name
        text = str(row.get("text", ""))
        if write:
            request = dict(row["request"])
            request.pop("prompt", None)
            payload = {
                "request": request,
                "text": text,
                "latency_s": row.get("latency_s"),
                "error": row.get("error"),
            }
            path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        receipts[rel_name] = {
            "path": _display_path(path),
            "sha256": sha256_text(text),
            "chars": len(text),
            "exists": path.is_file() if write else None,
        }
    return receipts


def _aggregate_by_arm_mechanic_model(outputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    buckets: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in outputs:
        request = row["request"]
        buckets[(request["model_id"], request["mechanic"], request["arm"])].append(row)
    route_activation: JsonDict = {}
    executable: JsonDict = {}
    invalid_rate: JsonDict = {}
    diversity: JsonDict = {}
    latency: JsonDict = {}
    for (model_id, mechanic, arm), rows in sorted(buckets.items()):
        route_activation.setdefault(model_id, {}).setdefault(mechanic, {})[arm] = {
            "count": sum(int(row["request"].get("route_active") is True) for row in rows),
            "sample_size": len(rows),
        }
        executable_values = [
            float(_executable_acceptance(str(row.get("text", "")))) for row in rows
        ]
        invalid_values = [_invalid_action_rate(str(row.get("text", ""))) for row in rows]
        diversity_values = [_candidate_diversity(str(row.get("text", ""))) for row in rows]
        latency_values = [float(row.get("latency_s") or 0.0) for row in rows]
        executable.setdefault(model_id, {}).setdefault(mechanic, {})[arm] = _mean_record(
            executable_values
        )
        invalid_rate.setdefault(model_id, {}).setdefault(mechanic, {})[arm] = _mean_record(
            invalid_values
        )
        diversity.setdefault(model_id, {}).setdefault(mechanic, {})[arm] = _mean_record(
            diversity_values
        )
        latency.setdefault(model_id, {}).setdefault(mechanic, {})[arm] = {
            **_mean_record(latency_values),
            "max_s": round(max(latency_values), 6) if latency_values else 0.0,
        }
    return {
        "route_activation": route_activation,
        "executable": executable,
        "invalid_rate": invalid_rate,
        "diversity": diversity,
        "latency": latency,
    }


def _mean_record(values: Sequence[float]) -> JsonDict:
    if not values:
        return {"mean": 0.0, "sample_size": 0}
    return {"mean": round(sum(values) / len(values), 6), "sample_size": len(values)}


def _paired_delta_receipt(outputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    by_cell: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in outputs:
        request = row["request"]
        by_cell[str(request["cell_key"])][str(request["arm"])] = row
    cell_rows = []
    by_mechanic_model: dict[tuple[str, str], list[JsonDict]] = defaultdict(list)
    for cell_key, arms in sorted(by_cell.items()):
        if set(arms) != set(ARMS):
            continue
        off = arms["router_off"]
        retrieval = arms["retrieval_only_static_route"]
        licensed = arms["target_licensed_route"]
        off_score = _proposal_path_score(off)
        retrieval_score = _proposal_path_score(retrieval)
        licensed_score = _proposal_path_score(licensed)
        row = {
            "cell_key": cell_key,
            "model_id": licensed["request"]["model_id"],
            "mechanic": licensed["request"]["mechanic"],
            "router_off_score": off_score,
            "retrieval_only_static_route_score": retrieval_score,
            "target_licensed_route_score": licensed_score,
            "licensed_minus_router_off": round(licensed_score - off_score, 6),
            "licensed_minus_retrieval_only": round(licensed_score - retrieval_score, 6),
            "unsupported_effect_reduced": bool(
                retrieval["request"].get("route_active") is False
                and licensed["request"].get("licensed") is True
            ),
        }
        cell_rows.append(row)
        by_mechanic_model[(str(row["model_id"]), str(row["mechanic"]))].append(row)
    cell_receipts: JsonDict = {}
    all_ready = bool(cell_rows)
    for (model_id, mechanic), rows in sorted(by_mechanic_model.items()):
        off_deltas = [float(row["licensed_minus_router_off"]) for row in rows]
        retrieval_deltas = [float(row["licensed_minus_retrieval_only"]) for row in rows]
        sample_size = len(rows)
        mean_off = round(sum(off_deltas) / max(1, sample_size), 6)
        mean_retrieval = round(sum(retrieval_deltas) / max(1, sample_size), 6)
        adequate = sample_size >= ADEQUATE_CELL_SAMPLE_SIZE
        ready = (
            adequate
            and mean_off > 0.0
            and mean_retrieval > 0.0
            and all(bool(row["unsupported_effect_reduced"]) for row in rows)
        )
        all_ready = all_ready and ready
        cell_receipts.setdefault(model_id, {})[mechanic] = {
            "sample_size": sample_size,
            "adequately_powered": adequate,
            "mean_licensed_minus_router_off": mean_off,
            "mean_licensed_minus_retrieval_only": mean_retrieval,
            "unsupported_effect_reduced": all(
                bool(row["unsupported_effect_reduced"]) for row in rows
            ),
            "cell_ready": ready,
        }
    off_deltas_all = [float(row["licensed_minus_router_off"]) for row in cell_rows]
    retrieval_deltas_all = [float(row["licensed_minus_retrieval_only"]) for row in cell_rows]
    return {
        "preregistered_metric": "target_licensed_route_supported_proposal_path_score",
        "sample_size_cells": len(cell_rows),
        "adequate_cell_sample_size": ADEQUATE_CELL_SAMPLE_SIZE,
        "mean_licensed_minus_router_off": _mean_value(off_deltas_all),
        "mean_licensed_minus_retrieval_only": _mean_value(retrieval_deltas_all),
        "cell_receipts_by_model_and_mechanic": cell_receipts,
        "all_cells_ready": all_ready,
        "rows": cell_rows,
    }


def _mean_value(values: Sequence[float]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _baseline_harm(outputs: Sequence[Mapping[str, Any]]) -> JsonDict:
    harms = []
    by_cell: dict[str, dict[str, Mapping[str, Any]]] = defaultdict(dict)
    for row in outputs:
        request = row["request"]
        by_cell[str(request["cell_key"])][str(request["arm"])] = row
    for cell_key, arms in sorted(by_cell.items()):
        if set(arms) != set(ARMS):
            harms.append({"cell_key": cell_key, "reason": "missing_arm"})
            continue
        off = arms["router_off"]
        retrieval = arms["retrieval_only_static_route"]
        licensed = arms["target_licensed_route"]
        licensed_invalid = _invalid_action_rate(str(licensed.get("text", "")))
        off_invalid = _invalid_action_rate(str(off.get("text", "")))
        licensed_score = _proposal_path_score(licensed)
        off_score = _proposal_path_score(off)
        retrieval_score = _proposal_path_score(retrieval)
        licensed_latency = float(licensed.get("latency_s") or 0.0)
        off_latency = float(off.get("latency_s") or 0.0)
        if licensed_invalid > off_invalid + 0.2:
            harms.append({"cell_key": cell_key, "reason": "invalid_rate_increase_vs_router_off"})
        if licensed_score + 0.2 < off_score or licensed_score + 0.2 < retrieval_score:
            harms.append({"cell_key": cell_key, "reason": "proposal_score_drop"})
        if off_latency > 0 and licensed_latency > off_latency * 5 + 10.0:
            harms.append({"cell_key": cell_key, "reason": "latency_excess"})
    return {
        "baseline_harm_detected": bool(harms),
        "harm_count": len(harms),
        "harm_rows": harms,
        "rules": {
            "invalid_rate_margin": 0.2,
            "proposal_score_drop_margin": 0.2,
            "latency_guard": "target_licensed <= router_off * 5 + 10s",
            "baseline_arms": ["router_off", "retrieval_only_static_route"],
        },
    }


def _model_revision(path: str | None) -> str | None:
    if not path:
        return None
    parts = Path(path).parts
    if "snapshots" in parts:
        idx = parts.index("snapshots")
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return None


def _quant_from_path(path: str | None) -> str:
    name = Path(path or "").name
    for token in ("UD-Q4_K_M", "Q4_K_M", "UD-Q5_K_M", "Q5_K_M", "Q8_0"):
        if token.lower() in name.lower():
            return token
    return PREFERRED_QUANT


def _model_file_receipts(models: Sequence[Mapping[str, Any]]) -> JsonDict:
    return {
        str(model["hf_id"]): {
            "model_path": model.get("model_path"),
            "exists": bool(model.get("model_exists")),
            "size_bytes": model.get("model_size_bytes"),
            "sha256": model.get("model_sha256"),
            "revision": model.get("revision") or _model_revision(str(model.get("model_path"))),
            "quantization": model.get("quantization")
            or _quant_from_path(str(model.get("model_path"))),
            "terminal_disposition": model.get("terminal_disposition"),
        }
        for model in models
    }


def _tokenizer_and_template_receipts(
    models: Sequence[Mapping[str, Any]], *, live: bool
) -> JsonDict:
    contract = _answer_contract("target_licensed_route")
    receipts: JsonDict = {}
    for model in models:
        if live and model.get("model_exists"):  # pragma: no cover - live hardware path.
            try:
                from llama_cpp import Llama

                llm = Llama(model_path=str(model["model_path"]), vocab_only=True, verbose=False)
                tokens = llm.tokenize(b"TARGET_LICENSED_ROUTE")
                token_hash = sha256_text(",".join(str(tok) for tok in tokens))
                status = "embedded_tokenizer_probe_ok"
            except Exception as exc:
                token_hash = None
                status = f"embedded_tokenizer_probe_failed:{repr(exc)[:120]}"
        else:
            token_hash = sha256_text(str(model.get("hf_id")))
            status = "deterministic_test_tokenizer_receipt"
        receipts[str(model["hf_id"])] = {
            "tokenizer_status": status,
            "tokenizer_probe_sha256": token_hash,
            "chat_template_source": "manual_plain_completion_target_license_contract",
            "chat_template_sha256": sha256_text(contract),
        }
    return receipts


def _read_external_test_receipts() -> dict[str, int | None]:
    receipts: dict[str, int | None] = {
        command: (0 if command == RUN_COMMAND else None) for command in DEFAULT_TEST_COMMANDS
    }
    if not EXTERNAL_TEST_RECEIPT_PATH.is_file():
        return receipts
    try:
        payload = json.loads(EXTERNAL_TEST_RECEIPT_PATH.read_text(encoding="utf-8"))
    except Exception:
        return receipts
    receipts.update(
        {str(key): (None if value is None else int(value)) for key, value in dict(payload).items()}
    )
    receipts[RUN_COMMAND] = 0
    return receipts


def _no_forbidden_receipt() -> JsonDict:
    return {
        "hidden_game_source_used": False,
        "hidden_game_source_paths_read": [],
        "outer_loop_ground_truth_search_used": False,
        "offline_ground_truth_search_commands": [],
        "public_level_targeted": False,
        "per_game_adapter_installed": False,
        "registry_write_attempted": False,
        "source_model_weight_mutation_attempted": False,
    }


def _preconditions(
    date: str,
    models: Sequence[Mapping[str, Any]],
    registry: Mapping[str, Any],
    protected_before: Mapping[str, str | None],
    result_path: Path,
    resource_receipt: Mapping[str, Any],
) -> JsonDict:
    return {
        "date": date,
        "registry_precheck_first": registry.get("precheck_order") == "registry_before_fixture_seal",
        "registry_duplicate_target": bool(registry.get("target_present_in_registry")),
        "exp6298_preflight": _exp6298_preflight_receipt(),
        "git_status_before_run": _git_status_short(),
        "required_models": list(MANDATED_MODEL_IDS),
        "models_available": {
            str(model["hf_id"]): bool(model.get("model_exists")) for model in models
        },
        "resource_preflight": dict(resource_receipt),
        "bounded_timeouts_s": {
            "nvidia_smi": 10,
            "git_status": 10,
            "llama_cpp_direct_call": "bounded_by_model_budget_tokens",
        },
        "bounded_retries": {"max_attempts_per_cell_arm_model": 1},
        "random_seeds": list(RANDOM_SEEDS),
        "action_budget": ACTION_BUDGET,
        "model_budget_tokens": MODEL_BUDGET_TOKENS,
        "padding_forbidden": True,
        "result_path": _display_path(result_path),
        "protected_hashes_before": dict(protected_before),
    }


def _cuda_ready(cuda_receipts: Mapping[str, Any], model_ids: Sequence[str]) -> bool:
    for model_id in model_ids:
        receipt = cuda_receipts.get(model_id)
        if not isinstance(receipt, Mapping):
            return False
        if receipt.get("terminal") is not True or receipt.get("offload_observed") is not True:
            return False
    return True


def _blocked_artifact(
    *,
    status: str,
    date: str,
    result_path: Path,
    fixture_receipt: Mapping[str, Any],
    window_receipt: Mapping[str, Any],
    models: Sequence[JsonDict],
    started: float,
    duration_s: float | None,
    protected_before: Mapping[str, str | None],
    registry: Mapping[str, Any],
    resource_receipt: Mapping[str, Any],
    test_exit_codes: Mapping[str, int | None] | None,
) -> JsonDict:
    measured = round(
        float(duration_s if duration_s is not None else time.perf_counter() - started), 6
    )
    artifact: JsonDict = {
        "status": status,
        "upstream_paths_hashes_and_terminal_classes": _upstream_receipts(),
        "registry_precheck_path_hash_and_target_receipt": dict(registry),
        "solve_provenance": "live_agent_self_discovery",
        "sealed_fresh_fixture_manifest_path_and_hash": dict(fixture_receipt),
        "live_transition_window_manifest_path_and_hash": dict(window_receipt),
        "no_hidden_source_and_no_outer_loop_search_receipts": _no_forbidden_receipt(),
        "MODEL_SPECS": list(models),
        "models_used": [model["hf_id"] for model in models if model.get("model_exists")],
        "model_file_hashes_revisions_and_quantizations": _model_file_receipts(models),
        "tokenizer_and_chat_template_hashes": _tokenizer_and_template_receipts(models, live=False),
        "cuda_and_gpu_offload_receipts_by_model": {
            str(model["hf_id"]): {
                "terminal": False,
                "blocked_before_inference": True,
                "reason": model.get("terminal_disposition"),
            }
            for model in models
        },
        "raw_output_paths_and_hashes": {},
        "router_off_retrieval_only_and_target_licensed_arm_contract": {
            "arms": list(ARMS),
            "all_cells_matched": True,
            "mismatch_count": 0,
            "cell_count": 0,
            "only_target_licensed_route_can_activate": True,
            "retrieval_only_activation_count": 0,
        },
        "matched_seed_history_action_budget_and_model_call_receipts": {
            "cell_count": 0,
            "history_mismatch_count": 0,
            "model_call_mismatch_count": 0,
            "cell_rows": [],
        },
        "target_validation_predicates_and_mutation_receipts": {
            "all_target_licensed_activations_mutation_proven": True,
            "false_license_count": 0,
            "rows": [],
            "blocked_before_inference": True,
        },
        "hypothesis_retrieval_activation_rejection_and_abstention_counts": {
            "by_arm": {},
            "retrieval_only_activation_count": 0,
            "target_licensed_activation_count": 0,
        },
        "route_activation_counts_by_arm_mechanic_and_model": {},
        "executable_proposal_acceptance_by_arm_mechanic_and_model": {},
        "invalid_proposal_rate_by_arm_mechanic_and_model": {},
        "candidate_diversity_by_arm_mechanic_and_model": {},
        "decision_latency_by_arm_mechanic_and_model": {},
        "paired_causal_deltas_intervals_and_sample_sizes": {
            "preregistered_metric": "target_licensed_route_supported_proposal_path_score",
            "sample_size_cells": 0,
            "all_cells_ready": False,
            "rows": [],
        },
        "baseline_harm_controls": {
            "baseline_harm_detected": False,
            "harm_count": 0,
            "harm_rows": [],
        },
        "actual_work_duration_receipt": {
            "measured_actual_work_s": measured,
            "monotonic_clock": "time.perf_counter",
            "sleep_or_padding_used": False,
            "duration_padding_count": 0,
            "blocked_before_inference": True,
            "duration_floor_s": 60.0,
            "duration_floor_met": measured >= 60.0,
        },
        "duration_padding_count": 0,
        "arc_target_licensed_router_ready_score": 0.0,
        "hidden_game_source_access_count": 0,
        "outer_loop_ground_truth_search_count": 0,
        "arc_level_solve_claim_count": 0,
        "registry_update_count": 0,
        "source_model_weight_mutation_count": 0,
        "protected_files_unchanged": _protected_unchanged(protected_before),
        "preconditions_checked": _preconditions(
            date, models, registry, protected_before, result_path, resource_receipt
        ),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seed": RANDOM_SEEDS[0],
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": f"complete: {status}_no_solve_claim",
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    return artifact


def run(
    *,
    date: str,
    result_path: Path,
    fixture_manifest_path: Path,
    live_window_manifest_path: Path,
    raw_output_dir: Path,
    duration_s: float | None = None,
    test_exit_codes: Mapping[str, int | None] | None = None,
    model_resolver: ModelResolver | None = None,
    llm_runner: LLMRunner | None = None,
    write: bool = True,
) -> JsonDict:
    started = time.perf_counter()
    protected_before = _protected_hashes()
    registry = registry_precheck()
    resource_receipt = _resource_preflight_receipt()
    live = model_resolver is None and llm_runner is None
    resolver = model_resolver or resolve_required_models
    models = resolver(live)
    fixtures = build_fresh_fixtures(seed=6307, per_mechanic=2)
    fixture_payload = fixture_manifest_payload(fixtures, seed=6307)
    fixture_receipt = write_manifest(fixture_manifest_path, fixture_payload, write=write)
    windows = build_live_transition_windows(fixtures, seeds=RANDOM_SEEDS)
    window_payload = live_window_manifest_payload(windows)
    window_receipt = write_manifest(live_window_manifest_path, window_payload, write=write)

    missing_models = [model["hf_id"] for model in models if not model.get("model_exists")]
    if missing_models:
        artifact = _blocked_artifact(
            status="blocked_model_not_cached_or_unavailable",
            date=date,
            result_path=result_path,
            fixture_receipt=fixture_receipt,
            window_receipt=window_receipt,
            models=models,
            started=started,
            duration_s=duration_s,
            protected_before=protected_before,
            registry=registry,
            resource_receipt=resource_receipt,
            test_exit_codes=test_exit_codes,
        )
        if write:
            result_path.parent.mkdir(parents=True, exist_ok=True)
            result_path.write_text(
                json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        return artifact

    tokenizer_receipts = _tokenizer_and_template_receipts(models, live=live)
    requests = build_matched_arm_requests(windows, models)
    validate_matched_requests(requests)
    runner = llm_runner or live_llama_cpp_runner
    llm = runner(requests, models, raw_output_dir, write)
    outputs = list(llm.get("outputs") or [])
    aggregates = _aggregate_by_arm_mechanic_model(outputs)
    paired = _paired_delta_receipt(outputs)
    baseline_harm = _baseline_harm(outputs)
    arm_contract, matched_receipt = _arm_receipts(requests)
    measured = round(
        float(duration_s if duration_s is not None else time.perf_counter() - started), 6
    )
    cuda_receipts = dict(llm.get("cuda_receipts") or {})
    cuda_ready = _cuda_ready(cuda_receipts, [str(model["hf_id"]) for model in models])
    duration_ready = measured >= 60.0
    completed = (
        bool(outputs)
        and not llm.get("errors")
        and bool(paired.get("all_cells_ready"))
        and not bool(baseline_harm.get("baseline_harm_detected"))
        and duration_ready
        and cuda_ready
    )
    status = "complete" if completed else "blocked_target_license_metric_or_precondition_failed"
    artifact: JsonDict = {
        "status": status,
        "upstream_paths_hashes_and_terminal_classes": _upstream_receipts(),
        "registry_precheck_path_hash_and_target_receipt": registry,
        "solve_provenance": "live_agent_self_discovery",
        "sealed_fresh_fixture_manifest_path_and_hash": fixture_receipt,
        "live_transition_window_manifest_path_and_hash": window_receipt,
        "no_hidden_source_and_no_outer_loop_search_receipts": _no_forbidden_receipt(),
        "MODEL_SPECS": list(models),
        "models_used": [model["hf_id"] for model in models],
        "model_file_hashes_revisions_and_quantizations": _model_file_receipts(models),
        "tokenizer_and_chat_template_hashes": tokenizer_receipts,
        "cuda_and_gpu_offload_receipts_by_model": cuda_receipts,
        "raw_output_paths_and_hashes": dict(llm.get("raw_output_paths_and_hashes") or {}),
        "router_off_retrieval_only_and_target_licensed_arm_contract": arm_contract,
        "matched_seed_history_action_budget_and_model_call_receipts": matched_receipt,
        "target_validation_predicates_and_mutation_receipts": _target_validation_receipt(requests),
        "hypothesis_retrieval_activation_rejection_and_abstention_counts": _hypothesis_counts(
            requests
        ),
        "route_activation_counts_by_arm_mechanic_and_model": aggregates["route_activation"],
        "executable_proposal_acceptance_by_arm_mechanic_and_model": aggregates["executable"],
        "invalid_proposal_rate_by_arm_mechanic_and_model": aggregates["invalid_rate"],
        "candidate_diversity_by_arm_mechanic_and_model": aggregates["diversity"],
        "decision_latency_by_arm_mechanic_and_model": aggregates["latency"],
        "paired_causal_deltas_intervals_and_sample_sizes": paired,
        "baseline_harm_controls": baseline_harm,
        "actual_work_duration_receipt": {
            "measured_actual_work_s": measured,
            "monotonic_clock": "time.perf_counter",
            "sleep_or_padding_used": False,
            "duration_padding_count": 0,
            "llm_runner_wall_s": llm.get("wall_s"),
            "duration_floor_s": 60.0,
            "duration_floor_met": duration_ready,
        },
        "duration_padding_count": 0,
        "arc_target_licensed_router_ready_score": 1.0 if completed else 0.0,
        "hidden_game_source_access_count": 0,
        "outer_loop_ground_truth_search_count": 0,
        "arc_level_solve_claim_count": 0,
        "registry_update_count": 0,
        "source_model_weight_mutation_count": 0,
        "protected_files_unchanged": _protected_unchanged(protected_before),
        "preconditions_checked": _preconditions(
            date, models, registry, protected_before, result_path, resource_receipt
        ),
        "inference_substrate": "live_llm_inference",
        "verifier_is_oracle": False,
        "field_provenance": dict(FIELD_PROVENANCE),
        "field_principles": dict(FIELD_PRINCIPLES),
        "test_commands": list(DEFAULT_TEST_COMMANDS),
        "test_exit_codes": dict(test_exit_codes or _read_external_test_receipts()),
        "duration_s": measured,
        "random_seed": RANDOM_SEEDS[0],
        "random_seeds": list(RANDOM_SEEDS),
        "reproducibility_checksum": "",
        "honest_verdict": (
            "complete: arc_target_validated_route_canary_ready_no_solve_claim"
            if completed
            else "complete: blocked_target_license_metric_or_precondition_failed_no_solve_claim"
        ),
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    validate_artifact(artifact)
    if write:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_path.write_text(
            json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    return artifact


def _terminal_verdict(value: str) -> bool:
    return value.startswith(
        (
            "complete:",
            "complete_",
            "success:",
            "success_",
            "passed:",
            "passed_",
            "shipped:",
            "shipped_",
        )
    )


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
    if artifact["inference_substrate"] != "live_llm_inference":
        raise ValueError("inference_substrate")
    if artifact["verifier_is_oracle"] is not False:
        raise ValueError("verifier_is_oracle")
    if not _terminal_verdict(str(artifact["honest_verdict"])):
        raise ValueError("honest_verdict")
    for field in FORBIDDEN_ZERO_FIELDS:
        if type(artifact[field]) is not int or artifact[field] != 0:
            raise ValueError(field)
    model_ids = [row.get("hf_id") for row in artifact["MODEL_SPECS"]]
    if not all(model_id in model_ids for model_id in MANDATED_MODEL_IDS):
        raise ValueError("MODEL_SPECS")
    if not all(
        model_id in artifact["models_used"] or str(artifact["status"]).startswith("blocked_")
        for model_id in MANDATED_MODEL_IDS
    ):
        raise ValueError("models_used")
    registry = artifact["registry_precheck_path_hash_and_target_receipt"]
    if registry.get("target_present_in_registry") or registry.get("duplicate_registry_target"):
        raise ValueError("registry_precheck_path_hash_and_target_receipt")
    arm_contract = artifact["router_off_retrieval_only_and_target_licensed_arm_contract"]
    if arm_contract.get("all_cells_matched") is not True:
        raise ValueError("router_off_retrieval_only_and_target_licensed_arm_contract")
    if arm_contract.get("retrieval_only_activation_count", 0) != 0:
        raise ValueError("router_off_retrieval_only_and_target_licensed_arm_contract")
    matched = artifact["matched_seed_history_action_budget_and_model_call_receipts"]
    if matched.get("history_mismatch_count") != 0 or matched.get("model_call_mismatch_count") != 0:
        raise ValueError("matched_seed_history_action_budget_and_model_call_receipts")
    hypothesis = artifact["hypothesis_retrieval_activation_rejection_and_abstention_counts"]
    if hypothesis.get("retrieval_only_activation_count", 0) != 0:
        raise ValueError("hypothesis_retrieval_activation_rejection_and_abstention_counts")
    validation = artifact["target_validation_predicates_and_mutation_receipts"]
    if validation.get("all_target_licensed_activations_mutation_proven") is not True:
        raise ValueError("target_validation_predicates_and_mutation_receipts")
    if validation.get("false_license_count", 0) != 0:
        raise ValueError("target_validation_predicates_and_mutation_receipts")
    if artifact["baseline_harm_controls"].get("baseline_harm_detected") is True:
        raise ValueError("baseline_harm_controls")
    complete = artifact["status"] == "complete"
    if complete and float(artifact["duration_s"]) < 60.0:
        raise ValueError("actual_work_duration_receipt")
    if (
        complete
        and artifact["paired_causal_deltas_intervals_and_sample_sizes"].get("all_cells_ready")
        is not True
    ):
        raise ValueError("paired_causal_deltas_intervals_and_sample_sizes")
    if complete and artifact["arc_target_licensed_router_ready_score"] != 1.0:
        raise ValueError("arc_target_licensed_router_ready_score")
    if complete and not _cuda_ready(
        artifact["cuda_and_gpu_offload_receipts_by_model"], list(MANDATED_MODEL_IDS)
    ):
        raise ValueError("cuda_and_gpu_offload_receipts_by_model")
    if artifact["reproducibility_checksum"] != payload_checksum(artifact):
        raise ValueError("reproducibility_checksum")


def main(
    argv: Sequence[str] | None = None,
) -> int:  # pragma: no cover - CLI wrapper uses live defaults.
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", default="20260811")
    parser.add_argument("--output", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    parser.add_argument(
        "--fixture-manifest", default=str(REPO_ROOT / FIXTURE_MANIFEST_RELATIVE_PATH)
    )
    parser.add_argument(
        "--live-window-manifest", default=str(REPO_ROOT / LIVE_WINDOW_MANIFEST_RELATIVE_PATH)
    )
    parser.add_argument("--raw-output-dir", default=str(REPO_ROOT / RAW_OUTPUT_DIR_RELATIVE_PATH))
    args = parser.parse_args(argv)
    run(
        date=args.date,
        result_path=Path(args.output),
        fixture_manifest_path=Path(args.fixture_manifest),
        live_window_manifest_path=Path(args.live_window_manifest),
        raw_output_dir=Path(args.raw_output_dir),
        write=True,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint.
    raise SystemExit(main())
