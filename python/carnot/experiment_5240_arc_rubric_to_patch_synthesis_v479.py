"""Exp 5240: synthesize one ARC live-path patch candidate from upstream evidence.

Spec refs: REQ-REPORT-5240, SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS,
SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from carnot.agentic import arc_solve_learning
from carnot.agentic import arc_typed_memory_provenance_guard as memory_guard


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5240_arc_rubric_to_patch_synthesis_v479"
EXPERIMENT_ID = 5240
SCHEMA = "carnot.arc_rubric_to_patch_synthesis_v479.v1"
RUN_DATE = "2026-07-04"
RESULT_RELATIVE_PATH = "results/experiment_5240_arc_rubric_to_patch_synthesis_v479.json"
PATCH_RELATIVE_PATH = "python/carnot/agentic/arc_solve_learning.py"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_REFS = (
    "REQ-REPORT-5240",
    "SCENARIO-REPORT-5240-LIVE-PATCH-SYNTHESIS",
    "SCENARIO-REPORT-5240-NO-PATCH-WITHOUT-EVIDENCE",
)
SOURCE_ARTIFACTS = (
    memory_guard.EXP5228_RUBRIC_GATE,
    memory_guard.EXP5239_MEMORY_ABLATION,
    REGISTRY_RELATIVE_PATH,
)
REQUIRED_ARTIFACT_FIELDS = (
    "recommended_live_patch_available",
    "patch_test_ready",
    "patch_path",
    "patch_failure_mode_targeted",
    "registry_precheck_done",
    "duplicate_solve_target_avoided",
    "live_agent_reachability_evidence",
    "model_specs",
    "inference_substrate",
    "duration_s",
    "honest_verdict",
)
FIELD_PRINCIPLES = {
    "recommended_live_patch_available": (
        "BARE top-level boolean for Exp 5241. True only when one evidence-backed "
        "live-path patch candidate exists."
    ),
    "patch_test_ready": (
        "BARE top-level boolean for Exp 5241. True only when a test or dry-run "
        "receipt proves the live agent can reach the candidate."
    ),
    "patch_path": "Path to the code or configuration patch, or null when no patch is justified.",
    "patch_failure_mode_targeted": (
        "One of skill_selection, skill_following, composition, reflection, "
        "provenance_routing, or none."
    ),
    "registry_precheck_done": (
        "True only when ops/arc_solve_registry.yaml was read before patch recommendation."
    ),
    "duplicate_solve_target_avoided": (
        "True only when the synthesis does not target an already reached live solve depth."
    ),
    "live_agent_reachability_evidence": "Test command, dry-run receipt, or null.",
    "model_specs": (
        "MODEL_SPECS with mandated SOTA GGUF if any LLM proposer was used; otherwise null."
    ),
    "inference_substrate": "Must be aggregation_from_upstream_artifacts.",
    "duration_s": (
        "Real measured wall-clock time of the aggregation pass (registry read + "
        "typed-memory-guard read + verdict computation); the aggregation_from_"
        "upstream_artifacts substrate floor is 0.0001s, per CLAUDE.md's "
        "Inference-Substrate Declaration Discipline."
    ),
    "honest_verdict": (
        "Must start with complete:/complete_/success:/success_ and state whether a "
        "live patch is gated for exp5241."
    ),
}


def load_registry_summary(root: Path | str = REPO_ROOT) -> JsonDict:
    """Load the public ARC solve registry summary without mutating it."""

    path = Path(root) / REGISTRY_RELATIVE_PATH
    if not path.exists():
        return {
            "present": False,
            "path": REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels": 0,
            "games": {},
        }
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {
            "present": False,
            "path": REGISTRY_RELATIVE_PATH,
            "reproducible_total_levels": 0,
            "games": {},
        }
    games = {
        str(row.get("game")): int(row.get("levels_reproduced") or 0)
        for row in data.get("games", []) or []
        if isinstance(row, Mapping) and row.get("game")
    }
    return {
        "present": True,
        "path": REGISTRY_RELATIVE_PATH,
        "reproducible_total_levels": int(data.get("reproducible_total_levels") or 0),
        "games": games,
    }


def build_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the Exp 5240 synthesis artifact from upstream evidence."""

    started = time.perf_counter()
    registry = load_registry_summary(root)
    guard = arc_solve_learning.typed_memory_provenance_guard(root=root)
    recommended = bool(guard.get("enabled") and registry.get("present"))
    duplicate_avoided = bool(registry.get("present"))
    test_receipts = [dict(item) for item in tests_run]
    tests_passed = bool(test_receipts) and all(bool(item.get("passed")) for item in test_receipts)
    patch_test_ready = bool(recommended and tests_passed)
    reachability = _reachability_evidence(test_receipts) if patch_test_ready else None
    failure_mode = str(guard.get("failure_mode_targeted") or "none") if recommended else "none"
    patch_path = PATCH_RELATIVE_PATH if recommended else None
    verdict = (
        "success: provenance-routing live patch is gated for exp5241 without a solve claim."
        if patch_test_ready
        else "complete: no evidence-backed live patch is gated for exp5241."
    )
    duration_s = time.perf_counter() - started
    return {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "run_date": RUN_DATE,
        "spec_refs": list(SPEC_REFS),
        "recommended_live_patch_available": recommended,
        "patch_test_ready": patch_test_ready,
        "patch_path": patch_path,
        "patch_failure_mode_targeted": failure_mode,
        "registry_precheck_done": bool(registry.get("present")),
        "duplicate_solve_target_avoided": duplicate_avoided,
        "live_agent_reachability_evidence": reachability,
        "model_specs": None,
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "duration_s": duration_s,
        "honest_verdict": verdict,
        "field_principles": dict(FIELD_PRINCIPLES),
        "patch_candidate": {
            "kind": "narrow_live_routing_rule" if recommended else "none",
            "summary": guard.get("routing_rule") if recommended else None,
            "guard": guard,
        },
        "registry_summary": registry,
        "source_artifacts": list(SOURCE_ARTIFACTS),
        "tests_run": test_receipts,
        "llm_proposer_used": False,
        "read_hidden_game_source": False,
        "offline_ground_truth_bfs": False,
        "hand_per_game_adapter_created": False,
        "arc_level_solve_claimed": False,
    }


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp 5240 synthesis artifact."""

    artifact = build_artifact(root=root, tests_run=tests_run)
    destination = Path(result_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def _reachability_evidence(tests_run: Sequence[Mapping[str, Any]]) -> str | None:
    commands = [str(item.get("command") or "").strip() for item in tests_run]
    commands = [command for command in commands if command]
    return "; ".join(commands) if commands else None


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(REPO_ROOT))
    parser.add_argument("--result-path", default=str(REPO_ROOT / RESULT_RELATIVE_PATH))
    args = parser.parse_args(argv)
    artifact = run(root=args.root, result_path=args.result_path)
    print(json.dumps({field: artifact[field] for field in REQUIRED_ARTIFACT_FIELDS}, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
