"""Experiment 5641: counterexample-patched executable ARC transition model.

Spec refs: REQ-ARC-WMTE-5641,
SCENARIO-ARC-WMTE-5641-COUNTEREXAMPLE-PATCH-REPLAY,
SCENARIO-ARC-WMTE-5641-CONTROLS-AND-ABSTENTION.

The experiment is a development proxy over already reproduced public-level
transition receipts. It does not read game source, use a per-game adapter, run
offline BFS, invoke an LLM, or claim solve credit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from carnot.agentic.arc_counterexample_executable_model import (
    PATCH_OPERATORS,
    TransitionReceipt,
    context_features,
    hypothesis_language_spec,
    make_unsupported_receipt,
    run_chronological_evaluation,
)


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5641_arc_counterexample_executable_model"
EXPERIMENT_ID = 5641
RESULT_RELATIVE_PATH = f"results/{EXPERIMENT}.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
INFERENCE_SUBSTRATE = "deterministic_counterexample_patched_executable_model"
SOLVE_PROVENANCE = "development_proxy"
DEFAULT_ROSTER = ("dc22", "bp35", "s5i5")
RANDOM_SEEDS = [5641]

REQUIRED_ARTIFACT_FIELDS = (
    "field_principles",
    "registry_precheck_receipt",
    "evaluation_levels",
    "solve_provenance",
    "agent_owned_evidence_only",
    "source_read",
    "game_adapter_used",
    "offline_ground_truth_bfs_used",
    "model_specs",
    "hypothesis_language",
    "patch_operator_set",
    "counterexample_count",
    "accepted_patch_count",
    "rejected_patch_count",
    "all_receipt_replay_pass",
    "heldout_transition_error_by_arm",
    "abstention_calibration",
    "mechanism_question_controls",
    "unsafe_patch_accept_count",
    "executable_model_ready_score",
    "inference_substrate",
    "random_seeds",
    "reproducibility_checksum",
    "honest_verdict",
    "source_access_guards",
)

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "field_principles": {
        "principle": "principle annotations make every required 5641 field auditable.",
    },
    "registry_precheck_receipt": {
        "principle": "development levels are already reproduced; no solve credit is claimed.",
    },
    "evaluation_levels": {
        "principle": "the development scope is fixed before trace loading.",
    },
    "solve_provenance": {
        "principle": "development_proxy -- the artifact evaluates a known-level method, not a level solve.",
    },
    "agent_owned_evidence_only": {
        "principle": "evidence provenance is credited to live-agent receipts only.",
    },
    "source_read": {
        "principle": "false excludes game source from the hypothesis and evaluation path.",
    },
    "game_adapter_used": {
        "principle": "false excludes per-game transition models.",
    },
    "offline_ground_truth_bfs_used": {
        "principle": "false excludes outer-loop exhaustive search labels.",
    },
    "model_specs": {
        "principle": "empty list because no LLM participates.",
    },
    "hypothesis_language": {
        "principle": "typed executable state is inspectable and excludes game constants.",
    },
    "patch_operator_set": {
        "principle": "revision is bounded to add, specialize, relax, and retire.",
    },
    "counterexample_count": {
        "principle": "every revision has falsifying transition evidence.",
    },
    "accepted_patch_count": {
        "principle": "successful revisions are auditable.",
    },
    "rejected_patch_count": {
        "principle": "fail-closed behavior is visible.",
    },
    "all_receipt_replay_pass": {
        "principle": "accepted theory has no wrong prediction on accumulated receipts.",
    },
    "heldout_transition_error_by_arm": {
        "principle": "utility is measured against generic controls.",
    },
    "abstention_calibration": {
        "principle": "unsupported cases are bounded and reported.",
    },
    "mechanism_question_controls": {
        "principle": "adaptive questions are auditable tests only.",
    },
    "unsafe_patch_accept_count": {
        "principle": "contradictions cannot publish as accepted patches.",
    },
    "executable_model_ready_score": {
        "principle": "downstream gating is scalar and mechanical.",
    },
    "inference_substrate": {
        "principle": "deterministic_counterexample_patched_executable_model -- no LLM or source participated.",
    },
    "random_seeds": {
        "principle": "development proxy replays deterministically.",
    },
    "reproducibility_checksum": {
        "principle": "development proxy inputs and decisions are content-addressed.",
    },
    "honest_verdict": {
        "principle": "terminal prefix records complete or blocked status; nulls retire the mechanism.",
    },
    "source_access_guards": {
        "principle": "guards are recorded before loading traces and exclude source, adapters, BFS, and LLMs.",
    },
}


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)


def _sha256(value: Any) -> str:
    return "sha256:" + hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def read_yaml(path: Path) -> JsonDict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def source_access_guard(*, recorded_before_trace_load: bool) -> JsonDict:
    return {
        "recorded_before_trace_load": bool(recorded_before_trace_load),
        "agent_owned_evidence_roots": ["data/arc_transition_corpus"],
        "allowed_metadata": ["ops/arc_solve_registry.yaml"],
        "forbidden_path_fragments": [
            "environment_files",
            "arc_game_adapters.py",
            "scripts/arc_loop_solve.py",
            "outer_loop",
            "game.py",
        ],
        "source_read": False,
        "game_adapter_used": False,
        "offline_ground_truth_bfs_used": False,
        "model_specs": [],
    }


def path_allowed_by_source_guard(path: str | Path, guard: Mapping[str, Any]) -> bool:
    normalized = str(path).replace("\\", "/")
    forbidden = guard.get("forbidden_path_fragments", [])
    return not any(str(fragment) in normalized for fragment in forbidden)


def _registry_rows(registry: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    rows = registry.get("games", {})
    if isinstance(rows, Mapping):
        return {str(game): row for game, row in rows.items() if isinstance(row, Mapping)}
    return {
        str(row.get("game")): row for row in rows if isinstance(row, Mapping) and row.get("game")
    }


def _is_reproduced(row: Mapping[str, Any] | None) -> bool:
    if not row:
        return False
    if str(row.get("reproducibility", "")).lower() == "reproduced":
        return True
    return bool(row.get("offline_reproduced")) or int(row.get("levels_reproduced") or 0) > 0


def registry_precheck(
    roster: Sequence[str],
    registry: Mapping[str, Any],
    transitions_by_game: Mapping[str, Sequence[TransitionReceipt]],
    guard: Mapping[str, Any],
) -> JsonDict:
    rows = _registry_rows(registry)
    selected: list[str] = []
    roster_rows: list[JsonDict] = []
    for game in roster:
        reg = rows.get(str(game))
        reproduced = _is_reproduced(reg)
        usable_count = sum(1 for row in transitions_by_game.get(str(game), ()) if _usable(row))
        usable = bool(reproduced and usable_count >= 4)
        if usable:
            selected.append(str(game))
        roster_rows.append(
            {
                "game": str(game),
                "registry_reproduced": bool(reproduced),
                "levels_reproduced": int((reg or {}).get("levels_reproduced") or 0),
                "agent_owned_transition_count": int(usable_count),
                "selected": bool(usable),
            }
        )
    return {
        "ok": len(selected) >= 1,
        "only_already_reproduced_levels": all(
            row["registry_reproduced"] for row in roster_rows if row["selected"]
        ),
        "development_levels_are_already_reproduced": True,
        "no_solve_credit_claimed": True,
        "source_access_guard_recorded_before_trace_load": bool(
            guard.get("recorded_before_trace_load")
        ),
        "selected_games": selected,
        "roster_rows": roster_rows,
        "registry_total_levels": int(registry.get("reproducible_total_levels") or 0),
    }


def load_agent_owned_receipts(
    roster: Sequence[str] = DEFAULT_ROSTER,
    *,
    guard: Mapping[str, Any],
    corpus_root: Path | None = None,
    max_per_game: int = 120,
) -> dict[str, list[TransitionReceipt]]:
    from carnot.agentic.arc_transition_capture import TransitionCorpus

    if not guard.get("recorded_before_trace_load"):
        raise ValueError("source access guard must be recorded before trace loading")
    corpus = TransitionCorpus(root=corpus_root)
    out: dict[str, list[TransitionReceipt]] = {}
    for game in roster:
        relative = f"data/arc_transition_corpus/{game}.npz"
        if not path_allowed_by_source_guard(relative, guard):
            raise ValueError(f"trace path rejected by source guard: {relative}")
        rows: list[TransitionReceipt] = []
        for index, transition in enumerate(corpus.load(str(game))):
            state = np.asarray(transition.grid, dtype=np.int16)
            successor = np.asarray(transition.next_grid, dtype=np.int16)
            if state.shape != successor.shape or not np.any(state != successor):
                continue
            reward = int(getattr(transition, "level_after", 0)) - int(
                getattr(transition, "level_before", 0)
            )
            receipt = TransitionReceipt(
                trace_id=str(game),
                episode=f"{game}-episode-{index // 16}",
                step=index,
                state=state,
                action=int(transition.action),
                data=transition.data,
                successor=successor,
                reward=reward,
                terminal=reward > 0,
                provenance="agent_owned_runtime_observation",
            )
            if _usable(receipt):
                rows.append(receipt)
            if len(rows) >= max_per_game:
                break
        out[str(game)] = rows
    return out


def build_artifact(
    *,
    transitions_by_game: Mapping[str, Sequence[TransitionReceipt]] | None = None,
    registry: Mapping[str, Any] | None = None,
    roster: Sequence[str] = DEFAULT_ROSTER,
    random_seed: int = 5641,
    root: Path = REPO_ROOT,
) -> JsonDict:
    guard = source_access_guard(recorded_before_trace_load=True)
    registry_data = dict(registry or read_yaml(root / REGISTRY_RELATIVE_PATH))
    traces = (
        {str(game): list(rows) for game, rows in transitions_by_game.items()}
        if transitions_by_game is not None
        else load_agent_owned_receipts(roster, guard=guard)
    )
    if not roster and transitions_by_game is not None:
        roster = tuple(sorted(traces))
    precheck = registry_precheck(roster, registry_data, traces, guard)
    selected = list(precheck["selected_games"])
    selected_traces = {game: list(traces.get(game, ())) for game in selected}
    unsupported_controls = [
        make_unsupported_receipt(rows[0])
        for rows in selected_traces.values()
        if rows
    ]
    evaluation = run_chronological_evaluation(
        selected_traces,
        unsupported_receipts=unsupported_controls,
        random_seed=random_seed,
    )
    error_interval = evaluation.abstention_calibration[
        "patched_vs_unpatched_error_reduction_interval"
    ]
    error_or_calibration_pass = float(error_interval["lower"]) > 0.0
    informative = float(evaluation.mechanism_question_controls["informative"]["score"])
    irrelevant = float(evaluation.mechanism_question_controls["irrelevant"]["score"])
    unsupported_abstain = float(
        evaluation.abstention_calibration["unsupported_abstention_rate"]
    ) == 1.0
    ready = float(
        bool(selected)
        and error_or_calibration_pass
        and bool(evaluation.all_receipt_replay_pass)
        and informative > irrelevant
        and unsupported_abstain
        and int(evaluation.unsafe_patch_accept_count) == 0
    )
    verdict = (
        "complete: counterexample_patched_executable_model_ready_development_proxy"
        if ready == 1.0
        else "blocked: counterexample_patched_executable_model_retired_terminal"
    )
    evaluation_levels = []
    by_game = _registry_rows(registry_data)
    for game in selected:
        row = by_game.get(game, {})
        evaluation_levels.append(
            {
                "game": game,
                "level_scope": f"L<= {int(row.get('levels_reproduced') or 0)}",
                "agent_owned_receipts": len(selected_traces.get(game, ())),
            }
        )
    artifact: JsonDict = {
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "schema": "carnot.exp5641.arc_counterexample_executable_model.v1",
        "date": "20260714",
        "field_principles": FIELD_PRINCIPLES,
        "registry_precheck_receipt": precheck,
        "evaluation_levels": evaluation_levels,
        "source_access_guards": guard,
        "solve_provenance": SOLVE_PROVENANCE,
        "agent_owned_evidence_only": True,
        "source_read": False,
        "game_adapter_used": False,
        "offline_ground_truth_bfs_used": False,
        "model_specs": [],
        "hypothesis_language": hypothesis_language_spec(),
        "patch_operator_set": list(PATCH_OPERATORS),
        "counterexample_count": int(evaluation.counterexample_count),
        "accepted_patch_count": int(evaluation.accepted_patch_count),
        "rejected_patch_count": int(evaluation.rejected_patch_count),
        "all_receipt_replay_pass": bool(evaluation.all_receipt_replay_pass),
        "heldout_transition_error_by_arm": evaluation.heldout_transition_error_by_arm,
        "abstention_calibration": evaluation.abstention_calibration,
        "mechanism_question_controls": evaluation.mechanism_question_controls,
        "adversarial_controls": evaluation.patch_controls,
        "unsafe_patch_accept_count": int(evaluation.unsafe_patch_accept_count),
        "executable_model_ready_score": ready,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "random_seeds": [int(random_seed)],
        "honest_verdict": verdict,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def payload_checksum(payload: Mapping[str, Any]) -> str:
    core = {key: value for key, value in payload.items() if key != "reproducibility_checksum"}
    return _sha256(core)


def _usable(row: TransitionReceipt) -> bool:
    return context_features(row).supported and int(row.effect_signature.get("changed_count", 0)) > 0


def main() -> None:
    write_json(REPO_ROOT / RESULT_RELATIVE_PATH, build_artifact())


if __name__ == "__main__":
    main()
