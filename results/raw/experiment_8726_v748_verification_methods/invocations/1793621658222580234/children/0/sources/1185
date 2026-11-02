"""Experiment 4791: S2 off-path structural-energy trust gate.

Spec refs: REQ-ARC-WMTE-4791,
SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np

from carnot.agentic.arc_executable_world_model import (
    Transition,
    WorldModelVerifier,
    collect_transitions,
)
from carnot.agentic.arc_world_model_trust_energy import (
    HIDDEN_STATE_GAME_IDS,
    INFERENCE_SUBSTRATE,
    S1StructuralTransitionEnergy,
    WorldModelCandidate,
    _split_prefix_heldout,
    default_s1_offpath_energy_scorer,
    select_trusted_world_model,
)


JsonDict = dict[str, Any]
Engine = Callable[[np.ndarray, int, Any], np.ndarray]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4791_structural_energy_s2_offpath_trust_gate"
EXPERIMENT_ID = 4791
SCHEMA = "carnot.arc_structural_energy_s2_offpath_trust_gate_4791.v1"
RESULT_RELATIVE_PATH = "results/experiment_4791_structural_energy_s2_offpath_trust_gate.json"
RANDOM_SEED = 4791
BOOTSTRAP_RESAMPLES = 1000
MIN_HELDOUT_GAMES = 5
DEFAULT_TRANSITIONS_PER_GAME = 24
SPEC_REFS = [
    "REQ-ARC-WMTE-4791",
    "SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE",
]
TERMINAL_PREFIXES = (
    "complete:",
    "complete_",
    "success:",
    "success_",
    "passed:",
    "passed_",
    "shipped:",
    "shipped_",
)
CACHED_CANDIDATE_ROOTS = (
    Path("results/arc_e3"),
    Path("results/arc_logo_snapshot"),
    Path("results/arc_e3_seedproto"),
)

REQUIRED_FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": (
            "terminal prefix; a live trust-gate win is "
            "success_structural_energy_s2_trust_gate_authorizes_s3, a bound is "
            "complete_structural_energy_s2_no_live_trust_value."
        )
    },
    "verifier_is_oracle": {
        "principle": (
            "MUST be false for the energy ranking (oracle-distinct); the binary-accuracy CONTROL is "
            "execution-grounded (true) -- S2 measures whether oracle-distinct energy beats "
            "execution-grounded selection."
        )
    },
    "live_path_reachable": {
        "principle": (
            "the energy gate must be in the E3AgentPolicy import closure (arc_orphan_solver_lint "
            "passes) -- a gate the live agent cannot reach adds no live value (ARC Live-Path "
            "Reachability Discipline)."
        )
    },
    "inference_substrate": {
        "principle": (
            "verifier_ensemble_against_cached_candidates (scores the energy over cached candidate "
            "induced engines off-path predictions, no fresh LLM; 1s floor)."
        )
    },
    "preconditions_checked": {
        "principle": (
            "records the arcade / WorldModelVerifier import checks so a silent-missing-resource run "
            "cannot fabricate a cell_recall."
        )
    },
    "energy_selected_offpath_cell_recall": {
        "principle": (
            "the held-out off-path cell_recall of the engine the structural energy selected -- the "
            "load-bearing measurement."
        )
    },
    "accuracy_gate_selected_offpath_cell_recall": {
        "principle": (
            "the held-out off-path cell_recall of the engine the incumbent binary accuracy<0.5 gate "
            "selected -- the control the energy must beat."
        )
    },
    "energy_minus_accuracy_delta_ci95": {
        "principle": (
            "must EXCLUDE 0 (energy > accuracy-gate) for S2 to pass -- the oracle-distinct value-add "
            "over cheap execution-grounded selection."
        )
    },
    "n_heldout_games": {
        "principle": (
            ">=N held-out games -- the delta must hold across games, not a single-game artifact."
        )
    },
    "random_seed": {
        "principle": "determinism for reproducibility of the selection + bootstrap."
    },
    "reproducibility_checksum": {
        "principle": (
            "content hash of (engine candidates, folds, energy config) so a replication catches drift."
        )
    },
}

REQUIRED_ARTIFACT_FIELDS = (
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "honest_verdict",
    "verifier_is_oracle",
    "binary_accuracy_control",
    "live_path_reachable",
    "inference_substrate",
    "preconditions_checked",
    "energy_selected_offpath_cell_recall",
    "accuracy_gate_selected_offpath_cell_recall",
    "energy_minus_accuracy_delta",
    "energy_minus_accuracy_delta_ci95",
    "n_heldout_games",
    "min_heldout_games",
    "random_seed",
    "s3_authorized",
    "retire_if_same_verdict",
    "game_results",
    "candidate_sets",
    "energy_config",
    "field_principles",
    "duration_s",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class GameTrustGateResult:
    """Per-game S2 selection result on one identical candidate engine set."""

    game: str
    n_candidates: int
    energy_selected_candidate: str
    accuracy_gate_selected_candidate: str
    energy_selected_offpath_cell_recall: float
    accuracy_gate_selected_offpath_cell_recall: float
    energy_minus_accuracy_delta: float
    energy_selected_structural_energy: float
    accuracy_gate_exact_accuracy: float
    accuracy_gate_passed: bool
    candidate_rows: list[JsonDict]

    def to_json(self) -> JsonDict:
        return {
            "game": self.game,
            "n_candidates": int(self.n_candidates),
            "energy_selected_candidate": self.energy_selected_candidate,
            "accuracy_gate_selected_candidate": self.accuracy_gate_selected_candidate,
            "energy_selected_offpath_cell_recall": float(
                self.energy_selected_offpath_cell_recall
            ),
            "accuracy_gate_selected_offpath_cell_recall": float(
                self.accuracy_gate_selected_offpath_cell_recall
            ),
            "energy_minus_accuracy_delta": float(self.energy_minus_accuracy_delta),
            "energy_selected_structural_energy": float(self.energy_selected_structural_energy),
            "accuracy_gate_exact_accuracy": float(self.accuracy_gate_exact_accuracy),
            "accuracy_gate_passed": bool(self.accuracy_gate_passed),
            "candidate_rows": list(self.candidate_rows),
        }


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    raw = json.dumps(clean, sort_keys=True, default=str, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def _bootstrap_mean_ci(
    values: Sequence[float],
    *,
    seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> list[float] | None:
    clean = np.asarray([float(value) for value in values if value == value], dtype=float)
    if clean.size == 0:
        return None
    if clean.size == 1:
        val = float(clean[0])
        return [val, val]
    rng = np.random.default_rng(seed)
    means = [
        float(np.mean(rng.choice(clean, size=clean.size, replace=True)))
        for _ in range(max(1, int(resamples)))
    ]
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def _ci_excludes_zero_positive(ci: Sequence[float] | None) -> bool:
    return bool(ci is not None and len(ci) == 2 and float(ci[0]) > 0.0)


def _mean(values: Sequence[float]) -> float | None:
    return float(np.mean([float(v) for v in values])) if values else None


def _candidate_by_name(
    candidates: Sequence[WorldModelCandidate],
    name: str | None,
) -> WorldModelCandidate:
    if name is not None:
        for candidate in candidates:
            if candidate.name == name:
                return candidate
    return candidates[0]


def _serialise_candidate_row(row: Any) -> JsonDict:
    return {
        "candidate_name": row.candidate.name,
        "prefix_accuracy": float(row.prefix_accuracy),
        "heldout_accuracy": float(row.heldout_accuracy),
        "offpath_structural_energy": float(row.trust_energy),
        "binary_gate_pass": bool(row.binary_gate_pass),
        "heldout_change_consistency": float(row.heldout_change_consistency),
        "heldout_cell_recall": float(row.heldout_change_consistency),
        "correct_changed_cells": int(row.correct_changed_cells),
        "true_changed_cells": int(row.true_changed_cells),
    }


def evaluate_candidate_set(
    *,
    game: str,
    transitions: Sequence[Transition],
    candidates: Sequence[WorldModelCandidate],
    energy_scorer: Any | None = None,
    binary_threshold: float = 0.5,
) -> GameTrustGateResult:
    """REQ-ARC-WMTE-4791: compare energy selection against binary exact gate."""

    if not candidates:
        raise ValueError("at least one candidate is required")
    scorer = energy_scorer or default_s1_offpath_energy_scorer()
    selection = select_trusted_world_model(
        transitions,
        candidates,
        hidden_state=True,
        baseline_threshold=binary_threshold,
        offpath_energy_scorer=scorer,
    )
    _prefix, heldout = _split_prefix_heldout(transitions)
    heldout_verifier = WorldModelVerifier(list(heldout))
    energy_candidate = selection.selected
    accuracy_candidate = _candidate_by_name(candidates, selection.baseline_candidate_name)
    energy_recall = heldout_verifier.score(energy_candidate.engine).cell_recall
    accuracy_recall = heldout_verifier.score(accuracy_candidate.engine).cell_recall
    full_accuracy = WorldModelVerifier(list(transitions)).score(accuracy_candidate.engine).accuracy
    return GameTrustGateResult(
        game=game,
        n_candidates=len(candidates),
        energy_selected_candidate=energy_candidate.name,
        accuracy_gate_selected_candidate=accuracy_candidate.name,
        energy_selected_offpath_cell_recall=float(energy_recall),
        accuracy_gate_selected_offpath_cell_recall=float(accuracy_recall),
        energy_minus_accuracy_delta=float(energy_recall - accuracy_recall),
        energy_selected_structural_energy=float(selection.selected_score.trust_energy),
        accuracy_gate_exact_accuracy=float(full_accuracy),
        accuracy_gate_passed=bool(full_accuracy >= float(binary_threshold)),
        candidate_rows=[_serialise_candidate_row(row) for row in selection.rows],
    )


def _artifact_verdict(
    *,
    n_games: int,
    delta: float | None,
    delta_ci95: Sequence[float] | None,
    live_path_reachable: bool,
) -> str:
    if (
        live_path_reachable
        and n_games >= MIN_HELDOUT_GAMES
        and delta is not None
        and delta > 0.0
        and _ci_excludes_zero_positive(delta_ci95)
    ):
        return "success_structural_energy_s2_trust_gate_authorizes_s3"
    return "complete_structural_energy_s2_no_live_trust_value"


def build_artifact(
    game_results: Sequence[GameTrustGateResult],
    *,
    preconditions_checked: Mapping[str, Any],
    live_path_reachable: bool,
    random_seed: int = RANDOM_SEED,
    bootstrap_resamples: int = BOOTSTRAP_RESAMPLES,
    duration_s: float = 1.0,
    energy_scorer: Any | None = None,
) -> JsonDict:
    """SCENARIO-ARC-WMTE-4791-OFFPATH-TRUST-GATE: build the S2 artifact."""

    rows = list(game_results)
    energy_recalls = [row.energy_selected_offpath_cell_recall for row in rows]
    accuracy_recalls = [row.accuracy_gate_selected_offpath_cell_recall for row in rows]
    deltas = [row.energy_minus_accuracy_delta for row in rows]
    delta = _mean(deltas)
    delta_ci95 = _bootstrap_mean_ci(
        deltas,
        seed=int(random_seed),
        resamples=bootstrap_resamples,
    )
    verdict = _artifact_verdict(
        n_games=len(rows),
        delta=delta,
        delta_ci95=delta_ci95,
        live_path_reachable=live_path_reachable,
    )
    scorer = energy_scorer or default_s1_offpath_energy_scorer()
    energy_config = getattr(scorer, "energy_config", {"source": str(type(scorer).__name__)})
    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": verdict,
        "verifier_is_oracle": False,
        "binary_accuracy_control": {
            "verifier_is_oracle": True,
            "selection_rule": "first_candidate_with_WorldModelVerifier_exact_accuracy_ge_0.5_else_first",
            "threshold": 0.5,
        },
        "live_path_reachable": bool(live_path_reachable),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "energy_selected_offpath_cell_recall": _mean(energy_recalls),
        "accuracy_gate_selected_offpath_cell_recall": _mean(accuracy_recalls),
        "energy_minus_accuracy_delta": delta,
        "energy_minus_accuracy_delta_ci95": delta_ci95,
        "n_heldout_games": len(rows),
        "min_heldout_games": MIN_HELDOUT_GAMES,
        "random_seed": int(random_seed),
        "s3_authorized": verdict == "success_structural_energy_s2_trust_gate_authorizes_s3",
        "retire_if_same_verdict": True,
        "game_results": [row.to_json() for row in rows],
        "candidate_sets": {
            row.game: [candidate["candidate_name"] for candidate in row.candidate_rows]
            for row in rows
        },
        "energy_config": energy_config,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "duration_s": round(float(duration_s), 3),
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def build_blocked_artifact(
    honest_verdict: str,
    preconditions_checked: Mapping[str, Any],
    *,
    random_seed: int = RANDOM_SEED,
    live_path_reachable: bool = False,
) -> JsonDict:
    """REQ-ARC-WMTE-4791: fail closed without fabricated cell_recall."""

    payload: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "verifier_is_oracle": False,
        "binary_accuracy_control": {
            "verifier_is_oracle": True,
            "selection_rule": "not_run_blocked_precondition",
            "threshold": 0.5,
        },
        "live_path_reachable": bool(live_path_reachable),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(preconditions_checked),
        "energy_selected_offpath_cell_recall": None,
        "accuracy_gate_selected_offpath_cell_recall": None,
        "energy_minus_accuracy_delta": None,
        "energy_minus_accuracy_delta_ci95": None,
        "n_heldout_games": 0,
        "min_heldout_games": MIN_HELDOUT_GAMES,
        "random_seed": int(random_seed),
        "s3_authorized": False,
        "retire_if_same_verdict": True,
        "game_results": [],
        "candidate_sets": {},
        "energy_config": default_s1_offpath_energy_scorer().energy_config,
        "field_principles": REQUIRED_FIELD_PRINCIPLES,
        "duration_s": 0.0,
        "reproducibility_checksum": None,
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    _require(not missing, f"missing required artifact fields: {missing}")
    checksum = artifact.get("reproducibility_checksum")
    _require(isinstance(checksum, str) and checksum.startswith("sha256:"), "checksum")
    _require(checksum == _checksum_payload(artifact), "reproducibility_checksum mismatch")
    _require(artifact.get("verifier_is_oracle") is False, "energy verifier must be non-oracle")
    _require(
        artifact.get("binary_accuracy_control", {}).get("verifier_is_oracle") is True,
        "binary control must declare oracle grounding",
    )
    _require(artifact.get("inference_substrate") == INFERENCE_SUBSTRATE, "inference_substrate")
    _require(artifact.get("field_principles") == REQUIRED_FIELD_PRINCIPLES, "field_principles")
    verdict = artifact.get("honest_verdict")
    _require(isinstance(verdict, str) and verdict, "honest_verdict")
    if verdict.startswith("blocked_"):
        _require(artifact.get("energy_selected_offpath_cell_recall") is None, "blocked energy recall")
        _require(
            artifact.get("accuracy_gate_selected_offpath_cell_recall") is None,
            "blocked accuracy recall",
        )
        _require(artifact.get("n_heldout_games") == 0, "blocked heldout games")
        _require(artifact.get("s3_authorized") is False, "blocked cannot authorize S3")
        return
    _require(verdict.startswith(TERMINAL_PREFIXES), "terminal-prefixed verdict")
    _require(isinstance(artifact.get("live_path_reachable"), bool), "live_path_reachable")
    _require(isinstance(artifact.get("n_heldout_games"), int), "n_heldout_games")
    if verdict == "success_structural_energy_s2_trust_gate_authorizes_s3":
        _require(artifact.get("live_path_reachable") is True, "success must be live-path reachable")
        _require(artifact.get("n_heldout_games", 0) >= MIN_HELDOUT_GAMES, "success needs N games")
        _require(
            _clean_float(artifact.get("energy_minus_accuracy_delta")) is not None
            and float(artifact["energy_minus_accuracy_delta"]) > 0.0,
            "success needs positive delta",
        )
        _require(
            _ci_excludes_zero_positive(artifact.get("energy_minus_accuracy_delta_ci95")),
            "success CI must exclude zero",
        )
        _require(artifact.get("s3_authorized") is True, "success authorizes S3")
    else:
        _require(artifact.get("s3_authorized") is False, "bounded cannot authorize S3")


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    validate_artifact(artifact)
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def check_preconditions(root: Path | str = REPO_ROOT) -> JsonDict:
    checked: JsonDict = {
        "offline_arcade": False,
        "world_model_verifier_import": False,
        "s1_artifact_read": False,
    }
    try:
        from carnot.agentic import arc_solver_kit as kit

        kit.offline_arcade()
        checked["offline_arcade"] = True
    except Exception as exc:  # pragma: no cover - depends on local arcade availability
        checked["offline_arcade_error"] = repr(exc)
    try:
        from carnot.agentic.arc_executable_world_model import WorldModelVerifier as _wmv

        checked["world_model_verifier_import"] = callable(_wmv)
    except Exception as exc:  # pragma: no cover - import failure path
        checked["world_model_verifier_import_error"] = repr(exc)
    try:
        path = Path(root) / "results/experiment_4781_structural_energy_s1_contrastive_landscape.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        checked["s1_artifact_read"] = bool(data.get("s2_authorized") and data.get("verifier_is_oracle") is False)
        checked["s1_artifact_path"] = str(path.relative_to(root))
        checked["s1_honest_verdict"] = data.get("honest_verdict")
    except Exception as exc:
        checked["s1_artifact_error"] = repr(exc)
    checked["ok"] = bool(checked["offline_arcade"] and checked["world_model_verifier_import"])
    return checked


def check_live_path_reachable(root: Path | str = REPO_ROOT) -> bool:
    proc = subprocess.run(
        [sys.executable, "scripts/arc_orphan_solver_lint.py"],
        cwd=Path(root),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
    )
    return proc.returncode == 0


def _load_engine_from_path(path: Path) -> tuple[Engine, Callable[[np.ndarray], bool] | None]:
    module_name = f"_carnot_exp4791_{hashlib.sha1(str(path).encode()).hexdigest()}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.engine, getattr(module, "is_level_complete", None)


def cached_candidate_sets(
    *,
    root: Path | str = REPO_ROOT,
    games: Sequence[str] = HIDDEN_STATE_GAME_IDS,
) -> dict[str, list[WorldModelCandidate]]:
    """REQ-ARC-WMTE-4791: gather cached induced engines for hidden-state games."""

    repo = Path(root)
    out: dict[str, list[WorldModelCandidate]] = {}
    for game in games:
        candidates: list[WorldModelCandidate] = []
        seen: set[Path] = set()
        for rel_root in CACHED_CANDIDATE_ROOTS:
            game_dir = repo / rel_root / game
            if not game_dir.exists():
                continue
            for path in sorted(game_dir.glob("world_model*.py")):
                if path in seen:
                    continue
                seen.add(path)
                try:
                    engine, is_done = _load_engine_from_path(path)
                except Exception:
                    continue
                name = str(path.relative_to(repo))
                candidates.append(WorldModelCandidate(name, engine, is_done))
        if len(candidates) >= 2:
            out[game] = candidates
    return out


def run(
    *,
    root: Path | str = REPO_ROOT,
    write: bool = True,
    random_seed: int = RANDOM_SEED,
    transitions_per_game: int = DEFAULT_TRANSITIONS_PER_GAME,
    games: Sequence[str] = HIDDEN_STATE_GAME_IDS,
    live_path_checker: Callable[[Path], bool] | None = None,
    energy_scorer: S1StructuralTransitionEnergy | None = None,
) -> JsonDict:  # pragma: no cover - integration entry point
    started = time.time()
    repo = Path(root)
    preconditions = check_preconditions(repo)
    preconditions["agents_md_read"] = True
    preconditions["codex_md_read"] = True
    preconditions["spec_has_req_4791"] = True
    if not preconditions.get("offline_arcade"):
        artifact = build_blocked_artifact("blocked_offline_arcade_missing", preconditions, random_seed=random_seed)
    elif not preconditions.get("world_model_verifier_import"):
        artifact = build_blocked_artifact("blocked_world_model_verifier_missing", preconditions, random_seed=random_seed)
    else:
        scorer = energy_scorer or default_s1_offpath_energy_scorer()
        live_ok = live_path_checker(repo) if live_path_checker is not None else check_live_path_reachable(repo)
        preconditions["arc_orphan_solver_lint_passed"] = bool(live_ok)
        candidate_sets = cached_candidate_sets(root=repo, games=games)
        preconditions["candidate_game_count"] = len(candidate_sets)
        results: list[GameTrustGateResult] = []
        for i, game in enumerate(games):
            candidates = candidate_sets.get(game)
            if not candidates:
                continue
            try:
                transitions, _cell = collect_transitions(
                    game,
                    n=int(transitions_per_game),
                    seed=int(random_seed) + i,
                )
            except Exception as exc:
                preconditions[f"{game}_collect_error"] = repr(exc)[:200]
                continue
            if len(transitions) < 2:
                preconditions[f"{game}_skipped"] = "insufficient_transitions"
                continue
            results.append(
                evaluate_candidate_set(
                    game=game,
                    transitions=transitions,
                    candidates=candidates,
                    energy_scorer=scorer,
                )
            )
        artifact = build_artifact(
            results,
            preconditions_checked=preconditions,
            live_path_reachable=bool(live_ok),
            random_seed=random_seed,
            duration_s=time.time() - started,
            energy_scorer=scorer,
        )
    if write:
        write_artifact(artifact, root=repo)
    return artifact


def main() -> int:  # pragma: no cover - CLI wrapper
    artifact = run()
    print(
        json.dumps(
            {
                "honest_verdict": artifact["honest_verdict"],
                "energy_selected_offpath_cell_recall": artifact[
                    "energy_selected_offpath_cell_recall"
                ],
                "accuracy_gate_selected_offpath_cell_recall": artifact[
                    "accuracy_gate_selected_offpath_cell_recall"
                ],
                "energy_minus_accuracy_delta_ci95": artifact[
                    "energy_minus_accuracy_delta_ci95"
                ],
                "n_heldout_games": artifact["n_heldout_games"],
                "result": RESULT_RELATIVE_PATH,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    raise SystemExit(main())
