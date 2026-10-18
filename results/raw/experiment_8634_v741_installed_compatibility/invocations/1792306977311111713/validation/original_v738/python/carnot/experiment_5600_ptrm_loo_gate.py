"""Exp 5600: PTRM Stage-1 multi-seed, pre-registered leave-one-game-out gate.

This is the falsifiable gate task 8 (`ops/known-issues.md`) has never reached:
`run_experiment_5574` always writes `loo_verdict_reached=False` regardless of
`heldout_games`, because a single training run with one seed cannot support a
statistically meaningful verdict -- v1/v2's single-seed pilots produced
directly contradictory rankings, and v3 only trusted a result after a
10-seed-per-combination paired significance test. This experiment applies
that same rigor to the (now wiring-fixed, see
`python/carnot/agentic/arc_ptrm_stage1_generator.py`) PTRM Stage-1 pipeline.

Spec refs: REQ-ARC-PTRM-5600-1, REQ-ARC-PTRM-5600-2,
SCENARIO-ARC-PTRM-5600-WIRING-FIX, SCENARIO-ARC-PTRM-5600-LOO-GATE.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from scipy import stats

from carnot.agentic.arc_ptrm_stage1_generator import (
    DEFAULT_CORPUS_DIR,
    REPO_ROOT,
    CarnotTrajectoryVerifier,
    PTRMActionSequenceGenerator,
    Stage1Config,
    Stage1DatasetBundle,
    Stage1Example,
    Stage1InputBatch,
    _infer_action_vocab,
    _load_corpus,
    build_stage1_dataset,
    generate_trajectories,
    select_trajectory,
)


JsonDict = dict[str, Any]
RESULT_FILENAME = "experiment_5600_ptrm_loo_gate.json"
RESULT_RELATIVE_PATH = f"results/{RESULT_FILENAME}"
DEFAULT_OUTPUT = REPO_ROOT / RESULT_RELATIVE_PATH

# Same five held-out games as the v3 pilot
# (docs/research-notes/trm-leave-one-game-out-pilot-results-2026-07-05.md),
# chosen there for size/imbalance diversity before this run existed --
# reused verbatim for direct comparability and to avoid post-hoc selection.
HELDOUT_GAMES: tuple[str, ...] = ("ft09", "m0r0", "vc33", "sk48", "cd82")
N_SEEDS = 10
BASE_SEED = 5600
WILCOXON_ALPHA = 0.05
MAJORITY_GAMES_REQUIRED = 3  # of 5 -- the pre-registered gate threshold

REQUIRED_ARTIFACT_FIELDS = (
    "track",
    "honest_verdict",
    "preconditions",
    "wiring_fix_note",
    "exp5574_artifact_mismatch_corrigendum",
    "heldout_games",
    "n_seeds",
    "gate_definition",
    "per_game_results",
    "games_ptrm_beats_non_recursive_significantly",
    "games_ptrm_beats_majority_baseline",
    "loo_verdict_reached",
    "heldout_generalization_signal",
    "retire_trm_generator_line",
    "corpus_won_proxy_caveat",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "gpu_device_receipt",
    "verifier_is_oracle",
    "solve_provenance",
    "inference_substrate",
    "no_level_solve_claim",
)


def preconditions(*, corpus_dir: Path = DEFAULT_CORPUS_DIR, require_cuda: bool = True) -> JsonDict:
    manifest_ok = (corpus_dir / "manifest.json").exists()
    cuda_ok = True
    cuda_detail = "cuda check waived"
    if require_cuda:
        cuda_ok = bool(torch.cuda.is_available())
        cuda_detail = f"torch_cuda_available={cuda_ok}"
    ok = manifest_ok and cuda_ok
    return {
        "corpus_manifest_present": manifest_ok,
        "cuda_available": cuda_ok,
        "cuda_detail": cuda_detail,
        "ok": ok,
    }


def _majority_action(examples: list[Stage1Example]) -> int:
    counter: Counter[int] = Counter()
    for example in examples:
        counter.update(example.target_actions)
    if not counter:
        return 0
    return counter.most_common(1)[0][0]


def _accuracy_and_exact_match(
    predicted_by_example: list[list[int]], examples: list[Stage1Example]
) -> tuple[float, float]:
    total_actions = 0
    matched_actions = 0
    exact_matches = 0
    for predicted, example in zip(predicted_by_example, examples, strict=True):
        target = example.target_actions
        matched_actions += sum(
            int(left == right) for left, right in zip(predicted, target, strict=False)
        )
        total_actions += len(target)
        exact_matches += int(list(predicted) == list(target))
    accuracy = matched_actions / float(max(total_actions, 1))
    exact_match_rate = exact_matches / float(max(len(examples), 1))
    return round(accuracy, 6), round(exact_match_rate, 6)


def _majority_baseline_predictions(
    majority_action: int, examples: list[Stage1Example]
) -> list[list[int]]:
    return [[majority_action] * len(example.target_actions) for example in examples]


def _non_recursive_predictions(
    model: PTRMActionSequenceGenerator,
    batch: Stage1InputBatch,
) -> list[list[int]]:
    model.eval()
    with torch.no_grad():
        logits = model(batch)
        predictions = torch.argmax(logits, dim=2)
    return [row.tolist() for row in predictions]


def _ptrm_predictions(
    batch: Stage1InputBatch,
    *,
    model: PTRMActionSequenceGenerator,
    verifier: CarnotTrajectoryVerifier,
    config: Stage1Config,
    action_vocab_size: int,
) -> list[list[int]]:
    predictions: list[list[int]] = []
    batch_size = int(batch.history_actions.shape[0])
    for input_index in range(batch_size):
        single = Stage1InputBatch(
            frame_features=batch.frame_features[input_index : input_index + 1],
            history_actions=batch.history_actions[input_index : input_index + 1],
            history_coords=batch.history_coords[input_index : input_index + 1],
            history_intents=batch.history_intents[input_index : input_index + 1],
            target_actions=(
                batch.target_actions[input_index : input_index + 1]
                if batch.target_actions is not None
                else None
            ),
        )
        trajectories = generate_trajectories(
            single,
            action_vocab_size=action_vocab_size,
            sequence_length=config.sequence_length,
            max_depth=config.max_depth,
            hidden_dim=config.hidden_dim,
            trajectories_per_input=config.trajectories_per_input,
            seed=config.seed * 1000 + input_index,
            noise_std=config.noise_std,
            verifier=verifier,
            model=model,
        )
        selected = select_trajectory(trajectories)
        predictions.append(list(selected.action_ids))
    return predictions


def _run_one_combination(
    *,
    bundle: Stage1DatasetBundle,
    game: str,
    seed_index: int,
    require_cuda: bool,
) -> JsonDict:
    """Trains and evaluates one (held-out game, seed) combination.

    Takes an already-built `Stage1DatasetBundle` -- `build_stage1_dataset`'s
    windowing is a deterministic function of (rows, config.heldout_games) and
    does NOT depend on `config.seed`, so the caller builds it once per held-out
    game and reuses it across all `n_seeds` draws, instead of re-scanning the
    whole corpus on every one of the 50 (game, seed) combinations.

    Runs on CPU regardless of `require_cuda`, matching `_train_proxy_model` /
    `_recursion_metrics` / `run_experiment_5574`'s own established behavior:
    the `cuda_3090_class` precondition (see `preconditions()`) is a hardware-
    provenance receipt confirming real 3090-class GPUs are present on this
    rig, not a mandate that every tensor op in this tiny (<100K-parameter)
    proxy model runs on them. `generate_trajectories`'s internal ops
    (`torch.Generator()`, `_position_bias`, `torch.bincount`) are CPU-only by
    construction; mixing in CUDA-resident tensors would raise a device
    mismatch without any accuracy benefit for a model this small.
    """

    cfg = Stage1Config(seed=BASE_SEED + seed_index, heldout_games=(game,))
    action_vocab_size = _infer_action_vocab(bundle, cfg.action_vocab_size)

    model = PTRMActionSequenceGenerator(
        history_length=cfg.history_length,
        sequence_length=cfg.sequence_length,
        action_vocab_size=action_vocab_size,
        hidden_dim=cfg.hidden_dim,
    )
    train_batch = Stage1InputBatch.from_examples(
        bundle.train_examples, action_vocab_size=action_vocab_size
    )
    _train_proxy_model_prebatched(model, train_batch, cfg, action_vocab_size)

    verifier = CarnotTrajectoryVerifier.from_sequences(
        [example.target_actions for example in bundle.train_examples],
        action_vocab_size=action_vocab_size,
    )
    majority_action = _majority_action(bundle.train_examples)

    heldout_examples = bundle.heldout_examples
    heldout_batch = Stage1InputBatch.from_examples(
        heldout_examples, action_vocab_size=action_vocab_size
    )

    ptrm_preds = _ptrm_predictions(
        heldout_batch,
        model=model,
        verifier=verifier,
        config=cfg,
        action_vocab_size=action_vocab_size,
    )
    non_recursive_preds = _non_recursive_predictions(model, heldout_batch)
    majority_preds = _majority_baseline_predictions(majority_action, heldout_examples)

    ptrm_acc, ptrm_exact = _accuracy_and_exact_match(ptrm_preds, heldout_examples)
    non_recursive_acc, non_recursive_exact = _accuracy_and_exact_match(
        non_recursive_preds, heldout_examples
    )
    majority_acc, majority_exact = _accuracy_and_exact_match(majority_preds, heldout_examples)

    return {
        "game": game,
        "seed_index": seed_index,
        "n_train_examples": len(bundle.train_examples),
        "n_heldout_examples": len(heldout_examples),
        "leakage_count": bundle.leakage_count,
        "ptrm_per_action_accuracy": ptrm_acc,
        "ptrm_exact_window_accuracy": ptrm_exact,
        "non_recursive_per_action_accuracy": non_recursive_acc,
        "non_recursive_exact_window_accuracy": non_recursive_exact,
        "majority_baseline_per_action_accuracy": majority_acc,
        "majority_baseline_exact_window_accuracy": majority_exact,
    }


def _train_proxy_model_prebatched(
    model: PTRMActionSequenceGenerator,
    train_batch: Stage1InputBatch,
    cfg: Stage1Config,
    action_vocab_size: int,
) -> JsonDict:
    """Same recipe as `_train_proxy_model` but taking an already-tensorized
    `Stage1InputBatch` directly, since `_run_one_combination` already built one
    for the whole held-in split (avoids re-tensorizing on every call)."""

    if train_batch.frame_features.shape[0] == 0:
        return {"train_token_accuracy": 0.0, "loss": 0.0}
    torch.manual_seed(int(cfg.seed))
    optimizer = torch.optim.Adam(model.parameters(), lr=float(cfg.learning_rate))
    target = train_batch.target_actions
    assert target is not None
    last_loss = torch.tensor(0.0)
    for _epoch in range(max(int(cfg.epochs), 1)):
        optimizer.zero_grad()
        logits = model(train_batch)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, int(action_vocab_size)),
            target.reshape(-1),
        )
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()
    return {"train_token_accuracy": 0.0, "loss": round(float(last_loss), 6)}


def _wilcoxon(ptrm: list[float], non_recursive: list[float]) -> JsonDict:
    diffs = [p - n for p, n in zip(ptrm, non_recursive, strict=True)]
    if all(d == 0.0 for d in diffs):
        return {"p_value": 1.0, "ptrm_wins": 0, "note": "all paired diffs zero"}
    try:
        result = stats.wilcoxon(ptrm, non_recursive)
        p_value = float(result.pvalue)
    except ValueError as exc:
        p_value = 1.0
        return {"p_value": p_value, "ptrm_wins": sum(1 for d in diffs if d > 0), "note": str(exc)}
    return {
        "p_value": round(p_value, 6),
        "ptrm_wins": sum(1 for d in diffs if d > 0),
        "ptrm_losses": sum(1 for d in diffs if d < 0),
        "ties": sum(1 for d in diffs if d == 0),
    }


def _gpu_device_receipt() -> JsonDict:
    available = bool(torch.cuda.is_available())
    smi = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "torch_cuda_available": available,
        "devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        "nvidia_smi_returncode": smi.returncode,
        "nvidia_smi_stdout": smi.stdout.strip(),
    }


def _stable_checksum(payload: JsonDict) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


def build_artifact(
    *,
    heldout_games: tuple[str, ...] = HELDOUT_GAMES,
    n_seeds: int = N_SEEDS,
    require_cuda: bool = True,
    corpus_dir: Path = DEFAULT_CORPUS_DIR,
) -> JsonDict:
    started = time.monotonic()
    preconds = preconditions(corpus_dir=corpus_dir, require_cuda=require_cuda)
    if not preconds["ok"]:
        return _blocked_artifact(preconds, started)

    rows, _dataset_hashes = _load_corpus(corpus_dir)

    per_game_results: JsonDict = {}
    for game in heldout_games:
        seed0_cfg = Stage1Config(seed=BASE_SEED, heldout_games=(game,))
        bundle = build_stage1_dataset(rows, config=seed0_cfg, heldout_games=(game,))
        game_rows = []
        for seed_index in range(n_seeds):
            row = _run_one_combination(
                bundle=bundle, game=game, seed_index=seed_index, require_cuda=require_cuda
            )
            game_rows.append(row)
        ptrm_series = [r["ptrm_per_action_accuracy"] for r in game_rows]
        non_recursive_series = [r["non_recursive_per_action_accuracy"] for r in game_rows]
        majority_value = game_rows[0]["majority_baseline_per_action_accuracy"]
        wilcoxon = _wilcoxon(ptrm_series, non_recursive_series)
        per_game_results[game] = {
            "n_seeds": n_seeds,
            "ptrm_mean_per_action_accuracy": round(sum(ptrm_series) / len(ptrm_series), 6),
            "non_recursive_mean_per_action_accuracy": round(
                sum(non_recursive_series) / len(non_recursive_series), 6
            ),
            "majority_baseline_per_action_accuracy": majority_value,
            "ptrm_exact_window_accuracy_mean": round(
                sum(r["ptrm_exact_window_accuracy"] for r in game_rows) / n_seeds, 6
            ),
            "wilcoxon_ptrm_vs_non_recursive": wilcoxon,
            "ptrm_beats_non_recursive_significantly": bool(
                wilcoxon["p_value"] < WILCOXON_ALPHA
                and wilcoxon.get("ptrm_wins", 0) > wilcoxon.get("ptrm_losses", 0)
            ),
            "ptrm_beats_majority_baseline": bool(
                round(sum(ptrm_series) / len(ptrm_series), 6) > majority_value
            ),
            "per_seed_rows": game_rows,
        }

    games_significant = [
        g for g, r in per_game_results.items() if r["ptrm_beats_non_recursive_significantly"]
    ]
    games_beat_baseline = [
        g for g, r in per_game_results.items() if r["ptrm_beats_majority_baseline"]
    ]
    games_gate_passed = [g for g in games_significant if g in games_beat_baseline]
    gate_passed = len(games_gate_passed) >= MAJORITY_GAMES_REQUIRED

    if gate_passed:
        heldout_generalization_signal = "loo_gate_passed_majority_5_games"
        honest_verdict = "complete: ptrm_loo_gate_passed_majority_of_heldout_games"
        retire = False
    else:
        heldout_generalization_signal = "loo_gate_failed_no_majority_significant_and_above_baseline"
        honest_verdict = "complete: ptrm_loo_gate_failed_no_majority_significant_and_above_baseline"
        retire = True

    artifact: JsonDict = {
        "track": "arc-trm-generator",
        "honest_verdict": honest_verdict,
        "preconditions": preconds,
        "wiring_fix_note": (
            "generate_trajectories previously ignored the trained "
            "PTRMActionSequenceGenerator entirely, seeding every trajectory from the "
            "untrained _base_action_logits frequency heuristic. Fixed 2026-07-13: model "
            "logits (mean-pooled over the K-step horizon) now seed generation when a "
            "trained model is supplied. See openspec/capabilities/arc-trm-generator/"
            "spec.md REQ-ARC-PTRM-5600-1 and SCENARIO-ARC-PTRM-5600-WIRING-FIX."
        ),
        "exp5574_artifact_mismatch_corrigendum": (
            "results/experiment_5574_ptrm_stochastic_generator_stage1.json (committed in "
            "70c857a69, an unrelated-titled commit) contains fields "
            "(exact_window_accuracy, per_action_accuracy, selection_eval, "
            "gpu_device_receipt.device_count, halting_distribution) that the "
            "_recursion_metrics/_gpu_device_receipt/build_stage1_artifact code committed "
            "in that same commit does not compute -- the artifact could not have been "
            "produced by the code it shipped with. adversarial_verify.py did not flag "
            "this (no artifact-schema-vs-source-code consistency check exists). The "
            "original artifact is preserved unmodified per never-prune discipline; its "
            "specific numbers should not be cited. This experiment (5600) supersedes it "
            "as the trustworthy, reproducible reference."
        ),
        "heldout_games": list(heldout_games),
        "n_seeds": n_seeds,
        "gate_definition": (
            f"PTRM supported only if, in >= {MAJORITY_GAMES_REQUIRED} of "
            f"{len(heldout_games)} held-out games, it both (i) beats the non-recursive "
            f"control with paired Wilcoxon p < {WILCOXON_ALPHA} across {n_seeds} seeds "
            "and (ii) has higher mean per-action accuracy than the majority-class "
            "baseline. Pre-registered before running, per REQ-ARC-PTRM-5600-2."
        ),
        "per_game_results": per_game_results,
        "games_ptrm_beats_non_recursive_significantly": games_significant,
        "games_ptrm_beats_majority_baseline": games_beat_baseline,
        "loo_verdict_reached": True,
        "heldout_generalization_signal": heldout_generalization_signal,
        "retire_trm_generator_line": retire,
        "corpus_won_proxy_caveat": (
            "Won-session filtering uses level_progress>=1.0 (inherited from exp5574 / "
            "build_stage1_dataset), which the v4 pilot found can mean 'reached this "
            "session's own highest recorded checkpoint,' not necessarily 'won the whole "
            "game.' Not fixed in this experiment's scope; disclosed per "
            "REQ-ARC-PTRM-5600-2."
        ),
        "random_seed": BASE_SEED,
        "gpu_device_receipt": _gpu_device_receipt(),
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "inference_substrate": "trained_ptrm_offline_development_proxy",
        "no_level_solve_claim": True,
    }
    artifact["duration_s"] = round(time.monotonic() - started, 6)
    artifact["reproducibility_checksum"] = _stable_checksum(
        {k: v for k, v in artifact.items() if k != "reproducibility_checksum"}
    )
    return artifact


def _blocked_artifact(preconds: JsonDict, started: float) -> JsonDict:
    return {
        "track": "arc-trm-generator",
        "honest_verdict": f"complete: blocked_{'cuda' if not preconds['cuda_available'] else 'corpus_manifest_missing'}",
        "preconditions": preconds,
        "wiring_fix_note": "not reached -- blocked before any training",
        "exp5574_artifact_mismatch_corrigendum": "not reached -- blocked before any training",
        "heldout_games": [],
        "n_seeds": 0,
        "gate_definition": "not reached -- blocked before any training",
        "per_game_results": {},
        "games_ptrm_beats_non_recursive_significantly": [],
        "games_ptrm_beats_majority_baseline": [],
        "loo_verdict_reached": False,
        "heldout_generalization_signal": "blocked_preconditions",
        "retire_trm_generator_line": False,
        "corpus_won_proxy_caveat": "not reached -- blocked before any training",
        "random_seed": BASE_SEED,
        "gpu_device_receipt": _gpu_device_receipt(),
        "verifier_is_oracle": False,
        "solve_provenance": "development_proxy",
        "inference_substrate": "trained_ptrm_offline_development_proxy",
        "no_level_solve_claim": True,
        "duration_s": round(time.monotonic() - started, 6),
        "reproducibility_checksum": "",
    }


def _write_json(path: Path, payload: JsonDict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8"
    )


def main() -> JsonDict:
    artifact = build_artifact()
    _write_json(DEFAULT_OUTPUT, artifact)
    return artifact


if __name__ == "__main__":
    main()
