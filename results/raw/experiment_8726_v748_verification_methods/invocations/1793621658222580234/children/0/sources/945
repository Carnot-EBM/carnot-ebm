"""Experiment 4545: ARC cross-game discriminative verifier v3 LOO gate.

Spec refs: REQ-LEARN-4476, SCENARIO-LEARN-4476-GATE.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from carnot.agentic.arc_value_learner import (
    DiscriminativeVerifier,
    cross_game_feature_names_v3,
    cross_game_feature_slices_v3,
    cross_game_features_v3,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULT_RELATIVE_PATH = "results/experiment_4545_cross_game_discrimination_v3.json"
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
RANDOM_SEED = 0
CHANCE_AUROC = 0.5
POSITIVE_CONTROL_THRESHOLD = 0.5
FEATURE_FAMILIES_ADDED = [
    "object_relational",
    "frame_delta",
    "action_conditioned",
    "predicate_distance",
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
FIELD_PRINCIPLES = {
    "honest_verdict": (
        "terminal prefix; success: cross_game_discrimination_loo_auroc_<n>_above_chance OR "
        "complete: cross_game_discrimination_still_chance_gap_sharpened_honest_null."
    ),
    "inference_substrate": (
        "verifier_ensemble_against_cached_candidates -- scores cached candidate features, no LLM load "
        "(1s floor)."
    ),
    "verifier_is_oracle": (
        "MUST be false -- the DiscriminativeVerifier ranks by a LEARNED signal, oracle-DISTINCT from "
        "the executable win-check; a circular win does not count (Circularity / Oracle-Distinctness "
        "Discipline)."
    ),
    "loo_auroc_mean": (
        "the HEADLINE -- mean leave-one-game-out AUROC; > 0.5 with CI-excl-0.5 is the only "
        "non-circular cross-game discrimination evidence."
    ),
    "loo_auroc_ci": (
        "bootstrap CI on the LOO AUROC; a claim above chance requires the CI to exclude 0.5 "
        "(the .419-class FALSE_NEGATIVE_RISK guard against a small-sample artifact)."
    ),
    "in_sample_auroc": (
        "the POSITIVE CONTROL -- train==test AUROC; must be > 0.5 or the harness is broken and a "
        "LOO null is uninformative."
    ),
    "feature_families_added": (
        "the relational/delta-frame/action-conditioned/predicate-distance families added to v3 -- "
        "traceable to the GAP-ARCH-FEATURES spec."
    ),
    "positive_control_passed": (
        "in-sample AUROC > 0.5; guards a silently-broken harness."
    ),
    "false_negative_risk_checked": (
        "a LOO null is valid only if the in-sample positive control passed."
    ),
    "missing_verifier_gaps": (
        "if LOO stays at chance, the sharpened discriminator still missing (Missing-Verifier Gap "
        "Logging) -- the input to the verifier-build backlog."
    ),
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "catches silent corpus drift on replay.",
    "preconditions_checked": (
        "records resources verified; pre-empts missing-resource fabrication."
    ),
}
REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES)
SPEC_REFS = ["REQ-LEARN-4476", "SCENARIO-LEARN-4476-GATE"]


def _clean_float(value: Any) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if out == out else None


def _checksum_payload(payload: Mapping[str, Any]) -> str:
    clean = {k: v for k, v in payload.items() if k != "reproducibility_checksum"}
    encoded = json.dumps(clean, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round_row(row: Sequence[float]) -> list[float]:
    return [round(float(v), 12) for v in row]


def corpus_checksum(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
    per_game: Mapping[str, Any],
) -> str:
    payload = {
        "feature_names": cross_game_feature_names_v3(),
        "per_game": per_game,
        "x": [_round_row(row) for row in x_rows],
        "y": [float(v) for v in y_rows],
    }
    encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def tie_aware_auroc(scores: Sequence[float], labels: Sequence[float]) -> float:
    pos = [float(s) for s, label in zip(scores, labels) if float(label) == 1.0]
    neg = [float(s) for s, label in zip(scores, labels) if float(label) == 0.0]
    if not pos or not neg:
        return 0.5

    order = sorted(range(len(scores)), key=lambda i: float(scores[i]))
    ranks: dict[int, float] = {}
    i = 0
    while i < len(order):
        j = i
        while j < len(order) and float(scores[order[j]]) == float(scores[order[i]]):
            j += 1
        avg_rank = (i + j + 1) / 2.0
        for k in range(i, j):
            ranks[order[k]] = avg_rank
        i = j
    pos_rank_sum = sum(ranks[i] for i in range(len(scores)) if float(labels[i]) == 1.0)
    return float((pos_rank_sum - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg)))


def evaluate_loo(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
    per_game: Mapping[str, Mapping[str, int]],
    *,
    iters: int = 800,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> dict[str, Any]:
    x_list = [[float(v) for v in row] for row in x_rows]
    y_list = [float(v) for v in y_rows]
    bounds: dict[str, tuple[int, int]] = {}
    cur = 0
    for game in sorted(per_game):
        n = int(per_game[game].get("pos", 0)) + int(per_game[game].get("neg", 0))
        bounds[game] = (cur, cur + n)
        cur += n

    per_game_auroc: dict[str, float] = {}
    for held, (lo, hi) in bounds.items():
        held_y = y_list[lo:hi]
        if (hi - lo) < 4 or not held_y or sum(held_y) in (0.0, float(len(held_y))):
            continue
        train_x = x_list[:lo] + x_list[hi:]
        train_y = y_list[:lo] + y_list[hi:]
        clf = DiscriminativeVerifier(lambda row: row).fit(train_x, train_y, iters=iters, lr=lr, l2=l2)
        scores = [clf.proba_features(row) for row in x_list[lo:hi]]
        per_game_auroc[held] = tie_aware_auroc(scores, held_y)

    full = DiscriminativeVerifier(lambda row: row).fit(x_list, y_list, iters=iters, lr=lr, l2=l2)
    in_sample_scores = [full.proba_features(row) for row in x_list]
    loo_values = list(per_game_auroc.values())
    return {
        "loo_auroc_mean": float(np.mean(loo_values)) if loo_values else None,
        "per_game_loo_auroc": per_game_auroc,
        "in_sample_auroc": tie_aware_auroc(in_sample_scores, y_list),
        "n_held_out_games": len(per_game_auroc),
        "n_pos": int(sum(1 for v in y_list if v == 1.0)),
        "n_neg": int(sum(1 for v in y_list if v == 0.0)),
    }


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    random_seed: int,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
) -> list[float | None]:
    arr = np.asarray([float(v) for v in values if _clean_float(v) is not None], dtype=float)
    if arr.size == 0:
        return [None, None]
    rng = np.random.default_rng(random_seed)
    samples = rng.choice(arr, size=(int(n_bootstrap), arr.size), replace=True)
    means = samples.mean(axis=1)
    alpha = (1.0 - float(confidence)) / 2.0
    return [
        float(np.quantile(means, alpha)),
        float(np.quantile(means, 1.0 - alpha)),
    ]


def _subset_rows(
    x_rows: Sequence[Sequence[float]],
    ranges: Sequence[tuple[int, int]],
) -> list[list[float]]:
    out: list[list[float]] = []
    for row in x_rows:
        vals: list[float] = []
        for start, stop in ranges:
            vals.extend(float(v) for v in row[start:stop])
        out.append(vals)
    return out


def evaluate_feature_classes(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
    per_game: Mapping[str, Mapping[str, int]],
    *,
    iters: int = 800,
    lr: float = 0.5,
    l2: float = 1e-3,
) -> dict[str, dict[str, float | None]]:
    slices = cross_game_feature_slices_v3()
    specs: dict[str, list[tuple[int, int]]] = {
        "v2": [slices["v2"]],
        "v2_plus_action_conditioned": [slices["v2"], slices["action_conditioned"]],
        "v2_plus_frame_delta": [slices["v2"], slices["frame_delta"]],
        "v2_plus_object_relational": [slices["v2"], slices["object_relational"]],
        "v2_plus_predicate_distance": [slices["v2"], slices["predicate_distance"]],
        "v3_full": [(0, len(cross_game_feature_names_v3()))],
    }
    loo: dict[str, float | None] = {}
    in_sample: dict[str, float | None] = {}
    for name, ranges in specs.items():
        metrics = evaluate_loo(
            _subset_rows(x_rows, ranges),
            y_rows,
            per_game,
            iters=iters,
            lr=lr,
            l2=l2,
        )
        loo[name] = _clean_float(metrics.get("loo_auroc_mean"))
        in_sample[name] = _clean_float(metrics.get("in_sample_auroc"))
    return {"loo_auroc": loo, "in_sample_auroc": in_sample}


def _feature_class_deltas(feature_class_metrics: Mapping[str, Mapping[str, Any]]) -> dict[str, float | None]:
    loo = feature_class_metrics.get("loo_auroc", {})
    baseline = _clean_float(loo.get("v2"))
    out: dict[str, float | None] = {}
    for key, value in sorted(loo.items()):
        if key == "v2":
            continue
        current = _clean_float(value)
        out[key] = None if baseline is None or current is None else float(current - baseline)
    return out


def _loo_ci_excludes_chance(mean: float | None, ci: Sequence[Any]) -> bool:
    if mean is None or len(ci) != 2:
        return False
    lo = _clean_float(ci[0])
    return bool(lo is not None and mean > CHANCE_AUROC and lo > CHANCE_AUROC)


def _missing_verifier_gaps(
    feature_class_metrics: Mapping[str, Mapping[str, Any]],
    *,
    success: bool,
) -> list[str]:
    if success:
        return []
    loo = feature_class_metrics.get("loo_auroc", {})
    in_sample = feature_class_metrics.get("in_sample_auroc", {})
    stalled: list[str] = []
    for key, family in (
        ("v2_plus_frame_delta", "frame_delta"),
        ("v2_plus_object_relational", "object_relational"),
        ("v2_plus_predicate_distance", "predicate_distance"),
        ("v2_plus_action_conditioned", "action_conditioned"),
    ):
        sample = _clean_float(in_sample.get(key))
        transfer = _clean_float(loo.get(key))
        if sample is not None and sample > POSITIVE_CONTROL_THRESHOLD and (
            transfer is None or transfer <= 0.55
        ):
            stalled.append(family)
    if stalled:
        return [
            f"{stalled[0]} moved in-sample-but-not-LOO; missing invariant mechanic/world-model "
            "state abstractions beyond one-step frame statistics."
        ]
    return [
        "v3 feature families did not produce a reliable in-sample or LOO discriminator; missing stronger "
        "negative labels and mechanic-specific predicate abstractions."
    ]


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict must start with a terminal prefix")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        errors.append("inference_substrate must be verifier_ensemble_against_cached_candidates")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle must be false")
    if not isinstance(artifact.get("preconditions_checked"), Mapping):
        errors.append("preconditions_checked must be a mapping")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping) or set(REQUIRED_ARTIFACT_FIELDS) - set(principles):
        errors.append("field_principles must annotate every required artifact field")
    ci = artifact.get("loo_auroc_ci")
    if not isinstance(ci, list) or len(ci) != 2:
        errors.append("loo_auroc_ci must be a two-item list")
    if artifact.get("false_negative_risk_checked") and not artifact.get("positive_control_passed"):
        errors.append("false_negative_risk_checked requires positive_control_passed")
    if artifact.get("loo_ci_excludes_chance") and not artifact.get("positive_control_passed"):
        errors.append("above-chance LOO claim requires positive_control_passed")
    checksum = artifact.get("reproducibility_checksum")
    if not isinstance(checksum, str) or not checksum:
        errors.append("reproducibility_checksum must be present")
    return errors


def build_artifact(
    *,
    metrics: Mapping[str, Any],
    feature_class_metrics: Mapping[str, Mapping[str, Any]],
    preconditions_checked: Mapping[str, Any],
    random_seed: int,
    corpus_checksum: str,
    tests_pass: bool,
) -> dict[str, Any]:
    loo_mean = _clean_float(metrics.get("loo_auroc_mean"))
    loo_ci = list(metrics.get("loo_auroc_ci", [None, None]))
    in_sample = _clean_float(metrics.get("in_sample_auroc"))
    positive_control = bool(in_sample is not None and in_sample > POSITIVE_CONTROL_THRESHOLD)
    ci_excludes = _loo_ci_excludes_chance(loo_mean, loo_ci)
    success = bool(positive_control and ci_excludes)
    false_negative_checked = positive_control

    if success:
        verdict = f"success: cross_game_discrimination_loo_auroc_{loo_mean:.3f}_above_chance"
    elif positive_control:
        verdict = "complete: cross_game_discrimination_still_chance_gap_sharpened_honest_null"
    else:
        verdict = "complete: cross_game_discrimination_positive_control_failed_harness_uninformative"

    payload: dict[str, Any] = {
        "experiment": "experiment_4545_cross_game_discrimination_v3",
        "schema": "carnot.arc_cross_game_discrimination_v3_4545.v1",
        "honest_verdict": verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "verifier_is_oracle": False,
        "loo_auroc_mean": loo_mean,
        "loo_auroc_ci": [_clean_float(v) for v in loo_ci],
        "loo_ci_excludes_chance": ci_excludes,
        "in_sample_auroc": in_sample,
        "feature_families_added": list(FEATURE_FAMILIES_ADDED),
        "positive_control_passed": positive_control,
        "false_negative_risk_checked": false_negative_checked,
        "missing_verifier_gaps": _missing_verifier_gaps(feature_class_metrics, success=success),
        "random_seed": int(random_seed),
        "preconditions_checked": dict(preconditions_checked),
        "field_principles": dict(FIELD_PRINCIPLES),
        "spec_refs": list(SPEC_REFS),
        "n_held_out_games": int(metrics.get("n_held_out_games", 0) or 0),
        "n_pos": int(metrics.get("n_pos", 0) or 0),
        "n_neg": int(metrics.get("n_neg", 0) or 0),
        "per_game_loo_auroc": dict(metrics.get("per_game_loo_auroc", {})),
        "feature_class_loo_auroc": dict(feature_class_metrics.get("loo_auroc", {})),
        "feature_class_in_sample_auroc": dict(feature_class_metrics.get("in_sample_auroc", {})),
        "feature_class_deltas": _feature_class_deltas(feature_class_metrics),
        "corpus_checksum": corpus_checksum,
        "tests_pass": bool(tests_pass),
    }
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def write_artifact(artifact: Mapping[str, Any], *, root: Path | str = REPO_ROOT) -> Path:
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    out = Path(root) / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(artifact), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out


def _load_train_module() -> Any:  # pragma: no cover - exercised by the required CLI run.
    script = REPO_ROOT / "scripts" / "arc_cross_game_verifier_train.py"
    spec = importlib.util.spec_from_file_location("arc_cross_game_verifier_train_4545", script)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {script}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def check_preconditions(train_mod: Any | None = None) -> tuple[dict[str, Any], Any | None]:  # pragma: no cover
    status: dict[str, Any] = {
        "arc_value_learner_import": True,
        "train_harness_import": False,
        "cached_cross_game_candidate_corpus": False,
        "missing_artifacts": [],
        "empty_action_artifacts": [],
    }
    try:
        mod = train_mod or _load_train_module()
        status["train_harness_import"] = True
    except Exception as exc:
        status["train_harness_error"] = repr(exc)
        return status, None

    try:
        mh = mod._metaharness()
        missing: list[str] = []
        empty: list[str] = []
        for game, artifact in sorted(mh.GAME_ARTIFACTS.items()):
            src = mh.RESOLVED_ARTIFACTS.get(game, artifact)
            if not (REPO_ROOT / src).exists():
                missing.append(src)
                continue
            if not mh.load_actions(src):
                empty.append(src)
        status["missing_artifacts"] = missing
        status["empty_action_artifacts"] = empty
        status["banked_game_count"] = len(mh.GAME_ARTIFACTS)
        status["cached_cross_game_candidate_corpus"] = not missing and not empty
    except Exception as exc:
        status["candidate_corpus_error"] = repr(exc)
    return status, mod


def _blocked_artifact(resource: str, preconditions: Mapping[str, Any], random_seed: int) -> dict[str, Any]:
    payload = build_artifact(
        metrics={
            "loo_auroc_mean": None,
            "loo_auroc_ci": [None, None],
            "in_sample_auroc": None,
            "per_game_loo_auroc": {},
            "n_held_out_games": 0,
            "n_pos": 0,
            "n_neg": 0,
        },
        feature_class_metrics={"loo_auroc": {}, "in_sample_auroc": {}},
        preconditions_checked=preconditions,
        random_seed=random_seed,
        corpus_checksum="blocked",
        tests_pass=False,
    )
    payload["honest_verdict"] = f"complete: blocked_{resource}"
    payload["missing_verifier_gaps"] = [f"blocked_{resource}: required cross-game corpus unavailable"]
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    payload["schema_errors"] = artifact_schema_errors(payload)
    payload["reproducibility_checksum"] = _checksum_payload(payload)
    return payload


def _write_sidecars(
    x_rows: Sequence[Sequence[float]],
    y_rows: Sequence[float],
    per_game: Mapping[str, Any],
    metrics: Mapping[str, Any],
    *,
    root: Path,
) -> None:  # pragma: no cover - exercised by the required CLI run.
    model_path = root / "models" / "arc_discriminative_verifier_v3.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    verifier = DiscriminativeVerifier(lambda row: row).fit(x_rows, y_rows)
    verifier.save(
        model_path,
        meta={
            "trained_games": list(sorted(per_game)),
            "feature_names": cross_game_feature_names_v3(),
            "provenance": (
                "Exp4545 v3 cross-game discriminative verifier; "
                f"in_sample_auroc={metrics.get('in_sample_auroc')}; "
                f"loo_auroc_mean={metrics.get('loo_auroc_mean')}"
            ),
        },
    )
    summary = {
        "n_pos": metrics.get("n_pos"),
        "n_neg": metrics.get("n_neg"),
        "per_game": per_game,
        "in_sample_auroc": metrics.get("in_sample_auroc"),
        "loo_auroc": metrics.get("loo_auroc_mean"),
        "loo_auroc_ci": metrics.get("loo_auroc_ci"),
        "n_held_out_games": metrics.get("n_held_out_games"),
        "checkpoint": "models/arc_discriminative_verifier_v3.json",
        "feature_names": "cross_game_features_v3",
        "honest_verdict": "complete_discriminative_cross_game_transfer_measured",
        "inference_substrate": INFERENCE_SUBSTRATE,
        "mode": "discriminative_win_reachability_off_path_negatives_v3",
    }
    out = root / "results" / "arc_discriminative_verifier_v3.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run(
    *,
    root: Path | str = REPO_ROOT,
    random_seed: int = RANDOM_SEED,
    neg_per_game: int = 14,
    n_bootstrap: int = 2000,
) -> dict[str, Any]:  # pragma: no cover - exercised by the required CLI run.
    root_path = Path(root)
    preconditions, train_mod = check_preconditions()
    if train_mod is None or not preconditions.get("cached_cross_game_candidate_corpus"):
        artifact = _blocked_artifact("cached_cross_game_candidate_corpus", preconditions, random_seed)
        write_artifact(artifact, root=root_path)
        return artifact

    x_rows, y_rows, per_game = train_mod.collect_discriminative(
        featurize=cross_game_features_v3,
        neg_per_game=neg_per_game,
        seed=random_seed,
    )
    if int(sum(y_rows)) < 10 or int(len(y_rows) - sum(y_rows)) < 10:
        blocked = {
            **preconditions,
            "n_pos": int(sum(y_rows)),
            "n_neg": int(len(y_rows) - sum(y_rows)),
        }
        artifact = _blocked_artifact("insufficient_cross_game_candidate_labels", blocked, random_seed)
        write_artifact(artifact, root=root_path)
        return artifact

    metrics = evaluate_loo(x_rows, y_rows, per_game)
    metrics["loo_auroc_ci"] = bootstrap_mean_ci(
        list(metrics["per_game_loo_auroc"].values()),
        random_seed=random_seed,
        n_bootstrap=n_bootstrap,
    )
    feature_metrics = evaluate_feature_classes(x_rows, y_rows, per_game)
    checksum = corpus_checksum(x_rows, y_rows, per_game)
    preconditions = {
        **preconditions,
        "feature_names": "cross_game_features_v3",
        "neg_per_game": int(neg_per_game),
        "seed": int(random_seed),
        "candidate_rows": len(x_rows),
    }
    artifact = build_artifact(
        metrics=metrics,
        feature_class_metrics=feature_metrics,
        preconditions_checked=preconditions,
        random_seed=random_seed,
        corpus_checksum=checksum,
        tests_pass=False,
    )
    _write_sidecars(x_rows, y_rows, per_game, metrics, root=root_path)
    write_artifact(artifact, root=root_path)
    return artifact


def main() -> int:  # pragma: no cover - thin CLI wrapper.
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0 if not str(artifact.get("honest_verdict", "")).startswith("complete: blocked_") else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
