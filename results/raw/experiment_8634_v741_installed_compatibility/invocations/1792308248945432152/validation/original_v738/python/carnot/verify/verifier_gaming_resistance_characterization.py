"""Exp 3790 FoVer verifier gaming-resistance characterization.

This module measures a narrow product question: if a cached FoVer wrong step is
edited to look less suspicious, do the shipped Exp 2837 verifier scores still
separate wrong steps from correct ones?  The run is deliberately cheap and
honest. It replays committed FoVer rows through the four existing scoring
helpers and applies deterministic text transforms only to rows already labeled
incorrect. It does not generate candidates with an LLM and does not change the
frozen 0.9131 headline.

Spec: REQ-VERIFY-3790, SCENARIO-VERIFY-3790.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import importlib.util
import json
import math
import re
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from carnot.eval import fover_memory_leakage_v3 as fover


REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_REL_PATH = Path("results/experiment_3790_verifier_gaming_resistance_characterization.json")
DEFAULT_RANDOM_SEED = 3790
DEFAULT_N_SAMPLES = 240
INFERENCE_SUBSTRATE = "verifier_ensemble_against_cached_candidates"
VERIFIER_NAMES = (
    "fr11_session_memory",
    "tier0r_curry_howard",
    "tier0s_arithmetic_gap",
    "tier0u_logical_consistency",
)
PRODUCTION_WEIGHTS = {
    "fr11_session_memory": 1.0,
    "tier0r_curry_howard": 0.9,
    "tier0s_arithmetic_gap": 0.0,
    "tier0u_logical_consistency": 0.1,
}
PERTURBATION_SET = (
    "arithmetic_result_plus_one",
    "operand_swap",
    "sign_flip",
    "irrelevant_truth_padding",
    "context_compaction",
)
HOLD_AUROC_DROP_TOLERANCE = 0.02
HOLD_FLAG_RATE_DROP_TOLERANCE = 0.05
SUCCESS_VERDICT = (
    "complete: verifier_gaming_resistance_characterized_degradation_curve_n240_"
    "holds_and_degrades_documented_not_a_moat_reopen_headline_unchanged"
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "gaming_degradation_curve",
    "n_samples",
    "perturbations_tested",
    "verifier_holds_where",
    "verifier_degrades_where",
    "not_a_moat_reopen",
    "headline_unchanged",
    "model_specs",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
)

FIELD_PRINCIPLES = {
    "honest_verdict": (
        "Terminal prefix; the robustness-characterization outcome; blocked_<resource> "
        "if a precondition failed."
    ),
    "inference_substrate": (
        "Re-scores cached and perturbed FoVer triples through the verifier ensemble; "
        "no live model."
    ),
    "gaming_degradation_curve": (
        "Per-perturbation step-error detection AUROC and flag rate versus clean."
    ),
    "n_samples": "Bare sample count; >=200 for the completed characterization.",
    "perturbations_tested": (
        "Transparent deterministic text transforms applied to wrong steps only."
    ),
    "verifier_holds_where": "Perturbations whose AUROC and wrong-step flag rate do not materially drop.",
    "verifier_degrades_where": "Perturbations whose AUROC or wrong-step flag rate drops materially.",
    "not_a_moat_reopen": "Bare bool; this is not the closed moat/independence thread.",
    "headline_unchanged": "Bare bool; the frozen 0.9131 headline is untouched.",
    "model_specs": "Names the four verifiers and perturbation set; no live model path.",
    "random_seed": "Determinism precondition.",
    "reproducibility_checksum": "Content hash catches drift.",
    "duration_s": "Measured wall-clock duration, not padded.",
}


@dataclass(frozen=True)
class ScorePanel:
    """Labels and shipped verifier scores for one clean or perturbed row panel."""

    labels: Sequence[int]
    ensemble_scores: Sequence[float]
    scores_by_verifier: Mapping[str, Sequence[float]]
    verifier_names: Sequence[str]


def success_verdict(n_samples: int) -> str:
    """Return the required terminal verdict with the measured sample count."""

    return (
        "complete: verifier_gaming_resistance_characterized_degradation_curve_"
        f"n{int(n_samples)}_holds_and_degrades_documented_not_a_moat_reopen_"
        "headline_unchanged"
    )


def probe_preconditions(repo_root: Path, *, n_samples: int) -> list[dict[str, Any]]:
    """Check interpreter, imports, verifier helpers, and cached corpus before scoring."""

    root = Path(repo_root)
    checks: list[dict[str, Any]] = []
    executable = Path(sys.executable)
    interpreter_ok = ".venv" in executable.parts and executable.name.startswith("python")
    checks.append(
        {
            "resource": "interpreter_not_venv",
            "available": interpreter_ok,
            "detail": str(executable),
        }
    )

    try:
        import sklearn  # noqa: F401

        import_ok = True
        import_detail = "numpy_and_sklearn_importable"
    except Exception as exc:  # noqa: BLE001 - surfaced as blocked precondition.
        import_ok = False
        import_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "python_deps_missing",
            "available": import_ok,
            "detail": import_detail,
        }
    )

    try:
        smoke = fover._score_text_verifiers(["1 + 1 = 2"])
        helper_ok = set(smoke) == set(VERIFIER_NAMES[1:]) and callable(fover._fr11_memory_score)
        helper_detail = "loaded=" + ",".join(["fr11_session_memory", *sorted(smoke)])
    except Exception as exc:  # noqa: BLE001 - surfaced as blocked precondition.
        helper_ok = False
        helper_detail = f"{type(exc).__name__}: {exc}"
    checks.append(
        {
            "resource": "four_verifiers_unavailable",
            "available": helper_ok,
            "detail": helper_detail,
        }
    )

    corpus_path = (root / "data" / "fover_corpus.jsonl").resolve()
    if not corpus_path.is_file():
        checks.append(
            {
                "resource": "fover_corpus_missing",
                "available": False,
                "detail": str(corpus_path),
            }
        )
    else:
        labeled_count = sum(1 for row in fover._read_fover_rows(corpus_path) if "label" in row)
        checks.append(
            {
                "resource": "fover_corpus_insufficient",
                "available": labeled_count >= int(n_samples),
                "detail": f"path={corpus_path}; labeled_rows={labeled_count}; required>={int(n_samples)}",
            }
        )
    return checks


def select_fover_sample(repo_root: Path, *, n_samples: int, random_seed: int) -> list[dict[str, Any]]:
    """Load a deterministic balanced sample of cached FoVer rows."""

    return fover._select_balanced_subset(
        fover._read_fover_rows(Path(repo_root) / "data" / "fover_corpus.jsonl"),
        seed=int(random_seed),
        n_examples=int(n_samples),
    )


def apply_perturbation_to_wrong_steps(
    rows: Sequence[Mapping[str, Any]],
    perturbation_name: str,
) -> list[dict[str, Any]]:
    """Apply one deterministic transform to rows labeled as wrong steps."""

    transform = _PERTURBATION_FUNCTIONS.get(str(perturbation_name))
    if transform is None:
        raise ValueError(f"unknown perturbation: {perturbation_name}")
    perturbed: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if fover._label_to_int(item.get("label")) == 1:
            item["step_text"] = transform(str(item.get("step_text", "")))
        perturbed.append(item)
    return perturbed


def score_rows_with_shipped_ensemble(
    rows: Sequence[Mapping[str, Any]],
    *,
    repo_root: Path | None = None,
    memory_index: Mapping[str, object] | None = None,
) -> ScorePanel:
    """Score rows with the four Exp 2837 verifier columns and production weights."""

    if memory_index is None:
        if repo_root is None:
            memory_index = {"question_ids": set(), "prompt_token_sets": []}
        else:
            memory_index = fover._load_fr11_memory_index(Path(repo_root))
    labels = [fover._label_to_int(row.get("label")) for row in rows]
    texts = [str(row.get("step_text", "")) for row in rows]
    text_scores = fover._score_text_verifiers(texts)
    fr11_scores = [float(fover._fr11_memory_score(dict(row), dict(memory_index))) for row in rows]
    scores_by_verifier = {
        "fr11_session_memory": fr11_scores,
        "tier0r_curry_howard": [float(value) for value in text_scores["tier0r_curry_howard"]],
        "tier0s_arithmetic_gap": [float(value) for value in text_scores["tier0s_arithmetic_gap"]],
        "tier0u_logical_consistency": [
            float(value) for value in text_scores["tier0u_logical_consistency"]
        ],
    }
    ensemble = [
        sum(PRODUCTION_WEIGHTS[name] * float(scores_by_verifier[name][idx]) for name in VERIFIER_NAMES)
        for idx in range(len(labels))
    ]
    return ScorePanel(
        labels=labels,
        ensemble_scores=ensemble,
        scores_by_verifier=scores_by_verifier,
        verifier_names=VERIFIER_NAMES,
    )


def build_artifact(
    repo_root: Path,
    *,
    started_s: float | None = None,
    now_s: float | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = DEFAULT_RANDOM_SEED,
    perturbation_names: Sequence[str] = PERTURBATION_SET,
) -> dict[str, Any]:
    """Build the Exp 3790 artifact from cached FoVer rows or fail closed."""

    root = Path(repo_root)
    start = time.time() if started_s is None else float(started_s)
    preconditions = probe_preconditions(root, n_samples=n_samples)
    failed = next((item for item in preconditions if not bool(item["available"])), None)
    if failed is not None:
        return _blocked_artifact(
            verdict=f"blocked_{failed['resource']}",
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=preconditions,
        )

    try:
        rows = select_fover_sample(root, n_samples=n_samples, random_seed=random_seed)
        memory_index = fover._load_fr11_memory_index(root)
        clean = score_rows_with_shipped_ensemble(rows, memory_index=memory_index)
        perturbed_by_name = {
            name: score_rows_with_shipped_ensemble(
                apply_perturbation_to_wrong_steps(rows, name),
                memory_index=memory_index,
            )
            for name in perturbation_names
        }
    except Exception as exc:  # noqa: BLE001 - terminal artifact must fail closed.
        return _blocked_artifact(
            verdict="blocked_scoring_unavailable",
            duration_s=_duration(start, now_s),
            random_seed=random_seed,
            preconditions=[
                *preconditions,
                {
                    "resource": "scoring_unavailable",
                    "available": False,
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            ],
        )

    return build_artifact_from_score_panels(
        clean,
        perturbed_by_name,
        started_s=start,
        now_s=now_s,
        n_samples=len(rows),
        random_seed=random_seed,
        perturbation_names=perturbation_names,
        corpus_path=(root / "data" / "fover_corpus.jsonl").resolve(),
        preconditions=preconditions,
    )


def build_artifact_from_score_panels(
    clean: ScorePanel,
    perturbed_by_name: Mapping[str, ScorePanel],
    *,
    started_s: float,
    now_s: float | None,
    n_samples: int,
    random_seed: int,
    perturbation_names: Sequence[str],
    corpus_path: Path,
    preconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Assemble the degradation curve from clean and perturbed score panels."""

    labels = np.asarray(clean.labels, dtype=np.int64)
    clean_scores = np.asarray(clean.ensemble_scores, dtype=np.float64)
    _validate_binary_panel(labels, clean_scores)
    threshold = clean_flag_threshold(labels, clean_scores)
    clean_auroc = fover.compute_auroc(labels.tolist(), clean_scores.tolist())
    clean_flag = wrong_step_flag_rate(labels, clean_scores, threshold)
    curve: dict[str, Any] = {
        "clean": {
            "auroc": _round_metric(clean_auroc),
            "wrong_step_flag_rate": _round_metric(clean_flag),
            "flag_threshold_from_clean_scores": _round_metric(threshold),
            "n_wrong_steps": int(np.sum(labels == 1)),
        }
    }
    holds: list[str] = []
    degrades: list[str] = []
    for name in perturbation_names:
        panel = perturbed_by_name[str(name)]
        pert_labels = np.asarray(panel.labels, dtype=np.int64)
        pert_scores = np.asarray(panel.ensemble_scores, dtype=np.float64)
        if not np.array_equal(labels, pert_labels):
            raise ValueError(f"{name} labels diverged from clean panel")
        auroc = fover.compute_auroc(labels.tolist(), pert_scores.tolist())
        flag = wrong_step_flag_rate(labels, pert_scores, threshold)
        auroc_delta = float(auroc - clean_auroc)
        flag_delta = float(flag - clean_flag)
        mean_wrong_delta = mean_wrong_score_delta(labels, clean_scores, pert_scores)
        classification = classify_perturbation(auroc_delta, flag_delta)
        if classification == "holds":
            holds.append(str(name))
        else:
            degrades.append(str(name))
        curve[str(name)] = {
            "auroc": _round_metric(auroc),
            "auroc_delta_vs_clean": _round_metric(auroc_delta),
            "wrong_step_flag_rate": _round_metric(flag),
            "wrong_step_flag_rate_delta_vs_clean": _round_metric(flag_delta),
            "mean_wrong_score_delta_vs_clean": _round_metric(mean_wrong_delta),
            "n_wrong_steps": int(np.sum(labels == 1)),
            "classification": classification,
        }

    artifact: dict[str, Any] = {
        "artifact": "experiment_3790_verifier_gaming_resistance_characterization",
        "schema": "carnot.verifier_gaming_resistance_characterization.v1",
        "honest_verdict": success_verdict(int(n_samples)),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "gaming_degradation_curve": curve,
        "n_samples": int(n_samples),
        "perturbations_tested": [str(name) for name in perturbation_names],
        "verifier_holds_where": holds,
        "verifier_degrades_where": degrades,
        "not_a_moat_reopen": True,
        "headline_unchanged": True,
        "model_specs": {
            "verifiers": list(VERIFIER_NAMES),
            "production_weights": dict(PRODUCTION_WEIGHTS),
            "perturbation_set": [str(name) for name in perturbation_names],
            "scoring_protocol": (
                "Exp 2837 shipped four-verifier production weighting over cached FoVer rows; "
                "labels are used only for AUROC and flag-rate measurement."
            ),
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": reproducibility_checksum(
            clean=clean,
            perturbed_by_name=perturbed_by_name,
            perturbation_names=perturbation_names,
            random_seed=random_seed,
            corpus_path=Path(corpus_path),
        ),
        "duration_s": _round_metric(_duration(float(started_s), now_s)),
        "cached_fover_corpus_path": str(Path(corpus_path)),
        "methodology_note": (
            "arXiv:2604.15149 is used as AS-REPORTED motivation only; this artifact "
            "is a Carnot measurement of deterministic perturbations over cached FoVer rows."
        ),
        "adversarial_verify_clean": False,
        "adversarial_verify_report": None,
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def clean_flag_threshold(labels: np.ndarray, clean_scores: np.ndarray) -> float:
    """Use the clean correct/wrong midpoint as a fixed flag threshold."""

    negatives = clean_scores[labels == 0]
    positives = clean_scores[labels == 1]
    if len(negatives) == 0 or len(positives) == 0:
        raise ValueError("flag threshold requires both labels")
    return float((np.mean(negatives) + np.mean(positives)) / 2.0)


def wrong_step_flag_rate(labels: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    """Fraction of wrong steps whose shipped ensemble score crosses threshold."""

    wrong_scores = scores[labels == 1]
    if len(wrong_scores) == 0:
        raise ValueError("wrong-step flag rate requires at least one wrong step")
    return float(np.mean(wrong_scores >= float(threshold)))


def mean_wrong_score_delta(labels: np.ndarray, clean_scores: np.ndarray, perturbed_scores: np.ndarray) -> float:
    """Average perturbed-minus-clean ensemble score over wrong steps."""

    wrong = labels == 1
    if not np.any(wrong):
        raise ValueError("wrong-score delta requires at least one wrong step")
    return float(np.mean(perturbed_scores[wrong] - clean_scores[wrong]))


def classify_perturbation(auroc_delta: float, flag_delta: float) -> str:
    """Classify a perturbation as held or degraded under fixed tolerances."""

    materially_lower_auroc = float(auroc_delta) < -HOLD_AUROC_DROP_TOLERANCE
    materially_lower_flag_rate = float(flag_delta) < -HOLD_FLAG_RATE_DROP_TOLERANCE
    return "degrades" if materially_lower_auroc or materially_lower_flag_rate else "holds"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the Exp 3790 schema before writing terminal JSON."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise ValueError(f"missing required artifact fields: {missing}")
    principles = artifact.get("field_principles")
    if not isinstance(principles, Mapping):
        raise ValueError("field_principles must be present")
    missing_principles = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in principles]
    if missing_principles:
        raise ValueError(f"missing field principles: {missing_principles}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str) or not (
        verdict.startswith("complete:") or verdict.startswith("blocked_")
    ):
        raise ValueError(f"unsupported honest_verdict: {verdict!r}")
    for field in ("not_a_moat_reopen", "headline_unchanged", "adversarial_verify_clean"):
        if type(artifact.get(field)) is not bool:
            raise ValueError(f"{field} must be a bare boolean")
    if artifact["inference_substrate"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be the verifier-scoring sentinel")
    if verdict.startswith("blocked_"):
        if int(artifact.get("n_samples", -1)) != 0:
            raise ValueError("blocked artifacts must report n_samples=0")
        return
    if int(artifact["n_samples"]) < 200:
        raise ValueError("completed artifact must report n_samples >= 200")
    if not artifact.get("perturbations_tested"):
        raise ValueError("perturbations_tested must be nonempty")
    if not isinstance(artifact.get("gaming_degradation_curve"), Mapping):
        raise ValueError("gaming_degradation_curve must be a mapping")
    curve = dict(artifact["gaming_degradation_curve"])
    if "clean" not in curve:
        raise ValueError("gaming_degradation_curve must include clean")
    for field in ("duration_s",):
        value = artifact.get(field)
        if not isinstance(value, int | float) or not math.isfinite(float(value)):
            raise ValueError(f"{field} must be finite")


def write_artifact(
    repo_root: Path,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build, adversarially verify, and write the Exp 3790 terminal artifact."""

    root = Path(repo_root)
    artifact = build_artifact(root, started_s=started_s, now_s=now_s)
    target = root / output_path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if not str(artifact["honest_verdict"]).startswith("blocked_"):
        report = run_adversarial_verify_report(target)
        artifact["adversarial_verify_clean"] = adversarial_report_is_clean(report)
        artifact["adversarial_verify_report"] = {
            "flag_count": int(report.get("flag_count", 0)),
            "max_severity": report.get("max_severity"),
            "flags": list(report.get("flags") or []),
        }
        validate_artifact(artifact)
        target.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def run_adversarial_verify_report(path: Path) -> dict[str, Any]:  # pragma: no cover
    """Run the repository artifact verifier and return its structured report."""

    script_path = REPO_ROOT / "scripts" / "adversarial_verify.py"
    spec = importlib.util.spec_from_file_location("carnot_adversarial_verify", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not import {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return dict(module.verify_artifact(Path(path)))


def adversarial_report_is_clean(report: Mapping[str, Any]) -> bool:
    """True when the artifact verifier emitted no critical or TAUTOLOGY flag."""

    for flag in list(report.get("flags") or []):
        item = dict(flag)
        if str(item.get("kind", "")) == "TAUTOLOGY" or str(item.get("severity", "")) == "critical":
            return False
    return True


def reproducibility_checksum(
    *,
    clean: ScorePanel,
    perturbed_by_name: Mapping[str, ScorePanel],
    perturbation_names: Sequence[str],
    random_seed: int,
    corpus_path: Path,
) -> str:
    """Hash labels, score vectors, perturbation order, seed, and corpus path."""

    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(clean.labels, dtype=np.int64).tobytes())
    digest.update(np.ascontiguousarray(clean.ensemble_scores, dtype=np.float64).tobytes())
    for name in perturbation_names:
        digest.update(str(name).encode("utf-8"))
        panel = perturbed_by_name[str(name)]
        digest.update(np.ascontiguousarray(panel.ensemble_scores, dtype=np.float64).tobytes())
    digest.update(json.dumps(list(clean.verifier_names), separators=(",", ":")).encode("utf-8"))
    digest.update(str(int(random_seed)).encode("ascii"))
    digest.update(str(Path(corpus_path)).encode("utf-8"))
    return digest.hexdigest()


def _blocked_artifact(
    *,
    verdict: str,
    duration_s: float,
    random_seed: int,
    preconditions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    payload = json.dumps(
        {"preconditions": [dict(item) for item in preconditions], "random_seed": random_seed},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    artifact: dict[str, Any] = {
        "artifact": "experiment_3790_verifier_gaming_resistance_characterization",
        "schema": "carnot.verifier_gaming_resistance_characterization.v1",
        "honest_verdict": str(verdict),
        "inference_substrate": INFERENCE_SUBSTRATE,
        "gaming_degradation_curve": {},
        "n_samples": 0,
        "perturbations_tested": list(PERTURBATION_SET),
        "verifier_holds_where": [],
        "verifier_degrades_where": [],
        "not_a_moat_reopen": True,
        "headline_unchanged": True,
        "model_specs": {
            "verifiers": list(VERIFIER_NAMES),
            "production_weights": dict(PRODUCTION_WEIGHTS),
            "perturbation_set": list(PERTURBATION_SET),
        },
        "random_seed": int(random_seed),
        "reproducibility_checksum": hashlib.sha256(payload).hexdigest(),
        "duration_s": _round_metric(duration_s),
        "adversarial_verify_clean": False,
        "adversarial_verify_report": None,
        "preconditions_checked": [dict(item) for item in preconditions],
        "field_principles": dict(FIELD_PRINCIPLES),
    }
    validate_artifact(artifact)
    return artifact


def _validate_binary_panel(labels: np.ndarray, scores: np.ndarray) -> None:
    if labels.shape[0] != scores.shape[0]:
        raise ValueError("labels and scores must have the same length")
    if set(labels.tolist()) != {0, 1}:
        raise ValueError("score panel requires both correct and incorrect labels")
    if not np.isfinite(scores).all():
        raise ValueError("scores must be finite")


def _round_metric(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _duration(started_s: float, now_s: float | None) -> float:
    now = time.time() if now_s is None else float(now_s)
    return max(0.0, now - float(started_s))


_NUMBER_RE = re.compile(r"(?<![A-Za-z])\d+(?![A-Za-z])")


def _perturb_arithmetic_result_plus_one(text: str) -> str:
    matches = list(_NUMBER_RE.finditer(text))
    if not matches:
        return text + " Plausible wrong arithmetic result: 1."
    match = matches[-1]
    replacement = str(int(match.group()) + 1)
    return text[: match.start()] + replacement + text[match.end() :]


def _perturb_operand_swap(text: str) -> str:
    pattern = re.compile(r"(\d+)\s*([+\-*/×])\s*(\d+)")
    swapped = pattern.sub(lambda m: f"{m.group(3)} {m.group(2)} {m.group(1)}", text, count=1)
    if swapped != text:
        return swapped
    matches = list(_NUMBER_RE.finditer(text))
    if len(matches) < 2:
        return text + " Swapped operands: 1 and 2."
    first, second = matches[0], matches[1]
    return (
        text[: first.start()]
        + second.group()
        + text[first.end() : second.start()]
        + first.group()
        + text[second.end() :]
    )


def _perturb_sign_flip(text: str) -> str:
    flipped = re.sub(r"\s\+\s", " - ", text, count=1)
    if flipped != text:
        return flipped
    match = _NUMBER_RE.search(text)
    if match is None:
        return text + " Sign-flipped check: -1."
    return text[: match.start()] + "-" + text[match.start() :]


def _perturb_irrelevant_truth_padding(text: str) -> str:
    return (
        text
        + "\nIrrelevant true check: 2 + 2 = 4 and 3 * 3 = 9. "
        "This does not change the preceding result."
    )


def _perturb_context_compaction(text: str) -> str:
    pieces = [piece.strip() for piece in re.split(r"[\n.]", text) if piece.strip()]
    preferred = [
        piece
        for piece in pieces
        if re.search(r"\b(therefore|answer|total|result|so)\b", piece, re.IGNORECASE)
    ]
    compact = (preferred[-1] if preferred else pieces[-1] if pieces else text[:120]).strip()
    return compact if compact else "Therefore, the stated result follows."


_PERTURBATION_FUNCTIONS = {
    "arithmetic_result_plus_one": _perturb_arithmetic_result_plus_one,
    "operand_swap": _perturb_operand_swap,
    "sign_flip": _perturb_sign_flip,
    "irrelevant_truth_padding": _perturb_irrelevant_truth_padding,
    "context_compaction": _perturb_context_compaction,
}
