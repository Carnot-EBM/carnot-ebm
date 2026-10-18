#!/usr/bin/env python3
"""Exp 5049: confirm the best powered MuSR verifier on the Exp 5044 corpus.

Spec refs: REQ-VERIFY-5049, SCENARIO-VERIFY-5049.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT / "python") not in sys.path:  # pragma: no cover - direct script execution
    sys.path.insert(0, str(REPO_ROOT / "python"))

from carnot import experiment_5031_lora_ebm_scorer_musr_v3 as d1  # noqa: E402
from carnot import experiment_5046_vpr_process_reward_repair as d2  # noqa: E402
from carnot import moat_benchmark_harness as harness  # noqa: E402
from carnot.moat_benchmark_harness import (  # noqa: E402
    DEFAULT_RANDOM_SEED,
    OracleDistinctnessError,
    evaluate_verifier,
)


JsonDict = dict[str, Any]
JsonMap = Mapping[str, Any]
ScoreFn = Callable[[Any, list[str]], list[float]]
Clock = Callable[[], float]

EXPERIMENT_ID = 5049
EXPERIMENT_NAME = "experiment_5049_second_corpus_confirmation"
SCHEMA = "carnot.experiment_5049_second_corpus_confirmation.v1"
RESULT_RELATIVE_PATH = "results/experiment_5049_second_corpus_confirmation.json"
EXP5044_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.json"
EXP5044_CACHE_RELATIVE_PATH = "results/experiment_5044_second_corpus_candidate_cache.jsonl"
EXP5045_RELATIVE_PATH = "results/experiment_5045_powered_lora_ebm_eorm_musr.json"
EXP5046_RELATIVE_PATH = "results/experiment_5046_vpr_process_reward_repair.json"
EXP5047_RELATIVE_PATH = "results/experiment_5047_kan_purm_energy_calibration.json"
SPEC_REFS = ["REQ-VERIFY-5049", "SCENARIO-VERIFY-5049"]
RANDOM_SEED = DEFAULT_RANDOM_SEED

ORACLE_CANDIDATE_KEYS = frozenset(
    {
        "gold",
        "label",
        "label_correct",
        "candidate_label",
        "solver_verdict",
        "solver_score_used_for_selection",
        "answer_index",
        "answer_choice",
        "model_id",
        "generation_model",
        "scoring_model",
        "source_checkpoint_path",
        "oracle_answer",
    }
)

REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "model_specs",
    "second_corpus_name",
    "second_corpus_confirmed",
    "best_arm",
    "best_arm_source",
    "n_questions_second",
    "genuine_sc_accuracy_second",
    "verifier_accuracy_second",
    "delta_vs_tuned_sc_second",
    "paired_ci95_second",
    "mcnemar_p_second",
    "headroom_present",
    "verifier_is_oracle",
    "schema",
    "experiment",
    "experiment_id",
    "spec_refs",
    "result_path",
    "candidate_cache_path",
    "source_artifacts",
    "oracle_distinctness_enforced",
    "duration_s",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class PoweredArm:
    """A deployable powered verifier selected from upstream MuSR evidence."""

    arm: str
    source: str
    scorer_kind: str
    delta_vs_tuned_sc: float
    selection_accuracy: float
    artifact_path: Path
    model_specs: JsonDict
    checkpoint_path: str | None = None


def _json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_object(path: Path) -> JsonDict | None:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return dict(payload) if isinstance(payload, Mapping) else None


def _read_jsonl(path: Path) -> list[JsonDict]:
    if not path.exists():
        return []
    rows: list[JsonDict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _number(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_") or "corpus"


def _arm_check(arm: str, source: str, available: bool, detail: str) -> JsonDict:
    return {
        "arm": arm,
        "source": source,
        "available": bool(available),
        "detail": detail,
    }


def _payload_common_ok(payload: JsonMap) -> tuple[bool, str]:
    if payload.get("verifier_is_oracle") is not False:
        return False, "verifier_is_oracle_not_false"
    if payload.get("headroom_present") is not True:
        return False, "musr_headroom_not_present"
    return True, "ok"


def _d1_from_payload(path: Path, payload: JsonMap) -> tuple[PoweredArm | None, JsonDict]:
    common_ok, detail = _payload_common_ok(payload)
    delta = _number(payload.get("delta_vs_tuned_sc"))
    accuracy = _number(payload.get("powered_lora_ebm_accuracy"))
    checkpoint = str(payload.get("checkpoint_path") or "")
    available = (
        common_ok
        and payload.get("powered_scorer_available") is True
        and payload.get("scorer_trained") is True
        and delta is not None
        and accuracy is not None
        and bool(checkpoint)
    )
    if not available:
        reason = detail if not common_ok else "powered_lora_ebm_not_deployable"
        return None, _arm_check("D1", EXP5045_RELATIVE_PATH, False, reason)
    return (
        PoweredArm(
            arm="D1",
            source=EXP5045_RELATIVE_PATH,
            scorer_kind="lora_ebm_lookup",
            delta_vs_tuned_sc=float(delta),
            selection_accuracy=float(accuracy),
            artifact_path=path,
            model_specs=dict(payload.get("model_specs") or {}),
            checkpoint_path=checkpoint,
        ),
        _arm_check("D1", EXP5045_RELATIVE_PATH, True, f"musr_delta={delta:.6f}"),
    )


def _d2_from_payload(path: Path, payload: JsonMap) -> tuple[PoweredArm | None, JsonDict]:
    common_ok, detail = _payload_common_ok(payload)
    delta = _number(payload.get("delta_vs_tuned_sc"))
    accuracy = _number(payload.get("process_reward_accuracy"))
    available = (
        common_ok
        and payload.get("process_reward_available") is True
        and payload.get("scalar_marker_only") is False
        and payload.get("oracle_distinctness_enforced") is True
        and delta is not None
        and accuracy is not None
    )
    if not available:
        reason = detail if not common_ok else "dense_process_reward_not_deployable"
        return None, _arm_check("D2", EXP5046_RELATIVE_PATH, False, reason)
    return (
        PoweredArm(
            arm="D2",
            source=EXP5046_RELATIVE_PATH,
            scorer_kind="dense_process_reward",
            delta_vs_tuned_sc=float(delta),
            selection_accuracy=float(accuracy),
            artifact_path=path,
            model_specs=dict(payload.get("model_specs") or {}),
        ),
        _arm_check("D2", EXP5046_RELATIVE_PATH, True, f"musr_delta={delta:.6f}"),
    )


def _d3_from_payload(path: Path, payload: JsonMap) -> tuple[PoweredArm | None, JsonDict]:
    common_ok, detail = _payload_common_ok(payload)
    delta = _number(payload.get("delta_vs_tuned_sc"))
    accuracy = _number(payload.get("calibrated_accuracy"))
    available = (
        common_ok
        and payload.get("calibration_available") is True
        and payload.get("degeneracy_guard_fired") is False
        and delta is not None
        and accuracy is not None
    )
    if not available:
        reason = detail if not common_ok else "kan_purm_calibration_degenerate_or_unavailable"
        return None, _arm_check("D3", EXP5047_RELATIVE_PATH, False, reason)
    return (
        PoweredArm(
            arm="D3",
            source=EXP5047_RELATIVE_PATH,
            scorer_kind="kan_purm_calibrated_d1",
            delta_vs_tuned_sc=float(delta),
            selection_accuracy=float(accuracy),
            artifact_path=path,
            model_specs=dict(payload.get("model_specs") or {}),
        ),
        _arm_check("D3", EXP5047_RELATIVE_PATH, True, f"musr_delta={delta:.6f}"),
    )


def select_best_powered_arm(root: Path = REPO_ROOT) -> tuple[PoweredArm | None, list[JsonDict]]:
    """REQ-VERIFY-5049: select the best non-oracle, non-degenerate MuSR arm."""

    root = Path(root)
    specs = (
        (EXP5045_RELATIVE_PATH, _d1_from_payload),
        (EXP5046_RELATIVE_PATH, _d2_from_payload),
        (EXP5047_RELATIVE_PATH, _d3_from_payload),
    )
    candidates: list[PoweredArm] = []
    checks: list[JsonDict] = []
    for relative_path, builder in specs:
        path = root / relative_path
        payload = _read_json_object(path)
        if payload is None:
            arm = {"5045": "D1", "5046": "D2", "5047": "D3"}[relative_path[19:23]]
            checks.append(_arm_check(arm, relative_path, False, "artifact_unavailable"))
            continue
        candidate, check = builder(path, payload)
        checks.append(check)
        if candidate is not None:
            candidates.append(candidate)
    if not candidates:
        return None, checks
    return max(candidates, key=lambda item: (item.delta_vs_tuned_sc, item.arm)), checks


def render_candidate_score_text(row: JsonMap, candidate: JsonMap) -> str:
    """Render the non-oracle text passed to the powered scorer."""

    answer = str(candidate.get("answer") or "").strip()
    question = str(row.get("question") or "").strip()
    context = str(row.get("context") or "").strip()[:6000]
    text = f"Candidate answer: {answer}\nQuestion: {question}"
    if context:
        text += f"\nContext:\n{context}"
    return text


def sanitize_second_corpus_rows(rows: Sequence[JsonMap]) -> list[JsonDict]:
    """Remove candidate-level oracle fields before verifier scoring."""

    sanitized: list[JsonDict] = []
    for row in rows:
        candidates = []
        for candidate in row.get("candidates", []) or []:
            clean = {
                str(key): value
                for key, value in dict(candidate).items()
                if str(key) not in ORACLE_CANDIDATE_KEYS
            }
            clean["text"] = render_candidate_score_text(row, clean)
            candidates.append(clean)
        if candidates:
            new_row = dict(row)
            new_row.pop("label", None)
            new_row["candidates"] = candidates
            sanitized.append(new_row)
    return sanitized


def oracle_distinctness_self_check(rows: Sequence[JsonMap]) -> bool:
    """Verify the shared guard still rejects direct gold access."""

    try:
        evaluate_verifier(rows, scorer=lambda candidate: candidate["gold"], bootstrap_samples=8)
    except OracleDistinctnessError:
        return True
    return False  # pragma: no cover - indicates a shared harness regression.


def _resolve_cache_path(root: Path, artifact: JsonMap) -> Path:
    raw = str(artifact.get("candidate_cache_path") or EXP5044_CACHE_RELATIVE_PATH)
    path = Path(raw)
    return path if path.is_absolute() else root / path


def load_exp5044_rows(root: Path) -> tuple[JsonDict | None, list[JsonDict], Path | None, str | None]:
    """Load and validate the Exp 5044 headroom-present second-corpus rows."""

    root = Path(root)
    artifact = _read_json_object(root / EXP5044_RELATIVE_PATH)
    if artifact is None:
        return None, [], None, "second_corpus_cache_unavailable"
    if artifact.get("verifier_is_oracle") is not False:
        return artifact, [], None, "second_corpus_oracle_tainted"
    if artifact.get("headroom_present") is not True:
        return artifact, [], None, "second_corpus_not_headroom_present"
    if artifact.get("second_corpus_cache_built") is not True:
        return artifact, [], None, "second_corpus_cache_not_built"
    cache_path = _resolve_cache_path(root, artifact)
    rows = sanitize_second_corpus_rows(_read_jsonl(cache_path))
    if not rows:
        return artifact, [], cache_path, "second_corpus_cache_empty"
    return artifact, rows, cache_path, None


def default_score_fn(checkpoint: Any, texts: list[str]) -> list[float]:  # pragma: no cover - live
    config = d1.TrainingConfig(seed=RANDOM_SEED)
    return list(d1.default_score_fn(config)(checkpoint, texts))


def _d1_energy_by_id(arm: PoweredArm, rows: Sequence[JsonMap], score_fn: ScoreFn) -> dict[str, float]:
    candidate_ids: list[str] = []
    texts: list[str] = []
    for row in rows:
        for candidate in row.get("candidates", []) or []:
            candidate_ids.append(str(candidate.get("candidate_id") or ""))
            texts.append(str(candidate.get("text") or ""))
    energies = list(score_fn(arm.checkpoint_path or "", texts))
    if len(energies) != len(candidate_ids):
        raise RuntimeError(f"score_fn returned {len(energies)} energies for {len(candidate_ids)} candidates")
    return {candidate_id: float(energy) for candidate_id, energy in zip(candidate_ids, energies)}


def evaluate_second_corpus(
    rows: Sequence[JsonMap],
    *,
    arm: PoweredArm,
    score_fn: ScoreFn,
    seed: int = RANDOM_SEED,
    bootstrap_samples: int = 2000,
) -> JsonDict:
    """SCENARIO-VERIFY-5049: evaluate selected powered verifier vs tuned-SC."""

    if arm.scorer_kind == "lora_ebm_lookup":
        energy_by_id = _d1_energy_by_id(arm, rows, score_fn)
        return evaluate_verifier(
            rows,
            scorer=d1.make_lookup_scorer(energy_by_id),
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            headroom_threshold=harness.HEADROOM_THRESHOLD,
        )
    if arm.scorer_kind == "dense_process_reward":  # pragma: no cover - current evidence does not select D2
        prepared = d2.prepare_rows_with_process_rewards(rows)
        return evaluate_verifier(
            prepared,
            scorer=d2.dense_process_reward_energy,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
            headroom_threshold=harness.HEADROOM_THRESHOLD,
        )
    raise RuntimeError(f"selected arm {arm.arm} has no deployable second-corpus scorer")


def _ci_excludes_zero_positive(ci95: Sequence[Any]) -> bool:
    return len(ci95) == 2 and float(ci95[0]) > 0.0 and float(ci95[1]) > 0.0


def _format_delta(delta: float) -> str:
    return f"{delta:+.3f}".replace("+", "plus_").replace("-", "minus_").replace(".", "p")


def _checksum(artifact: JsonMap) -> str:
    basis = {
        "experiment_id": artifact.get("experiment_id"),
        "honest_verdict": artifact.get("honest_verdict"),
        "best_arm": artifact.get("best_arm"),
        "second_corpus_name": artifact.get("second_corpus_name"),
        "delta_vs_tuned_sc_second": artifact.get("delta_vs_tuned_sc_second"),
        "paired_ci95_second": artifact.get("paired_ci95_second"),
    }
    return "sha256:" + hashlib.sha256(_json_dumps(basis).encode("utf-8")).hexdigest()


def _base_artifact(
    *,
    honest_verdict: str,
    root: Path,
    duration_s: float,
    second_corpus: JsonMap | None,
    best_arm: PoweredArm | None,
    candidate_cache_path: Path | None,
    arm_checks: Sequence[JsonMap],
    blocked_error: str | None = None,
) -> JsonDict:
    candidate_specs = dict(second_corpus.get("model_specs") or {}) if second_corpus else {}
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT_NAME,
        "experiment_id": EXPERIMENT_ID,
        "spec_refs": list(SPEC_REFS),
        "result_path": str(root / RESULT_RELATIVE_PATH),
        "honest_verdict": honest_verdict,
        "model_specs": {
            "candidate_rows": candidate_specs,
            "best_powered_verifier": best_arm.model_specs if best_arm else {},
            "inherited_candidate_models": EXP5044_RELATIVE_PATH,
            "inherited_verifier_models": best_arm.source if best_arm else None,
        },
        "second_corpus_name": second_corpus.get("second_corpus_name") if second_corpus else None,
        "second_corpus_confirmed": False,
        "best_arm": best_arm.arm if best_arm else None,
        "best_arm_source": best_arm.source if best_arm else None,
        "n_questions_second": 0,
        "genuine_sc_accuracy_second": None,
        "verifier_accuracy_second": None,
        "delta_vs_tuned_sc_second": None,
        "paired_ci95_second": None,
        "mcnemar_p_second": None,
        "headroom_present": bool(second_corpus.get("headroom_present")) if second_corpus else False,
        "verifier_is_oracle": False,
        "candidate_cache_path": candidate_cache_path.as_posix() if candidate_cache_path else None,
        "source_artifacts": {
            "second_corpus": EXP5044_RELATIVE_PATH,
            "best_arm": best_arm.source if best_arm else None,
        },
        "oracle_distinctness_enforced": False,
        "upstream_arm_checks": list(arm_checks),
        "duration_s": round(float(duration_s), 6),
        "reproducibility_checksum": "",
    }
    if blocked_error:
        artifact["blocked_error"] = blocked_error[:1000]
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def _complete_artifact(
    *,
    root: Path,
    duration_s: float,
    second_corpus: JsonMap,
    best_arm: PoweredArm,
    candidate_cache_path: Path,
    rows: Sequence[JsonMap],
    evaluation: JsonMap,
    arm_checks: Sequence[JsonMap],
) -> JsonDict:
    tuned = dict(evaluation.get("tuned_self_consistency") or {})
    verifier = dict(evaluation.get("verifier") or {})
    tuned_accuracy = float(tuned.get("accuracy") or 0.0)
    verifier_accuracy = float(verifier.get("accuracy") or 0.0)
    delta = float(evaluation.get("verifier_minus_tuned_sc_delta") or 0.0)
    ci95 = [float(value) for value in evaluation.get("verifier_minus_tuned_sc_ci95", [0.0, 0.0])]
    confirmed = bool(evaluation.get("headroom_present")) and delta > 0.0 and _ci_excludes_zero_positive(ci95)
    corpus_slug = _slug(str(second_corpus.get("second_corpus_name") or "second_corpus"))
    verdict = (
        f"success_second_corpus_confirms_musr_margin_{corpus_slug}_{_format_delta(delta)}"
        if confirmed
        else f"complete_second_corpus_musr_margin_did_not_transfer_{corpus_slug}_{_format_delta(delta)}"
    )
    artifact = _base_artifact(
        honest_verdict=verdict,
        root=root,
        duration_s=duration_s,
        second_corpus=second_corpus,
        best_arm=best_arm,
        candidate_cache_path=candidate_cache_path,
        arm_checks=arm_checks,
    )
    artifact.update(
        {
            "second_corpus_confirmed": confirmed,
            "n_questions_second": int(evaluation.get("n_rows") or len(rows)),
            "genuine_sc_accuracy_second": round(tuned_accuracy, 6),
            "verifier_accuracy_second": round(verifier_accuracy, 6),
            "delta_vs_tuned_sc_second": round(delta, 6),
            "paired_ci95_second": ci95,
            "mcnemar_p_second": float(evaluation.get("mcnemar_p") or 0.0),
            "headroom_present": bool(evaluation.get("headroom_present")),
            "oracle_distinctness_enforced": True,
            "evaluation": evaluation,
            "best_arm_validation": {
                "musr_delta_vs_tuned_sc": best_arm.delta_vs_tuned_sc,
                "musr_selection_accuracy": best_arm.selection_accuracy,
                "scorer_kind": best_arm.scorer_kind,
            },
        }
    )
    artifact["reproducibility_checksum"] = _checksum(artifact)
    return artifact


def run(
    *,
    root: Path = REPO_ROOT,
    artifact_path: Path | None = None,
    score_fn: ScoreFn = default_score_fn,
    bootstrap_samples: int = 2000,
    now: Clock = time.monotonic,
    write: bool = True,
) -> JsonDict:
    """Run Exp 5049 and optionally write the terminal artifact."""

    root = Path(root)
    artifact_path = Path(artifact_path) if artifact_path else root / RESULT_RELATIVE_PATH
    start = float(now())
    second_corpus, rows, cache_path, cache_error = load_exp5044_rows(root)
    best_arm, arm_checks = select_best_powered_arm(root)

    if cache_error is not None:
        artifact = _base_artifact(
            honest_verdict=f"blocked_{cache_error}",
            root=root,
            duration_s=float(now()) - start,
            second_corpus=second_corpus,
            best_arm=best_arm,
            candidate_cache_path=cache_path,
            arm_checks=arm_checks,
            blocked_error=cache_error,
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    if best_arm is None:
        artifact = _base_artifact(
            honest_verdict="blocked_no_non_degenerate_powered_verifier",
            root=root,
            duration_s=float(now()) - start,
            second_corpus=second_corpus,
            best_arm=None,
            candidate_cache_path=cache_path,
            arm_checks=arm_checks,
            blocked_error="no D1/D2/D3 powered verifier passed non-oracle non-degenerate gates",
        )
        if write:
            write_json(artifact_path, artifact)
        return artifact
    assert second_corpus is not None
    assert cache_path is not None
    try:
        if not oracle_distinctness_self_check(rows):
            raise OracleDistinctnessError("shared guard did not block gold access")
        evaluation = evaluate_second_corpus(
            rows,
            arm=best_arm,
            score_fn=score_fn,
            seed=RANDOM_SEED,
            bootstrap_samples=bootstrap_samples,
        )
        artifact = _complete_artifact(
            root=root,
            duration_s=float(now()) - start,
            second_corpus=second_corpus,
            best_arm=best_arm,
            candidate_cache_path=cache_path,
            rows=rows,
            evaluation=evaluation,
            arm_checks=arm_checks,
        )
    except Exception as exc:
        artifact = _base_artifact(
            honest_verdict="blocked_second_corpus_scoring_unavailable",
            root=root,
            duration_s=float(now()) - start,
            second_corpus=second_corpus,
            best_arm=best_arm,
            candidate_cache_path=cache_path,
            arm_checks=arm_checks,
            blocked_error=f"{type(exc).__name__}: {exc}",
        )
    if write:
        write_json(artifact_path, artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    """Return terminal schema errors; empty means valid."""

    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(field)
    if artifact.get("schema") != SCHEMA:
        errors.append("schema")
    if artifact.get("experiment") != EXPERIMENT_NAME:
        errors.append("experiment")
    if artifact.get("spec_refs") != SPEC_REFS:
        errors.append("spec_refs")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle")
    for field in ("headroom_present", "second_corpus_confirmed", "oracle_distinctness_enforced"):
        if not isinstance(artifact.get(field), bool):
            errors.append(field)
    if not isinstance(artifact.get("model_specs"), Mapping):
        errors.append("model_specs")
    if not isinstance(artifact.get("source_artifacts"), Mapping):
        errors.append("source_artifacts")
    if not isinstance(artifact.get("n_questions_second"), int) or int(
        artifact.get("n_questions_second", -1)
    ) < 0:
        errors.append("n_questions_second")
    for field in ("genuine_sc_accuracy_second", "verifier_accuracy_second", "mcnemar_p_second"):
        value = artifact.get(field)
        if value is not None and not (
            isinstance(value, (int, float)) and 0.0 <= float(value) <= 1.0
        ):
            errors.append(field)
    delta = artifact.get("delta_vs_tuned_sc_second")
    if delta is not None and not isinstance(delta, (int, float)):
        errors.append("delta_vs_tuned_sc_second")
    ci95 = artifact.get("paired_ci95_second")
    if ci95 is not None and (
        not isinstance(ci95, list)
        or len(ci95) != 2
        or not all(isinstance(value, (int, float)) for value in ci95)
    ):
        errors.append("paired_ci95_second")
    verdict = str(artifact.get("honest_verdict") or "")
    if not verdict.startswith(("blocked_", "complete_", "success_")):
        errors.append("honest_verdict")
    return sorted(set(errors))


def main(argv: Sequence[str] | None = None) -> int:  # pragma: no cover - CLI entrypoint
    _ = argv
    artifact = run()
    errors = artifact_schema_errors(artifact)
    print(
        json.dumps(
            {
                "result_path": str(REPO_ROOT / RESULT_RELATIVE_PATH),
                "honest_verdict": artifact.get("honest_verdict"),
                "best_arm": artifact.get("best_arm"),
                "second_corpus_confirmed": artifact.get("second_corpus_confirmed"),
                "delta_vs_tuned_sc_second": artifact.get("delta_vs_tuned_sc_second"),
                "paired_ci95_second": artifact.get("paired_ci95_second"),
                "mcnemar_p_second": artifact.get("mcnemar_p_second"),
            },
            sort_keys=True,
        )
    )
    if errors:
        print(f"schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main(sys.argv[1:]))
