"""Exp 1395 FR-11 self-learning v5 with the DVI v2 + SECL verifier.

The v5 run treats Exp 1394's combined checkpoint as the active semantic
verifier.  It scans the local FoVer corpus without fresh LLM inference, builds
deterministic certificate states from the FoVer labels, and promotes only rows
where the DVI v2 prediction agrees with that certificate state after SECL
confidence calibration.  Exp 1388's promoted IDs are excluded so the reported
fresh count measures new memory rather than replaying the prior 59 cases.

Spec: REQ-LEARN-1395, SCENARIO-LEARN-1395.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from carnot.reporting import dvi_discriminative_verifier_training_v1 as dvi
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1388_FILE = "experiment_1388_fr11_self_learning_v4_dvi_grpo_integration.json"
EXP1393_FILE = "experiment_1393_grpo_v8_ngrpo_zero_reward_fix.json"
EXP1394_FILE = "experiment_1394_dvi_v2_secl_combined.json"
OUTPUT_FILE = "experiment_1395_fr11_self_learning_v5.json"

DEFAULT_EXP1388_PATH = DEFAULT_RESULTS_DIR / EXP1388_FILE
DEFAULT_EXP1393_PATH = DEFAULT_RESULTS_DIR / EXP1393_FILE
DEFAULT_EXP1394_PATH = DEFAULT_RESULTS_DIR / EXP1394_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1395_fr11_self_learning_v5"
SCHEMA = "fr11_self_learning_v5_dvi_v2_secl_v1"
RUN_DATE = "20260506"
EXP1388_BASELINE_FRESH_VERIFIED_COUNT = 59
DVI_INCORRECT_THRESHOLD = 0.72
SECL_CONFIDENCE_THRESHOLD = 0.5

PATH_DVI_V2_ONLY = "dvi_v2_secl_fover_self_learning"
PATH_DVI_V2_GRPO = "dvi_v2_secl_fover_self_learning_grpo_v8"

POLICY_PROMOTE = "promote"
POLICY_DEMOTE = "demote"
STATE_SAT = "SAT"
STATE_REPAIR_HINT = "REPAIR_HINT"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "path_used",
    "dvi_v2_checkpoint_active",
    "replay_cases_used",
    "fresh_verified_sample_count",
    "grpo_v8_cases_integrated",
    "self_learning_delta_overall",
    "headline_result_allowed",
    "honest_verdict",
)


@dataclass(frozen=True)
class DviV2CheckpointState:
    """Readable DVI v2 + SECL state extracted from the Exp 1394 checkpoint."""

    checkpoint_path: str
    metric: np.ndarray
    bias: float
    secl_bin_values: np.ndarray
    secl_global_value: float
    secl_n_bins: int
    source_fresh_cases_used: int


@dataclass(frozen=True)
class DviV2Activation:
    """Activation result that keeps checkpoint blockers explicit in artifacts."""

    active: bool
    path: str | None
    blocker: str | None
    state: DviV2CheckpointState | None


@dataclass(frozen=True)
class FoVerSelfLearningCase:
    """One FoVer row normalized for the v5 self-learning replay path."""

    case_id: str
    question: str
    response: str
    is_incorrect: bool
    source: str

    @property
    def certificate_state(self) -> str:
        """Return the deterministic certificate state implied by the FoVer label."""

        return STATE_REPAIR_HINT if self.is_incorrect else STATE_SAT


def _write_json(path: Path | str, artifact: Mapping[str, Any]) -> dict[str, Any]:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(artifact)
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return payload


def _metadata(project_root: str | Path, run_date: str) -> dict[str, str]:
    return {"project_root": str(project_root), "run_date": run_date}


def write_in_progress_artifact(
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """REQ-LEARN-1395-1: persist a visible bootstrap artifact first."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "source_artifacts": [],
            "status": "in_progress",
            "path_used": None,
            "dvi_v2_checkpoint_active": False,
            "dvi_v2_checkpoint_path": None,
            "dvi_v2_checkpoint_blocker": None,
            "replay_cases_used": 0,
            "fresh_verified_sample_count": 0,
            "grpo_v8_cases_integrated": 0,
            "self_learning_delta_overall": 0,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load an artifact and reject non-object JSON payloads."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")
    return payload


def activate_dvi_v2_checkpoint(exp1394_artifact: Mapping[str, Any]) -> DviV2Activation:
    """REQ-LEARN-1395-2: load the combined DVI v2 + SECL checkpoint."""

    raw_path = exp1394_artifact.get("checkpoint_path")
    if exp1394_artifact.get("dvi_v2_deployed") is not True:
        return DviV2Activation(
            active=False,
            path=str(raw_path) if raw_path else None,
            blocker="exp1394_dvi_v2_not_deployed",
            state=None,
        )
    if not raw_path:
        return DviV2Activation(
            active=False,
            path=None,
            blocker="exp1394_checkpoint_path_missing",
            state=None,
        )
    path = Path(str(raw_path))
    if not path.exists():
        return DviV2Activation(
            active=False,
            path=str(path),
            blocker="dvi_v2_checkpoint_file_missing",
            state=None,
        )

    try:
        with np.load(path, allow_pickle=False) as data:
            required = {"metric", "bias", "secl_bin_values", "secl_global_value", "secl_n_bins"}
            missing = sorted(required.difference(data.files))
            if missing:
                return DviV2Activation(
                    active=False,
                    path=str(path),
                    blocker=f"dvi_v2_checkpoint_missing_fields:{','.join(missing)}",
                    state=None,
                )
            metric = np.asarray(data["metric"], dtype=np.float32)
            bias = float(np.asarray(data["bias"], dtype=np.float32).reshape(-1)[0])
            bin_values = np.asarray(data["secl_bin_values"], dtype=np.float64)
            global_value = float(
                np.asarray(data["secl_global_value"], dtype=np.float64).reshape(-1)[0]
            )
            n_bins = int(np.asarray(data["secl_n_bins"], dtype=np.int32).reshape(-1)[0])
            fresh_cases = (
                int(np.asarray(data["fresh_cases_used"], dtype=np.int32).reshape(-1)[0])
                if "fresh_cases_used" in data.files
                else _int(exp1394_artifact.get("fresh_cases_used"))
            )
    except Exception as exc:
        return DviV2Activation(
            active=False,
            path=str(path),
            blocker=f"dvi_v2_checkpoint_unreadable:{type(exc).__name__}",
            state=None,
        )

    if metric.ndim != 1 or metric.size == 0:
        return DviV2Activation(
            active=False,
            path=str(path),
            blocker="dvi_v2_checkpoint_metric_invalid",
            state=None,
        )
    if n_bins <= 0 or bin_values.size < n_bins:
        return DviV2Activation(
            active=False,
            path=str(path),
            blocker="dvi_v2_checkpoint_secl_head_invalid",
            state=None,
        )

    return DviV2Activation(
        active=True,
        path=str(path),
        blocker=None,
        state=DviV2CheckpointState(
            checkpoint_path=str(path),
            metric=metric,
            bias=bias,
            secl_bin_values=bin_values[:n_bins],
            secl_global_value=global_value,
            secl_n_bins=n_bins,
            source_fresh_cases_used=fresh_cases,
        ),
    )


def exp1388_fresh_case_ids(exp1388_artifact: Mapping[str, Any]) -> set[str]:
    """REQ-LEARN-1395-3: return Exp 1388's already promoted DVI case IDs."""

    promoted = exp1388_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        return set()
    prefix = "dvi:exp1382:"
    return {str(item)[len(prefix) :] for item in promoted if str(item).startswith(prefix)}


def normalize_fover_cases(rows: Sequence[Mapping[str, Any]]) -> list[FoVerSelfLearningCase]:
    """Normalize labeled FoVer rows while preserving Exp 1382-style duplicate IDs."""

    cases: list[FoVerSelfLearningCase] = []
    seen: dict[str, int] = {}
    for index, row in enumerate(rows):
        text = dvi.row_text(row)
        correctness = secl.row_is_correct(row)
        if not text or correctness is None:
            continue
        raw_id = str(
            row.get("question_id")
            or row.get("case_id")
            or row.get("id")
            or row.get("question_index")
            or f"fover_{index}"
        )
        ordinal = seen.get(raw_id, 0)
        seen[raw_id] = ordinal + 1
        case_id = raw_id if ordinal == 0 else f"{raw_id}_{ordinal}"
        cases.append(
            FoVerSelfLearningCase(
                case_id=case_id,
                question=str(row.get("question") or row.get("prompt") or ""),
                response=text,
                is_incorrect=not correctness,
                source=str(row.get("source") or "fover_corpus"),
            )
        )
    return cases


def sample_fresh_fover_cases(
    rows: Sequence[Mapping[str, Any]],
    *,
    exclude_case_ids: set[str] | None = None,
    max_cases: int | None = None,
) -> list[FoVerSelfLearningCase]:
    """Sample fresh FoVer rows deterministically after removing replayed IDs."""

    excluded = exclude_case_ids or set()
    cases = [case for case in normalize_fover_cases(rows) if case.case_id not in excluded]
    if max_cases is None:
        return cases
    return cases[: max(0, int(max_cases))]


def verify_cases_with_dvi_v2(
    cases: Sequence[FoVerSelfLearningCase],
    state: DviV2CheckpointState,
    *,
    incorrect_threshold: float = DVI_INCORRECT_THRESHOLD,
    secl_confidence_threshold: float = SECL_CONFIDENCE_THRESHOLD,
) -> list[dict[str, Any]]:
    """REQ-LEARN-1395-4: verify generated certificate states with DVI v2 + SECL."""

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    variants: list[dict[str, Any]] = []
    for case in cases:
        incorrect_probability = dvi.predict_incorrect_probability(
            verifier,
            state.metric,
            state.bias,
            case.response,
        )
        dvi_predicts_incorrect = incorrect_probability >= float(incorrect_threshold)
        semantic_result = STATE_REPAIR_HINT if dvi_predicts_incorrect else STATE_SAT
        predicted_state_probability = (
            incorrect_probability if dvi_predicts_incorrect else 1.0 - incorrect_probability
        )
        secl_confidence = secl_confidence_for_probability(state, predicted_state_probability)
        certificate_state = case.certificate_state
        constraint_passed = certificate_state == semantic_result and secl_confidence >= float(
            secl_confidence_threshold
        )
        variants.append(
            {
                "variant_id": f"dvi_v2:fover:{case.case_id}",
                "source": "fover_dvi_v2_secl_semantic_validation",
                "case_id": case.case_id,
                "memory_action": POLICY_PROMOTE if constraint_passed else POLICY_DEMOTE,
                "support": 1,
                "certificate_generation_method": "fover_label_conditioned_certificate_state",
                "certificate_answer": generated_certificate_answer(certificate_state),
                "dvi_score_source": "exp1394_dvi_v2_secl_checkpoint",
                "evidence_summary": {
                    "expected_state": certificate_state,
                    "certificate_state": certificate_state,
                    "semantic_result": semantic_result,
                    "constraint_evaluated": True,
                    "constraint_passed": constraint_passed,
                    "dvi_incorrect_probability": round(float(incorrect_probability), 6),
                    "dvi_incorrect_threshold": round(float(incorrect_threshold), 6),
                    "secl_predicted_state_confidence": round(float(secl_confidence), 6),
                    "secl_confidence_threshold": round(float(secl_confidence_threshold), 6),
                    "fover_label": "incorrect" if case.is_incorrect else "correct",
                    "source": case.source,
                    "failure_reason": None
                    if constraint_passed
                    else _semantic_failure_reason(
                        certificate_state=certificate_state,
                        semantic_result=semantic_result,
                        secl_confidence=secl_confidence,
                        secl_confidence_threshold=secl_confidence_threshold,
                    ),
                },
            }
        )
    return variants


def generated_certificate_answer(state: str) -> str:
    """Return a small deterministic certificate answer for the replayed FoVer row."""

    return json.dumps({"certificate_state": state}, sort_keys=True)


def secl_confidence_for_probability(state: DviV2CheckpointState, probability: float) -> float:
    """Apply the fixed-bin SECL confidence head saved inside the checkpoint."""

    clipped = min(1.0, max(0.0, float(probability)))
    index = min(state.secl_n_bins - 1, int(np.floor(clipped * state.secl_n_bins)))
    if 0 <= index < state.secl_bin_values.size:
        return float(state.secl_bin_values[index])
    return float(state.secl_global_value)


def build_grpo_v8_memory_variants(exp1393_artifact: Mapping[str, Any]) -> list[dict[str, Any]]:
    """REQ-LEARN-1395-5: integrate only verifier-matched positive GRPO v8 rows."""

    variants: list[dict[str, Any]] = []
    seen: set[str] = set()
    for row in _rows(exp1393_artifact, ("training_reward_rows",)):
        if not _training_grpo_row_verified(row):
            continue
        case_id = str(row.get("case_id") or f"training_{len(variants)}")
        variant = _grpo_variant(case_id=case_id, row=row, source_stage="training_reward_rows")
        if variant["variant_id"] not in seen:
            variants.append(variant)
            seen.add(variant["variant_id"])
    for row in _rows(exp1393_artifact, ("heldout_evaluation_rows",)):
        if not _heldout_grpo_row_verified(row):
            continue
        case_id = str(row.get("case_id") or f"heldout_{len(variants)}")
        variant = _grpo_variant(case_id=case_id, row=row, source_stage="heldout_evaluation_rows")
        if variant["variant_id"] not in seen:
            variants.append(variant)
            seen.add(variant["variant_id"])
    return variants


def build_artifact(
    *,
    exp1388_artifact: Mapping[str, Any],
    exp1393_artifact: Mapping[str, Any],
    exp1394_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    source_artifacts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1395: build the terminal v5 self-learning artifact."""

    activation = activate_dvi_v2_checkpoint(exp1394_artifact)
    grpo_improvement = _float(exp1393_artifact.get("grpo_v8_improvement_pp"), 0.0)
    path_used = PATH_DVI_V2_GRPO if grpo_improvement > 0.0 else PATH_DVI_V2_ONLY
    baseline_count = _int(
        exp1388_artifact.get("fresh_verified_sample_count"),
        EXP1388_BASELINE_FRESH_VERIFIED_COUNT,
    )
    replay_cases_used = _int(exp1388_artifact.get("replay_cases_used"))
    dvi_variants: list[dict[str, Any]] = []
    fresh_cases: list[FoVerSelfLearningCase] = []

    if activation.active and activation.state is not None:
        fresh_cases = sample_fresh_fover_cases(
            fover_rows,
            exclude_case_ids=exp1388_fresh_case_ids(exp1388_artifact),
        )
        dvi_variants = verify_cases_with_dvi_v2(fresh_cases, activation.state)

    grpo_variants = (
        build_grpo_v8_memory_variants(exp1393_artifact) if grpo_improvement > 0.0 else []
    )
    variants = dvi_variants + grpo_variants
    memory_updates = apply_memory_updates(variants)
    dvi_verified_count = _support_count(
        variant for variant in dvi_variants if variant.get("memory_action") == POLICY_PROMOTE
    )
    grpo_v8_cases_integrated = _support_count(
        variant for variant in grpo_variants if variant.get("memory_action") == POLICY_PROMOTE
    )
    fresh_verified_sample_count = dvi_verified_count + grpo_v8_cases_integrated
    self_learning_delta = fresh_verified_sample_count - baseline_count
    headline_allowed = (
        activation.active and fresh_verified_sample_count > EXP1388_BASELINE_FRESH_VERIFIED_COUNT
    )
    status = "complete" if activation.active else "blocked"
    artifact = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "artifact_metadata": _metadata(project_root, run_date),
        "run_date": run_date,
        "started_at": started_at or datetime.now(tz=UTC).isoformat(),
        "finished_at": datetime.now(tz=UTC).isoformat(),
        "duration_s": round(float(duration_s), 3),
        "status": status,
        "spec": ["REQ-LEARN-1395", "SCENARIO-LEARN-1395"],
        "source_artifacts": list(source_artifacts or _default_source_artifacts()),
        "source_honest_verdicts": {
            "exp1388": exp1388_artifact.get("honest_verdict"),
            "exp1393": exp1393_artifact.get("honest_verdict"),
            "exp1394": exp1394_artifact.get("honest_verdict"),
        },
        "path_used": path_used,
        "dvi_v2_checkpoint_active": activation.active,
        "dvi_v2_checkpoint_path": activation.path,
        "dvi_v2_checkpoint_blocker": activation.blocker,
        "dvi_v2_checkpoint_source_fresh_cases_used": (
            activation.state.source_fresh_cases_used if activation.state is not None else None
        ),
        "dvi_incorrect_threshold": DVI_INCORRECT_THRESHOLD,
        "secl_confidence_threshold": SECL_CONFIDENCE_THRESHOLD,
        "replay_cases_used": replay_cases_used,
        "exp1388_baseline_fresh_verified_sample_count": baseline_count,
        "exp1388_prior_fresh_ids_excluded": len(exp1388_fresh_case_ids(exp1388_artifact)),
        "fover_rows_available": len(normalize_fover_cases(fover_rows)),
        "fover_rows_sampled": len(fresh_cases),
        "dvi_v2_verified_fover_case_count": dvi_verified_count,
        "fresh_verified_sample_count": fresh_verified_sample_count,
        "grpo_v8_cases_integrated": grpo_v8_cases_integrated,
        "grpo_v8_improvement_pp": grpo_improvement,
        "self_learning_delta_overall": self_learning_delta,
        "self_learning_delta_ratio_vs_exp1388": _ratio_delta(
            fresh_verified_sample_count,
            baseline_count,
        ),
        "headline_result_allowed": headline_allowed,
        "memory_updates": memory_updates,
        "promoted_memory_count": memory_updates["promoted_memory_count"],
        "demoted_memory_count": memory_updates["demoted_memory_count"],
        "certificate_generation_method": "fover_label_conditioned_certificate_state",
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": derive_honest_verdict(
            status=status,
            headline_result_allowed=headline_allowed,
            fresh_verified_sample_count=fresh_verified_sample_count,
            self_learning_delta=self_learning_delta,
            grpo_v8_cases_integrated=grpo_v8_cases_integrated,
        ),
        "measurement_note": (
            "Exp 1395 activates the Exp 1394 DVI v2 + SECL checkpoint and scans "
            "the local FoVer corpus without fresh LLM inference. Exp 1388's 59 "
            "DVI-promoted IDs are excluded from the fresh count. GRPO v8 cases "
            "are integrated only when Exp 1393 reports positive improvement."
        ),
    }
    validate_artifact(artifact)
    return artifact


def run(
    *,
    exp1388_path: Path | str = DEFAULT_EXP1388_PATH,
    exp1393_path: Path | str = DEFAULT_EXP1393_PATH,
    exp1394_path: Path | str = DEFAULT_EXP1394_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run Exp 1395 end-to-end and write the final artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1388 = load_json(exp1388_path)
    exp1393 = load_json(exp1393_path)
    exp1394 = load_json(exp1394_path)
    fover_rows = dvi.load_jsonl_rows(fover_path)
    artifact = build_artifact(
        exp1388_artifact=exp1388,
        exp1393_artifact=exp1393,
        exp1394_artifact=exp1394,
        fover_rows=fover_rows,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        source_artifacts=[
            f"results/{EXP1388_FILE}",
            f"results/{EXP1393_FILE}",
            f"results/{EXP1394_FILE}",
            "data/fover_corpus.jsonl",
        ],
    )
    return _write_json(out_path, artifact)


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Count support-weighted promotions and demotions for the memory pool."""

    promoted: list[str] = []
    demoted: list[str] = []
    promoted_count = 0
    demoted_count = 0
    for variant in variants:
        variant_id = str(variant.get("variant_id") or variant.get("case_id") or "unknown")
        support = max(_int(variant.get("support")), 1)
        if variant.get("memory_action") == POLICY_PROMOTE:
            promoted.append(variant_id)
            promoted_count += support
        elif variant.get("memory_action") == POLICY_DEMOTE:
            demoted.append(variant_id)
            demoted_count += support
    return {
        "promoted": promoted,
        "demoted": demoted,
        "promoted_memory_count": promoted_count,
        "demoted_memory_count": demoted_count,
    }


def derive_honest_verdict(
    *,
    status: str,
    headline_result_allowed: bool,
    fresh_verified_sample_count: int,
    self_learning_delta: int,
    grpo_v8_cases_integrated: int,
) -> str:
    """Name the active path and headline boundary without overstating the result."""

    if status == "blocked":
        return "fr11_self_learning_v5_blocked_dvi_v2_checkpoint_inactive"
    if headline_result_allowed:
        return (
            "fr11_self_learning_v5_dvi_v2_secl_headline_allowed_"
            f"fresh_{fresh_verified_sample_count}_delta_{self_learning_delta}_"
            f"grpo_{grpo_v8_cases_integrated}"
        )
    if fresh_verified_sample_count <= EXP1388_BASELINE_FRESH_VERIFIED_COUNT:
        return "fr11_self_learning_v5_dvi_v2_secl_no_fresh_delta_non_headline"
    return "fr11_self_learning_v5_dvi_v2_secl_non_headline"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1395-6/7: enforce required fields and headline invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    if artifact["path_used"] not in {None, PATH_DVI_V2_ONLY, PATH_DVI_V2_GRPO}:
        raise AssertionError(f"unsupported path_used: {artifact['path_used']}")
    for field in (
        "replay_cases_used",
        "fresh_verified_sample_count",
        "grpo_v8_cases_integrated",
    ):
        if not isinstance(artifact[field], int) or artifact[field] < 0:
            raise AssertionError(f"{field} must be a non-negative integer")
    if artifact["headline_result_allowed"]:
        if artifact["dvi_v2_checkpoint_active"] is not True:
            raise AssertionError("headline_result_allowed requires active DVI v2 checkpoint")
        if artifact["fresh_verified_sample_count"] <= EXP1388_BASELINE_FRESH_VERIFIED_COUNT:
            raise AssertionError("headline_result_allowed requires fresh count > 59")
    if artifact["path_used"] == PATH_DVI_V2_ONLY and artifact["grpo_v8_cases_integrated"] != 0:
        raise AssertionError("DVI v2-only path cannot integrate GRPO v8 cases")
    if artifact["status"] == "complete" and artifact["dvi_v2_checkpoint_active"] is not True:
        raise AssertionError("complete status requires active DVI v2 checkpoint")


def _semantic_failure_reason(
    *,
    certificate_state: str,
    semantic_result: str,
    secl_confidence: float,
    secl_confidence_threshold: float,
) -> str:
    if certificate_state != semantic_result:
        return "dvi_v2_prediction_disagrees_with_certificate_state"
    if secl_confidence < secl_confidence_threshold:
        return "secl_confidence_below_threshold"
    return "unknown_semantic_gate_failure"


def _training_grpo_row_verified(row: Mapping[str, Any]) -> bool:
    expected = str(row.get("expected_answer") or "")
    verifier_result = str(row.get("verifier_result") or "")
    if not expected:
        return verifier_result == "VERIFIED"
    if verifier_result == expected or verifier_result == "VERIFIED":
        return True
    candidate = str(row.get("candidate_answer") or "")
    rewards = row.get("raw_rewards")
    has_positive_reward = (
        isinstance(rewards, Sequence)
        and not isinstance(rewards, (str, bytes))
        and any(_float(value, 0.0) > 0.0 for value in rewards)
    )
    return candidate == expected and has_positive_reward


def _heldout_grpo_row_verified(row: Mapping[str, Any]) -> bool:
    expected = str(row.get("expected_answer") or "")
    verifier_result = str(row.get("post_grpo_verifier_result") or "")
    if not expected or verifier_result != expected:
        return False
    answer = row.get("post_grpo_answer")
    return answer is None or str(answer) == expected


def _grpo_variant(
    *,
    case_id: str,
    row: Mapping[str, Any],
    source_stage: str,
) -> dict[str, Any]:
    return {
        "variant_id": f"grpo_v8:exp1393:{source_stage}:{case_id}",
        "source": "exp1393_grpo_v8",
        "case_id": case_id,
        "memory_action": POLICY_PROMOTE,
        "support": 1,
        "dvi_score_source": source_stage,
        "evidence_summary": dict(row),
    }


def _rows(artifact: Mapping[str, Any], keys: Sequence[str]) -> list[dict[str, Any]]:
    for key in keys:
        value = artifact.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            return [dict(row) for row in value if isinstance(row, Mapping)]
    return []


def _support_count(variants: Sequence[Mapping[str, Any]] | Any) -> int:
    return sum(max(_int(variant.get("support")), 1) for variant in variants)


def _default_source_artifacts() -> list[str]:
    return [
        f"results/{EXP1388_FILE}",
        f"results/{EXP1393_FILE}",
        f"results/{EXP1394_FILE}",
        "data/fover_corpus.jsonl",
    ]


def _ratio_delta(current: int, baseline: int) -> float:
    if baseline <= 0:
        return 0.0
    return round((current - baseline) / baseline, 6)


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


if __name__ == "__main__":
    print(json.dumps(run(), indent=2, sort_keys=True))
