"""Exp 1433 FR-11 self-learning v6 gated on the deployed DVI v3 checkpoint.

The v6 run is intentionally gate-first. Exp 1432 repaired DVI v3 deployment by
calibrating the replay nonforgetting threshold, so this module treats that
checkpoint as the active verifier only when the deployment flag, checkpoint
file, and nonforgetting rate are all present. It then scans FoVer rows not
already promoted by Exp 1395 and counts only new DVI v3 verified promotions as
fresh self-learning growth.

Spec: REQ-LEARN-1433, SCENARIO-LEARN-1433, SCENARIO-LEARN-1434.
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
from carnot.reporting import dvi_v3_1508_fresh_cases as dvi_v3
from carnot.reporting import fr11_self_learning_v5 as fr11
from carnot.reporting import secl_discriminative_self_calibration as secl
from carnot.verify.sc_energy_verifier import SCEnergyVerifier


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RESULTS_DIR = REPO_ROOT / "results"
DEFAULT_FOVER_PATH = REPO_ROOT / "data" / "fover_corpus.jsonl"

EXP1395_FILE = "experiment_1395_fr11_self_learning_v5.json"
EXP1432_FILE = "experiment_1432_dvi_v3_nonforgetting_replay_balanced.json"
OUTPUT_FILE = "experiment_1433_fr11_self_learning_v6_dvi_v3_gated.json"

DEFAULT_EXP1395_PATH = DEFAULT_RESULTS_DIR / EXP1395_FILE
DEFAULT_EXP1432_PATH = DEFAULT_RESULTS_DIR / EXP1432_FILE
DEFAULT_OUTPUT_PATH = DEFAULT_RESULTS_DIR / OUTPUT_FILE

EXPERIMENT = "1433_fr11_self_learning_v6_dvi_v3_gated"
SCHEMA = "fr11_self_learning_v6_dvi_v3_gated_v1"
RUN_DATE = "20260506"
EXP1395_BASELINE_COUNT = 1508
MIN_NONFORGETTING_RATE = 0.99
DVI_V3_ARTIFACT_USED = f"results/{EXP1432_FILE}"
PROMOTED_PREFIX = "dvi_v3:fover:"

REQUIRED_ARTIFACT_FIELDS = (
    "status",
    "dvi_v3_artifact_used",
    "fresh_verified_sample_count",
    "baseline_fresh_verified_sample_count",
    "self_learning_delta_overall",
    "nonforgetting_rate",
    "session_memory_updated",
    "headline_result_allowed",
    "honest_verdict",
)


@dataclass(frozen=True)
class DviV3CheckpointState:
    """Loaded DVI v3 verifier state plus the calibrated memory gate thresholds."""

    checkpoint_path: str
    metric: np.ndarray
    bias: float
    confidence_head: secl.HistogramECEConfidenceHead
    dvi_incorrect_threshold: float
    secl_confidence_threshold: float


@dataclass(frozen=True)
class DviV3Activation:
    """Gate result for deciding whether FR-11 v6 may run."""

    active: bool
    path: str | None
    blocker: str | None
    nonforgetting_rate: float | None
    state: DviV3CheckpointState | None


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
    """REQ-LEARN-1433-1: persist a visible bootstrap artifact before source loading."""

    return _write_json(
        out_path,
        {
            "experiment": EXPERIMENT,
            "schema": SCHEMA,
            "artifact_metadata": _metadata(project_root, run_date),
            "run_date": run_date,
            "status": "in_progress",
            "dvi_v3_artifact_used": None,
            "fresh_verified_sample_count": None,
            "baseline_fresh_verified_sample_count": EXP1395_BASELINE_COUNT,
            "self_learning_delta_overall": None,
            "nonforgetting_rate": None,
            "session_memory_updated": None,
            "headline_result_allowed": False,
            "honest_verdict": "in_progress",
            "fresh_llm_inference_used": False,
            "cpu_only": True,
        },
    )


def load_json(path: Path | str) -> dict[str, Any]:
    """Load one experiment artifact as a JSON object."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"artifact must be a JSON object: {path}")  # pragma: no cover
    return payload


def activate_dvi_v3_checkpoint(
    exp1432_artifact: Mapping[str, Any],
    *,
    min_nonforgetting_rate: float = MIN_NONFORGETTING_RATE,
) -> DviV3Activation:
    """REQ-LEARN-1433-2/3: load deployed DVI v3 or return an explicit blocker."""

    raw_path = exp1432_artifact.get("dvi_v3_checkpoint_path")
    nonforgetting_rate = float(exp1432_artifact.get("nonforgetting_rate") or 0.0)
    if exp1432_artifact.get("dvi_v3_deployed") is not True:
        return DviV3Activation(
            active=False,
            path=str(raw_path) if raw_path else None,
            blocker="exp1432_dvi_v3_not_deployed",
            nonforgetting_rate=nonforgetting_rate,
            state=None,
        )
    if nonforgetting_rate < float(min_nonforgetting_rate):
        return DviV3Activation(  # pragma: no cover
            active=False,
            path=str(raw_path) if raw_path else None,
            blocker="exp1432_nonforgetting_below_gate",
            nonforgetting_rate=nonforgetting_rate,
            state=None,
        )
    if not raw_path:
        return DviV3Activation(  # pragma: no cover
            active=False,
            path=None,
            blocker="exp1432_dvi_v3_checkpoint_path_missing",
            nonforgetting_rate=nonforgetting_rate,
            state=None,
        )

    path = Path(str(raw_path))
    if not path.exists():
        return DviV3Activation(  # pragma: no cover
            active=False,
            path=str(path),
            blocker="dvi_v3_checkpoint_file_missing",
            nonforgetting_rate=nonforgetting_rate,
            state=None,
        )

    with np.load(path, allow_pickle=False) as data:
        required = {"metric", "bias", "secl_bin_values", "secl_global_value", "secl_n_bins"}
        missing = sorted(required.difference(data.files))
        if missing:
            return DviV3Activation(  # pragma: no cover
                active=False,
                path=str(path),
                blocker=f"dvi_v3_checkpoint_missing_fields:{','.join(missing)}",
                nonforgetting_rate=nonforgetting_rate,
                state=None,
            )
        metric = np.asarray(data["metric"], dtype=np.float32)
        bias = float(np.asarray(data["bias"], dtype=np.float32).reshape(-1)[0])
        n_bins = int(np.asarray(data["secl_n_bins"], dtype=np.int32).reshape(-1)[0])
        confidence_head = secl.HistogramECEConfidenceHead(
            bin_values=np.asarray(data["secl_bin_values"], dtype=np.float64)[:n_bins],
            global_value=float(
                np.asarray(data["secl_global_value"], dtype=np.float64).reshape(-1)[0]
            ),
            n_bins=n_bins,
        )
        dvi_threshold = _scalar_field(
            data,
            "dvi_incorrect_threshold",
            fr11.DVI_INCORRECT_THRESHOLD,
        )
        secl_threshold = _scalar_field(
            data,
            "secl_confidence_threshold",
            fr11.SECL_CONFIDENCE_THRESHOLD,
        )

    return DviV3Activation(
        active=True,
        path=str(path),
        blocker=None,
        nonforgetting_rate=nonforgetting_rate,
        state=DviV3CheckpointState(
            checkpoint_path=str(path),
            metric=metric,
            bias=bias,
            confidence_head=confidence_head,
            dvi_incorrect_threshold=dvi_threshold,
            secl_confidence_threshold=secl_threshold,
        ),
    )


def exp1395_promoted_case_ids(exp1395_artifact: Mapping[str, Any]) -> set[str]:
    """REQ-LEARN-1433-4: return FoVer IDs already counted by v5."""

    promoted = exp1395_artifact.get("memory_updates", {}).get("promoted", [])
    if not isinstance(promoted, Sequence) or isinstance(promoted, (str, bytes)):
        raise ValueError("Exp 1395 memory_updates.promoted must be a list")  # pragma: no cover
    return {str(item).split(":")[-1] for item in promoted}


def candidate_cases_from_fover(
    fover_rows: Sequence[Mapping[str, Any]],
    *,
    exclude_case_ids: set[str],
) -> list[dvi.DviCase]:
    """Return labeled FoVer rows that v6 can evaluate for new memory promotion."""

    candidates: list[dvi.DviCase] = []
    for row in fover_rows:
        case_id = str(row.get("question_id") or "")
        if not case_id or case_id in exclude_case_ids:
            continue
        is_correct = secl.row_is_correct(row)
        text = secl.row_text(row)
        if is_correct is None or not text:
            continue  # pragma: no cover
        candidates.append(
            dvi.DviCase(
                case_id=case_id,
                text=text,
                label=0 if is_correct else 1,
                source=str(row.get("source") or "fover"),
            )
        )
    return candidates


def verify_cases_with_dvi_v3(
    cases: Sequence[dvi.DviCase],
    state: DviV3CheckpointState,
) -> list[dict[str, Any]]:
    """REQ-LEARN-1433-3: score v6 candidate cases with deployed DVI v3."""

    verifier = SCEnergyVerifier(model_name="deterministic", hidden_dim=int(state.metric.size))
    variants: list[dict[str, Any]] = []
    for case in cases:
        action = dvi_v3.memory_action_for_case(
            case,
            verifier=verifier,
            metric=state.metric,
            bias=state.bias,
            confidence_head=state.confidence_head,
            incorrect_threshold=state.dvi_incorrect_threshold,
            secl_confidence_threshold=state.secl_confidence_threshold,
        )
        variants.append(
            {
                "variant_id": f"{PROMOTED_PREFIX}{case.case_id}",
                "source": "exp1432_dvi_v3_fover_heldout_verification",
                "case_id": case.case_id,
                "memory_action": action,
                "support": 1,
            }
        )
    return variants


def apply_memory_updates(variants: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Collect v6 SessionMemory-equivalent promotions and demotions."""

    promoted = [
        str(variant["variant_id"])
        for variant in variants
        if variant.get("memory_action") == fr11.POLICY_PROMOTE
    ]
    demoted = [
        str(variant["variant_id"])
        for variant in variants
        if variant.get("memory_action") == fr11.POLICY_DEMOTE
    ]
    return {
        "promoted": promoted,
        "demoted": demoted,
        "promoted_memory_count": len(promoted),
        "demoted_memory_count": len(demoted),
    }


def build_artifact(
    *,
    exp1395_artifact: Mapping[str, Any],
    exp1432_artifact: Mapping[str, Any],
    fover_rows: Sequence[Mapping[str, Any]],
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
    started_at: str | None = None,
    duration_s: float = 0.0,
    source_artifacts: Sequence[str] | None = None,
) -> dict[str, Any]:
    """REQ-LEARN-1433: build the terminal v6 self-learning artifact."""

    activation = activate_dvi_v3_checkpoint(exp1432_artifact)
    baseline_count = int(
        exp1395_artifact.get("fresh_verified_sample_count", EXP1395_BASELINE_COUNT)
    )
    variants: list[dict[str, Any]] = []
    candidates: list[dvi.DviCase] = []
    if activation.active and activation.state is not None:
        candidates = candidate_cases_from_fover(
            fover_rows,
            exclude_case_ids=exp1395_promoted_case_ids(exp1395_artifact),
        )
        variants = verify_cases_with_dvi_v3(candidates, activation.state)

    memory_updates = apply_memory_updates(variants)
    new_promoted_count = int(memory_updates["promoted_memory_count"])
    fresh_verified_count = baseline_count + new_promoted_count
    delta = fresh_verified_count - baseline_count
    nonforgetting_rate = activation.nonforgetting_rate
    nonforgetting_preserved = bool(
        activation.active
        and nonforgetting_rate is not None
        and float(nonforgetting_rate) >= MIN_NONFORGETTING_RATE
    )
    headline_allowed = delta > 0 and nonforgetting_preserved
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
        "spec": ["REQ-LEARN-1433", "SCENARIO-LEARN-1433", "SCENARIO-LEARN-1434"],
        "source_artifacts": list(
            source_artifacts
            or [f"results/{EXP1395_FILE}", DVI_V3_ARTIFACT_USED, "data/fover_corpus.jsonl"]
        ),
        "dvi_v3_artifact_used": DVI_V3_ARTIFACT_USED,
        "dvi_v3_checkpoint_active": bool(activation.active),
        "dvi_v3_checkpoint_path": activation.path,
        "dvi_v3_checkpoint_blocker": activation.blocker,
        "fover_rows_available": len(fover_rows),
        "v6_candidate_cases_evaluated": len(candidates),
        "v6_new_promoted_count": new_promoted_count,
        "memory_updates": memory_updates,
        "fresh_verified_sample_count": fresh_verified_count,
        "baseline_fresh_verified_sample_count": baseline_count,
        "self_learning_delta_overall": delta,
        "nonforgetting_rate": round(float(nonforgetting_rate), 6)
        if nonforgetting_rate is not None
        else None,
        "session_memory_updated": new_promoted_count > 0,
        "headline_result_allowed": headline_allowed,
        "fresh_llm_inference_used": False,
        "cpu_only": True,
        "honest_verdict": derive_honest_verdict(
            status=status,
            blocker=activation.blocker,
            self_learning_delta=delta,
            nonforgetting_preserved=nonforgetting_preserved,
        ),
        "measurement_note": (
            "Exp 1433 uses the deployed Exp 1432 DVI v3 checkpoint as a verifier "
            "over FoVer rows not already promoted by Exp 1395. The reported fresh "
            "count is cumulative versus the 1508-case v5 baseline."
        ),
    }
    validate_artifact(artifact)
    return artifact


def derive_honest_verdict(
    *,
    status: str,
    blocker: str | None,
    self_learning_delta: int,
    nonforgetting_preserved: bool,
) -> str:
    """Name the v6 headline boundary without implying growth that was not found."""

    if status == "blocked":
        return f"fr11_self_learning_v6_blocked_{blocker or 'dvi_v3_unavailable'}"
    if self_learning_delta > 0 and nonforgetting_preserved:
        return "fr11_self_learning_v6_dvi_v3_headline_allowed_positive_growth"
    return "fr11_self_learning_v6_dvi_v3_no_positive_growth_non_headline"


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """REQ-LEARN-1433-5/6: enforce required fields and headline invariants."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")  # pragma: no cover
    if artifact["status"] == "in_progress":
        return

    fresh = int(artifact["fresh_verified_sample_count"])
    baseline = int(artifact["baseline_fresh_verified_sample_count"])
    delta = fresh - baseline
    if int(artifact["self_learning_delta_overall"]) != delta:
        raise AssertionError("self_learning_delta_overall must equal fresh minus baseline")
    nonforgetting = artifact["nonforgetting_rate"]
    if artifact["headline_result_allowed"]:
        if delta <= 0:
            raise AssertionError("headline_result_allowed requires positive self-learning delta")
        if nonforgetting is None or float(nonforgetting) < MIN_NONFORGETTING_RATE:
            raise AssertionError("headline_result_allowed requires preserved nonforgetting")
    if artifact["session_memory_updated"] and delta <= 0:
        raise AssertionError("session_memory_updated requires positive self-learning delta")


def run(
    *,
    exp1395_path: Path | str = DEFAULT_EXP1395_PATH,
    exp1432_path: Path | str = DEFAULT_EXP1432_PATH,
    fover_path: Path | str = DEFAULT_FOVER_PATH,
    out_path: Path | str = DEFAULT_OUTPUT_PATH,
    project_root: str | Path = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> dict[str, Any]:
    """Run Exp 1433 end-to-end and write the final artifact."""

    started_at = datetime.now(tz=UTC).isoformat()
    t0 = time.perf_counter()
    write_in_progress_artifact(out_path, project_root=project_root, run_date=run_date)
    exp1395 = load_json(exp1395_path)
    exp1432 = load_json(exp1432_path)
    fover_rows = dvi.load_jsonl_rows(fover_path)
    artifact = build_artifact(
        exp1395_artifact=exp1395,
        exp1432_artifact=exp1432,
        fover_rows=fover_rows,
        project_root=project_root,
        run_date=run_date,
        started_at=started_at,
        duration_s=time.perf_counter() - t0,
        source_artifacts=[
            f"results/{EXP1395_FILE}",
            DVI_V3_ARTIFACT_USED,
            "data/fover_corpus.jsonl",
        ],
    )
    return _write_json(out_path, artifact)


def _scalar_field(data: Any, field: str, default: float) -> float:
    if field not in data.files:
        return float(default)  # pragma: no cover
    return float(np.asarray(data[field], dtype=np.float32).reshape(-1)[0])


if __name__ == "__main__":  # pragma: no cover
    print(json.dumps(run(), indent=2, sort_keys=True))
