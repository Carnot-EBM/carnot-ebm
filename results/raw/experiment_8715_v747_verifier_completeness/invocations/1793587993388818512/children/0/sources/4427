"""Build the Exp 3324 Phase-3 path-de-risking capstone artifact.

Spec refs: REQ-REPORT-3324, SCENARIO-REPORT-3324.

This module aggregates the Exp 3322 Kona-premise test and Exp 3323 verifier
diversity audit. It does not run the premise test or the verifier audit; it
only records their terminal readout, runs the stable publication gate, and
chooses the next Phase-3 gap from those inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Mapping


JsonDict = dict[str, Any]
REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260529"
SCHEMA_VERSION = "carnot.milestone_capstone.v307_phase3_path_derisking.v1"
EXPERIMENT_ID = "exp3324"
TASK_ID = "exp3324-capstone-v307"
ARTIFACT = "experiment_3324_capstone_v307"
MILESTONE = "2026.05.307"
INFERENCE_SUBSTRATE = "aggregation_from_upstream_artifacts"
RANDOM_SEED = 3324

OUTPUT_REL_PATH = Path("results/experiment_3324_capstone_v307.json")
KONA_PREMISE_REL_PATH = Path(
    "results/experiment_3322_energy_descent_vs_autoregressive_premise_v1.json"
)
GROUNDING_AUDIT_REL_PATH = Path(
    "results/experiment_3323_verifier_ensemble_lambda_min_diversity_audit_v1.json"
)
PUBLICATION_GATE_REL_PATH = Path("scripts/publication_gate.py")

TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:", "shipped_")
KONA_OUTCOMES = {"validated", "viable_not_superior", "unsupported"}
GROUNDING_OUTCOMES = {"holds", "at_risk"}
DEFAULT_UNMET_GATES = ["G1", "G2", "G3", "G4"]
REQUIRED_ARTIFACT_FIELDS = {
    "capstone_v307_ready",
    "kona_premise_outcome",
    "grounding_keystone_outcome",
    "next_top_gap",
    "paper_ready",
    "publication_gate_unmet",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def read_json_object(path: Path) -> JsonDict:
    """Read JSON evidence while treating absent or malformed sources as empty."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return dict(payload) if isinstance(payload, Mapping) else {}


def sha256_file(path: Path) -> str | None:
    """Hash exact source bytes so the capstone can prove what it summarized."""

    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_publication_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Run `scripts/publication_gate.py --json` and return its JSON payload."""

    root_path = Path(root)
    try:
        completed = subprocess.run(
            [sys.executable, str(root_path / PUBLICATION_GATE_REL_PATH), "--json"],
            cwd=root_path,
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:  # pragma: no cover - defensive OS boundary
        return _publication_gate_failure(str(exc))
    if completed.returncode != 0:
        return _publication_gate_failure(completed.stderr.strip() or "publication gate failed")
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:  # pragma: no cover - defensive script boundary
        return _publication_gate_failure("publication gate emitted invalid JSON")
    return (
        dict(payload)
        if isinstance(payload, Mapping)
        else _publication_gate_failure("publication gate emitted non-object JSON")
    )


def build_artifact(
    root: Path | str = REPO_ROOT,
    *,
    publication_gate_result: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> JsonDict:
    """REQ-REPORT-3324: aggregate Exp 3322, Exp 3323, and the G1-G4 gate."""

    root_path = Path(root)
    start = time.perf_counter() if started_s is None else float(started_s)
    kona_artifact = read_json_object(root_path / KONA_PREMISE_REL_PATH)
    grounding_artifact = read_json_object(root_path / GROUNDING_AUDIT_REL_PATH)
    publication_gate_raw = (
        dict(publication_gate_result)
        if publication_gate_result is not None
        else run_publication_gate(root_path)
    )
    publication_gate = _normalise_publication_gate(publication_gate_raw)
    kona_outcome = _kona_premise_outcome(kona_artifact)
    grounding_outcome = _grounding_keystone_outcome(grounding_artifact)
    capstone_ready = not _is_blocked(kona_outcome) and not _is_blocked(grounding_outcome)

    artifact: JsonDict = {
        "schema": SCHEMA_VERSION,
        "schema_version": SCHEMA_VERSION,
        "artifact": ARTIFACT,
        "experiment_id": EXPERIMENT_ID,
        "task_id": TASK_ID,
        "run_date": RUN_DATE,
        "milestone": MILESTONE,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "capstone_v307_ready": capstone_ready,
        "kona_premise_outcome": kona_outcome,
        "grounding_keystone_outcome": grounding_outcome,
        "next_top_gap": _next_top_gap(kona_outcome, grounding_outcome),
        "paper_ready": publication_gate["paper_ready"],
        "publication_gate_unmet": publication_gate["publication_gate_unmet"],
        "publication_gate_source": "scripts/publication_gate.py --json",
        "publication_gate": publication_gate_raw,
        "kona_premise_summary": _kona_summary(kona_artifact),
        "grounding_keystone_summary": _grounding_summary(grounding_artifact),
        "source_artifacts": _source_artifacts(root_path, kona_artifact, grounding_artifact),
        "source_checksums": _source_checksums(root_path),
        "no_new_model_execution": True,
        "no_new_cuda_probe": True,
        "no_new_teacher_labeling": True,
        "no_new_kan_training": True,
        "no_new_garak_run": True,
        "no_new_dataflip_run": True,
        "no_new_repair_run": True,
        "no_new_verifier_run": True,
        "no_new_fr11_weight_update": True,
        "no_new_hardware_run": True,
        "no_conductor_execution": True,
        "no_external_submission_or_publication": True,
        "no_push": True,
        "ops_status_modified_by_this_task": False,
        "ops_changelog_modified_by_this_task": False,
        "traceability_modified_by_this_task": False,
        "random_seed": RANDOM_SEED,
        "duration_s": _duration(start, now_s),
        "reproducibility_checksum": "",
        "honest_verdict": "",
    }
    artifact["reproducibility_checksum"] = _reproducibility_checksum(artifact)
    artifact["honest_verdict"] = _honest_verdict(artifact)
    validate_artifact(artifact)
    return artifact


def write_artifact(
    root: Path | str = REPO_ROOT,
    *,
    output_path: Path | str = OUTPUT_REL_PATH,
    publication_gate_result: Mapping[str, Any] | None = None,
    started_s: float | None = None,
    now_s: float | None = None,
) -> Path:
    """Build and persist the Exp 3324 capstone JSON."""

    root_path = Path(root)
    out_path = Path(output_path)
    if not out_path.is_absolute():
        out_path = root_path / out_path
    artifact = build_artifact(
        root_path,
        publication_gate_result=publication_gate_result,
        started_s=started_s,
        now_s=now_s,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return out_path


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Reject capstone JSON that omits schema fields or overclaims execution."""

    missing = sorted(REQUIRED_ARTIFACT_FIELDS - set(artifact))
    if missing:
        raise ValueError(f"missing required fields: {missing}")
    if artifact.get("experiment_id") != EXPERIMENT_ID:
        raise ValueError("experiment_id must be exp3324")
    if artifact.get("task_id") != TASK_ID:
        raise ValueError("task_id must be exp3324-capstone-v307")
    if artifact.get("milestone") != MILESTONE:
        raise ValueError("milestone must be 2026.05.307")
    if artifact.get("random_seed") != RANDOM_SEED:
        raise ValueError("random_seed must be 3324")
    if artifact.get("inference_substrate") != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate must be aggregation_from_upstream_artifacts")
    kona = str(artifact.get("kona_premise_outcome") or "")
    grounding = str(artifact.get("grounding_keystone_outcome") or "")
    if kona not in KONA_OUTCOMES and not _is_blocked(kona):
        raise ValueError("kona_premise_outcome is not valid")
    if grounding not in GROUNDING_OUTCOMES and not _is_blocked(grounding):
        raise ValueError("grounding_keystone_outcome is not valid")
    if not isinstance(artifact.get("capstone_v307_ready"), bool):
        raise ValueError("capstone_v307_ready must be a boolean")
    if not isinstance(artifact.get("paper_ready"), bool):
        raise ValueError("paper_ready must be a boolean")
    if not isinstance(artifact.get("publication_gate_unmet"), list):
        raise ValueError("publication_gate_unmet must be a list")
    if not str(artifact.get("next_top_gap") or ""):
        raise ValueError("next_top_gap must be non-empty")
    if not str(artifact.get("reproducibility_checksum") or ""):
        raise ValueError("reproducibility_checksum must be non-empty")
    if not _terminal_prefix_ok(str(artifact.get("honest_verdict") or "")):
        raise ValueError("honest_verdict must begin with a terminal success prefix")
    if artifact.get("no_push") is not True:
        raise ValueError("no_push must remain true")


def _source_artifacts(
    root: Path,
    kona_artifact: Mapping[str, Any],
    grounding_artifact: Mapping[str, Any],
) -> list[JsonDict]:
    return [
        _source_record(root, KONA_PREMISE_REL_PATH, "exp3322_kona_premise", kona_artifact),
        _source_record(
            root, GROUNDING_AUDIT_REL_PATH, "exp3323_grounding_keystone", grounding_artifact
        ),
    ]


def _source_record(root: Path, path: Path, role: str, payload: Mapping[str, Any]) -> JsonDict:
    full_path = root / path
    return {
        "role": role,
        "path": path.as_posix(),
        "present": full_path.is_file(),
        "readable_json_object": bool(payload),
        "reported_experiment_id": str(payload.get("experiment_id") or ""),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "sha256": sha256_file(full_path),
    }


def _source_checksums(root: Path) -> JsonDict:
    checksums = {}
    for path in (KONA_PREMISE_REL_PATH, GROUNDING_AUDIT_REL_PATH):
        digest = sha256_file(root / path)
        if digest:
            checksums[path.as_posix()] = digest
    return checksums


def _kona_premise_outcome(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "blocked_exp3322_missing"
    verdict = str(payload.get("honest_verdict") or "").lower()
    blocked = _blocked_token(verdict)
    if blocked:
        return blocked
    if "beats_ar" in verdict or "premise_validated" in verdict:
        return "validated"
    if "viable_not_superior" in verdict:
        return "viable_not_superior"
    if "below_ar" in verdict or "premise_unsupported" in verdict:
        return "unsupported"
    if payload.get("energy_descent_vs_autoregressive_premise_v1_ready") is False:
        return "blocked_exp3322_not_ready"
    energy_accuracy = _float_value(payload.get("energy_descent_accuracy"))
    ar_accuracy = _float_value(payload.get("ar_baseline_accuracy"))
    if energy_accuracy == 0.0 and ar_accuracy == 0.0:
        return "blocked_exp3322_indeterminate"
    delta = _float_value(payload.get("accuracy_delta"))
    p_value = _float_value(_as_mapping(payload.get("paired_significance")).get("p_value"))
    if energy_accuracy >= ar_accuracy:
        return "validated" if delta > 0.0 and 0.0 < p_value < 0.05 else "viable_not_superior"
    return "unsupported"


def _grounding_keystone_outcome(payload: Mapping[str, Any]) -> str:
    if not payload:
        return "blocked_exp3323_missing"
    verdict = str(payload.get("honest_verdict") or "").lower()
    blocked = _blocked_token(verdict)
    if blocked:
        return blocked
    if "grounding_holds" in verdict or "diversity_sufficient" in verdict:
        return "holds"
    if "grounding_at_risk" in verdict or "null_space_collapse" in verdict:
        return "at_risk"
    if payload.get("verifier_ensemble_lambda_min_diversity_audit_v1_ready") is False:
        return "blocked_exp3323_not_ready"
    lambda_min = _float_value(payload.get("lambda_min_sigma"))
    effective_k = _float_value(payload.get("effective_k_participation_ratio"))
    if lambda_min == 0.0 and effective_k == 0.0:
        return "blocked_exp3323_indeterminate"
    return "holds" if lambda_min > 0.1 and effective_k >= 3.0 else "at_risk"


def _next_top_gap(kona_outcome: str, grounding_outcome: str) -> str:
    if _is_blocked(kona_outcome) or _is_blocked(grounding_outcome):
        return "complete_phase3_path_derisking_upstreams"
    if kona_outcome == "validated" and grounding_outcome == "holds":
        return "scale_substrate_intermediate (exp_NEXT_E 100-300M)"
    if kona_outcome == "unsupported":
        return "reconsider_foundation_model_endgame"
    if grounding_outcome == "at_risk":
        return "redesign_verifier_grounding_source"
    if kona_outcome == "viable_not_superior":
        return "prove_energy_descent_superiority_before_scale"
    return "complete_phase3_path_derisking_upstreams"


def _kona_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "present": bool(payload),
        "task_name": str(payload.get("task_name") or ""),
        "n_problems": int(payload.get("n_problems") or 0)
        if isinstance(payload.get("n_problems"), int)
        else 0,
        "ar_baseline_accuracy": _float_value(payload.get("ar_baseline_accuracy")),
        "energy_descent_accuracy": _float_value(payload.get("energy_descent_accuracy")),
        "accuracy_delta": _float_value(payload.get("accuracy_delta")),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "reproducibility_checksum": str(payload.get("reproducibility_checksum") or ""),
    }


def _grounding_summary(payload: Mapping[str, Any]) -> JsonDict:
    return {
        "present": bool(payload),
        "k_verifiers": int(payload.get("k_verifiers") or 0)
        if isinstance(payload.get("k_verifiers"), int)
        else 0,
        "lambda_min_sigma": _float_value(payload.get("lambda_min_sigma")),
        "effective_k_participation_ratio": _float_value(
            payload.get("effective_k_participation_ratio")
        ),
        "pairwise_max_correlation": _float_value(payload.get("pairwise_max_correlation")),
        "honest_verdict": str(payload.get("honest_verdict") or ""),
        "reproducibility_checksum": str(payload.get("reproducibility_checksum") or ""),
    }


def _normalise_publication_gate(payload: Mapping[str, Any]) -> JsonDict:
    paper_ready = payload.get("paper_ready") is True
    raw_unmet = payload.get(
        "unmet_gates", payload.get("publication_gate_unmet", payload.get("unmet"))
    )
    if isinstance(raw_unmet, list):
        unmet = _list_of_strings(raw_unmet)
    else:
        unmet = [] if paper_ready else list(DEFAULT_UNMET_GATES)
    return {
        "paper_ready": paper_ready,
        "publication_gate_unmet": unmet,
    }


def _publication_gate_failure(reason: str) -> JsonDict:
    return {
        "paper_ready": False,
        "unmet_gates": list(DEFAULT_UNMET_GATES),
        "error": reason,
    }


def _reproducibility_checksum(artifact: Mapping[str, Any]) -> str:
    stable = {
        "experiment_id": artifact.get("experiment_id"),
        "task_id": artifact.get("task_id"),
        "capstone_v307_ready": artifact.get("capstone_v307_ready"),
        "kona_premise_outcome": artifact.get("kona_premise_outcome"),
        "grounding_keystone_outcome": artifact.get("grounding_keystone_outcome"),
        "next_top_gap": artifact.get("next_top_gap"),
        "paper_ready": artifact.get("paper_ready"),
        "publication_gate_unmet": artifact.get("publication_gate_unmet"),
        "kona_premise_summary": artifact.get("kona_premise_summary"),
        "grounding_keystone_summary": artifact.get("grounding_keystone_summary"),
        "source_checksums": artifact.get("source_checksums"),
    }
    payload = json.dumps(stable, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _honest_verdict(artifact: Mapping[str, Any]) -> str:
    return (
        "complete: capstone_v307_ready="
        f"{str(artifact.get('capstone_v307_ready') is True).lower()}; "
        f"kona={artifact.get('kona_premise_outcome')}; "
        f"grounding={artifact.get('grounding_keystone_outcome')}; "
        f"paper_ready={str(artifact.get('paper_ready') is True).lower()}; "
        f"publication_gate_unmet={artifact.get('publication_gate_unmet')}; "
        f"next_top_gap={artifact.get('next_top_gap')}"
    )


def _duration(started_s: float, now_s: float | None) -> float:
    end = time.perf_counter() if now_s is None else float(now_s)
    return round(max(0.0, end - started_s), 6)


def _blocked_token(verdict: str) -> str:
    for token in verdict.replace(":", " ").replace(";", " ").split():
        if token.startswith("blocked_"):
            return token
    return ""


def _is_blocked(outcome: str) -> bool:
    return outcome.startswith("blocked_")


def _terminal_prefix_ok(verdict: str) -> bool:
    return verdict.startswith(TERMINAL_PREFIXES)


def _as_mapping(value: Any) -> JsonDict:
    return dict(value) if isinstance(value, Mapping) else {}


def _list_of_strings(value: Any) -> list[str]:
    return [str(item) for item in value] if isinstance(value, list) else []


def _float_value(value: Any) -> float:
    return float(value) if isinstance(value, int | float) and not isinstance(value, bool) else 0.0
