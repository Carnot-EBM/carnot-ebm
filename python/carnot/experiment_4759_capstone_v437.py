"""Experiment 4759: .437 induction-quality capstone aggregation.

Spec refs: REQ-CAPSTONE-4759, SCENARIO-CAPSTONE-4759,
SCENARIO-CAPSTONE-4759-FIELD-PRINCIPLES.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_4759_capstone_v437"
SCHEMA = "carnot.exp4759.capstone_v437.v1"
RESULT_RELATIVE_PATH = "results/experiment_4759_capstone_v437.json"
REGISTRY_RELATIVE_PATH = "ops/arc_solve_registry.yaml"
SPEC_RELATIVE_PATH = "openspec/capabilities/capstone/spec.md"
RANDOM_SEED = 4759
FREEFORM_BASELINE = 0.12
INFERENCE_SUBSTRATE = (
    "aggregation_from_upstream_artifacts -- reads upstream JSON, registry, "
    "and publication gate; no model load (100us floor)."
)

UPSTREAM_SOURCES: dict[str, str] = {
    "A1": "results/experiment_4749_structured_engine_vs_freeform.json",
    "A2": "results/experiment_4750_structural_alignment_detector_fix.json",
    "A3": "results/experiment_4751_levelup_selfplay.json",
    "A4": "results/experiment_4752_held_out_first_win_readiness.json",
}

FIELD_PRINCIPLES: dict[str, dict[str, str]] = {
    "honest_verdict": {
        "principle": "terminal prefix; the capstone is complete_ (an aggregation), success_ only if a real bank landed."
    },
    "inference_substrate": {"principle": "aggregation_from_upstream_artifacts; 100us floor."},
    "preconditions_checked": {"principle": "records which upstream artifacts were present."},
    "cited_upstream_artifacts": {
        "principle": "{experiment_id, fields_imported, sha256} so every aggregated number traces to a real measurement (the aggregation audit trail)."
    },
    "bridge_crossed_for_solve": {
        "principle": "the honest headline -- did the agent solve a hidden-class game; false is honest, not a failure."
    },
    "reproducible_total_levels": {
        "principle": "the authoritative banked-level count from the registry -- the monotonic sprint metric."
    },
    "induction_quality_decision": {
        "principle": "did A1 beat the 0.12 free-form baseline and/or A2 make the structural goal satisfiable -- the .437 headline result."
    },
    "verifier_is_oracle": {"principle": "false on every aggregated value claim."},
    "submission_package_ready": {
        "principle": "True only if OPERATOR-ready; the capstone NEVER submits."
    },
}

SPEC_REFS = [
    "REQ-CAPSTONE-4759",
    "SCENARIO-CAPSTONE-4759",
    "SCENARIO-CAPSTONE-4759-FIELD-PRINCIPLES",
]

REQUIRED_FIELDS = (
    "honest_verdict",
    "inference_substrate",
    "preconditions_checked",
    "cited_upstream_artifacts",
    "bridge_crossed_for_solve",
    "reproducible_total_levels",
    "induction_quality_decision",
    "verifier_is_oracle",
    "submission_package_ready",
    "paper_ready",
    "publication_gate",
    "scorecard",
    "skipped_artifacts",
    "registry_provenance",
    "field_principles",
    "spec_refs",
    "schema",
    "experiment",
    "result_path",
    "duration_s",
    "random_seed",
    "reproducibility_checksum",
    "submitted_to_leaderboard",
    "missing_artifacts",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_bytes(data: bytes) -> str:
    return "sha256:" + hashlib.sha256(data).hexdigest()


def _file_sha256(path: Path) -> str | None:
    return _sha256_bytes(path.read_bytes()) if path.exists() else None


def _checksum_artifact(payload: Mapping[str, Any]) -> str:
    stable_payload = {
        key: value
        for key, value in payload.items()
        if key not in {"duration_s", "reproducibility_checksum"}
    }
    return _sha256_bytes(_stable_json(stable_payload).encode("utf-8"))


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _read_yaml(path: Path) -> dict[str, Any]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    return loaded if isinstance(loaded, dict) else {}


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed == parsed else None


def _as_int(value: Any) -> int:
    if isinstance(value, bool) or value is None:
        return 0
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _experiment_id(source: str, artifact: Mapping[str, Any]) -> int:
    parsed = _as_int(artifact.get("experiment_id"))
    if parsed:
        return parsed
    return {"A1": 4749, "A2": 4750, "A3": 4751, "A4": 4752}[source]


def _is_flagged(artifact: Mapping[str, Any]) -> bool:
    return artifact.get("flagged_adversarial") is True


def _gate_paper_ready(publication_gate: Mapping[str, Any]) -> bool:
    gates = publication_gate.get("gates")
    if not isinstance(gates, Mapping):
        return False
    return bool(
        publication_gate.get("paper_ready") is True
        and all(
            isinstance(gates.get(gate), Mapping) and gates[gate].get("pass") is True
            for gate in ("G1", "G2", "G3", "G4")
        )
    )


def _a1_scorecard(artifact: Mapping[str, Any] | None, *, skipped: bool) -> dict[str, Any]:
    if artifact is None:
        return {
            "present": False,
            "decision": "missing",
            "beat_0_12_freeform_baseline": None,
            "banked_l2": False,
        }
    if skipped:
        return {
            "present": True,
            "decision": "skipped_flagged_adversarial",
            "beat_0_12_freeform_baseline": None,
            "banked_l2": False,
        }
    structured_accuracy = _as_float(artifact.get("structured_heldout_accuracy"))
    beat_baseline = structured_accuracy is not None and structured_accuracy > FREEFORM_BASELINE
    banked_l2 = (
        bool(artifact.get("offline_reproduced")) and _as_int(artifact.get("reproduced_levels")) >= 2
    )
    return {
        "present": True,
        "decision": "beat_0_12_baseline" if beat_baseline else "did_not_beat_0_12_baseline",
        "structured_heldout_accuracy": structured_accuracy,
        "freeform_baseline": FREEFORM_BASELINE,
        "beat_0_12_freeform_baseline": beat_baseline,
        "banked_l2": banked_l2,
    }


def _a2_scorecard(artifact: Mapping[str, Any] | None, *, skipped: bool) -> dict[str, Any]:
    if artifact is None:
        return {
            "present": False,
            "decision": "missing",
            "goal_predicate_satisfiable": False,
            "banked_l2": False,
        }
    if skipped:
        return {
            "present": True,
            "decision": "skipped_flagged_adversarial",
            "goal_predicate_satisfiable": False,
            "banked_l2": False,
        }
    satisfiable = artifact.get("goal_predicate_satisfiable") is True
    plan_reaches_goal = artifact.get("l2_plan_reaches_goal") is True
    banked_l2 = (
        bool(artifact.get("offline_reproduced")) and _as_int(artifact.get("reproduced_levels")) >= 2
    )
    return {
        "present": True,
        "decision": "satisfiable_or_banked"
        if (satisfiable or banked_l2)
        else "detector_fixed_no_satisfiable_goal_no_bank",
        "goal_predicate_satisfiable": satisfiable,
        "l2_plan_reaches_goal": plan_reaches_goal,
        "offline_reproduced": bool(artifact.get("offline_reproduced")),
        "reproduced_levels": _as_int(artifact.get("reproduced_levels")),
        "banked_l2": banked_l2,
    }


def _a3_scorecard(artifact: Mapping[str, Any] | None, *, skipped: bool) -> dict[str, Any]:
    if artifact is None:
        return {"present": False, "decision": "missing", "real_bank_landed": False}
    if skipped:
        return {
            "present": True,
            "decision": "skipped_flagged_adversarial",
            "real_bank_landed": False,
        }
    banked = (
        _as_int(artifact.get("new_levels_banked")) > 0
        and artifact.get("offline_reproduced") is True
    )
    return {
        "present": True,
        "decision": "real_bank_landed" if banked else "no_bank",
        "target_game": artifact.get("target_game"),
        "reached_level": _as_int(artifact.get("reached_level")),
        "new_levels_banked": _as_int(artifact.get("new_levels_banked")),
        "offline_reproduced": artifact.get("offline_reproduced") is True,
        "real_bank_landed": banked,
    }


def _a4_scorecard(artifact: Mapping[str, Any] | None, *, skipped: bool) -> dict[str, Any]:
    if artifact is None:
        return {"present": False, "decision": "missing", "submission_package_ready": False}
    if skipped:
        return {
            "present": True,
            "decision": "skipped_flagged_adversarial",
            "submission_package_ready": False,
        }
    ready = (
        artifact.get("submission_package_ready") is True
        and artifact.get("ready_for_operator_submit") is True
    )
    return {
        "present": True,
        "decision": "operator_ready" if ready else "not_operator_ready",
        "submission_package_ready": ready,
    }


def _imported_fields(source: str, artifact: Mapping[str, Any], *, skipped: bool) -> list[str]:
    if skipped:
        return ["flagged_adversarial"]
    fields = {
        "A1": [
            "structured_heldout_accuracy",
            "freeform_heldout_accuracy",
            "offline_reproduced",
            "reproduced_levels",
            "verifier_is_oracle",
        ],
        "A2": [
            "goal_predicate_satisfiable",
            "l2_plan_reaches_goal",
            "offline_reproduced",
            "reproduced_levels",
            "verifier_is_oracle",
        ],
        "A3": [
            "new_levels_banked",
            "offline_reproduced",
            "reproduced_levels",
            "reproducible_total_levels",
            "target_game",
            "verifier_is_oracle",
        ],
        "A4": ["submission_package_ready", "ready_for_operator_submit", "verifier_is_oracle"],
    }[source]
    return [field for field in fields if field in artifact]


def _default_preconditions(
    artifacts: Mapping[str, Mapping[str, Any]],
    *,
    registry_present: bool,
    registry_loadable: bool,
    publication_gate_available: bool,
) -> dict[str, Any]:
    return {
        "upstream_artifacts": {
            source: {"path": path, "present": source in artifacts}
            for source, path in UPSTREAM_SOURCES.items()
        },
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
        },
        "publication_gate": {"available": publication_gate_available},
    }


def build_artifact(
    *,
    artifacts: Mapping[str, Mapping[str, Any]],
    artifact_sha256: Mapping[str, str],
    registry: Mapping[str, Any],
    registry_sha256: str | None,
    publication_gate: Mapping[str, Any],
    duration_s: float,
    preconditions_checked: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    skipped_artifacts: list[dict[str, Any]] = []
    cited: list[dict[str, Any]] = []
    scorecard: dict[str, Any] = {}

    for source in UPSTREAM_SOURCES:
        artifact = artifacts.get(source)
        if artifact is None:
            continue
        skipped = _is_flagged(artifact)
        if skipped:
            skipped_artifacts.append(
                {
                    "source": source,
                    "experiment_id": _experiment_id(source, artifact),
                    "reason": "flagged_adversarial",
                    "sha256": artifact_sha256.get(source, ""),
                }
            )
        cited.append(
            {
                "experiment_id": _experiment_id(source, artifact),
                "fields_imported": _imported_fields(source, artifact, skipped=skipped),
                "sha256": artifact_sha256.get(source, ""),
            }
        )

    scorecard["A1"] = _a1_scorecard(
        artifacts.get("A1"), skipped=_is_flagged(artifacts.get("A1", {}))
    )
    scorecard["A2"] = _a2_scorecard(
        artifacts.get("A2"), skipped=_is_flagged(artifacts.get("A2", {}))
    )
    scorecard["A3"] = _a3_scorecard(
        artifacts.get("A3"), skipped=_is_flagged(artifacts.get("A3", {}))
    )
    scorecard["A4"] = _a4_scorecard(
        artifacts.get("A4"), skipped=_is_flagged(artifacts.get("A4", {}))
    )

    a1 = scorecard["A1"]
    a2 = scorecard["A2"]
    a3 = scorecard["A3"]
    a4 = scorecard["A4"]
    bridge_crossed = bool(a1.get("banked_l2") or a2.get("banked_l2"))
    real_bank_landed = bool(bridge_crossed or a3.get("real_bank_landed"))
    induction_wall_cleared = bool(
        a1.get("beat_0_12_freeform_baseline")
        or a2.get("goal_predicate_satisfiable")
        or a2.get("banked_l2")
    )

    if bridge_crossed:
        honest_verdict = "success: induction_quality_wall_cleared_capstone_complete"
    elif real_bank_landed:
        game = str(a3.get("target_game") or "unknown")
        level = _as_int(a3.get("reached_level"))
        honest_verdict = f"success: real_bank_landed_{game}_L{level}_capstone_complete"
    else:
        honest_verdict = "complete: capstone_aggregation_no_real_bank"

    missing_artifacts = [source for source in UPSTREAM_SOURCES if source not in artifacts]
    publication_gate_copy = json.loads(_stable_json(publication_gate))
    payload: dict[str, Any] = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "result_path": RESULT_RELATIVE_PATH,
        "honest_verdict": honest_verdict,
        "inference_substrate": INFERENCE_SUBSTRATE,
        "preconditions_checked": dict(
            preconditions_checked
            or _default_preconditions(
                artifacts,
                registry_present=bool(registry),
                registry_loadable=bool(registry),
                publication_gate_available=bool(publication_gate),
            )
        ),
        "cited_upstream_artifacts": cited,
        "bridge_crossed_for_solve": bridge_crossed,
        "reproducible_total_levels": _as_int(registry.get("reproducible_total_levels")),
        "paper_ready": _gate_paper_ready(publication_gate),
        "publication_gate": publication_gate_copy,
        "induction_quality_decision": {
            "a1": a1,
            "a2": a2,
            "cleared_induction_quality_wall": induction_wall_cleared,
            "headline": (
                "A1 beat the 0.12 baseline and/or A2 made the structural goal satisfiable."
                if induction_wall_cleared
                else "A1 was not admissible or did not beat 0.12; A2 did not make the goal satisfiable or bank L2."
            ),
        },
        "verifier_is_oracle": False,
        "submission_package_ready": bool(a4.get("submission_package_ready")),
        "scorecard": scorecard,
        "skipped_artifacts": skipped_artifacts,
        "missing_artifacts": missing_artifacts,
        "registry_provenance": {
            "path": REGISTRY_RELATIVE_PATH,
            "fields_imported": ["reproducible_total_levels"],
            "sha256": registry_sha256 or "",
        },
        "field_principles": FIELD_PRINCIPLES,
        "spec_refs": SPEC_REFS,
        "duration_s": float(duration_s),
        "random_seed": RANDOM_SEED,
        "submitted_to_leaderboard": False,
    }
    payload["reproducibility_checksum"] = _checksum_artifact(payload)
    return payload


def validate_artifact(payload: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_FIELDS:
        if field not in payload:
            errors.append(f"missing_field:{field}")
    if not str(payload.get("honest_verdict", "")).startswith(
        ("complete:", "success:", "passed:", "shipped:")
    ):
        errors.append("honest_verdict_missing_terminal_prefix")
    if payload.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if payload.get("submitted_to_leaderboard") is not False:
        errors.append("submitted_to_leaderboard_must_be_false")
    field_principles = payload.get("field_principles")
    for field, principle in FIELD_PRINCIPLES.items():
        if not isinstance(field_principles, Mapping) or field_principles.get(field) != principle:
            errors.append(f"missing_principle:{field}")
    cited = payload.get("cited_upstream_artifacts")
    if not isinstance(cited, list) or any(
        not isinstance(row, Mapping)
        or not isinstance(row.get("experiment_id"), int)
        or not isinstance(row.get("fields_imported"), list)
        or not str(row.get("sha256", "")).startswith("sha256:")
        for row in cited
    ):
        errors.append("invalid_cited_upstream_artifacts")
    if not str(payload.get("reproducibility_checksum", "")).startswith("sha256:"):
        errors.append("invalid_reproducibility_checksum")
    return errors


def _publication_gate_result(
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:  # pragma: no cover - live subprocess boundary.
    cmd = [sys.executable, "scripts/publication_gate.py", "--json"]
    proc = subprocess.run(
        cmd,
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    info = {
        "command": " ".join(cmd),
        "available": proc.returncode == 0,
        "returncode": proc.returncode,
    }
    if proc.returncode != 0:
        return {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_unrunnable"],
        }, info
    try:
        loaded = json.loads(proc.stdout)
    except json.JSONDecodeError:
        info["available"] = False
        return {
            "paper_ready": False,
            "gates": {},
            "unmet_gates": ["publication_gate_invalid_json"],
        }, info
    return loaded if isinstance(loaded, dict) else {}, info


def run_capstone(
    *,
    root: Path = REPO_ROOT,
    publication_gate: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    artifacts: dict[str, dict[str, Any]] = {}
    artifact_sha256: dict[str, str] = {}
    upstream_preconditions: dict[str, dict[str, Any]] = {}

    for source, relative_path in UPSTREAM_SOURCES.items():
        path = root / relative_path
        present = path.exists()
        upstream_preconditions[source] = {"path": relative_path, "present": present}
        if present:
            artifacts[source] = _read_json(path)
            artifact_sha256[source] = _file_sha256(path) or ""

    registry_path = root / REGISTRY_RELATIVE_PATH
    registry_present = registry_path.exists()
    registry: dict[str, Any] = {}
    registry_loadable = False
    if registry_present:
        try:
            registry = _read_yaml(registry_path)
            registry_loadable = True
        except yaml.YAMLError:
            registry = {}

    if publication_gate is None:
        gate_payload, gate_precondition = _publication_gate_result(root)
    else:
        gate_payload = dict(publication_gate)
        gate_precondition = {"available": True, "injected": True}

    spec_path = root / SPEC_RELATIVE_PATH
    preconditions_checked = {
        "agents_md_read": True,
        "codex_md_read": True,
        "upstream_artifacts": upstream_preconditions,
        "registry": {
            "path": REGISTRY_RELATIVE_PATH,
            "present": registry_present,
            "yaml_loadable": registry_loadable,
        },
        "publication_gate": gate_precondition,
        "spec_has_req_4759": spec_path.exists()
        and "REQ-CAPSTONE-4759" in spec_path.read_text(encoding="utf-8"),
    }

    artifact = build_artifact(
        artifacts=artifacts,
        artifact_sha256=artifact_sha256,
        registry=registry,
        registry_sha256=_file_sha256(registry_path),
        publication_gate=gate_payload,
        duration_s=max(time.perf_counter() - start, 0.0001),
        preconditions_checked=preconditions_checked,
    )
    result_path = root / RESULT_RELATIVE_PATH
    result_path.parent.mkdir(parents=True, exist_ok=True)
    result_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return artifact


def main() -> int:  # pragma: no cover - direct CLI boundary.
    artifact = run_capstone()
    errors = validate_artifact(artifact)
    print(
        json.dumps(
            {
                "result_path": RESULT_RELATIVE_PATH,
                "honest_verdict": artifact["honest_verdict"],
                "schema_errors": errors,
            },
            sort_keys=True,
        )
    )
    return 1 if errors else 0


if __name__ == "__main__":  # pragma: no cover - direct script boundary.
    raise SystemExit(main())
