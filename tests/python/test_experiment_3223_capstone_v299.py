"""Tests for Exp 3223 milestone .299 capstone.

Spec refs: REQ-REPORT-3223, SCENARIO-REPORT-3223.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v299_3223 as mod


REQUIRED_FIELDS = {
    "capstone_v299_ready",
    "paper_ready",
    "publication_blocker_count",
    "next_top_gap",
    "v4_outcome",
    "random_seed",
    "reproducibility_checksum",
    "duration_s",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prior_capstone(
    *,
    ready: bool = True,
    paper_ready: bool = False,
    blockers: int = 100,
    next_gap: str = "repair_system_driver_cuda_runtime_boundary_to_unblock_cuda_offload_receipt",
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.milestone_capstone.v298_matrix_v32_terminal_aggregation.v1",
        "experiment_id": "exp3232",
        "milestone": "2026.05.298",
        "capstone_ready": ready,
        "paper_ready": paper_ready,
        "publication_blocker_count": blockers,
        "next_top_gap": next_gap,
        "honest_verdict": "complete: capstone_ready=true",
    }


def _v4_payload(
    verdict: str,
    *,
    gates: dict[str, bool] | None = None,
    model_specs: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": "carnot.prompt_injection_kan_distill.v4_15k",
        "experiment_id": "exp3222",
        "milestone": "2026.05.299",
        "honest_verdict": verdict,
        "random_seed": 3222,
        "reproducibility_checksum": "a" * 64,
        "model_specs": model_specs or {"teacher": "gpt-oss-safeguard-20b-Q4_K_M.gguf"},
        "gate_results": gates
        or {
            "gate_1_replacement_grade": True,
            "gate_2_ood_floor": True,
            "gate_3_adversarial_floor": False,
        },
        "auroc_paired_test": 0.912,
        "delong_pvalue_vs_teacher": 0.01,
        "cross_dataset_auroc": 0.87,
        "garak_auroc_per_probe": {"worst_case": 0.71, "median": 0.81},
    }


def _write_sources(
    root: Path,
    *,
    v4: dict[str, Any] | None = None,
    prior: dict[str, Any] | None = None,
) -> None:
    _write_json(root, mod.PRIOR_CAPSTONE_REL_PATH, prior or _prior_capstone())
    if v4 is not None:
        _write_json(root, mod.V4_RESULT_REL_PATH, v4)


def test_req_report_3223_spec_anchor_and_script_exist() -> None:
    """REQ-REPORT-3223: OpenSpec declares the v299 capstone before code."""

    spec = (mod.REPO_ROOT / "openspec/capabilities/research-reporting/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-REPORT-3223" in spec
    assert "SCENARIO-REPORT-3223" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert mod.SCRIPT_REL_PATH.exists()


def test_scenario_report_3223_garak_partial_sets_v4_gap(tmp_path: Path) -> None:
    """SCENARIO-REPORT-3223: Gates 1+2 with partial Gate 3 selects Garak expansion."""

    _write_sources(
        tmp_path,
        v4=_v4_payload("complete: prompt_injection_v4_publication_grade_garak_partial"),
    )

    artifact = mod.build_artifact(tmp_path, started_s=1.0, now_s=3.5)
    second = mod.build_artifact(tmp_path, started_s=10.0, now_s=11.0)
    sources = {row["role"]: row for row in artifact["source_artifacts"]}

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["schema_version"] == mod.SCHEMA_VERSION
    assert artifact["experiment_id"] == "exp3223"
    assert artifact["task_id"] == mod.TASK_ID
    assert artifact["milestone"] == "2026.05.299"
    assert artifact["prior_capstone_artifact"] == mod.PRIOR_CAPSTONE_REL_PATH.as_posix()
    assert artifact["v4_result_artifact"] == mod.V4_RESULT_REL_PATH.as_posix()
    assert artifact["capstone_v299_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["prior_publication_blocker_count"] == 100
    assert artifact["publication_blocker_delta"] == -1
    assert artifact["publication_blocker_count"] == 99
    assert artifact["v4_outcome"] == "publication_grade_garak_partial"
    assert artifact["next_top_gap"] == mod.GARAK_NEXT_TOP_GAP
    assert artifact["gate_summary"] == {
        "gate_1_replacement_grade": True,
        "gate_2_ood_floor": True,
        "gate_3_adversarial_floor": False,
    }
    assert artifact["source_v4_summary"]["honest_verdict"].endswith(
        "prompt_injection_v4_publication_grade_garak_partial"
    )
    assert artifact["random_seed"] == mod.RANDOM_SEED
    assert len(artifact["reproducibility_checksum"]) == 64
    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["honest_verdict"].startswith("complete:")
    assert sources["prior_capstone_v298"]["sha256"] == _sha256(
        tmp_path / mod.PRIOR_CAPSTONE_REL_PATH
    )
    assert sources["prompt_injection_kan_v4"]["sha256"] == _sha256(
        tmp_path / mod.V4_RESULT_REL_PATH
    )
    assert artifact["no_new_model_execution"] is True
    assert artifact["no_new_garak_run"] is True
    assert artifact["ops_docs_reconciliation_left_to_conductor"] is True


def test_req_report_3223_replacement_can_close_safety_tier_only(tmp_path: Path) -> None:
    """REQ-REPORT-3223: replacement-grade reduces blockers by three, not more."""

    _write_sources(
        tmp_path,
        prior=_prior_capstone(paper_ready=True, blockers=3),
        v4=_v4_payload(
            "complete: prompt_injection_v4_replacement_grade",
            gates={
                "gate_1_replacement_grade": True,
                "gate_2_ood_floor": True,
                "gate_3_adversarial_floor": True,
            },
        ),
    )

    artifact = mod.build_artifact(tmp_path)

    assert artifact["v4_outcome"] == "replacement_grade"
    assert artifact["publication_blocker_delta"] == -3
    assert artifact["publication_blocker_count"] == 0
    assert artifact["next_top_gap"] == mod.DEFAULT_NEXT_TOP_GAP
    assert artifact["paper_ready"] is True


@pytest.mark.parametrize(
    ("verdict", "expected"),
    [
        ("complete: prompt_injection_v4_overfit_to_training_distribution", "overfit_to_training"),
        ("complete: prompt_injection_v4_below_replacement_threshold", "below_replacement_threshold"),
    ],
)
def test_req_report_3223_non_replacement_outcomes_leave_blockers(
    tmp_path: Path, verdict: str, expected: str
) -> None:
    """REQ-REPORT-3223: weak v4 outcomes do not retire publication blockers."""

    _write_sources(tmp_path, v4=_v4_payload(verdict))

    artifact = mod.build_artifact(tmp_path)

    assert artifact["v4_outcome"] == expected
    assert artifact["publication_blocker_delta"] == 0
    assert artifact["publication_blocker_count"] == 100
    assert artifact["next_top_gap"] == mod.DEFAULT_NEXT_TOP_GAP
    assert artifact["paper_ready"] is False


def test_req_report_3223_missing_v4_is_blocked_resource_outcome(tmp_path: Path) -> None:
    """REQ-REPORT-3223: missing exp3222 is reported honestly without fabrication."""

    _write_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=4.0, now_s=4.0)

    assert artifact["capstone_v299_ready"] is True
    assert artifact["paper_ready"] is False
    assert artifact["v4_outcome"] == "blocked_missing_exp3222_result"
    assert artifact["publication_blocker_delta"] == 0
    assert artifact["publication_blocker_count"] == 100
    assert artifact["next_top_gap"] == mod.DEFAULT_NEXT_TOP_GAP
    assert artifact["source_v4_summary"]["present"] is False
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_report_3223_write_artifact_and_fail_closed_edges(tmp_path: Path) -> None:
    """REQ-REPORT-3223: writer and helpers preserve bounded aggregation semantics."""

    _write_sources(
        tmp_path,
        v4=_v4_payload("complete: prompt_injection_v4_replacement_grade"),
    )

    output = mod.write_artifact(tmp_path, started_s=2.0, now_s=5.0)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["capstone_v299_ready"] is True
    assert saved["duration_s"] == pytest.approx(3.0)

    empty = mod.build_artifact(tmp_path / "empty", started_s=3.0, now_s=2.0)
    assert empty["capstone_v299_ready"] is False
    assert empty["v4_outcome"] == "blocked_missing_exp3222_result"
    assert empty["honest_verdict"].startswith("complete:")
    assert "prior capstone v298 authority is missing or not ready" in empty["invariant_violations"]

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    list_json = tmp_path / "list.json"
    list_json.write_text("[]", encoding="utf-8")

    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(list_json) == {}
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    assert mod.sha256_file(tmp_path / "missing.json") is None
    assert mod._as_mapping([]) == {}
    assert mod._int_value(True) == 0
    assert mod._int_value(7) == 7
    assert mod._nonnegative_count(2, -5) == 0
    assert mod._outcome_from_verdict("blocked_cuda_unavailable") == "blocked_cuda_unavailable"
    assert mod._outcome_from_verdict("complete: unknown") == "blocked_unclassified_v4_outcome"
    assert mod._blocker_delta("replacement_grade") == -3
    assert mod._blocker_delta("publication_grade_garak_partial") == -1
    assert mod._blocker_delta("overfit_to_training") == 0
    assert mod._blocker_delta("blocked_cuda_unavailable") == 0
    assert mod._next_top_gap(
        "publication_grade_garak_partial",
        {
            "gate_1_replacement_grade": True,
            "gate_2_ood_floor": True,
            "gate_3_adversarial_floor": False,
        },
    ) == mod.GARAK_NEXT_TOP_GAP
    assert mod._next_top_gap("publication_grade_garak_partial", {}) == mod.DEFAULT_NEXT_TOP_GAP
    assert mod._gate_summary(
        {
            "gate_results": {
                "replacement_grade": True,
                "cross_dataset": True,
                "garak": False,
            }
        }
    ) == {
        "gate_1_replacement_grade": True,
        "gate_2_ood_floor": True,
        "gate_3_adversarial_floor": False,
    }
    assert mod._gate_summary(
        {
            "auroc_paired_test": 0.9,
            "delong_pvalue_vs_teacher": 0.04,
            "cross_dataset_auroc": 0.85,
            "garak_auroc_per_probe": {"worst_case": 0.75},
        }
    ) == {
        "gate_1_replacement_grade": True,
        "gate_2_ood_floor": True,
        "gate_3_adversarial_floor": True,
    }
    assert mod._required_fields_are_typed({}) == [
        "capstone_v299_ready missing_or_wrong_type",
        "paper_ready missing_or_wrong_type",
        "publication_blocker_count missing_or_wrong_type",
        "next_top_gap missing_or_wrong_type",
        "v4_outcome missing_or_wrong_type",
        "random_seed missing_or_wrong_type",
        "reproducibility_checksum missing_or_wrong_type",
        "duration_s missing_or_wrong_type",
        "honest_verdict missing_or_wrong_type",
    ]
