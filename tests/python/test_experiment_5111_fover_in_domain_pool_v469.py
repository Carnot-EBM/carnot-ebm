"""Tests for Exp 5111 FoVer in-domain pool retraction handling.

Spec refs: REQ-REPORT-5111, SCENARIO-REPORT-5111,
SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5111_fover_in_domain_pool_v469 as mod
from scripts import experiment_5111_fover_in_domain_pool_v469 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
CORRECTED_RESULT_PATH = REPO / mod.CORRECTED_RESULT_RELATIVE_PATH
KNOWN_ISSUES_PATH = REPO / mod.KNOWN_ISSUES_RELATIVE_PATH


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5111")
    end = spec.index("### REQ-REPORT-5010", start)
    return spec[start:end]


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        corrected_result_text=CORRECTED_RESULT_PATH.read_text(encoding="utf-8"),
        known_issues_text=KNOWN_ISSUES_PATH.read_text(encoding="utf-8"),
        duration_s=1.0,
        run_date="20260701",
        tests_run=["tests/python/test_experiment_5111_fover_in_domain_pool_v469.py"],
    )


def test_req_report_5111_spec_declares_retracted_pool_contract() -> None:
    """REQ-REPORT-5111: OpenSpec declares the FoVer pool retraction gate."""

    section = _spec_section()

    assert "REQ-REPORT-5111" in section
    assert "SCENARIO-REPORT-5111" in section
    assert "SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION" in section
    assert mod.EXPERIMENT_ID in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.CORRECTED_RESULT_RELATIVE_PATH in section
    assert "blocked_fover_indomain_pool_retracted_see_*" in section
    assert "learned-verifier AUROC `0.9663`" in section
    assert "cheap baseline AUROC `0.9635`" in section
    assert "delta AUROC `0.0028`" in section
    assert "`beats_cheap_baseline=false`" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_retracted_pool_artifact_cites_corrected_result_for_scenario_report_5111() -> None:
    """SCENARIO-REPORT-5111: corrected FoVer result is the terminal answer."""

    artifact = _artifact()

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.BLOCKED_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["pool_path"] is None
    assert artifact["pool_sha256"] is None
    assert artifact["pool_n"] == 0
    assert artifact["candidates_per_item"] == 0
    assert artifact["vote_at_1"] is None
    assert artifact["tuned_self_consistency"] is None
    assert artifact["oracle_at_k"] is None
    assert artifact["headroom_present"] is False
    assert artifact["verifier_is_oracle"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["tests_run"] == [
        "tests/python/test_experiment_5111_fover_in_domain_pool_v469.py"
    ]

    summary = artifact["corrected_result_summary"]
    assert summary["n_rows"] == 6548
    assert summary["verifier_auroc"] == 0.9663
    assert summary["cheap_baseline_auroc"] == 0.9635
    assert summary["delta_auroc"] == 0.0028
    assert summary["delta_auroc_ci95"] == [-0.0244, 0.0347]
    assert summary["beats_cheap_baseline"] is False
    assert "no natural multi-candidate structure" in summary["framing_change_from_retracted_claim"]
    assert "length" in summary["cheap_baseline_root_cause"].lower()

    assert artifact["corrected_result_path"] == mod.CORRECTED_RESULT_RELATIVE_PATH
    assert artifact["corrected_result_sha256"].startswith("sha256:")
    assert artifact["model_specs"] == {
        "generative_llms_used": [],
        "corrected_result_embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "corrected_result_cheap_baseline_model": (
            "sklearn.linear_model.LogisticRegression "
            "(8 hand-engineered text-statistical features, no learned embeddings)"
        ),
    }
    assert artifact["seeds_or_checksums"]["corrected_result_random_seed"] == 20260701
    assert artifact["preconditions_checked"]["corrected_result_read"] is True
    assert artifact["preconditions_checked"]["known_issues_retraction_found"] is True
    assert artifact["preconditions_checked"]["candidate_pool_generation_attempted"] is False
    assert artifact["retracted_claims"]["candidate_selection_headroom_claim_retracted"] is True
    assert artifact["retracted_claims"]["synthetic_pool_must_not_be_built"] is True
    assert artifact["retraction_sources"] == [
        "ops/known-issues.md#NUDGE-2026-07-01-RETRACTED",
        "ops/known-issues.md#MOAT-REDIRECT-2026-06-30-RETRACTED",
    ]


def test_missing_corrected_result_fails_closed_for_scenario_report_5111() -> None:
    """SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION: no pool is fabricated."""

    artifact = mod.build_artifact(
        corrected_result_text="{bad json",
        known_issues_text="no retraction marker here",
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert artifact["honest_verdict"] == mod.MISSING_CORRECTED_RESULT_VERDICT
    assert artifact["preconditions_checked"]["corrected_result_read"] is False
    assert artifact["preconditions_checked"]["corrected_result_parse_error"]
    assert artifact["preconditions_checked"]["known_issues_retraction_found"] is False
    assert artifact["pool_n"] == 0
    assert artifact["headroom_present"] is False
    assert artifact["corrected_result_summary"] is None
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("corrected_result_text", "parse_error"),
    [
        ("[]", "not a JSON object"),
        ("{}", "missing fields"),
    ],
)
def test_malformed_corrected_result_variants_fail_closed_for_req_report_5111(
    corrected_result_text: str,
    parse_error: str,
) -> None:
    """SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION: malformed JSON fails closed."""

    artifact = mod.build_artifact(
        corrected_result_text=corrected_result_text,
        known_issues_text=KNOWN_ISSUES_PATH.read_text(encoding="utf-8"),
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert artifact["honest_verdict"] == mod.MISSING_CORRECTED_RESULT_VERDICT
    assert parse_error in artifact["preconditions_checked"]["corrected_result_parse_error"]
    assert artifact["corrected_result_summary"] is None
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.468"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "complete_fake_pool"}, "honest_verdict"),
        (
            lambda artifact: artifact | {"inference_substrate": "live_llm_inference"},
            "inference_substrate",
        ),
        (lambda artifact: artifact | {"pool_path": "results/fake_pool.json"}, "pool_path"),
        (lambda artifact: artifact | {"pool_sha256": "sha256:abc"}, "pool_sha256"),
        (lambda artifact: artifact | {"pool_n": 150}, "pool_n"),
        (lambda artifact: artifact | {"candidates_per_item": 4}, "candidates_per_item"),
        (lambda artifact: artifact | {"vote_at_1": 0.25}, "candidate-selection metrics"),
        (
            lambda artifact: artifact | {"tuned_self_consistency": 0.50},
            "candidate-selection metrics",
        ),
        (lambda artifact: artifact | {"oracle_at_k": 0.75}, "candidate-selection metrics"),
        (lambda artifact: artifact | {"headroom_present": True}, "headroom_present"),
        (lambda artifact: artifact | {"verifier_is_oracle": True}, "verifier_is_oracle"),
        (lambda artifact: artifact | {"flagged_adversarial": True}, "flagged_adversarial"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (
            lambda artifact: artifact | {"corrected_result_path": "results/other.json"},
            "corrected_result_path",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "preconditions_checked": artifact["preconditions_checked"]
                    | {"candidate_pool_generation_attempted": True}
                }
            ),
            "candidate_pool_generation_attempted",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "preconditions_checked": artifact["preconditions_checked"]
                    | {"local_llm_generation_attempted": True}
                }
            ),
            "local_llm_generation_attempted",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "preconditions_checked": artifact["preconditions_checked"]
                    | {"pool_fabrication_blocked": False}
                }
            ),
            "pool_fabrication_blocked",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "retracted_claims": artifact["retracted_claims"]
                    | {"synthetic_pool_must_not_be_built": False}
                }
            ),
            "retracted_claims",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "corrected_result_summary": artifact["corrected_result_summary"]
                    | {"beats_cheap_baseline": True}
                }
            ),
            "beats_cheap_baseline",
        ),
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "preconditions_checked"
            },
            "required field",
        ),
        (
            lambda artifact: artifact | {"corrected_result_summary": None},
            "missing corrected_result_summary",
        ),
    ],
)
def test_validator_rejects_fabricated_pool_claims_for_req_report_5111(
    mutate: object,
    message: str,
) -> None:
    """REQ-REPORT-5111: validator rejects stale pool/headroom claims."""

    bad_artifact = mutate(copy.deepcopy(_artifact()))

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_write_artifact_is_stable_for_req_report_5111(tmp_path: Path) -> None:
    """REQ-REPORT-5111: writer emits a stable blocked artifact."""

    (tmp_path / mod.CORRECTED_RESULT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.KNOWN_ISSUES_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.CORRECTED_RESULT_RELATIVE_PATH).write_text(
        CORRECTED_RESULT_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        KNOWN_ISSUES_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    first = mod.write_artifact(
        root=tmp_path,
        duration_s=0.75,
        run_date="20260701",
        tests_run=["focused"],
    )
    second = mod.write_artifact(
        root=tmp_path,
        duration_s=0.75,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert second == first
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    assert json.loads(artifact_path.read_text(encoding="utf-8")) == first
    assert first["honest_verdict"] == mod.BLOCKED_VERDICT


def test_writer_missing_inputs_remains_blocked_for_req_report_5111(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5111-BLOCKED-MISSING-CORRECTION: missing files fail closed."""

    artifact = mod.write_artifact(
        root=tmp_path,
        duration_s=0.2,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert artifact["honest_verdict"] == mod.MISSING_CORRECTED_RESULT_VERDICT
    assert artifact["preconditions_checked"]["corrected_result_read"] is False
    assert artifact["preconditions_checked"]["known_issues_retraction_found"] is False
    assert artifact["pool_path"] is None
    mod.validate_artifact(artifact)


def test_script_main_delegates_to_tested_module_for_req_report_5111(tmp_path: Path) -> None:
    """REQ-REPORT-5111: CLI wrapper writes the same validated artifact."""

    (tmp_path / mod.CORRECTED_RESULT_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.KNOWN_ISSUES_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.CORRECTED_RESULT_RELATIVE_PATH).write_text(
        CORRECTED_RESULT_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / mod.KNOWN_ISSUES_RELATIVE_PATH).write_text(
        KNOWN_ISSUES_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )

    artifact_path = script_mod.main(
        root=tmp_path,
        date="20260701",
        duration_s=0.25,
        tests_run=["script wrapper"],
    )

    artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    assert artifact["tests_run"] == ["script wrapper"]
    assert artifact["run_date"] == "20260701"
    mod.validate_artifact(artifact)
