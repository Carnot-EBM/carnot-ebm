"""Tests for Exp 5110 V469 source-freshness SOTA ingestion.

Spec refs: REQ-REPORT-5110, SCENARIO-REPORT-5110,
SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from carnot import experiment_5110_sota_ingestion_v469 as mod
from scripts import experiment_5110_sota_ingestion_v469 as script_mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/research-reporting/spec.md"
REFERENCES_PATH = REPO / mod.REFERENCES_RELATIVE_PATH
ROADMAP_PATH = REPO / mod.ROADMAP_RELATIVE_PATH


def _spec_section() -> str:
    spec = SPEC_PATH.read_text(encoding="utf-8")
    start = spec.index("### REQ-REPORT-5110")
    end = spec.index("### REQ-REPORT-5010", start)
    return spec[start:end]


def _artifact() -> dict[str, object]:
    return mod.build_artifact(
        references_text=REFERENCES_PATH.read_text(encoding="utf-8"),
        roadmap_text=ROADMAP_PATH.read_text(encoding="utf-8"),
        duration_s=1.25,
        run_date="20260701",
        tests_run=["tests/python/test_experiment_5110_sota_ingestion_v469.py"],
    )


def test_req_report_5110_spec_declares_source_freshness_contract() -> None:
    """REQ-REPORT-5110: OpenSpec declares the V469 source-freshness contract."""

    section = _spec_section()

    assert "REQ-REPORT-5110" in section
    assert "SCENARIO-REPORT-5110" in section
    assert "SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE" in section
    assert mod.RESULT_RELATIVE_PATH in section
    assert mod.EXPERIMENT_ID in section
    assert "FoVer" in section
    assert "FormalRewardBench" in section
    assert "Extropic TSU/XTR-0" in section
    assert "Logical Intelligence Kona/Aleph" in section
    assert "research-roadmap.yaml" in section
    assert "scripts/research_conductor.py" in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle == mod.FIELD_PRINCIPLES[field]
        assert principle in section


def test_req_report_5110_artifact_maps_fresh_sources_to_v469_tasks() -> None:
    """REQ-REPORT-5110: artifact maps actionable V469 sources to exp5111-exp5120."""

    artifact = _artifact()

    mod.validate_artifact(artifact)
    assert mod.REQUIRED_USER_ARTIFACT_FIELDS.issubset(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["honest_verdict"] == mod.COMPLETE_VERDICT
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["references_section_found"] is True
    assert artifact["active_roadmap_modified"] is False
    assert artifact["flagged_adversarial"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert artifact["tests_run"] == ["tests/python/test_experiment_5110_sota_ingestion_v469.py"]

    required_topics = {source["required_topic"] for source in artifact["sources_checked"]}
    assert required_topics == mod.REQUIRED_TOPICS
    assert len(artifact["sources_checked"]) >= len(mod.REQUIRED_TOPICS)
    assert all(source["urls"] for source in artifact["sources_checked"])
    assert all(url["url"].startswith("https://") for source in artifact["sources_checked"] for url in source["urls"])
    assert all(url["status"] != "dead" for source in artifact["sources_checked"] for url in source["urls"])

    self_verification = next(
        source for source in artifact["sources_checked"] if source["source_id"] == "self_verification_cliff"
    )
    assert self_verification["source_access_status"] == "partial_openreview_challenge_metadata_verified"
    assert "browser challenge" in self_verification["access_note"]

    assert set(artifact["task_mapping"]) == mod.ACTIONABLE_SOURCE_IDS
    for source_id, task_rows in artifact["task_mapping"].items():
        assert source_id in mod.ACTIONABLE_SOURCE_IDS
        assert task_rows
        for row in task_rows:
            assert row["task_id"] in mod.ALLOWED_TASK_IDS
            assert row["task_id"] != "exp5121"
            assert row["reason"]

    background_ids = {row["source_id"] for row in artifact["background_only_sources"]}
    assert mod.REQUIRED_BACKGROUND_SOURCE_IDS.issubset(background_ids)
    assert not background_ids.intersection(artifact["task_mapping"])
    assert artifact["training_or_hardware_execution_claims_detected"] is False
    assert "EBT/ARM-EBM/Kona/TSU kept as architecture pressure" in artifact["claim_boundary_findings"]


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: artifact | {"references_section_found": False},
            "references_section_found",
        ),
        (
            lambda artifact: artifact | {"active_roadmap_modified": True},
            "active_roadmap_modified",
        ),
        (
            lambda artifact: artifact | {"training_or_hardware_execution_claims_detected": True},
            "training or hardware",
        ),
        (
            lambda artifact: artifact
            | {
                "task_mapping": {
                    key: value
                    for key, value in artifact["task_mapping"].items()
                    if key != "fover"
                }
            },
            "actionable",
        ),
        (
            lambda artifact: artifact
            | {
                "task_mapping": artifact["task_mapping"]
                | {"ebt_primary": [{"task_id": "exp5114", "reason": "bad training claim"}]}
            },
            "background-only",
        ),
        (
            lambda artifact: artifact
            | {
                "task_mapping": artifact["task_mapping"]
                | {"fover": [{"task_id": "exp5121", "reason": "capstone is outside allowed range"}]}
            },
            "exp5111-exp5120",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source
                    | {
                        "urls": [
                            url | {"status": "dead"}
                            for url in source["urls"]
                        ]
                    }
                    if source["source_id"] == "falcon"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "dead or stale",
        ),
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "sources_checked"
            },
            "required field",
        ),
    ],
)
def test_validator_rejects_stale_sources_and_false_claims_for_scenario_report_5110(
    mutate: object,
    message: str,
) -> None:
    """SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE: bad claims fail closed."""

    artifact = _artifact()
    bad_artifact = mutate(copy.deepcopy(artifact))

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_missing_references_section_writes_partial_artifact_for_scenario_report_5110() -> None:
    """SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE: missing section is partial."""

    artifact = mod.build_artifact(
        references_text="# no v469 references here\n",
        roadmap_text=ROADMAP_PATH.read_text(encoding="utf-8"),
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert artifact["references_section_found"] is False
    assert artifact["honest_verdict"] == mod.PARTIAL_REFERENCES_VERDICT
    assert artifact["sources_checked"]
    assert artifact["task_mapping"]
    mod.validate_artifact(artifact)


def test_dead_source_builds_partial_artifact_for_scenario_report_5110(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-5110-BLOCKED-OR-PARTIAL-SOURCE: dead source is preserved."""

    patched_sources = [
        source
        | {
            "urls": [
                url | {"status": "dead"}
                for url in source["urls"]
            ]
        }
        if source["source_id"] == "falcon"
        else source
        for source in mod.SOURCE_CHECKS
    ]
    monkeypatch.setattr(mod, "SOURCE_CHECKS", patched_sources)
    artifact = mod.build_artifact(
        references_text=REFERENCES_PATH.read_text(encoding="utf-8"),
        roadmap_text=ROADMAP_PATH.read_text(encoding="utf-8"),
        duration_s=0.5,
        run_date="20260701",
        tests_run=["focused"],
    )

    assert artifact["honest_verdict"] == mod.PARTIAL_SOURCE_VERDICT
    assert any(
        url["status"] == "dead"
        for source in artifact["sources_checked"]
        for url in source["urls"]
    )
    mod.validate_artifact(artifact)


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda artifact: artifact | {"experiment_id": "bad"}, "experiment_id"),
        (lambda artifact: artifact | {"milestone": "2026.07.468"}, "milestone"),
        (lambda artifact: artifact | {"honest_verdict": "finished"}, "honest_verdict"),
        (lambda artifact: artifact | {"inference_substrate": "live_llm_inference"}, "inference_substrate"),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source | {"required_topic": "missing-topic"}
                    if source["source_id"] == "fover"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "source coverage",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source | {"urls": []}
                    if source["source_id"] == "fover"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "no real URL",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source | {"source_id": "rogue_actionable"}
                    if source["source_id"] == "fover"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "unknown actionable",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source
                    | {
                        "urls": [
                            url | {"url": "http://example.invalid"}
                            for url in source["urls"]
                        ]
                    }
                    if source["source_id"] == "fover"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "not HTTPS",
        ),
        (
            lambda artifact: artifact | {"background_only_sources": []},
            "background-only architecture",
        ),
        (
            lambda artifact: artifact
            | {
                "sources_checked": [
                    source | {"source_id": "fover_missing", "claim_boundary": "background_only"}
                    if source["source_id"] == "fover"
                    else source
                    for source in artifact["sources_checked"]
                ]
            },
            "unknown source",
        ),
        (
            lambda artifact: artifact
            | {"task_mapping": artifact["task_mapping"] | {"fover": []}},
            "no task rows",
        ),
        (
            lambda artifact: artifact
            | {
                "task_mapping": artifact["task_mapping"]
                | {"fover": [{"task_id": "exp5111", "task_name": "bad", "reason": ""}]}
            },
            "needs a reason",
        ),
        (
            lambda artifact: artifact | {"claim_boundary_findings": []},
            "claim_boundary_findings",
        ),
    ],
)
def test_validator_covers_schema_error_branches_for_req_report_5110(
    mutate: object,
    message: str,
) -> None:
    """REQ-REPORT-5110: validator rejects each malformed schema branch."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(copy.deepcopy(_artifact())))


def test_write_artifact_is_stable_for_req_report_5110(tmp_path: Path) -> None:
    """REQ-REPORT-5110: writer emits a stable source-freshness artifact."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(
        REFERENCES_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        ROADMAP_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / "research-roadmap.yaml").write_text("milestone: 2026.07.468\n", encoding="utf-8")

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
    assert first["active_roadmap_modified"] is False
    assert (tmp_path / "research-roadmap.yaml").read_text(encoding="utf-8") == "milestone: 2026.07.468\n"


def test_script_main_delegates_to_tested_module_for_req_report_5110(tmp_path: Path) -> None:
    """REQ-REPORT-5110: CLI wrapper writes the same validated artifact."""

    (tmp_path / mod.REFERENCES_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).parent.mkdir(parents=True, exist_ok=True)
    (tmp_path / mod.REFERENCES_RELATIVE_PATH).write_text(
        REFERENCES_PATH.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (tmp_path / mod.ROADMAP_RELATIVE_PATH).write_text(
        ROADMAP_PATH.read_text(encoding="utf-8"), encoding="utf-8"
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
