"""Tests for Exp 5258 V481 execution-time SOTA refresh.

Spec refs: REQ-REPORT-5258, SCENARIO-REPORT-5258-APPEND-DELTAS,
SCENARIO-REPORT-5258-NOOP.
"""

from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5258_sota_refresh_v481 as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "research-reporting" / "spec.md"
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH


def test_req_report_5258_spec_declares_v481_refresh_contract() -> None:
    """REQ-REPORT-5258: OpenSpec anchors the execution-time SOTA refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5258") : spec.index("### REQ-REPORT-5162")]

    for marker in (
        "REQ-REPORT-5258",
        "SCENARIO-REPORT-5258-APPEND-DELTAS",
        "SCENARIO-REPORT-5258-NOOP",
        str(mod.RESULT_RELATIVE_PATH),
        "literature_ingestion_network_sources",
        "Semantic Scholar",
        "arXiv:2507.02092",
        "arXiv:2512.15605",
        "/deep-research",
        "research-roadmap.yaml",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_required_fields_and_new_deltas() -> None:
    """REQ-REPORT-5258: artifact records source-verified actionable deltas."""

    artifact = mod.build_artifact(
        duration_s=3.5,
        tests_run=["tests/python/test_experiment_5258_sota_refresh_v481.py"],
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["honest_verdict"]["value"].startswith("complete:")
    assert "new actionable" in artifact["honest_verdict"]["value"]
    assert artifact["new_references_added"]["value"] == len(mod.ACTIONABLE_DELTAS)
    assert artifact["references_md_updated"]["value"] is True
    assert artifact["retired_scope_reopened"]["value"] is False
    assert artifact["no_deep_research_used"] is True
    assert artifact["research_roadmap_yaml_modified"] is False
    assert artifact["research_conductor_modified"] is False
    assert artifact["plan_change_required"] is False
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES

    sources = artifact["sources_checked"]["value"]
    assert set(mod.REQUIRED_SOURCE_FAMILIES).issubset(sources)
    assert sources["semantic_scholar"]["status"] == "ok"
    assert sources["github"]["status"] == "ok"

    deltas = artifact["actionable_deltas"]["value"]
    assert [row["title"] for row in deltas] == [row["title"] for row in mod.ACTIONABLE_DELTAS]
    assert all(row["source_url"].startswith("https://") for row in deltas)
    assert {row["planned_task_impact"] for row in deltas} == {"no_plan_edit"}
    assert {row["retired_scope_risk"] for row in deltas} == {"none"}
    assert any(row["arxiv_id_or_repo"] == "2606.29961" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "2606.32034" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "2607.01071" for row in deltas)
    assert any(row["arxiv_id_or_repo"] == "xiaohanma-oss/PLN-THRML" for row in deltas)

    semantic = artifact["semantic_scholar_status"]["value"]
    assert semantic["EBT"]["arxiv_id"] == "2507.02092"
    assert semantic["EBT"]["citationCount"] == 26
    assert semantic["ARM-EBM"]["arxiv_id"] == "2512.15605"
    assert semantic["ARM-EBM"]["citationCount"] == 8
    assert "LoopUS" in {row["title"] for row in semantic["EBT"]["citation_samples"]}


def test_scenario_report_5258_appends_refresh_section_idempotently() -> None:
    """SCENARIO-REPORT-5258-APPEND-DELTAS: references append once without rewriting history."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n## V481 Research Update - 2026-07-05\nOld V481 section.\n"

    updated = mod.append_refresh_section(original, artifact)
    second = mod.append_refresh_section(updated, artifact)

    assert updated == second
    assert updated.startswith(original)
    assert updated.count(mod.REFRESH_HEADING) == 1
    assert "DuoMem" in updated
    assert "QVal" in updated
    assert "MemSyco-Bench" in updated
    assert "PLN-THRML" in updated
    assert "No executable `.481` task edit is required" in updated
    assert mod.REFRESH_END_MARKER in updated


def test_scenario_report_5258_noop_leaves_references_unchanged() -> None:
    """SCENARIO-REPORT-5258-NOOP: zero-new refresh does not churn references."""

    artifact = mod.build_artifact(actionable_deltas=[])
    original = "# Research References\n\n## V481 Research Update - 2026-07-05\nOld V481 section.\n"

    mod.validate_artifact(artifact)
    assert artifact["new_references_added"]["value"] == 0
    assert artifact["references_md_updated"]["value"] is False
    assert artifact["actionable_deltas"]["value"] == []
    assert "no new actionable" in artifact["honest_verdict"]["value"]
    assert mod.append_refresh_section(original, artifact) == original


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: {
                key: value for key, value in artifact.items() if key != "sources_checked"
            },
            "missing required fields",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "honest_verdict": {
                        "value": "done",
                        "principle": mod.FIELD_PRINCIPLES["honest_verdict"],
                    }
                }
            ),
            "honest_verdict",
        ),
        (
            lambda artifact: artifact | {"honest_verdict": "complete: unwrapped"},
            "principle-wrapped",
        ),
        (
            lambda artifact: (
                artifact | {"honest_verdict": {"value": "complete: ok", "principle": "wrong"}}
            ),
            "declared principle",
        ),
        (
            lambda artifact: (
                artifact | {"honest_verdict": {"principle": mod.FIELD_PRINCIPLES["honest_verdict"]}}
            ),
            "missing value",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "inference_substrate": {
                        "value": "cached_fixture_replay_no_llm",
                        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    }
                }
            ),
            "inference_substrate",
        ),
        (lambda artifact: artifact | {"no_deep_research_used": False}, "deep-research"),
        (
            lambda artifact: artifact | {"research_roadmap_yaml_modified": True},
            "research-roadmap.yaml",
        ),
        (
            lambda artifact: artifact | {"research_conductor_modified": True},
            "research_conductor",
        ),
        (lambda artifact: artifact | {"field_principles": {}}, "field_principles"),
        (
            lambda artifact: (
                artifact
                | {
                    "references_md_updated": {
                        "value": False,
                        "principle": mod.FIELD_PRINCIPLES["references_md_updated"],
                    }
                }
            ),
            "references_md_updated",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": {
                        "value": 0,
                        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
                    }
                }
            ),
            "new_references_added",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": "none",
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "actionable_deltas",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0] | {"source_url": "arxiv:bad"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "verified URL",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0]
                            | {"planned_task_impact": "plan_edit"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "active plan",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [
                            artifact["actionable_deltas"]["value"][0]
                            | {"retired_scope_risk": "reopened"}
                        ],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "retired scopes",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "actionable_deltas": {
                        "value": [{"title": "Incomplete"}],
                        "principle": mod.FIELD_PRINCIPLES["actionable_deltas"],
                    }
                }
            ),
            "rows must include",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": {
                        "value": {"arxiv": artifact["sources_checked"]["value"]["arxiv"]},
                        "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                    }
                }
            ),
            "sources_checked",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "semantic_scholar_status": {
                        "value": {"EBT": artifact["semantic_scholar_status"]["value"]["EBT"]},
                        "principle": mod.FIELD_PRINCIPLES["semantic_scholar_status"],
                    }
                }
            ),
            "Semantic Scholar",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "semantic_scholar_status": {
                        "value": artifact["semantic_scholar_status"]["value"]
                        | {
                            "EBT": artifact["semantic_scholar_status"]["value"]["EBT"]
                            | {"citationCount": "26"}
                        },
                        "principle": mod.FIELD_PRINCIPLES["semantic_scholar_status"],
                    }
                }
            ),
            "integer citationCount",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "semantic_scholar_status": {
                        "value": artifact["semantic_scholar_status"]["value"]
                        | {
                            "ARM-EBM": artifact["semantic_scholar_status"]["value"]["ARM-EBM"]
                            | {"citation_samples": []}
                        },
                        "principle": mod.FIELD_PRINCIPLES["semantic_scholar_status"],
                    }
                }
            ),
            "citation samples",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "retired_scope_reopened": {
                        "value": True,
                        "principle": mod.FIELD_PRINCIPLES["retired_scope_reopened"],
                    }
                }
            ),
            "retired_scope_reopened",
        ),
        (lambda artifact: artifact | {"tests_run": []}, "tests_run"),
    ],
)
def test_validate_artifact_rejects_invalid_or_unverified_rows(
    mutate: Callable[[dict[str, Any]], dict[str, Any]], message: str
) -> None:
    """REQ-REPORT-5258: invalid refresh artifacts fail closed before writing."""

    artifact = mod.build_artifact()

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


def test_write_outputs_writes_json_and_refresh_section(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5258-APPEND-DELTAS: writer emits stable JSON and references."""

    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path.write_text(
        "# Research References\n\n## V481 Research Update - 2026-07-05\nFixture.\n",
        encoding="utf-8",
    )

    artifact = mod.write_outputs(
        root=tmp_path,
        references_path=references_path,
        result_path=result_path,
        tests_run=["focused"],
    )
    second = mod.write_outputs(
        root=tmp_path,
        references_path=references_path,
        result_path=result_path,
        tests_run=["focused"],
    )

    assert second == artifact
    assert json.loads(result_path.read_text(encoding="utf-8")) == artifact
    references = references_path.read_text(encoding="utf-8")
    assert references.count(mod.REFRESH_HEADING) == 1
    assert artifact["references_md_updated"]["value"] is True


def test_req_report_5258_repository_artifact_matches_schema() -> None:
    """REQ-REPORT-5258: checked-in deliverable is a valid refresh artifact."""

    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["inference_substrate"]["value"] == mod.INFERENCE_SUBSTRATE
    assert artifact["new_references_added"]["value"] == len(mod.ACTIONABLE_DELTAS)
    assert artifact["references_md_updated"]["value"] is True
    assert artifact["retired_scope_reopened"]["value"] is False
