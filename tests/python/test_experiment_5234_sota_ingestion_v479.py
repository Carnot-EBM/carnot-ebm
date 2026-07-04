"""Tests for Exp 5234 reserved V479 SOTA ingestion.

Spec refs: REQ-REPORT-5234, SCENARIO-REPORT-5234,
SCENARIO-REPORT-5234-BLOCKED-METADATA.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5234_sota_ingestion_v479 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")


def _value(field: dict[str, Any]) -> Any:
    return field["value"]


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact = dict(artifact)
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_report_5234_spec_declares_reserved_ingestion_contract() -> None:
    """REQ-REPORT-5234: OpenSpec anchors the reserved V479 refresh."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-REPORT-5234") : spec.index("### REQ-REPORT-5221")]

    for marker in (
        "REQ-REPORT-5234",
        "SCENARIO-REPORT-5234",
        "SCENARIO-REPORT-5234-BLOCKED-METADATA",
        str(mod.RESULT_RELATIVE_PATH),
        "arXiv:2507.02092",
        "arXiv:2512.15605",
        "research-references.md",
        "ops/known-issues.md",
        "literature_ingestion",
        "scripts/research_conductor.py",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert principle in section


def test_build_artifact_records_v479_noop_refresh_with_retry_metadata() -> None:
    """SCENARIO-REPORT-5234: no new actionable finding is an honest result."""

    artifact = mod.build_artifact(
        tests_run=["tests/python/test_experiment_5234_sota_ingestion_v479.py"]
    )

    mod.validate_artifact(artifact)
    assert artifact["experiment_id"] == mod.EXPERIMENT_ID
    assert artifact["milestone"] == mod.MILESTONE
    assert artifact["field_principles"] == mod.FIELD_PRINCIPLES
    assert _value(artifact["new_references_added"]) == 0
    assert artifact["references_md_updated"]["value"] is False
    assert artifact["no_deep_research_used"]["value"] is True
    assert artifact["research_conductor_py_untouched_confirmed"]["value"] is True
    assert _value(artifact["retired_scope_reopened"]) is False
    assert _value(artifact["inference_substrate"]) == mod.INFERENCE_SUBSTRATE
    assert _value(artifact["honest_verdict"]).startswith("complete:")
    assert "no new actionable" in _value(artifact["honest_verdict"])

    ebt_status = _value(artifact["ebt_semantic_scholar_status"])
    assert ebt_status["title"] == "Energy-Based Transformers are Scalable Learners and Thinkers"
    assert ebt_status["year"] == 2025
    assert ebt_status["citation_count"] == 26
    assert ebt_status["blocked"] is False
    assert ebt_status["backoff_behavior"] == [
        "attempt=1 http_code=429",
        "attempt=2 http_code=200",
    ]

    arm_status = _value(artifact["arm_ebm_semantic_scholar_status"])
    assert (
        arm_status["title"] == "Autoregressive Language Models are Secretly Energy-Based Models: "
        "Insights into the Lookahead Capabilities of Next-Token Prediction"
    )
    assert arm_status["year"] == 2025
    assert arm_status["citation_count"] == 8
    assert arm_status["blocked"] is False
    assert arm_status["backoff_behavior"] == [
        "attempt=1 http_code=429",
        "attempt=2 http_code=200",
    ]

    checked_groups = {row["group"] for row in _value(artifact["sources_checked"])}
    assert {
        "local V478/V479 planning context",
        "arXiv API and primary abstract refresh",
        "OpenReview page status",
        "Hugging Face Papers",
        "Semantic Scholar Graph API",
        "Extropic public writing",
        "Logical Intelligence public posts",
        "GitHub repository refresh",
    }.issubset(checked_groups)

    mappings = {
        row["source"]: row["mapped_task_or_defer"]
        for row in _value(artifact["sota_to_experiment_mapping"])
    }
    assert mappings["Free-Energy / FLaG hallucination diagnostics"] == "defer/no action"
    assert mappings["JANUS / Hard-CSP constraint refresh"] == "exp5236"
    assert mappings["RW-EBR / AgentCL controlled memory refresh"] == "exp5239"
    assert mappings["HF July 3 memory and skill papers"] == "exp5239"
    assert mappings["Analog KAN / KAN abstraction refresh"] == "exp5242"
    assert mappings["Extropic / p-bit hardware refresh"] == "exp5243"
    assert mappings["Logical Intelligence Kona/Aleph posts"] == "defer/no action"
    assert mappings["Semantic Scholar EBT metadata retry"] == "defer/no action"
    assert mappings["Semantic Scholar ARM-EBM metadata retry"] == "defer/no action"


def test_noop_refresh_leaves_references_text_unchanged() -> None:
    """SCENARIO-REPORT-5234: no new references means no references append."""

    artifact = mod.build_artifact()
    original = "# Research References\n\n## V479 Research Update - 2026-07-04\nExisting.\n"

    updated = mod.maybe_append_references(original, artifact)
    second = mod.maybe_append_references(updated, artifact)

    assert updated == original
    assert second == original


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
            lambda artifact: artifact | {"sources_checked": []},
            "principle-wrapped",
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
            lambda artifact: (
                artifact
                | {
                    "inference_substrate": {
                        "value": "live_llm",
                        "principle": mod.FIELD_PRINCIPLES["inference_substrate"],
                    }
                }
            ),
            "inference_substrate",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": {
                        "value": 1,
                        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
                    }
                }
            ),
            "references_md_updated",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "references_md_updated": {
                        "value": True,
                        "principle": mod.EXTRA_FIELD_PRINCIPLES["references_md_updated"],
                    }
                }
            ),
            "new_references_added",
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
            "retired_scope",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "ebt_semantic_scholar_status": {
                        "value": {"blocked": False, "citation_count": "26"},
                        "principle": mod.FIELD_PRINCIPLES["ebt_semantic_scholar_status"],
                    }
                }
            ),
            "EBT",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "arm_ebm_semantic_scholar_status": {
                        "value": {"blocked": False, "citation_count": 8},
                        "principle": mod.FIELD_PRINCIPLES["arm_ebm_semantic_scholar_status"],
                    }
                }
            ),
            "ARM-EBM",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sota_to_experiment_mapping": {
                        "value": [
                            *_value(artifact["sota_to_experiment_mapping"]),
                            {
                                "source": "new paper",
                                "mapped_task_or_defer": "exp9999-new-task",
                                "reason": "bad",
                            },
                        ],
                        "principle": mod.FIELD_PRINCIPLES["sota_to_experiment_mapping"],
                    }
                }
            ),
            "existing .479",
        ),
        (
            lambda artifact: artifact | {"tests_run": []},
            "tests_run",
        ),
    ],
)
def test_validate_artifact_fails_closed_for_fabrication_or_scope_creep(
    mutate: object, message: str
) -> None:
    """SCENARIO-REPORT-5234-BLOCKED-METADATA: invalid metadata cannot pass."""

    artifact = mod.build_artifact(
        tests_run=["tests/python/test_experiment_5234_sota_ingestion_v479.py"]
    )

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": {
                        "value": _value(artifact["sources_checked"]),
                        "principle": "wrong",
                    }
                }
            ),
            "declared principle",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "no_deep_research_used": {
                        "value": False,
                        "principle": mod.EXTRA_FIELD_PRINCIPLES["no_deep_research_used"],
                    }
                }
            ),
            "deep-research",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "research_conductor_py_untouched_confirmed": {
                        "value": False,
                        "principle": mod.EXTRA_FIELD_PRINCIPLES[
                            "research_conductor_py_untouched_confirmed"
                        ],
                    }
                }
            ),
            "conductor",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": {
                        "value": [],
                        "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                    }
                }
            ),
            "sources_checked value",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sources_checked": {
                        "value": [{"group": "incomplete"}],
                        "principle": mod.FIELD_PRINCIPLES["sources_checked"],
                    }
                }
            ),
            "sources_checked rows",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "new_references_added": {
                        "value": -1,
                        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
                    }
                }
            ),
            "new_references_added must be",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "ebt_semantic_scholar_status": {
                        "value": {"blocked": False, "citation_count": -1},
                        "principle": mod.FIELD_PRINCIPLES["ebt_semantic_scholar_status"],
                    }
                }
            ),
            "EBT",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "arm_ebm_semantic_scholar_status": {
                        "value": "rate_limited",
                        "principle": mod.FIELD_PRINCIPLES["arm_ebm_semantic_scholar_status"],
                    }
                }
            ),
            "ARM-EBM",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "arm_ebm_semantic_scholar_status": {
                        "value": {
                            "blocked": "unknown",
                            "citation_count": "blocked",
                            "raw_status": "HTTP/2 429 Too Many Requests",
                            "backoff_behavior": ["attempt=1 http_code=429"],
                        },
                        "principle": mod.FIELD_PRINCIPLES["arm_ebm_semantic_scholar_status"],
                    }
                }
            ),
            "ARM-EBM",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sota_to_experiment_mapping": {
                        "value": [],
                        "principle": mod.FIELD_PRINCIPLES["sota_to_experiment_mapping"],
                    }
                }
            ),
            "sota_to_experiment_mapping value",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sota_to_experiment_mapping": {
                        "value": ["bad"],
                        "principle": mod.FIELD_PRINCIPLES["sota_to_experiment_mapping"],
                    }
                }
            ),
            "mapping rows must be objects",
        ),
        (
            lambda artifact: (
                artifact
                | {
                    "sota_to_experiment_mapping": {
                        "value": [{"source": "x"}],
                        "principle": mod.FIELD_PRINCIPLES["sota_to_experiment_mapping"],
                    }
                }
            ),
            "mapping rows must include",
        ),
        (
            lambda artifact: artifact | {"source_urls": []},
            "source_urls",
        ),
        (
            lambda artifact: artifact | {"source_api_responses": {}},
            "source_api_responses",
        ),
        (
            lambda artifact: artifact | {"source_api_responses": []},
            "source_api_responses",
        ),
        (
            lambda artifact: artifact | {"field_principles": {"wrong": "value"}},
            "field_principles",
        ),
        (
            lambda artifact: artifact | {"reproducibility_checksum": "bad"},
            "reproducibility_checksum",
        ),
    ],
)
def test_validate_artifact_covers_additional_fail_closed_branches(
    mutate: object, message: str
) -> None:
    """REQ-REPORT-5234: defensive validation branches are covered directly."""

    artifact = mod.build_artifact(tests_run=["focused"])

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(mutate(artifact))


@pytest.mark.parametrize(
    "status",
    [
        {
            "raw_status": "HTTP/2 429 Too Many Requests",
            "title": None,
            "year": None,
            "citation_count": "blocked",
            "blocked": True,
            "backoff_behavior": ["attempt=1 http_code=429"],
        },
        {
            "raw_status": "error: Semantic Scholar temporary outage",
            "title": None,
            "year": None,
            "citation_count": "blocked",
            "blocked": True,
            "backoff_behavior": ["attempt=1 error=timeout"],
        },
    ],
)
def test_validate_artifact_accepts_blocked_semantic_scholar_status(
    status: dict[str, Any],
) -> None:
    """SCENARIO-REPORT-5234-BLOCKED-METADATA: exact blocked statuses may vary."""

    artifact = mod.build_artifact(tests_run=["focused"])
    artifact["ebt_semantic_scholar_status"] = {
        "value": status,
        "principle": mod.FIELD_PRINCIPLES["ebt_semantic_scholar_status"],
    }

    mod.validate_artifact(_with_checksum(artifact))


def test_maybe_append_references_handles_positive_append_cases() -> None:
    """REQ-REPORT-5234: positive reference additions append idempotently."""

    artifact = mod.build_artifact(tests_run=["focused"])
    artifact["new_references_added"] = {
        "value": 1,
        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
    }
    artifact["references_md_updated"] = {
        "value": True,
        "principle": mod.EXTRA_FIELD_PRINCIPLES["references_md_updated"],
    }
    artifact["reference_appendix"] = "## V479 Execution Refresh\nNew item.\n"

    appended = mod.maybe_append_references("Existing", artifact)
    already_present = mod.maybe_append_references(appended, artifact)
    appended_after_newline = mod.maybe_append_references("Existing\n", artifact)

    assert appended == "Existing\n## V479 Execution Refresh\nNew item.\n"
    assert already_present == appended
    assert appended_after_newline == "Existing\n## V479 Execution Refresh\nNew item.\n"


def test_maybe_append_references_requires_appendix_for_positive_count() -> None:
    """REQ-REPORT-5234: positive counts cannot imply unstated reference text."""

    artifact = mod.build_artifact(tests_run=["focused"])
    artifact["new_references_added"] = {
        "value": 1,
        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
    }

    with pytest.raises(ValueError, match="reference_appendix"):
        mod.maybe_append_references("Existing", artifact)


def test_write_outputs_appends_when_builder_reports_new_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-REPORT-5234: writer path supports the positive branch when present."""

    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path.parent.mkdir(parents=True, exist_ok=True)
    references_path.write_text("Existing\n", encoding="utf-8")
    artifact = mod.build_artifact(tests_run=["patched"])
    artifact["new_references_added"] = {
        "value": 1,
        "principle": mod.FIELD_PRINCIPLES["new_references_added"],
    }
    artifact["references_md_updated"] = {
        "value": True,
        "principle": mod.EXTRA_FIELD_PRINCIPLES["references_md_updated"],
    }
    artifact["reference_appendix"] = "## V479 Execution Refresh\nNew item.\n"
    artifact = _with_checksum(artifact)

    monkeypatch.setattr(mod, "build_artifact", lambda **_: artifact)

    written = mod.write_outputs(
        root=tmp_path,
        references_path=references_path,
        result_path=result_path,
        tests_run=["ignored"],
    )

    assert written == artifact
    assert "New item." in references_path.read_text(encoding="utf-8")


def test_write_outputs_writes_json_and_preserves_references(tmp_path: Path) -> None:
    """SCENARIO-REPORT-5234: writer emits stable JSON and no references delta."""

    references_path = tmp_path / mod.REFERENCES_RELATIVE_PATH
    result_path = tmp_path / mod.RESULT_RELATIVE_PATH
    references_path.parent.mkdir(parents=True, exist_ok=True)
    original = "# Research References\n\n## V479 Research Update - 2026-07-04\nExisting.\n"
    references_path.write_text(original, encoding="utf-8")

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

    assert artifact["reproducibility_checksum"] == second["reproducibility_checksum"]
    assert references_path.read_text(encoding="utf-8") == original
    written = json.loads(result_path.read_text(encoding="utf-8"))
    assert written == artifact
    assert _value(written["new_references_added"]) == 0
    assert written["references_md_updated"]["value"] is False


def test_main_writes_artifact_and_prints_json(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-5234: CLI entrypoint uses the same deterministic writer."""

    assert mod.main(["--root", str(tmp_path), "--test-run", "cli-focused"]) == 0

    captured = capsys.readouterr()
    printed = json.loads(captured.out)
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert printed == written
    assert written["tests_run"] == ["cli-focused"]
