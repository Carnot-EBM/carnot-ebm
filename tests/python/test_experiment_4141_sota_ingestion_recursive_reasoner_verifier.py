"""Tests for REQ-REPORT-4141 / SCENARIO-REPORT-4141."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import experiment_4141_sota_ingestion_recursive_reasoner_verifier as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-recursive-reasoner-verifier-2026-06-13.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4141_sota_ingestion_recursive_reasoner_verifier.json"
)
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "name": "GRAM stochastic-latent generator",
            "arxiv_id_or_url": "2605.19376",
            "url": "https://arxiv.org/abs/2605.19376",
            "implementation_over_stack": (
                "Treat GRAM as the .384 generator candidate only after the TRM graft "
                "shows non-oracle verifier value and best-of-K headroom."
            ),
            "failure_mode": (
                "A stronger generator can erase rerank headroom; without verifier_value_added "
                "the graft would only benchmark GRAM rather than Carnot verifier value."
            ),
        },
        {
            "name": "TRM thinking reward for RLVR/GRPO",
            "arxiv_id_or_url": "2602.08498",
            "url": "https://arxiv.org/abs/2602.08498",
            "implementation_over_stack": (
                "Use verified-correct trace filtering as the precedent for the .383 "
                "RFT A-vs-B de-confound: verifier-certified labels versus vote labels."
            ),
            "failure_mode": (
                "If correctness filtering is mixed with label-source effects, the run "
                "cannot distinguish verifier reward from generic adaptation compute."
            ),
        },
        {
            "name": "Weaver weak-verifier weighted ensemble",
            "arxiv_id_or_url": "2506.18203",
            "url": "https://arxiv.org/abs/2506.18203",
            "implementation_over_stack": (
                "Make the .383 non-oracle ensemble-rerank headline a weighted weak-verifier "
                "baseline rather than a single executable-oracle rerank."
            ),
            "failure_mode": (
                "Weak-verifier weights can overfit correlated errors; oracle(best-of-K) must "
                "beat vote before any null rerank is interpreted."
            ),
        },
    ]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v384="gram_as_generator_if_verifier_value_added_and_headroom_present_v384",
    )


def test_req_report_4141_spec_anchor_exists() -> None:
    """REQ-REPORT-4141: OpenSpec declares recursive-reasoner ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4141" in spec
    assert "SCENARIO-REPORT-4141" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert "flagged_for_v384" in spec
    assert "arXiv:2605.19376" in spec
    assert "arXiv:2602.08498" in spec
    assert "arXiv:2506.18203" in spec


def test_build_artifact_has_required_schema_fields_for_req_report_4141() -> None:
    """REQ-REPORT-4141: artifact exposes the required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": "complete: sota_ingestion_recursive_reasoner_verifier_mapped",
        "methods_mapped": _valid_methods(),
        "flagged_for_v384": (
            "gram_as_generator_if_verifier_value_added_and_headroom_present_v384"
        ),
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method/source MUST carry a real arXiv ID or canonical doc URL; "
                "an ingestion note without verifiable citations is treated as fabrication."
            ),
            "flagged_for_v384": (
                "Closes the discover->ingest->plan loop: names the strongest method "
                "for the next planner (candidate: GRAM-as-generator IF verifier_value_added)."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods_mapped": []}, "at least three"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "at least three"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "implementation_over_stack": "fake",
                        "failure_mode": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv ID or canonical URL",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "GRAM",
                        "arxiv_id_or_url": "2605.19376",
                        "url": "https://example.com/2605.19376",
                        "implementation_over_stack": "use it",
                        "failure_mode": "breaks",
                    }
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [{"name": "GRAM", "arxiv_id_or_url": "2605.19376"}] + _valid_methods()[1:]},
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}] + _valid_methods()[1:]},
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v384": ""}, "flagged_for_v384"),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4141(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4141: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_extra_and_bad_method_rows() -> None:
    """SCENARIO-REPORT-4141: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "aggregation_from_upstream_artifacts"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {"methods_mapped": ["not-a-dict"] + _valid_methods()[1:]}
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4141_sections() -> None:
    """SCENARIO-REPORT-4141: note maps sources to implementation work and risks."""

    note = """
    # SOTA ingestion recursive reasoner verifier

    ## Current .383 recursive-reasoner plus verifier anchor
    arXiv:2605.19376, arXiv:2602.08498, and arXiv:2506.18203 are the source anchors.

    ## GRAM stochastic-latent generator
    arXiv:2605.19376.
    Implementation over nano-trm + Carnot-verifier stack: use GRAM after verifier_value_added.
    Pitfalls / where it fails: no headroom means no verifier claim.

    ## TRM thinking reward for RLVR/GRPO
    arXiv:2602.08498.
    Implementation over nano-trm + Carnot-verifier stack: de-confound labels.
    Pitfalls / where it fails: reward signal is mixed with adaptation.

    ## Weaver weak-verifier weighted ensemble
    arXiv:2506.18203.
    Implementation over nano-trm + Carnot-verifier stack: weighted non-oracle rerank.
    Pitfalls / where it fails: correlated verifier errors.

    ## Flagged for the .384 roadmap
    gram_as_generator_if_verifier_value_added_and_headroom_present_v384
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_roadmap_flag() -> None:
    """SCENARIO-REPORT-4141: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Flagged for the .384 roadmap"):
        mod.validate_markdown_note(
            "## Current .383 recursive-reasoner plus verifier anchor\n"
            "## GRAM stochastic-latent generator\n"
            "arXiv:2605.19376.\n"
            "Implementation over nano-trm + Carnot-verifier stack\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_sources() -> None:
    """SCENARIO-REPORT-4141: every mapped method cites a verified source."""

    note = """
    ## Current .383 recursive-reasoner plus verifier anchor
    ## GRAM stochastic-latent generator
    arXiv:2605.19376.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## TRM thinking reward for RLVR/GRPO
    arXiv:2602.08498.
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Weaver weak-verifier weighted ensemble
    Implementation over nano-trm + Carnot-verifier stack.
    Pitfalls / where it fails.
    ## Flagged for the .384 roadmap
    gram_as_generator_if_verifier_value_added_and_headroom_present_v384
    """

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(note)


def test_write_outputs_updates_files_idempotently_for_req_report_4141(tmp_path: Path) -> None:
    """REQ-REPORT-4141: writer emits note, artifact, and one studying section."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\nExisting body.\n", encoding="utf-8")

    artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )
    second_artifact = mod.write_outputs(
        note_path=note_path,
        artifact_path=artifact_path,
        studying_path=studying_path,
    )

    mod.validate_artifact(artifact)
    mod.validate_artifact(second_artifact)
    mod.validate_markdown_note(note_path.read_text(encoding="utf-8"))
    saved_artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    studying = studying_path.read_text(encoding="utf-8")

    assert saved_artifact == artifact
    assert studying.count("2026-06-13 Exp 4141") == 1
    assert "Flagged for .384" in studying


def test_studying_section_update_handles_heading_layouts_for_req_report_4141() -> None:
    """REQ-REPORT-4141: studying update works before or between sections."""

    without_marker = "# Research Studying\n\n## Existing\nBody.\n"
    with_marker_and_next = mod._with_studying_section(without_marker)
    refreshed = mod._with_studying_section(with_marker_and_next)
    no_heading = mod._with_studying_section("# Research Studying\nOnly body.\n")
    marker_at_end = mod._with_studying_section(with_marker_and_next.split("\n## Existing")[0])

    assert with_marker_and_next.index("2026-06-13 Exp 4141") < with_marker_and_next.index(
        "## Existing"
    )
    assert refreshed.count("2026-06-13 Exp 4141") == 1
    assert "## Existing\nBody." in refreshed
    assert no_heading.rstrip().endswith("not as an unconditional rerank claim.")
    assert marker_at_end.count("2026-06-13 Exp 4141") == 1


def test_deliverable_files_validate_against_req_report_4141() -> None:
    """REQ-REPORT-4141: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v384"] == (
        "gram_as_generator_if_verifier_value_added_and_headroom_present_v384"
    )
    assert (
        "2026-06-13 Exp 4141 - .383 recursive-reasoner/verifier SOTA ingestion ingested"
        in studying
    )
    assert (
        "Flagged for .384: "
        "`gram_as_generator_if_verifier_value_added_and_headroom_present_v384`"
    ) in studying
