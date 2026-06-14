"""Tests for REQ-REPORT-4192 / SCENARIO-REPORT-4192."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4192_sota_ingestion_efficiency_gap4_diffusion as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-efficiency-gap4-diffusion-v389-2026-06-14.md"
)
ARTIFACT_PATH = Path(
    "results/experiment_4192_sota_ingestion_efficiency_gap4_diffusion.json"
)
WRAPPER_PATH = Path(
    "results/experiment_4192_sota_ingestion_efficiency_gap4_diffusion.py"
)
STUDYING_PATH = Path("research-studying.md")
EXCLUSION_PATH = Path("ops/exclusion_manifest.yaml")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_cem_flag() -> dict[str, object]:
    return dict(mod.DEFAULT_CEM_OPERATOR_AUTHORIZATION_FLAG)


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        cem_operator_authorization_flag=_valid_cem_flag(),
        flagged_for_v389=mod.DEFAULT_FLAGGED_FOR_V389,
    )


def test_req_report_4192_spec_anchor_exists() -> None:
    """REQ-REPORT-4192: OpenSpec declares the .389 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4192" in spec
    assert "SCENARIO-REPORT-4192" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "cem_operator_authorization_flag" in spec
    assert "flagged_for_v389" in spec
    assert "auto_activation_recommended == false" in spec
    for source in mod.VERIFIED_SOURCE_URLS:
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4192() -> None:
    """REQ-REPORT-4192: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "cem_operator_authorization_flag": _valid_cem_flag(),
        "flagged_for_v389": mod.DEFAULT_FLAGGED_FOR_V389,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note "
                "without verifiable citations is treated as fabrication "
                "(adversarial_verify discipline)."
            ),
            "cem_operator_authorization_flag": (
                "Explicitly records that CEM (2510.20607) needs operator "
                "authorization before activation (the retired "
                "trained-content-energy selector lineage) - closes the loop "
                "honestly instead of silently dropping or auto-running it."
            ),
            "flagged_for_v389": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the next planner."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (
            _valid_artifact() | {"field_principles": {"honest_verdict": "loose"}},
            "field_principles",
        ),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "at least three"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "carnot_stack_mapping": "fake",
                        "implication": "fake",
                        "failure_mode": "fake",
                        "experiment_mapping": "fake",
                    }
                ]
                + _valid_methods()[1:]
            },
            "verified source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"url": "https://example.com/2602.22871"}
                ]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {"name": "diffusion", "arxiv_id_or_url": "2602.22871"}
                ]
                + _valid_methods()[1:]
            },
            "exactly",
        ),
        (
            _valid_artifact()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"failure_mode": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v389": ""}, "flagged_for_v389"),
        (
            _valid_artifact()
            | {"flagged_for_v389": "cem_gap3_stage2_compositional_arc_energy_v389"},
            "must not auto-select CEM",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4192(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4192: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


@pytest.mark.parametrize(
    ("flag_patch", "message"),
    [
        ({"source_id": "2602.22871"}, "CEM source"),
        ({"operator_authorization_required": False}, "operator authorization"),
        ({"auto_activation_recommended": True}, "auto-activation"),
        ({"retirement_marker": "missing"}, "retirement marker"),
        ({"reason": ""}, "non-empty string"),
    ],
)
def test_validate_artifact_rejects_bad_cem_operator_flag_for_req_report_4192(
    flag_patch: dict[str, object], message: str
) -> None:
    """REQ-REPORT-4192: CEM cannot be silently dropped or auto-activated."""

    artifact = _valid_artifact() | {
        "cem_operator_authorization_flag": _valid_cem_flag() | flag_patch
    }

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(artifact)


def test_validate_artifact_rejects_missing_extra_and_bad_method_rows() -> None:
    """SCENARIO-REPORT-4192: artifact and method fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["inference_substrate"] = "manual_ingestion"
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)

    missing_flag_field = _valid_artifact()
    cem_flag = dict(missing_flag_field["cem_operator_authorization_flag"])
    cem_flag.pop("required_gate")
    missing_flag_field["cem_operator_authorization_flag"] = cem_flag
    with pytest.raises(ValueError, match="CEM flag"):
        mod.validate_artifact(missing_flag_field)


def test_validate_markdown_note_checks_scenario_report_4192_sections() -> None:
    """SCENARIO-REPORT-4192: note maps sources to required axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweeps and WebSearch/WebFetch verified the sources.
    ## SOTA -> experiment mapping
    ## Reward-Guided Stitching DiffusionGemma scale-up
    arXiv:2602.22871. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## S^3 verifier-guided denoising search
    arXiv:2604.06260. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Self-Rewarding SMC particle guidance
    arXiv:2602.01849. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## OpenReview cve4NOiyVp judge-cost tuning
    OpenReview:cve4NOiyVp. arXiv:2501.17178. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## When To Solve/Verify compute-normalized verifier bar
    arXiv:2504.01005. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## ThinkPRM process-verifier comparator
    arXiv:2504.16828. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## CEM operator authorization flag
    arXiv:2510.20607. operator authorization. auto-activation. Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Flagged for .389
    s3_diffusiongemma_verifier_guided_search_scaleup_v389
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_flag() -> None:
    """SCENARIO-REPORT-4192: note must cite each source and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .389"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2602.22871\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("arXiv:2510.20607", "CEM")
        )

    with pytest.raises(ValueError, match="operator authorization"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("operator authorization", "manual review")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4192(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4192: writer emits note, artifact, and studying entry."""

    note_path = tmp_path / "note.md"
    artifact_path = tmp_path / "artifact.json"
    studying_path = tmp_path / "research-studying.md"
    studying_path.write_text("# Research Studying\n\n## Existing\nBody.\n", encoding="utf-8")

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
    assert studying.count("2026-06-14 Exp 4192") == 1
    assert "cem_operator_authorization_flag" in studying
    assert "flagged_for_v389" in studying
    assert "Flagged for .389" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4192() -> None:
    """REQ-REPORT-4192: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(
        studying_once.split("\n## Existing")[0]
    )
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-14 Exp 4192") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-14 Exp 4192")
    assert studying_refreshed.count("2026-06-14 Exp 4192") == 1
    assert marker_at_end.count("2026-06-14 Exp 4192") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "operator authorization is granted and gate-1R is passed."
    )


def test_main_prints_terminal_verdict_for_req_report_4192(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4192: CLI entry point writes default repo-root outputs."""

    calls: dict[str, Path] = {}

    def fake_write_outputs(
        *,
        note_path: Path,
        artifact_path: Path,
        studying_path: Path,
    ) -> dict[str, object]:
        calls["note_path"] = note_path
        calls["artifact_path"] = artifact_path
        calls["studying_path"] = studying_path
        return {"honest_verdict": mod.DEFAULT_HONEST_VERDICT}

    monkeypatch.setattr(mod, "write_outputs", fake_write_outputs)

    assert mod.main() == 0

    captured = capsys.readouterr()
    assert captured.out.strip() == mod.DEFAULT_HONEST_VERDICT
    assert calls["note_path"].as_posix().endswith(NOTE_PATH.as_posix())
    assert calls["artifact_path"].as_posix().endswith(ARTIFACT_PATH.as_posix())
    assert calls["studying_path"].as_posix().endswith(STUDYING_PATH.as_posix())


def test_module_main_guard_exits_zero_for_req_report_4192(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4192: direct module execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_wrapper_script_runs_module_for_req_report_4192(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4192: required results/ wrapper delegates to the module."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_deliverable_files_validate_against_req_report_4192() -> None:
    """REQ-REPORT-4192: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")
    exclusion = EXCLUSION_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v389"] == mod.DEFAULT_FLAGGED_FOR_V389
    assert (
        artifact["cem_operator_authorization_flag"]["operator_authorization_required"]
        is True
    )
    assert (
        artifact["cem_operator_authorization_flag"]["auto_activation_recommended"]
        is False
    )
    assert mod.RETIREMENT_MARKER in exclusion
    assert (
        "2026-06-14 Exp 4192 - .388 planning sweep SOTA ingestion ingested"
        in studying
    )
    assert (
        "Flagged for .389: "
        "`s3_diffusiongemma_verifier_guided_search_scaleup_v389`"
    ) in studying
