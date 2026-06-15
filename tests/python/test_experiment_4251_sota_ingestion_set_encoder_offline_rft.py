"""Tests for REQ-REPORT-4251 / SCENARIO-REPORT-4251."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4251_sota_ingestion_set_encoder_offline_rft as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-set-encoder-offline-rft-v394-2026-06-15.md"
)
ARTIFACT_PATH = Path("results/experiment_4251_sota_ingestion_set_encoder_offline_rft.json")
WRAPPER_PATH = Path("results/experiment_4251_sota_ingestion_set_encoder_offline_rft.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v394=mod.DEFAULT_FLAGGED_FOR_V394,
    )


def test_req_report_4251_spec_anchor_exists() -> None:
    """REQ-REPORT-4251: OpenSpec declares the .394 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4251" in spec
    assert "SCENARIO-REPORT-4251" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v394" in spec
    assert "Exp 4245 reports a clean ARC oracle-distinct Set-Encoder beats-vote win" in spec
    assert "blocked_code_second_corpus_missing" in spec
    assert "harness_smoke_passed=false" in spec
    for source in mod.VERIFIED_SOURCE_URLS.values():
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4251() -> None:
    """REQ-REPORT-4251: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v394": mod.DEFAULT_FLAGGED_FOR_V394,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note "
                "without verifiable citations is treated as fabrication "
                "(adversarial_verify discipline)."
            ),
            "flagged_for_v394": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the next planner, conditioned on the A3/A4/B2 outcomes."
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
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "three to eight"),
        (
            _valid_artifact()
            | {
                "methods_mapped": [
                    {
                        "name": "fake",
                        "arxiv_id_or_url": "9999.99999",
                        "url": "https://arxiv.org/abs/9999.99999",
                        "carnot_stack_mapping": "fake",
                        "a3_arc_mapping": "fake",
                        "a4_code_mapping": "fake",
                        "b2_reward_mapping": "fake",
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
                    _valid_methods()[0] | {"url": "https://example.com/2505.15433"}
                ]
                + _valid_methods()[1:]
            },
            "url",
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
                    _valid_methods()[0] | {"a3_arc_mapping": ""}
                ]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v394": ""}, "flagged_for_v394"),
        (
            _valid_artifact() | {"flagged_for_v394": "raft_reward_weighted_sft_v394"},
            "AggLM",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4251(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4251: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4251: artifact fields are exact."""

    missing_artifact = _valid_artifact()
    missing_artifact.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_artifact(missing_artifact)

    extra_artifact = _valid_artifact()
    extra_artifact["outcomes_mapped"] = {}
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_artifact(extra_artifact)

    malformed_methods = _valid_artifact() | {
        "methods_mapped": ["not-a-dict"] + _valid_methods()[1:]
    }
    with pytest.raises(ValueError, match="exactly"):
        mod.validate_artifact(malformed_methods)


def test_validate_markdown_note_checks_scenario_report_4251_sections() -> None:
    """SCENARIO-REPORT-4251: note maps sources to A3/A4/B2 axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch HTTP 429.
    ## Exp 4245 ARC A3, Exp 4246 code A4, and Exp 4248 offline B2 read
    headline_outcome=arc_oracle_distinct_set_encoder_beats_vote
    set_encoder_minus_vote_delta=0.4423076923 CI95 [0.3076923077, 0.5961538462]
    margin_override_minus_vote=0.4230769231 matched_control_delta=0.4807692308
    oracle_at_k=0.8269230769 held_out_task_n=52 oracle_distinct_beats_vote=true.
    blocked_code_second_corpus_missing no distinct viable hidden-label candidate pool.
    blocked_gate_check_failed harness_smoke_passed=false steps_run=0 trainable_param_count=0.
    ## SOTA -> experiment mapping
    arXiv:2505.15433 arXiv:2509.06870 arXiv:2605.26172 arXiv:2510.14913
    arXiv:2504.11343 arXiv:2502.11026 arXiv:2506.10947 arXiv:2512.15146.
    Set-LLM AggLM ARBITER budget-aware discriminative verification RAFT VAR
    Spurious Rewards SCOPE synthesizes a corrected grid
    SCOPE per-region evidence bigger pool same-base random-label.
    Carnot stack mapping. A3 ARC mapping. A4 code mapping. B2 reward mapping.
    Failure mode. Experiment mapping.
    ## Flagged for .394
    agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_conditioning() -> None:
    """SCENARIO-REPORT-4251: note must cite sources and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .394"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2505.15433\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(mod.NOTE_MARKDOWN.replace("arXiv:2505.15433", "Set-LLM"))

    with pytest.raises(ValueError, match="set_encoder_minus_vote_delta=0.4423076923"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("set_encoder_minus_vote_delta=0.4423076923", "pending")
        )

    with pytest.raises(ValueError, match="blocked_code_second_corpus_missing"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("blocked_code_second_corpus_missing", "replicated")
        )

    with pytest.raises(ValueError, match="harness_smoke_passed=false"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("harness_smoke_passed=false", "harness_smoke_passed=true")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4251(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4251: writer emits note, artifact, and studying entry."""

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
    assert studying.count("2026-06-15 Exp 4251") == 1
    assert "flagged_for_v394" in studying
    assert "Flagged for .394" in studying
    assert "arc_oracle_distinct_set_encoder_beats_vote" in studying
    assert "blocked_code_second_corpus_missing" in studying
    assert "harness_smoke_passed=false" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4251() -> None:
    """REQ-REPORT-4251: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-15 Exp 4251") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-15 Exp 4251")
    assert studying_refreshed.count("2026-06-15 Exp 4251") == 1
    assert marker_at_end.count("2026-06-15 Exp 4251") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "treat reward-weighted SFT as an owed gate after the harness proves real training."
    )


def test_main_prints_terminal_verdict_for_req_report_4251(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4251: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4251(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4251: direct module execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_wrapper_script_runs_module_for_req_report_4251(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4251: required results/ wrapper delegates to the module."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_deliverable_files_validate_against_req_report_4251() -> None:
    """REQ-REPORT-4251: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v394"] == mod.DEFAULT_FLAGGED_FOR_V394
    assert "2026-06-15 Exp 4251 - .393 planning sweep SOTA ingestion ingested" in studying
    assert (
        "Flagged for .394: `agglm_synthesize_corrected_grid_from_set_encoder_evidence_v394`"
        in studying
    )
