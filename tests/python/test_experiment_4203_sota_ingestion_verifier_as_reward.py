"""Tests for REQ-REPORT-4203 / SCENARIO-REPORT-4203."""

from __future__ import annotations

import json
import runpy
from pathlib import Path

import pytest

from carnot import experiment_4203_sota_ingestion_verifier_as_reward as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path("docs/research-notes/sota-ingestion-verifier-as-reward-v390-2026-06-14.md")
ARTIFACT_PATH = Path("results/experiment_4203_sota_ingestion_verifier_as_reward.json")
WRAPPER_PATH = Path("results/experiment_4203_sota_ingestion_verifier_as_reward.py")
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [dict(method) for method in mod.DEFAULT_METHODS_MAPPED]


def _valid_artifact() -> dict[str, object]:
    return mod.build_artifact(
        methods_mapped=_valid_methods(),
        flagged_for_v390=mod.DEFAULT_FLAGGED_FOR_V390,
    )


def test_req_report_4203_spec_anchor_exists() -> None:
    """REQ-REPORT-4203: OpenSpec declares the .390 ingestion artifact."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4203" in spec
    assert "SCENARIO-REPORT-4203" in spec
    assert NOTE_PATH.as_posix() in spec
    assert ARTIFACT_PATH.as_posix() in spec
    assert WRAPPER_PATH.as_posix() in spec
    assert "flagged_for_v390" in spec
    assert "non-Qwen" in spec
    assert "same-generator random-label A-vs-B replication" in spec
    for source in mod.VERIFIED_SOURCE_URLS:
        assert source in spec


def test_build_artifact_has_required_fields_for_req_report_4203() -> None:
    """REQ-REPORT-4203: artifact exposes required principle fields."""

    artifact = _valid_artifact()

    assert artifact == {
        "honest_verdict": mod.DEFAULT_HONEST_VERDICT,
        "methods_mapped": _valid_methods(),
        "flagged_for_v390": mod.DEFAULT_FLAGGED_FOR_V390,
        "field_principles": {
            "honest_verdict": (
                "Terminal-prefixed. Records ingestion completed with verifiable citations."
            ),
            "methods_mapped": (
                "Each method MUST carry a real arXiv ID/URL; an ingestion note "
                "without verifiable citations is treated as fabrication "
                "(adversarial_verify discipline)."
            ),
            "flagged_for_v390": (
                "Closes discover->ingest->plan: names the strongest method for "
                "the next planner (e.g. the non-Qwen replication or the "
                "math-process-reward fork)."
            ),
        },
    }


@pytest.mark.parametrize(
    ("bad_artifact", "message"),
    [
        (_valid_artifact() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_artifact() | {"field_principles": {"honest_verdict": "loose"}}, "field_principles"),
        (_valid_artifact() | {"methods_mapped": _valid_methods()[:2]}, "five to eight"),
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
                "methods_mapped": [_valid_methods()[0] | {"url": "https://example.com/2506.10947"}]
                + _valid_methods()[1:]
            },
            "url",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [{"name": "confound", "arxiv_id_or_url": "2506.10947"}]
                + _valid_methods()[1:]
            },
            "exactly",
        ),
        (
            _valid_artifact() | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate source",
        ),
        (
            _valid_artifact()
            | {
                "methods_mapped": [_valid_methods()[0] | {"failure_mode": ""}]
                + _valid_methods()[1:]
            },
            "non-empty string",
        ),
        (_valid_artifact() | {"flagged_for_v390": ""}, "flagged_for_v390"),
        (
            _valid_artifact() | {"flagged_for_v390": "math_process_reward_fork_v390"},
            "non-Qwen",
        ),
    ],
)
def test_validate_artifact_rejects_schema_violations_for_scenario_report_4203(
    bad_artifact: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4203: invalid mapping artifacts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_artifact(bad_artifact)


def test_validate_artifact_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4203: artifact fields are exact."""

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


def test_validate_markdown_note_checks_scenario_report_4203_sections() -> None:
    """SCENARIO-REPORT-4203: note maps sources to required axes."""

    note = """
    ## Fresh-pass provenance
    reliable-channel sweep_clusters.py sweep_semscholar.py WebSearch/WebFetch.
    ## Exp 4199 A-vs-B status
    blocked_gate_check_failed training_launched false.
    ## SOTA -> experiment mapping
    ## CONFOUND hardening
    arXiv:2506.10947 arXiv:2601.11061 arXiv:2601.04411 arXiv:2509.20837.
    non-Qwen base same-generator random-label Youden-J memorization-shortcut diagnostic.
    Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Code-RLVR baselines
    arXiv:2410.02089 arXiv:2601.12186 arXiv:2602.17684.
    Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Process/self-distill fork
    arXiv:2604.03128 arXiv:2605.19436.
    Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Cost-crossover hybrid
    arXiv:2510.14913 arXiv:2504.01005 arXiv:2504.16828.
    Carnot stack mapping. Implication. Failure mode. Experiment mapping.
    ## Flagged for .390
    non_qwen_same_generator_random_label_ablation_v390
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_sources_or_flag() -> None:
    """SCENARIO-REPORT-4203: note must cite sources and close the loop."""

    with pytest.raises(ValueError, match="Flagged for .390"):
        mod.validate_markdown_note("## Fresh-pass provenance\narXiv:2506.10947\n")

    with pytest.raises(ValueError, match="verified source citations"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("arXiv:2506.10947", "Spurious Rewards")
        )

    with pytest.raises(ValueError, match="blocked_gate_check_failed"):
        mod.validate_markdown_note(
            mod.NOTE_MARKDOWN.replace("blocked_gate_check_failed", "pending")
        )


def test_write_outputs_updates_files_idempotently_for_req_report_4203(
    tmp_path: Path,
) -> None:
    """REQ-REPORT-4203: writer emits note, artifact, and studying entry."""

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
    assert studying.count("2026-06-14 Exp 4203") == 1
    assert "flagged_for_v390" in studying
    assert "Flagged for .390" in studying
    assert "blocked_gate_check_failed" in studying


def test_section_updates_handle_heading_layouts_for_req_report_4203() -> None:
    """REQ-REPORT-4203: markdown updates work before or between sections."""

    without_marker = "# Doc\n\n## Existing\nBody.\n"
    studying_once = mod._with_studying_section(without_marker)
    starts_with_heading = mod._with_studying_section("## Existing\nBody.\n")
    studying_refreshed = mod._with_studying_section(studying_once)
    marker_at_end = mod._with_studying_section(studying_once.split("\n## Existing")[0])
    no_heading = mod._with_studying_section("# Doc\nOnly body.\n")

    assert studying_once.index("2026-06-14 Exp 4203") < studying_once.index("## Existing")
    assert starts_with_heading.startswith("## 2026-06-14 Exp 4203")
    assert studying_refreshed.count("2026-06-14 Exp 4203") == 1
    assert marker_at_end.count("2026-06-14 Exp 4203") == 1
    assert "## Existing\nBody." in studying_refreshed
    assert no_heading.rstrip().endswith(
        "random-label A-vs-B replication before any math-process-reward fork."
    )


def test_main_prints_terminal_verdict_for_req_report_4203(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-REPORT-4203: CLI entry point writes default repo-root outputs."""

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


def test_module_main_guard_exits_zero_for_req_report_4203(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4203: direct module execution exits after writing outputs."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(Path(mod.__file__)), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_wrapper_script_runs_module_for_req_report_4203(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-REPORT-4203: required results/ wrapper delegates to the module."""

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(WRAPPER_PATH), run_name="__main__")

    assert exc_info.value.code == 0
    assert capsys.readouterr().out.strip() == mod.DEFAULT_HONEST_VERDICT


def test_deliverable_files_validate_against_req_report_4203() -> None:
    """REQ-REPORT-4203: committed note and JSON artifact satisfy the contract."""

    artifact = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_artifact(artifact)
    mod.validate_markdown_note(note)
    assert len(artifact["methods_mapped"]) >= 3
    assert artifact["flagged_for_v390"] == mod.DEFAULT_FLAGGED_FOR_V390
    assert "2026-06-14 Exp 4203 - .389 planning sweep SOTA ingestion ingested" in studying
    assert "Flagged for .390: `non_qwen_same_generator_random_label_ablation_v390`" in studying
