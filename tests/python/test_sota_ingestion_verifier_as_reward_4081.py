"""Tests for REQ-REPORT-4081 / SCENARIO-REPORT-4081."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from carnot import sota_ingestion_verifier_as_reward_4081 as mod


SPEC_PATH = Path("openspec/capabilities/research-reporting/spec.md")
NOTE_PATH = Path(
    "docs/research-notes/sota-ingestion-verifier-as-reward-2026-06-11.md"
)
RECEIPT_PATH = Path(
    "results/experiment_4081_sota_ingestion_verifier_as_reward_receipt.json"
)
STUDYING_PATH = Path("research-studying.md")


def _valid_methods() -> list[dict[str, str]]:
    return [
        {
            "arxiv_id": "2411.15124",
            "one_line": "Tulu 3 supplies the open SFT/DPO/RLVR recipe to adapt.",
        },
        {
            "arxiv_id": "2507.14843",
            "one_line": "Invisible Leash gates latent support before RLVR/RFT spend.",
        },
        {
            "arxiv_id": "2505.14216",
            "one_line": "RL-vs-distillation separates accuracy lift from new capability.",
        },
        {
            "arxiv_id": "2604.03128",
            "one_line": "Self-Distilled RLVR keeps verifier feedback as update direction.",
        },
        {
            "arxiv_id": "2203.14465",
            "one_line": "STaR is the minimal verifier-certified rationale loop.",
        },
        {
            "arxiv_id": "2308.08998",
            "one_line": "ReST gives the offline generate-filter-improve cadence.",
        },
        {
            "arxiv_id": "2601.17223",
            "one_line": "VPRM turns rule verifiers into dense process rewards.",
        },
        {
            "arxiv_id": "2605.10325",
            "one_line": "VPR extends verifiable process rewards to agentic trajectories.",
        },
    ]


def _valid_receipt() -> dict[str, object]:
    return mod.build_receipt(
        methods_mapped=_valid_methods(),
        strongest_for_next_roadmap=[
            "latent_vs_absent_precision_gate_before_rft",
            "process_reward_weighted_sft_over_trace_certification",
        ],
    )


def test_req_report_4081_spec_anchor_exists() -> None:
    """REQ-REPORT-4081: OpenSpec declares the verifier-as-reward ingestion."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-REPORT-4081" in spec
    assert "SCENARIO-REPORT-4081" in spec
    assert NOTE_PATH.as_posix() in spec
    assert RECEIPT_PATH.as_posix() in spec
    assert "methods_mapped" in spec


def test_build_receipt_has_required_schema_fields_for_req_report_4081() -> None:
    """REQ-REPORT-4081: receipt exposes only the required artifact fields."""

    receipt = _valid_receipt()

    assert receipt == {
        "honest_verdict": "complete: sota_ingestion_verifier_as_reward_mapped",
        "methods_mapped": _valid_methods(),
        "strongest_for_next_roadmap": [
            "latent_vs_absent_precision_gate_before_rft",
            "process_reward_weighted_sft_over_trace_certification",
        ],
        "inference_substrate": "aggregation_from_upstream_artifacts",
    }


@pytest.mark.parametrize(
    ("bad_receipt", "message"),
    [
        (_valid_receipt() | {"honest_verdict": "draft"}, "terminal prefix"),
        (_valid_receipt() | {"methods_mapped": []}, "non-empty list"),
        (
            _valid_receipt() | {"methods_mapped": _valid_methods()[:4]},
            "five to eight",
        ),
        (
            _valid_receipt()
            | {
                "methods_mapped": [
                    {"arxiv_id": "9999.99999", "one_line": "fake"}
                ]
                + _valid_methods()[1:]
            },
            "verified arxiv",
        ),
        (
            _valid_receipt()
            | {
                "methods_mapped": [
                    {"arxiv_id": "2411.15124", "title": "Tulu"}
                ]
                + _valid_methods()[1:]
            },
            "exactly arxiv_id and one_line",
        ),
        (
            _valid_receipt()
            | {"methods_mapped": [_valid_methods()[0]] + _valid_methods()[:-1]},
            "duplicate method",
        ),
        (
            _valid_receipt()
            | {
                "methods_mapped": [
                    _valid_methods()[0] | {"one_line": ""}
                ]
                + _valid_methods()[1:]
            },
            "one_line",
        ),
        (
            _valid_receipt()
            | {"strongest_for_next_roadmap": []},
            "strongest_for_next_roadmap",
        ),
        (
            _valid_receipt() | {"inference_substrate": "manual_guess"},
            "aggregation_from_upstream_artifacts",
        ),
    ],
)
def test_validate_receipt_rejects_schema_violations_for_scenario_report_4081(
    bad_receipt: dict[str, object], message: str
) -> None:
    """SCENARIO-REPORT-4081: invalid receipts fail closed."""

    with pytest.raises(ValueError, match=message):
        mod.validate_receipt(bad_receipt)


def test_validate_receipt_rejects_missing_and_extra_fields() -> None:
    """SCENARIO-REPORT-4081: receipt fields are bare and exact."""

    missing_receipt = _valid_receipt()
    missing_receipt.pop("methods_mapped")
    with pytest.raises(ValueError, match="missing required fields"):
        mod.validate_receipt(missing_receipt)

    extra_receipt = _valid_receipt()
    extra_receipt["methods_mapped_count"] = 8
    with pytest.raises(ValueError, match="unexpected fields"):
        mod.validate_receipt(extra_receipt)


def test_validate_markdown_note_checks_scenario_report_4081_sections() -> None:
    """SCENARIO-REPORT-4081: note maps methods to pipeline changes and pitfalls."""

    note = """
    # SOTA ingestion verifier as reward

    ## Verifier-certified RFT over the current RFT pipeline
    Tulu 3 arXiv:2411.15124 and STaR arXiv:2203.14465.
    Implementation over current RFT pipeline: build matched corpora.
    Pitfalls / where it fails: poisoned labels collapse the train.

    ## RLVR / Tulu 3 open post-training recipe
    Invisible Leash arXiv:2507.14843.
    Implementation over current RFT pipeline: add latent-vs-absent gate.
    Pitfalls / where it fails: no support means no RLVR escape.

    ## Invisible Leash latent-vs-absent diagnostic
    RL-vs-distillation arXiv:2505.14216 separates capability ceilings.
    Implementation over current RFT pipeline: check pass@k support before training.
    Pitfalls / where it fails: pass@1 lift can hide a flat ceiling.

    ## Process-reward distillation
    Self-Distilled RLVR arXiv:2604.03128, VPRM arXiv:2601.17223 and
    VPR arXiv:2605.10325.
    Implementation over current RFT pipeline: use dense verifier scores.
    Pitfalls / where it fails: local validity may not imply final correctness.

    ## RFT / STaR / ReST self-training
    ReST arXiv:2308.08998 and RL-vs-distillation arXiv:2505.14216.
    Implementation over current RFT pipeline: offline generate-filter-improve.
    Pitfalls / where it fails: distillation without new support only sharpens easy cases.

    ## Bottom line for the .378 roadmap
    Prioritize latent-vs-absent precision gate and process-reward weighted SFT.
    """

    mod.validate_markdown_note(note)


def test_validate_markdown_note_rejects_missing_bottom_line() -> None:
    """SCENARIO-REPORT-4081: note must close the discover-to-plan loop."""

    with pytest.raises(ValueError, match="Bottom line"):
        mod.validate_markdown_note(
            "## Verifier-certified RFT over the current RFT pipeline\n"
            "Implementation over current RFT pipeline\n"
            "Pitfalls / where it fails\n"
        )


def test_validate_markdown_note_rejects_missing_verified_citations() -> None:
    """SCENARIO-REPORT-4081: every mapped method claim cites a verified ID."""

    note = """
    ## Verifier-certified RFT over the current RFT pipeline
    arXiv:2411.15124 arXiv:2203.14465.
    Implementation over current RFT pipeline.
    Pitfalls / where it fails.
    ## RLVR / Tulu 3 open post-training recipe
    arXiv:2507.14843.
    Implementation over current RFT pipeline.
    Pitfalls / where it fails.
    ## Invisible Leash latent-vs-absent diagnostic
    arXiv:2505.14216.
    Implementation over current RFT pipeline.
    Pitfalls / where it fails.
    ## Process-reward distillation
    arXiv:2601.17223 arXiv:2605.10325.
    Implementation over current RFT pipeline.
    Pitfalls / where it fails.
    ## RFT / STaR / ReST self-training
    arXiv:2308.08998.
    Implementation over current RFT pipeline.
    Pitfalls / where it fails.
    ## Bottom line for the .378 roadmap
    Prioritize the precision gate.
    """

    with pytest.raises(ValueError, match="verified arxiv citations"):
        mod.validate_markdown_note(note)


def test_deliverable_files_validate_against_req_report_4081() -> None:
    """REQ-REPORT-4081: committed note and JSON receipt satisfy the contract."""

    receipt = json.loads(RECEIPT_PATH.read_text(encoding="utf-8"))
    note = NOTE_PATH.read_text(encoding="utf-8")
    studying = STUDYING_PATH.read_text(encoding="utf-8")

    mod.validate_receipt(receipt)
    mod.validate_markdown_note(note)
    assert len(receipt["methods_mapped"]) >= 5
    assert (
        "2026-06-11 Exp 4081 - .377 verifier-as-reward SOTA ingestion ingested"
        in studying
    )
    assert "Bottom line for the .378 roadmap" in studying
