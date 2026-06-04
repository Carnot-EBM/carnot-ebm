"""Tests for the Exp 3789 CLI abstention batch surface.

Spec: REQ-SPOE-3789, SCENARIO-SPOE-3789.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from carnot import cli
from carnot.pipeline import certified_abstention_surface as abstention
from carnot.pipeline import second_pair_detector as spd


def _domain_examples(domain: str = "math", *, n: int = 80) -> list[spd.LabeledDetectorExample]:
    examples: list[spd.LabeledDetectorExample] = []
    for idx in range(n):
        label = 1 if idx < n // 2 else 0
        ensemble = 0.95 - 0.004 * idx if label else 0.05 + 0.001 * (idx - n // 2)
        confidence_error = 0.82 - 0.003 * idx if label else 0.18 + 0.001 * (idx - n // 2)
        examples.append(
            spd.LabeledDetectorExample(
                domain=domain,
                label=label,
                ensemble_energy=ensemble,
                confidence_error=confidence_error,
                example_id=f"{domain}-3789-{idx}",
            )
        )
    return examples


def _batch_candidates() -> list[dict[str, object]]:
    return [
        {
            "candidate_id": "confident-error",
            "domain": "math",
            "text": "We compute 8 + 5 = 14.",
            "confidence_error": 1.0,
            "ensemble_energy": 1.0,
        },
        {
            "candidate_id": "uncertain-midpoint",
            "domain": "math",
            "text": "We compute 8 + 5 = 13.",
            "confidence_error": 0.5,
            "ensemble_energy": 0.5,
        },
    ]


def test_req_spoe_3789_load_candidate_batch_accepts_json_array_and_line_rows(
    tmp_path: Path,
) -> None:
    """REQ-SPOE-3789: batch files accept JSON arrays and one-row-per-line input."""

    json_array = tmp_path / "candidates.json"
    json_array.write_text(json.dumps(_batch_candidates()), encoding="utf-8")
    assert cli._load_candidate_batch(candidates_file=str(json_array)) == _batch_candidates()

    line_file = tmp_path / "candidates.jsonl"
    line_file.write_text(
        json.dumps(_batch_candidates()[0]) + "\n" + "raw candidate text\n",
        encoding="utf-8",
    )
    loaded = cli._load_candidate_batch(candidates_file=str(line_file))

    assert loaded[0]["candidate_id"] == "confident-error"
    assert loaded[1] == {"candidate_id": "line-2", "text": "raw candidate text"}


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ("{}", "must be a JSON list"),
        ("[]", "candidate batch is empty"),
        ("[1]", "candidate at index 0"),
    ],
)
def test_req_spoe_3789_load_candidate_batch_rejects_invalid_payloads(
    payload: str,
    match: str,
) -> None:
    """REQ-SPOE-3789: malformed batch payloads fail closed."""

    with pytest.raises(ValueError, match=match):
        cli._load_candidate_batch(candidates_json=payload)


def test_scenario_spoe_3789_verify_batch_cli_default_off_and_abstention_on(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SPOE-3789: CLI batch mode preserves default-off and opt-in verdicts."""

    monkeypatch.setattr(
        spd,
        "load_cached_labeled_examples",
        lambda _root, **_kwargs: (
            _domain_examples("math") + _domain_examples("code"),
            {"math": {"status": "synthetic"}, "code": {"status": "synthetic"}},
        ),
    )
    candidates_file = tmp_path / "batch.json"
    candidates_file.write_text(json.dumps(_batch_candidates()), encoding="utf-8")

    default_args = argparse.Namespace(
        candidates_file=str(candidates_file),
        domain="math",
        abstention_mode=False,
        abstention_threshold=None,
    )
    assert cli.cmd_verify_batch(default_args) == 0
    default_payload = json.loads(capsys.readouterr().out)

    assert default_payload["cli_surface"] == "verify-batch"
    assert default_payload["batch"]["n_candidates"] == 2
    assert default_payload["batch"]["abstention_mode_enabled"] is False
    assert all("abstention_verdict" not in row for row in default_payload["scores"])

    enabled_args = argparse.Namespace(
        candidates_file=str(candidates_file),
        domain="math",
        abstention_mode=True,
        abstention_threshold=None,
    )
    assert cli.cmd_verify_batch(enabled_args) == 0
    enabled_payload = json.loads(capsys.readouterr().out)
    rows = {row["candidate_id"]: row for row in enabled_payload["scores"]}

    assert enabled_payload["batch"]["n_candidates"] == 2
    assert enabled_payload["batch"]["abstention_mode_enabled"] is True
    assert rows["confident-error"]["abstention_verdict"] == abstention.CONFIDENT_ERROR_VERDICT
    assert rows["confident-error"]["route_to_review"] is False
    assert rows["uncertain-midpoint"]["abstention_verdict"] == abstention.ABSTAIN_VERDICT
    assert rows["uncertain-midpoint"]["route_to_review"] is True
    assert rows["uncertain-midpoint"]["certified_abstention"]["delta"] == pytest.approx(0.05)
    assert rows["uncertain-midpoint"]["certified_abstention"]["threshold_source"].endswith(
        "results/experiment_3771_certified_abstention_operating_point.json"
    )


def test_scenario_spoe_3789_main_routes_verify_batch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-SPOE-3789: packaged CLI dispatch reaches the batch surface."""

    monkeypatch.setattr(
        spd,
        "load_cached_labeled_examples",
        lambda _root, **_kwargs: (
            _domain_examples("math") + _domain_examples("code"),
            {"math": {"status": "synthetic"}, "code": {"status": "synthetic"}},
        ),
    )
    candidates_file = tmp_path / "batch.jsonl"
    candidates_file.write_text(
        "\n".join(json.dumps(candidate) for candidate in _batch_candidates()),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "carnot",
            "verify-batch",
            "--candidates-file",
            str(candidates_file),
            "--domain",
            "math",
            "--abstention-mode",
        ],
    )

    assert cli.main() == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cli_surface"] == "verify-batch"
    assert payload["batch"]["n_candidates"] == 2
    assert payload["scores"][1]["abstention_verdict"] == abstention.ABSTAIN_VERDICT


def test_req_spoe_3789_verify_batch_reports_input_errors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-SPOE-3789: missing or malformed batch files return CLI errors."""

    args = argparse.Namespace(
        candidates_file=str(tmp_path / "missing.json"),
        domain="math",
        abstention_mode=False,
        abstention_threshold=None,
    )

    assert cli.cmd_verify_batch(args) == 1
    assert "Error reading candidates" in capsys.readouterr().err
