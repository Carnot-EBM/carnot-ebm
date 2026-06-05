"""Tests for Exp 3858 balanced step-error corpus v2.

Spec: REQ-DATA-3858,
      SCENARIO-DATA-3858,
      SCENARIO-DATA-3858-FALLBACK.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

from carnot.data import step_error_balanced_v2 as mod


AXES = [
    "circular",
    "confidence",
    "counterfactual",
    "deception",
    "domain_inconsistency",
    "missing_condition",
    "multi_solutions",
    "redundency",
    "step_contradiction",
]


def _prmbench_record(idx: int, axis: str) -> dict[str, Any]:
    return {
        "idx": f"row-{idx}",
        "question": f"Question {idx}",
        "modified_process": [
            f"row {idx} step 1 ok",
            f"row {idx} step 2 bad",
            f"row {idx} step 3 ok",
            f"row {idx} step 4 bad",
        ],
        "error_steps": [2, 4],
        "classification": axis,
    }


def _config(tmp_path: Path, *, target_n: int = 18, min_incorrect: int = 8) -> mod.BuildConfig:
    return mod.BuildConfig(
        repo_root=tmp_path,
        output_path=tmp_path / "data" / "step_error_balanced_v2.json",
        random_seed=123,
        target_n=target_n,
        min_incorrect_steps=min_incorrect,
    )


def test_scenario_data_3858_prmbench_primary_balances_and_preserves_axes(
    tmp_path: Path,
) -> None:
    """SCENARIO-DATA-3858: PRMBench rows map to a balanced scoreable artifact."""

    records = [_prmbench_record(idx, AXES[idx % len(AXES)]) for idx in range(18)]
    artifact = mod.build_corpus_artifact(
        _config(tmp_path),
        availability=mod.SourceAvailability(
            prmbench_reachable=True,
            fover_v3_paths=[],
            preconditions_checked=[
                {"resource": "prmbench_hf", "available": True, "detail": "fixture"},
            ],
        ),
        prmbench_records=records,
        schema_validator=lambda item: True,
        started_s=1.0,
        now_s=3.25,
    )

    assert artifact["primary_source"] == "prmbench"
    assert artifact["n_items"] == 18
    assert artifact["n_incorrect_steps"] == 9
    assert isinstance(artifact["n_incorrect_steps"], int)
    assert artifact["incorrect_fraction"] == 0.5
    assert artifact["schema_compatible_with_2837"] is True
    assert isinstance(artifact["schema_compatible_with_2837"], bool)
    assert artifact["error_axis_coverage"] == sorted(AXES)
    assert "error_steps" in artifact["label_mapping_note"]
    assert artifact["duration_s"] == 2.25
    assert artifact["honest_verdict"].startswith(
        "complete: balanced_step_error_corpus_v2_n18_nincorrect9_sourceprmbench_"
    )
    assert set(mod.REQUIRED_FIELD_PRINCIPLES) <= set(artifact["field_principles"])

    labels = {item["label"] for item in artifact["items"]}
    assert labels == {"correct", "incorrect"}
    assert all(item["error_axis"] in AXES for item in artifact["items"])
    assert all({"question_id", "question", "step_text", "label"} <= set(item) for item in artifact["items"])


def test_req_data_3858_fover_fallback_dedupes_and_does_not_pad(tmp_path: Path) -> None:
    """REQ-DATA-3858: FoVer fallback reports the true largest balanced set."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    rows = [
        {"question_id": "i1", "step_text": "bad one", "label": "incorrect"},
        {"question_id": "i1", "step_text": "bad one", "label": "incorrect"},
        {"question_id": "i2", "step_text": "bad two", "label": "incorrect"},
        {"question_id": "i3", "step_text": "bad three", "label": "incorrect"},
        {"question_id": "c1", "step_text": "ok one", "label": "correct"},
        {"question_id": "c2", "step_text": "ok two", "label": "correct"},
    ]
    (data_dir / "fover_corpus_v3.json").write_text(json.dumps(rows), encoding="utf-8")

    artifact = mod.build_corpus_artifact(
        _config(tmp_path, target_n=10, min_incorrect=4),
        availability=mod.SourceAvailability(
            prmbench_reachable=False,
            fover_v3_paths=[data_dir / "fover_corpus_v3.json"],
            preconditions_checked=[
                {"resource": "prmbench_hf", "available": False, "detail": "fixture"},
                {"resource": "fover_v3_fallback", "available": True, "detail": "fixture"},
            ],
        ),
        schema_validator=lambda item: True,
        started_s=0.0,
        now_s=1.0,
    )

    assert artifact["primary_source"] == "fover_v3_fallback"
    assert artifact["n_items"] == 4
    assert artifact["n_incorrect_steps"] == 2
    assert artifact["incorrect_fraction"] == 0.5
    assert artifact["error_axis_coverage"] == []
    assert all(item["error_axis"] is None for item in artifact["items"])
    assert artifact["honest_verdict"] == (
        "complete: balanced_corpus_v2_fover_fallback_n4_nincorrect2_"
        "below_target_scissor_will_widen_or_inconclusive"
    )

    target_met = mod.build_corpus_artifact(
        _config(tmp_path, target_n=4, min_incorrect=2),
        availability=mod.SourceAvailability(
            prmbench_reachable=False,
            fover_v3_paths=[data_dir / "fover_corpus_v3.json"],
            preconditions_checked=[],
        ),
        schema_validator=lambda item: True,
        started_s=0.0,
        now_s=1.0,
    )
    assert target_met["honest_verdict"] == (
        "complete: balanced_step_error_corpus_v2_n4_nincorrect2_"
        "sourcefover_v3_fallback_9axisfalse_schema_ok"
    )


def test_req_data_3858_blocks_when_no_source_available(tmp_path: Path) -> None:
    """REQ-DATA-3858: no PRMBench and no FoVer source blocks honestly."""

    artifact = mod.build_corpus_artifact(
        _config(tmp_path),
        availability=mod.SourceAvailability(
            prmbench_reachable=False,
            fover_v3_paths=[],
            preconditions_checked=[
                {"resource": "prmbench_hf", "available": False, "detail": "fixture"},
                {"resource": "fover_v3_fallback", "available": False, "detail": "fixture"},
            ],
        ),
        schema_validator=lambda item: True,
        started_s=4.0,
        now_s=4.5,
    )

    assert artifact["honest_verdict"] == "blocked_no_step_error_source"
    assert artifact["items"] == []
    assert artifact["n_items"] == 0
    assert artifact["n_incorrect_steps"] == 0
    assert artifact["schema_compatible_with_2837"] is False
    assert artifact["preconditions_checked"][0]["resource"] == "prmbench_hf"


def test_req_data_3858_schema_validator_uses_exp2837_hooks() -> None:
    """REQ-DATA-3858: scoreability validation uses exp2837 label and scorer hooks."""

    calls: list[str] = []

    def label_to_int(label: Any) -> int:
        calls.append(f"label:{label}")
        return 1

    def scorer(texts: list[str]) -> dict[str, list[float]]:
        calls.append(f"score:{texts[0]}")
        return {"tier0r_curry_howard": [0.75]}

    assert (
        mod.validate_schema_compatible_with_2837(
            {"question_id": "q1", "step_text": "bad arithmetic", "label": "incorrect"},
            label_to_int=label_to_int,
            score_text_verifiers=scorer,
        )
        is True
    )
    assert calls == ["label:incorrect", "score:bad arithmetic"]


def test_req_data_3858_schema_validator_rejects_unscoreable_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DATA-3858: schema validation rejects malformed or unscorable rows."""

    assert mod.validate_schema_compatible_with_2837({"step_text": "x", "label": "correct"}) is False
    assert (
        mod.validate_schema_compatible_with_2837(
            {"question_id": "q", "step_text": "", "label": "correct"}
        )
        is False
    )
    assert (
        mod.validate_schema_compatible_with_2837(
            {"question_id": "q", "step_text": "x", "label": "maybe"}
        )
        is False
    )
    assert (
        mod.validate_schema_compatible_with_2837(
            {"question_id": "q", "step_text": "x", "label": "correct"},
            label_to_int=lambda label: (_ for _ in ()).throw(ValueError(label)),
            score_text_verifiers=lambda texts: {"score": [1.0]},
        )
        is False
    )

    fake_exp2837 = ModuleType("carnot.eval.fover_memory_leakage_v3")
    fake_exp2837._label_to_int = lambda label: 0
    fake_exp2837._score_text_verifiers = lambda texts: {"score": [0.25]}
    monkeypatch.setitem(sys.modules, "carnot.eval.fover_memory_leakage_v3", fake_exp2837)
    assert (
        mod.validate_schema_compatible_with_2837(
            {"question_id": "q", "step_text": "score me", "label": "correct"}
        )
        is True
    )


def test_req_data_3858_preconditions_and_hf_jsonl_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-DATA-3858: preconditions use curl and HF JSONL loading is deterministic."""

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "fover_test_v3.json").write_text("[]", encoding="utf-8")

    def failed_curl(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert args[:3] == ["curl", "-sf", "-o"]
        assert kwargs["timeout"] == 20
        return subprocess.CompletedProcess(args, 22, "", "")

    availability = mod.check_source_availability(_config(tmp_path), command_runner=failed_curl)
    assert availability.prmbench_reachable is False
    assert availability.fover_v3_paths == [data_dir / "fover_test_v3.json"]
    assert availability.preconditions_checked[0]["detail"] == "curl_exit_22"

    jsonl_path = tmp_path / "prmbench.jsonl"
    jsonl_path.write_text(json.dumps({"idx": "a"}) + "\n\n", encoding="utf-8")
    fake_hub = ModuleType("huggingface_hub")
    fake_hub.hf_hub_download = lambda **kwargs: str(jsonl_path)
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hub)
    assert mod.load_prmbench_records(_config(tmp_path)) == [{"idx": "a"}]


def test_req_data_3858_normalization_and_parsing_edge_cases(tmp_path: Path) -> None:
    """REQ-DATA-3858: malformed source rows are skipped instead of fabricated."""

    assert mod.normalize_label(True) == "correct"
    assert mod.normalize_label(False) == "incorrect"
    with pytest.raises(ValueError, match="unsupported step-error label"):
        mod.normalize_label("unknown")

    parsed = mod.parse_prmbench_records(
        [
            {"modified_process": "not-list"},
            {
                "idx": "edge",
                "modified_process": ["", "ok", "bad"],
                "error_steps": ["nope", 3],
                "modified_question": "Fallback question",
                "classification": "",
            },
            {
                "modified_process": ["plain ok"],
                "error_steps": "not-list",
                "original_question": "Original question",
            },
        ]
    )
    assert [item["label"] for item in parsed] == ["correct", "incorrect", "correct"]
    assert parsed[0]["question"] == "Fallback question"
    assert parsed[-1]["question"] == "Original question"

    assert mod.select_balanced_items([{"label": "correct"}], _config(tmp_path)) == []

    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    path = data_dir / "fover_corpus_v3.json"
    path.write_text(
        json.dumps(
            [
                {"question_id": "empty", "step_text": "", "label": "correct"},
                {"question_id": "bad-label", "step_text": "x", "label": "unknown"},
            ]
        ),
        encoding="utf-8",
    )
    assert mod.load_fover_fallback_items([path]) == []


def test_req_data_3858_schema_blocked_and_module_main(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """REQ-DATA-3858: schema validation can block and the module CLI returns zero."""

    artifact = mod.build_corpus_artifact(
        _config(tmp_path, target_n=2, min_incorrect=1),
        availability=mod.SourceAvailability(
            prmbench_reachable=True,
            fover_v3_paths=[],
            preconditions_checked=[],
        ),
        prmbench_records=[_prmbench_record(1, "confidence")],
        schema_validator=lambda item: False,
        started_s=0.0,
        now_s=1.0,
    )
    assert artifact["honest_verdict"] == "blocked_schema_compatible_with_2837"

    monkeypatch.setattr(
        mod,
        "write_corpus_artifact",
        lambda config: {"honest_verdict": "complete: fake_schema_ok"},
    )
    assert mod.main() == 0
    assert "complete: fake_schema_ok" in capsys.readouterr().out


def test_req_data_3858_write_artifact_is_deterministic(tmp_path: Path) -> None:
    """REQ-DATA-3858: fixed seed and checksum make the written artifact reproducible."""

    records = [_prmbench_record(idx, AXES[idx % len(AXES)]) for idx in range(12)]
    config = _config(tmp_path, target_n=12, min_incorrect=6)
    availability = mod.SourceAvailability(
        prmbench_reachable=True,
        fover_v3_paths=[],
        preconditions_checked=[{"resource": "prmbench_hf", "available": True, "detail": "fixture"}],
    )

    first = mod.write_corpus_artifact(
        config,
        availability=availability,
        prmbench_records=records,
        schema_validator=lambda item: True,
        started_s=0.0,
        now_s=1.0,
    )
    second = mod.write_corpus_artifact(
        config,
        availability=availability,
        prmbench_records=records,
        schema_validator=lambda item: True,
        started_s=0.0,
        now_s=1.0,
    )

    assert first == second
    saved = json.loads(config.output_path.read_text(encoding="utf-8"))
    assert saved == first
    assert saved["reproducibility_checksum"] == first["reproducibility_checksum"]
