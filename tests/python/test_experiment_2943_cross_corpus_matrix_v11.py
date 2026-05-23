"""Tests for Exp 2943 cross-corpus matrix v11.

Spec refs: REQ-REPORT-2943, SCENARIO-REPORT-2943.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import cross_corpus_matrix_v11_2943 as mod


REQUIRED_FIELDS = {
    "honest_verdict",
    "inference_substrate",
    "rows_clean",
    "rows_flagged",
    "rows_blocked",
    "per_corpus_auprc",
    "kv260_same_schedule_speedup_recorded",
    "kv260_n_crossover_measured",
    "cited_upstream_artifacts",
    "duration_s",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_ready_sources(root: Path) -> None:
    _write_json(
        root,
        mod.MATRIX_V10_REL_PATH,
        {
            "honest_verdict": "complete: matrix v10 fixture",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "clean_rows": ["corpus:FoVer", "exp2910_sota_codegen"],
            "flagged_rows": ["exp2911_code_hallucination_verifier"],
            "blocked_rows": ["exp2914_gatemate_toolchain"],
            "matrix_rows": [
                {"row_id": "corpus:FoVer", "row_class": "clean"},
                {"row_id": "exp2911_code_hallucination_verifier", "row_class": "flagged"},
            ],
        },
    )
    _write_json(
        root,
        mod.EXP2938_REL_PATH,
        {
            "honest_verdict": "complete: kv260_mmd_vs_cpu_sequential_gibbs_recorded",
            "inference_substrate": "hardware_smoke",
            "distributions_distinguishable": True,
            "per_seed_mmd_squared": [1.3, 1.4, 1.5],
            "per_seed_mmd_pvalue": [0.001, 0.001, 0.001],
            "per_seed_ks_statistic": [0.99, 0.98, 0.97],
            "paper_v6_recommendation": "retract exact-sampling claim",
        },
    )
    _write_json(
        root,
        mod.EXP2939_REL_PATH,
        {
            "honest_verdict": "complete: kv260_slower_than_same_schedule_cpu_at_n64",
            "inference_substrate": "live_llm_inference",
            "kv260_speedup_vs_same_schedule_cpu": {"value": 0.5},
            "paper_v6_recommendation": "retract speedup claim",
        },
    )
    _write_json(
        root,
        mod.EXP2940_REL_PATH,
        {
            "honest_verdict": "complete: verifier provides meaningful information",
            "inference_substrate": mod.INFERENCE_SUBSTRATE,
            "code_corpus_auprc": 0.75,
            "code_corpus_baseline_random_auprc": {"value": 0.075},
            "fover_corpus_auprc": {"value": 0.9},
            "paper_v6_recommendation": {"value": "retain"},
        },
    )
    _write_json(
        root,
        mod.EXP2942_REL_PATH,
        {
            "honest_verdict": "complete: kv260_fixed_n64_latency_profile_recorded",
            "inference_substrate": "hardware_smoke",
            "bitstream_supports_variable_n": False,
            "measured_crossover_n": None,
            "per_n_results": [{"n": 64, "per_sample_us_median": 25.54, "per_sample_us_p95": 25.77}],
        },
    )


def test_req_report_2943_spec_anchor_exists() -> None:
    """REQ-REPORT-2943: OpenSpec declares the v11 matrix contract first."""

    spec = Path("openspec/capabilities/research-reporting/spec.md").read_text(encoding="utf-8")

    assert "REQ-REPORT-2943" in spec
    assert "SCENARIO-REPORT-2943" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec


def test_scenario_report_2943_builds_v11_from_upstream_artifacts(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2943: v11 carries v10 buckets plus AUPRC/corrigenda fields."""

    _write_ready_sources(tmp_path)

    artifact = mod.build_artifact(tmp_path, started_s=10.0, now_s=12.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["duration_s"] == pytest.approx(2.25)
    assert artifact["rows_clean"] == [
        "corpus:FoVer",
        "exp2910_sota_codegen",
        "exp2938_kv260_mmd_corrigendum",
        "exp2939_same_schedule_speedup_corrigendum",
        "exp2940_code_corpus_auprc_corrigendum",
        "exp2942_kv260_n_scaling_corrigendum",
    ]
    assert artifact["rows_flagged"] == ["exp2911_code_hallucination_verifier"]
    assert artifact["rows_blocked"] == ["exp2914_gatemate_toolchain"]
    assert artifact["per_corpus_auprc"] == {
        "FoVer": {
            "source_experiment_id": "exp2940",
            "source_field": "fover_corpus_auprc.value",
            "value": 0.9,
        },
        "code_corpora": {
            "baseline_random_auprc": 0.075,
            "source_experiment_id": "exp2940",
            "source_field": "code_corpus_auprc",
            "value": 0.75,
        },
    }
    assert artifact["kv260_same_schedule_speedup_recorded"] == pytest.approx(0.5)
    assert artifact["kv260_n_crossover_measured"] == 0
    assert (
        artifact["deep_think_corrigenda_outcomes"]["exp2938"]["distributions_distinguishable"]
        is True
    )
    assert artifact["deep_think_corrigenda_outcomes"]["exp2942"]["measured_crossover_n"] == 0

    cited = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert set(cited) == {"exp2935", "exp2938", "exp2939", "exp2940", "exp2942"}
    assert cited["exp2940"]["sha256"] == _sha256(tmp_path / mod.EXP2940_REL_PATH)
    assert cited["exp2940"]["fields_imported"] == [
        "code_corpus_auprc",
        "code_corpus_baseline_random_auprc.value",
        "fover_corpus_auprc.value",
        "paper_v6_recommendation",
    ]


def test_req_report_2943_write_artifact_persists_json(tmp_path: Path) -> None:
    """REQ-REPORT-2943: write_artifact emits the stable deliverable JSON."""

    _write_ready_sources(tmp_path)

    output = mod.write_artifact(tmp_path, started_s=1.0, now_s=1.5)
    saved = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert saved["honest_verdict"].startswith("complete:")
    assert saved["kv260_same_schedule_speedup_recorded"] == pytest.approx(0.5)


def test_req_report_2943_blocks_missing_or_malformed_required_upstream(tmp_path: Path) -> None:
    """REQ-REPORT-2943: missing or malformed upstreams fail closed with required fields."""

    _write_ready_sources(tmp_path)
    (tmp_path / mod.EXP2938_REL_PATH).write_text("[not-an-object]\n", encoding="utf-8")
    (tmp_path / mod.EXP2942_REL_PATH).unlink()

    artifact = mod.build_artifact(tmp_path, started_s=2.0, now_s=2.25)

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["honest_verdict"] == "blocked_required_upstream_missing"
    assert artifact["rows_clean"] == ["corpus:FoVer", "exp2910_sota_codegen"]
    assert artifact["rows_flagged"] == ["exp2911_code_hallucination_verifier"]
    assert artifact["rows_blocked"] == ["exp2914_gatemate_toolchain"]
    assert artifact["per_corpus_auprc"] == {}
    assert artifact["kv260_same_schedule_speedup_recorded"] == 0.0
    assert artifact["kv260_n_crossover_measured"] == 0
    assert {
        (error["experiment_id"], error["reason"]) for error in artifact["required_upstream_errors"]
    } == {
        ("exp2938", "missing_or_malformed_artifact"),
        ("exp2942", "missing_or_malformed_artifact"),
    }
    cited = {item["experiment_id"]: item for item in artifact["cited_upstream_artifacts"]}
    assert cited["exp2938"]["sha256"] == _sha256(tmp_path / mod.EXP2938_REL_PATH)
    assert cited["exp2942"]["sha256"] is None

    _write_ready_sources(tmp_path)
    _write_json(
        tmp_path,
        mod.EXP2939_REL_PATH,
        {
            "honest_verdict": "complete: speedup field missing",
            "inference_substrate": "live_llm_inference",
            "paper_v6_recommendation": "blocked in fixture",
        },
    )

    missing_field = mod.build_artifact(tmp_path, started_s=3.0, now_s=3.1)

    assert missing_field["honest_verdict"] == "blocked_required_upstream_missing"
    assert missing_field["required_upstream_errors"] == [
        {
            "artifact_path": mod.EXP2939_REL_PATH.as_posix(),
            "experiment_id": "exp2939",
            "missing_fields": ["kv260_speedup_vs_same_schedule_cpu.value"],
            "reason": "missing_required_field",
        }
    ]


def test_req_report_2943_helper_edges_keep_numeric_fields_honest() -> None:
    """REQ-REPORT-2943: helper edges do not invent speedups or crossover rows."""

    assert mod._kv260_n_crossover_measured({"kv260_n_crossover_measured": 256}) == 256
    assert mod._kv260_n_crossover_measured({"measured_crossover_n": 512}) == 512
    assert mod._kv260_n_crossover_measured({"measured_crossover_n": None}) == 0
    assert mod._unique_strings(["a", 1, "a"]) == ["a", "1"]
    assert mod._v10_bucket({"rows_clean": ["legacy"]}, "clean") == ["legacy"]
    assert mod._v10_bucket({"clean_rows": ["current"]}, "clean") == ["current"]
    assert mod._get_path({"a": {"b": 3}}, "a.b") == 3
    assert mod._get_path({"a": 1}, "a.b") is None
    with pytest.raises(ValueError, match="numeric"):
        mod._as_float("not-a-number", "bad_field")
