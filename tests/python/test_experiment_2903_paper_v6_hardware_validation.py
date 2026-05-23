"""Tests for Exp 2903 paper-v6 hardware-validation snippet staging.

Spec refs: REQ-PUBLISH-035, SCENARIO-PUBLISH-035, SCENARIO-PUBLISH-035B.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import paper_v6_hardware_validation_2903 as exp2903


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256(root: Path, rel_path: str | Path) -> str:
    return hashlib.sha256((root / rel_path).read_bytes()).hexdigest()


def _exp2898_payload() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
        "bitstream_sha256": "a90028a7931de505edd38caadfccfc1d7d5aa21e8abca5ae48bfb8a63a9876cb",
        "bitstream_sha256_source": "board:/lib/firmware/xilinx/carnot_ising_v4",
        "ising_problem_spec": {"n_spins": 64},
        "preconditions_checked": [
            {"resource": "kv260_ssh", "available": True},
            {"resource": "kv260_overlay", "available": True},
            {"resource": "kv260_uio0", "available": True},
        ],
        "per_seed_results": [
            {
                "seed": 42,
                "n_samples": 10000,
                "per_sample_wall_clock_us_median": 24.05,
                "per_sample_wall_clock_us_p95": 24.38,
            },
            {
                "seed": 137,
                "n_samples": 10000,
                "per_sample_wall_clock_us_median": 24.04,
                "per_sample_wall_clock_us_p95": 24.36,
            },
            {
                "seed": 271,
                "n_samples": 10000,
                "per_sample_wall_clock_us_median": 24.01,
                "per_sample_wall_clock_us_p95": 24.33,
            },
        ],
        "sample_count_sweep_results": [
            {"seed": 42, "n_samples": 100, "failed_samples": 0},
            {"seed": 137, "n_samples": 1000, "failed_samples": 0},
            {"seed": 271, "n_samples": 10000, "failed_samples": 0},
        ],
    }


def test_req_publish_035_spec_anchor_exists() -> None:
    """REQ-PUBLISH-035: the hardware-validation staging contract is in OpenSpec."""

    spec = (exp2903.REPO_ROOT / "openspec/capabilities/publication/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-PUBLISH-035" in spec
    assert "SCENARIO-PUBLISH-035" in spec
    assert "SCENARIO-PUBLISH-035B" in spec
    assert "experiment_2903_paper_v6_hardware_validation_section_v1.json" in spec
    assert "hardware-validation-v1.tex" in spec


def test_scenario_publish_035_stages_snippet_and_artifact_without_main_tex_edit(
    tmp_path: Path,
) -> None:
    """SCENARIO-PUBLISH-035: clean KV260 evidence becomes a standalone snippet."""

    _write_json(tmp_path, exp2903.EXP2898_REL_PATH, _exp2898_payload())
    main_tex = tmp_path / exp2903.PAPER_REL_PATH
    main_tex.parent.mkdir(parents=True, exist_ok=True)
    original_main = "\\section{Hardware Acceleration}\\nNo staged input yet.\\n"
    main_tex.write_text(original_main, encoding="utf-8")

    out = exp2903.write_outputs(tmp_path, started_s=10.0, now_s=12.5)

    artifact = json.loads(out.read_text(encoding="utf-8"))
    snippet = (tmp_path / exp2903.SNIPPET_REL_PATH).read_text(encoding="utf-8")

    required = {
        "honest_verdict",
        "inference_substrate",
        "latex_snippet_path",
        "kv260_latency_cited_p50_us",
        "kv260_latency_cited_p95_us",
        "bitstream_sha256_cited",
        "cited_upstream_artifacts",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["latex_snippet_path"] == exp2903.SNIPPET_REL_PATH.as_posix()
    assert artifact["kv260_latency_cited_p50_us"] == pytest.approx(24.04)
    assert artifact["kv260_latency_cited_p95_us"] == pytest.approx(24.38)
    assert artifact["bitstream_sha256_cited"] == _exp2898_payload()["bitstream_sha256"]
    assert artifact["duration_s"] == pytest.approx(2.5)
    assert artifact["snippet_written"] is True
    assert artifact["main_tex_modified"] is False
    assert artifact["operator_only_external_publication"] is True
    assert artifact["cited_upstream_artifacts"] == [
        {
            "experiment_id": "exp2898",
            "artifact_path": exp2903.EXP2898_REL_PATH.as_posix(),
            "fields_imported": exp2903.FIELDS_IMPORTED,
            "sha256": _sha256(tmp_path, exp2903.EXP2898_REL_PATH),
        }
    ]

    assert "\\subsection{Hardware Validation}" in snippet
    assert "Xilinx Kria KV260" in snippet
    assert "carnot\\_ising\\_v2\\_n64" in snippet
    assert "n\\_spins=64" in snippet
    assert "24.05" in snippet and "24.38" in snippet
    assert "24.04" in snippet and "24.36" in snippet
    assert "24.01" in snippet and "24.33" in snippet
    assert "No same-basis CPU baseline has been measured yet" in snippet
    assert "no FPGA speedup claim" in snippet
    assert "\\input" not in (tmp_path / exp2903.PAPER_REL_PATH).read_text(encoding="utf-8")
    assert main_tex.read_text(encoding="utf-8") == original_main


def test_scenario_publish_035_blocks_unclean_upstream_without_snippet(tmp_path: Path) -> None:
    """SCENARIO-PUBLISH-035B: failed upstream gates do not create paper text."""

    payload = _exp2898_payload()
    payload["sample_count_sweep_results"][1]["failed_samples"] = 2
    payload["acceptance_gate_results"] = [{"name": "latency_rows", "passed": False}]
    _write_json(tmp_path, exp2903.EXP2898_REL_PATH, payload)

    out = exp2903.write_outputs(tmp_path, started_s=4.0, now_s=5.0)
    artifact = json.loads(out.read_text(encoding="utf-8"))

    assert artifact["honest_verdict"].startswith("blocked:")
    assert artifact["snippet_written"] is False
    assert artifact["kv260_latency_cited_p50_us"] == pytest.approx(0.0)
    assert artifact["kv260_latency_cited_p95_us"] == pytest.approx(0.0)
    assert "sample_count_sweep_results[1].failed_samples=2" in artifact["blocked_reasons"]
    assert "acceptance_gate_results[0] failed" in artifact["blocked_reasons"]
    assert not (tmp_path / exp2903.SNIPPET_REL_PATH).exists()


def test_req_publish_035_input_validation_and_read_json_fail_closed(tmp_path: Path) -> None:
    """REQ-PUBLISH-035: malformed, missing, or unavailable evidence fails closed."""

    assert exp2903.read_json(tmp_path / "missing.json") == {}
    bad = tmp_path / "bad.json"
    bad.write_text("{bad", encoding="utf-8")
    assert exp2903.read_json(bad) == {}
    array = tmp_path / "array.json"
    array.write_text("[1]", encoding="utf-8")
    assert exp2903.read_json(array) == {}

    missing_artifact = exp2903.build_artifact(tmp_path, started_s=0.0, now_s=0.25)
    assert missing_artifact["honest_verdict"].startswith("blocked:")
    assert missing_artifact["blocked_reasons"] == ["exp2898_artifact_missing_or_malformed"]
    assert missing_artifact["cited_upstream_artifacts"] == []

    non_terminal = {**_exp2898_payload(), "honest_verdict": "running"}
    assert "honest_verdict_not_complete_or_success" in exp2903.validate_upstream(non_terminal)

    wrong_substrate = {**_exp2898_payload(), "inference_substrate": "software_proxy"}
    assert "inference_substrate_not_hardware_smoke" in exp2903.validate_upstream(
        wrong_substrate
    )

    unavailable = {
        **_exp2898_payload(),
        "preconditions_checked": [{"resource": "kv260_ssh", "available": False}],
    }
    assert "preconditions_not_all_available" in exp2903.validate_upstream(unavailable)

    no_rows = {**_exp2898_payload(), "per_seed_results": []}
    assert "per_seed_results_missing" in exp2903.validate_upstream(no_rows)

    bad_latency = _exp2898_payload()
    bad_latency["per_seed_results"][0]["per_sample_wall_clock_us_p95"] = 0
    assert "per_seed_results_have_nonpositive_latency" in exp2903.validate_upstream(bad_latency)

    malformed_row = _exp2898_payload()
    malformed_row["per_seed_results"][0] = "not a row"
    assert "per_seed_results_have_nonpositive_latency" in exp2903.validate_upstream(
        malformed_row
    )

    gate_dict = _exp2898_payload()
    gate_dict["gate_results"] = {"paper_gate": {"status": "failed"}}
    assert "gate_results[paper_gate] failed" in exp2903.validate_upstream(gate_dict)
