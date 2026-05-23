"""Tests for the Exp 2908 milestone .274 capstone artifact.

Spec refs: REQ-REPORT-2908, SCENARIO-REPORT-2908.

These tests construct each .274 source artifact with the minimum fields the
classifier consults, then verify the capstone:

- classifies adversarially-flagged inputs as flagged even when ready booleans pass,
- routes deliberate pilot-only artifacts into the pilot bucket,
- marks ``hardware_portfolio_reactivated`` True only when at least three of
  four hardware tracks land clean and the fourth is honestly blocked,
- marks ``kv260_first_latency_recorded`` True only when overlay+UIO+transcript+sha exist,
- marks ``gatemate_bitstream_built`` False when nextpnr-gatemate is missing,
- marks ``polarfire_smoke_verified`` True when riscv64+hash match,
- marks ``thrml_import_repaired`` True when import+version recorded,
- marks ``cross_corpus_matrix_v8_built`` True only when matrix v8 is clean.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.reporting import capstone_v274_2908 as exp2908


def _write_json(root: Path, rel_path: str | Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _exp2897_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: archive_ready=true; activated_milestone=2026.05.274",
        "activated_milestone": "2026.05.274",
        "archive_already_present": True,
    }


def _exp2898_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_hardware_latency_transcript_recorded",
        "inference_substrate": "hardware_smoke",
        "kv260_overlay_loaded": "carnot_ising_v2_n64",
        "kv260_uio_devices_present": ["/dev/uio0", "/dev/uio1", "/dev/uio2"],
        "bitstream_sha256": "a90028a7931de505edd38caadfccfc1d7d5aa21e8abca5ae48bfb8a63a9876cb",
        "board_transcript_path": "results/experiment_2898_kv260_transcript.log",
        "reproducibility_checksum": "aabfc490ef6288444abcabc6ccb27cf3dcdc8b436431e72564b8b42bc6c5b79f",
        "per_seed_results": [
            {
                "seed": 1,
                "n_samples": 1000,
                "per_sample_wall_clock_us_median": 24.0,
                "per_sample_wall_clock_us_p95": 24.5,
                "final_energy": -42.0,
            }
        ],
    }


def _exp2899_blocked_toolchain() -> dict[str, Any]:
    return {
        "honest_verdict": "blocked_gatemate_toolchain_missing",
        "inference_substrate": "hardware_smoke",
        "synth_succeeded": False,
        "place_and_route_succeeded": False,
        "bitstream_path": None,
        "bitstream_sha256": None,
    }


def _exp2900_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: polarfire_riscv64_constraint_scorer_hash_verified",
        "inference_substrate": "hardware_smoke",
        "scorer_output_hash_verified": True,
        "scorer_output_sha256": "9205fd7c5370cbe3e66ad246a4de4e91e628897951681c550bd52a2e31f27cde",
        "expected_scorer_output_sha256": "9205fd7c5370cbe3e66ad246a4de4e91e628897951681c550bd52a2e31f27cde",
        "polarfire_arch": "riscv64",
        "no_fpga_fabric_claim": True,
    }


def _exp2901_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: thrml_import_repaired_n16_parity_passed_no_hardware_claim",
        "inference_substrate": "live_llm_inference",
        "thrml_import_succeeded": True,
        "thrml_version_installed": "0.1.3",
        "parity_energy_delta": 2.47955e-07,
        "random_seed": 202605232901,
        "reproducibility_checksum": "14bf58d1512dff765f165866457a776a8d0fc2419f93da3c39456602be2e024f",
    }


def _exp2902_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: cross-corpus matrix v8 aggregated; clean=6; flagged=2",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "rows_clean": [
            "corpus:FoVer",
            "corpus:HaluEval_FEVER",
            "corpus:TruthfulQA",
            "exp2890_code_structural_dependency",
            "exp2892_vericot",
            "exp2898_kv260_hardware",
        ],
        "rows_flagged": ["corpus:MBPP", "corpus:HumanEval"],
        "rows_blocked": [],
        "rows_pilot_only": ["corpus:MBPP", "corpus:HumanEval", "exp2891_cctu"],
    }


def _exp2903_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: paper_v6_hardware_validation_section_staged_from_exp2898",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "snippet_written": True,
        "latex_snippet_path": "docs/arxiv-paper/sections/hardware-validation-v1.tex",
        "main_tex_modified": False,
        "operator_only_external_publication": True,
        "kv260_latency_cited_p50_us": 24.04,
        "kv260_latency_cited_p95_us": 24.38,
        "n_spins": 64,
    }


def _exp2904_clean() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: KAN hardware complexity accounting v2 aggregated; no KAN synthesis claim"
        ),
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "kan_node_count": 2,
        "kv260_lut_used": 52564,
        "kan_synthesis_claim_made": False,
        "new_board_execution_claim_made": False,
    }


def _exp2905_flagged() -> dict[str, Any]:
    return {
        "honest_verdict": (
            "complete: bounded-budget k=8 live SOTA code generation executed with"
            " pass@1=0.5000 and pass@k=0.5000"
        ),
        "inference_substrate": "live_llm_inference",
        "aggregate_pass_at_1": 0.5,
        "aggregate_pass_at_k": 0.5,
        "flagged_adversarial": True,
        "corrigendum_pending": [
            {"kind": "TAUTOLOGY", "severity": "critical"},
            {"kind": "METHODOLOGY_MISSING", "severity": "warn"},
        ],
    }


def _exp2906_pilot_only() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: kv260_replay_dispatch_path_validated_pilot_only",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "pilot_only": True,
        "no_hardware_performance_claim": True,
        "dispatch_path_validated": True,
    }


def _exp2907_clean() -> dict[str, Any]:
    return {
        "honest_verdict": "complete: operator_hardware_portfolio_status_aggregated",
        "inference_substrate": "aggregation_from_upstream_artifacts",
        "per_board_status": {
            "kv260": {
                "state": "ready_live_latency_recorded",
                "last_artifact": "results/experiment_2898_kv260_ising_sampler_hardware_latency_benchmark_v1.json",
                "next_step": "Add same-basis CPU baseline before speedup claims.",
            },
            "gatemate": {
                "state": "blocked_gatemate_toolchain_missing",
                "last_artifact": "results/experiment_2899_gatemate_a1_n16_ising_tile_bitstream_build_v1.json",
                "next_step": "Provision nextpnr-gatemate, rerun build, do not flash until bitstream exists.",
            },
            "polarfire": {
                "state": "ready_riscv64_cpu_dispatch_verified",
                "last_artifact": "results/experiment_2900_polarfire_carnot_dispatch_smoke_v1.json",
                "next_step": "Treat as CPU-dispatch proof; fabric acceleration remains separate.",
            },
            "thrml": {
                "state": "ready_software_parity_no_tsu_claim",
                "last_artifact": "results/experiment_2901_thrml_local_import_repair_v1.json",
                "next_step": "Use import/parity evidence; require TSU access before hardware claims.",
            },
        },
    }


def _write_all_sources(root: Path) -> None:
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2897"], _exp2897_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2898"], _exp2898_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2899"], _exp2899_blocked_toolchain())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2900"], _exp2900_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2901"], _exp2901_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2902"], _exp2902_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2903"], _exp2903_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2904"], _exp2904_clean())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2905"], _exp2905_flagged())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2906"], _exp2906_pilot_only())
    _write_json(root, exp2908.EXPECTED_ARTIFACTS["exp2907"], _exp2907_clean())


def test_req_report_2908_required_top_level_fields_present(tmp_path: Path) -> None:
    """REQ-REPORT-2908: capstone surface contains every operator-contracted field."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=10.0, now_s=11.25)

    required = {
        "honest_verdict",
        "inference_substrate",
        "milestone",
        "clean_artifacts",
        "flagged_artifacts",
        "blocked_artifacts",
        "missing_artifacts",
        "paper_ready",
        "hardware_portfolio_reactivated",
        "kv260_first_latency_recorded",
        "gatemate_bitstream_built",
        "polarfire_smoke_verified",
        "thrml_import_repaired",
        "cross_corpus_matrix_v8_built",
        "top_3_next_actions",
        "gaps_for_275",
        "cited_upstream_artifacts",
        "duration_s",
    }
    assert required <= artifact.keys()
    assert artifact["honest_verdict"].startswith("complete:")
    assert artifact["milestone"] == "2026.05.274"
    assert artifact["inference_substrate"] == "aggregation_from_upstream_artifacts"
    assert artifact["duration_s"] == pytest.approx(1.25)


def test_scenario_report_2908_hardware_portfolio_reactivated_with_gatemate_blocked(
    tmp_path: Path,
) -> None:
    """SCENARIO-REPORT-2908: three clean tracks + honestly-blocked GateMate = reactivated."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert artifact["kv260_first_latency_recorded"] is True
    assert artifact["polarfire_smoke_verified"] is True
    assert artifact["thrml_import_repaired"] is True
    assert artifact["gatemate_bitstream_built"] is False
    assert artifact["hardware_portfolio_reactivated"] is True
    # exp2899 honestly blocked (toolchain missing) — not flagged, not missing.
    assert "exp2899" in artifact["blocked_artifacts"]
    assert "exp2899" not in artifact["flagged_artifacts"]


def test_scenario_report_2908_flagged_sota_kept_out_of_clean(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: exp2905 TAUTOLOGY flag keeps it out of clean bucket."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert "exp2905" in artifact["flagged_artifacts"]
    assert "exp2905" not in artifact["clean_artifacts"]
    # Top-3 actions must mention the SOTA re-run because the artifact is flagged.
    assert any("SOTA" in action for action in artifact["top_3_next_actions"])
    # Gaps for .275 must surface the tautology.
    assert any("TAUTOLOGY" in gap for gap in artifact["gaps_for_275"])


def test_scenario_report_2908_pilot_only_fr11_routed_separately(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: exp2906 dispatch pilot stays out of clean bucket."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert "exp2906" in artifact["pilot_only_artifacts"]
    assert "exp2906" not in artifact["clean_artifacts"]
    assert "exp2906" not in artifact["flagged_artifacts"]


def test_scenario_report_2908_matrix_v8_drives_paper_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: clean matrix v8 with FoVer + others -> paper_ready=True."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert artifact["cross_corpus_matrix_v8_built"] is True
    assert artifact["paper_ready"] is True
    rows = artifact["headline_eligible_rows"]
    assert any("FoVer" in r for r in rows)
    assert len(rows) >= 2


def test_scenario_report_2908_missing_matrix_blocks_paper_ready(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: absent matrix v8 forces paper_ready=False."""

    _write_all_sources(tmp_path)
    # Delete the matrix v8 artifact and rerun.
    (tmp_path / exp2908.EXPECTED_ARTIFACTS["exp2902"]).unlink()

    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert "exp2902" in artifact["missing_artifacts"]
    assert artifact["cross_corpus_matrix_v8_built"] is False
    assert artifact["paper_ready"] is False
    assert artifact["headline_eligible_rows"] == []


def test_scenario_report_2908_missing_kv260_unsets_hardware_flags(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: KV260 absence drops latency and portfolio booleans."""

    _write_all_sources(tmp_path)
    (tmp_path / exp2908.EXPECTED_ARTIFACTS["exp2898"]).unlink()

    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    assert artifact["kv260_first_latency_recorded"] is False
    # KV260 missing + GateMate blocked = only 2 clean tracks; portfolio not reactivated.
    assert artifact["hardware_portfolio_reactivated"] is False
    assert "exp2898" in artifact["missing_artifacts"]


def test_scenario_report_2908_polarfire_hash_mismatch_blocks_smoke(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: polarfire smoke needs hash match."""

    _write_all_sources(tmp_path)
    bad_polarfire = _exp2900_clean()
    bad_polarfire["scorer_output_sha256"] = "deadbeef" * 8
    _write_json(tmp_path, exp2908.EXPECTED_ARTIFACTS["exp2900"], bad_polarfire)

    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert artifact["polarfire_smoke_verified"] is False


def test_scenario_report_2908_thrml_import_failure_unsets_flag(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: thrml_import_succeeded=False unsets thrml_import_repaired."""

    _write_all_sources(tmp_path)
    bad_thrml = _exp2901_clean()
    bad_thrml["thrml_import_succeeded"] = False
    _write_json(tmp_path, exp2908.EXPECTED_ARTIFACTS["exp2901"], bad_thrml)

    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert artifact["thrml_import_repaired"] is False


def test_scenario_report_2908_gatemate_clean_sets_built(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: a hypothetical clean GateMate run flips bitstream_built True."""

    _write_all_sources(tmp_path)
    good_gatemate = {
        "honest_verdict": "complete: gatemate_n16_ising_tile_bitstream_built",
        "inference_substrate": "hardware_smoke",
        "synth_succeeded": True,
        "place_and_route_succeeded": True,
        "bitstream_sha256": "f00dface" * 8,
    }
    _write_json(tmp_path, exp2908.EXPECTED_ARTIFACTS["exp2899"], good_gatemate)

    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)
    assert artifact["gatemate_bitstream_built"] is True
    # When all four hardware tracks are clean, the portfolio is reactivated.
    assert artifact["hardware_portfolio_reactivated"] is True


def test_scenario_report_2908_cited_upstream_artifacts_have_sha256(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: every present source artifact carries a sha256 citation."""

    _write_all_sources(tmp_path)
    artifact = exp2908.build_artifact(tmp_path, started_s=0.0, now_s=0.1)

    citations = artifact["cited_upstream_artifacts"]
    assert len(citations) == len(exp2908.EXPECTED_ARTIFACTS)
    for citation in citations:
        assert isinstance(citation["sha256"], str)
        assert len(citation["sha256"]) == 64
        assert citation["artifact_path"].startswith("results/")


def test_scenario_report_2908_write_artifact_persists_json(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: write_artifact persists the JSON file at the target path."""

    _write_all_sources(tmp_path)
    out_path = exp2908.write_artifact(
        tmp_path,
        output_path=tmp_path / "results/experiment_2908_capstone_v274.json",
        started_s=0.0,
        now_s=0.1,
    )

    assert out_path.is_file()
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["milestone"] == "2026.05.274"
    assert payload["schema"] == exp2908.SCHEMA
    # The on-disk artifact is sorted by key — sanity-check via a known key.
    assert "hardware_portfolio_reactivated" in payload


def test_scenario_report_2908_classify_artifact_handles_missing_and_malformed() -> None:
    """SCENARIO-REPORT-2908: classifier returns 'missing' for absent or junk payloads."""

    assert exp2908.classify_artifact("exp2898", {}, present=False) == "missing"
    assert exp2908.classify_artifact("exp2898", {}, present=True) == "missing"
    # Terminal verdict but required boolean unsatisfied — downgraded to blocked.
    # exp2899 requires synth_succeeded: True; absent means blocked.
    assert (
        exp2908.classify_artifact(
            "exp2899",
            {"honest_verdict": "complete: but_synth_not_recorded"},
            present=True,
        )
        == "blocked"
    )
    # Blocked verdict prefix routes to blocked.
    assert (
        exp2908.classify_artifact(
            "exp2899", {"honest_verdict": "blocked_gatemate_toolchain_missing"}, present=True
        )
        == "blocked"
    )
    # Non-terminal verdict routes to missing.
    assert (
        exp2908.classify_artifact(
            "exp2898", {"honest_verdict": "partial: something"}, present=True
        )
        == "missing"
    )


def test_scenario_report_2908_read_json_returns_empty_on_bad_file(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: read_json swallows errors and returns {}."""

    bad = tmp_path / "garbage.json"
    bad.write_text("{not json", encoding="utf-8")
    assert exp2908.read_json(bad) == {}

    nonexistent = tmp_path / "nope.json"
    assert exp2908.read_json(nonexistent) == {}

    # Non-dict payload also returns {}.
    nondict = tmp_path / "list.json"
    nondict.write_text("[1, 2, 3]", encoding="utf-8")
    assert exp2908.read_json(nondict) == {}


def test_scenario_report_2908_flag_detection_modes(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: every flag mechanism triggers the flagged bucket."""

    # adversarial_verify_flags list non-empty triggers flag.
    assert exp2908._has_flags({"adversarial_verify_flags": [{"kind": "X"}]}) is True
    # adversarial_verify_summary with flag_count > 0 triggers flag.
    assert (
        exp2908._has_flags({"adversarial_verify_summary": {"flag_count": 1}}) is True
    )
    # adversarial_verify_passed == False triggers flag.
    assert exp2908._has_flags({"adversarial_verify_passed": False}) is True
    # No flag mechanisms -> no flag.
    assert exp2908._has_flags({"honest_verdict": "complete: x"}) is False


def test_scenario_report_2908_top_3_actions_padded_when_clean(tmp_path: Path) -> None:
    """SCENARIO-REPORT-2908: top_3_next_actions always returns exactly three entries."""

    actions = exp2908._top_3_next_actions(
        statuses={},
        gatemate_built=True,
        kv260_latency=False,
        paper_ready=True,
        sota_flagged=False,
    )
    assert len(actions) == 3
