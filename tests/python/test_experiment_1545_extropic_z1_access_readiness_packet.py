"""Tests for Exp 1545 Extropic Z1 access-readiness packet.

Spec refs: REQ-SAMPLE-055, SCENARIO-SAMPLE-083.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.reporting.extropic_z1_access_readiness_packet as exp1545


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _exp1543_payload(*, ready: bool = True, no_hardware: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "20260508",
        "thrml_parity_n256_schedule_ready": ready,
        "parity_passed": ready,
        "simulator_only": no_hardware,
        "no_tsu_hardware_claim": no_hardware,
        "n_spins": 256,
        "samples_per_schedule": 4096,
        "n_samples_per_backend": 12288,
        "mean_energy_delta": 0.004920247396,
        "max_energy_delta": 0.011440429688,
        "kl_divergence": 0.002662339801,
        "autocorrelation_delta": 0.01929397015,
        "schedule_manifest": [
            {
                "schedule_id": "low_beta_short_warmup",
                "beta": 0.9,
                "n_warmup": 384,
                "steps_per_sample": 3,
                "seed_offset": 0,
                "use_checkerboard": True,
            },
            {
                "schedule_id": "baseline_n128_style",
                "beta": 1.0,
                "n_warmup": 512,
                "steps_per_sample": 4,
                "seed_offset": 17,
                "use_checkerboard": True,
            },
            {
                "schedule_id": "high_beta_longer_thinning",
                "beta": 1.1,
                "n_warmup": 640,
                "steps_per_sample": 6,
                "seed_offset": 31,
                "use_checkerboard": True,
            },
        ],
        "schedule_results": {
            "low_beta_short_warmup": {"seeds": [20260508, 20360508]},
            "baseline_n128_style": {"seeds": [20260525, 20360525]},
            "high_beta_longer_thinning": {"seeds": [20260539, 20360539]},
        },
        "parity_report_path": "results/thrml_carnot_parity_n256_schedule_stress_1543.jsonl",
        "metadata": {
            "thrml_version": "0.1.3",
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "tsu_hardware_execution": False,
        },
    }


def _exp1544_payload(*, ready: bool = True, no_hardware: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "milestone": "2026.04.118",
        "diverse_topology_parity_n64_ready": ready,
        "parity_passed": ready,
        "simulator_only": no_hardware,
        "no_tsu_hardware_claim": no_hardware,
        "n_spins": 64,
        "topologies_tested": ["complete", "sparse_random", "lattice", "scale_free"],
        "topologies_passed": ["complete", "sparse_random", "lattice", "scale_free"],
        "topology_seeds": {
            "sparse_random": 20260510,
            "scale_free": 20260511,
        },
        "seeds": [20260508, 20260509, 20260510, 20260511, 20260512],
        "n_samples_per_backend": 40960,
        "mean_energy_delta": 0.001493994141,
        "max_energy_delta": 0.006869335937,
        "kl_divergence": 0.000728807813,
        "mean_energy_delta_by_topology": {
            "complete": 0.002992578125,
            "lattice": 0.0007546875,
            "scale_free": 0.006869335937,
            "sparse_random": 0.00285390625,
        },
        "kl_divergence_by_topology": {
            "complete": 0.003264431492,
            "lattice": 0.00259363621,
            "scale_free": 0.00523750856,
            "sparse_random": 0.001759267228,
        },
        "parity_report_path": "results/thrml_diverse_topology_parity_n64_1544.jsonl",
        "metadata": {
            "thrml_version": "0.1.3",
            "z1_hardware_execution": False,
            "xtr0_hardware_execution": False,
            "tsu_hardware_execution": False,
        },
    }


def test_spec_mentions_exp1545_contract() -> None:
    """REQ-SAMPLE-055, SCENARIO-SAMPLE-083: Exp1545 is spec-anchored."""

    spec = (exp1545.REPO_ROOT / "openspec/capabilities/training-inference/spec.md").read_text(
        encoding="utf-8"
    )

    assert "REQ-SAMPLE-055" in spec
    assert "SCENARIO-SAMPLE-083" in spec
    assert "experiment_1545_extropic_z1_access_readiness_packet.json" in spec
    assert "ops/extropic_z1_readiness_packet.md" in spec
    assert "no_hardware_execution_claim" in spec


def test_req_sample_055_builds_packet_schema_and_ready_artifact(tmp_path: Path) -> None:
    """REQ-SAMPLE-055: packet fields, schema fields, and blockers are explicit."""

    exp1543_path = tmp_path / "results" / "experiment_1543.json"
    exp1544_path = tmp_path / "results" / "experiment_1544.json"
    _write_json(exp1543_path, _exp1543_payload())
    _write_json(exp1544_path, _exp1544_payload())

    artifact, packet_text, transcript_schema = exp1545.build_artifact(
        exp1543=exp1545.load_json(exp1543_path),
        exp1544=exp1545.load_json(exp1544_path),
        simulator_artifact_paths=[exp1543_path, exp1544_path],
        hardware_wishlist_text=(
            "No Extropic hardware access, Z1/XTR-0 execution, or TSU latency claim "
            "without authenticated hardware evidence."
        ),
        research_references_text=(
            "Z1 early access 2026; readiness packet and benchmark spec, not a hardware claim."
        ),
        known_issues_text="THRML/Carnot parity independent-RNG audit remains a .119 gate.",
        packet_path="ops/extropic_z1_readiness_packet.md",
        transcript_schema_path="ops/extropic_z1_transcript_schema.json",
        focused_checks_passed=True,
    )

    assert exp1545.REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["status"] == "complete"
    assert artifact["milestone"] == "2026.04.118"
    assert artifact["extropic_z1_readiness_packet_ready"] is True
    assert artifact["readiness_packet_path"] == "ops/extropic_z1_readiness_packet.md"
    assert artifact["transcript_schema_path"] == "ops/extropic_z1_transcript_schema.json"
    assert artifact["benchmark_cases_included"] == 7
    assert artifact["no_hardware_execution_claim"] is True
    assert artifact["focused_checks_passed"] is True
    assert artifact["honest_verdict"].startswith("complete:")
    assert "no_authenticated_extropic_z1_or_xtr0_device_access" in artifact["access_blockers"]
    assert len(artifact["simulator_artifacts_referenced"]) == 2
    assert "authenticated_access_proof" in artifact["required_device_evidence_fields"]
    assert "output_samples_sha256" in artifact["required_device_evidence_fields"]
    assert "n256_schedule_stress:baseline_n128_style" in packet_text
    assert "n64_diverse_topology:scale_free" in packet_text
    assert "No Hardware Execution Claim" in packet_text
    assert "Rollback Criteria" in packet_text
    assert transcript_schema["required"] == exp1545.TRANSCRIPT_REQUIRED_FIELDS
    assert exp1545.validate_transcript_schema(transcript_schema) is True


def test_req_sample_055_rejects_hardware_claim_or_incomplete_sources(tmp_path: Path) -> None:
    """REQ-SAMPLE-055: readiness cannot be true from hardware-claiming sources."""

    good_1544 = _exp1544_payload()
    hardware_claiming_1543 = _exp1543_payload(no_hardware=False)

    with pytest.raises(ValueError, match="no-TSU simulator boundary"):
        exp1545.build_artifact(
            exp1543=hardware_claiming_1543,
            exp1544=good_1544,
            simulator_artifact_paths=[],
            hardware_wishlist_text="No Extropic hardware access.",
            research_references_text="readiness packet, not a hardware claim.",
            known_issues_text="",
            packet_path="ops/extropic_z1_readiness_packet.md",
            transcript_schema_path="ops/extropic_z1_transcript_schema.json",
            focused_checks_passed=True,
        )

    incomplete_1544 = _exp1544_payload(ready=False)
    with pytest.raises(ValueError, match="Exp1544"):
        exp1545.build_artifact(
            exp1543=_exp1543_payload(),
            exp1544=incomplete_1544,
            simulator_artifact_paths=[],
            hardware_wishlist_text="No Extropic hardware access.",
            research_references_text="readiness packet, not a hardware claim.",
            known_issues_text="",
            packet_path="ops/extropic_z1_readiness_packet.md",
            transcript_schema_path="ops/extropic_z1_transcript_schema.json",
            focused_checks_passed=True,
        )

    schema = exp1545.build_transcript_schema()
    schema["required"] = [field for field in schema["required"] if field != "device_identifier"]
    with pytest.raises(ValueError, match="device_identifier"):
        exp1545.validate_transcript_schema(schema)


def test_req_sample_055_source_validation_rejects_malformed_artifacts() -> None:
    """REQ-SAMPLE-055: every readiness gate has an explicit failure mode."""

    exp1543_cases = [
        ({"status": "blocked"}, "Exp1543 is not complete"),
        ({**_exp1543_payload(), "parity_passed": False}, "parity_passed"),
        ({**_exp1543_payload(), "n_spins": 128}, "n_spins=256"),
        ({**_exp1543_payload(), "schedule_manifest": []}, "schedule_manifest"),
        ({**_exp1543_payload(), "metadata": "bad"}, "metadata"),
        (
            {
                **_exp1543_payload(),
                "metadata": {"z1_hardware_execution": True},
            },
            "hardware execution",
        ),
    ]
    for payload, message in exp1543_cases:
        with pytest.raises(ValueError, match=message):
            exp1545.validate_source_artifacts(payload, _exp1544_payload())

    exp1544_cases = [
        ({**_exp1544_payload(), "parity_passed": False}, "parity_passed"),
        ({**_exp1544_payload(no_hardware=False)}, "no-TSU simulator boundary"),
        ({**_exp1544_payload(), "n_spins": 32}, "n_spins=64"),
        ({**_exp1544_payload(), "topologies_tested": ["complete"]}, "four required"),
        (
            {**_exp1544_payload(), "metadata": {"xtr0_hardware_execution": True}},
            "hardware execution",
        ),
    ]
    for payload, message in exp1544_cases:
        with pytest.raises(ValueError, match=message):
            exp1545.validate_source_artifacts(_exp1543_payload(), payload)


def test_req_sample_055_defensive_manifest_paths_are_covered(tmp_path: Path) -> None:
    """REQ-SAMPLE-055: optional malformed manifest inputs degrade deterministically."""

    exp1543 = {
        **_exp1543_payload(),
        "schedule_manifest": ["not-a-schedule", _exp1543_payload()["schedule_manifest"][0]],
        "schedule_results": {"low_beta_short_warmup": "bad-result"},
    }
    exp1544 = {
        **_exp1544_payload(),
        "per_topology_results": "bad-results",
    }
    cases = exp1545.build_benchmark_cases(exp1543, exp1544)

    assert cases[0]["case_id"] == "n256_schedule_stress:low_beta_short_warmup"
    assert cases[0]["seeds"] == []
    assert len(cases) == 5
    nonmapping_schedule_results = exp1545.build_benchmark_cases(
        {**_exp1543_payload(), "schedule_results": "bad-results"},
        _exp1544_payload(),
    )
    assert nonmapping_schedule_results[0]["seeds"] == []
    nonmapping_topology_row = exp1545.build_benchmark_cases(
        _exp1543_payload(),
        {**_exp1544_payload(), "per_topology_results": {"complete": "bad-result"}},
    )
    assert nonmapping_topology_row[3]["case_id"] == "n64_diverse_topology:complete"
    assert nonmapping_topology_row[3]["simulator_baseline_metrics"] == {}
    assert exp1545._relative_path(
        tmp_path / "plain-file.txt", repo_root=exp1545.REPO_ROOT
    ).endswith("plain-file.txt")
    assert exp1545._sha256_file(tmp_path / "missing.json") is None
    rendered = exp1545._render_case_table(
        [
            {
                "case_id": "manual",
                "n_spins": 1,
                "topology": "toy",
                "schedule": "not-a-dict",
                "seeds": "seed-string",
            }
        ]
    )
    assert "seed-string" in rendered[-1]


def test_req_sample_055_transcript_schema_validation_failures() -> None:
    """REQ-SAMPLE-055: transcript schema corruption fails before readiness."""

    schema = exp1545.build_transcript_schema()
    corruptions = [
        ({"title": "wrong"}, "title"),
        ({**schema, "type": "array"}, "object"),
        ({**schema, "properties": []}, "properties"),
        (
            {
                **schema,
                "properties": {
                    key: value
                    for key, value in schema["properties"].items()
                    if key != "device_identifier"
                },
            },
            "device_identifier",
        ),
        (
            {
                **schema,
                "properties": {
                    **schema["properties"],
                    "hardware_execution_performed": {"type": "boolean", "const": False},
                },
            },
            "hardware_execution_performed",
        ),
        (
            {
                **schema,
                "properties": {
                    **schema["properties"],
                    "simulator_fallback_used": {"type": "boolean", "const": True},
                },
            },
            "simulator_fallback_used",
        ),
    ]
    for bad_schema, message in corruptions:
        with pytest.raises(ValueError, match=message):
            exp1545.validate_transcript_schema(bad_schema)


def test_req_sample_055_terminal_validation_failure_modes(tmp_path: Path) -> None:
    """REQ-SAMPLE-055: terminal readiness fields cannot be silently weakened."""

    exp1543_path = tmp_path / "results" / "experiment_1543.json"
    exp1544_path = tmp_path / "results" / "experiment_1544.json"
    _write_json(exp1543_path, _exp1543_payload())
    _write_json(exp1544_path, _exp1544_payload())
    artifact, packet_text, transcript_schema = exp1545.build_artifact(
        exp1543=exp1545.load_json(exp1543_path),
        exp1544=exp1545.load_json(exp1544_path),
        simulator_artifact_paths=[exp1543_path, exp1544_path],
        hardware_wishlist_text="No Extropic hardware access.",
        research_references_text="readiness packet, not a hardware claim.",
        known_issues_text="independent RNG followup",
        packet_path="ops/extropic_z1_readiness_packet.md",
        transcript_schema_path="ops/extropic_z1_transcript_schema.json",
        focused_checks_passed=True,
    )

    with pytest.raises(ValueError, match="missing required fields"):
        incomplete = dict(artifact)
        incomplete.pop("status")
        exp1545.validate_terminal_artifact(
            incomplete,
            packet_text=packet_text,
            transcript_schema=transcript_schema,
        )
    for field, value, message in [
        ("no_hardware_execution_claim", False, "no_hardware_execution_claim"),
        ("extropic_z1_readiness_packet_ready", False, "readiness packet"),
        ("honest_verdict", "blocked_without_prefix", "honest_verdict"),
        ("access_blockers", [], "access_blockers"),
    ]:
        invalid = dict(artifact)
        invalid[field] = value
        with pytest.raises(ValueError, match=message):
            exp1545.validate_terminal_artifact(
                invalid,
                packet_text=packet_text,
                transcript_schema=transcript_schema,
            )
    with pytest.raises(ValueError, match="Benchmark Case List"):
        exp1545.validate_terminal_artifact(
            artifact,
            packet_text="No Hardware Execution Claim\nno_hardware_execution_claim: true",
            transcript_schema=transcript_schema,
        )


def test_req_sample_055_git_protected_file_probe_runs_in_repo() -> None:
    """REQ-SAMPLE-055: protected roadmap/conductor files are checked."""

    result = exp1545._protected_files_unchanged(exp1545.REPO_ROOT)

    assert set(result) == {"research-roadmap.yaml", "scripts/research_conductor.py"}
    assert all(isinstance(value, bool) for value in result.values())


def test_scenario_sample_083_run_writes_packet_schema_and_terminal_json(tmp_path: Path) -> None:
    """SCENARIO-SAMPLE-083: run writes terminal JSON with no hardware claim."""

    root = tmp_path
    _write_json(
        root / "results" / "experiment_1543_thrml_carnot_parity_n256_schedule_stress.json",
        _exp1543_payload(),
    )
    _write_json(
        root / "results" / "experiment_1544_thrml_diverse_topology_parity_n64.json",
        _exp1544_payload(),
    )
    (root / "research-hardware-wishlist.md").write_text(
        "No Extropic hardware access.", encoding="utf-8"
    )
    (root / "research-references.md").write_text(
        "Build an Extropic/Z1 access-readiness packet, not a hardware claim.",
        encoding="utf-8",
    )
    (root / "ops").mkdir(parents=True, exist_ok=True)
    (root / "ops" / "known-issues.md").write_text(
        "THRML/Carnot parity independent-RNG audit remains a .119 gate.",
        encoding="utf-8",
    )

    artifact = exp1545.run(repo_root=root, focused_checks_passed=True)

    out_path = root / "results" / "experiment_1545_extropic_z1_access_readiness_packet.json"
    packet_path = root / "ops" / "extropic_z1_readiness_packet.md"
    schema_path = root / "ops" / "extropic_z1_transcript_schema.json"

    assert out_path.exists()
    assert packet_path.exists()
    assert schema_path.exists()
    assert json.loads(out_path.read_text(encoding="utf-8")) == artifact
    assert artifact["extropic_z1_readiness_packet_ready"] is True
    assert artifact["benchmark_cases_included"] == 7
    assert artifact["no_hardware_execution_claim"] is True
    assert "no_hardware_execution_claim" in packet_path.read_text(encoding="utf-8")
    assert json.loads(schema_path.read_text(encoding="utf-8"))["title"] == (
        "Extropic Z1 authenticated benchmark transcript"
    )
