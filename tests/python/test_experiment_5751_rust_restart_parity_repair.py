"""Artifact contract tests for Exp5751 restart parity repair.

Spec coverage: REQ-SAMPLE-5751, SCENARIO-SAMPLE-5751.
"""

from __future__ import annotations

import json
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = REPO / "results/experiment_5751_rust_restart_parity_repair.json"
SPEC_PATH = REPO / "openspec/capabilities/samplers/spec.md"

REQUIRED_FIELDS = (
    "field_principles",
    "preconditions_checked",
    "spec_refs",
    "upstream_artifact_hashes",
    "rust_toolchain",
    "python_version",
    "pyo3_version",
    "release_build_receipt",
    "reproduced_failure_receipts",
    "first_divergence_receipt",
    "root_cause",
    "changed_files",
    "checkpoint_schema_before",
    "checkpoint_schema_after",
    "migration_receipt",
    "interruption_injection_manifest",
    "energy_parity",
    "proposal_parity",
    "scheduler_parity",
    "rng_parity",
    "restart_parity",
    "checkpoint_parity",
    "sample_count_parity",
    "distributional_parity",
    "fallback_equivalence",
    "production_backend_reachable",
    "restart_parity_ready_score",
    "timing_claimed",
    "hardware_speedup_claimed",
    "random_seeds",
    "test_commands",
    "test_exit_codes",
    "reproducibility_checksum",
    "honest_verdict",
)


def test_req_sample_5751_artifact_schema_and_no_speed_claims() -> None:
    """REQ-SAMPLE-5751: artifact fields, principles, and no-speed gates hold."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))

    assert tuple(payload) == REQUIRED_FIELDS
    assert sorted(payload["field_principles"]) == sorted(REQUIRED_FIELDS)
    assert payload["spec_refs"] == ["REQ-SAMPLE-5751", "SCENARIO-SAMPLE-5751"]
    assert payload["restart_parity_ready_score"] == 1.0
    assert payload["timing_claimed"] is False
    assert payload["hardware_speedup_claimed"] is False
    assert payload["honest_verdict"].startswith("complete:")
    assert payload["reproducibility_checksum"]


def test_scenario_sample_5751_artifact_records_failure_and_repair() -> None:
    """SCENARIO-SAMPLE-5751: original failure and repaired parity are recorded."""
    payload = json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))
    divergence = payload["first_divergence_receipt"]

    assert payload["reproduced_failure_receipts"][0]["reproduced"] is True
    assert divergence["size"] == 96
    assert divergence["field"] == "log_ratio"
    assert divergence["rust_legacy_json_value"] == "-0.0"
    assert divergence["python_json_value"] == "0.0"
    assert divergence["semantic_state_equal"] is True
    assert payload["restart_parity"]["all_repaired_suffix_hashes_match"] is True
    assert payload["checkpoint_parity"]["corrupted_checkpoint_rejected"] is True
    assert payload["fallback_equivalence"]["exact_fallback_equivalence"] is True


def test_req_sample_5751_spec_status_is_reconciled() -> None:
    """REQ-SAMPLE-5751: OpenSpec implementation status names artifact tests."""
    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("## Implementation Status (REQ-SAMPLE-5751)") :]
    section = section[: section.index("### REQ-SAMPLE-1746")]

    assert "Implemented" in section
    assert "results/experiment_5751_rust_restart_parity_repair.json" in section
    assert "tests/python/test_experiment_5751_rust_restart_parity_repair.py" in section
