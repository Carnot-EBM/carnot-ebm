"""Tests for Exp 1516 KAN/KAEM shape-normalization preflight.

Spec refs: REQ-KAN-1516, SCENARIO-KAN-1516.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_PYTHON_DIR = _PROJECT_ROOT / "python"
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
if str(_PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(_PYTHON_DIR))

import carnot.hardware.kan_shape_normalization_preflight as preflight  # noqa: E402
from carnot.hardware.kan_shape_normalization_preflight import (  # noqa: E402
    HARDWARE_ACCOUNTING_SHAPE_FIELDS,
    REQUIRED_ARTIFACT_FIELDS,
    artifact_has_required_fields,
    build_gated_artifact,
    build_normalized_shape_manifest,
    build_terminal_artifact,
    load_json,
    manifest_has_required_schema,
    run_preflight,
)


def _exp1502() -> dict[str, object]:
    return {
        "experiment": 1502,
        "status": "complete",
        "run_date": "20260507",
        "accounting_only_no_synthesis_claim": True,
        "hardware_claim_allowed": False,
        "accounting_table": [
            {
                "variant": "naive_full_precision_soskan",
                "rm_per_inference": 2547,
                "bop_per_inference": 81504,
                "nabs_per_inference": 2352,
                "memory_bytes": 13592,
                "lut_proxy": 27822,
                "bram36_blocks": 3,
                "accuracy_boundary": "full_precision_reference_only_no_hardware_measurement",
            },
            {
                "variant": "quantkan_3bit_lut_soskan",
                "rm_per_inference": 24,
                "bop_per_inference": 72,
                "nabs_per_inference": 75,
                "memory_bytes": 7065,
                "lut_proxy": 6298,
                "bram36_blocks": 2,
                "accuracy_boundary": "requires_empirical_auroc_gate_before_deployment",
            },
            {
                "variant": "kaem_univariate_table_approx",
                "rm_per_inference": 6,
                "bop_per_inference": 48,
                "nabs_per_inference": 9,
                "memory_bytes": 768,
                "lut_proxy": 240,
                "bram36_blocks": 1,
                "accuracy_boundary": "only_safe_for_separable_or_revalidated_verifier_features",
            },
        ],
        "naive_proxy_estimates": {
            "variant": "naive_full_precision_soskan",
            "rm_per_inference": 2547,
            "bop_per_inference": 81504,
            "nabs_per_inference": 2352,
            "memory_bytes": 13592,
            "lut_proxy": 27822,
            "bram36_blocks": 3,
            "accuracy_boundary": "full_precision_reference_only_no_hardware_measurement",
            "accuracy_reference": {
                "auroc_original": 0.9902,
                "source": "results/experiment_1148_metacluster_sos_kan_compression.json",
            },
        },
        "quantkan_proxy_estimates": {
            "variant": "quantkan_3bit_lut_soskan",
            "rm_per_inference": 24,
            "bop_per_inference": 72,
            "nabs_per_inference": 75,
            "memory_bytes": 7065,
            "lut_proxy": 6298,
            "bram36_blocks": 2,
            "accuracy_boundary": "requires_empirical_auroc_gate_before_deployment",
            "accuracy_reference": {
                "quantkan_3bit_auroc": 0.9801,
                "lut_kan_speedup": 2.5,
                "q3_model_bytes_from_exp1199_q4": 921,
                "table_bytes_from_exp1162": 6144,
                "source": "results/experiment_1266_quantkan_3bit_lut_kan.json",
            },
        },
        "kaem_proxy_estimates": {
            "variant": "kaem_univariate_table_approx",
            "rm_per_inference": 6,
            "bop_per_inference": 48,
            "nabs_per_inference": 9,
            "memory_bytes": 768,
            "lut_proxy": 240,
            "bram36_blocks": 1,
            "accuracy_boundary": "only_safe_for_separable_or_revalidated_verifier_features",
            "accuracy_reference": {
                "univariate_separable_assumption": True,
                "n_inputs": 3,
                "n_splines_in_sos_reference": 8,
                "n_lut_points": 256,
                "source": "python/carnot/models/kaem_energy.py",
            },
        },
        "source_artifacts": {
            "experiment_1162": "results/experiment_1162_kanele_sos_kan_fpga_blueprint.json",
            "experiment_1199": "results/experiment_1199_kantize_soskan_4bit_quantization.json",
            "experiment_1266": "results/experiment_1266_quantkan_3bit_lut_kan.json",
        },
        "blockers": [
            "quantkan_and_kaem_proxy_shapes_must_be_normalized_before_any_future_synthesis"
        ],
        "honest_verdict": "complete: kan hardware accounting ready; no synthesis or hardware claim",
    }


def _exp1506(prior_recorded: bool = True) -> dict[str, object]:
    return {
        "status": "complete",
        "run_date": "20260507",
        "prior_kan_shape_blocker_recorded": prior_recorded,
        "honest_verdict": "complete: milestone_116_activation_complete_115_archived_gate_fields_ready",
    }


def _exp1162() -> dict[str, object]:
    return {
        "sos_kan_n_inputs": 3,
        "sos_kan_k_splines": 8,
        "sos_kan_n_knots": 8,
        "sos_kan_rank": 4,
        "sos_kan_hidden_dim": 16,
        "n_lut_points": 256,
        "lut_storage_bytes": 6144,
        "rm_per_inference": 24,
        "bop_per_inference": 192,
        "nabs_per_inference": 75,
    }


def _exp1199() -> dict[str, object]:
    return {
        "soskan_4bit_size_mb": 0.001228,
        "soskan_4bit_auroc": 0.990137,
        "soskan_4bit_inference_latency_ms": 0.038038,
    }


def _exp1266() -> dict[str, object]:
    return {
        "quantkan_3bit_auroc": 0.9801,
        "lut_kan_speedup": 2.5,
        "lut_table_size_kb": 12.5,
        "simulation": {
            "lut_grid_points": 256,
            "lut_value_dtype": "int8",
            "direct_latency_ns": 1000.0,
            "lut_latency_ns": 400.0,
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n")


def test_manifest_maps_proxy_shapes_to_hardware_accounting_shapes() -> None:
    """REQ-KAN-1516: proxy, model, and hardware-accounting dimensions are mapped."""
    manifest = build_normalized_shape_manifest(
        exp1502=_exp1502(),
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        run_date="20260508",
    )

    assert manifest_has_required_schema(manifest)
    assert manifest["no_synthesis_claim"] is True
    assert manifest["no_board_claim"] is True
    assert manifest["hardware_accounting_shape_fields"] == HARDWARE_ACCOUNTING_SHAPE_FIELDS
    assert {item["variant"] for item in manifest["normalized_shapes"]} == {
        "naive_full_precision_soskan",
        "quantkan_3bit_lut_soskan",
        "kaem_univariate_table_approx",
    }

    quantkan = next(
        item
        for item in manifest["normalized_shapes"]
        if item["variant"] == "quantkan_3bit_lut_soskan"
    )
    assert quantkan["model_shape"]["feature_dim"]["value"] == 3
    assert quantkan["proxy_dimensions"]["quantization_bits"]["value"] == 3
    assert quantkan["proxy_dimensions"]["lut_grid_points"]["value"] == 256
    assert quantkan["hardware_accounting_shape"]["rm_per_inference"]["value"] == 24
    assert quantkan["hardware_accounting_shape"]["rm_per_inference"]["provenance"]

    kaem = next(
        item
        for item in manifest["normalized_shapes"]
        if item["variant"] == "kaem_univariate_table_approx"
    )
    assert kaem["proxy_dimensions"]["separable_univariate_tables"]["value"] is True
    assert kaem["excluded_assumptions"] == [
        "cross_feature_interactions_preserved_by_univariate_kaem_proxy"
    ]
    assert "batch_size_gt_1" in {
        item["assumption"] for item in manifest["excluded_shape_assumptions"]
    }
    assert "token_sequence_length" in {
        item["assumption"] for item in manifest["excluded_shape_assumptions"]
    }


def test_terminal_artifact_requires_manifest_and_claim_gates() -> None:
    """REQ-KAN-1516: terminal artifacts expose required fields and no-claim gates."""
    manifest = build_normalized_shape_manifest(
        exp1502=_exp1502(),
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        run_date="20260508",
    )

    artifact = build_terminal_artifact(manifest=manifest, duration_s=0.25)

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact_has_required_fields(artifact)
    assert artifact["status"] == "complete"
    assert artifact["kan_shape_manifest_ready"] is True
    assert artifact["gated_inputs_present"] is True
    assert artifact["no_synthesis_claim"] is True
    assert artifact["no_board_claim"] is True
    assert artifact["proxy_shapes_loaded"] is True
    assert artifact["normalized_shapes_written"] is True
    assert artifact["honest_verdict"].startswith("complete:")

    artifact["no_synthesis_claim"] = False
    assert not artifact_has_required_fields(artifact)


def test_missing_prior_blocker_writes_terminal_gated_artifact(tmp_path: Path) -> None:
    """REQ-KAN-1516: absent Exp 1506 blocker record prevents manifest readiness."""
    exp1502 = tmp_path / "exp1502.json"
    exp1506 = tmp_path / "exp1506.json"
    exp1162 = tmp_path / "exp1162.json"
    exp1199 = tmp_path / "exp1199.json"
    exp1266 = tmp_path / "exp1266.json"
    manifest_path = tmp_path / "manifest.json"
    deliverable_path = tmp_path / "experiment_1516.json"
    _write_json(exp1502, _exp1502())
    _write_json(exp1506, _exp1506(prior_recorded=False))
    _write_json(exp1162, _exp1162())
    _write_json(exp1199, _exp1199())
    _write_json(exp1266, _exp1266())

    artifact = run_preflight(
        exp1502_path=exp1502,
        exp1506_path=exp1506,
        exp1162_path=exp1162,
        exp1199_path=exp1199,
        exp1266_path=exp1266,
        manifest_path=manifest_path,
        deliverable_path=deliverable_path,
    )

    assert artifact == json.loads(deliverable_path.read_text())
    assert artifact["status"] == "blocked"
    assert artifact["kan_shape_manifest_ready"] is False
    assert artifact["gated_inputs_present"] is False
    assert artifact["no_synthesis_claim"] is True
    assert artifact["no_board_claim"] is True
    assert not manifest_path.exists()
    assert artifact_has_required_fields(artifact)


def test_run_preflight_writes_manifest_and_deliverable(tmp_path: Path) -> None:
    """SCENARIO-KAN-1516: runner writes normalized manifest and terminal artifact."""
    exp1502 = tmp_path / "exp1502.json"
    exp1506 = tmp_path / "exp1506.json"
    exp1162 = tmp_path / "exp1162.json"
    exp1199 = tmp_path / "exp1199.json"
    exp1266 = tmp_path / "exp1266.json"
    manifest_path = tmp_path / "manifest.json"
    deliverable_path = tmp_path / "experiment_1516.json"
    _write_json(exp1502, _exp1502())
    _write_json(exp1506, _exp1506())
    _write_json(exp1162, _exp1162())
    _write_json(exp1199, _exp1199())
    _write_json(exp1266, _exp1266())

    artifact = run_preflight(
        exp1502_path=exp1502,
        exp1506_path=exp1506,
        exp1162_path=exp1162,
        exp1199_path=exp1199,
        exp1266_path=exp1266,
        manifest_path=manifest_path,
        deliverable_path=deliverable_path,
    )

    manifest = json.loads(manifest_path.read_text())
    payload = json.loads(deliverable_path.read_text())
    assert payload == artifact
    assert manifest_has_required_schema(manifest)
    assert artifact_has_required_fields(payload)
    assert payload["shape_manifest_path"] == str(manifest_path)
    assert payload["hardware_accounting_shape_fields"] == HARDWARE_ACCOUNTING_SHAPE_FIELDS
    assert payload["excluded_shape_assumptions"] == manifest["excluded_shape_assumptions"]


def test_invalid_manifest_inputs_and_claim_drift_fail_clearly() -> None:
    """REQ-KAN-1516: malformed accounting shapes and claim drift are rejected."""
    bad_exp1502 = _exp1502()
    bad_exp1502.pop("accounting_table")

    with pytest.raises(ValueError, match="missing Exp 1502 field: accounting_table"):
        build_normalized_shape_manifest(
            exp1502=bad_exp1502,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    gated = build_gated_artifact("missing_prior_kan_shape_blocker_recorded")
    assert artifact_has_required_fields(gated)
    assert gated["kan_shape_manifest_ready"] is False

    manifest = build_normalized_shape_manifest(
        exp1502=_exp1502(),
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        run_date="20260508",
    )
    manifest["future_synthesis_claim_gate"]["future_synthesis_claim_allowed"] = True
    assert not manifest_has_required_schema(manifest)


def test_loader_and_source_shape_validation_errors_are_explicit(tmp_path: Path) -> None:
    """REQ-KAN-1516: source-shape loader failures name the malformed field."""
    bad_array = tmp_path / "bad_array.json"
    bad_array.write_text("[]")
    with pytest.raises(ValueError, match="expected JSON object"):
        load_json(bad_array)

    bad_exp1502 = _exp1502()
    bad_exp1502["quantkan_proxy_estimates"] = "bad"
    with pytest.raises(ValueError, match="expected Exp 1502.quantkan_proxy_estimates"):
        build_normalized_shape_manifest(
            exp1502=bad_exp1502,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    bad_exp1502 = _exp1502()
    bad_exp1502.pop("quantkan_proxy_estimates")
    with pytest.raises(ValueError, match="missing Exp 1502 field: quantkan_proxy_estimates"):
        build_normalized_shape_manifest(
            exp1502=bad_exp1502,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    bad_exp1502 = _exp1502()
    bad_exp1502["accounting_table"] = {}
    with pytest.raises(ValueError, match="expected Exp 1502.accounting_table to be a list"):
        build_normalized_shape_manifest(
            exp1502=bad_exp1502,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    bad_exp1162 = _exp1162()
    bad_exp1162.pop("sos_kan_n_inputs")
    with pytest.raises(ValueError, match="missing Exp 1162 field: sos_kan_n_inputs"):
        build_normalized_shape_manifest(
            exp1502=_exp1502(),
            exp1506=_exp1506(),
            exp1162=bad_exp1162,
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )


def test_accounting_variant_validation_catches_missing_rows_and_drift() -> None:
    """REQ-KAN-1516: variant rows and proxy estimates must remain in sync."""
    missing_variant = _exp1502()
    missing_variant["accounting_table"] = [
        row
        for row in missing_variant["accounting_table"]
        if row["variant"] != "kaem_univariate_table_approx"
    ]
    with pytest.raises(ValueError, match="missing Exp 1502 accounting_table variant"):
        build_normalized_shape_manifest(
            exp1502=missing_variant,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    missing_field = _exp1502()
    missing_field["naive_proxy_estimates"].pop("rm_per_inference")
    with pytest.raises(ValueError, match="missing Exp 1502 naive_proxy_estimates field"):
        build_normalized_shape_manifest(
            exp1502=missing_field,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    drifted = _exp1502()
    drifted["accounting_table"][0]["rm_per_inference"] = 999
    with pytest.raises(ValueError, match="disagrees with proxy estimate"):
        build_normalized_shape_manifest(
            exp1502=drifted,
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )


def test_input_gate_failures_and_fallback_provenance_are_explicit() -> None:
    """REQ-KAN-1516: all prerequisite gates fail closed before manifest readiness."""
    with pytest.raises(ValueError, match="prior_kan_shape_blocker_recorded"):
        build_normalized_shape_manifest(
            exp1502=_exp1502(),
            exp1506=_exp1506(prior_recorded=False),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )

    for field, message, value in [
        ("status", "not complete", "running"),
        ("accounting_only_no_synthesis_claim", "no-synthesis accounting gate", False),
        ("hardware_claim_allowed", "hardware claim gate", True),
    ]:
        bad_exp1502 = _exp1502()
        bad_exp1502[field] = value
        with pytest.raises(ValueError, match=message):
            build_normalized_shape_manifest(
                exp1502=bad_exp1502,
                exp1506=_exp1506(),
                exp1162=_exp1162(),
                exp1199=_exp1199(),
                exp1266=_exp1266(),
                run_date="20260508",
            )

    no_sources = _exp1502()
    no_sources.pop("source_artifacts")
    manifest = build_normalized_shape_manifest(
        exp1502=no_sources,
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199={},
        exp1266={},
        run_date="20260508",
    )
    assert manifest_has_required_schema(manifest)
    assert manifest["quantization_assumptions"][1]["normalized_value"] is None


def test_schema_validators_reject_each_claim_drift_shape() -> None:
    """REQ-KAN-1516: validators reject incomplete manifests and artifacts."""
    manifest = build_normalized_shape_manifest(
        exp1502=_exp1502(),
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        run_date="20260508",
    )
    manifest_mutations = [
        ("status", "blocked"),
        ("kan_shape_manifest_ready", False),
        ("no_synthesis_claim", False),
        ("hardware_accounting_shape_fields", []),
        ("future_synthesis_claim_gate", "bad"),
        ("excluded_shape_assumptions", []),
        ("normalized_shapes", []),
        ("honest_verdict", "not_terminal"),
    ]
    for key, value in manifest_mutations:
        candidate = json.loads(json.dumps(manifest))
        candidate[key] = value
        assert not manifest_has_required_schema(candidate)

    candidate = json.loads(json.dumps(manifest))
    candidate["future_synthesis_claim_gate"]["shape_provenance_explicit"] = False
    assert not manifest_has_required_schema(candidate)

    shape_mutations = [
        "not_a_dict",
        {"synthesis_claim_ready": True, "hardware_accounting_shape": {}},
        {"synthesis_claim_ready": False, "hardware_accounting_shape": "bad"},
        {"synthesis_claim_ready": False, "hardware_accounting_shape": {}},
    ]
    for shape in shape_mutations:
        candidate = json.loads(json.dumps(manifest))
        candidate["normalized_shapes"] = [shape]
        assert not manifest_has_required_schema(candidate)

    candidate = json.loads(json.dumps(manifest))
    del candidate["normalized_shapes"][0]["hardware_accounting_shape"]["rm_per_inference"]["value"]
    assert not manifest_has_required_schema(candidate)

    candidate = json.loads(json.dumps(manifest))
    candidate["normalized_shapes"][0]["hardware_accounting_shape"]["rm_per_inference"][
        "provenance"
    ] = []
    assert not manifest_has_required_schema(candidate)

    artifact = build_terminal_artifact(manifest=manifest, duration_s=0.1)
    for key, value in [
        ("honest_verdict", "not_terminal"),
        ("no_board_claim", False),
        ("hardware_accounting_shape_fields", []),
        ("status", "weird"),
    ]:
        candidate = dict(artifact)
        candidate[key] = value
        assert not artifact_has_required_fields(candidate)

    missing = dict(artifact)
    missing.pop("status")
    assert not artifact_has_required_fields(missing)


def test_defensive_terminal_builder_errors_are_covered(monkeypatch: pytest.MonkeyPatch) -> None:
    """REQ-KAN-1516: defensive builder checks fail closed if validation drifts."""
    manifest = build_normalized_shape_manifest(
        exp1502=_exp1502(),
        exp1506=_exp1506(),
        exp1162=_exp1162(),
        exp1199=_exp1199(),
        exp1266=_exp1266(),
        run_date="20260508",
    )
    bad_manifest = dict(manifest)
    bad_manifest["status"] = "bad"
    with pytest.raises(ValueError, match="invalid shape manifest"):
        build_terminal_artifact(manifest=bad_manifest, duration_s=0.1)

    monkeypatch.setattr(preflight, "artifact_has_required_fields", lambda _artifact: False)
    with pytest.raises(RuntimeError, match="terminal artifact is missing"):
        build_terminal_artifact(manifest=manifest, duration_s=0.1)


def test_defensive_manifest_builder_validation_error_is_covered(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-KAN-1516: manifest construction fails closed if final validation fails."""
    monkeypatch.setattr(preflight, "manifest_has_required_schema", lambda _manifest: False)
    with pytest.raises(RuntimeError, match="shape manifest is missing"):
        preflight.build_normalized_shape_manifest(
            exp1502=_exp1502(),
            exp1506=_exp1506(),
            exp1162=_exp1162(),
            exp1199=_exp1199(),
            exp1266=_exp1266(),
            run_date="20260508",
        )
