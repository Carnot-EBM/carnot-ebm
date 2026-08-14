"""Tests for Exp6415 exact Boolean WCSP CCG kernelization.

Spec refs: REQ-CONSTRAINT-VERIFY-6415,
SCENARIO-CONSTRAINT-VERIFY-6415-EXACT-PRESERVATION,
SCENARIO-CONSTRAINT-VERIFY-6415-ATTACKS,
SCENARIO-CONSTRAINT-VERIFY-6415-NO-SPEEDUP-CLAIM.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6415_boolean_wcsp_ccg_kernelization as mod


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec/capabilities/constraint-verification/spec.md"


def test_req_constraint_verify_6415_spec_declares_contract_fields() -> None:
    """REQ-CONSTRAINT-VERIFY-6415: OpenSpec anchors schema, fields, and gates."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-CONSTRAINT-VERIFY-6415") :]
    normalized = " ".join(section.split())

    for marker in (
        mod.MODULE_RELATIVE_PATH.as_posix(),
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "SCENARIO-CONSTRAINT-VERIFY-6415-EXACT-PRESERVATION",
        "SCENARIO-CONSTRAINT-VERIFY-6415-ATTACKS",
        "SCENARIO-CONSTRAINT-VERIFY-6415-NO-SPEEDUP-CLAIM",
        "at least 48 small instances",
        "verifier_is_oracle",
        "quantum_advantage_claimed",
        "hardware_speedup_claimed",
    ):
        assert marker in section

    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_req_constraint_verify_6415_wcsp_schema_is_canonical_and_reversible() -> None:
    """REQ-CONSTRAINT-VERIFY-6415: duplicate and negative costs canonicalize exactly."""

    raw = {
        "instance_id": "schema_duplicate_negative",
        "n_variables": 2,
        "terms": [
            {"term_id": "u0a", "scope": [0], "costs": {"0": 0, "1": -3}},
            {"term_id": "u0b", "scope": [0], "costs": {"0": 2, "1": 5}},
            {"term_id": "p01", "scope": [1, 0], "costs": {"00": 7, "01": 11, "10": 13, "11": 17}},
            {"term_id": "const", "scope": [], "costs": {"": -4}},
        ],
        "classes": ["adversarial_weight"],
        "seed": 6415,
    }

    instance = mod.BooleanWCSP.from_mapping(raw)
    reversed_instance = mod.BooleanWCSP.from_mapping(
        {**raw, "terms": list(reversed(raw["terms"]))}
    )

    assert instance.canonical_terms == reversed_instance.canonical_terms
    assert instance.canonical_hash() == reversed_instance.canonical_hash()
    assert instance.evaluate({0: 1, 1: 0}) == 9
    assert instance.source_mapping["p01"]["source_scope"] == [1, 0]
    assert instance.source_mapping["p01"]["canonical_scope"] == [0, 1]

    restored = mod.BooleanWCSP.from_mapping(instance.to_json())
    assert restored.canonical_hash() == instance.canonical_hash()

    too_large = deepcopy(raw)
    too_large["terms"][0]["costs"]["1"] = mod.MAX_ABS_COST + 1
    with pytest.raises(ValueError, match="cost bound"):
        mod.BooleanWCSP.from_mapping(too_large)


def test_scenario_constraint_verify_6415_kernelizer_fixes_only_certified_variables() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6415-EXACT-PRESERVATION: certificates prove fixes."""

    instance = mod.BooleanWCSP.from_mapping(
        {
            "instance_id": "forced_chain_with_free_tail",
            "n_variables": 3,
            "terms": [
                {"term_id": "force0", "scope": [0], "costs": {"0": 7, "1": 0}},
                {"term_id": "tie01", "scope": [0, 1], "costs": {"00": 0, "01": 5, "10": 5, "11": 0}},
            ],
            "classes": ["unary", "pairwise"],
            "seed": 6415,
        }
    )

    source = mod.exhaustive_reference(instance)
    result = mod.kernelize_with_ccg(instance, source)
    checks = mod.independent_certificate_checks(instance, result, source)
    completion = mod.exact_completion(instance, result.fixed_assignments)

    assert source["optimum_cost"] == 0
    assert result.fixed_assignments == {0: 1, 1: 1}
    assert 2 not in result.fixed_assignments
    assert all(row["passed"] for row in checks)
    assert completion["optimum_cost"] == source["optimum_cost"]
    assert completion["verifier_calls"] == 2

    ccg = mod.build_ccg(instance)
    assert mod.validate_ccg_contract(instance, ccg) is True

    missing_aux = deepcopy(ccg.to_json())
    missing_aux["nodes"] = [node for node in missing_aux["nodes"] if node["kind"] != "auxiliary_term"]
    with pytest.raises(ValueError, match="auxiliary"):
        mod.validate_ccg_contract(instance, mod.CCG.from_json(missing_aux))


def test_req_constraint_verify_6415_defensive_validators_cover_malformed_inputs() -> None:
    """REQ-CONSTRAINT-VERIFY-6415: malformed schemas and CCG mappings fail closed."""

    valid = {
        "instance_id": "validator_base",
        "n_variables": 2,
        "terms": [{"term_id": "force0", "scope": [0], "costs": {"0": 2, "1": 0}}],
        "classes": ["attack"],
        "seed": 6415,
    }
    for expected, patch in (
        ("n_variables bound", lambda data: data.__setitem__("n_variables", 0)),
        ("duplicate term_id", lambda data: data["terms"].append(dict(data["terms"][0]))),
        ("scope arity", lambda data: data["terms"][0].__setitem__("scope", [0, 0])),
        ("scope variable", lambda data: data["terms"][0].__setitem__("scope", [9])),
        ("cost table shape", lambda data: data["terms"][0].__setitem__("costs", {"0": 0})),
    ):
        raw = deepcopy(valid)
        patch(raw)
        with pytest.raises(ValueError, match=expected):
            mod.BooleanWCSP.from_mapping(raw)

    instance = mod.BooleanWCSP.from_mapping(
        {
            "instance_id": "validator_constant",
            "n_variables": 2,
            "terms": [
                {"term_id": "const", "scope": [], "costs": {"": -1}},
                {"term_id": "force0", "scope": [0], "costs": {"0": 2, "1": 0}},
                {"term_id": "eq01", "scope": [0, 1], "costs": {"00": 0, "01": 1, "10": 1, "11": 0}},
            ],
            "classes": ["attack"],
            "seed": 6415,
        }
    )
    assert mod.build_ccg(instance).graph_cut_constant == -1
    with pytest.raises(ValueError, match="assignment label"):
        instance.evaluate({0: 2, 1: 0})
    with pytest.raises(ValueError, match="no feasible assignment"):
        mod.exhaustive_reference(instance, {0: 2})
    assert mod._raises(lambda: None) is False

    ccg = mod.build_ccg(instance)
    no_source = deepcopy(ccg.to_json())
    no_source["nodes"] = [node for node in no_source["nodes"] if node["node_id"] != "source"]
    with pytest.raises(ValueError, match="source sink"):
        mod.validate_ccg_contract(instance, mod.CCG.from_json(no_source))

    no_variable = deepcopy(ccg.to_json())
    no_variable["nodes"] = [node for node in no_variable["nodes"] if node["node_id"] != "var:1"]
    with pytest.raises(ValueError, match="variable node"):
        mod.validate_ccg_contract(instance, mod.CCG.from_json(no_variable))

    bad_endpoint = deepcopy(ccg.to_json())
    bad_endpoint["graph_cut_edges"][0]["to"] = "var:99"
    with pytest.raises(ValueError, match="edge endpoint"):
        mod.validate_ccg_contract(instance, mod.CCG.from_json(bad_endpoint))

    bad_energy = deepcopy(ccg.to_json())
    bad_energy["graph_cut_edges"][0]["capacity"] += 1
    with pytest.raises(ValueError, match="graph energy mapping"):
        mod.validate_ccg_contract(instance, mod.CCG.from_json(bad_energy))


def test_scenario_constraint_verify_6415_attack_matrix_blocks_unsound_reductions() -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6415-ATTACKS: each attack has a failing control path."""

    attacks = mod.run_attack_matrix()
    expected = {
        "sign_inversion",
        "zero_negative_weights",
        "duplicate_constraints",
        "disconnected_components",
        "auxiliary_node_omission",
        "mapping_reversal",
        "integer_overflow",
        "unsound_fixed_variable",
        "nonunique_optima",
    }

    assert {row["attack_id"] for row in attacks} == expected
    assert all(row["passed"] is True for row in attacks)
    assert any(row["mechanism"] == "abstained_non_submodular_ccg" for row in attacks)


def test_scenario_constraint_verify_6415_artifact_is_complete_and_gated(tmp_path: Path) -> None:
    """SCENARIO-CONSTRAINT-VERIFY-6415-NO-SPEEDUP-CLAIM: artifact is exact-only."""

    output = tmp_path / mod.RESULT_RELATIVE_PATH.name
    manifest = tmp_path / "experiment_6415_boolean_wcsp_frozen_manifest.json"
    artifact = mod.write_artifact(
        output_path=output,
        manifest_path=manifest,
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run=["focused-exp6415"],
    )
    loaded = json.loads(output.read_text(encoding="utf-8"))

    assert loaded == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) == set(artifact)
    assert mod.validate_artifact(artifact) is True
    assert artifact["status"] == "complete_ready"
    assert artifact["honest_verdict"].startswith("complete_ready:")
    assert artifact["optimum_preservation_rate"] == 1.0
    assert artifact["ccg_kernelization_exact_ready_score"] == 1.0
    assert artifact["quantum_advantage_claimed"] is False
    assert artifact["hardware_speedup_claimed"] is False
    assert artifact["verifier_is_oracle"]["value"] is True
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["frozen_manifest_path_hash_counts_classes_and_seeds"]["total_instances"] >= 48
    assert artifact["frozen_manifest_path_hash_counts_classes_and_seeds"]["sha256"] == mod.sha256_file(manifest)
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)


def test_req_constraint_verify_6415_artifact_mutations_fail_closed(tmp_path: Path) -> None:
    """REQ-CONSTRAINT-VERIFY-6415: readiness rejects missing evidence and bad claims."""

    artifact = mod.write_artifact(
        output_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        manifest_path=tmp_path / "manifest.json",
        root=REPO,
        run_date="20260814",
        duration_s=0.0,
        tests_run=["focused-exp6415"],
    )
    mutations = [
        ("required field", lambda data: data.pop("field_principles")),
        ("required field set", lambda data: data.__setitem__("extra", True)),
        ("field_principles", lambda data: data.__setitem__("field_principles", {})),
        ("optimum_preservation_rate", lambda data: data.__setitem__("optimum_preservation_rate", 0.99)),
        ("quantum_advantage_claimed", lambda data: data.__setitem__("quantum_advantage_claimed", True)),
        ("hardware_speedup_claimed", lambda data: data.__setitem__("hardware_speedup_claimed", True)),
        ("verifier_is_oracle", lambda data: data["verifier_is_oracle"].__setitem__("kernelizer_is_oracle", True)),
        ("attack_matrix", lambda data: data["sign_weight_duplicate_component_auxiliary_mapping_overflow_fixed_variable_and_nonunique_attack_matrix"][0].__setitem__("passed", False)),
        ("ccg_kernelization_exact_ready_score", lambda data: data.__setitem__("ccg_kernelization_exact_ready_score", 0.5)),
        ("status", lambda data: data.__setitem__("status", "bad")),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        (
            "ccg_kernelization_exact_ready_score",
            lambda data: (
                data["fixed_variable_certificates_and_independent_checks"].__setitem__(
                    "all_passed", False
                ),
                data.__setitem__("ccg_kernelization_exact_ready_score", 0.0),
                data.__setitem__("status", "blocked"),
                data.__setitem__(
                    "honest_verdict",
                    "blocked: exact Boolean WCSP CCG kernelization readiness gates failed.",
                ),
            ),
        ),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "sha256:bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {
            "required field",
            "required field set",
            "field_principles",
            "attack_matrix",
            "ccg_kernelization_exact_ready_score",
            "status",
            "honest_verdict",
            "reproducibility_checksum",
        }:
            bad["ccg_kernelization_exact_ready_score"] = mod.ready_score(bad)
            bad["status"] = mod.status(bad)
            bad["honest_verdict"] = mod.honest_verdict(bad)
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        elif expected not in {"required field", "reproducibility_checksum"}:
            bad["reproducibility_checksum"] = mod.reproducibility_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)
