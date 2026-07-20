"""Tests for Exp5745 ARC causal gate schema corrigendum.

Spec refs: REQ-ARC-WMTE-5745,
SCENARIO-ARC-WMTE-5745-NEGATIVE-CONTROL-NORMALIZATION,
SCENARIO-ARC-WMTE-5745-SCALAR-COVERAGE-GATE,
SCENARIO-ARC-WMTE-5745-HASH-LINKED-NO-CREDIT-CONTRACT.
"""

from __future__ import annotations

from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


REPO = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO / "python/carnot/experiment_5745_arc_causal_gate_schema_corrigendum.py"
SPEC = importlib.util.spec_from_file_location(
    "carnot.experiment_5745_arc_causal_gate_schema_corrigendum", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)

SPEC_PATH = REPO / "openspec/capabilities/arc-world-model-trust-energy/spec.md"
SOURCE_PATH = REPO / mod.SOURCE_ARTIFACT_RELATIVE_PATH
RESULT_PATH = REPO / mod.RESULT_RELATIVE_PATH
SCRIPTS_DIR = REPO / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from conductor_gates import evaluate_gates  # noqa: E402


def _source_artifact() -> dict[str, Any]:
    return json.loads(SOURCE_PATH.read_text(encoding="utf-8"))


def test_req_arc_wmte_5745_spec_declares_corrigendum_contract() -> None:
    """REQ-ARC-WMTE-5745: OpenSpec lists the scalar gate fields and principles."""

    spec = SPEC_PATH.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-WMTE-5745") :]
    section = section[: section.index("### REQ-ARC-WMTE-4738")]
    normalized = " ".join(section.split())

    for marker in (
        str(mod.RESULT_RELATIVE_PATH),
        "without rerunning primitive mining",
        "detected-and-rejected leakage canaries",
        "counterfactual_receipt_coverage_score=1.0",
        "SCENARIO-ARC-WMTE-5745-NEGATIVE-CONTROL-NORMALIZATION",
        "SCENARIO-ARC-WMTE-5745-SCALAR-COVERAGE-GATE",
        "SCENARIO-ARC-WMTE-5745-HASH-LINKED-NO-CREDIT-CONTRACT",
    ):
        assert marker in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_arc_wmte_5745_builds_hash_linked_scalar_artifact(
    tmp_path: Path,
) -> None:
    """SCENARIO-ARC-WMTE-5745-HASH-LINKED-NO-CREDIT-CONTRACT."""

    artifact = mod.build_artifact(
        root=REPO,
        test_commands=["unit: exp5745"],
        test_exit_codes={"unit: exp5745": 0},
    )
    saved_path = mod.write_output(tmp_path, artifact)
    saved = json.loads(saved_path.read_text(encoding="utf-8"))

    assert saved == artifact
    assert set(saved["field_principles"]) == set(saved)
    assert saved["source_artifact_path"] == str(mod.SOURCE_ARTIFACT_RELATIVE_PATH)
    assert saved["source_artifact_hash"] == mod.EXPECTED_SOURCE_ARTIFACT_HASH
    assert saved["source_schema_version"] == mod.SOURCE_SCHEMA_VERSION
    assert saved["normalized_schema_version"] == mod.NORMALIZED_SCHEMA_VERSION
    assert saved["positive_causal_primitive_count"] == 7
    assert saved["frozen_primitive_ids"] == list(mod.FROZEN_PRIMITIVE_IDS)
    assert saved["frozen_effect_hash"] == mod.EXPECTED_FROZEN_EFFECT_HASH
    assert (
        saved["original_counterfactual_receipt_coverage"]
        == _source_artifact()["counterfactual_receipt_coverage"]
    )
    assert saved["counterfactual_receipt_coverage_score"] == 1.0
    assert saved["detected_source_leak_canary_count"] == 1
    assert saved["detected_game_identity_leak_canary_count"] == 2
    assert saved["admitted_source_leak_count"] == 0
    assert saved["admitted_game_identity_leak_count"] == 0
    assert saved["registry_precheck"]["public_game_count"] == 25
    assert saved["registry_precheck"]["reproducible_total_levels"] == 183
    assert saved["registry_precheck"]["completed_levels"] == 183
    assert saved["registry_precheck"]["all_public_games_complete"] is True
    assert saved["solve_provenance"] == "development_proxy"
    assert saved["arc_registry_delta"] == 0
    assert saved["arc_solve_credited"] is False
    assert saved["science_rerun"] is False
    assert saved["live_policy_modified"] is False
    assert saved["preconditions_checked"]["source_artifact_hash_verified"] is True
    assert saved["preconditions_checked"]["trace_manifest_hashes_verified"] is True
    assert saved["preconditions_checked"]["primitive_count_verified"] is True
    assert saved["preconditions_checked"]["deletion_effect_hash_verified"] is True
    assert saved["preconditions_checked"]["paired_replay_count_verified"] is True
    assert saved["preconditions_checked"]["exact_replay_receipts_verified"] is True
    assert saved["preconditions_checked"]["scripts_research_conductor_modified"] is False
    assert saved["reproducibility_checksum"] == mod.payload_checksum(saved)
    assert saved["honest_verdict"].startswith("complete:")
    mod.validate_artifact(saved)


def test_scenario_arc_wmte_5745_detected_rejected_canaries_are_not_admitted() -> None:
    """SCENARIO-ARC-WMTE-5745-NEGATIVE-CONTROL-NORMALIZATION."""

    source = _source_artifact()
    counts = mod.derive_leak_counts(source)

    assert counts == {
        "detected_source_leak_canary_count": 1,
        "detected_game_identity_leak_canary_count": 2,
        "admitted_source_leak_count": 0,
        "admitted_game_identity_leak_count": 0,
    }

    admitted_source = deepcopy(source)
    admitted_source["primitive_candidates"][0]["admitted_leak_classes"] = ["source"]
    admitted_counts = mod.derive_leak_counts(admitted_source)
    assert admitted_counts["detected_source_leak_canary_count"] == 1
    assert admitted_counts["admitted_source_leak_count"] == 1

    unrejected = deepcopy(source)
    for row in unrejected["negative_controls"]:
        if row["control"] == "source_derived_rule":
            row["rejected"] = False
    with pytest.raises(ValueError, match="negative_controls"):
        mod.derive_leak_counts(unrejected)


def test_scenario_arc_wmte_5745_object_coverage_normalizes_deterministically() -> None:
    """SCENARIO-ARC-WMTE-5745-SCALAR-COVERAGE-GATE."""

    source = _source_artifact()

    assert mod.normalize_counterfactual_coverage(source, root=REPO) == 1.0
    assert mod.verify_trace_manifest_hashes(source, root=REPO)["verified"] is True

    too_few_replays = deepcopy(source)
    for row in too_few_replays["primitive_candidates"]:
        if row["causal_retained"]:
            row["paired_replay_count"] = mod.MIN_PAIRED_REPLAYS - 1
            break
    with pytest.raises(ValueError, match="paired_replay"):
        mod.normalize_counterfactual_coverage(too_few_replays, root=REPO)

    bad_trace = deepcopy(source)
    bad_trace["trace_manifest"][0]["sha256"] = "sha256:" + "0" * 64
    with pytest.raises(ValueError, match="trace_manifest"):
        mod.verify_trace_manifest_hashes(bad_trace, root=REPO)


def test_req_arc_wmte_5745_downstream_conductor_gates_pass(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5745: normalized scalar fields satisfy the intended Exp5753 gate."""

    artifact = mod.build_artifact(root=REPO)
    mod.write_output(tmp_path, artifact)
    task = {
        "id": "exp5753-arc-generic-primitive-live-registry-ab",
        "gated_on": [
            {
                "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                "artifact_field": "counterfactual_receipt_coverage_score",
                "op": ">=",
                "value": 1.0,
            },
            {
                "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                "artifact_field": "admitted_source_leak_count",
                "op": "==",
                "value": 0,
            },
            {
                "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                "artifact_field": "admitted_game_identity_leak_count",
                "op": "==",
                "value": 0,
            },
            {
                "upstream": "exp5745-arc-causal-gate-schema-corrigendum",
                "artifact_field": "positive_causal_primitive_count",
                "op": ">=",
                "value": 1,
            },
        ],
    }

    gate_result = evaluate_gates(task, results_dir=tmp_path)

    assert gate_result.passed is True
    assert gate_result.summary == "4 gate(s) satisfied"


def test_req_arc_wmte_5745_validation_rejects_manual_overclaims() -> None:
    """REQ-ARC-WMTE-5745: schema, provenance, registry, and checksum edits fail closed."""

    artifact = mod.build_artifact(root=REPO)
    mutations = [
        ("missing required fields", lambda data: data.pop("field_principles")),
        ("field_principles missing", lambda data: data.__setitem__("unprincipled_extra", True)),
        ("field_principles", lambda data: data["field_principles"].__setitem__("bad", "bad")),
        ("source_artifact_hash", lambda data: data.__setitem__("source_artifact_hash", "bad")),
        ("source_schema_version", lambda data: data.__setitem__("source_schema_version", "bad")),
        (
            "normalized_schema_version",
            lambda data: data.__setitem__("normalized_schema_version", "bad"),
        ),
        (
            "positive_causal_primitive_count",
            lambda data: data.__setitem__("positive_causal_primitive_count", 6),
        ),
        ("frozen_primitive_ids", lambda data: data.__setitem__("frozen_primitive_ids", [])),
        ("frozen_effect_hash", lambda data: data.__setitem__("frozen_effect_hash", "bad")),
        (
            "original_counterfactual_receipt_coverage",
            lambda data: data.__setitem__("original_counterfactual_receipt_coverage", {}),
        ),
        ("counterfactual_receipt_coverage_score", lambda data: data.__setitem__("counterfactual_receipt_coverage_score", 0.0)),
        (
            "detected_source_leak_canary_count",
            lambda data: data.__setitem__("detected_source_leak_canary_count", 0),
        ),
        (
            "detected_game_identity_leak_canary_count",
            lambda data: data.__setitem__("detected_game_identity_leak_canary_count", 0),
        ),
        ("admitted_source_leak_count", lambda data: data.__setitem__("admitted_source_leak_count", 1)),
        ("admitted_game_identity_leak_count", lambda data: data.__setitem__("admitted_game_identity_leak_count", 1)),
        (
            "registry_precheck",
            lambda data: data.__setitem__(
                "registry_precheck", {**data["registry_precheck"], "all_public_games_complete": False}
            ),
        ),
        (
            "registry_precheck count",
            lambda data: data.__setitem__(
                "registry_precheck", {**data["registry_precheck"], "completed_levels": 182}
            ),
        ),
        ("arc_registry_delta", lambda data: data.__setitem__("arc_registry_delta", 1)),
        ("arc_solve_credited", lambda data: data.__setitem__("arc_solve_credited", True)),
        ("science_rerun", lambda data: data.__setitem__("science_rerun", True)),
        ("live_policy_modified", lambda data: data.__setitem__("live_policy_modified", True)),
        ("solve_provenance", lambda data: data.__setitem__("solve_provenance", "live_agent_self_discovery")),
        ("preconditions_checked", lambda data: data.__setitem__("preconditions_checked", [])),
        (
            "preconditions_checked source_artifact_hash_verified",
            lambda data: data["preconditions_checked"].__setitem__(
                "source_artifact_hash_verified", False
            ),
        ),
        (
            "preconditions_checked scripts_research_conductor_modified",
            lambda data: data["preconditions_checked"].__setitem__(
                "scripts_research_conductor_modified", True
            ),
        ),
        ("test_commands", lambda data: data.__setitem__("test_commands", "bad")),
        (
            "test_exit_codes",
            lambda data: (
                data.__setitem__("test_commands", ["unit"]),
                data.__setitem__("test_exit_codes", {"unit": 1}),
            ),
        ),
        ("honest_verdict", lambda data: data.__setitem__("honest_verdict", "done")),
        ("reproducibility_checksum", lambda data: data.__setitem__("reproducibility_checksum", "bad")),
    ]

    for expected, mutate in mutations:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected != "reproducibility_checksum":
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        with pytest.raises(ValueError, match=expected):
            mod.validate_artifact(bad)


def test_req_arc_wmte_5745_helper_branches_fail_closed(tmp_path: Path) -> None:
    """REQ-ARC-WMTE-5745: malformed source receipts fail before normalization."""

    source = _source_artifact()

    bad_candidates = deepcopy(source)
    bad_candidates["primitive_candidates"] = {}
    with pytest.raises(ValueError, match="primitive_candidates"):
        mod.derive_leak_counts(bad_candidates)

    bad_ids = deepcopy(source)
    bad_ids["primitive_candidates"][0]["primitive"] = "not_frozen"
    with pytest.raises(ValueError, match="frozen_primitive_ids"):
        mod.derive_leak_counts(bad_ids)

    bad_utility = deepcopy(source)
    bad_utility["counterfactual_trajectory_utility"] = []
    with pytest.raises(ValueError, match="counterfactual_trajectory_utility"):
        mod._retained_effects(bad_utility)

    missing_effect = deepcopy(source)
    missing_effect["counterfactual_trajectory_utility"].pop(mod.FROZEN_PRIMITIVE_IDS[0])
    with pytest.raises(ValueError, match="counterfactual_trajectory_utility"):
        mod._retained_effects(missing_effect)

    changed_effect = deepcopy(source)
    changed_effect["counterfactual_trajectory_utility"][mod.FROZEN_PRIMITIVE_IDS[0]][
        "composite_utility_delta"
    ] = 0.0
    with pytest.raises(ValueError, match="frozen_effect_hash"):
        mod._retained_effects(changed_effect)

    effects = mod._retained_effects(source)
    no_hash = deepcopy(effects)
    no_hash[mod.FROZEN_PRIMITIVE_IDS[0]]["baseline_decision_hash"] = "bad"
    with pytest.raises(ValueError, match="exact_replay_receipts"):
        mod._exact_replay_receipts(no_hash)

    same_hash = deepcopy(effects)
    primitive = mod.FROZEN_PRIMITIVE_IDS[0]
    same_hash[primitive]["deletion_decision_hash"] = same_hash[primitive][
        "baseline_decision_hash"
    ]
    with pytest.raises(ValueError, match="exact_replay_receipts"):
        mod._exact_replay_receipts(same_hash)

    no_downstream = deepcopy(effects)
    no_downstream[primitive]["downstream_decision_hash_changed_count"] = 0
    with pytest.raises(ValueError, match="exact_replay_receipts"):
        mod._exact_replay_receipts(no_downstream)

    bad_controls = deepcopy(source)
    bad_controls["negative_controls"] = {}
    with pytest.raises(ValueError, match="negative_controls"):
        mod.derive_leak_counts(bad_controls)

    bad_control_row = deepcopy(source)
    bad_control_row["negative_controls"][0] = "bad"
    with pytest.raises(ValueError, match="negative_controls"):
        mod.derive_leak_counts(bad_control_row)

    for key, expected_source, expected_identity in (
        ("source_rule", 1, 0),
        ("game", 0, 1),
        ("learner_visible", 1, 1),
    ):
        leaked = deepcopy(source)
        if key == "learner_visible":
            leaked["primitive_candidates"][0][key] = {"source_rule": "x", "game_id": "x"}
        else:
            leaked["primitive_candidates"][0][key] = "x"
        counts = mod.derive_leak_counts(leaked)
        assert counts["admitted_source_leak_count"] == expected_source
        assert counts["admitted_game_identity_leak_count"] == expected_identity

    admitted_identity = deepcopy(source)
    admitted_identity["primitive_candidates"][0]["live_state_leak_classes"] = [
        "game_identity"
    ]
    assert mod.derive_leak_counts(admitted_identity)["admitted_game_identity_leak_count"] == 1

    bad_trace = deepcopy(source)
    bad_trace["trace_manifest"] = {}
    with pytest.raises(ValueError, match="trace_manifest"):
        mod.verify_trace_manifest_hashes(bad_trace, root=REPO)

    bad_trace_row = deepcopy(source)
    bad_trace_row["trace_manifest"][0] = "bad"
    with pytest.raises(ValueError, match="trace_manifest"):
        mod.verify_trace_manifest_hashes(bad_trace_row, root=REPO)

    with pytest.raises(ValueError, match="trace_manifest referenced receipt"):
        mod.verify_trace_manifest_hashes(source, root=tmp_path)

    wrong_file_root = tmp_path / "wrong_file"
    (wrong_file_root / "results").mkdir(parents=True)
    (wrong_file_root / "results/arc_live_oracle_gap.json").write_text("bad", encoding="utf-8")
    with pytest.raises(ValueError, match="trace_manifest file hash"):
        mod.verify_trace_manifest_hashes(source, root=wrong_file_root)

    partial_root = tmp_path / "partial"
    (partial_root / "results").mkdir(parents=True)
    first_rel = Path(source["trace_manifest"][0]["path"])
    (partial_root / first_rel).write_bytes((REPO / first_rel).read_bytes())
    partial_trace = deepcopy(source)
    partial_trace["trace_manifest"] = [partial_trace["trace_manifest"][0]]
    with pytest.raises(ValueError, match="trace_manifest did not verify"):
        mod.verify_trace_manifest_hashes(partial_trace, root=partial_root)

    bad_coverage = deepcopy(source)
    bad_coverage["counterfactual_receipt_coverage"] = []
    with pytest.raises(ValueError, match="counterfactual_receipt_coverage"):
        mod.normalize_counterfactual_coverage(bad_coverage, root=REPO)

    for key, value, expected in (
        ("minimum_positive_candidate_paired_replay_count", 999, "paired_replay coverage minimum"),
        ("paired_replay_count", 999, "paired_replay coverage total"),
        ("trace_step_count", 999, "trace_step_count"),
        ("meets_minimum_n", False, "minimum flag"),
    ):
        corrupted = deepcopy(source)
        corrupted["counterfactual_receipt_coverage"][key] = value
        with pytest.raises(ValueError, match=expected):
            mod.normalize_counterfactual_coverage(corrupted, root=REPO)

    bad_registry_root = tmp_path / "bad_registry"
    (bad_registry_root / "ops").mkdir(parents=True)
    (bad_registry_root / mod.REGISTRY_RELATIVE_PATH).write_text("games: bad\n", encoding="utf-8")
    with pytest.raises(ValueError, match="registry_precheck games"):
        mod.registry_precheck(root=bad_registry_root)

    incomplete_registry_root = tmp_path / "incomplete_registry"
    (incomplete_registry_root / "ops").mkdir(parents=True)
    (incomplete_registry_root / mod.REGISTRY_RELATIVE_PATH).write_text(
        "reproducible_total_games: 0\ngames: []\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="registry_precheck expected"):
        mod.registry_precheck(root=incomplete_registry_root)

    bad_source_root = tmp_path / "bad_source"
    (bad_source_root / "results").mkdir(parents=True)
    (bad_source_root / mod.SOURCE_ARTIFACT_RELATIVE_PATH).write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="source_artifact_hash"):
        mod.build_artifact(root=bad_source_root)


def test_scenario_arc_wmte_5745_checked_in_artifact_is_stable_when_present() -> None:
    """SCENARIO-ARC-WMTE-5745-HASH-LINKED-NO-CREDIT-CONTRACT."""

    if not RESULT_PATH.exists():
        pytest.skip("Exp5745 artifact has not been emitted yet")
    artifact = json.loads(RESULT_PATH.read_text(encoding="utf-8"))

    mod.validate_artifact(artifact)
    assert artifact["source_artifact_hash"] == mod.EXPECTED_SOURCE_ARTIFACT_HASH
    assert artifact["frozen_effect_hash"] == mod.EXPECTED_FROZEN_EFFECT_HASH
    assert artifact["counterfactual_receipt_coverage_score"] == 1.0
    assert artifact["arc_registry_delta"] == 0
    assert artifact["arc_solve_credited"] is False
