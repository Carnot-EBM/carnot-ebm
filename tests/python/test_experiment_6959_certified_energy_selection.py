"""Tests for label-blind selection from the frozen mapping proposal bank.

Spec refs: REQ-VERIFY-6959 and SCENARIO-VERIFY-6959-*.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_6959_certified_energy_selection as exp
from carnot import experiment_6958_convex_factor_energy_canary as energy_exp


ROOT = Path(__file__).resolve().parents[2]
BANK = ROOT / "results/experiment_6956_three_family_reformulation_bank.json"
CERTIFICATES = ROOT / "results/experiment_6957_smt_mapping_certification.json"
ENERGY = ROOT / "results/experiment_6958_convex_factor_energy_canary.json"


def _candidate(
    key: str,
    variant: str,
    scores: dict[str, float | None],
    *,
    raw_hash: str | None = None,
) -> dict[str, object]:
    """REQ-VERIFY-6959 tests use the smallest label-blind candidate payload."""

    return {
        "attempt_key": key,
        "group_id": "model|pair",
        "pair_id": "pair",
        "model_family": "model_a",
        "problem_family": "family_a",
        "prompt_variant_id": variant,
        "raw_sha256": raw_hash or f"sha256:{key}",
        "scores": scores,
    }


def _selection(
    group_id: str,
    pair_id: str,
    arm: str,
    correct: bool,
    *,
    model_family: str = "model_a",
    problem_family: str = "family_a",
) -> dict[str, object]:
    """REQ-VERIFY-6959 synthetic rows isolate group-weighted reducers."""

    return {
        "group_id": group_id,
        "pair_id": pair_id,
        "arm": arm,
        "model_family": model_family,
        "problem_family": problem_family,
        "difficulty": "standard",
        "candidate_diversity": "unique",
        "selected_attempt_key": f"{group_id}|selected",
        "selected_exact_correct": correct,
        "selected_false_acceptance": False,
        "abstained": False,
        "terminal": True,
    }


def test_req_verify_6959_spec_precedes_code_and_declares_contract() -> None:
    """REQ-VERIFY-6959 owns every required field and failure scenario."""

    text = (ROOT / exp.SPEC_PATH).read_text(encoding="utf-8")
    section = text[text.index("### REQ-VERIFY-6959") :]

    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(
        f"SCENARIO-VERIFY-6959-{name}" in section
        for name in (
            "PRECONDITIONS",
            "LEAKAGE",
            "ORDER",
            "SCORES",
            "DUPLICATES",
            "HEADROOM",
            "AGGREGATES",
            "GATES",
        )
    )


def test_scenario_verify_6959_leakage_is_recursive_and_fails_closed() -> None:
    """SCENARIO-VERIFY-6959-LEAKAGE rejects direct and nested oracle fields."""

    safe = {"mapping": {"variables": []}, "metadata": {"confidence": 0.8}}
    assert exp.audit_selector_payload("safe", safe)["passed"] is True

    for field in exp.FORBIDDEN_SELECTOR_FIELDS:
        leaked = {"mapping": {"nested": {field: "secret"}}}
        row = exp.audit_selector_payload("leaked", leaked)
        assert row["passed"] is False
        assert row["forbidden_paths"] == [f"mapping.nested.{field}"]


def test_scenario_verify_6959_score_directions_ties_and_missing_values() -> None:
    """SCENARIO-VERIFY-6959-SCORES applies directions and preserves missing scores."""

    low = [
        _candidate("a", "objective_first", {exp.ARM_CONVEX: 2.0}),
        _candidate("b", "direct_affine", {exp.ARM_CONVEX: 1.0}),
        _candidate("c", "domain_first", {exp.ARM_CONVEX: 3.0}),
    ]
    selected = exp.rank_group(low, exp.ARM_CONVEX)
    assert selected["selected_attempt_key"] == "b"
    assert selected["score_direction"] == "min"

    high = [
        _candidate("a", "objective_first", {exp.ARM_CONFIDENCE: 0.9}),
        _candidate("b", "direct_affine", {exp.ARM_CONFIDENCE: 0.7}),
        _candidate("c", "domain_first", {exp.ARM_CONFIDENCE: 0.8}),
    ]
    assert exp.rank_group(high, exp.ARM_CONFIDENCE)["selected_attempt_key"] == "a"

    tied = [
        _candidate("a", "objective_first", {exp.ARM_LINEAR: 1.0}),
        _candidate("b", "domain_first", {exp.ARM_LINEAR: 1.0}),
        _candidate("c", "direct_affine", {exp.ARM_LINEAR: 1.0}),
    ]
    tied_row = exp.rank_group(tied, exp.ARM_LINEAR)
    assert tied_row["selected_attempt_key"] == "c"
    assert tied_row["tie_count"] == 3
    assert tied_row["tied_attempt_keys"] == ["c", "b", "a"]

    unavailable = [
        _candidate("a", "direct_affine", {exp.ARM_LIKELIHOOD: None}),
        _candidate("b", "domain_first", {exp.ARM_LIKELIHOOD: None}),
    ]
    missing_row = exp.rank_group(unavailable, exp.ARM_LIKELIHOOD)
    assert missing_row["terminal"] is True
    assert missing_row["abstained"] is True
    assert missing_row["selected_attempt_key"] is None
    assert missing_row["unavailable_score_count"] == 2


def test_scenario_verify_6959_candidate_order_is_not_a_hidden_selector() -> None:
    """SCENARIO-VERIFY-6959-ORDER keeps scores and identities stable under permutation."""

    candidates = [
        _candidate("a", "objective_first", {exp.ARM_MLP: 1.0}),
        _candidate("b", "domain_first", {exp.ARM_MLP: 0.0}),
        _candidate("c", "direct_affine", {exp.ARM_MLP: 2.0}),
    ]
    forward = exp.rank_group(candidates, exp.ARM_MLP)
    reverse = exp.rank_group(list(reversed(candidates)), exp.ARM_MLP)
    assert forward["selected_attempt_key"] == reverse["selected_attempt_key"] == "b"
    assert forward["selection_frozen_hash"] == reverse["selection_frozen_hash"]

    tied = deepcopy(candidates)
    for row in tied:
        row["scores"][exp.ARM_MLP] = 0.0
    assert exp.rank_group(tied, exp.ARM_MLP)["selected_attempt_key"] == "c"
    assert exp.rank_group(list(reversed(tied)), exp.ARM_MLP)["selected_attempt_key"] == "c"


def test_scenario_verify_6959_duplicates_and_no_headroom_stay_visible() -> None:
    """SCENARIO-VERIFY-6959-DUPLICATES and HEADROOM keep degenerate groups explicit."""

    duplicate = [
        _candidate("a", "direct_affine", {exp.ARM_FIXED: 0.0}, raw_hash="same"),
        _candidate("b", "domain_first", {exp.ARM_FIXED: 1.0}, raw_hash="same"),
        _candidate("c", "objective_first", {exp.ARM_FIXED: 2.0}, raw_hash="other"),
    ]
    diversity = exp.candidate_diversity_row(duplicate)
    assert diversity["candidate_count"] == 3
    assert diversity["unique_candidate_count"] == 2
    assert diversity["duplicate_candidate_count"] == 1
    assert diversity["candidate_diversity"] == "contains_duplicates"

    labels = {"a": False, "b": False, "c": False}
    oracle = exp.oracle_upper_bound_row(duplicate, labels)
    assert oracle["correct_candidate_count"] == 0
    assert oracle["group_has_correct_candidate"] is False
    no_correct = exp.headroom_row(
        oracle, baseline_correct=False, convex_correct=False, baseline_arm=exp.ARM_FIXED
    )
    assert no_correct["available_headroom"] == 0
    assert no_correct["headroom_captured"] is None

    labels["c"] = True
    oracle = exp.oracle_upper_bound_row(duplicate, labels)
    no_headroom = exp.headroom_row(
        oracle, baseline_correct=True, convex_correct=True, baseline_arm=exp.ARM_FIXED
    )
    assert no_headroom["available_headroom"] == 0
    assert no_headroom["headroom_captured"] is None


def test_scenario_verify_6959_pair_bootstrap_and_family_reducer_use_groups() -> None:
    """SCENARIO-VERIFY-6959-AGGREGATES bootstraps pair IDs and gives each group one vote."""

    paired = [
        {"pair_id": "p1", "paired_top1_delta": 1.0},
        {"pair_id": "p1", "paired_top1_delta": -1.0},
        {"pair_id": "p2", "paired_top1_delta": 1.0},
    ]
    interval = exp.paired_bootstrap_by_pair(paired, seed=7, samples=100)
    assert interval["paired_pair_count"] == 2
    assert interval["paired_group_count"] == 3
    assert interval["bootstrap_unit"] == "pair_id"

    rows = [
        _selection("g1", "p1", exp.ARM_FIXED, True),
        _selection("g2", "p2", exp.ARM_FIXED, False),
    ]
    aggregate = exp.aggregate_selection_rows(rows, "model_family", "model_a", exp.ARM_FIXED)
    assert aggregate["group_count"] == 2
    assert aggregate["top1_accuracy"] == 0.5

    imbalanced = rows + [deepcopy(rows[0]) for _ in range(9)]
    with pytest.raises(ValueError, match="duplicate_group_arm"):
        exp.aggregate_selection_rows(imbalanced, "model_family", "model_a", exp.ARM_FIXED)


def test_scenario_verify_6959_positive_gate_needs_top1_and_headroom_not_auroc() -> None:
    """SCENARIO-VERIFY-6959-GATES does not turn high AUROC alone into a selection win."""

    controls = {"leakage": True, "ties": True, "shuffled": True, "fixed": True}
    assert (
        exp.positive_gate(
            complete=True,
            paired_ci_lower=0.0,
            headroom_captured=1.0,
            control_checks=controls,
            candidate_auroc=1.0,
        )
        is False
    )
    assert (
        exp.positive_gate(
            complete=True,
            paired_ci_lower=0.01,
            headroom_captured=0.19,
            control_checks=controls,
            candidate_auroc=1.0,
        )
        is False
    )
    assert (
        exp.positive_gate(
            complete=True,
            paired_ci_lower=0.01,
            headroom_captured=0.2,
            control_checks=controls,
            candidate_auroc=0.5,
        )
        is True
    )
    broken = dict(controls, leakage=False)
    assert (
        exp.positive_gate(
            complete=True,
            paired_ci_lower=0.01,
            headroom_captured=0.2,
            control_checks=broken,
            candidate_auroc=1.0,
        )
        is False
    )


def test_scenario_verify_6959_preconditions_fail_closed_with_exact_observation(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6959-PRECONDITIONS reports missing rows and incompatible scores."""

    bank = json.loads(BANK.read_text(encoding="utf-8"))
    certificates = json.loads(CERTIFICATES.read_text(encoding="utf-8"))
    energy = json.loads(ENERGY.read_text(encoding="utf-8"))
    arms = exp.freeze_arm_policy()
    ready = exp.check_preconditions(ROOT, bank, certificates, energy, arms)
    assert ready["passed"] is True

    broken = deepcopy(certificates)
    broken["smt_certification_run_complete_score"] = 0
    broken["proposal_rows"].pop()
    summary = exp.check_preconditions(ROOT, bank, broken, energy, arms)
    failed = {row["check"]: row for row in summary["failed_checks"]}
    assert failed["smt_certification_run_complete_score"]["expected"] == 1
    assert failed["smt_certification_run_complete_score"]["observed"] == 0
    assert failed["certificate_row_count"]["expected"] == exp.EXPECTED_CANDIDATE_COUNT

    blocked = exp.build_blocked_artifact(
        date="20260904",
        preconditions=summary,
        source_hashes={},
        duration_s=0.01,
    )
    assert blocked["certified_selection_run_complete_score"] == 0
    assert blocked["certified_energy_positive_score"] == 0
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"] == "blocked_certified_energy_selection"
    assert set(blocked["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    exp.validate_artifact(blocked)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """REQ-VERIFY-6959 builds the full frozen selection artifact once for integration tests."""

    work = tmp_path_factory.mktemp("exp6959")
    return exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        bank_path=BANK,
        certificate_path=CERTIFICATES,
        energy_path=ENERGY,
        replay_manifest_path=work / "replay.json",
        bootstrap_samples=200,
    )


def test_req_verify_6959_complete_artifact_has_identical_arm_groups(
    artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-6959 requires every arm to terminate on every frozen group."""

    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["inference_substrate"] == exp.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["certified_selection_run_complete_score"] == 1
    assert artifact["verdict_class"] in {"positive", "null"}
    assert artifact["honest_verdict"].startswith("complete_")
    assert len(artifact["candidate_group_rows"]) == exp.EXPECTED_GROUP_COUNT
    assert len(artifact["candidate_rows"]) == exp.EXPECTED_CANDIDATE_COUNT
    assert len(artifact["selection_rows"]) == exp.EXPECTED_GROUP_COUNT * len(exp.ARM_ORDER)
    assert {row["arm"] for row in artifact["arm_rows"]} == set(exp.ARM_ORDER)
    assert all(row["terminal"] for row in artifact["selection_rows"])
    assert all(row["oracle_used_for_selection"] is False for row in artifact["selection_rows"])
    assert all(row["passed"] for row in artifact["leakage_rows"])
    assert all(row["replay_matches"] for row in artifact["fresh_process_replay_rows"])
    assert len(artifact["likelihood_rows"]) == exp.EXPECTED_CANDIDATE_COUNT
    assert any(row["score"] is None for row in artifact["likelihood_rows"])
    exp.validate_artifact(artifact)


def test_scenario_verify_6959_aggregate_mismatch_and_checksum_are_rejected(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-6959-AGGREGATES rejects a headline not supported by rows."""

    mismatch = deepcopy(artifact)
    mismatch["arm_rows"][0]["top1_accuracy"] += 0.1
    mismatch["reproducibility_checksum"] = exp.payload_checksum(mismatch)
    with pytest.raises(ValueError, match="aggregate_row_mismatch"):
        exp.validate_artifact(mismatch)

    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:wrong"
    with pytest.raises(ValueError, match="reproducibility_checksum_mismatch"):
        exp.validate_artifact(checksum)


def test_scenario_verify_6959_fresh_process_replay_rebuilds_headlines(
    artifact: dict[str, object], tmp_path: Path
) -> None:
    """SCENARIO-VERIFY-6959-AGGREGATES replays metrics from serialized rows only."""

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(exp.replay_manifest_payload(artifact), sort_keys=True), encoding="utf-8"
    )
    replay = exp.replay_manifest(manifest)
    assert (
        replay["headline_checksum"] == artifact["fresh_process_replay_rows"][0]["headline_checksum"]
    )
    assert replay["metrics"] == exp.replay_headline_metrics(artifact["selection_rows"])


def test_req_verify_6959_validation_and_run_surface(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6959 validates bindings and writes through the public run surface."""

    variants: list[tuple[dict[str, object], str]] = []
    missing = deepcopy(artifact)
    missing.pop("rows")
    variants.append((missing, "missing_artifact_fields"))
    principles = deepcopy(artifact)
    principles["field_principles"] = {}
    variants.append((principles, "field_principles_mismatch"))
    substrate = deepcopy(artifact)
    substrate["inference_substrate"] = "wrong"
    variants.append((substrate, "inference_substrate_mismatch"))
    oracle = deepcopy(artifact)
    oracle["verifier_is_oracle"] = True
    variants.append((oracle, "verifier_is_oracle_mismatch"))
    positive = deepcopy(artifact)
    positive["certified_selection_run_complete_score"] = 0
    positive["certified_energy_positive_score"] = 1
    variants.append((positive, "positive_without_completion"))
    wrong_class = deepcopy(artifact)
    wrong_class["verdict_class"] = "partial"
    variants.append((wrong_class, "verdict_class_score_mismatch"))
    for value, reason in variants:
        value["reproducibility_checksum"] = exp.payload_checksum(value)
        with pytest.raises(ValueError, match=reason):
            exp.validate_artifact(value)

    payload = deepcopy(artifact)
    output = tmp_path / "artifact.json"
    monkeypatch.setattr(exp, "build_artifact", lambda **kwargs: payload)
    observed = exp.run("20260904", repo_root=ROOT, output_path=output)
    assert observed == payload
    assert json.loads(output.read_text(encoding="utf-8")) == payload


def test_req_verify_6959_defensive_score_and_replay_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-VERIFY-6959 keeps malformed metadata and replay failures explicit."""

    assert exp._unique_row({}, "name", "x") is None
    assert exp._opposite_aggregation("min") == "max"
    assert exp._opposite_aggregation("unknown") is None
    assert exp._likelihood_score({"runtime_receipt": "not-a-row"}) is None
    assert exp._likelihood_score({"mean_logprob": -0.25}) == -0.25
    assert exp._calibration([]) == (None, None)
    assert exp.paired_bootstrap_by_pair([], seed=1, samples=10)["mean_delta"] is None
    with pytest.raises(ValueError, match="unknown_selection_arm"):
        exp.rank_group([_candidate("a", "direct_affine", {})], "oracle")

    bad_manifest = tmp_path / "bad-replay.json"
    bad_manifest.write_text('{"schema_version":"wrong"}', encoding="utf-8")
    with pytest.raises(ValueError, match="replay_manifest_schema_mismatch"):
        exp.replay_manifest(bad_manifest)

    monkeypatch.setattr(
        exp.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"returncode": 1, "stderr": "failed"})(),
    )
    with pytest.raises(RuntimeError, match="fresh_process_replay_failed"):
        exp._fresh_process_replay(ROOT, bad_manifest)

    positive = {
        "honest_verdict": "complete_positive_certified_energy_selection",
        "certified_energy_positive_score": 1,
        "certified_selection_run_complete_score": 1,
    }
    partial = dict(
        positive,
        honest_verdict="partial_certified_energy_selection",
        certified_energy_positive_score=0,
        certified_selection_run_complete_score=0,
    )
    assert exp._expected_verdict(positive) == "positive"
    assert exp._expected_verdict(partial) == "partial"


def test_scenario_verify_6959_malformed_structural_factors_fail_closed() -> None:
    """SCENARIO-VERIFY-6959-SCORES maps malformed affine and objective fields to violations."""

    bank = json.loads(BANK.read_text(encoding="utf-8"))
    attempt = deepcopy(
        next(row for row in bank["attempt_rows"] if row["parse"]["parsed_candidate"] is not None)
    )
    attempt["parse"]["parsed_candidate"]["mapping"]["variables"][0]["scale"] = "bad"
    factors = exp.structural_factors(attempt)
    assert factors[0][2] == 1.0

    attempt["parse"]["parsed_candidate"]["mapping"]["objective"]["offset"] = "bad"
    factors = exp.structural_factors(attempt)
    assert factors[-2][3] == 1.0
    assert factors[-1][4] == 1.0


def test_scenario_verify_6959_checkpoint_compatibility_rejects_bad_controls(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-6959-PRECONDITIONS rejects hash drift and unknown checkpoint arms."""

    convex_path = tmp_path / "convex.pt"
    convex = energy_exp.make_model(energy_exp.ARM_CONVEX, 7)
    energy_exp._save_checkpoint(convex_path, energy_exp.ARM_CONVEX, 7, convex)
    hand_path = tmp_path / "hand.pt"
    hand = energy_exp.make_model(energy_exp.ARM_HAND, 8)
    energy_exp._save_checkpoint(hand_path, energy_exp.ARM_HAND, 8, hand)
    broken_path = tmp_path / "broken.pt"
    broken_path.write_text("not a checkpoint", encoding="utf-8")
    energy = {
        "checkpoint_paths": [convex_path.name, hand_path.name, broken_path.name],
        "fresh_process_replay_rows": [
            {
                "arm": energy_exp.ARM_CONVEX,
                "seed": 7,
                "checkpoint_sha256": "sha256:wrong",
            },
            {
                "arm": energy_exp.ARM_HAND,
                "seed": 8,
                "checkpoint_sha256": exp.sha256_path(hand_path),
            },
        ],
    }
    count, paths = exp._checkpoint_compatibility(tmp_path, energy)
    assert count == 0
    assert paths == []


def test_scenario_verify_6959_build_artifact_uses_blocked_preflight(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-6959-PRECONDITIONS stops before scoring after a failed gate."""

    certificates = json.loads(CERTIFICATES.read_text(encoding="utf-8"))
    certificates["smt_certification_run_complete_score"] = 0
    broken_path = tmp_path / "certificates.json"
    broken_path.write_text(json.dumps(certificates), encoding="utf-8")
    blocked = exp.build_artifact(
        date="20260904",
        repo_root=ROOT,
        bank_path=BANK,
        certificate_path=broken_path,
        energy_path=ENERGY,
        replay_manifest_path=tmp_path / "replay.json",
        bootstrap_samples=10,
    )
    assert blocked["verdict_class"] == "blocked"
    assert blocked["rows"] == []
