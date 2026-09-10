"""Tests for REQ-VERIFY-7182 and SCENARIO-VERIFY-7182-*.

The tests use the checked-in frozen inputs. All new result and checkpoint
writes stay below private temporary directories.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
import os
from pathlib import Path
import sys

import pytest

from carnot import experiment_7182_v633_grounding_energy_audit as exp
from carnot import experiment_7182_v633_grounding_energy_independent_audit as auditor


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
TRACE_PATH = REPO / "results/experiment_7181_v633_qwen38_symbolic_traces.json"
GENERATION_PATH = REPO / "results/experiment_7180_v633_symbolic_edit_fixture_generation_view.jsonl"
AUTHORITY_PATH = REPO / "results/experiment_7180_v633_symbolic_edit_fixture_authority.jsonl"


@pytest.fixture(scope="module")
def source_rows() -> dict[str, list[dict[str, object]]]:
    """Load immutable source rows once so mutations cannot touch repository files."""

    trace = json.loads(TRACE_PATH.read_text(encoding="utf-8"))["rows"]
    generation = [
        json.loads(line) for line in GENERATION_PATH.read_text(encoding="utf-8").splitlines()
    ]
    authority = [
        json.loads(line) for line in AUTHORITY_PATH.read_text(encoding="utf-8").splitlines()
    ]
    return {"trace": trace, "generation": generation, "authority": authority}


@pytest.fixture(scope="module")
def features(source_rows: dict[str, list[dict[str, object]]]) -> list[dict[str, object]]:
    """Reparse raw completions without opening the private authority sidecar."""

    return exp.build_candidate_features(source_rows["trace"], source_rows["generation"])


@pytest.fixture(scope="module")
def frozen(
    features: list[dict[str, object]], source_rows: dict[str, list[dict[str, object]]]
) -> dict[str, object]:
    """Fit the only adjustable value from calibration-family labels."""

    calibration = [row for row in source_rows["authority"] if row["split"] == "calibration"]
    return exp.select_threshold(features, calibration)


@pytest.fixture(scope="module")
def evaluated(
    features: list[dict[str, object]],
    frozen: dict[str, object],
    source_rows: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Evaluate all five arms on the exact held-out roster."""

    fixture = json.loads(
        (REPO / "results/experiment_7180_v633_symbolic_edit_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    evaluation = [row for row in source_rows["authority"] if row["split"] == "evaluation"]
    return exp.evaluate_arms(features, evaluation, frozen, fixture["score_contract"])


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], dict[str, Path]]:
    """Build one full audit through its real fresh-process auditor boundary."""

    directory = tmp_path_factory.mktemp("exp7182")
    paths = {
        "result": directory / "result.json",
        "checkpoint": directory / "checkpoints" / "running.json",
        "audit_request": directory / "checkpoints" / "audit_request.json",
        "audit_result": directory / "checkpoints" / "audit_result.json",
    }
    paths["checkpoint"].parent.mkdir()
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=paths["result"],
        checkpoint_path=paths["checkpoint"],
        audit_request_path=paths["audit_request"],
        audit_result_path=paths["audit_result"],
        duration_s=0.5,
    )
    return artifact, paths


def test_req_verify_7182_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7182 names every focused scenario and required field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7182") :]
    for scenario in (
        "PREFLIGHT",
        "LEAKAGE",
        "PARSE",
        "PAIRS",
        "METRICS",
        "AUDIT",
        "VERDICT",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7182-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7182_leakage_rejects_authority_fields(
    source_rows: dict[str, list[dict[str, object]]],
    features: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-LEAKAGE fails if truth enters candidate inputs."""

    assert len(features) == 192
    serialized = exp.canonical_json(features)
    assert "expected_response" not in serialized
    assert '"split"' not in serialized
    assert '"variant"' not in serialized

    leaked = deepcopy(source_rows["trace"])
    leaked[0]["expected_response"] = {"direct_decision": "supported"}
    with pytest.raises(ValueError, match="authority_field_exposed"):
        exp.build_candidate_features(leaked, source_rows["generation"])


def test_scenario_verify_7182_reparses_raw_bytes_not_producer_projection(
    source_rows: dict[str, list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7182-LEAKAGE derives tuples from raw model output."""

    changed = deepcopy(source_rows["trace"])
    changed[0]["parse_status"] = "valid"
    changed[0]["direct_decision"] = "unsupported"
    changed[0]["extracted_tuples"] = {"claim_tuple": {}, "evidence_tuple": {}}
    rebuilt = exp.build_candidate_features(changed, source_rows["generation"])
    assert rebuilt[0]["parse_status"] == "failed"
    assert rebuilt[0]["direct_decision"] is None
    assert rebuilt[0]["raw_output_sha256"] == changed[0]["raw_output_sha256"]


def test_scenario_verify_7182_threshold_uses_only_calibration_labels(
    features: list[dict[str, object]],
    frozen: dict[str, object],
    source_rows: dict[str, list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7182-LEAKAGE freezes threshold before evaluation truth."""

    calibration = [row for row in source_rows["authority"] if row["split"] == "calibration"]
    changed_evaluation = deepcopy(source_rows["authority"])
    for row in changed_evaluation:
        if row["split"] == "evaluation":
            decision = row["expected_response"]["direct_decision"]
            row["expected_response"]["direct_decision"] = (
                "unsupported" if decision == "supported" else "supported"
            )
    assert exp.select_threshold(features, calibration) == frozen
    with pytest.raises(ValueError, match="calibration_authority_only"):
        exp.select_threshold(features, changed_evaluation)
    assert frozen["fit_relation_families"] == ["precedes", "follows"]
    assert frozen["evaluation_truth_accessed"] is False
    assert frozen["threshold"] in frozen["candidate_thresholds"]


def test_scenario_verify_7182_parse_failures_are_five_arm_abstentions(
    evaluated: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-PARSE retains every failed parse in each arm."""

    by_unit: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in evaluated:
        by_unit[str(row["unit_id"])].append(row)
    assert len(evaluated) == 5 * 128
    assert all(len(rows) == 5 for rows in by_unit.values())
    failed = [rows for rows in by_unit.values() if rows[0]["parse_status"] == "failed"]
    assert failed
    assert all(row["prediction"] == "abstain" for rows in failed for row in rows)
    assert all(row["abstention"] is True for rows in failed for row in rows)

    metrics = exp.summarize_arms(evaluated)
    assert all(value["denominator"] == 128 for value in metrics.values())
    assert all(value["parse_failure_count"] == len(failed) for value in metrics.values())


def test_scenario_verify_7182_pairs_reject_drop_duplicate_and_shuffle(
    evaluated: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-PAIRS rejects changed denominators and pair IDs."""

    assert exp.evaluation_row_errors(evaluated) == []
    dropped = deepcopy(evaluated[:-1])
    assert "row_count" in exp.evaluation_row_errors(dropped)
    duplicate = deepcopy(evaluated)
    duplicate[-1]["unit_id"] = duplicate[-2]["unit_id"]
    assert any("unit_roster" in error for error in exp.evaluation_row_errors(duplicate))
    shuffled = deepcopy(evaluated)
    first = next(index for index, row in enumerate(shuffled) if row["arm"] == exp.ARMS[1])
    shuffled[first]["unit_id"], shuffled[first + 1]["unit_id"] = (
        shuffled[first + 1]["unit_id"],
        shuffled[first]["unit_id"],
    )
    assert any("ordered_pair_ids" in error for error in exp.evaluation_row_errors(shuffled))


def test_scenario_verify_7182_metrics_keep_metamorphic_denominators(
    evaluated: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-METRICS reports each required count and rate."""

    metrics = exp.summarize_arms(evaluated)
    assert set(metrics) == set(exp.ARMS)
    required = {
        "parse_rate",
        "coverage",
        "false_accept_count",
        "false_reject_count",
        "accuracy",
        "harmful_flip_count",
        "rename_invariance",
        "semantic_edit_sensitivity",
    }
    assert all(required <= set(metric) for metric in metrics.values())
    assert all(metric["rename_pair_denominator"] == 32 for metric in metrics.values())
    assert all(metric["semantic_pair_denominator"] == 64 for metric in metrics.values())


def test_scenario_verify_7182_bootstrap_is_clustered_stratified_and_fixed(
    evaluated: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-PAIRS uses 10,000 family-stratified base draws."""

    first = exp.paired_cluster_bootstrap(evaluated, exp.BOOTSTRAP_SEED, draws=10_000)
    second = exp.paired_cluster_bootstrap(evaluated, exp.BOOTSTRAP_SEED, draws=10_000)
    assert first == second
    assert first["draw_count"] == 10_000
    assert first["cluster_count"] == 32
    assert first["clusters_per_family"] == 8
    assert first["stratified_by"] == "relation_family"
    assert first["cluster_key"] == "base_id"
    assert len(first["paired_unit_ids"]) == 128
    for name in ("accuracy_delta", "false_accept_rate_delta"):
        assert set(first["energy_vs_direct"][name]) == {"estimate", "ci95_lower", "ci95_upper"}


def test_scenario_verify_7182_auditor_matches_and_runs_fresh(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7182-AUDIT preserves independent rows and interventions."""

    artifact, paths = built
    audit = json.loads(paths["audit_result"].read_text(encoding="utf-8"))
    assert audit["process_id"] != os.getpid()
    assert audit["candidate_module_imported"] is False
    assert audit["threshold_adapted"] is False
    assert audit["formula_implemented_independently"] is True
    assert audit["disagreement_count"] == 0
    assert len(artifact["independent_audit_rows"]) == 128
    assert len(artifact["feature_lineage_rows"]) == 256
    counts = Counter(row["intervention"] for row in artifact["intervention_rows"])
    assert counts["evidence_swap_within_family"] == 128
    assert counts["label_permutation"] == 128
    for term in exp.ENERGY_TERMS:
        assert counts[f"delete_term:{term}"] == 128


def test_scenario_verify_7182_auditor_formula_does_not_import_candidate(
    source_rows: dict[str, list[dict[str, object]]],
    features: list[dict[str, object]],
    frozen: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7182-AUDIT directly checks the independent implementation."""

    valid = next(row for row in features if row["parse_status"] == "valid")
    independent = auditor.independent_energy(
        valid["parsed_response"], valid["source_text"], exp.ENERGY_WEIGHTS
    )
    assert independent == valid["energy_terms"]
    request = {
        "threshold": frozen["threshold"],
        "weights": exp.ENERGY_WEIGHTS,
        "candidate_rows": [],
        "expected_input_hashes": {},
        "random_seed": exp.RANDOM_SEED,
    }
    assert auditor.validate_request(request) == []
    request["evaluation_labels"] = []
    assert "request_contains_evaluation_labels" in auditor.validate_request(request)


def test_scenario_verify_7182_auditor_direct_replay_covers_all_controls(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7182-AUDIT directly replays the full independent core."""

    _, paths = built
    request = json.loads(paths["audit_request"].read_text(encoding="utf-8"))
    replay = auditor.run_audit(request)
    assert len(replay["independent_audit_rows"]) == 128
    assert len(replay["feature_lineage_rows"]) == 256
    assert len(replay["intervention_rows"]) == 896
    assert replay["disagreement_count"] == 0
    assert replay["input_hashes"] == request["expected_input_hashes"]


def test_scenario_verify_7182_auditor_parser_rejects_each_defect(
    features: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-PARSE keeps every independent parser failure explicit."""

    valid_feature = next(row for row in features if row["parse_status"] == "valid")
    prompt = (
        f"Evidence:\n{valid_feature['source_text']}\n\nClaim:\n{valid_feature['claim_text']}"
        "\n\nReturn one JSON object that matches the supplied response schema."
    )
    valid = deepcopy(valid_feature["parsed_response"])
    mutations: list[tuple[object, str]] = [([], "root_not_object")]
    missing = deepcopy(valid)
    missing.pop("direct_decision")
    mutations.append((missing, "field_set_mismatch"))
    decision = deepcopy(valid)
    decision["direct_decision"] = "maybe"
    mutations.append((decision, "direct_decision_invalid"))
    claim = deepcopy(valid)
    claim["claim_tuple"] = None
    mutations.append((claim, "claim_tuple_invalid"))
    evidence = deepcopy(valid)
    evidence["evidence_tuple"] = {"subject": "only"}
    mutations.append((evidence, "evidence_tuple_invalid"))
    missing_fields = deepcopy(valid)
    missing_fields["missing_fields"] = [1]
    mutations.append((missing_fields, "missing_fields_invalid"))
    span = deepcopy(valid)
    span["source_start"] = 0
    span["source_end"] = len(str(valid_feature["source_text"])) + 1
    mutations.append((span, "source_span_invalid"))
    assert auditor.parse_raw_output("{", prompt)["parse_error"] == "invalid_json"
    for value, expected in mutations:
        assert auditor.parse_raw_output(json.dumps(value), prompt)["parse_error"] == expected
    assert auditor.parse_raw_output(json.dumps(valid), prompt)["parse_status"] == "valid"


def test_scenario_verify_7182_candidate_guards_each_input_identity(
    source_rows: dict[str, list[dict[str, object]]],
) -> None:
    """SCENARIO-VERIFY-7182-PAIRS fails each changed raw-input identity."""

    trace = source_rows["trace"]
    generation = source_rows["generation"]
    with pytest.raises(ValueError, match="candidate_source_row_count"):
        exp.build_candidate_features(trace[:-1], generation)
    changed_generation = deepcopy(generation)
    changed_generation[0]["extra"] = True
    with pytest.raises(ValueError, match="generation_shape"):
        exp.build_candidate_features(trace, changed_generation)
    changed_trace = deepcopy(trace)
    changed_trace[0]["unit_id"] = "u-changed"
    with pytest.raises(ValueError, match="candidate_unit_order"):
        exp.build_candidate_features(changed_trace, generation)
    changed_trace = deepcopy(trace)
    changed_trace[0]["prompt"] += " "
    with pytest.raises(ValueError, match="candidate_prompt"):
        exp.build_candidate_features(changed_trace, generation)
    changed_trace = deepcopy(trace)
    changed_trace[0]["raw_output_sha256"] = "sha256:changed"
    with pytest.raises(ValueError, match="raw_output_hash_mismatch"):
        exp.build_candidate_features(changed_trace, generation)
    with pytest.raises(ValueError, match="prompt_shape_invalid"):
        exp._prompt_parts("bad prompt")


def test_scenario_verify_7182_candidate_guards_calibration_and_evaluation_contracts(
    features: list[dict[str, object]],
    frozen: dict[str, object],
    source_rows: dict[str, list[dict[str, object]]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7182-LEAKAGE rejects incomplete split and control inputs."""

    calibration = [row for row in source_rows["authority"] if row["split"] == "calibration"]
    missing_feature = deepcopy(calibration)
    missing_feature[0]["unit_id"] = "u-missing"
    with pytest.raises(ValueError, match="calibration_feature_missing"):
        exp.select_threshold(features, missing_feature)

    evaluation = [row for row in source_rows["authority"] if row["split"] == "evaluation"]
    fixture = json.loads(
        (REPO / "results/experiment_7180_v633_symbolic_edit_fixture.json").read_text(
            encoding="utf-8"
        )
    )
    contract = fixture["score_contract"]
    with pytest.raises(ValueError, match="evaluation_authority_only"):
        exp.evaluate_arms(features, evaluation[:-1], frozen, contract)
    unsealed = deepcopy(frozen)
    unsealed["evaluation_truth_accessed"] = True
    with pytest.raises(ValueError, match="threshold_not_blind"):
        exp.evaluate_arms(features, evaluation, unsealed, contract)
    missing = [row for row in features if row["unit_id"] != evaluation[0]["unit_id"]]
    with pytest.raises(ValueError, match="evaluation_feature_missing"):
        exp.evaluate_arms(missing, evaluation, frozen, contract)
    no_control = deepcopy(contract)
    no_control["seeded_shuffle_controls"] = []
    with pytest.raises(ValueError, match="shuffle_control_missing"):
        exp.evaluate_arms(features, evaluation, frozen, no_control)
    short_control = deepcopy(contract)
    removed_unit = evaluation[0]["unit_id"]
    short_control["seeded_shuffle_controls"][0]["mapping"] = [
        row
        for row in short_control["seeded_shuffle_controls"][0]["mapping"]
        if row["unit_id"] != removed_unit
    ]
    with pytest.raises(ValueError, match="shuffle_control_roster"):
        exp.evaluate_arms(features, evaluation, frozen, short_control)
    valid = next(row for row in features if row["parse_status"] == "valid")
    failed = next(row for row in features if row["parse_status"] == "failed")
    assert exp._shuffled_score(failed, valid)[0] == "abstain"
    monkeypatch.setattr(exp, "evaluation_row_errors", lambda _rows: ["forced"])
    with pytest.raises(ValueError, match="evaluation_rows_invalid"):
        exp.evaluate_arms(features, evaluation, frozen, contract)


def test_scenario_verify_7182_pair_and_bootstrap_guards_are_fail_closed(
    evaluated: list[dict[str, object]],
) -> None:
    """SCENARIO-VERIFY-7182-PAIRS covers every roster and bootstrap guard."""

    baseline_duplicate = deepcopy(evaluated)
    baseline_duplicate[1]["unit_id"] = baseline_duplicate[0]["unit_id"]
    assert "baseline_unit_roster" in exp.evaluation_row_errors(baseline_duplicate)
    wrong_order = deepcopy(evaluated)
    wrong_order[0]["row_order"] = 9
    assert "baseline_direct:row_order" in exp.evaluation_row_errors(wrong_order)
    wrong_family = deepcopy(evaluated)
    removed_family = exp.EVALUATION_FAMILIES[0]
    replacement = exp.EVALUATION_FAMILIES[1]
    for row in wrong_family:
        if row["relation_family"] == removed_family:
            row["relation_family"] = replacement
    assert "evaluation_families" in exp.evaluation_row_errors(wrong_family)
    wrong_base = deepcopy(evaluated)
    changed_unit = wrong_base[0]["unit_id"]
    for row in wrong_base:
        if row["unit_id"] == changed_unit:
            row["base_id"] = "b-new"
    assert "base_variant_roster" in exp.evaluation_row_errors(wrong_base)
    with pytest.raises(ValueError, match="cannot_summarize_invalid_rows"):
        exp.summarize_arms(evaluated[:-1])
    with pytest.raises(ValueError, match="cannot_bootstrap_invalid_rows"):
        exp.paired_cluster_bootstrap(evaluated[:-1], exp.BOOTSTRAP_SEED)
    with pytest.raises(ValueError, match="bootstrap_draws_positive"):
        exp.paired_cluster_bootstrap(evaluated, exp.BOOTSTRAP_SEED, draws=0)

    unbalanced = deepcopy(evaluated)
    base_id = unbalanced[0]["base_id"]
    for row in unbalanced:
        if row["base_id"] == base_id:
            row["relation_family"] = replacement
    assert exp.evaluation_row_errors(unbalanced) == []
    with pytest.raises(ValueError, match="bootstrap_family_cluster_count"):
        exp.paired_cluster_bootstrap(unbalanced, exp.BOOTSTRAP_SEED, draws=1)


def test_scenario_verify_7182_independent_auditor_rejects_contract_drift(
    built: tuple[dict[str, object], dict[str, Path]],
    source_rows: dict[str, list[dict[str, object]]],
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7182-AUDIT rejects every changed independent input."""

    _, paths = built
    request = json.loads(paths["audit_request"].read_text(encoding="utf-8"))
    assert auditor.validate_request({}) == ["request_fields_missing", "weight_terms_mismatch"]
    with pytest.raises(ValueError, match="prompt_shape_invalid"):
        auditor._prompt_parts("bad")
    assert auditor.independent_energy({}, "", exp.ENERGY_WEIGHTS)["missing_required_fields"] == 1
    with pytest.raises(ValueError, match="source_row_count_mismatch"):
        auditor._features([source_rows["trace"][0]], [], exp.ENERGY_WEIGHTS)
    changed_trace = deepcopy(source_rows["trace"][:1])
    changed_trace[0]["unit_id"] = "u-changed"
    with pytest.raises(ValueError, match="source_unit_order_mismatch"):
        auditor._features(changed_trace, source_rows["generation"][:1], exp.ENERGY_WEIGHTS)
    changed_trace = deepcopy(source_rows["trace"][:1])
    changed_trace[0]["prompt"] += " "
    with pytest.raises(ValueError, match="source_prompt_mismatch"):
        auditor._features(changed_trace, source_rows["generation"][:1], exp.ENERGY_WEIGHTS)
    with pytest.raises(ValueError, match="audit_request_invalid"):
        auditor.run_audit({})
    changed = deepcopy(request)
    changed["expected_input_hashes"]["trace_artifact"] = "sha256:changed"
    with pytest.raises(ValueError, match="audit_input_hash_mismatch"):
        auditor.run_audit(changed)
    for field, expected in (
        ("candidate_rows", "candidate_evaluation_roster_mismatch"),
        ("shuffle_mapping", "shuffle_mapping_roster_mismatch"),
        ("label_permutation_mapping", "label_mapping_roster_mismatch"),
    ):
        changed = deepcopy(request)
        changed[field].pop()
        with pytest.raises(ValueError, match=expected):
            auditor.run_audit(changed)
    output = tmp_path / "atomic" / "audit.json"
    auditor._atomic_write_json(output, {"ok": True})
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": True}


def test_scenario_verify_7182_precondition_guards_each_contract_layer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7182-PREFLIGHT stops at hashes, gates, tools, and paths."""

    paths = exp._resolve_paths(REPO, None)
    output = tmp_path / "out"
    output.mkdir()
    args = (REPO, exp.RUN_DATE, paths, output / "r", output / "c", output / "q", output / "a")
    wrong_hashes = deepcopy(exp.PINNED_HASHES)
    wrong_hashes["trace_artifact"] = "sha256:wrong"
    monkeypatch.setattr(exp, "PINNED_HASHES", wrong_hashes)
    checks, _ = exp._preconditions(*args)
    assert checks[-1]["check"] == "trace_artifact_hash"
    monkeypatch.undo()

    no_requirement = tmp_path / "spec.md"
    no_requirement.write_text("no requirement\n", encoding="utf-8")
    changed_paths = dict(paths)
    changed_paths["constraint_spec"] = no_requirement
    checks, _ = exp._preconditions(
        REPO, exp.RUN_DATE, changed_paths, output / "r", output / "c", output / "q", output / "a"
    )
    assert checks[-1]["check"] == "constraint_spec_requirement"

    expected_fixture = deepcopy(exp.EXP7180_EXPECTED_FIELDS)
    expected_fixture["status"] = "changed"
    monkeypatch.setattr(exp, "EXP7180_EXPECTED_FIELDS", expected_fixture)
    checks, _ = exp._preconditions(*args)
    assert checks[-1]["check"] == "exp7180_same_milestone_gate_fields"
    monkeypatch.undo()
    expected_trace = deepcopy(exp.EXP7181_EXPECTED_FIELDS)
    expected_trace["status"] = "changed"
    monkeypatch.setattr(exp, "EXP7181_EXPECTED_FIELDS", expected_trace)
    checks, _ = exp._preconditions(*args)
    assert checks[-1]["check"] == "exp7181_same_milestone_gate_fields"
    monkeypatch.undo()

    raw = json.loads(paths["raw_manifest"].read_text(encoding="utf-8"))
    raw["raw_rows"] = raw["raw_rows"][:-1]
    changed_raw = tmp_path / "raw.json"
    changed_raw.write_text(json.dumps(raw), encoding="utf-8")
    changed_paths = dict(paths)
    changed_paths["raw_manifest"] = changed_raw
    pinned = deepcopy(exp.PINNED_HASHES)
    pinned["raw_manifest"] = exp.sha256_file(changed_raw)
    monkeypatch.setattr(exp, "PINNED_HASHES", pinned)
    checks, _ = exp._preconditions(
        REPO, exp.RUN_DATE, changed_paths, output / "r", output / "c", output / "q", output / "a"
    )
    assert checks[-1]["check"] == "raw_manifest_roster"
    monkeypatch.undo()

    monkeypatch.setattr(sys, "executable", "/bin/false")
    checks, _ = exp._preconditions(*args)
    assert checks[-1]["check"] == "python_executable"
    monkeypatch.undo()
    checks, _ = exp._preconditions(
        REPO,
        exp.RUN_DATE,
        paths,
        tmp_path / "missing" / "r",
        output / "c",
        output / "q",
        output / "a",
    )
    assert checks[-1]["check"] == "terminal_output_directory"


def test_scenario_verify_7182_preflight_writes_terminal_block(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7182-PREFLIGHT records the first exact source failure."""

    checkpoint = tmp_path / "checkpoints" / "running.json"
    checkpoint.parent.mkdir()
    result = tmp_path / "result.json"
    blocked = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=result,
        checkpoint_path=checkpoint,
        audit_request_path=tmp_path / "checkpoints" / "audit_request.json",
        audit_result_path=tmp_path / "checkpoints" / "audit_result.json",
        source_paths={"trace_artifact": tmp_path / "missing.json"},
        duration_s=0.1,
    )
    running = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert running["status"] == "running"
    assert set(running) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["inference_substrate_class"] == "blocked_no_run"
    assert blocked["grounding_measurement_complete_score"] == 0
    assert blocked["gate_check_summary"] == {
        "failed_check": "trace_artifact_path",
        "upstream": "results/experiment_7181_v633_qwen38_symbolic_traces.json",
        "field": "path",
        "expected_value": "readable_file",
        "observed_value": "missing_or_unreadable",
        "passed": False,
    }


def test_scenario_verify_7182_complete_artifact_replays_and_preserves_null(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7182-ARTIFACT cold-validates the complete measured result."""

    artifact, paths = built
    assert paths["result"].is_file()
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260910"
    assert artifact["inference_substrate_class"] == "no_model_load"
    assert artifact["grounding_measurement_complete_score"] == 1
    assert artifact["grounding_value_score"] in {0, 1}
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] in {"positive", "null"}
    assert len(artifact["rows"]) == 640
    assert exp.validate_artifact(artifact, root=REPO, check_source_hashes=True) == []

    changed = deepcopy(artifact)
    changed["rows"].pop()
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    errors = exp.validate_artifact(changed, root=REPO, check_source_hashes=False)
    assert "row_count" in errors


def test_scenario_verify_7182_validator_rejects_each_terminal_mutation(
    built: tuple[dict[str, object], dict[str, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7182-ARTIFACT covers every terminal consistency gate."""

    artifact, paths = built
    assert exp.validate_artifact(paths["result"], root=REPO, check_source_hashes=False) == []
    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_unreadable"]

    monkeypatch.setattr(exp, "summarize_arms", lambda _rows: artifact["arm_metrics"])
    monkeypatch.setattr(
        exp, "paired_cluster_bootstrap", lambda _rows, _seed, draws: artifact["paired_metrics"]
    )

    def errors_after(change: object) -> list[str]:
        changed = deepcopy(artifact)
        change(changed)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        return exp.validate_artifact(changed, root=REPO, check_source_hashes=False)

    mutations = [
        (lambda value: value.__setitem__("extra", True), "artifact_fields"),
        (lambda value: value["field_principles"].pop("rows"), "field_principles"),
        (lambda value: value.__setitem__("run_date", "20260909"), "run_date"),
        (lambda value: value.__setitem__("random_seed", 0), "random_seed"),
        (
            lambda value: value.__setitem__("inference_substrate_class", "model_full_generation"),
            "inference_substrate_class",
        ),
        (lambda value: value.__setitem__("arm_metrics", {}), "arm_metrics"),
        (lambda value: value.__setitem__("paired_metrics", {}), "paired_metrics"),
        (lambda value: value["independent_audit_rows"].pop(), "independent_audit_rows"),
        (lambda value: value["feature_lineage_rows"].pop(), "feature_lineage_rows"),
        (lambda value: value["intervention_rows"].pop(), "intervention_rows"),
        (
            lambda value: value["audit_receipt"].__setitem__("candidate_module_imported", True),
            "candidate_module_imported",
        ),
        (
            lambda value: value["audit_receipt"].__setitem__("threshold_adapted", True),
            "threshold_adapted",
        ),
        (
            lambda value: value["audit_receipt"].__setitem__("disagreement_count", 1),
            "audit_disagreement_count",
        ),
        (
            lambda value: value["independent_audit_rows"][0].__setitem__(
                "candidate_prediction", "changed"
            ),
            "independent_decision_mismatch",
        ),
        (
            lambda value: value["feature_lineage_rows"][0].__setitem__(
                "authority_used_for_features", True
            ),
            "authority_leakage",
        ),
        (
            lambda value: value["preconditions_checked"][0].__setitem__("passed", False),
            "preconditions_checked",
        ),
        (
            lambda value: value.__setitem__("grounding_measurement_complete_score", 2),
            "grounding_measurement_complete_score",
        ),
        (
            lambda value: value.__setitem__("grounding_value_score", 2),
            "grounding_value_score",
        ),
        (lambda value: value.__setitem__("verdict_class", "unknown"), "verdict_class"),
    ]
    for mutation, expected in mutations:
        assert expected in errors_after(mutation)

    changed = deepcopy(artifact)
    changed["reproducibility_checksum"] = "sha256:changed"
    assert "reproducibility_checksum" in exp.validate_artifact(changed, check_source_hashes=False)
    changed = deepcopy(artifact)
    changed["source_artifact_hashes"]["trace_artifact"] = "sha256:changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "source_artifact_hashes" in exp.validate_artifact(
        changed, root=REPO, check_source_hashes=True
    )

    blocked = exp._base_artifact(exp.RUN_DATE)
    blocked.update(
        {
            "status": "blocked",
            "verdict_class": "blocked",
            "gate_check_summary": {"passed": False},
            "honest_verdict": "blocked_test",
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    assert exp.validate_artifact(blocked, check_source_hashes=False) == []
    for field, replacement, expected in (
        ("verdict_class", "null", "blocked_verdict"),
        ("grounding_measurement_complete_score", 1, "blocked_measurement_score"),
        ("inference_substrate_class", "no_model_load", "blocked_substrate_class"),
        ("gate_check_summary", {"passed": True}, "blocked_gate_summary"),
    ):
        changed = deepcopy(blocked)
        changed[field] = replacement
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed, check_source_hashes=False)
    running = exp._base_artifact(exp.RUN_DATE)
    running["reproducibility_checksum"] = exp.artifact_checksum(running)
    assert "terminal_status" in exp.validate_artifact(running, check_source_hashes=False)


def test_scenario_verify_7182_scope_answer_keeps_positive_narrow() -> None:
    """SCENARIO-VERIFY-7182-VERDICT keeps even a positive answer pilot-scoped."""

    metrics = {
        "energy_from_extracted_tuples": {"accuracy": 0.75},
        "baseline_direct": {"accuracy": 0.5},
    }
    answer = exp._scope_answer(metrics, {"grounding_value_score": 1})
    assert "passed every preregistered" in answer
    assert "32-base pilot supports no broad" in answer


def test_scenario_verify_7182_builder_refuses_invalid_terminal_write(
    built: tuple[dict[str, object], dict[str, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7182-ARTIFACT stops before writing an invalid result."""

    _, built_paths = built
    audit = json.loads(built_paths["audit_result"].read_text(encoding="utf-8"))

    def fake_auditor(
        _root: Path, _auditor: Path, _request: Path, output: Path
    ) -> dict[str, object]:
        exp.atomic_write_json(output, audit, allow_override=False, sort_keys=True)
        return audit

    monkeypatch.setattr(exp, "_run_auditor", fake_auditor)
    monkeypatch.setattr(exp, "validate_artifact", lambda *args, **kwargs: ["forced"])
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    result = tmp_path / "result.json"
    with pytest.raises(ValueError, match="terminal_artifact_invalid:forced"):
        exp.build_artifact(
            REPO,
            exp.RUN_DATE,
            result_path=result,
            checkpoint_path=checkpoint_dir / "running.json",
            audit_request_path=checkpoint_dir / "request.json",
            audit_result_path=checkpoint_dir / "audit.json",
            duration_s=0.5,
        )
    assert not result.exists()


def test_scenario_verify_7182_verdict_requires_all_value_gates() -> None:
    """SCENARIO-VERIFY-7182-VERDICT separates complete measurement from value."""

    passing = {
        "accuracy_ci95_lower": 0.01,
        "false_accept_ci95_upper": 0.0,
        "no_leakage": True,
        "semantic_sensitivity": 0.8,
        "syntax_sensitivity": 0.5,
        "shuffle_sensitivity": 0.4,
        "audit_disagreement_count": 0,
        "provenance_corrupt": False,
    }
    assert exp.classify_verdict(passing) == {
        "grounding_measurement_complete_score": 1,
        "grounding_value_score": 1,
        "verifier_is_oracle": False,
        "verdict_class": "positive",
        "honest_verdict": "complete_positive_grounding_energy_pilot_value_gate_passed_narrow_scope",
    }
    null = deepcopy(passing)
    null["accuracy_ci95_lower"] = 0.0
    assert exp.classify_verdict(null)["verdict_class"] == "null"
    corrupt = deepcopy(passing)
    corrupt["provenance_corrupt"] = True
    assert exp.classify_verdict(corrupt)["verdict_class"] == "disqualified"
    circular = deepcopy(passing)
    circular["verifier_is_oracle"] = True
    classified = exp.classify_verdict(circular)
    assert classified["verdict_class"] == "circular_positive"
    assert classified["verifier_is_oracle"] is True


def test_req_verify_7182_field_principles_and_rosters(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """REQ-VERIFY-7182 explains every field and retains identical arm rosters."""

    artifact, _ = built
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"]["field_principles"] == (
        "Echo each field reason so the artifact explains its evidence contract."
    )
    rosters = {
        arm: [row["unit_id"] for row in artifact["rows"] if row["arm"] == arm] for arm in exp.ARMS
    }
    assert len({tuple(roster) for roster in rosters.values()}) == 1
    assert Counter(row["relation_family"] for row in artifact["rows"][:128]) == Counter(
        {family: 32 for family in exp.EVALUATION_FAMILIES}
    )
