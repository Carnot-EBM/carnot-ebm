"""Tests for REQ-VERIFY-7158 and SCENARIO-VERIFY-7158-*.

The checked-in relational fixture is read-only. All artifact writes use a
private temporary path, so this suite cannot rewrite the research record.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7158_v630_entity_evidence_fixture as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
UPSTREAM_PATH = REPO / "results/experiment_7138_v627_relational_fixture.json"


@pytest.fixture(scope="module")
def upstream() -> dict[str, object]:
    """Load the frozen source fixture without changing it."""

    return json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def materialized(upstream: dict[str, object]) -> dict[str, object]:
    """Build deterministic in-memory rows for focused checks."""

    return exp.materialize_fixture(upstream)


@pytest.fixture(scope="module")
def artifact(tmp_path_factory: pytest.TempPathFactory) -> dict[str, object]:
    """Build one complete artifact at a private output path."""

    path = tmp_path_factory.mktemp("exp7158") / "artifact.json"
    return exp.build_artifact(REPO, exp.RUN_DATE, result_path=path, duration_s=0.25)


def _rows_by_condition(materialized: dict[str, object]) -> dict[str, list[dict[str, object]]]:
    """Group primary rows by the controlled condition name."""

    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in materialized["rows"]:
        grouped[row["condition"]].append(row)
    return grouped


def _vectors_by_base(materialized: dict[str, object]) -> dict[str, dict[str, dict[str, int]]]:
    """Compute every candidate vector without reading its exact label."""

    grouped: dict[str, dict[str, dict[str, int]]] = defaultdict(dict)
    for row in materialized["rows"]:
        grouped[row["base_id"]][row["condition"]] = exp.compute_energy_vector(
            exp.energy_input(row)
        )
    return grouped


def test_req_verify_7158_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7158 owns every focused scenario and required field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7158") :]
    for scenario in (
        "PREFLIGHT",
        "PAIRS",
        "SPANS",
        "SPLITS",
        "SEALING",
        "ENERGY",
        "MUTATIONS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7158-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7158_pairs_cover_all_exact_conditions(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-PAIRS fixes nine rows for each source-backed base."""

    rows = materialized["rows"]
    assert len(rows) == exp.BASE_COUNT * len(exp.CONDITIONS) == 648
    assert Counter(row["source_family"] for row in rows) == Counter(
        {family: 162 for family in exp.SOURCE_FAMILIES}
    )
    assert Counter(row["condition"] for row in rows) == Counter(
        {condition: exp.BASE_COUNT for condition in exp.CONDITIONS}
    )
    assert {row["support_label"] for row in rows} == {"supported", "unsupported"}
    assert all(
        row["support_label"] == exp.CONDITION_LABELS[row["condition"]] for row in rows
    )
    assert all(
        row["expected_answer"] == ("yes" if row["support_label"] == "supported" else "no")
        for row in rows
    )
    per_base = Counter(row["base_id"] for row in rows)
    assert set(per_base.values()) == {len(exp.CONDITIONS)}


def test_scenario_verify_7158_entity_substitution_and_alias_dedup(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-MUTATIONS isolates entity changes and alias no-ops."""

    vectors = _vectors_by_base(materialized)
    for conditions in vectors.values():
        baseline = conditions["supported"]
        assert exp.changed_energy_terms(baseline, conditions["entity_substitution"]) == [
            "entity_presence"
        ]
        assert exp.changed_energy_terms(baseline, conditions["duplicate_alias"]) == []
    aliases = _rows_by_condition(materialized)["duplicate_alias"]
    assert all("duplicate" in row["perturbation"] for row in aliases)


def test_scenario_verify_7158_relation_negation_and_numeric_units(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-MUTATIONS isolates role, polarity, and unit terms."""

    vectors = _vectors_by_base(materialized)
    for conditions in vectors.values():
        baseline = conditions["supported"]
        assert exp.changed_energy_terms(baseline, conditions["relation_reversal"]) == [
            "relation_role_agreement"
        ]
        assert exp.changed_energy_terms(baseline, conditions["negation"]) == ["polarity"]
        assert exp.changed_energy_terms(baseline, conditions["numeric_unit_change"]) == [
            "quantity_unit_agreement"
        ]


def test_scenario_verify_7158_evidence_removal_and_irrelevant_controls(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-MUTATIONS detects removal and preserves irrelevant additions."""

    vectors = _vectors_by_base(materialized)
    for conditions in vectors.values():
        baseline = conditions["supported"]
        assert exp.changed_energy_terms(baseline, conditions["evidence_removal"]) == [
            "entity_presence",
            "counterfactual_sensitivity",
        ]
        assert exp.changed_energy_terms(baseline, conditions["irrelevant_evidence_only"]) == [
            "entity_presence",
            "counterfactual_sensitivity",
        ]
        assert exp.changed_energy_terms(baseline, conditions["irrelevant_evidence_added"]) == []


def test_scenario_verify_7158_spans_replay_exact_offsets(
    materialized: dict[str, object], upstream: dict[str, object]
) -> None:
    """SCENARIO-VERIFY-7158-SPANS checks source, claim, and entity slices."""

    assert exp.span_errors(materialized["rows"], materialized["entity_evidence_rows"], upstream) == []
    changed = deepcopy(materialized["rows"])
    changed[0]["claim_span"]["end"] -= 1
    assert "claim_span_mismatch" in exp.span_errors(
        changed, materialized["entity_evidence_rows"], upstream
    )[0]


def test_scenario_verify_7158_splits_are_frozen_by_source_family(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-SPLITS keeps families isolated and evaluation large."""

    split_rows = materialized["split_rows"]
    assert {row["source_family"] for row in split_rows} == set(exp.SOURCE_FAMILIES)
    assert {row["split"] for row in split_rows} == {"train", "calibration", "evaluation"}
    family_splits: dict[str, set[str]] = defaultdict(set)
    for row in materialized["rows"]:
        family_splits[row["source_family"]].add(row["split"])
    assert all(len(values) == 1 for values in family_splits.values())
    evaluation = [row for row in materialized["rows"] if row["split"] == "evaluation"]
    assert len(evaluation) >= 48
    assert len({row["base_id"] for row in evaluation}) >= 30
    assert exp.split_errors(materialized["rows"], split_rows) == []


def test_scenario_verify_7158_evaluation_truth_is_sealed(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-SEALING excludes truth and condition aliases."""

    evaluation = next(row for row in materialized["rows"] if row["split"] == "evaluation")
    prompt = exp.render_generation_prompt(evaluation)
    scorer_input = exp.energy_input(evaluation)
    serialized = (prompt + exp.canonical_json(scorer_input)).lower()
    assert all(token not in serialized for token in exp.SEALED_TOKENS)
    assert set(scorer_input) == set(exp.ENERGY_INPUT_FIELDS)
    assert exp.sealing_errors(materialized["rows"], materialized["sealed_field_rows"]) == []

    exposed = deepcopy(materialized["sealed_field_rows"])
    exposed[0]["energy_input_fields"].append("support_label")
    assert "sealed_energy_fields" in exp.sealing_errors(materialized["rows"], exposed)[0]


def test_scenario_verify_7158_calibration_freezes_energy_without_evaluation_truth(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-ENERGY keeps evaluation truth outside fitting."""

    rows = materialized["rows"]
    contract = exp.freeze_energy_contract(rows)
    assert contract["term_order"] == list(exp.ENERGY_TERMS)
    assert contract["fit_split"] == "calibration"
    assert contract["fit_source_families"] == ["Recent News"]
    assert len(contract["fit_fixture_ids"]) == 162
    assert set(contract["fit_fixture_ids"]) == {
        row["fixture_id"] for row in rows if row["split"] == "calibration"
    }
    assert contract["evaluation_truth_accessed"] is False
    assert contract["tie_rule"] == "energy_equal_threshold_is_supported"

    changed = deepcopy(rows)
    for row in changed:
        if row["split"] == "evaluation":
            row["support_label"] = "unsupported"
            row["expected_answer"] = "no"
    assert exp.freeze_energy_contract(changed) == contract


def test_scenario_verify_7158_mutation_receipts_are_per_unit(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-MUTATIONS keeps every intended term check visible."""

    receipts = materialized["mutation_test_rows"]
    assert len(receipts) == exp.BASE_COUNT * (len(exp.CONDITIONS) - 1)
    assert all(row["passed"] is True for row in receipts)
    assert {row["condition"] for row in receipts} == set(exp.CONDITIONS) - {"supported"}
    assert all(row["changed_terms"] == row["expected_changed_terms"] for row in receipts)


def test_scenario_verify_7158_preflight_writes_complete_blocked_state_first(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7158-PREFLIGHT writes complete shape before a missing input."""

    writes: list[dict[str, object]] = []
    original = exp.atomic_write_json

    def capture(path: Path, payload: dict[str, object], **kwargs: object) -> Path:
        writes.append(deepcopy(payload))
        return original(path, payload, **kwargs)

    monkeypatch.setattr(exp, "atomic_write_json", capture)
    missing = tmp_path / "missing-upstream.json"
    blocked = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "blocked.json",
        source_paths={"exp7138_artifact": missing},
        duration_s=0.1,
    )
    assert set(writes[0]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert writes[0]["status"] == "running"
    assert writes[0]["inference_substrate_class"] == "blocked_no_run"
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["counterfactual_fixture_ready_score"] == 0
    assert blocked["gate_check_summary"] == {
        "failed_check": "exp7138_artifact_path",
        "expected_value": "readable_file",
        "observed_value": "missing_or_unreadable",
        "passed": False,
    }
    assert exp.validate_artifact(blocked, root=REPO) == []


def test_scenario_verify_7158_artifact_is_complete_and_detects_tampering(
    artifact: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7158-ARTIFACT replays rows, sources, state, and checksum."""

    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["inference_substrate"] == "exact_source_fixture_construction"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["counterfactual_fixture_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive")
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert exp.validate_artifact(artifact, root=REPO) == []

    changed = deepcopy(artifact)
    changed["rows"][0]["support_label"] = "unsupported"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert any("rows" in error or "label" in error for error in exp.validate_artifact(changed, root=REPO))

    changed_hash = deepcopy(artifact)
    changed_hash["source_artifact_hashes"]["exp7138_artifact"] = "sha256:changed"
    changed_hash["reproducibility_checksum"] = exp.artifact_checksum(changed_hash)
    assert "source_artifact_hashes_mismatch" in exp.validate_artifact(changed_hash, root=REPO)


def test_scenario_verify_7158_sources_include_license_and_modules(
    artifact: dict[str, object],
) -> None:
    """REQ-VERIFY-7158 records admissible data and every checked source input."""

    hashes = artifact["source_artifact_hashes"]
    assert set(hashes) == set(exp.SOURCE_PATHS)
    assert all(str(value).startswith("sha256:") for value in hashes.values())
    license_check = next(
        row for row in artifact["preconditions_checked"] if row["check"] == "ragtruth_license"
    )
    assert license_check["passed"] is True
    assert license_check["observed_value"]["spdx_id"] == "MIT"
    assert license_check["observed_value"]["upstream"] == "ParticleMedia/RAGTruth"


def test_req_verify_7158_cli_build_validate_and_rejects_wrong_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-7158 exposes one deterministic build and validation command."""

    result_path = tmp_path / "cli-artifact.json"
    assert exp.main(["--date", exp.RUN_DATE, "--result-path", str(result_path)]) == 0
    output = capsys.readouterr().out
    assert '"counterfactual_fixture_ready_score":1' in output.replace(" ", "")
    assert exp.main(["--validate", str(result_path)]) == 0
    assert '"valid":true' in capsys.readouterr().out.replace(" ", "")
    assert exp.main(["--date", "20260908", "--result-path", str(tmp_path / "wrong.json")]) == 2


def test_scenario_verify_7158_adversarial_local_validators_cover_failures(
    materialized: dict[str, object], upstream: dict[str, object]
) -> None:
    """SCENARIO-VERIFY-7158-ARTIFACT exposes every local integrity failure."""

    with pytest.raises(ValueError, match="expected 72"):
        exp.materialize_fixture({"rows": []})
    wrong_family = deepcopy(upstream)
    wrong_family["rows"][0]["source_family"] = "unknown"
    with pytest.raises(ValueError, match="source families"):
        exp.materialize_fixture(wrong_family)
    assert exp._span_matches("abc", {"start": "0", "end": 1}) is False

    rows = deepcopy(materialized["rows"])
    rows[0]["source_fixture_id"] = "missing-source"
    rows[0]["evidence_text_sha256"] = "sha256:wrong"
    entity_rows = deepcopy(materialized["entity_evidence_rows"])
    complete = next(
        row
        for row in entity_rows
        if row["source_spans"] and row["claim_spans"] and row["evidence_spans"]
    )
    missing_parent = deepcopy(complete)
    missing_parent["fixture_id"] = "missing-parent"
    entity_rows.append(missing_parent)
    complete["source_spans"][0]["end"] -= 1
    claim_receipt = next(row for row in entity_rows if row["claim_spans"] and row is not complete)
    claim_receipt["claim_spans"][0]["end"] -= 1
    evidence_receipt = next(
        row for row in entity_rows if row["evidence_spans"] and row not in (complete, claim_receipt)
    )
    evidence_receipt["evidence_spans"][0]["end"] -= 1
    errors = exp.span_errors(rows, entity_rows, upstream)
    assert {
        "source_span_mismatch",
        "evidence_text_hash_mismatch",
        "entity_parent_missing",
        "entity_source_span_mismatch",
        "entity_claim_span_mismatch",
        "entity_evidence_span_mismatch",
    } <= {error.split(":", 1)[0] for error in errors}

    leaked = deepcopy(materialized["rows"])
    leaked[0]["split"] = "train" if leaked[0]["split"] != "train" else "evaluation"
    assert "source_family_split_leakage" in exp.split_errors(leaked, materialized["split_rows"])
    undersized = list(materialized["rows"][:10])
    split_failures = exp.split_errors(undersized, materialized["split_rows"][:-1])
    assert {
        "split_family_roster_mismatch",
        "frozen_family_split_mismatch",
        "evaluation_row_shortfall",
        "evaluation_pair_shortfall",
    } <= set(split_failures)

    sealed = deepcopy(materialized["sealed_field_rows"])
    assert "sealed_evaluation_roster_mismatch" in exp.sealing_errors(
        materialized["rows"], sealed[1:]
    )
    exposed_rows = deepcopy(materialized["rows"])
    exposed_id = sealed[0]["fixture_id"]
    exposed_row = next(row for row in exposed_rows if row["fixture_id"] == exposed_id)
    exposed_row["evidence_text"] += " support_label"
    exposed_row["support_label"] = "tampered"
    sealed[0]["energy_input_sha256"] = "sha256:wrong"
    sealing_failures = exp.sealing_errors(exposed_rows, sealed)
    assert {
        "sealed_token_exposure",
        "sealed_prompt_hash",
        "sealed_energy_hash",
        "sealed_truth_hash",
    } <= {error.split(":", 1)[0] for error in sealing_failures}

    broken = deepcopy(materialized)
    broken["rows"] = broken["rows"][:-1]
    broken["rows"][0]["support_label"] = "unsupported"
    broken["mutation_test_rows"] = broken["mutation_test_rows"][:-1]
    broken["energy_term_contract"] = {}
    fixture_failures = exp._fixture_errors(broken, upstream)
    assert {
        "row_count_mismatch",
        "condition_count_mismatch",
        "exact_label_mismatch",
        "mutation_tests_mismatch",
        "energy_term_contract_mismatch",
    } <= set(fixture_failures)


def test_scenario_verify_7158_precondition_failure_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7158-PREFLIGHT retains exact receipts for each gate."""

    paths = exp._resolved_source_paths(REPO)
    _, failure, _, _ = exp._preconditions(
        REPO, paths, tmp_path / "unused.json", run_date="20260908"
    )
    assert failure["check"] == "run_date"
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "sha256_file", lambda _path: (_ for _ in ()).throw(OSError()))
        _, failure, _, _ = exp._preconditions(REPO, paths, tmp_path / "unused.json")
        assert failure["check"] == "exp7138_artifact_path"

    altered = tmp_path / "altered.json"
    altered.write_text("{}\n", encoding="utf-8")
    changed_paths = dict(paths, exp7138_artifact=altered)
    _, failure, _, _ = exp._preconditions(REPO, changed_paths, tmp_path / "unused.json")
    assert failure["check"] == "exp7138_artifact_hash"

    bad_license = tmp_path / "LICENSE"
    bad_license.write_text("not a license\n", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(
            exp,
            "PINNED_HASHES",
            {"exp7138_artifact": exp.sha256_file(paths["exp7138_artifact"])},
        )
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, ragtruth_license=bad_license), tmp_path / "unused.json"
        )
        assert failure["check"] == "ragtruth_license"

    bad_spec = tmp_path / "spec.md"
    bad_spec.write_text("no requirement\n", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, constraint_spec=bad_spec), tmp_path / "unused.json"
        )
        assert failure["check"] == "constraint_spec_requirement"

    invalid_json = tmp_path / "invalid.json"
    invalid_json.write_text("{", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, exp7138_artifact=invalid_json), tmp_path / "unused.json"
        )
        assert failure["check"] == "exp7138_artifact_json"

    list_json = tmp_path / "list.json"
    list_json.write_text("[]\n", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, exp7138_artifact=list_json), tmp_path / "unused.json"
        )
        assert failure["check"] == "exp7138_artifact_shape"

    from carnot import experiment_7138_v627_relational_fixture as exp7138

    with monkeypatch.context() as scoped:
        scoped.setattr(exp7138, "validate_artifact", lambda _value: ["forced"])
        _, failure, _, _ = exp._preconditions(REPO, paths, tmp_path / "unused.json")
        assert failure["check"] == "exp7138_fixture_validation"

    upstream = json.loads(paths["exp7138_artifact"].read_text(encoding="utf-8"))
    not_ready = deepcopy(upstream)
    not_ready["source_grounding_fixture_ready_score"] = 0
    not_ready_path = tmp_path / "not-ready.json"
    not_ready_path.write_text(json.dumps(not_ready), encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        scoped.setattr(exp7138, "validate_artifact", lambda _value: [])
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, exp7138_artifact=not_ready_path), tmp_path / "unused.json"
        )
        assert failure["check"] == "exp7138_fixture_ready_score"

    wrong_families = deepcopy(upstream)
    wrong_families["rows"][0]["source_family"] = "unknown"
    wrong_families_path = tmp_path / "wrong-families.json"
    wrong_families_path.write_text(json.dumps(wrong_families), encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        scoped.setattr(exp7138, "validate_artifact", lambda _value: [])
        _, failure, _, _ = exp._preconditions(
            REPO, dict(paths, exp7138_artifact=wrong_families_path), tmp_path / "unused.json"
        )
        assert failure["check"] == "exp7138_source_families"

    _, failure, _, _ = exp._preconditions(REPO, paths, tmp_path / "never-written.json")
    assert failure["check"] == "output_path"


def test_scenario_verify_7158_terminal_validator_failure_matrix(
    artifact: dict[str, object], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7158-ARTIFACT rejects malformed and inconsistent states."""

    assert exp.validate_artifact(tmp_path / "missing.json") == ["artifact_missing"]
    unreadable = tmp_path / "unreadable.json"
    unreadable.write_text("{", encoding="utf-8")
    assert exp.validate_artifact(unreadable) == ["artifact_unreadable"]
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp.validate_artifact(scalar) == ["artifact_not_object"]
    assert exp.validate_artifact(7) == ["artifact_not_object"]
    assert "artifact_fields_mismatch" in exp.validate_artifact({})[0]

    malformed = deepcopy(artifact)
    malformed.update(
        {
            "field_principles": {},
            "run_date": "wrong",
            "inference_substrate": "wrong",
            "execution_venue": "wrong",
            "duration_s": -1,
            "random_seed": -1,
            "verifier_is_oracle": True,
            "status": "wrong",
            "inference_substrate_class": "wrong",
            "counterfactual_fixture_ready_score": 0,
            "gate_check_summary": {},
            "verdict_class": "partial",
            "honest_verdict": "wrong",
            "reproducibility_checksum": "wrong",
        }
    )
    malformed_errors = exp.validate_artifact(malformed, root=REPO)
    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "duration_s_invalid",
        "random_seed_mismatch",
        "verifier_is_oracle_mismatch",
        "reproducibility_checksum_mismatch",
        "complete_status_mismatch",
        "complete_inference_substrate_class_mismatch",
        "counterfactual_fixture_ready_score_mismatch",
        "complete_gate_check_summary_mismatch",
        "verdict_class_mismatch",
        "honest_verdict_mismatch",
    } <= set(malformed_errors)

    failure = exp._gate("forced", True, False, False)
    blocked = exp._blocked_artifact(exp._base_artifact(exp.RUN_DATE), [failure], failure, {}, 0.0)
    blocked.update(
        {
            "status": "wrong",
            "inference_substrate_class": "wrong",
            "gate_check_summary": {},
            "verdict_class": "wrong",
            "counterfactual_fixture_ready_score": 1,
            "honest_verdict": "wrong",
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_errors = exp.validate_artifact(blocked, root=REPO)
    assert {
        "blocked_status_mismatch",
        "blocked_inference_substrate_class_mismatch",
        "blocked_gate_check_summary_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_readiness_mismatch",
        "blocked_honest_verdict_mismatch",
    } <= set(blocked_errors)

    assert "source_artifact_missing" in exp.validate_artifact(artifact, root=tmp_path)
    for name, relative in exp.SOURCE_PATHS.items():
        destination = tmp_path / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((REPO / relative).read_bytes())
    (tmp_path / exp.SOURCE_PATHS["exp7138_artifact"]).write_text("{", encoding="utf-8")
    assert any(
        error.startswith("independent_replay_failed")
        for error in exp.validate_artifact(artifact, root=tmp_path)
    )

    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "_fixture_errors", lambda _fixture, _upstream: ["forced"])
        blocked_build = exp.build_artifact(
            REPO,
            exp.RUN_DATE,
            result_path=tmp_path / "forced-blocked.json",
            duration_s=0.1,
        )
    assert blocked_build["gate_check_summary"]["failed_check"] == "fixture_integrity"
