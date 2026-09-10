"""Tests for REQ-VERIFY-7180 and SCENARIO-VERIFY-7180-*.

The checked-in Exp7158 artifact stays read-only. Tests write checkpoints,
sidecars, and terminal artifacts only below private temporary directories.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7180_v633_symbolic_edit_fixture as exp


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "openspec/capabilities/constraint-verification/spec.md"
UPSTREAM_PATH = REPO / "results/experiment_7158_v630_entity_evidence_fixture.json"


@pytest.fixture(scope="module")
def upstream() -> dict[str, object]:
    """Load the frozen source fixture without changing the evidence record."""

    return json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def materialized(upstream: dict[str, object]) -> dict[str, object]:
    """Build the complete symbolic fixture once for focused assertions."""

    return exp.materialize_fixture(upstream)


@pytest.fixture(scope="module")
def built(tmp_path_factory: pytest.TempPathFactory) -> tuple[dict[str, object], dict[str, Path]]:
    """Build one complete artifact and both sidecars in a private directory."""

    directory = tmp_path_factory.mktemp("exp7180")
    paths = {
        "result": directory / "artifact.json",
        "checkpoint": directory / "checkpoints" / "checkpoint.json",
        "generation": directory / "generation.jsonl",
        "authority": directory / "authority.jsonl",
    }
    paths["checkpoint"].parent.mkdir()
    artifact = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=paths["result"],
        checkpoint_path=paths["checkpoint"],
        generation_view_path=paths["generation"],
        authority_sidecar_path=paths["authority"],
        duration_s=0.25,
    )
    return artifact, paths


def test_req_verify_7180_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7180 owns every focused scenario and required artifact field."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-VERIFY-7180") :]
    for scenario in (
        "PREFLIGHT",
        "SPLITS",
        "VARIANTS",
        "AUTHORITY",
        "BLINDING",
        "ENERGY",
        "CONTROLS",
        "ARTIFACT",
    ):
        assert f"SCENARIO-VERIFY-7180-{scenario}" in section
    assert all(f"`{field}`" in section for field in exp.REQUIRED_ARTIFACT_FIELDS)


def test_scenario_verify_7180_splits_use_six_whole_families(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-SPLITS freezes 16 calibration and 32 evaluation bases."""

    rows = materialized["internal_rows"]
    manifest = materialized["split_manifest"]
    assert len({row["base_id"] for row in rows}) == exp.BASE_COUNT == 48
    assert Counter(row["relation_family"] for row in rows) == Counter(
        {family: 32 for family in exp.RELATION_FAMILIES}
    )
    assert {row["relation_family"] for row in rows if row["split"] == "calibration"} == set(
        exp.CALIBRATION_FAMILIES
    )
    assert {row["relation_family"] for row in rows if row["split"] == "evaluation"} == set(
        exp.EVALUATION_FAMILIES
    )
    assert manifest["calibration_base_count"] == 16
    assert manifest["evaluation_base_count"] == 32
    assert manifest["calibration_row_count"] == 64
    assert manifest["evaluation_row_count"] == 128
    base_splits: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        base_splits[row["base_id"]].add(row["split"])
    assert all(len(splits) == 1 for splits in base_splits.values())


def test_scenario_verify_7180_variants_preserve_and_change_labels(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-VARIANTS distinguishes surface and semantic edits."""

    rows = materialized["internal_rows"]
    labels = {row["unit_id"]: row for row in materialized["authority_rows"]}
    assert len(rows) == exp.BASE_COUNT * len(exp.VARIANTS) == 192
    assert Counter(row["variant"] for row in rows) == Counter(
        {variant: exp.BASE_COUNT for variant in exp.VARIANTS}
    )
    grouped: dict[str, dict[str, dict[str, object]]] = defaultdict(dict)
    for row in rows:
        grouped[row["base_id"]][row["variant"]] = labels[row["unit_id"]]
    for variants in grouped.values():
        original = variants["original"]["expected_response"]["direct_decision"]
        assert (
            variants["bijective_entity_rename"]["expected_response"]["direct_decision"] == original
        )
        assert (
            variants["relation_or_polarity_flip"]["expected_response"]["direct_decision"]
            != original
        )
        assert variants["evidence_deletion"]["expected_response"]["direct_decision"] != original
    rename_rows = [row for row in rows if row["variant"] == "bijective_entity_rename"]
    assert all(len(set(row["rename_map"].values())) == 2 for row in rename_rows)


def test_scenario_verify_7180_authorities_agree_independently(
    materialized: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7180-AUTHORITY gets exact agreement without scorer reuse."""

    rows = materialized["internal_rows"]
    symbolic = {row["unit_id"]: exp.authority_interpret(row) for row in rows}
    sqlite = exp.sqlite_cross_check(rows)
    assert symbolic == sqlite
    assert len(symbolic) == 192

    response = next(iter(symbolic.values()))
    source = next(row["source_text"] for row in rows if row["unit_id"] == next(iter(symbolic)))
    monkeypatch.setattr(
        exp, "authority_interpret", lambda _row: (_ for _ in ()).throw(AssertionError)
    )
    monkeypatch.setattr(
        exp, "sqlite_cross_check", lambda _rows: (_ for _ in ()).throw(AssertionError)
    )
    assert exp.compute_candidate_energy(response, source)["total"] == 0


def test_scenario_verify_7180_generation_view_is_minimal_and_label_invariant(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-BLINDING keeps labels outside exact generation bytes."""

    generation = materialized["generation_rows"]
    authority = materialized["authority_rows"]
    assert len(generation) == 192
    assert all(set(row) == {"unit_id", "text", "response_schema"} for row in generation)
    assert exp.generation_view_errors(generation, authority) == []
    before = exp.jsonl_bytes(generation)
    changed_labels = deepcopy(authority)
    changed_labels[0]["expected_response"]["direct_decision"] = "unsupported"
    assert exp.jsonl_bytes(generation) == before

    exposed = deepcopy(generation)
    exposed[0]["split"] = "evaluation"
    assert "generation_view_shape" in exp.generation_view_errors(exposed, authority)[0]


def test_scenario_verify_7180_energy_checks_tuples_units_fields_and_spans(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-ENERGY scores only a response and supplied source bytes."""

    original = next(row for row in materialized["internal_rows"] if row["variant"] == "original")
    response = exp.authority_interpret(original)
    assert exp.compute_candidate_energy(response, original["source_text"]) == {
        "tuple_alignment": 0,
        "polarity": 0,
        "quantity_unit_agreement": 0,
        "literal_span_validity": 0,
        "missing_required_fields": 0,
        "total": 0,
    }

    bad = deepcopy(response)
    bad["evidence_tuple"]["polarity"] = "negative"
    assert exp.compute_candidate_energy(bad, original["source_text"])["polarity"] == 1
    bad = deepcopy(response)
    bad["evidence_tuple"]["unit"] = "bytes"
    assert (
        exp.compute_candidate_energy(bad, original["source_text"])["quantity_unit_agreement"] == 1
    )
    bad = deepcopy(response)
    bad["source_end"] -= 1
    assert exp.compute_candidate_energy(bad, original["source_text"])["literal_span_validity"] == 1
    bad = deepcopy(response)
    del bad["claim_tuple"]
    assert (
        exp.compute_candidate_energy(bad, original["source_text"])["missing_required_fields"] == 1
    )


def test_scenario_verify_7180_controls_are_frozen_before_inference(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-CONTROLS fixes arms, denominators, and three controls."""

    contract = materialized["score_contract"]
    assert contract["frozen_before_inference"] is True
    assert contract["fit_split"] == "calibration"
    assert contract["scoring_denominators"] == {
        "all_model_visible_rows": 192,
        "calibration_rows": 64,
        "evaluation_rows_per_arm": 128,
    }
    assert set(contract["comparison_arms"]) == set(exp.COMPARISON_ARMS)
    assert len(contract["seeded_shuffle_controls"]) == 2
    assert contract["label_permutation_control"]["seed"] == exp.LABEL_PERMUTATION_SEED
    assert contract["oracle_upper_bound"]["deployable_extraction_arm"] is False

    changed = deepcopy(materialized["authority_rows"])
    for row in changed:
        if row["split"] == "evaluation":
            row["expected_response"]["direct_decision"] = "abstain"
    assert exp.freeze_score_contract(materialized["internal_rows"], changed) == contract


def test_scenario_verify_7180_mutations_have_exact_independent_receipts(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-VARIANTS retains rename invariance and negative rows."""

    mutations = materialized["mutation_rows"]
    assert len(mutations) == exp.BASE_COUNT * 3 == 144
    assert all(row["passed"] is True for row in mutations)
    assert Counter(row["variant"] for row in mutations) == Counter(
        {
            "bijective_entity_rename": 48,
            "relation_or_polarity_flip": 48,
            "evidence_deletion": 48,
        }
    )
    assert sum(row["decision_changed"] for row in mutations) == 96
    assert all(
        row["decision_changed"] is False
        for row in mutations
        if row["variant"] == "bijective_entity_rename"
    )


def test_scenario_verify_7180_preflight_writes_checkpoint_then_blocked_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7180-PREFLIGHT retains exact failure evidence."""

    writes: list[tuple[Path, dict[str, object]]] = []
    original = exp.atomic_write_json

    def capture(path: Path, payload: dict[str, object], **kwargs: object) -> Path:
        writes.append((Path(path), deepcopy(payload)))
        return original(path, payload, **kwargs)

    monkeypatch.setattr(exp, "atomic_write_json", capture)
    checkpoint = tmp_path / "checkpoints" / "checkpoint.json"
    checkpoint.parent.mkdir()
    result = tmp_path / "result.json"
    blocked = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=result,
        checkpoint_path=checkpoint,
        generation_view_path=tmp_path / "generation.jsonl",
        authority_sidecar_path=tmp_path / "authority.jsonl",
        source_paths={"exp7158_artifact": tmp_path / "missing.json"},
        duration_s=0.1,
    )
    assert writes[0][0] == checkpoint
    assert writes[0][1]["status"] == "running"
    assert set(writes[0][1]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert all(path != result for path, payload in writes[:-1] if payload["status"] == "running")
    assert blocked["status"] == "blocked"
    assert blocked["verdict_class"] == "blocked"
    assert blocked["fixture_ready_score"] == 0
    assert blocked["gate_check_summary"] == {
        "failed_check": "exp7158_artifact_path",
        "upstream": "results/experiment_7158_v630_entity_evidence_fixture.json",
        "field": "path",
        "expected_value": "readable_file",
        "observed_value": "missing_or_unreadable",
        "passed": False,
    }


def test_scenario_verify_7180_complete_artifact_and_sidecars_replay(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """SCENARIO-VERIFY-7180-ARTIFACT validates the full file-to-parser path."""

    artifact, paths = built
    assert set(artifact) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete"
    assert artifact["run_date"] == "20260910"
    assert artifact["inference_substrate"] == "exact_source_fixture_construction"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert artifact["execution_venue"] == "host"
    assert artifact["fixture_ready_score"] == 1
    assert artifact["verifier_is_oracle"] is False
    assert artifact["verdict_class"] == "positive"
    assert str(artifact["honest_verdict"]).startswith("complete_positive")
    assert len(artifact["rows"]) == 192
    assert len(paths["generation"].read_text(encoding="utf-8").splitlines()) == 192
    assert len(paths["authority"].read_text(encoding="utf-8").splitlines()) == 192
    assert (
        exp.validate_artifact(
            artifact,
            root=REPO,
            generation_view_path=paths["generation"],
            authority_sidecar_path=paths["authority"],
        )
        == []
    )

    changed = deepcopy(artifact)
    changed["rows"][0]["generation_row_sha256"] = "sha256:changed"
    changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
    assert "rows_mismatch" in exp.validate_artifact(
        changed,
        root=REPO,
        generation_view_path=paths["generation"],
        authority_sidecar_path=paths["authority"],
    )


def test_req_verify_7180_field_principles_and_exact_source_receipts(
    built: tuple[dict[str, object], dict[str, Path]],
) -> None:
    """REQ-VERIFY-7180 explains every field and binds the exact upstream bytes."""

    artifact, _ = built
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["field_principles"]["field_principles"] == (
        "Echo each field reason so the artifact explains its evidence contract."
    )
    assert (
        artifact["source_artifact_hashes"]["exp7158_artifact"]
        == exp.PINNED_HASHES["exp7158_artifact"]
    )
    gate_names = {row["check"] for row in artifact["preconditions_checked"]}
    assert {
        "run_date",
        "constraint_spec_requirement",
        "exp7158_artifact_hash",
        "exp7158_same_milestone_fields",
        "sqlite_version",
        "python_executable",
        "terminal_output_directory",
        "checkpoint_output_directory",
    } <= gate_names


def test_req_verify_7180_cli_build_validate_and_wrong_date(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """REQ-VERIFY-7180 exposes the executable build and cold-validation path."""

    checkpoint = tmp_path / "checkpoints" / "checkpoint.json"
    checkpoint.parent.mkdir()
    args = [
        "--date",
        exp.RUN_DATE,
        "--result-path",
        str(tmp_path / "artifact.json"),
        "--checkpoint-path",
        str(checkpoint),
        "--generation-view-path",
        str(tmp_path / "generation.jsonl"),
        "--authority-sidecar-path",
        str(tmp_path / "authority.jsonl"),
    ]
    assert exp.main(args) == 0
    assert '"fixture_ready_score":1' in capsys.readouterr().out.replace(" ", "")
    assert (
        exp.main(
            [
                "--validate",
                str(tmp_path / "artifact.json"),
                "--generation-view-path",
                str(tmp_path / "generation.jsonl"),
                "--authority-sidecar-path",
                str(tmp_path / "authority.jsonl"),
            ]
        )
        == 0
    )
    assert '"valid":true' in capsys.readouterr().out.replace(" ", "")
    assert exp.main(["--date", "20260909"]) == 2


def test_scenario_verify_7180_generation_and_energy_failure_matrix(
    materialized: dict[str, object],
) -> None:
    """SCENARIO-VERIFY-7180-BLINDING and ENERGY reject each malformed input class."""

    generation = deepcopy(materialized["generation_rows"])
    authority = materialized["authority_rows"]
    generation[0]["unit_id"] = "descriptive-id"
    generation[1]["response_schema"] = {}
    generation[2]["text"] += " expected_response"
    generation[3]["unit_id"] = generation[4]["unit_id"]
    failures = exp.generation_view_errors(generation, authority)
    assert {
        "generation_unit_id_not_opaque",
        "generation_response_schema",
        "generation_private_token",
        "generation_unit_id_duplicate",
        "generation_authority_roster_mismatch",
    } <= {failure.split(":", 1)[0] for failure in failures}

    deletion = next(
        row for row in materialized["internal_rows"] if row["variant"] == "evidence_deletion"
    )
    response = exp.authority_interpret(deletion)
    assert exp.compute_candidate_energy(response, "")["tuple_alignment"] == 1
    response["source_start"] = 0
    assert exp.compute_candidate_energy(response, "unexpected")["literal_span_validity"] == 1

    original = next(row for row in materialized["internal_rows"] if row["variant"] == "original")
    response = exp.authority_interpret(original)
    response["evidence_tuple"]["subject"] = "different"
    assert exp.compute_candidate_energy(response, original["source_text"])["tuple_alignment"] == 1
    response = exp.authority_interpret(original)
    response["claim_tuple"]["polarity"] = "unknown"
    assert (
        exp.compute_candidate_energy(response, original["source_text"])["missing_required_fields"]
        == 1
    )
    response = exp.authority_interpret(original)
    response["evidence_tuple"]["extra"] = "forbidden"
    assert (
        exp.compute_candidate_energy(response, original["source_text"])["missing_required_fields"]
        == 1
    )
    response = exp.authority_interpret(original)
    response["source_start"] = True
    assert (
        exp.compute_candidate_energy(response, original["source_text"])["literal_span_validity"]
        == 1
    )
    response = exp.authority_interpret(original)
    assert (
        exp.compute_candidate_energy(response, "x" * len(original["source_text"]))[
            "literal_span_validity"
        ]
        == 1
    )


def test_scenario_verify_7180_structural_failure_matrix(
    materialized: dict[str, object], upstream: dict[str, object]
) -> None:
    """SCENARIO-VERIFY-7180-ARTIFACT recomputes each structural readiness rule."""

    with pytest.raises(ValueError, match="expected 72"):
        exp.materialize_fixture({"rows": []})
    broken = deepcopy(materialized)
    removed_base = broken["internal_rows"][0]["base_id"]
    broken["internal_rows"] = [
        row for row in broken["internal_rows"] if row["base_id"] != removed_base
    ]
    broken["internal_rows"][0]["relation_family"] = "unknown"
    broken["internal_rows"][0]["variant"] = "unknown"
    broken["internal_rows"][0]["split"] = (
        "evaluation" if broken["internal_rows"][1]["split"] == "calibration" else "calibration"
    )
    broken["authority_rows"] = broken["authority_rows"][:-1]
    broken["authority_rows"][0]["authority_agreement"] = False
    broken["mutation_rows"] = broken["mutation_rows"][:-1]
    broken["mutation_rows"][0]["passed"] = False
    broken["split_manifest"] = {}
    broken["score_contract"] = {}
    failures = set(exp.structural_errors(broken))
    assert {
        "model_visible_row_count",
        "base_count",
        "relation_family_counts",
        "variant_counts",
        "split_leakage",
        "authority_agreement",
        "mutation_receipts",
        "split_manifest",
        "score_contract",
    } <= failures
    assert len(exp.materialize_fixture(upstream)["rows"]) == 192


def test_scenario_verify_7180_precondition_failure_matrix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7180-PREFLIGHT keeps every external failure exact."""

    paths = exp._resolved_source_paths(REPO)
    outputs = (
        tmp_path / "result.json",
        tmp_path / "checkpoint.json",
        tmp_path / "generation.jsonl",
        tmp_path / "authority.jsonl",
    )

    def failure_for(
        candidate_paths: dict[str, Path], *, run_date: str = exp.RUN_DATE
    ) -> dict[str, object]:
        _, failure, _, _ = exp._preconditions(REPO, candidate_paths, *outputs, run_date=run_date)
        assert failure is not None
        return failure

    assert failure_for(paths, run_date="20260909")["check"] == "run_date"
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "sha256_file", lambda _path: (_ for _ in ()).throw(OSError()))
        assert failure_for(paths)["check"] == "exp7158_artifact_path"
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {"exp7158_artifact": "sha256:wrong"})
        assert failure_for(paths)["check"] == "exp7158_artifact_hash"

    bad_spec = tmp_path / "spec.md"
    bad_spec.write_text("no requirement\n", encoding="utf-8")
    assert failure_for(dict(paths, constraint_spec=bad_spec))["check"] == (
        "constraint_spec_requirement"
    )
    invalid = tmp_path / "invalid.json"
    invalid.write_text("{", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        assert failure_for(dict(paths, exp7158_artifact=invalid))["check"] == (
            "exp7158_artifact_json"
        )
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]\n", encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        assert failure_for(dict(paths, exp7158_artifact=scalar))["check"] == (
            "exp7158_artifact_shape"
        )

    upstream = json.loads(UPSTREAM_PATH.read_text(encoding="utf-8"))
    damaged = deepcopy(upstream)
    damaged["rows"] = damaged["rows"][:-1]
    damaged_path = tmp_path / "damaged.json"
    damaged_path.write_text(json.dumps(damaged), encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        assert failure_for(dict(paths, exp7158_artifact=damaged_path))["check"] == (
            "exp7158_artifact_validation"
        )
    changed_gate = deepcopy(upstream)
    changed_gate["status"] = "blocked"
    from carnot import experiment_7158_v630_entity_evidence_fixture as exp7158

    changed_gate["reproducibility_checksum"] = exp7158.artifact_checksum(changed_gate)
    changed_gate_path = tmp_path / "changed-gate.json"
    changed_gate_path.write_text(json.dumps(changed_gate), encoding="utf-8")
    with monkeypatch.context() as scoped:
        scoped.setattr(exp, "PINNED_HASHES", {})
        assert failure_for(dict(paths, exp7158_artifact=changed_gate_path))["check"] == (
            "exp7158_same_milestone_fields"
        )
    with monkeypatch.context() as scoped:
        scoped.setattr(exp.sqlite3, "sqlite_version", "")
        assert failure_for(paths)["check"] == "sqlite_version"
    missing_parent = tmp_path / "missing" / "result.json"
    _, failure, _, _ = exp._preconditions(
        REPO,
        paths,
        missing_parent,
        outputs[1],
        outputs[2],
        outputs[3],
    )
    assert failure["check"] == "terminal_output_directory"
    assert exp._display_path(REPO, REPO / "results/example.json") == "results/example.json"


def test_scenario_verify_7180_build_blocks_on_structural_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7180-ARTIFACT cannot write ready sidecars after contract drift."""

    checkpoint = tmp_path / "checkpoints" / "checkpoint.json"
    checkpoint.parent.mkdir()
    original = exp.freeze_score_contract
    calls = 0

    def drifting(rows: object, authority: object) -> dict[str, object]:
        nonlocal calls
        calls += 1
        return original(rows, authority) if calls == 1 else {}

    monkeypatch.setattr(exp, "freeze_score_contract", drifting)
    blocked = exp.build_artifact(
        REPO,
        exp.RUN_DATE,
        result_path=tmp_path / "blocked.json",
        checkpoint_path=checkpoint,
        generation_view_path=tmp_path / "never-generation.jsonl",
        authority_sidecar_path=tmp_path / "never-authority.jsonl",
        duration_s=0.1,
    )
    assert blocked["status"] == "blocked"
    assert blocked["gate_check_summary"]["failed_check"] == "fixture_integrity"
    assert not (tmp_path / "never-generation.jsonl").exists()


def test_scenario_verify_7180_validator_failure_matrix(
    built: tuple[dict[str, object], dict[str, Path]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-VERIFY-7180-ARTIFACT rejects malformed state, sources, and sidecars."""

    artifact, paths = built
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
            "study_question": "wrong",
            "status": "wrong",
            "inference_substrate_class": "wrong",
            "fixture_ready_score": 0,
            "gate_check_summary": {},
            "verdict_class": "null",
            "honest_verdict": "wrong",
            "scope_answer": "wrong",
            "source_artifact_hashes": {},
            "structural_checks": [],
            "reproducibility_checksum": "wrong",
        }
    )
    failures = set(
        exp.validate_artifact(
            malformed,
            root=REPO,
            generation_view_path=paths["generation"],
            authority_sidecar_path=paths["authority"],
        )
    )
    assert {
        "field_principles_mismatch",
        "run_date_mismatch",
        "inference_substrate_mismatch",
        "execution_venue_mismatch",
        "duration_s_invalid",
        "random_seed_mismatch",
        "verifier_is_oracle_mismatch",
        "study_question_mismatch",
        "reproducibility_checksum_mismatch",
        "complete_status_mismatch",
        "inference_substrate_class_mismatch",
        "fixture_ready_score_mismatch",
        "gate_check_summary_mismatch",
        "verdict_class_mismatch",
        "honest_verdict_mismatch",
        "scope_answer_mismatch",
        "source_artifact_hashes_mismatch",
        "structural_checks_mismatch",
    } <= failures

    failure = exp._gate("forced", "upstream", "field", True, False, False)
    blocked = exp._blocked_artifact(
        exp._base_artifact(REPO, exp.RUN_DATE, paths["generation"], paths["authority"]),
        [failure],
        failure,
        {},
        0.0,
    )
    blocked.update(
        {
            "status": "wrong",
            "inference_substrate_class": "wrong",
            "gate_check_summary": {},
            "verdict_class": "wrong",
            "fixture_ready_score": 1,
            "honest_verdict": "wrong",
        }
    )
    blocked["reproducibility_checksum"] = exp.artifact_checksum(blocked)
    blocked_failures = set(exp.validate_artifact(blocked, root=REPO))
    assert {
        "blocked_status_mismatch",
        "blocked_inference_substrate_class_mismatch",
        "blocked_gate_check_summary_mismatch",
        "blocked_verdict_class_mismatch",
        "blocked_fixture_ready_score_mismatch",
        "blocked_honest_verdict_mismatch",
    } <= blocked_failures

    assert "source_artifact_missing" in exp.validate_artifact(artifact, root=tmp_path)
    with monkeypatch.context() as scoped:
        scoped.setattr(
            exp, "materialize_fixture", lambda _upstream: (_ for _ in ()).throw(ValueError())
        )
        assert any(
            error.startswith("independent_replay_failed")
            for error in exp.validate_artifact(artifact, root=REPO)
        )
    assert "sidecar_missing" in exp.validate_artifact(
        artifact,
        root=REPO,
        generation_view_path=tmp_path / "missing-generation.jsonl",
        authority_sidecar_path=paths["authority"],
    )

    bad_sidecar = tmp_path / "bad.jsonl"
    bad_sidecar.write_text("[]\n", encoding="utf-8")
    assert any(
        error.startswith("sidecar_unreadable")
        for error in exp.validate_artifact(
            artifact,
            root=REPO,
            generation_view_path=bad_sidecar,
            authority_sidecar_path=paths["authority"],
        )
    )
    changed_generation = tmp_path / "changed-generation.jsonl"
    changed_generation.write_bytes(
        paths["generation"].read_bytes().replace(b"Evidence", b"Changed!", 1)
    )
    changed_authority = tmp_path / "changed-authority.jsonl"
    changed_authority.write_bytes(
        paths["authority"].read_bytes().replace(b"supported", b"abstained", 1)
    )
    sidecar_failures = set(
        exp.validate_artifact(
            artifact,
            root=REPO,
            generation_view_path=changed_generation,
            authority_sidecar_path=changed_authority,
        )
    )
    assert {
        "generation_view_rows_mismatch",
        "authority_sidecar_rows_mismatch",
        "sidecar_hashes_mismatch",
    } <= sidecar_failures
    assert exp.validate_artifact(artifact, root=REPO) == []
