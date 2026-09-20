"""Tests for REQ-AUTO-7436 and SCENARIO-AUTO-7436-*.

These tests keep policy selection separate from certification. They also check
that reused public evidence cannot become a deployment certificate.
"""

from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7436_v652_selection_protocol as exp


def _rows(count: int = 24, *, role: str = "policy_tuning") -> list[dict[str, object]]:
    """Make independent source groups with both labels and ordered scores."""

    return [
        {
            "row_key": f"row-{index:03d}",
            "group_id": f"group-{index:03d}",
            "source_group": f"group-{index:03d}",
            "role": role,
            "partition": "probability_calibration"
            if role != "certification"
            else "policy_calibration",
            "label": int(index >= count // 2),
            "probability": (index + 1) / (count + 1),
        }
        for index in range(count)
    ]


def _passing_receipts() -> list[dict[str, object]]:
    """Build the exact affected and terminal receipt names for a fixture."""

    return [
        {"name": name, "required": True, "passed": True, "exit_code": 0}
        for name in (*exp.AFFECTED_CHECK_NAMES, *exp.TERMINAL_CHECK_NAMES)
    ]


def test_req_auto_7436_spec_precedes_implementation() -> None:
    """REQ-AUTO-7436: the implementation has a complete driving contract."""

    text = exp.SPEC_PATH.read_text(encoding="utf-8")
    assert "### REQ-AUTO-7436:" in text
    for number in range(1, 7):
        assert f"SCENARIO-AUTO-7436-{number:02d}" in text


def test_scenario_auto_7436_01_label_blind_roles_reject_mutations() -> None:
    """SCENARIO-AUTO-7436-01 rejects hidden labels, duplicates, and overlap."""

    predictors = [
        {"row_key": "b", "group_id": "g1", "partition": "probability_calibration"},
        {"row_key": "a", "group_id": "g1", "partition": "probability_calibration"},
        {"row_key": "c", "group_id": "g2", "partition": "probability_calibration"},
    ]
    representatives = exp.label_blind_representatives(predictors)
    assert [row["row_key"] for row in representatives] == ["a", "c"]
    halves = exp.split_probability_groups(predictors)
    assert set(halves) == {"probability_calibration", "policy_tuning"}
    assert not (
        {row["group_id"] for row in halves["probability_calibration"]}
        & {row["group_id"] for row in halves["policy_tuning"]}
    )

    with pytest.raises(ValueError, match="hidden label"):
        exp.label_blind_representatives([{**predictors[0], "label": 1}])
    with pytest.raises(ValueError, match="duplicate row"):
        exp.label_blind_representatives([predictors[0], predictors[0]])
    with pytest.raises(ValueError, match="role overlap"):
        exp.validate_role_groups({"policy_tuning": ["g1"], "certification": ["g1"]})


def test_scenario_auto_7436_02_seed_average_precedes_calibration() -> None:
    """SCENARIO-AUTO-7436-02 averages five heads before scalar calibration."""

    seed_scores = np.asarray([[0.1, 0.9]] * 5, dtype=np.float64)
    averaged = exp.average_seed_probabilities(seed_scores)
    assert averaged.tolist() == pytest.approx([0.1, 0.9])
    with pytest.raises(ValueError, match="five seed"):
        exp.average_seed_probabilities(seed_scores[:4])
    with pytest.raises(ValueError, match="bounded"):
        exp.average_seed_probabilities(np.asarray([[2.0]] * 5))

    sparse = {"calibration": [0.2, 0.8], "tuning": [0.3, 0.7]}
    dense = deepcopy(sparse)
    assert exp.dense_equivalence_errors(sparse, dense) == []
    dense["tuning"][1] = 0.71
    assert exp.dense_equivalence_errors(sparse, dense) == [
        "dense_sparse_probability_mismatch:tuning"
    ]


def test_scenario_auto_7436_03_quantile_policy_uses_frozen_utility() -> None:
    """SCENARIO-AUTO-7436-03 selects one pair with asymmetric harm costs."""

    rows = _rows()
    quantiles = exp.score_quantiles([float(row["probability"]) for row in rows])
    assert set(quantiles) == {f"q{index * 10:02d}" for index in range(11)}
    candidates = exp.candidate_thresholds([float(row["probability"]) for row in rows])
    assert exp.NO_ACCEPT_SENTINEL in candidates["accept"]
    assert exp.NO_REJECT_SENTINEL in candidates["reject"]

    selected, all_candidates = exp.select_tuned_policy(rows)
    assert len(all_candidates) > 1
    assert selected["candidate_index"] in {row["candidate_index"] for row in all_candidates}
    assert selected["selection_rule"] == "utility_then_coverage_then_registered_order"
    assert selected["utility_weights"] == {
        "correct_action": 1,
        "harmful_accept": -20,
        "harmful_reject": -10,
        "escalation": 0,
    }
    assert sum(row.get("selected") is True for row in all_candidates) == 1

    harmful_accept = exp.policy_utility([0], ["accept"])
    harmful_reject = exp.policy_utility([1], ["reject"])
    assert harmful_accept["total_utility"] == -20
    assert harmful_reject["total_utility"] == -10
    with pytest.raises(ValueError, match="same non-empty"):
        exp.policy_utility([], [])


def test_scenario_auto_7436_04_exact_checks_diagnose_empty_and_support() -> None:
    """SCENARIO-AUTO-7436-04 keeps empty selection distinct from low support."""

    delta = exp.ALPHA_PER_CHECK
    requirement = exp.zero_error_sample_requirement(0.05, delta)
    assert requirement == math.ceil(math.log(delta) / math.log(0.95))
    assert exp.zero_error_sample_requirement(0.10, delta) < requirement
    with pytest.raises(ValueError, match="risk budget"):
        exp.zero_error_sample_requirement(0.0, delta)

    disabled = exp.exact_action_check([], risk_budget=0.05, enabled=False)
    assert disabled["applicable"] is False
    assert disabled["upper_risk_bound"] is None
    assert disabled["alpha_spent"] == 0.0

    empty = exp.exact_action_check([], risk_budget=0.05, enabled=True)
    assert empty["diagnosis"] == "empty_selection"
    assert empty["upper_risk_bound"] is None
    assert "multiplicity" not in empty["diagnosis"]

    insufficient = exp.exact_action_check([0], risk_budget=0.05, enabled=True)
    assert insufficient["diagnosis"] == "insufficient_certification_support"
    assert insufficient["selected_groups"] == 1
    risky = exp.exact_action_check([1, 0, 0], risk_budget=0.05, enabled=True)
    assert risky["diagnosis"] == "excessive_observed_risk"
    certified = exp.exact_action_check([0] * requirement, risk_budget=0.05, enabled=True)
    assert certified["passed"] is True
    assert certified["diagnosis"] == "certified"
    with pytest.raises(ValueError, match="binary"):
        exp.exact_action_check([2], risk_budget=0.05, enabled=True)


def test_scenario_auto_7436_04_failed_conjunction_disables_policy() -> None:
    """SCENARIO-AUTO-7436-04 disables the whole policy without reselection."""

    rows = _rows(120, role="certification")
    tuned = {
        "accept_threshold": 0.75,
        "reject_threshold": 0.25,
        "accept_enabled": True,
        "reject_enabled": True,
        "candidate_index": 7,
    }
    certified = exp.certify_policy(rows, tuned)
    assert certified["frozen_candidate_index"] == 7
    assert certified["thresholds_reselected"] is False
    assert len(certified["checks"]) == 3
    assert certified["deployed_policy"] in {"frozen_typed_policy", "all_escalate"}
    if not certified["all_checks_passed"]:
        assert certified["deployed_policy"] == "all_escalate"
        assert certified["deployed_accept_enabled"] is False
        assert certified["deployed_reject_enabled"] is False

    disabled = exp.certify_policy(
        rows,
        {
            "accept_threshold": exp.NO_ACCEPT_SENTINEL,
            "reject_threshold": exp.NO_REJECT_SENTINEL,
            "accept_enabled": False,
            "reject_enabled": False,
            "candidate_index": 0,
        },
    )
    assert disabled["accept_check"]["applicable"] is False
    assert disabled["reject_check"]["applicable"] is False
    assert disabled["coverage_check"]["passed"] is False
    assert disabled["deployed_policy"] == "all_escalate"


def test_scenario_auto_7436_04_old_threshold_diagnosis_covers_each_action() -> None:
    """REQ-AUTO-7436 diagnoses old empty, unsupported, and risky selections."""

    rows = _rows(30, role="certification")
    diagnosis = exp.diagnose_old_thresholds("raw_l2_logistic", rows)
    assert len(diagnosis) == len(exp.OLD_ACCEPT_THRESHOLDS) + len(exp.OLD_REJECT_THRESHOLDS)
    assert {row["action"] for row in diagnosis} == {"accept", "reject"}
    assert all("score_quantiles" in row for row in diagnosis)
    assert all(row["selected_groups"] >= 0 for row in diagnosis)
    assert all(row["diagnosis"] in exp.SELECTION_DIAGNOSES for row in diagnosis)


def test_scenario_auto_7436_01_protocol_manifest_binds_disjoint_roles(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7436-01/06 seals roles before the measured trial."""

    roles = {
        "fit": ["fit-a", "fit-b"],
        "probability_calibration": ["cal-a"],
        "policy_tuning": ["tune-a"],
        "certification": ["cert-a", "cert-b"],
    }
    manifest = exp.build_protocol_manifest(roles)
    assert manifest["role_disjointness"]["passed"] is True
    assert manifest["bootstrap_draws"] == 10_000
    assert manifest["familywise_check_count"] == 9
    assert manifest["deployed_heads"] == list(exp.DEPLOYED_HEADS)
    path = tmp_path / "protocol.json"
    exp.seal_protocol(path, manifest)
    assert exp.validate_protocol_manifest(json.loads(path.read_text(encoding="utf-8"))) == []
    exp.seal_protocol(path, manifest)
    changed = deepcopy(manifest)
    changed["bootstrap_draws"] = 9_999
    with pytest.raises(FileExistsError, match="protocol seal conflict"):
        exp.seal_protocol(path, changed)

    overlap = deepcopy(manifest)
    overlap["roles"]["policy_tuning"]["group_ids"] = ["cert-a"]
    assert "role_overlap" in exp.validate_protocol_manifest(overlap)


def test_scenario_auto_7436_05_fixture_is_ready_but_never_deployment_valid(
    tmp_path: Path,
) -> None:
    """SCENARIO-AUTO-7436-05 keeps readiness independent from rollout."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    assert artifact["selection_protocol_ready_score"] == 1
    assert artifact["deployment_certificate_valid"] is False
    assert artifact["certificate_scope"] == "exploratory_reused_corpus"
    assert artifact["promotion_score"] == 0
    assert artifact["MODEL_SPECS"] == []
    assert artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == exp.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate_class"] == "aggregation"
    assert artifact["execution_venue"] == "host"
    assert "previously exposed" in artifact["evaluation_reuse_disclosure"]
    assert exp.validate_artifact(artifact, root=tmp_path) == []
    reduced = exp.independent_reduce(artifact, root=tmp_path)
    assert reduced["selection_protocol_ready_score"] == 1
    assert reduced["deployment_certificate_valid"] is False


@pytest.mark.parametrize(
    ("field", "replacement", "error"),
    [
        ("deployment_certificate_valid", True, "deployment_certificate_valid_mismatch"),
        ("certificate_scope", "fresh", "certificate_scope_mismatch"),
        ("promotion_score", 1, "promotion_score_mismatch"),
        (
            "inference_substrate_class",
            "no_model_load",
            "declaration_mismatch:inference_substrate_class",
        ),
        ("reproducibility_checksum", "sha256:bad", "reproducibility_checksum_mismatch"),
    ],
)
def test_scenario_auto_7436_06_artifact_mutations_fail_closed(
    tmp_path: Path, field: str, replacement: object, error: str
) -> None:
    """SCENARIO-AUTO-7436-06 rejects safety, scope, and checksum changes."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    artifact[field] = replacement
    assert error in exp.validate_artifact(artifact, root=tmp_path)


def test_req_auto_7436_blocked_and_cli_boundaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7436 preserves missing prerequisites and dispatches readers."""

    failed = exp.precondition_row("missing", "upstream", "path", "field", 1, None)
    blocked = exp.build_blocked_artifact(failed)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["honest_verdict"].startswith("blocked_")
    assert blocked["gate_check_summary"]["blocked_observed"] is None
    assert blocked["sample_size_budget"]["unstarted_units"] == 3

    candidate = tmp_path / "candidate.json"
    candidate.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(exp, "cold_replay", lambda path, root: [])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 0
    )
    monkeypatch.setattr(exp, "cold_replay", lambda path, root: ["bad"])
    assert (
        exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path), "--cold-replay", str(candidate)])
        == 1
    )
    with pytest.raises(SystemExit, match="--date"):
        exp.main(["--date", "20200101", "--root", str(tmp_path)])


def test_req_auto_7436_defensive_numeric_and_identity_boundaries(tmp_path: Path) -> None:
    """REQ-AUTO-7436 rejects malformed numeric, identity, and JSONL inputs."""

    assert exp._load_object(tmp_path / "missing.json") == {}
    malformed = tmp_path / "malformed.json"
    malformed.write_text("{", encoding="utf-8")
    assert exp._load_object(malformed) == {}
    scalar = tmp_path / "scalar.json"
    scalar.write_text("[]", encoding="utf-8")
    assert exp._load_object(scalar) == {}

    with pytest.raises(ValueError, match="row and group"):
        exp.label_blind_representatives([{"row_key": "", "group_id": "g"}])
    assert exp.dense_equivalence_errors({"a": [0.1]}, {"a": []}) == ["dense_sparse_shape_invalid:a"]
    for scores, match in (([], "non-empty"), ([math.nan], "finite"), ([1.1], "bounded")):
        with pytest.raises(ValueError, match=match):
            exp.score_quantiles(scores)
    with pytest.raises(ValueError, match="lower"):
        exp._actions(
            [0.5],
            accept_threshold=0.5,
            reject_threshold=0.5,
            accept_enabled=True,
            reject_enabled=True,
        )
    with pytest.raises(ValueError, match="bounded"):
        exp._actions(
            [math.inf],
            accept_threshold=0.8,
            reject_threshold=0.2,
            accept_enabled=True,
            reject_enabled=True,
        )
    with pytest.raises(ValueError, match="binary"):
        exp.policy_utility([2], ["accept"])
    with pytest.raises(ValueError, match="typed"):
        exp.policy_utility([1], ["other"])
    with pytest.raises(ValueError, match="required"):
        exp.select_tuned_policy([])
    with pytest.raises(ValueError, match="delta"):
        exp.zero_error_sample_requirement(0.1, 1.0)
    with pytest.raises(ValueError, match="ordered"):
        exp._clopper_pearson_lower(2, 1, exp.ALPHA_PER_CHECK)
    with pytest.raises(ValueError, match="typed"):
        exp.exact_coverage_check([])

    bad_jsonl = tmp_path / "bad.jsonl"
    bad_jsonl.write_text("{\n", encoding="utf-8")
    with pytest.raises(ValueError, match="jsonl_invalid"):
        exp._jsonl_rows(bad_jsonl)
    non_object = tmp_path / "non-object.jsonl"
    non_object.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not_object"):
        exp._jsonl_rows(non_object)


def test_scenario_auto_7436_06_protocol_and_shard_mutations(tmp_path: Path) -> None:
    """SCENARIO-AUTO-7436-06 names each changed protocol and row boundary."""

    roles = {
        "fit": ["fit"],
        "probability_calibration": ["cal"],
        "policy_tuning": ["tune"],
        "certification": ["cert"],
    }
    manifest = exp.build_protocol_manifest(roles)
    mutations: list[tuple[dict[str, object], str]] = []
    wrong_constant = deepcopy(manifest)
    wrong_constant["bootstrap_draws"] = 1
    mutations.append((wrong_constant, "protocol_field_mismatch:bootstrap_draws"))
    missing_roles = deepcopy(manifest)
    missing_roles["roles"] = None
    mutations.append((missing_roles, "roles_invalid"))
    malformed_role = deepcopy(manifest)
    malformed_role["roles"]["fit"] = None
    mutations.append((malformed_role, "role_invalid:fit"))
    wrong_count = deepcopy(manifest)
    wrong_count["roles"]["fit"]["group_count"] = 2
    mutations.append((wrong_count, "role_count_mismatch:fit"))
    wrong_hash = deepcopy(manifest)
    wrong_hash["roles"]["fit"]["group_identity_sha256"] = "sha256:bad"
    mutations.append((wrong_hash, "role_hash_mismatch:fit"))
    wrong_names = deepcopy(manifest)
    del wrong_names["roles"]["fit"]
    mutations.append((wrong_names, "role_names_mismatch"))
    for mutated, expected in mutations:
        assert expected in exp.validate_protocol_manifest(mutated)

    invalid_path = tmp_path / "invalid-protocol.json"
    with pytest.raises(ValueError, match="manifest invalid"):
        exp.seal_protocol(invalid_path, wrong_constant)

    rows_path = tmp_path / "rows.jsonl"
    row_manifest = exp._write_rows(rows_path, [{"row": 1}])
    assert exp._load_bound_rows(tmp_path, row_manifest) == [{"row": 1}]
    wrong_bytes = deepcopy(row_manifest)
    wrong_bytes["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="bytes_mismatch"):
        exp._load_bound_rows(tmp_path, wrong_bytes)
    wrong_count_manifest = deepcopy(row_manifest)
    wrong_count_manifest["rows"] = 2
    with pytest.raises(ValueError, match="count_mismatch"):
        exp._load_bound_rows(tmp_path, wrong_count_manifest)


def test_scenario_auto_7436_01_join_and_narrow_loader_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-AUTO-7436-01 authenticates only the permitted role shards."""

    predictor = {"row_key": "r", "group_id": "g", "partition": "probability_calibration"}
    evaluator = {
        "row_key": "r",
        "group_id": "g",
        "partition": "probability_calibration",
        "primary_label": 1,
    }
    assert exp._attach_labels([predictor], [evaluator])[0]["label"] == 1
    with pytest.raises(ValueError, match="unique"):
        exp._attach_labels([predictor], [evaluator, evaluator])
    with pytest.raises(ValueError, match="missing"):
        exp._attach_labels([{**predictor, "row_key": "x"}], [evaluator])
    with pytest.raises(ValueError, match="label invalid"):
        exp._attach_labels([predictor], [{**evaluator, "primary_label": None}])
    with pytest.raises(ValueError, match="identity mismatch"):
        exp._attach_labels([predictor], [{**evaluator, "group_id": "other"}])

    corpus = tmp_path / exp.CORPUS_DIR
    corpus.mkdir(parents=True)
    manifest_path = tmp_path / exp.CORPUS_MANIFEST_PATH
    manifest_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    with pytest.raises(ValueError, match="manifest_identity"):
        exp.load_allowed_protocol_rows(tmp_path)

    shards: list[dict[str, object]] = []
    for partition in exp.ALLOWED_PARTITIONS:
        for kind in ("predictor", "evaluator"):
            path = corpus / f"{kind}-{partition}-000.jsonl"
            path.write_text(json.dumps({"partition": partition}) + "\n", encoding="utf-8")
            shards.append(
                {
                    "kind": kind,
                    "path": path.name,
                    "rows": 1,
                    "sha256": exp.sha256_file(path),
                }
            )
    manifest_path.write_text(json.dumps({"shards": ["bad", *shards]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    with pytest.raises(ValueError, match="manifest_invalid"):
        exp.load_allowed_protocol_rows(tmp_path)

    manifest_path.write_text(json.dumps({"shards": shards}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    predictors, evaluators, receipts = exp.load_allowed_protocol_rows(tmp_path)
    assert set(predictors) == set(exp.ALLOWED_PARTITIONS)
    assert set(evaluators) == set(exp.ALLOWED_PARTITIONS)
    assert len(receipts) == 6

    bad_hash = deepcopy(shards)
    bad_hash[0]["sha256"] = "sha256:bad"
    manifest_path.write_text(json.dumps({"shards": bad_hash}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    with pytest.raises(ValueError, match="hash_mismatch"):
        exp.load_allowed_protocol_rows(tmp_path)

    bad_count = deepcopy(shards)
    bad_count[0]["rows"] = 2
    manifest_path.write_text(json.dumps({"shards": bad_count}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    with pytest.raises(ValueError, match="count_mismatch"):
        exp.load_allowed_protocol_rows(tmp_path)

    skipped = [*shards, {"kind": "other", "path": "predictor-fit-ignored.jsonl"}]
    manifest_path.write_text(json.dumps({"shards": skipped}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    assert len(exp.load_allowed_protocol_rows(tmp_path)[2]) == 6

    predictor_only_fit = [
        row for row in shards if not (row["kind"] == "evaluator" and "-fit-" in str(row["path"]))
    ]
    manifest_path.write_text(json.dumps({"shards": predictor_only_fit}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    assert len(exp.load_allowed_protocol_rows(tmp_path)[2]) == 5

    manifest_path.write_text(json.dumps({"shards": shards[:2]}) + "\n", encoding="utf-8")
    monkeypatch.setattr(exp, "EXPECTED_CORPUS_HASH", exp.sha256_file(manifest_path))
    with pytest.raises(ValueError, match="incomplete"):
        exp.load_allowed_protocol_rows(tmp_path)


def test_req_auto_7436_real_inputs_authenticate_and_reduce() -> None:
    """REQ-AUTO-7436 runs the affected aggregation on authenticated local evidence."""

    root = exp.REPO_ROOT
    checks, hashes, upstreams = exp.collect_preconditions(root)
    assert checks and all(row["passed"] is True for row in checks)
    assert (
        hashes[exp.UPSTREAM_PATHS["static_decisions"].as_posix()]["original_flagged_adversarial"]
        is False
    )
    predictors, evaluators, receipts = exp.load_allowed_protocol_rows(root)
    assert {row["partition"] for row in receipts} == set(exp.ALLOWED_PARTITIONS)
    states, checkpoint_receipts = exp._load_frozen_states(root, upstreams["static_decisions"])
    assert all(len(states[head]) == 5 for head in exp.ALL_HEADS)
    assert len(checkpoint_receipts) == 20
    scored = exp._score_protocol(
        root,
        predictors,
        evaluators,
        states,
        run_started=exp.time.monotonic(),
    )
    policies, raw_rows, diagnosis, equivalence, bootstraps = scored
    assert len(policies) == len(exp.DEPLOYED_HEADS)
    assert raw_rows and diagnosis and len(bootstraps) == len(exp.DEPLOYED_HEADS)
    assert equivalence == []


def test_req_auto_7436_checkpoint_and_replay_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-AUTO-7436 rejects changed checkpoint identity and reader inputs."""

    checkpoint = tmp_path / "checkpoint.json"
    checkpoint.write_text(
        json.dumps({"seed": exp.TRAINING_SEEDS[0], "arm": exp.DEPLOYED_HEADS[0], "checkpoint": {}}),
        encoding="utf-8",
    )
    row = {
        "condition": "full_source",
        "arm": exp.DEPLOYED_HEADS[0],
        "seed": exp.TRAINING_SEEDS[0],
        "path": checkpoint.name,
        "sha256": exp.sha256_file(checkpoint),
    }
    with pytest.raises(ValueError, match="five_frozen"):
        exp._load_frozen_states(tmp_path, {"checkpoint_manifest": [row]})
    bad_hash = deepcopy(row)
    bad_hash["sha256"] = "sha256:bad"
    with pytest.raises(ValueError, match="checkpoint_hash"):
        exp._load_frozen_states(tmp_path, {"checkpoint_manifest": [bad_hash]})
    checkpoint.write_text(
        json.dumps({"seed": 0, "arm": exp.DEPLOYED_HEADS[0], "checkpoint": {}}),
        encoding="utf-8",
    )
    bad_identity = {**row, "sha256": exp.sha256_file(checkpoint)}
    with pytest.raises(ValueError, match="checkpoint_identity"):
        exp._load_frozen_states(tmp_path, {"checkpoint_manifest": [bad_identity]})

    assert exp.cold_replay(tmp_path / "missing.json", root=tmp_path) == [
        "artifact_unreadable_or_not_object"
    ]
    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(artifact), encoding="utf-8")
    assert exp.cold_replay(candidate, root=tmp_path) == []

    span = exp._span("test", exp.time.monotonic(), exp.time.monotonic(), 1)
    assert span["completed_units"] == 1
    external = exp._protocol_reference(tmp_path / "other", tmp_path / "fixture-protocol.json")
    assert Path(external["path"]).is_absolute()

    monkeypatch.setattr(exp, "run_experiment", lambda root, date, output_path: {})
    assert exp.main(["--date", exp.RUN_DATE, "--root", str(tmp_path)]) == 0
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(candidate),
            ]
        )
        == 0
    )
    malformed = tmp_path / "malformed-candidate.json"
    malformed.write_text("{", encoding="utf-8")
    assert (
        exp.main(
            [
                "--date",
                exp.RUN_DATE,
                "--root",
                str(tmp_path),
                "--independent-reduce",
                str(malformed),
            ]
        )
        == 1
    )


def test_req_auto_7436_validator_defensive_branches(tmp_path: Path) -> None:
    """REQ-AUTO-7436 rejects malformed verdict, principles, and broken evidence."""

    artifact = exp.build_fixture_artifact(tmp_path, validation_receipts=_passing_receipts())
    mutations = []
    wrong_verdict = deepcopy(artifact)
    wrong_verdict["verdict_class"] = "other"
    mutations.append((wrong_verdict, "verdict_class_invalid"))
    wrong_principles = deepcopy(artifact)
    wrong_principles["field_principles"] = {}
    mutations.append((wrong_principles, "field_principles_mismatch"))
    blocked = deepcopy(artifact)
    blocked["verdict_class"] = "blocked"
    blocked["honest_verdict"] = "complete_wrong"
    mutations.append((blocked, "blocked_verdict_prefix_invalid"))
    broken_rows = deepcopy(artifact)
    broken_rows["selection_row_shard"]["sha256"] = "sha256:bad"
    broken_rows["reproducibility_checksum"] = exp.artifact_checksum(broken_rows)
    mutations.append((broken_rows, "independent_reduction_failed"))
    invalid_source_row = deepcopy(artifact)
    invalid_source_row["fixture_artifact"] = False
    invalid_source_row["source_artifact_hashes"] = {"bad": []}
    invalid_source_row["reproducibility_checksum"] = exp.artifact_checksum(invalid_source_row)
    mutations.append((invalid_source_row, "source_artifact_hash_row_invalid"))
    changed_source = deepcopy(artifact)
    changed_source["fixture_artifact"] = False
    changed_source["source_artifact_hashes"] = {
        "missing": {"path": "missing-source", "sha256": "sha256:bad"}
    }
    changed_source["reproducibility_checksum"] = exp.artifact_checksum(changed_source)
    mutations.append((changed_source, "source_artifact_hash_mismatch:missing-source"))
    for mutated, expected in mutations:
        assert any(expected in error for error in exp.validate_artifact(mutated, root=tmp_path))
