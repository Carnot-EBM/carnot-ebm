"""REQ-REPORT-7701 and REQ-VERIFY-7701 sealed cohort tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import fresh_relation_cohort as base
from carnot.reporting import sealed_source_cohort as sealed
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot import experiment_7701_v671_sealed_cohort as runner


COUNTS = {"fit": 2, "evaluation": 1}


def public(split: str, index: int) -> dict:
    """Build one complete public family with unique source and answer."""
    return {
        "official_split": split,
        "context": f"{index} def unique_{index}(): return {index}",
        "question": f"Where is unique_{index} defined?",
        "answer": f"unique_{index} is defined at line {index}",
        "metadata": {"instance_id": f"case-{index}"},
    }


def fixture_selection() -> list[dict]:
    """Keep distinct official splits without reading labels."""
    rows = [public("train", i) for i in (1, 2, 3)] + [public("test", 4)]
    planned = sealed.plan(rows, {"family_ids": set()}, set(), COUNTS, "fixture-salt")
    return planned["selected"]


def test_scenario_report_7701_custody_private_fixture_tree(tmp_path: Path):
    # SCENARIO-REPORT-7701-CUSTODY: disqualified old selections are still exposed.
    rows = [public("train", i) for i in (1, 2, 3, 5)] + [public("test", 4)]
    old_id = base.cluster_public_rows([rows[0]])[0][0]["family_id"]
    planned = sealed.plan(rows, {"family_ids": set()}, {old_id}, COUNTS, "fixture-salt")
    assert planned["selected_prior_exposure_count"] == 0
    assert planned["excluded_prior_exposure_count"] == 1
    assert len(planned["selected"]) == 3
    assert all(item["family_id"] != old_id for item in planned["selected"])
    assert {x["official_split"] for x in planned["selected"] if x["role"] == "evaluation"} == {
        "test"
    }
    (tmp_path / "fixture.json").write_text(json.dumps(planned["inventory"]))
    assert json.loads((tmp_path / "fixture.json").read_text())["public_rows"] == 5


def test_scenario_report_7701_custody_missing_branches():
    # SCENARIO-REPORT-7701-CUSTODY: failures cannot quietly resize a role.
    a = public("train", 1)
    b = deepcopy(a)
    b["official_split"] = "test"
    families, excluded = base.cluster_public_rows([a, b])
    assert families == [] and excluded[0]["reason"] == "cross_official_split_family"
    absent = public("train", 2)
    absent["question"] = ""
    assert base.cluster_public_rows([absent])[1][0]["reason"] == "complete_question_missing"
    with pytest.raises(ValueError, match="duplicate_family"):
        base.assign_roles([{"family_id": "x"}, {"family_id": "x"}], {}, "fixture")
    with pytest.raises(ValueError, match="underfilled"):
        sealed.plan([public("train", 1)], {"family_ids": set()}, set(), COUNTS, "fixture")
    bad = public("train", 3)
    bad["answer"] = None
    with pytest.raises(ValueError, match="public_row_incomplete"):
        base.cluster_public_rows([bad])


def test_scenario_verify_7701_isolation_seals_before_canary_labels(tmp_path: Path):
    # SCENARIO-VERIFY-7701-ISOLATION: only projected fields enter the model store.
    selected = fixture_selection()
    events: list[str] = []

    def labels(item: dict) -> int:
        assert (tmp_path / "public_protocol.json").is_file()
        events.append(item["family_id"])
        return 1

    protocol = sealed.seal(tmp_path, selected, COUNTS, "fixture-salt", labels)
    assert len(events) == 3
    assert sealed.cold_reduce(tmp_path / "protocol.json", COUNTS)["families"] == 3
    for role, info in protocol["roles"].items():
        model_path = tmp_path / info["model_inputs"]
        model = json.loads(model_path.read_text().splitlines()[0])
        assert set(model) == base.PREDICTOR_FIELDS
        assert "label" not in model
        evaluator = tmp_path / protocol["evaluator_stores"][role]["path"]
        assert evaluator.stat().st_mode & 0o777 == 0o600
        assert '"label":1' in evaluator.read_text()


def test_scenario_verify_7701_isolation_child_denies_evaluator_open(tmp_path: Path):
    # SCENARIO-VERIFY-7701-ISOLATION: the child has only a public path.
    protocol = sealed.seal(tmp_path, fixture_selection(), COUNTS, "fixture-salt", lambda _: 1)
    model_path = tmp_path / protocol["roles"]["fit"]["model_inputs"]
    evaluator = tmp_path / protocol["evaluator_stores"]["fit"]["path"]
    child = """\
import json, sys
from pathlib import Path
from carnot.reporting.sealed_source_cohort import predictor_inputs
model, forbidden = map(Path, sys.argv[1:])
def deny(event, args):
    if event == 'open' and Path(args[0]).resolve() == forbidden.resolve():
        raise PermissionError('evaluator_open_denied')
sys.addaudithook(deny)
rows = predictor_inputs(model, 'fit')
assert all('label' not in row for row in rows)
try:
    forbidden.open()
except PermissionError:
    print(json.dumps({'rows': len(rows), 'evaluator_denied': True}))
else:
    raise AssertionError('evaluator_open_allowed')
"""
    result = subprocess.run(
        [sys.executable, "-c", child, str(model_path), str(evaluator)],
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["evaluator_denied"] is True
    assert len(sealed.predictor_inputs(model_path, "fit")) == 2


def test_scenario_report_7701_custody_closes_v669_missing_branches():
    # SCENARIO-REPORT-7701-CUSTODY: all diagnosed public-boundary branches fire.
    text_meta = public("train", 10)
    text_meta["metadata"] = '{"instance_id":"ten"}'
    assert base.cluster_public_rows([text_meta])[0][0]["instance_ids"] == ["ten"]
    text_meta["metadata"] = "{}"
    assert base.cluster_public_rows([text_meta])[0][0]["instance_ids"] == []
    text_meta["metadata"] = None
    assert base.cluster_public_rows([text_meta])[0][0]["instance_ids"] == []
    bad = public("train", 10)
    bad["context"] = ""
    with pytest.raises(ValueError, match="public_row_incomplete"):
        base.cluster_public_rows([bad])
    with pytest.raises(ValueError, match="duplicate_family"):
        base.assign_roles([{"family_id": "a"}, {"family_id": "a"}], {}, "fixture")
    with pytest.raises(ValueError, match="role_split"):
        base.predictor_view(public("train", 10), "a", "evaluation")
    with pytest.raises(ValueError, match="role_invalid"):
        base.predictor_view(public("train", 10), "a", "invalid")
    model = base.predictor_view(public("train", 10), "a", "fit")
    for key, value, message in (
        ("labels_accessible", True, "label_access"),
        ("answer_sha256", "bad", "answer_hash"),
    ):
        altered = {**model, key: value}
        with pytest.raises(ValueError, match=message):
            base.validate_predictor(altered, "fit")
    with pytest.raises(ValueError, match="duplicate_family"):
        base.feature_rows([model, model], "fit")


def test_scenario_report_7701_terminal_extra_evaluator_key_fails(tmp_path: Path):
    # SCENARIO-REPORT-7701-TERMINAL: a rehashed extra label key is still invalid.
    protocol = sealed.seal(tmp_path, fixture_selection(), COUNTS, "fixture-salt", lambda _: 1)
    path = tmp_path / protocol["evaluator_stores"]["fit"]["path"]
    labels = [json.loads(line) for line in path.read_text().splitlines()]
    labels[0]["extra_label"] = 1
    path.write_text("".join(json.dumps(label) + "\n" for label in labels))
    protocol["evaluator_stores"]["fit"]["sha256"] = sha256_file(path)
    atomic_json(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match="label_schema"):
        sealed.cold_reduce(tmp_path / "protocol.json", COUNTS)


@pytest.mark.parametrize("mutation", ["source", "role", "label", "bytes"])
def test_scenario_report_7701_terminal_cold_rejects_mutations(tmp_path: Path, mutation: str):
    # SCENARIO-REPORT-7701-TERMINAL: hash and schema checks survive cold replay.
    protocol = sealed.seal(tmp_path, fixture_selection(), COUNTS, "fixture-salt", lambda _: 0)
    path = tmp_path / protocol["roles"]["fit"]["model_inputs"]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if mutation == "source":
        rows[0]["complete_source"] += " altered"
    elif mutation == "role":
        rows[0]["role"] = "evaluation"
    elif mutation == "label":
        rows[0]["canary_label"] = 1
    else:
        path.write_bytes(path.read_bytes() + b"corrupt")
    if mutation != "bytes":
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError, match="hash|label|role|source|json"):
        sealed.cold_reduce(tmp_path / "protocol.json", COUNTS)


def fixture_runner(monkeypatch: pytest.MonkeyPatch, root: Path, *, blocked: bool = False):
    """Place owned fake input files under one private worktree."""
    module = root / "python/carnot/reporting/sealed_source_cohort.py"
    module.parent.mkdir(parents=True)
    module.write_text("fixture reducer")
    monkeypatch.setattr(runner, "ROOT", root)
    failure = runner.check("fixture_input", "fixture", root / "missing", "exists", True, False)
    monkeypatch.setattr(
        runner,
        "preconditions",
        lambda _root, _start: (
            [failure] if blocked else [],
            {"producers": {}, "pre_gate_receipts": {}, "missing_evidence": []},
        ),
    )
    monkeypatch.setattr(
        runner, "prior_selected", lambda _root: ({f"prior-{i}" for i in range(480)}, [], {})
    )
    monkeypatch.setattr(
        runner,
        "_load_public_rows",
        lambda _start: (
            [public("train", i) for i in range(1, 362)]
            + [public("test", i) for i in range(362, 403)]
        ),
    )
    monkeypatch.setattr(
        runner,
        "exposure_ledger",
        lambda _root, _start: (
            {"files": [], "uncertainties": []},
            {"family_ids": set(), "source_hashes": set(), "answer_hashes": set()},
            [],
        ),
    )
    monkeypatch.setattr(runner, "label_reader", lambda _start: lambda _item: 0)

    def scoped(*_args, **kwargs):
        assert kwargs["basetemp"].is_dir()
        assert kwargs["coverage_file"].parent.is_dir()
        return {"validation_receipts": [], "required_checks_passed": True, "repository_health": {}}

    monkeypatch.setattr(
        runner.validation,
        "run_scoped_validation",
        scoped,
    )
    monkeypatch.setattr(
        runner.validation,
        "run_commands",
        lambda *_args, **_kwargs: [
            {"name": name, "passed": True, "exit_code": 0, "log_sha256": "fixture"}
            for name in ("cold_reduction", "adversarial_verify", "verdict_row_consistency_strict")
        ],
    )


def test_scenario_report_7701_seal_private_400_family_e2e(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-SEAL: exactly 400 disjoint, label-isolated roles.
    fixture_runner(monkeypatch, tmp_path)
    artifact = runner.run_experiment(tmp_path, "20260926", Path("result.json"))
    assert artifact["honest_verdict"] == "complete_null_cohort_readiness"
    assert artifact["cohort_ready_score"] == 1
    assert artifact["fresh_source_score"] == 1
    assert artifact["selected_prior_exposure_count"] == 0
    assert len(artifact["rows"]) == 400
    assert runner.validate_candidate(tmp_path / "result.json")["families"] == 400
    assert (
        sealed.cold_reduce(tmp_path / runner.RAW / "protocol.json", runner.ROLE_COUNTS)[
            "role_counts"
        ]
        == runner.ROLE_COUNTS
    )


def test_scenario_report_7701_custody_private_blocked_result(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-CUSTODY: absent external bytes stay blocked, not partial.
    fixture_runner(monkeypatch, tmp_path, blocked=True)
    artifact = runner.run_experiment(tmp_path, "20260926", Path("result.json"))
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "complete_blocked_precondition"
    assert artifact["cohort_ready_score"] == 0
    assert artifact["gate_check_summary"][0]["field"] == "exists"
    assert runner.validate_candidate(tmp_path / "result.json")["families"] == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("public_hash", "public_protocol_hash"),
        ("role_protocol", "role_protocol"),
        ("evaluator_hash", "evaluator_hash"),
        ("count", "role_count"),
        ("roster", "role_hash"),
        ("role_hash", "role_hash"),
        ("source_hash", "source_hash"),
        ("cross_role", "cross_role"),
        ("label_join", "label_join"),
        ("split", "role_split"),
        ("prior", "selected_prior"),
        ("json", "jsonl_corrupt"),
    ],
)
def test_scenario_report_7701_terminal_rehashed_corruptions(
    tmp_path: Path, mutation: str, message: str
):
    # SCENARIO-REPORT-7701-TERMINAL: cold reader checks semantics after rehash.
    protocol = sealed.seal(tmp_path, fixture_selection(), COUNTS, "fixture-salt", lambda _: 0)
    public_protocol = json.loads((tmp_path / "public_protocol.json").read_text())
    role = "evaluation" if mutation == "cross_role" else "fit"
    info = public_protocol["roles"][role]
    model_path = tmp_path / info["model_inputs"]
    label_path = tmp_path / protocol["evaluator_stores"][role]["path"]
    models = [json.loads(line) for line in model_path.read_text().splitlines()]
    labels = [json.loads(line) for line in label_path.read_text().splitlines()]
    if mutation == "public_hash":
        (tmp_path / "public_protocol.json").write_text("{}")
    elif mutation == "role_protocol":
        protocol["roles"]["fit"]["group_count"] = 999
    elif mutation == "evaluator_hash":
        label_path.write_text(label_path.read_text() + " ")
    elif mutation == "count":
        models.pop()
    elif mutation == "roster":
        info["families"] = list(reversed(info["families"]))
    elif mutation == "role_hash":
        info["role_hash"] = "wrong"
    elif mutation == "source_hash":
        info["source_hashes"][0] = "wrong"
    elif mutation == "cross_role":
        models[0]["component_hash"] = public_protocol["roles"]["fit"]["families"][0]
        info["families"] = [models[0]["component_hash"]]
        info["role_hash"] = base.stable_hash(info["families"])
    elif mutation == "label_join":
        labels[0]["family_id"] = "wrong"
    elif mutation == "split":
        models[0]["official_split"] = "test"
    elif mutation == "prior":
        models[0]["historically_exposed"] = True
    else:
        model_path.write_text("{bad json}\n")
    if mutation in {"count", "cross_role", "split", "prior"}:
        model_path.write_text("".join(json.dumps(row) + "\n" for row in models))
    if mutation == "label_join":
        label_path.write_text("".join(json.dumps(row) + "\n" for row in labels))
        protocol["evaluator_stores"][role]["sha256"] = sha256_file(label_path)
    if mutation in {"count", "cross_role", "split", "prior", "json"}:
        info["model_inputs_sha256"] = sha256_file(model_path)
    if mutation not in {"public_hash", "role_protocol", "evaluator_hash"}:
        atomic_json(tmp_path / "public_protocol.json", public_protocol)
        protocol["roles"] = public_protocol["roles"]
        protocol["public_protocol_sha256"] = sha256_file(tmp_path / "public_protocol.json")
    atomic_json(tmp_path / "protocol.json", protocol)
    with pytest.raises(ValueError, match=message):
        sealed.cold_reduce(tmp_path / "protocol.json", COUNTS)


def test_scenario_report_7701_seal_rejects_bad_roster_or_label(tmp_path: Path):
    # SCENARIO-REPORT-7701-SEAL: a role or evaluator error cannot be sealed.
    selected = fixture_selection()
    with pytest.raises(ValueError, match="underfilled"):
        sealed.seal(tmp_path, selected[:1], COUNTS, "fixture", lambda _: 0)
    duplicate = deepcopy(selected)
    duplicate[2]["family_id"] = duplicate[0]["family_id"]
    with pytest.raises(ValueError, match="cross_role_family"):
        sealed.seal(tmp_path, duplicate, COUNTS, "fixture", lambda _: 0)
    with pytest.raises(ValueError, match="label_invalid"):
        sealed.seal(tmp_path, selected, COUNTS, "fixture", lambda _: 7)


def test_scenario_report_7701_custody_prior_selected_authentication(tmp_path: Path):
    # SCENARIO-REPORT-7701-CUSTODY: roster and result bytes must both exist.
    ids, checks, receipts = runner.prior_selected(tmp_path)
    assert ids == set() and len(checks) == 2 and receipts == {}
    protocol_path = (
        tmp_path / "results/raw/experiment_7673_v669_fresh_relation_cohort/protocol.json"
    )
    result_path = tmp_path / "results/experiment_7673_v669_fresh_relation_cohort.json"
    protocol_path.parent.mkdir(parents=True)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    selected = [f"old-{index}" for index in range(480)]
    atomic_json(protocol_path, {"roles": {"fit": {"families": selected}}})
    atomic_json(result_path, {"exposure_ledger": {"selected_family_ids": selected}})
    ids, checks, receipts = runner.prior_selected(tmp_path)
    assert len(ids) == 480 and all(check["passed"] for check in checks)
    assert len(receipts) == 2


def test_scenario_report_7701_terminal_candidate_rejects_claim_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-TERMINAL: serialized claims cannot outrun stores.
    fixture_runner(monkeypatch, tmp_path)
    runner.run_experiment(tmp_path, "20260926", Path("result.json"))
    path = tmp_path / "result.json"
    original = json.loads(path.read_text())
    cases = [
        ("honest_verdict", "running", "nonterminal_verdict"),
        ("MODEL_SPECS", [{"model_id": "invented"}], "false_model_provenance"),
        ("role_manifest_sha256", "wrong", "protocol_hash"),
        ("rows", [], "candidate_row"),
        (
            "rows",
            [{**original["rows"][0], "role": "wrong"}] + original["rows"][1:],
            "candidate_role",
        ),
        ("selected_prior_exposure_count", 1, "selected_prior_exposure"),
    ]
    for field, value, message in cases:
        altered = {**original, field: value}
        atomic_json(path, altered)
        with pytest.raises(ValueError, match=message):
            runner.validate_candidate(path)
    changed = deepcopy(original)
    changed["source_artifact_hashes"]["producers"][str(path.resolve())] = "self"
    atomic_json(path, changed)
    with pytest.raises(ValueError, match="output_self_input"):
        runner.validate_candidate(path)


def test_scenario_report_7701_custody_rejects_wrong_date(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-CUSTODY: no accidental cross-milestone run.
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="repo_or_date"):
        runner.run_experiment(tmp_path, "20260925", Path("result.json"))


def test_scenario_report_7701_custody_rejects_broken_exposure_subtraction(
    monkeypatch: pytest.MonkeyPatch,
):
    # SCENARIO-REPORT-7701-CUSTODY: selected prior families fail even if subtraction regresses.
    rows = [public("train", i) for i in (1, 2)] + [public("test", 3)]
    exposed_id = base.cluster_public_rows([rows[0]])[0][0]["family_id"]
    monkeypatch.setattr(base, "subtract_exposure", lambda families, _ledger: (families, []))
    with pytest.raises(ValueError, match="selected_prior_exposure"):
        sealed.plan(rows, {"family_ids": set()}, {exposed_id}, COUNTS, "fixture-salt")


def test_scenario_report_7701_seal_late_label_reader_caches_shard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-SEAL: the evaluator opens one label column late.
    import pyarrow.parquet as parquet

    calls: list[Path] = []

    class Column:
        def to_pylist(self):
            return [[{"start": 0, "end": 1}]]

    class Table:
        def column(self, name: str):
            assert name == "labels"
            return Column()

    def read(path: Path, *, columns: list[str]):
        assert columns == ["labels"]
        calls.append(path)
        return Table()

    monkeypatch.setattr(parquet, "read_table", read)
    monkeypatch.setattr(runner, "DATA_ROOT", tmp_path)
    monkeypatch.setattr(
        runner, "annotation_binary_label", lambda answer, spans: int(bool(answer and spans))
    )
    callback = runner.label_reader(0.0)
    item = {"view": {"_shard": "fixture.parquet", "_row_index": 0, "answer": "answer"}}
    assert callback(item) == 1
    assert callback(item) == 1
    assert calls == [tmp_path / "data/fixture.parquet"]


def test_scenario_report_7701_custody_underfilled_run_stays_blocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    # SCENARIO-REPORT-7701-CUSTODY: a short public roster is a blocked gate.
    fixture_runner(monkeypatch, tmp_path)
    monkeypatch.setattr(runner, "_load_public_rows", lambda _start: [public("train", 1)])
    artifact = runner.run_experiment(tmp_path, "20260926", Path("result.json"))
    assert artifact["verdict_class"] == "blocked"
    assert artifact["gate_check_summary"][0]["check"] == "fixed_role_counts"
    assert artifact["gate_check_summary"][0]["field"] == "role_allocation"


@pytest.mark.parametrize("failed_check", ["validation", "terminal"])
def test_scenario_report_7701_terminal_failed_checks_zero_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_check: str
):
    # SCENARIO-REPORT-7701-TERMINAL: a failed check cannot retain readiness.
    fixture_runner(monkeypatch, tmp_path)
    if failed_check == "validation":
        monkeypatch.setattr(
            runner.validation,
            "run_scoped_validation",
            lambda *_args, **_kwargs: {
                "validation_receipts": [],
                "required_checks_passed": False,
                "repository_health": {},
            },
        )
    else:
        monkeypatch.setattr(
            runner.validation,
            "run_commands",
            lambda *_args, **_kwargs: [
                {"name": name, "passed": index != 1, "exit_code": int(index == 1)}
                for index, name in enumerate(
                    ("cold_reduction", "adversarial_verify", "verdict_row_consistency_strict")
                )
            ],
        )
    artifact = runner.run_experiment(tmp_path, "20260926", Path("result.json"))
    assert artifact["verdict_class"] == "disqualified"
    assert artifact["cohort_ready_score"] == 0
    assert artifact["fresh_source_score"] == 0
    assert artifact["flagged_adversarial"] is (failed_check == "terminal")


def test_scenario_report_7701_terminal_cli_modes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    # SCENARIO-REPORT-7701-TERMINAL: cold CLI reports only serialized evidence.
    blocked = runner.check("fixture", "fixture", tmp_path, "exists", True, False)
    value = runner.base_artifact("20260926", [blocked], {"producers": {}}, 0.0)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert runner.main(["--cold-replay", str(path)]) == 0
    assert json.loads(capsys.readouterr().out)["verdict_class"] == "blocked"
    calls: list[tuple] = []
    monkeypatch.setattr(runner, "run_experiment", lambda *args: calls.append(args))
    assert runner.main(["--date", "20260926", "--output", "fixture.json"]) == 0
    assert calls[0][2] == Path("fixture.json")
