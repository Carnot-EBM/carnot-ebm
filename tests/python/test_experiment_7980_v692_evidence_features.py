"""REQ-REPORT-7980: authenticated custody and private terminal CLI routes."""

import copy
import itertools
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest

from carnot import experiment_7980_v692_evidence_features as e
from carnot.reporting import evidence_features_custody_7980 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.source_boundary_7892 import EIGHT_ROLES
from test_evidence_features_7980 import public


def fixture(tmp_path):
    upstream = {}
    roles = {}
    for role, count in EIGHT_ROLES.items():
        rows = [
            public(
                f"{role}-{i}",
                source=f"{role} record {i}. It has {i} units.",
                answer=f"It has {i + i % 2} units.",
            )
            for i in range(count)
        ]
        labels = [
            dict(
                family_id=r["family_id"],
                role=role,
                source_id=r["family_id"],
                source_cluster_id="sha256:"
                + __import__("hashlib").sha256(bytes.fromhex(r["source_bytes"])).hexdigest(),
                response_id=r["family_id"],
                response_sha256="sha256:"
                + __import__("hashlib").sha256(bytes.fromhex(r["answer_bytes"])).hexdigest(),
                y=i % 2,
                status="completed",
                exclusion_reason=None,
            )
            for i, r in enumerate(rows)
        ]
        p, v = tmp_path / f"public/{role}.json", tmp_path / f"evaluator/{role}.json"
        atomic_json(p, dict(role=role, request_rows=rows, boundaries=e.targets.freeze(rows)[1]))
        atomic_json(v, dict(role=role, rows=labels, annotation_rows=[]))
        roles[role] = dict(public=c.reference(p), evaluator=c.reference(v))
    for eid, spec in c.INPUTS.items():
        value = dict(
            experiment_id=eid,
            task_id=spec[1],
            run_date=spec[2],
            milestone=spec[3],
            verdict_class="null",
            flagged_adversarial=False,
            honest_verdict="complete_null_fixture",
            **{spec[4]: 1},
        )
        upstream[eid] = value
    upstream[7968].update(
        public_role_manifests={r: v["public"] for r, v in roles.items()},
        evaluator_role_manifests={r: v["evaluator"] for r, v in roles.items()},
    )
    upstream[7969]["rows"] = [
        dict(family_id=f"{role}-{i}", role=role, parsed=dict(probability=0.5), status="completed")
        for role in ("fit", "tune", "policy_design", "calibration_replay")
        for i in range(EIGHT_ROLES[role])
    ]
    upstream[7958]["rows"] = [
        dict(
            family_id=f"evaluation-{i}",
            arm="full_source",
            parsed=dict(probability=0.5),
            status="completed",
        )
        for i in range(64)
    ]
    heads = tmp_path / "heads.json"
    atomic_json(heads, dict(heads=dict(gibbs=[dict(arm="raw_qwen")])))
    upstream[7972]["heads_seal"] = c.reference(heads)
    pins = {}
    for eid, spec in c.INPUTS.items():
        p = tmp_path / spec[0]
        atomic_json(p, upstream[eid])
        pins[eid] = sha256_file(p)
    sources = [
        dict(source_id=f"fresh-{i}", source_info=f"Fresh record {i}. It has {i} units.")
        for i in range(97)
    ]
    responses = [
        dict(
            id=f"r-{i}",
            source_id=f"fresh-{i}",
            response=f"It has {i} units.",
            split="train",
            quality="good",
            labels=[],
        )
        for i in range(97)
    ]
    data = tmp_path / "data/ragtruth"
    data.mkdir(parents=True)
    # This malformed TEST row must be skipped before either child decodes it.
    responses.append(dict(split="test"))
    for name, rows in [("source_info", sources), ("response", responses)]:
        (data / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return pins


def test_missing_and_authenticated_inputs(tmp_path):
    failures, _ = c.authenticate(tmp_path)
    assert failures and all(
        set(("upstream_id", "path", "hash", "field", "op", "expected", "observed")) <= set(r)
        for r in failures
    )
    pins = fixture(tmp_path)
    with patch.dict(c.PINS, pins):
        failures, plan = c.authenticate(tmp_path)
    assert failures == [] and len(plan["upstream"]) == 4
    p = tmp_path / c.INPUTS[7968][0]
    value = json.loads(p.read_text())
    value["run_date"] = "19990101"
    atomic_json(p, value)
    with patch.dict(c.PINS, {7968: sha256_file(p)}):
        failures, _ = c.authenticate(tmp_path)
    assert failures
    with pytest.raises(ValueError, match="hash"):
        c.checked(dict(path=str(p), sha256="bad"))
    p.unlink()
    with pytest.raises(ValueError, match="hash"):
        c.checked(dict(path=str(p), sha256="bad"))


def test_missing_official_input_names_an_external_gate(tmp_path):
    pins = fixture(tmp_path)
    (tmp_path / "data/ragtruth/response.jsonl").unlink()
    with patch.dict(c.PINS, pins):
        failed, _ = c.authenticate(tmp_path)
    assert any(
        r["field"] == "exists" and r["observed"] is False and "response.jsonl" in r["path"]
        for r in failed
    )


def test_history_inventory_and_training_projection(tmp_path):
    fixture(tmp_path)
    path = tmp_path / "results/raw/old/manifest.json"
    atomic_json(
        path,
        dict(
            source_id="fresh-1",
            source_bytes=b"Fresh record 2. It has 2 units.".hex(),
            source_cluster_id="sha256:" + "a" * 64,
        ),
    )
    ids, hashes, audit = c.exposure(tmp_path, tmp_path / "results/raw/current")
    assert "fresh-1" in ids and e.f.normalized(b"Fresh record 2. It has 2 units.") in hashes
    assert audit["search_inventory"] and audit["discovered_exclusions"]
    malformed = tmp_path / "results/raw/old/nontext_manifest.json"
    atomic_json(malformed, dict(source_bytes="ff"))
    with patch.object(c.time, "monotonic", side_effect=itertools.count(0, 60)):
        _, _, audit = c.exposure(tmp_path, tmp_path / "results/raw/current")
    assert audit["invalid_source_byte_records"][0]["path"] == str(malformed)
    assert audit["invalid_source_byte_records"][0]["sha256"] == sha256_file(malformed)
    assert not audit["complete_history_custody"]
    projected = c.train_public(tmp_path)
    assert all(set(r) == {"id", "source_id", "split", "response"} for r in projected[1])
    p = tmp_path / "data/ragtruth/response.jsonl"
    with p.open("a") as stream:
        stream.write('{"split": "test", "labels": BROKEN}\n')
    assert c.train_public(tmp_path) == projected
    with (tmp_path / "data/ragtruth/source_info.jsonl").open("a") as stream:
        stream.write('{"source_id": "test-only", "source_info": BROKEN}\n')
    assert c.train_public(tmp_path) == projected


def test_q_joins_reject_duplicates_bad_probability_and_preserve_invalid():
    row = dict(family_id="one", status="completed", parsed=dict(probability=0.3))
    data = {7969: dict(rows=[row]), 7958: dict(rows=[dict(row, arm="shuffled_source")])}
    assert c.q_rows(data, [public()])[0]["q"] == 0.3
    data[7969]["rows"][0] = dict(row, status="excluded")
    assert c.q_rows(data, [public()])[0]["q"] is None
    data[7969]["rows"][0] = dict(row, parsed=dict(probability=2))
    with pytest.raises(ValueError, match="q_value"):
        c.q_rows(data, [public()])
    data[7969]["rows"] = [row, row]
    with pytest.raises(ValueError, match="q_duplicate"):
        c.q_rows(data, [public()])


def test_owned_measure_seals_roles_predicates_and_replay(tmp_path):
    pins = fixture(tmp_path)
    with patch.dict(c.PINS, pins):
        failures, plan = c.authenticate(tmp_path)
    assert not failures
    raw = tmp_path / "output/raw/run"
    value = e.measure(plan, tmp_path, raw)
    assert len(value["rows"]) == 640 and value["feature_views_ready_score"] == 1
    assert len(value["predicate_bank"]["unary"]) == 16
    assert value["fresh_panel_ready_score"] == 0  # No complete historical custody claim.
    assert value["label_access_events"][0]["public_seal_sha256"]
    assert not value["label_access_events"][-1]["labels_exported_to_parent"]
    assert value["metadata_invariance_receipt"]["cold_subprocess"]
    e.replay(value)
    bad = copy.deepcopy(value)
    bad["rows"][0]["feature_hash"] = "bad"
    with pytest.raises(ValueError, match="rows_drift"):
        e.replay(bad)
    bad = copy.deepcopy(value)
    bad["predicate_bank"]["unary"][0]["threshold"] = 999
    with pytest.raises(ValueError, match="predicate_drift"):
        e.replay(bad)
    bad = copy.deepcopy(value)
    bad["rows"][0]["role"] = "tune"
    with pytest.raises(ValueError, match="role_drift"):
        e.replay(bad)
    bad = copy.deepcopy(value)
    bad["rows"][0]["status"] = "failed"
    with pytest.raises(ValueError, match="custody_rows_drift"):
        e.replay(bad)


def test_cross_role_custody_and_q_absence(tmp_path):
    pins = fixture(tmp_path)
    with patch.dict(c.PINS, pins):
        _, plan = c.authenticate(tmp_path)
    p = c.checked(plan["upstream"][7968]["public_role_manifests"]["tune"])
    v = json.loads(p.read_text())
    v["request_rows"][0]["source_bytes"] = public("fit-0", source="fit record 0. It has 0 units.")[
        "source_bytes"
    ]
    atomic_json(p, v)
    plan["upstream"][7968]["public_role_manifests"]["tune"] = c.reference(p)
    with pytest.raises(ValueError, match="cross_role"):
        e.measure(plan, tmp_path, tmp_path / "raw")
    assert c.q_rows({7969: dict(rows=[]), 7958: dict(rows=[])}, [public()]) == [
        dict(family_id="one", q=None, origin=None)
    ]


def cli(args, tmp_path):
    argv = [
        str(e.ROOT / ".venv/bin/python"),
        "-u",
        str(e.ROOT / f"scripts/experiments/{e.NAME}.py"),
        *args,
    ]
    if os.environ.get("CARNOT_7980_COVERAGE_CONFIG"):
        argv[1:2] = [
            "-m",
            "coverage",
            "run",
            "--rcfile=" + os.environ["CARNOT_7980_COVERAGE_CONFIG"],
        ]
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(argv, cwd=tmp_path, env=env, text=True, capture_output=True, timeout=120)


def test_real_private_cli_fixture_blocked_replay_and_rejection(tmp_path):
    pins = fixture(tmp_path / "inputs")
    with patch.dict(c.PINS, pins):
        _, plan = c.authenticate(tmp_path / "inputs")
    fixture_path = tmp_path / "fixture.json"
    # Integer producer keys are restored by the fixture route after JSON serialization.
    atomic_json(fixture_path, dict(root=str(tmp_path / "inputs"), plan=plan))
    output = tmp_path / "success" / (e.NAME + ".json")
    done = cli(["--fixture-plan", str(fixture_path), "--output", str(output)], tmp_path)
    assert done.returncode == 0, done.stdout + done.stderr
    value = json.loads(output.read_text())
    assert (
        value["verdict_class"] == "circular_positive"
        and value["sample_size_budget"]["independent"] == 0
    )
    assert cli(["--cold-replay", str(output)], tmp_path).returncode == 0
    assert not any(value["model_invocation_counts"].values())
    report = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
    assert report["primary_sha256"] == sha256_file(output)
    blocked = tmp_path / "blocked" / (e.NAME + ".json")
    done = cli(
        ["--root", str(tmp_path / "absent"), "--validation-worker", "--output", str(blocked)],
        tmp_path,
    )
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.loads(blocked.read_text())["verdict_class"] == "blocked"
    assert cli(["--cold-replay", str(blocked)], tmp_path).returncode == 0
    for args in [
        ["--cold-replay", str(tmp_path / "missing")],
        ["--output", str(tmp_path / "wrong.json")],
        ["--validation-worker"],
        ["--date", "20260930"],
        ["--public-extract", str(fixture_path)],
        ["--reserved-join", str(fixture_path), "--feature-output", str(tmp_path / "bad.json")],
    ]:
        assert cli(args, tmp_path).returncode != 0


def test_validation_manifest_nonempty_coverage_and_failure_preservation(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    manifest = e.freeze_commands(tmp_path / "raw", scratch)
    names = {r["name"] for r in manifest["commands"]}
    assert {
        "changed_statement_coverage",
        "strict_mypy",
        "scoped_spec_coverage",
        "E2E_015_019",
        "repository_health",
    } <= names
    assert all(r["timeout_s"] <= 600 for r in manifest["commands"])
    value = e.base([])
    receipts = [
        dict(name="repository_health", passed=False, exit_code=2),
        dict(name="owned", passed=False, exit_code=1),
    ]
    e.apply_validation(value, receipts)
    assert value["verdict_class"] == "disqualified" and value["feature_views_ready_score"] == 0
    assert value["repository_health"]["exit_code"] == 2
    clean = e.base([])
    e.apply_validation(
        clean,
        [
            dict(name="repository_health", passed=False, exit_code=2),
            dict(name="owned", passed=True, exit_code=0),
        ],
    )
    assert clean["verdict_class"] == "null"


def test_parent_validation_coverage_archive_and_terminal_failure(tmp_path):
    fake = [dict(name="owned", passed=True, exit_code=0)]
    counts = {p: dict(summary=dict(num_statements=1, missing_lines=0)) for p in e.OWNED}

    def execute(root, commands, *, log_dir, **kwargs):
        cov = Path(
            json.loads((tmp_path / "out/raw" / e.NAME / "validation_commands.json").read_text())[
                "coverage_json"
            ]
        )
        atomic_json(cov, dict(files=counts))
        cov.with_name(".coverage.fake").write_bytes(b"receipt")
        return fake

    out = tmp_path / "out" / (e.NAME + ".json")
    historical = tmp_path / "out/raw" / e.NAME / "development_checks/revision/receipts.json"
    atomic_json(
        historical, dict(receipts=[dict(name="earlier_failure", passed=False, exit_code=1)])
    )
    with (
        patch.object(e, "run_commands", side_effect=execute),
        patch.object(e, "terminal_check", return_value=dict(passed=True, receipts=[])),
    ):
        assert e.main(["--root", str(tmp_path / "absent"), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["coverage_statement_counts"] and value["verdict_class"] == "blocked"
    assert any(
        r.get("scope") == "earlier_owned_revision" for r in value["historical_required_failures"]
    )
    value = e.base([])
    with patch.object(
        e, "terminal_check", return_value=dict(passed=False, receipts=[], flagged_adversarial=True)
    ):
        e.publish(tmp_path / "failed" / (e.NAME + ".json"), value)
    assert value["verdict_class"] == "disqualified" and value["flagged_adversarial"]
    with (
        patch.object(e, "reader_receipt", return_value=dict(passed=False)),
        patch.object(e, "terminal_check", return_value=dict(passed=True, receipts=[])),
    ):
        with pytest.raises(ValueError, match="primary_resolution"):
            e.publish(tmp_path / "bad-reader" / (e.NAME + ".json"), e.base([]))


@pytest.mark.parametrize(
    "case",
    [
        "public_role",
        "family_duplicate",
        "train_projection",
        "public_original",
        "evaluator_roster",
        "source_id_duplicate",
        "reserved_join",
        "metadata",
    ],
)
def test_measure_rejection_boundaries(tmp_path, monkeypatch, case):
    pins = fixture(tmp_path)
    with patch.dict(c.PINS, pins):
        _, plan = c.authenticate(tmp_path)
    original_extract = e.f.extract

    def local_child(name, argv, raw):
        output = Path(argv[argv.index("--feature-output") + 1])
        if name == "train_projection":
            sources, responses = c.train_public(tmp_path)
            atomic_json(output, dict(sources=sources, responses=responses))
        elif name == "reserved_join":
            seal = Path(argv[argv.index("--reserved-join") + 1])
            c.reserved_join(tmp_path, seal, output)
            if case == "metadata":
                monkeypatch.setattr(
                    e.f, "extract", lambda row: dict(original_extract(row), feature_hash="drift")
                )
        else:
            src = Path(argv[argv.index("--public-extract") + 1])
            e.f.extract_file(src, output)
        return dict(name=name, passed=name != case, exit_code=int(name == case))

    monkeypatch.setattr(e, "child", local_child)
    field = (
        "public_role_manifests"
        if case in {"public_role", "family_duplicate"}
        else "evaluator_role_manifests"
    )
    if case in {"public_role", "family_duplicate", "evaluator_roster", "source_id_duplicate"}:
        item = plan["upstream"][7968][field]["tune"]
        path = c.checked(item)
        view = json.loads(path.read_text())
        if case == "public_role":
            view["role"] = "wrong"
        elif case == "family_duplicate":
            view["request_rows"][0]["family_id"] = "fit-0"
            view["boundaries"] = e.targets.freeze(view["request_rows"])[1]
        elif case == "evaluator_roster":
            view["rows"].pop()
        else:
            view["rows"][0]["source_id"] = "fit-0"
        atomic_json(path, view)
        plan["upstream"][7968][field]["tune"] = c.reference(path)
    expected = {
        "public_role": "original_role_roster",
        "family_duplicate": "original_family_roster",
        "train_projection": "train_projection_failed",
        "public_original": "public_child_failed",
        "evaluator_roster": "evaluator_role_roster",
        "source_id_duplicate": "cross_role_source_id",
        "reserved_join": "reserved_join_failed",
        "metadata": "metadata_invariance_failed",
    }[case]
    with pytest.raises(ValueError, match=expected):
        e.measure(plan, tmp_path, tmp_path / "output/raw/run")


@pytest.mark.parametrize("coverage", ["missing", "empty", "uncovered"])
def test_parent_fails_closed_on_missing_or_incomplete_coverage(tmp_path, coverage):
    out = tmp_path / (e.NAME + ".json")

    def execute(root, commands, *, log_dir, **kwargs):
        manifest = json.loads((tmp_path / "raw" / e.NAME / "validation_commands.json").read_text())
        if coverage != "missing":
            files = (
                {}
                if coverage == "empty"
                else {p: dict(summary=dict(num_statements=1, missing_lines=1)) for p in e.OWNED}
            )
            atomic_json(Path(manifest["coverage_json"]), dict(files=files))
        return [dict(name="owned", passed=True, exit_code=0)]

    with (
        patch.object(e, "run_commands", side_effect=execute),
        patch.object(e, "terminal_check", return_value=dict(passed=True, receipts=[])),
    ):
        assert e.main(["--root", str(tmp_path / "absent"), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["verdict_class"] == "disqualified" and value["feature_views_ready_score"] == 0


def test_private_public_route_denies_evaluator_files(tmp_path):
    src = tmp_path / "evaluator/input.json"
    atomic_json(src, dict(request_rows=[public()]))
    done = cli(
        ["--public-extract", str(src), "--feature-output", str(tmp_path / "features.json")],
        tmp_path,
    )
    assert done.returncode == 1 and "predictor_evaluator_access" in done.stdout
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(dict(e.base([]), feature_views_ready_score=1))
