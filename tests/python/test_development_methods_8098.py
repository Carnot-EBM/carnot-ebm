"""REQ-VERIFY-8098-PUBLIC/CUSTODY; REQ-REPORT-8098 private protocol checks."""

from copy import deepcopy
import json
from pathlib import Path
import os
import subprocess
import sys

import pytest

from carnot.verify import development_methods_8098 as m
from carnot import experiment_8098_v701_development_methods as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file

ROOT = Path(__file__).resolve().parents[2]
CLI = ROOT / "scripts/experiments/experiment_8098_v701_development_methods.py"


def fixture(root: Path, count: int = 640) -> Path:
    """Private corpus bytes cannot overwrite the historical research corpus."""
    data = root / "data/ragtruth"
    data.mkdir(parents=True)
    sources, responses = [], []
    for i in range(count):
        sid, text = str(i), f"source{i} fact{i} evidence{i} item{i} token{i}."
        sources.append(dict(source_id=sid, task_type="Summary", source_info=text))
        span = dict(
            start=0,
            end=6,
            text="Answer",
            label_type="baseless",
            implicit_true=False,
            due_to_null=False,
            meta=None,
        )
        responses.append(
            dict(
                id=sid,
                source_id=sid,
                split="train",
                model="fixture",
                response="Answer.",
                quality="good",
                labels=[span] if i % 3 else [],
            )
        )
    for name, rows in (("source_info", sources), ("response", responses)):
        (data / (name + ".jsonl")).write_text("".join(json.dumps(r) + "\n" for r in rows))
    return root


def test_public_projection_and_hash_selection(tmp_path):
    root = fixture(tmp_path)
    path = root / "data/ragtruth/response.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    rows[0]["labels"] = dict(id="injected", split="test", quality="secret")
    rows.append(dict(rows[1], id="extra", model="alternative"))
    rows.append(dict(rows[2], id="test-only", split="test"))
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    sources, public = m.public_training(root)
    assert all(set(r) == m.RESPONSE_KEYS for r in public)
    assert "injected" not in {r["id"] for r in public}
    roster, predictors, exclusions, clusters = m.select(sources, public)
    assert len(roster) == len(clusters) == 640 and not exclusions
    selected = next(r for r in roster if r["source_id"] == "1")
    assert selected["response_id"] == min(
        ("1", "extra"), key=lambda x: m.sha(b"V701-response" + x.encode())
    )
    assert all(set(r) == m.PUBLIC_KEYS for r in predictors)
    assert {k: sum(r["role"] == k for r in roster) for k in m.ROLES} == m.ROLES


def test_structured_transitive_duplicates_and_collisions(monkeypatch):
    base = " ".join(f"word{i}" for i in range(45))
    sources = [
        dict(source_id=str(i), task_type="QA", source_info=s)
        for i, s in enumerate(
            (
                base,
                base + " x y",
                base + " x y z w",
                {"z": [1, 2], "a": "full"},
                " ＨＥＬＬＯ ",
                "hello",
            )
        )
    ]
    responses = [
        dict(id=str(i), source_id=str(i), split="train", model="f", response="Answer.")
        for i in range(6)
    ]
    roster, public, _, clusters = m.select(sources, responses)
    assert len(clusters) == 3 and not roster and not public
    assert any(c["source_ids"] == ["0", "1", "2"] for c in clusters)
    assert m.render({"z": [1, 2], "a": "full"}) == b'{"a":"full","z":[1,2]}'
    monkeypatch.setattr(m, "sha", lambda value: "collision")
    with pytest.raises(ValueError, match="source_hash_collision"):
        m.select(sources, responses)


def test_invalid_public_sources_and_fields(tmp_path):
    sources = [dict(source_id="0", task_type="QA", source_info=" ")]
    responses = [dict(id="0", source_id="0", split="train", model="f", response="")]
    assert m.select(sources, responses)[2][0]["exclusion_reason"] == "empty_source"
    with pytest.raises(ValueError, match="public_metadata"):
        m.select(sources, [dict(responses[0], y=1)])
    root = fixture(tmp_path, 1)
    path = root / "data/ragtruth/response.jsonl"
    path.write_text('{"id":"0","source_id":"0","split":"train","response":9,"model":"x"}\n')
    with pytest.raises(ValueError, match="public_metadata"):
        m.public_training(root)


def test_custody_seal_support_and_unknown(tmp_path):
    root = fixture(tmp_path / "corpus")
    plan = m.seal(root, tmp_path / "raw")
    assert plan["cohort_ready_score"] == plan["fit_support_ready_score"] == 1
    assert plan["selection_sealed_before_labels"] is True
    assert len(plan["rows"]) == 640
    assert sum(r["denominator"] for r in plan["rows"]) == 640
    assert all(r["issued_state"] == "methods_sealed" for r in plan["rows"])
    ev = json.loads(Path(plan["evaluator_label_manifests"]["fit"]["path"]).read_text())
    assert ev["annotation_rows"] and ev["annotation_rows"][0]["start_byte"] == 0
    assert len(ev["original_response_records"]) == 128
    assert Path(plan["evaluator_label_manifests"]["fit"]["path"]).stat().st_mode & 0o777 == 0o600
    selected = json.loads(Path(plan["selection"]["path"]).read_text())
    with pytest.raises(ValueError, match="immutable_drift"):
        m.immutable(Path(plan["selection"]["path"]), dict(selected, roster=[]))
    assert m.immutable(Path(plan["selection"]["path"]), selected) == plan["selection"]
    r = json.loads((root / "data/ragtruth/response.jsonl").read_text().splitlines()[0])
    r.pop("labels")
    target, spans = m.target(r, b"Answer.")
    assert target["y"] is None and spans == []
    r.update(labels=[], quality="truncated")
    assert m.target(r, b"Answer.")[0]["y"] is None


@pytest.mark.parametrize("mutation", ["contamination", "duplicate", "roles"])
def test_private_mutations(tmp_path, mutation):
    root = fixture(tmp_path / "corpus")
    plan = m.seal(root, tmp_path / "raw", mutation)
    assert plan["cohort_ready_score"] == plan["fit_support_ready_score"] == 0
    assert plan["owned_failure"] and not plan["evaluator_label_manifests"]


def test_blocked_and_missing_annotations(tmp_path):
    assert m.seal(tmp_path / "missing", tmp_path / "raw")["failures"]
    root = fixture(tmp_path / "corpus", 1)
    plan = m.seal(root, tmp_path / "small")
    assert plan["cohort_ready_score"] == 0 and plan["failures"][-1]["observed"] == 1
    root = fixture(tmp_path / "unknown", 640)
    path = root / "data/ragtruth/response.jsonl"
    rows = [json.loads(s) for s in path.read_text().splitlines()]
    for r in rows:
        r.pop("labels")
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    plan = m.seal(root, tmp_path / "unknown-raw")
    assert plan["cohort_ready_score"] == 1 and plan["fit_support_ready_score"] == 0
    assert all(r["status"] == "excluded" for r in plan["rows"])
    ev = json.loads(Path(plan["evaluator_label_manifests"]["fit"]["path"]).read_text())
    assert all("labels" not in r for r in ev["original_response_records"])


def test_cli_external_success_block_mutation_replay(tmp_path):
    """SCENARIO-REPORT-8098-CLI uses the actual script without ambient imports."""
    fixture(tmp_path / "corpus")
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    for mode, expected in (
        ("success", "null"),
        ("blocked", "blocked"),
        ("contamination", "disqualified"),
        ("duplicate", "disqualified"),
    ):
        output = tmp_path / mode / "experiment_8098_v701_development_methods.json"
        args = [
            sys.executable,
            "-u",
            str(CLI),
            "--fixture-root",
            str(tmp_path / ("missing" if mode == "blocked" else "corpus")),
            "--output",
            str(output),
        ]
        if mode in {"contamination", "duplicate"}:
            args += ["--mutate", mode]
        result = subprocess.run(
            args, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(output.read_text())
        assert value["verdict_class"] == expected
        replay = subprocess.run(
            [sys.executable, str(CLI), "--cold-replay", str(output)],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert replay.returncode == 0, replay.stdout + replay.stderr
        value["completed_count"] += 1
        output.write_text(json.dumps(value))
        assert (
            subprocess.run(
                [sys.executable, str(CLI), "--cold-replay", str(output)],
                cwd=tmp_path,
                env=env,
                capture_output=True,
                timeout=60,
            ).returncode
            == 1
        )


def test_projection_missing_fields_and_unicode_spans():
    with pytest.raises(ValueError, match="public_metadata"):
        m.public_fields("{}")
    row = dict(
        response="é.",
        quality="good",
        labels=[
            dict(
                start=0,
                end=1,
                text="é",
                label_type="baseless",
                implicit_true=False,
                due_to_null=False,
                meta=None,
            )
        ],
    )
    target, spans = m.target(row, "é.".encode())
    assert target["y"] == 1 and spans[0]["end_byte"] == 2
    assert m.target(dict(row, response="changed"), "é.".encode())[0]["y"] is None


def private_inputs(root):
    """Toy authenticated history cannot modify historical primary bytes."""
    for name in e.INPUTS:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
    history = root / e.HISTORY
    terminal = history.parent / "raw" / history.stem / "terminal.json"
    sidecar = terminal.parent / "validators" / "toy.json"
    atomic_json(
        history,
        dict(
            terminal_validation_sidecar_path=str(terminal),
            honest_verdict="complete_blocked_available_history",
            known_excluded_count=2515,
            gate_check_summary=[dict(check="available_history", observed=[{}] * 789)],
        ),
    )
    digest = sha256_file(history)
    atomic_json(sidecar, dict(primary_sha256=digest, report=dict(passed=True)))
    atomic_json(terminal, dict(publication=dict(primary_sha256=digest, sidecar_path=str(sidecar))))
    literature = root / "paper.html"
    literature.write_text("toy primary byte-custody fixture")
    atomic_json(
        root / e.LITERATURE,
        dict(rows=[dict(path=str(literature), sha256=sha256_file(literature), status=200)]),
    )
    return history, terminal, sidecar


def test_preconditions_and_historical_operands(tmp_path):
    root = tmp_path / "inputs"
    history, terminal, sidecar = private_inputs(root)
    assert not e.preconditions(root)["failures"]
    (root / "data/ragtruth/source_info.jsonl").unlink()
    plan = e.produce(root, tmp_path / "sealed", "", False)
    assert plan["historical_operands"]["malformed_history_observations"] == 789
    assert plan["historical_operands"]["known_exclusions"] == 2515
    assert plan["failures"]
    sidecar.write_text("{}")
    assert e.preconditions(root)["failures"][-1]["check"] == "authenticated_terminal"
    assert e.produce(tmp_path / "missing", tmp_path / "blocked", "", False)["failures"]


def test_frozen_protocol_and_independent_replay(tmp_path):
    root = fixture(tmp_path / "corpus")
    plan = e.produce(root, tmp_path / "raw", "", True)
    methods = plan["method_config"]
    assert methods["additive_basis"]["basis_columns"] == 63
    assert methods["statistical_plan"]["primary_block"] == 16
    assert methods["precision_bounds"]["64"]["zero_event_95_upper"] > 0.04
    value = e.build(plan, dict(passed=True, receipts=[]), tmp_path / "raw", 1, True)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    value["class_support"]["fit"]["eligible"] -= 1
    atomic_json(path, value)
    assert not e.replay(path)
    value["class_support"]["fit"]["eligible"] += 1
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(path, value)
    assert not e.replay(path)
    assert not e.replay(tmp_path / "missing.json")


def test_validation_and_terminal_supervision(tmp_path, monkeypatch):
    """Synthetic supervisor receipts exercise errors, not scientific readiness."""

    def fake(root, spec, private, durable):
        if spec["name"] == "coverage_json":
            atomic_json(
                private / "coverage.json",
                dict(
                    files={
                        n: dict(summary=dict(num_statements=1, covered_lines=1), missing_lines=[])
                        for n in e.OWNED
                    }
                ),
            )
        return dict(passed=True, test_only_synthetic_receipt=True)

    monkeypatch.setattr(e, "run_check", fake)
    assert e.validate(tmp_path / "passing")["passed"]
    assert e.terminal(tmp_path / "candidate.json", tmp_path / "terminal")["passed"]
    monkeypatch.setattr(
        e, "run_check", lambda *args: dict(passed=False, test_only_synthetic_receipt=True)
    )
    assert not e.validate(tmp_path / "failed")["passed"]
    assert not e.terminal(tmp_path / "candidate.json", tmp_path / "failed-terminal")["passed"]


def test_readiness_and_code_receipt_mutations(tmp_path):
    root = fixture(tmp_path / "corpus")
    response = root / "data/ragtruth/response.jsonl"
    records = [json.loads(line) for line in response.read_text().splitlines()]
    for row in records:
        row.pop("labels")
    response.write_text("".join(json.dumps(row) + "\n" for row in records))
    plan = e.produce(root, tmp_path / "raw", "", True)
    value = e.build(plan, dict(passed=True, receipts=[]), tmp_path / "raw", 1, True)
    path = tmp_path / "candidate.json"
    atomic_json(path, value)
    assert e.replay(path)
    for field in ("fit_support_ready_score", "cohort_ready_score"):
        changed = deepcopy(value)
        changed[field] = 1 - changed[field]
        atomic_json(path, changed)
        assert not e.replay(path)
    changed = deepcopy(value)
    changed["code_config_hashes"][e.OWNED[0]] = "changed"
    atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["validation_receipts"] = [
        dict(log_path=str(tmp_path / "absent.log"), log_sha256="missing", passed=True)
    ]
    atomic_json(path, changed)
    assert not e.replay(path)
    log = tmp_path / "changed.log"
    log.write_text("actual changed validation bytes")
    changed["validation_receipts"] = [dict(log_path=str(log), log_sha256="incorrect", passed=True)]
    atomic_json(path, changed)
    assert not e.replay(path)
    changed = deepcopy(value)
    changed["fixture_protocol_only"] = False
    atomic_json(path, changed)
    assert not e.replay(path)


def test_malformed_sources_are_preserved_exclusions():
    sources = [
        dict(source_id=str(i), task_type="QA", source_info=v)
        for i, v in enumerate((None, 5, {}, [], True))
    ]
    responses = [
        dict(id=str(i), source_id=str(i), split="train", model="fixture", response="Answer.")
        for i in range(len(sources))
    ]
    roster, public, excluded, clusters = m.select(sources, responses)
    assert not roster and not public and not clusters
    assert len(excluded) == 5 and all(
        r["exclusion_reason"] == "malformed_public_source" for r in excluded
    )


def test_parent_publication_requires_normal_child_and_validation(tmp_path, monkeypatch):
    root = fixture(tmp_path / "corpus")

    def child_run(checkout, spec, private, durable):
        sealed = Path(spec["argv"][spec["argv"].index("--seal-output") + 1])
        atomic_json(sealed, e.produce(root, sealed.parent, "", True))
        log = tmp_path / "child.log"
        log.write_text("private synthetic supervision fixture")
        return dict(
            passed=True,
            name="seal_child",
            log_path=str(log),
            log_sha256=sha256_file(log),
            test_only_synthetic_receipt=True,
        )

    monkeypatch.setattr(e, "run_check", child_run)
    monkeypatch.setattr(e, "validate", lambda raw: dict(passed=True, receipts=[]))
    monkeypatch.setattr(e, "terminal", lambda *args: dict(passed=False, receipts=[]))
    output = tmp_path / "publication" / (e.NAME + ".json")
    health = tmp_path / "health.json"
    atomic_json(health, dict(status="test_only"))
    assert (
        e.main(
            [
                "--root",
                str(root),
                "--output",
                str(output),
                "--health-receipt",
                str(health),
                "--mutate",
                "roles",
            ]
        )
        == 0
    )
    value = json.loads(output.read_text())
    assert value["verdict_class"] == "disqualified" and value["cohort_ready_score"] == 0
    assert value["required_checks_passed"] is False and value["flagged_adversarial"] is True
    assert e.main(["--output", str(output)]) == 0
    assert e.main(["--cold-replay", str(output)]) == 0
    value["completed_count"] += 1
    atomic_json(output, value)
    assert e.main(["--output", str(output)]) == 1
    assert e.main(["--cold-replay", str(output)]) == 1
    sealed = tmp_path / "seal-child" / "plan.json"
    assert e.main(["--fixture-root", str(root), "--seal-output", str(sealed)]) == 0
    assert json.loads(sealed.read_text())["cohort_ready_score"] == 1
    monkeypatch.setattr(e, "run_check", lambda *args: dict(passed=False, name="seal_child"))
    monkeypatch.setattr(e, "terminal", lambda *args: dict(passed=True, receipts=[]))
    blocked_output = tmp_path / "child-failed" / (e.NAME + ".json")
    assert e.main(["--root", str(root), "--output", str(blocked_output)]) == 0
    assert json.loads(blocked_output.read_text())["verdict_class"] == "disqualified"
    fixture_output = tmp_path / "fixture-publication" / (e.NAME + ".json")
    assert e.main(["--fixture-root", str(root), "--output", str(fixture_output)]) == 0
