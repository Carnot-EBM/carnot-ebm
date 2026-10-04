"""REQ-REPORT-8084, REQ-VERIFY-8084: new source roles precede human targets."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot import experiment_8084_v700_fresh_cohort_methods as m
from carnot.verify import fresh_cohort_8084 as c


def fixture(root, count=768):
    """Private official-shaped records let controls test custody without live labels."""
    data = root / "data/ragtruth"
    data.mkdir(parents=True)
    sources, responses = [], []
    for index in range(count):
        text = f"Source{index} has distinctive word{index} fact{index} value{index} token{index}."
        sid = f"source-{index}"
        sources.append(dict(source_id=sid, source_info=text))
        responses.append(
            dict(
                id=str(index * 2),
                source_id=sid,
                split="train",
                response=text,
                quality="good",
                labels=[],
            )
        )
        responses.append(
            dict(
                id=str(index * 2 + 1),
                source_id=sid,
                split="train",
                response="Other.",
                quality="good",
                labels=[],
            )
        )
    for name, rows in [("source_info", sources), ("response", responses)]:
        (data / (name + ".jsonl")).write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    (root / "results").mkdir()
    return sources, responses


def test_inventory_and_selection(tmp_path):
    """SCENARIO-VERIFY-8084-POOL: IDs, exact content and near clusters are excluded."""
    sources, responses = fixture(tmp_path, 770)
    history = tmp_path / "results/prior.json"
    history.write_text(
        json.dumps(
            dict(source_ids=["source-0"], source_bytes=sources[1]["source_info"].encode().hex())
        )
    )
    inventory = c.inventory(tmp_path, sources, tmp_path / "new")
    assert inventory["known_source_ids"] == ["source-0", "source-1"]
    assert inventory["declarations"] and inventory["files"]
    roster, public, exclusions, clusters = c.select(sources, responses, inventory)
    assert len(roster) == len(public) == 768
    assert len(exclusions) == 2
    assert all(int(r["response_id"]) % 2 == 0 for r in roster)
    assert [r["selection_hash"] for r in roster] == sorted(r["selection_hash"] for r in roster)
    assert {role: sum(r["role"] == role for r in roster) for role in c.ROLES} == c.ROLES
    assert clusters == []
    changed = deepcopy(responses)
    changed[0]["quality"] = "unknown"
    assert c.select(sources, changed, inventory)[:2] == (roster, public)
    assert c.select(sources, responses + [dict(responses[0], split="test")], inventory)[0] == roster
    cached = c.immutable(tmp_path / "cached.json", inventory)
    reused = c.inventory(tmp_path, sources, tmp_path / "new", Path(cached["path"]))
    assert reused["known_source_ids"] == inventory["known_source_ids"]
    history.write_text("changed history")
    assert any(
        r["reason"] == "history_input_hash"
        for r in c.inventory(tmp_path, sources, tmp_path / "new", Path(cached["path"]))[
            "unknown_history"
        ]
    )
    sources[3]["source_info"] += " More public text stays identical."
    sources[2]["source_info"] = sources[3]["source_info"] + " extra"
    assert c.select(sources, responses, inventory)[3]


def test_seal_unknown_and_mutations(tmp_path):
    """SCENARIO-VERIFY-8084-SEPARATION: unknown labels stay excluded after sealing."""
    _, responses = fixture(tmp_path)
    responses[0].pop("labels")
    (tmp_path / "data/ragtruth/response.jsonl").write_text(
        "\n".join(json.dumps(r) for r in responses)
    )
    raw = tmp_path / "sealed"
    plan = c.seal(tmp_path, raw)
    assert not plan["failures"]
    assert plan["selection_sealed_before_labels"] is True
    assert sum(r["eligible"] for r in plan["rows"]) == 767
    assert any(r["exclusion_reason"] == "annotation_custody:annotations" for r in plan["rows"])
    for role, ref in plan["role_manifests"].items():
        rows = json.loads(Path(ref["path"]).read_text())["rows"]
        assert not any("y" in row or "labels" in row for row in rows)
        assert len(rows) == c.ROLES[role]
    assert all(Path(ref["path"]).stat().st_mode & 0o222 == 0 for ref in plan["raw_shard_hashes"])
    for route in ["contamination", "duplicate", "unavailable_history"]:
        blocked = c.seal(tmp_path, tmp_path / route, mutation=route)
        assert blocked["failures"]
    missing = c.seal(tmp_path / "absent", tmp_path / "missing")
    assert missing["failures"][0]["observed"] is False
    short = tmp_path / "short"
    fixture(short, 1)
    assert c.seal(short, short / "seal")["failures"][0]["field"] == "disjoint_public_groups"


def test_bad_history_and_duplicate_groups(tmp_path):
    """REQ-VERIFY-8084: unavailable history and malformed declarations cannot pass."""
    sources, responses = fixture(tmp_path, 2)
    bad = tmp_path / "results/bad.json"
    bad.write_text('{"source_bytes":"ff"}')
    inv = c.inventory(tmp_path, sources, tmp_path / "new")
    assert inv["unknown_history"]
    bad.unlink()
    bad.write_text('{"source_ids":[{}]}')
    opaque = tmp_path / "checkpoints/opaque.npz"
    opaque.parent.mkdir()
    opaque.write_bytes(b"opaque")
    c.immutable(
        tmp_path / "results/hash.json",
        dict(source_hashes=[c.normalized(sources[0]["source_info"].encode())]),
    )
    inv = c.inventory(tmp_path, sources, tmp_path / "new")
    assert inv["known_source_ids"] == ["source-0"]
    assert any(r["reason"] == "opaque_checkpoint" for r in inv["unknown_history"])
    bad.write_text('{"source_hashes":[{}]}')
    assert any(
        r["blocking"] for r in c.inventory(tmp_path, sources, tmp_path / "new")["unknown_history"]
    )
    bad.unlink()
    sources[1]["source_info"] = sources[0]["source_info"]
    roster = c.select(sources, responses, dict(known_source_ids=[]))[0]
    assert len(roster) == 1
    public = [
        dict(family_id="a", source_bytes=b"Same".hex(), answer_bytes=b"Answer".hex()),
        dict(family_id="b", source_bytes=b"same".hex(), answer_bytes=b"Answer".hex()),
    ]
    with pytest.raises(ValueError, match="cross_role_duplicate"):
        c.separation(public, [dict(role="fit"), dict(role="tune")])
    public[0]["y"] = 0
    with pytest.raises(ValueError, match="public_fields"):
        c.separation(public[:1], [dict(role="fit")])


def test_methods_and_reduction(tmp_path):
    """REQ-REPORT-8084: frozen Gaussian methods have no automatic paper guarantee."""
    fixture(tmp_path, 1)
    plan = c.seal(tmp_path, tmp_path / "raw")
    methods = m.methods()
    assert len(methods["features"]) == 9
    assert methods["hypothesis_family"] == [
        "H1_radial_source_decisions",
        "H2_feedback_grown_memory",
    ]
    assert methods["memory"]["initial_centers"] == 16
    assert methods["memory"]["maximum_additions"] == 12
    value = m.build(plan, [], {}, 1.0, fixture=True)
    assert value["verdict_class"] == "blocked"
    assert value["generalized_learning_benefit_score"] == value["cohort_ready_score"] == 0
    assert set(value) - {"field_principles"} <= set(value["field_principles"])
    assert m.build(plan, [], {}, 1.0, fixture=False)["verdict_class"] == "disqualified"


@pytest.mark.parametrize(
    "route", ["success", "blocked", "contamination", "duplicate", "unavailable_history"]
)
def test_private_cli(tmp_path, route):
    """SCENARIO-REPORT-8084-CLI: real subprocesses have no ambient checkout path."""
    root = tmp_path / "input"
    fixture(root, 768 if route != "blocked" else 1)
    output = tmp_path / "output" / (m.NAME + ".json")
    cmd = [
        sys.executable,
        "-u",
        str(m.ROOT / m.CLI),
        "--fixture-output",
        str(output),
        "--root",
        str(root),
    ]
    if route not in {"success", "blocked"}:
        cmd += ["--mutate", route]
    config = os.environ.get("CARNOT_8084_COVERAGE_CONFIG")
    if config:
        cmd = [sys.executable, "-m", "coverage", "run", "--rcfile=" + config, *cmd[2:]]
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    env["PYTHONUNBUFFERED"] = "1"
    print("8084 CLI before " + route, flush=True)
    child = subprocess.run(cmd, cwd=tmp_path, env=env, capture_output=True, text=True, timeout=90)
    print("8084 CLI after " + route, flush=True)
    assert child.returncode == 0, child.stdout + child.stderr
    value = json.loads(output.read_text())
    assert value["verdict_class"] == ("null" if route == "success" else "blocked")
    assert m.replay(output)
    replay_cmd = [sys.executable, "-u", str(m.ROOT / m.CLI), "--cold-replay", str(output)]
    replay = subprocess.run(replay_cmd, cwd=tmp_path, env=env, capture_output=True, timeout=30)
    assert replay.returncode == 0
    assert m.main(["--cold-replay", str(output)]) == 0
    value["eligible_count"] += 1
    m.atomic_json(output, value)
    assert m.main(["--cold-replay", str(output)]) == 1


def test_replay_and_owned_failure(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8084-VALIDATION: real receipts own readiness; drift is rejected."""
    root = tmp_path / "input"
    fixture(root, 1)
    output = tmp_path / (m.NAME + ".json")
    monkeypatch.setattr(m, "terminal", lambda p: dict(passed=True))
    assert m.main(["--fixture-output", str(output), "--root", str(root)]) == 0
    original = json.loads(output.read_text())
    for key in ["code_config_hashes", "raw_shard_hashes"]:
        changed = deepcopy(original)
        if key == "code_config_hashes":
            changed[key][m.MODULE] = "wrong"
        else:
            changed[key][0]["sha256"] = "wrong"
        m.atomic_json(output, changed)
        assert not m.replay(output)
    m.atomic_json(output, original)
    assert m.replay(output)
    assert m.main(["--fixture-output", str(output), "--root", str(root)]) == 1
    output.write_text("invalid")
    assert not m.replay(output)
    coverage = {p: {"summary": {"num_statements": 1, "missing_lines": 0}} for p in m.OWNED}

    def check(repo, spec, private, durable):
        if spec["name"] == "seal_child":
            raw = Path(spec["argv"][-1]).parent
            plan = c.seal(root, raw)
            m.atomic_json(raw / "seal.json", plan)
        m.atomic_json(private / "coverage.json", dict(files=coverage))
        log = durable / (spec["name"] + ".log")
        log.parent.mkdir(parents=True, exist_ok=True)
        log.write_text("private validation control")
        return dict(
            spec, passed=True, exit_code=0, log_path=str(log), log_sha256=m.sha256_file(log)
        )

    monkeypatch.setattr(m, "run_check", check)
    good = tmp_path / "production" / (m.NAME + ".json")
    assert m.main(["--output", str(good), "--root", str(root)]) == 0
    assert m.replay(good)
    good_value = json.loads(good.read_text())
    health = tmp_path / "health.json"
    m.atomic_json(health, good_value["repository_health"][0])
    reused = tmp_path / "reused" / (m.NAME + ".json")
    assert (
        m.main(["--output", str(reused), "--root", str(root), "--health-receipt", str(health)]) == 0
    )
    receipt = json.loads(health.read_text())
    receipt["log_sha256"] = "wrong"
    m.atomic_json(health, receipt)
    with pytest.raises(ValueError, match="unauthenticated_health_receipt"):
        m.main(
            [
                "--output",
                str(tmp_path / "invalid" / (m.NAME + ".json")),
                "--health-receipt",
                str(health),
            ]
        )
    good_value["validation_receipts"][0]["log_sha256"] = "wrong"
    m.atomic_json(good, good_value)
    assert not m.replay(good)
    bad = tmp_path / "failed" / (m.NAME + ".json")
    monkeypatch.setattr(m, "run_check", lambda *args: dict(name=args[1]["name"], passed=False))
    assert m.main(["--output", str(bad), "--root", str(root)]) == 1


def test_immutable_and_preconditions(tmp_path):
    """REQ-REPORT-8084: seals refuse changed bytes and tools report actual paths."""
    p = tmp_path / "immutable.json"
    c.immutable(p, dict(value=1))
    with pytest.raises(ValueError, match="immutable_drift"):
        c.immutable(p, dict(value=2))
    assert m.preconditions(tmp_path)["failures"]
    actual = m.preconditions(m.ROOT)
    assert actual["checks"] and actual["references"]
    assert c.text_bytes(dict(b=2, a=1)) == b'{"a":1,"b":2}'


def test_worker_and_primitive_drift(tmp_path, monkeypatch):
    """REQ-REPORT-8084: normal exited workers and independent row rebuild resist drift."""
    root = tmp_path / "input"
    fixture(root)
    (root / "results/experiment_7980_private.json").write_text("invalid")
    assert any(r["field"] == "authenticated_terminal" for r in m.preconditions(root)["failures"])
    assert m.main(["--root", str(root), "--seal-output", str(tmp_path / "worker/seal.json")]) == 0
    reference = dict(
        path=str(root / "data/ragtruth/source_info.jsonl"),
        sha256=m.sha256_file(root / "data/ragtruth/source_info.jsonl"),
    )
    monkeypatch.setattr(
        m,
        "preconditions",
        lambda r: dict(checks=[], failures=[], references=[reference.copy(), reference.copy()]),
    )
    assert (
        m.main(
            ["--root", str(root), "--seal-output", str(tmp_path / "duplicate-snapshots/seal.json")]
        )
        == 0
    )
    monkeypatch.setattr(m, "terminal", lambda path: dict(passed=True))
    output = tmp_path / "success" / (m.NAME + ".json")
    assert m.main(["--root", str(root), "--fixture-output", str(output)]) == 0
    value = json.loads(output.read_text())
    raw = Path(value["terminal_validation_sidecar_path"]).parent
    seal = raw / "seal.json"
    plan = json.loads(seal.read_text())
    plan["rows"][0]["source"] = "wrong"
    seal.chmod(0o600)
    m.atomic_json(seal, plan)
    next(ref for ref in value["raw_shard_hashes"] if ref["path"] == str(seal))["sha256"] = (
        m.sha256_file(seal)
    )
    m.atomic_json(output, value)
    assert not m.replay(output)
