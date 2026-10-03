"""REQ-REPORT-8019: original roles and unknown targets survive the real CLI."""

import copy
import json
import os
from pathlib import Path
import subprocess

import pytest

from carnot import experiment_8019_v695_eligible_targets as e
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify.sentence_labels_7942 import digest


def fixture():
    """Complete artificial annotations test custody without natural evidence credit."""
    data = dict(public={}, evaluator={}, original_slot_roster=[], exposure_rows=[], exclusions={})
    for role, n in e.SLOTS.items():
        data["public"][role] = []
        data["evaluator"][role] = dict(rows=[], annotation_rows=[])
        for i in range(n):
            fid = f"{role}-{i}"
            source, answer = f"Source {fid}.".encode(), b"Answer."
            roster = dict(family_id=fid, role=role, source_cluster_id=normalized(source), slot=i)
            data["original_slot_roster"].append(roster)
            data["public"][role].append(
                dict(
                    roster,
                    source_bytes=source.hex(),
                    answer_bytes=answer.hex(),
                    q=0.5,
                    features=[0.0] * 8,
                    status="generated",
                    exclusion_reason=None,
                    capture_identity="fixture",
                    response_id=fid,
                )
            )
            data["evaluator"][role]["rows"].append(
                dict(
                    family_id=fid,
                    role=role,
                    y=i % 2,
                    quality="good",
                    custody_passed=True,
                    completely_annotated=True,
                    annotation_count=i % 2,
                    response_sha256=digest(answer),
                    response_id=fid,
                    exclusion_reason=None,
                )
            )
            if i % 2:
                data["evaluator"][role]["annotation_rows"].append(
                    dict(
                        family_id=fid,
                        response_id=fid,
                        response_sha256=digest(answer),
                        start_byte=0,
                        end_byte=6,
                        text="Answer",
                        text_equal=True,
                    )
                )
    data["feedback_schedule"] = dict(label_delay_slots=20, stream_slots=256)
    return data


def cli(tmp_path, data, *extra):
    """Run the script from private scratch and instrument its actual CLI statements."""
    src = tmp_path / "input.json"
    atomic_json(src, data)
    out = tmp_path / (e.NAME + ".json")
    prefix = [str(e.ROOT / ".venv/bin/python")]
    config = os.environ.get("CARNOT_8019_COVERAGE_CONFIG")
    if config:
        prefix += [
            "-m",
            "coverage",
            "run",
            "--data-file=" + str(Path(config).parent / ".coverage"),
            "--rcfile=" + config,
        ]
    env = dict(os.environ, PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    env.pop("PYTHONPATH", None)
    result = subprocess.run(
        prefix
        + [str(e.ROOT / e.OWNED[-1]), "--fixture-input", str(src), "--output", str(out), *extra],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return result, out


@pytest.mark.parametrize("mutation", ["missing", "partial", "truncated", "context"])
def test_unknown_targets_cli(tmp_path, mutation):
    """SCENARIO-REPORT-8019-MASK: absent or incomplete targets never become zero."""
    data = fixture()
    row = data["evaluator"]["calibration"]["rows"][1]
    if mutation == "missing":
        row["y"] = None
    elif mutation == "partial":
        row["annotation_count"] = 2
    elif mutation == "truncated":
        row["quality"] = "truncated"
        row["completely_annotated"] = False
    else:
        data["public"]["calibration"][1].update(
            status="excluded", q=None, features=None, exclusion_reason="context_truncated"
        )
    result, out = cli(tmp_path, data)
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(out.read_text())
    assert len(value["eligibility_rows"]) == 704
    assert value["support_by_role"]["calibration"]["eligible"] == 63
    target = next(r for r in value["eligibility_rows"] if r["family_id"] == "calibration-1")
    assert not target["target_eligible"] and target["denominator"] == 1
    public = json.loads(Path(value["public_manifests"]["calibration"]["path"]).read_text())
    assert all("y" not in r for r in public["rows"])
    private = json.loads(Path(value["evaluator_manifests"]["calibration"]["path"]).read_text())
    assert private["rows"][1]["eligible_y"] is None
    assert value["target_roles_ready_score"] == 0  # Fixtures cannot invent owned checks.
    assert e.replay(out)["passed"]


@pytest.mark.parametrize("mutation", ["overlap", "wrong_role", "label_leak", "target_field"])
def test_contract_rejection_cli(tmp_path, mutation):
    """SCENARIO-REPORT-8019-MASK: source overlap and malformed roles fail closed."""
    data = fixture()
    if mutation == "overlap":
        data["public"]["tune"][0]["source_bytes"] = data["public"]["fit"][0]["source_bytes"]
    elif mutation == "wrong_role":
        data["evaluator"]["stream"]["rows"][0]["role"] = "fit"
    elif mutation == "label_leak":
        data["public"]["fit"][0]["y"] = 1
    else:
        del data["evaluator"]["calibration"]["rows"][0]["y"]
    result, out = cli(tmp_path, data)
    assert result.returncode == 1 and not out.exists()


def test_mask_ignores_correctness_and_keeps_time():
    """SCENARIO-REPORT-8019-MASK: class support cannot select the eligibility mask."""
    data = fixture()
    first = e.reduce(data)
    data["evaluator"]["calibration"]["rows"][0]["y"] = None
    second = e.reduce(data)
    assert first["support_by_role"]["calibration"]["eligible"] == 64
    assert second["support_by_role"]["calibration"]["eligible"] == 63
    assert second["support_by_role"]["calibration"]["passed"]
    stream = [r for r in second["eligibility_rows"] if r["role"] == "stream"]
    assert [r["slot"] for r in stream] == list(range(256))
    assert stream[-1]["reveal_at_slot"] == 275 and stream[-1]["censor_status"]
    assert e.COSTS["escalate"] == 0.5


def test_real_custody_loader(tmp_path):
    """REQ-REPORT-8019: recover sealed original bytes, capture IDs and exposure receipts."""
    data = e.load_inputs(e.ROOT, tmp_path)
    value = e.reduce(data)
    assert value["support_by_role"]["calibration"]["eligible"] == 61
    assert value["support_by_role"]["calibration"]["class_counts"] == {"0": 50, "1": 11}
    assert all(r["passed"] for r in value["support_by_role"].values())
    assert all(Path(r["path"]).is_relative_to(tmp_path) for r in data["references"])
    assert all(Path(r["path"]).is_relative_to(tmp_path) for r in data["exposure_rows"])


def test_upstream_block_operands(tmp_path):
    """SCENARIO-REPORT-8019-PUBLICATION: missing and changed bytes retain exact operands."""
    with pytest.raises(e.InputBlock) as error:
        e.load_inputs(tmp_path, tmp_path / "raw")
    assert error.value.check["artifact_field"] == "checkpoints.bundle"
    path = tmp_path / "results/experiment_8008_v694_conditioned_energy_fit.json"
    atomic_json(path, {})
    with pytest.raises(e.InputBlock) as error:
        e.load_inputs(tmp_path, tmp_path / "raw")
    assert error.value.check["observed"] == "MISSING_CONTRACT_FIELD"
    atomic_json(path, dict(checkpoints=dict(bundle=dict(path=str(path), sha256="sha256:wrong"))))
    with pytest.raises(e.InputBlock) as error:
        e.load_inputs(tmp_path, tmp_path / "raw")
    assert error.value.check["artifact_field"] == "sha256"


def test_terminal_and_natural_main(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8019-PUBLICATION: readiness needs all owned measurements and checks."""

    calls = []

    def run(root, commands, **kwargs):
        calls.extend(c.name for c in commands)
        if commands[0].name == "unit_consumers_e2e015_019":
            scratch = Path(kwargs["extra_env"]["CARNOT_8019_COVERAGE_CONFIG"]).parent
            atomic_json(
                scratch / "coverage.json",
                dict(
                    files={
                        p: dict(summary=dict(missing_lines=0, num_statements=1)) for p in e.OWNED
                    }
                ),
            )
        return [dict(name=c.name, scope=c.scope, passed=True, exit_code=0) for c in commands]

    monkeypatch.setattr(e, "run_commands", run)
    data = fixture()
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: data)
    out = tmp_path / "good" / (e.NAME + ".json")
    assert e.main(["--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["target_roles_ready_score"] == 1
    assert e.main(["--output", str(out)]) == 0
    assert calls.count("repository_health") == 1
    assert e.terminal(out)["passed"]
    assert e.main(["--cold-replay", str(out)]) == 0
    value["target_roles_ready_score"] = 1
    value["verdict_class"] = "blocked"
    atomic_json(out, value)
    with pytest.raises(ValueError, match="unsafe_readiness"):
        e.replay(out)
    assert e.main(["--cold-replay", str(out)]) == 1
    data["evaluator"]["retention"]["rows"][:50] = [
        dict(r, y=None) for r in data["evaluator"]["retention"]["rows"][:50]
    ]
    blocked = tmp_path / "support" / (e.NAME + ".json")
    assert e.main(["--output", str(blocked)]) == 0
    value = json.loads(blocked.read_text())
    assert value["verdict_class"] == "blocked" and value["retention_targets_ready_score"] == 0
    assert value["fit_targets_ready_score"] == 1 and value["gate_check_summary"]
    monkeypatch.setattr(e, "run_commands", lambda *args, **kwargs: [])
    disqualified = tmp_path / "unchecked" / (e.NAME + ".json")
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: fixture())
    assert e.main(["--output", str(disqualified)]) == 0
    assert json.loads(disqualified.read_text())["verdict_class"] == "disqualified"


def test_cold_cli_changed_bytes(tmp_path):
    """SCENARIO-REPORT-8019-PUBLICATION: the actual replay CLI rejects changed sealed bytes."""
    result, out = cli(tmp_path, fixture())
    assert result.returncode == 0
    value = json.loads(out.read_text())
    checkpoint = Path(value["checkpoint_references"][0]["path"])
    checkpoint.write_text(checkpoint.read_text() + " ")
    result, _ = cli(tmp_path, fixture(), "--cold-replay", str(out))
    assert result.returncode == 1 and "bound_bytes_changed" in result.stdout


def test_replay_aggregate_and_manifest_drift(tmp_path):
    """SCENARIO-REPORT-8019-PUBLICATION: fresh reduction checks projections as well as totals."""
    result, out = cli(tmp_path, fixture())
    assert result.returncode == 0
    original = json.loads(out.read_text())
    value = copy.deepcopy(original)
    value["rows"][0]["numerator"] = 42
    atomic_json(out, value)
    with pytest.raises(ValueError, match="reduction_drift"):
        e.replay(out)
    path = Path(original["public_manifests"]["fit"]["path"])
    value = json.loads(path.read_text())
    value["rows"][0]["q"] = 0.75
    atomic_json(path, value)
    for ref in original["raw_shard_hashes"] + [original["public_manifests"]["fit"]]:
        if ref["path"] == str(path):
            ref["sha256"] = sha256_file(path)
    atomic_json(out, original)
    with pytest.raises(ValueError, match="manifest_drift"):
        e.replay(out)


@pytest.mark.parametrize(
    "mutation", ["roles", "count", "roster", "identity", "response", "spans", "value", "extra"]
)
def test_reducer_contracts(mutation):
    """SCENARIO-REPORT-8019-MASK: contracts distinguish wrong bytes from partial coverage."""
    data = fixture()
    if mutation == "roles":
        del data["public"]["fit"]
    elif mutation == "count":
        data["public"]["fit"].pop()
    elif mutation == "roster":
        data["evaluator"]["fit"]["rows"].pop()
    elif mutation == "identity":
        data["original_slot_roster"][0]["slot"] = 1
    elif mutation == "response":
        data["public"]["fit"][0]["answer_bytes"] = b"changed".hex()
    elif mutation == "spans":
        data["evaluator"]["fit"]["annotation_rows"][0]["text"] = "wrong"
        assert not e.reduce(data)["rows"][1]["target_eligible"]
        return
    elif mutation == "value":
        data["evaluator"]["fit"]["rows"][0]["y"] = 1
    else:
        data["original_slot_roster"].append(dict(family_id="extra"))
    with pytest.raises((ValueError, KeyError)):
        e.reduce(data)


def test_immutable_shard_reuse_and_damage(tmp_path):
    """SCENARIO-REPORT-8019-PUBLICATION: immutable names reject changed existing bytes."""
    ref = e.shard(tmp_path, "public", dict(value=1))
    assert e.shard(tmp_path, "public", dict(value=1)) == ref
    Path(ref["path"]).write_text("changed")
    with pytest.raises(ValueError, match="immutable_shard_changed"):
        e.shard(tmp_path, "public", dict(value=1))


@pytest.mark.parametrize(
    "mutation", ["producer", "identity", "fit_bytes", "reserved_bytes", "local_bytes"]
)
def test_loader_rejects_capture_drift(tmp_path, mutation):
    """REQ-REPORT-8019: authenticated captures bind exact historical producer bytes."""
    original = json.loads(
        (e.ROOT / "results" / "experiment_8008_v694_conditioned_energy_fit.json").read_text()
    )
    bundle = json.loads(Path(original["checkpoints"]["bundle"]["path"]).read_text())
    root = tmp_path / "root"

    def save(value, name):
        path = tmp_path / name
        atomic_json(path, value)
        return dict(path=str(path), sha256=sha256_file(path))

    if mutation == "producer":
        producer = json.loads(Path(bundle["references"][0]["path"]).read_text())
        producer["experiment_id"] = 0
        bundle["references"][0] = save(producer, "wrong_producer.json")
    elif mutation in {"identity", "fit_bytes"}:
        hist = json.loads(Path(bundle["references"][2]["path"]).read_text())
        ref = next(r for r in hist["cited_upstream_artifacts"] if r.get("producer_id") == 7969)
        cap = json.loads(Path(ref["path"]).read_text())
        row = next(r for r in cap["rows"] if r["role"] == "fit")
        row["capture_identity" if mutation == "identity" else "q"] = "changed"
        ref.update(save(cap, "wrong_capture.json"))
        bundle["references"][2] = save(hist, "wrong_history.json")
    elif mutation == "reserved_bytes":
        cap = json.loads(Path(bundle["references"][1]["path"]).read_text())
        cap["rows"][0]["q"] = "changed"
        bundle["references"][1] = save(cap, "wrong_reserved.json")
    else:
        raw = tmp_path / "raw"
        target = (
            raw / "custody" / (original["checkpoints"]["bundle"]["sha256"].split(":")[-1] + ".json")
        )
        target.parent.mkdir(parents=True)
        target.write_text("changed")
    if mutation != "local_bytes":
        original["checkpoints"]["bundle"] = save(bundle, "bundle.json")
    atomic_json(root / "results/experiment_8008_v694_conditioned_energy_fit.json", original)
    with pytest.raises(ValueError):
        e.load_inputs(root, tmp_path / "raw")


def test_missing_upstream_and_reader_rejection(tmp_path, monkeypatch):
    """SCENARIO-REPORT-8019-PUBLICATION: terminal blocks and consumer failures do not retry."""
    monkeypatch.setattr(e, "run_commands", lambda *args, **kwargs: [])
    out = tmp_path / "absent" / (e.NAME + ".json")
    assert e.main(["--root", str(tmp_path / "absent-inputs"), "--output", str(out)]) == 0
    value = json.loads(out.read_text())
    assert value["honest_verdict"] == "complete_blocked_eligible_targets"
    assert value["gate_check_summary"][0]["observed"] is None
    assert not value["target_roles_ready_score"] and e.replay(out)["passed"]
    monkeypatch.setattr(e, "load_inputs", lambda root, raw: fixture())
    monkeypatch.setattr(e, "reader_receipt", lambda *args, **kwargs: dict(passed=False))
    assert e.main(["--output", str(tmp_path / "reader" / (e.NAME + ".json"))]) == 1


def test_coverage_environment_override(tmp_path):
    """SCENARIO-REPORT-8019-PUBLICATION: explicit owned coverage resists health environment drift."""
    commands = e.validation_plan(tmp_path)
    assert "--data-file=" + str(tmp_path / ".coverage") in commands[0].argv
