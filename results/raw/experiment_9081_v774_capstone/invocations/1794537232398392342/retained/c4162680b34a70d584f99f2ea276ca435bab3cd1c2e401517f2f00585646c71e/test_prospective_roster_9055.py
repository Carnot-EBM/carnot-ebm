"""REQ-REPORT-9055-CONTINUATION: private deterministic construction and refusal checks."""

from copy import deepcopy
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from carnot.reporting import prospective_roster_9055 as d
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash


def private_prefix():
    """Small real proposals exercise mechanics without being production measurements."""
    with patch.object(d.old, "progress"):
        return d.old.construct(
            dict(old_graphs=[], old_seeds=[]),
            Path("/tmp") / "carnot-9055-prefix",
            proposal_limit=40,
        )


def test_cap_crossing_and_exact_resume(tmp_path):
    """SCENARIO-REPORT-9055-CAP-RESUME: crossing the old cap preserves every prefix byte."""
    prefix = private_prefix()
    inventory = dict(old_graphs=[], old_seeds=[])
    before = canonical_hash(prefix)
    result = d.construct(prefix, inventory, tmp_path / "one", proposal_limit=20002)
    assert result["proposal_count"] == 20002
    assert result["graph_manifest"][: len(prefix["graph_manifest"])] == prefix["graph_manifest"]
    assert result["proposal_rows"][:40] == prefix["proposal_rows"]
    assert canonical_hash(prefix) == before
    first = d.construct(prefix, inventory, tmp_path / "resume", proposal_limit=100)
    resumed = d.construct(prefix, inventory, tmp_path / "resume", proposal_limit=20002)
    for key in ("graph_manifest", "proposal_rows", "proposal_count"):
        assert resumed[key] == result[key]
    assert first["proposal_count"] == 100
    assert result["terminal_reason"] == "proposal_budget_exhausted"


def test_complete_time_and_tampered_ledger(tmp_path):
    """REQ-REPORT-9055-CONTINUATION: no smaller cohort and no forged resume state."""
    prefix = private_prefix()
    inventory = dict(old_graphs=[], old_seeds=[])
    result = d.construct(prefix, inventory, tmp_path, proposal_limit=45)
    ledger = tmp_path / "continuation.jsonl"
    rows = ledger.read_text().splitlines()
    value = json.loads(rows[0])
    value["graph_hash"] = "changed"
    rows[0] = json.dumps(value)
    ledger.write_text("\n".join(rows) + "\n")
    with pytest.raises(ValueError, match="continuation_ledger"):
        d.construct(prefix, inventory, tmp_path, proposal_limit=46)
    result = d.construct(prefix, inventory, tmp_path / "timed", seconds=0)
    assert result["terminal_reason"] == "construction_time_exhausted"
    with patch.object(d, "TARGET", len(prefix["graph_manifest"])):
        result = d.construct(prefix, inventory, tmp_path / "complete")
    assert result["complete"] and result["terminal_reason"] is None


def test_prefix_and_late_exposure_refused(tmp_path):
    """SCENARIO-VERIFY-V773-PROSPECTIVE: accepted objects cannot be replaced or exposed."""
    prefix = private_prefix()
    altered = deepcopy(prefix)
    altered["proposal_rows"][0]["graph_hash"] = "changed"
    with pytest.raises(ValueError, match="original_ledger"):
        d.construct(altered, dict(old_graphs=[], old_seeds=[]), tmp_path)
    if prefix["graph_manifest"]:
        exposed = dict(old_graphs=[prefix["graph_manifest"][0]], old_seeds=[])
        with pytest.raises(ValueError, match="changed_prefix_exposure"):
            d.construct(prefix, exposed, tmp_path / "late")
    with pytest.raises(ValueError, match="incomplete_history"):
        d.construct(
            prefix,
            dict(old_graphs=[d.g.source.candidate(1)], old_seeds=[]),
            tmp_path / "history",
            seconds=0,
        )


def test_missing_and_changed_input_bytes(tmp_path):
    """SCENARIO-REPORT-9055-QUALIFICATION: missing bytes and changed authority stay distinct."""
    missing = d.authenticate(tmp_path, tmp_path / "raw")
    assert missing["failures"][0]["observed"] == "missing_file"
    target = tmp_path / d.history.ROSTER
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("{}")
    changed = d.authenticate(tmp_path, tmp_path / "changed")
    assert any(r["observed"] == d.reference(target)["sha256"] for r in changed["failures"])


def test_historical_prefix_recount():
    """REQ-REPORT-9055-CONTINUATION: independently pinned production ledger is unchanged."""
    original = json.loads((d.ROOT / d.history.ROSTER).read_bytes())
    prefix = json.loads(
        Path(
            next(
                r["path"]
                for r in original["raw_shard_hashes"]
                if r["path"].endswith("/construction.json")
            )
        ).read_bytes()
    )
    assert d.recount(prefix) == dict(
        accepted=414,
        rejection_counts={"unmatched_cardinality": 19586},
        block_yields=[112, 102, 107, 93],
    )


def test_authenticated_original_bytes_and_source_inventory(tmp_path):
    """REQ-REPORT-9055-CONTINUATION: production pins bind original objects and frozen fits."""
    evidence = d.authenticate(d.ROOT, tmp_path / "authenticated")
    assert not evidence["failures"]
    assert len(evidence["prefix"]["graph_manifest"]) == 414
    assert evidence["original_manifest"]["state_hashes"] == evidence["state_hashes"]
    with patch.object(
        d,
        "reference",
        side_effect=lambda path: (
            dict(path=str(path), sha256="changed")
            if path.name == "construction.json"
            else d.old.reference(path)
        ),
    ):
        failed = d.authenticate(d.ROOT, tmp_path / "changed")
    assert any(r["check"] == "original_primitive_bytes" for r in failed["failures"])


def test_later_source_objects_and_response_exposure(tmp_path):
    """SCENARIO-VERIFY-V773-PROSPECTIVE: later exposed topology extends the overlap authority."""
    prefix = private_prefix()
    graph = d.g.source.candidate(2)
    item = dict(graph, seed=2, graph_hash=d.g.source.graph_hash(graph))
    path = tmp_path / "results/experiment_9050_private.json"
    atomic_json(
        path,
        dict(
            graph_manifest=[item],
            response_rows=[
                dict(unit_id=prefix["graph_manifest"][0]["unit_id"], response_sha256="private")
            ],
        ),
    )
    evidence = dict(
        prefix=prefix, original_inventory=dict(old_graphs=[], old_seeds=[]), references=[]
    )
    later = d.exposure(tmp_path, evidence, tmp_path / "raw")
    assert later["exposed"] == [prefix["graph_manifest"][0]["unit_id"]]
    assert later["inventory"]["additional_seeds"] == [2]
    assert len(later["inventory"]["additional_graphs"]) == 1
    sidecar = tmp_path / "results/sidecar.json"
    atomic_json(sidecar, dict(graph_manifest=[item]))
    ref = dict(path=str(d.ROOT / "results/sidecar.json"), sha256=d.reference(sidecar)["sha256"])
    atomic_json(path, dict(graph_manifest_reference=ref))
    assert not d.exposure(tmp_path, evidence, tmp_path / "sidecar")["exposed"]
    ref["sha256"] = "changed"
    atomic_json(path, dict(graph_manifest_reference=ref))
    with pytest.raises(ValueError, match="later_source_bytes"):
        d.exposure(tmp_path, evidence, tmp_path / "bad")


def test_original_prefix_mutation_collision_and_heartbeat(tmp_path):
    """REQ-REPORT-9055-CONTINUATION: metadata cannot hide overlap or suppress progress."""
    prefix = private_prefix()
    changed = deepcopy(prefix)
    changed["graph_manifest"][0]["role"] = "changed"
    with pytest.raises(ValueError, match="original_prefix"):
        d.construct(changed, dict(old_graphs=[], old_seeds=[]), tmp_path / "changed")
    seed = 2**52 + d.SEED * 10000000
    graph = d.g.source.candidate(seed)
    topology = d.g.nx_graph(graph)
    row, accepted = d.proposal(0, 0, {d.g.topology_key(topology): [seed]}, set())
    assert row["reason"] == "isomorphic_relabeling" and accepted is None
    import itertools

    ticks = itertools.count(0, 31)
    with patch.object(d.time, "monotonic", side_effect=lambda: next(ticks)):
        result = d.construct(
            prefix,
            dict(old_graphs=[d.g.source.candidate(3)], old_seeds=[]),
            tmp_path / "heartbeat",
            seconds=100000,
            proposal_limit=45,
        )
    assert result["proposal_count"] == 45


def private_work(tmp_path):
    """No fixture graph can become a current scientific measurement."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    with patch.object(r.prior, "qualify", return_value=[dict(name="private", passed=True)]):
        return r.freeze(tmp_path / "missing", tmp_path / "raw")


def test_freeze_positive_late_exposure_and_changed_authority(tmp_path):
    """SCENARIO-REPORT-9055-QUALIFICATION: real source gates precede any construction."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    prefix = private_prefix()
    evidence = dict(
        checks=[],
        failures=[],
        references=[],
        state_hashes={},
        prefix=prefix,
        original_manifest=dict(state_hashes={}),
        original_inventory=dict(old_graphs=[], old_seeds=[]),
    )
    with (
        patch.object(d, "authenticate", return_value=evidence),
        patch.object(r.prior, "qualify", return_value=[dict(passed=True)]),
    ):
        with patch.object(
            d,
            "exposure",
            return_value=dict(
                inventory=dict(old_graphs=[], old_seeds=[]), rows=[], exposed=["late"]
            ),
        ):
            work = r.freeze(tmp_path, tmp_path / "late")
            assert work["evidence"]["failures"][-1]["check"] == "late_exposure"
        evidence["checks"].clear()
        with patch.object(d, "exposure", side_effect=ValueError("later_source_bytes")):
            work = r.freeze(tmp_path, tmp_path / "changed")
            assert work["evidence"]["failures"][-1]["observed"] == "later_source_bytes"
        evidence["checks"].clear()
        with (
            patch.object(
                d,
                "exposure",
                return_value=dict(inventory=dict(old_graphs=[], old_seeds=[]), rows=[], exposed=[]),
            ),
            patch.object(d, "construct", return_value=prefix),
        ):
            work = r.freeze(tmp_path, tmp_path / "ready")
            assert work["result"] == prefix


def test_build_readiness_counts_and_no_model(tmp_path):
    """REQ-REPORT-9055-CONTINUATION: qualification and fixed cohort both gate readiness."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    work = private_work(tmp_path)
    validation = dict(receipts=[], coverage={}, scope=dict(code_test_hashes=[]))
    blocked = r.build(work, validation, True)
    assert blocked["verdict_class"] == "blocked"
    assert blocked["fresh_source_ready_score"] == 0
    assert sum(blocked["model_invocation_counts"].values()) == 0
    assert blocked["MODEL_SPECS"] == blocked["model_specs"] == []
    assert r.build(work, validation, False)["verdict_class"] == "disqualified"
    work["evidence"]["checks"] = []
    work["result"]["graph_manifest"] = [dict(unit_id=f"private-{i}") for i in range(1024)]
    work["result"]["complete"] = True
    ready = r.build(work, validation, True)
    assert ready["fresh_source_ready_score"] == 1
    assert ready["honest_verdict"] == "complete_null_prospective_roster"
    assert ready["expected_seal_fields"]["expected_counts"] == dict(
        detector_test=640, online=3840, retention=46080
    )


def foreign_cli(tmp_path, *args):
    """Real child imports must work without checkout-relative environment paths."""
    import os
    import subprocess
    import sys

    command = ["env", "-u", "PYTHONPATH", "-C", str(tmp_path), sys.executable]
    if os.environ.get("COVERAGE_RCFILE"):
        command += ["-m", "coverage", "run", "--rcfile=" + os.environ["COVERAGE_RCFILE"]]
    command += [str(d.ROOT / d.CLI), *map(str, args)]
    return subprocess.run(command, capture_output=True, text=True, timeout=60)


def test_real_child_interruption_and_resume(tmp_path):
    """SCENARIO-REPORT-9055-CAP-RESUME: exit73 retains exact proposals and prefix."""
    prefix = private_prefix()
    request = tmp_path / "request.json"
    raw = tmp_path / "resumed"
    atomic_json(
        request,
        dict(
            prefix=prefix,
            inventory=dict(old_graphs=[], old_seeds=[]),
            raw=str(raw),
            proposal_limit=60,
        ),
    )
    killed = foreign_cli(tmp_path, "--construct-worker", request, "--interrupt-after", "50")
    assert killed.returncode == 73, killed.stdout + killed.stderr
    assert json.loads((raw / "checkpoint.json").read_bytes())["proposal_count"] == 50
    resumed = foreign_cli(tmp_path, "--construct-worker", request)
    assert resumed.returncode == 0, resumed.stdout + resumed.stderr
    baseline = d.construct(
        prefix, dict(old_graphs=[], old_seeds=[]), tmp_path / "baseline", proposal_limit=60
    )
    result = json.loads((raw / "construction.json").read_bytes())
    assert result["proposal_rows"] == baseline["proposal_rows"]
    assert result["graph_manifest"] == baseline["graph_manifest"]


def test_foreign_cli_missing_replay_and_rehashed_tamper(tmp_path):
    """SCENARIO-REPORT-9055-QUALIFICATION: private CLI blocks and refuses rehashed changes."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    output = tmp_path / (d.NAME + ".json")
    process = foreign_cli(tmp_path, "--fixture-e2e", output, "--input-root", tmp_path / "missing")
    assert process.returncode == 0, process.stdout + process.stderr
    value = json.loads(output.read_bytes())
    assert value["honest_verdict"].startswith("complete_blocked_")
    assert value["fresh_source_ready_score"] == 0
    assert foreign_cli(tmp_path, "--cold-replay", output).returncode == 0
    value["proposal_count"] = 1024
    value["reproducibility_checksum"] = r.p.checksum(value)
    atomic_json(output, value)
    assert foreign_cli(tmp_path, "--cold-replay", output).returncode == 1
    with pytest.raises(SystemExit):
        r.main(["--fixture-e2e", str(d.ROOT / "results/private.json")])


def candidate_from_work(work, validation):
    """Bind private primitive rows so tampering tests exercise the actual reader."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    raw = Path(work["raw"])
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] = [d.reference(raw / "work.json"), d.reference(raw / "validation.json")]
    value = r.build(work, validation, work["accepted"])
    candidate = raw / "candidate.json"
    atomic_json(candidate, value)
    return candidate


def test_semantic_replay_and_current_validation_refusals(tmp_path):
    """SCENARIO-REPORT-9055-QUALIFICATION: rehashing cannot change roster or validation."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    work = private_work(tmp_path)
    work["evidence"]["prefix"] = private_prefix()
    work["result"] = work["evidence"]["prefix"]
    work["accepted"] = False
    validation = dict(receipts=[], coverage={}, scope=dict(code_test_hashes=[], frozen_ns=0))
    candidate = candidate_from_work(work, validation)
    assert r.replay(candidate)["passed"]
    value = json.loads(candidate.read_bytes())
    value["raw_shard_hashes"][0]["sha256"] = "changed"
    atomic_json(candidate, value)
    assert r.replay(candidate)["reason"] == "primitive_bytes"
    candidate = candidate_from_work(work, validation)
    with patch.object(d, "construct", return_value=dict(work["result"], complete=True)):
        assert r.replay(candidate)["reason"] == "proposal_replay_drift"
    work["accepted"] = True
    candidate = candidate_from_work(work, validation)
    with patch.object(r.checks, "receipts_pass", return_value=False):
        assert r.replay(candidate)["reason"] == "validation_drift"
    with (
        patch.object(r.checks, "receipts_pass", return_value=True),
        patch.object(r, "verify_measured", return_value=True),
    ):
        assert r.replay(candidate)["passed"]


def test_owned_failure_preserves_primary_and_success_publishes(tmp_path, monkeypatch):
    """SCENARIO-REPORT-9055-QUALIFICATION: failed current checks cannot publish readiness."""
    from carnot.reporting import prospective_roster_runner_9055 as r

    work = private_work(tmp_path)
    frozen = dict(code_test_hashes=[], frozen_ns=0)
    monkeypatch.setattr(d, "ROOT", tmp_path)
    monkeypatch.setattr(r, "freeze", lambda *args: deepcopy(work))
    monkeypatch.setattr(r, "plan", lambda private: ([object()], frozen))
    monkeypatch.setattr(r.checks, "run_check", lambda *args: dict(passed=True))
    monkeypatch.setattr(r, "measured", lambda *args: dict(rows=[], percent=100))
    monkeypatch.setattr(r.checks, "receipts_pass", lambda *args: False)
    before = tmp_path / "results" / (d.NAME + ".json")
    before.parent.mkdir(parents=True, exist_ok=True)
    before.write_text("preserved")
    monkeypatch.delenv("COVERAGE_RCFILE", raising=False)
    assert r.main([]) == 1
    assert before.read_text() == "preserved"
    monkeypatch.setenv("COVERAGE_RCFILE", "preserved-coverage-config")
    assert r.main([]) == 1
    monkeypatch.setattr(r.checks, "receipts_pass", lambda *args: True)
    monkeypatch.setattr(r, "verify_measured", lambda *args: True)
    monkeypatch.setattr(r.terminal, "scanners", lambda *args: dict(passed=True))
    before.unlink()
    assert r.main([]) == 0
    assert json.loads(before.read_bytes())["required_checks_passed"]


def test_resume_authority_binding_and_added_history(tmp_path):
    """SCENARIO-REPORT-9055-CAP-RESUME: changed authority cannot resume an old scan."""
    prefix = private_prefix()
    inventory = dict(
        old_graphs=[],
        old_seeds=[],
        additional_graphs=[d.g.source.candidate(4)],
        additional_seeds=[5],
    )
    result = d.construct(prefix, inventory, tmp_path, proposal_limit=50)
    assert result["proposal_rows"][:40] == prefix["proposal_rows"]
    changed = dict(inventory, additional_seeds=[6])
    with pytest.raises(ValueError, match="checkpoint_authority"):
        d.construct(prefix, changed, tmp_path, proposal_limit=60)
    later = dict(old_graphs=[], old_seeds=[], additional_graphs=[prefix["graph_manifest"][0]])
    with pytest.raises(ValueError, match="changed_prefix_exposure"):
        d.construct(prefix, later, tmp_path / "late")
    rejected = next(r for r in prefix["proposal_rows"] if r["reason"])
    later = dict(old_graphs=[], old_seeds=[], additional_seeds=[rejected["seed"]])
    continued = d.construct(prefix, later, tmp_path / "rejected", proposal_limit=50)
    assert continued["proposal_rows"][:40] == prefix["proposal_rows"]


def test_resume_accepted_rows_and_long_loop_progress(tmp_path):
    """SCENARIO-REPORT-9055-CAP-RESUME: durable accepted rows and pending counts survive replay."""
    import itertools

    prefix = private_prefix()
    inventory = dict(old_graphs=[], old_seeds=[], additional_seeds=[4])
    first = d.construct(prefix, inventory, tmp_path, proposal_limit=200)
    assert len(first["graph_manifest"]) > len(prefix["graph_manifest"])
    ticks = itertools.count(0, 31)
    with patch.object(d.time, "monotonic", side_effect=lambda: next(ticks)):
        resumed = d.construct(prefix, inventory, tmp_path, proposal_limit=201, seconds=100000)
    assert resumed["proposal_rows"][:200] == first["proposal_rows"]


def test_unpublished_collection_evidence_blocks(tmp_path):
    """SCENARIO-VERIFY-V773-PROSPECTIVE: unowned collection files cannot hide behind absent primaries."""
    receipt = tmp_path / "results/raw/experiment_9045_private/inflight.json"
    atomic_json(receipt, dict(unit_id="v772-graph-0000-k2", status="started"))
    evidence = d.authenticate(tmp_path, tmp_path / "raw")
    failed = next(r for r in evidence["failures"] if r["check"] == "no_collection_since_prefix")
    assert failed["expected"] == []
    assert failed["observed"][0]["sha256"] == d.reference(receipt)["sha256"]


def test_source_inventory_skips_nonnumeric_experiment_names(tmp_path):
    """REQ-REPORT-9055-CONTINUATION: historical letter-suffixed names are outside this ID range."""
    atomic_json(tmp_path / "results/experiment_224c_private.json", dict(private=True))
    evidence = dict(
        prefix=private_prefix(), original_inventory=dict(old_graphs=[], old_seeds=[]), references=[]
    )
    assert d.exposure(tmp_path, evidence, tmp_path / "raw")["rows"] == []


def test_large_checkpoint_cursor_is_metadata_for_unchanged_scanner(tmp_path):
    """SCENARIO-REPORT-9055-QUALIFICATION: a cursor is administrative metadata, not a second measured metric."""
    from carnot.reporting import prospective_roster_runner_9055 as r
    from scripts import adversarial_verify

    work = private_work(tmp_path)
    work["result"]["proposal_count"] = 47987
    value = r.build(work, dict(receipts=[], coverage={}, scope=dict(code_test_hashes=[])), False)
    assert value["proposal_count"] == 47987
    assert value["continuation_counter"] == dict(initial=20000, last_completed=47986, next=47987)
    flags = []
    adversarial_verify.check_tautology(value, flags)
    assert not flags
