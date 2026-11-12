"""REQ-REPORT-9055-CONTINUATION: qualify bounded source construction before publication."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import prospective_roster_9055 as d
from carnot.reporting import fresh_seal_methods_9042 as old
from carnot.reporting import fresh_seal_custody_9042 as custody
from carnot.reporting import v773_branch_evidence as history
from carnot.reporting import fresh_seal_runner_9042 as prior
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting import retained_shard_runner_9030 as terminal
from carnot.reporting import invocation_provenance_9000 as p
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
DIRECT = [
    "tests/python/test_fresh_seal_methods_9042.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
]
POLICY = dict(
    custody.POLICY,
    test="Exp9063 final_evaluation only after byte-bound Exp9061 and Exp9062 prediction seals; no early truth reads",
    calibration="Exp9061 calibration only; private Exp9060 admission returns readiness only",
)
SEALS = dict(
    required_fields=custody.SEAL_FIELDS,
    prediction_experiments=[9061, 9062],
    authorized_test_reducer=9063,
    expected_counts=dict(detector_test=640, online=3840, retention=46080),
    detector_arms=5,
    test_graphs=128,
    online_streams=24,
    queries_per_stream=32,
    retention_checkpoints=[0, 16, 32],
)


def freeze(root: Path, raw: Path, construction_raw: Path | None = None) -> Json:
    """Only authenticated source inputs may enter the fixed construction budget."""
    started = time.monotonic()
    d.progress("preconditions_before")
    environment = p.environment(raw)
    evidence = d.authenticate(root, raw)
    evidence["checks"].extend(environment["checks"])
    d.progress(
        "preconditions_after",
        len(evidence["checks"]),
        sum(not row["passed"] for row in evidence["checks"]),
    )
    result: Json = dict(
        graph_manifest=[],
        proposal_rows=[],
        proposal_count=0,
        complete=False,
        construction_duration_s=0,
        terminal_reason="preconditions_blocked",
    )
    overlap: list[Json] = []
    inventory: Json = dict(old_graphs=[], old_seeds=[])
    if all(r["passed"] for r in evidence["checks"]):
        result = deepcopy(evidence["prefix"])
        try:
            later = d.exposure(root, evidence, raw)
            inventory, overlap = later["inventory"], later["rows"]
            evidence["checks"].append(
                old.gate(raw, "accepted_prefix_exposure", [], later["exposed"], "late_exposure")
            )
            if not later["exposed"]:
                d.progress("construction_before", 414, 610)
                result = d.construct(
                    evidence["prefix"], inventory, construction_raw or raw / "roster"
                )
        except ValueError as error:
            evidence["checks"].append(
                old.gate(raw, "construction_authority", "unchanged", str(error), str(error))
            )
    evidence["failures"] = [r for r in evidence["checks"] if not r["passed"]]
    atomic_json(raw / "historical_inventory.json", inventory)
    manifest = deepcopy(evidence.get("original_manifest", dict(state_hashes={})))
    manifest.update(
        graph_manifest=result["graph_manifest"],
        complete=result["complete"],
        intended_count=1024,
        fixture_only=False,
        consumers=[
            "exp9058-fresh-corpus-a",
            "exp9059-fresh-corpus-b",
            "exp9060-sealed-corpus-admission",
            "exp9061-sealed-calibrated-decisions",
            "exp9062-sealed-online-streams",
            "exp9063-sealed-learning-audit",
        ],
        label_access_policy=POLICY,
        expected_seal_fields=SEALS,
        historical_inventory_reference=p.reference(raw / "historical_inventory.json"),
    )
    atomic_json(raw / "fresh_role_manifest.json", manifest)
    d.progress("custody_controls_before", 0, 1)
    controls = prior.qualify(raw / "custody_controls")
    d.progress("custody_controls_after", len(controls), 0)
    return dict(
        raw=str(raw),
        input_root=str(root),
        fixture_only=False,
        evidence=evidence,
        environment=environment,
        result=result,
        controls=controls,
        manifest_reference=p.reference(raw / "fresh_role_manifest.json"),
        overlap_rows=overlap,
        raw_references=[],
        duration_s=time.monotonic() - started,
        phase_spans=[
            dict(phase="authentication_construction_custody", duration_s=time.monotonic() - started)
        ],
    )


def build(work: Json, validation: Json, accepted: bool) -> Json:
    """A completed administrative roster earns no generalization or model-call credit."""
    value = old.build(work, validation, accepted)
    result = work["result"]
    value.update(
        experiment_id=9055,
        task_id=d.TASK,
        milestone="2026.11.773",
        label_access_policy=POLICY,
        expected_seal_fields=SEALS,
        original_prefix_sha256=canonical_hash(
            work["evidence"].get("prefix", {}).get("graph_manifest", [])
        ),
        original_prefix_reference=next(
            (
                r
                for r in work["evidence"]["references"]
                if r.get("source_path", "").endswith("/construction.json")
            ),
            None,
        ),
        continuation_counter=dict(
            initial=20000,
            last_completed=result["proposal_count"] - 1 if result["proposal_count"] else None,
            next=result["proposal_count"],
        ),
        proposal_count=result["proposal_count"],
        rejection_counts=d.recount(result)["rejection_counts"],
        original_block_yields=d.recount(work["evidence"].get("prefix", dict(proposal_rows=[])))[
            "block_yields"
        ],
        construction_duration_s=result["construction_duration_s"],
        verifier_is_oracle=True,
        claim_scope="Prospective source identities and custody only. Executable graph truth is an oracle; no independent generalization or detector benefit is measured.",
        exposure_scope="Original exposed graphs excluded; prospective roster has no model responses or predictions.",
        methodology_note="Continue seed 2**52 + 772042*10000000 + counter from20000. Preserve prefix and original predicate, roles and fits. Stop at1024 accepted,100000 total proposals or300 construction seconds. Replay every prior proposal and retain rejections. Private custody controls measure mechanics only. No model is loaded or fitted.",
    )
    value["honest_verdict"] = value["honest_verdict"].replace(
        "fresh_seal_methods", "prospective_roster"
    )
    if value["fresh_source_ready_score"]:
        value["honest_verdict"] = "complete_null_prospective_roster"
    value["cited_upstream_artifacts"].extend(
        [
            dict(
                experiment_id=n,
                sha256=history.PINS[path],
                fields_imported=["preserved_source_and_verdict"],
            )
            for n, path in [(9042, history.ROSTER), (9053, history.CAPSTONE)]
        ]
    )
    value["field_principles"] = dict.fromkeys(
        [*value],
        "Immutable source bytes and exact current checks qualify custody only; incomplete external support remains blocked.",
    )
    value["reproducibility_checksum"] = p.checksum(value)
    return dict(value)


def replay(path: Path) -> Json:
    """Recompute source selection and summaries; rehashed headline tampering still fails."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if p.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("primitive_bytes")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        validation = json.loads((Path(work["raw"]) / "validation.json").read_bytes())
        if work["accepted"] and not (
            checks.receipts_pass(
                dict(
                    validation_scope_manifest=validation["scope"],
                    validation=validation,
                    frozen_ns=validation["scope"]["frozen_ns"],
                )
            )
            and verify_measured(validation["coverage"], d.OWNED)
        ):
            raise ValueError("validation_drift")
        work["raw_references"] = value["raw_shard_hashes"]
        if build(work, validation, work["accepted"]) != value:
            raise ValueError("aggregate_drift")
        if work["result"]["proposal_count"]:
            inventory = json.loads((Path(work["raw"]) / "historical_inventory.json").read_bytes())
            rebuilt = d.construct(
                work["evidence"]["prefix"],
                inventory,
                Path(tempfile.mkdtemp(prefix="carnot-9055-replay-")),
                proposal_limit=work["result"]["proposal_count"],
            )
            for field in ["graph_manifest", "proposal_rows", "complete", "proposal_count"]:
                if rebuilt[field] != work["result"][field]:
                    raise ValueError("proposal_replay_drift")
        return dict(passed=True, reason="primitive_replay_passed")
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        return dict(passed=False, reason=str(error))


def plan(private: Path) -> tuple[list[Any], Json]:
    """The original scoped harness freezes real commands and measures only new paths."""
    with patch.multiple(prior, d=d, DIRECT=DIRECT):
        commands, scope = prior.plan(private)
    rc = private / "coverage.ini"
    rc.write_text(rc.read_text().replace("[run]\n", "[run]\npatch=_exit\n"))
    scope["code_test_hashes"].extend(
        p.reference(d.ROOT / name)
        for name in [
            *old.OWNED,
            "python/carnot/reporting/graph_source_8931.py",
            "python/carnot/reporting/relational_methods_8959.py",
            "python/carnot/reporting/v773_branch_evidence.py",
        ]
    )
    return commands, scope


def main(argv: list[str] | None = None) -> int:
    """Publish only after current checks pass, while preserving all rejected candidates."""
    os.environ.update(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--construct-worker", type=Path)
    parser.add_argument("--interrupt-after", type=int)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    if args.construct_worker:
        request = json.loads(args.construct_worker.read_bytes())
        d.construct(
            request["prefix"],
            request["inventory"],
            Path(request["raw"]),
            proposal_limit=request["proposal_limit"],
            interrupt_after=args.interrupt_after,
        )
        return 0
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if args.fixture_e2e and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    raw = output.absolute().parent / "raw" / d.NAME / "invocations" / str(time.time_ns())
    private = Path(tempfile.mkdtemp(prefix="carnot-9055-validation-"))
    commands, scope = plan(private)
    atomic_json(raw / "validation_scope_manifest.json", scope)
    started = time.monotonic()
    construction_raw = raw.parent.parent / "construction"
    work = freeze(args.input_root, raw, construction_raw)
    work["fixture_only"] = args.fixture_e2e is not None
    validation: Json = dict(receipts=[], coverage={}, scope=scope)
    accepted = False
    if not args.fixture_e2e:
        before = os.environ.get("COVERAGE_RCFILE")
        os.environ["COVERAGE_RCFILE"] = str(private / "coverage.ini")
        try:
            for index, command in enumerate(commands):
                d.progress("validation", index, len(commands) - index)
                validation["receipts"].append(checks.run_check(command, raw / "validation", scope))
        finally:
            if before is None:
                os.environ.pop("COVERAGE_RCFILE", None)
            else:
                os.environ["COVERAGE_RCFILE"] = before
        validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "measured")
        validation["coverage"].pop("rows", None)
        accepted = checks.receipts_pass(
            dict(
                validation_scope_manifest=scope, validation=validation, frozen_ns=scope["frozen_ns"]
            )
        ) and verify_measured(validation["coverage"], d.OWNED)
    work.update(accepted=accepted, duration_s=time.monotonic() - started)
    work["phase_spans"].append(
        dict(
            phase="owned_validation",
            duration_s=work["duration_s"] - sum(s["duration_s"] for s in work["phase_spans"]),
        )
    )
    atomic_json(raw / "validation.json", validation)
    atomic_json(raw / "work.json", work)
    work["raw_references"] = [
        p.reference(raw / "work.json"),
        p.reference(raw / "validation.json"),
        work["manifest_reference"],
        p.reference(raw / "historical_inventory.json"),
    ]
    work["raw_references"].extend(
        p.retain(path, raw) for path in sorted(construction_raw.glob("*")) if path.is_file()
    )
    value = build(work, validation, accepted)
    atomic_json(raw / "candidate.json", value)
    if not args.fixture_e2e and not value["required_checks_passed"]:
        d.progress("owned_checks_failed_primary_preserved", 0, 1)
        return 1
    if args.fixture_e2e:
        publish_primary(output, value, replay)
    else:
        with patch.object(terminal, "d", d):
            publish_primary(
                output, value, lambda path: terminal.scanners(path, raw / "terminal", scope)
            )
    d.progress("terminal_published", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                path=str(output),
                honest_verdict=value["honest_verdict"],
                fresh_source_ready_score=value["fresh_source_ready_score"],
                proposal_count=value["proposal_count"],
                construction_duration_s=value["construction_duration_s"],
            )
        ),
        flush=True,
    )
    return 0
