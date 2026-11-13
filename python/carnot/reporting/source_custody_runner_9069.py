"""REQ-REPORT-9069-V774: private metadata audit with qualified atomic publication."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import source_custody_9069 as d
from carnot.reporting import fresh_corpus_runner_9058 as prior
from carnot.reporting import retained_shard_runner_9030 as terminal
from carnot.reporting import scoped_provenance_9014 as checks
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from carnot.reporting.v756_branch_qualification import measured, verify_measured

Json = dict[str, Any]
HEALTH = Path("/var/tmp/carnot-9069-global-health/receipt.json")
DIRECT = [
    d.a.TEST,
    "tests/python/test_fresh_seal_methods_9042.py",
    "tests/python/test_current_work_receipt.py",
    "tests/python/test_primary_publication_7928.py",
]


def foreign_argv(*args: str) -> tuple[str, ...]:
    """The shipped CLI must import itself outside the checkout without PYTHONPATH."""
    with patch.object(checks, "CLI", d.CLI):
        return checks.foreign_argv(*args)


def plan(private: Path) -> tuple[list[Any], Json]:
    """Reuse frozen consumer, real E2E and zero-exclusion coverage commands unchanged."""
    with patch.multiple(prior, d=d, DIRECT=DIRECT):
        return prior.plan(private)


def mutation_controls(raw: Path) -> list[Json]:
    """Synthetic negative controls exercise rejection without opening production TEST labels."""
    manifest: Json = dict(
        graph_manifest=[dict(unit_id="private-test", role="test")],
        state_hashes={},
        first_response_budget=1,
        generator_settings=d.source.SETTINGS,
    )
    primary = dict(state_hashes={}, model_tokenizer_hashes=dict(d.TOKENIZER))
    changed = deepcopy(manifest)
    changed["graph_manifest"][0]["role"] = "calibration"
    roles = not all(
        r["passed"] for r in d.contract_checks(changed, primary, raw, expected_roles={"test": 1})
    )
    primary["model_tokenizer_hashes"]["tokenizer"] = "changed"
    tokenizer = not all(
        r["passed"] for r in d.contract_checks(manifest, primary, raw, expected_roles={"test": 1})
    )
    orphan = raw / "results/raw/experiment_9059_private/invocations/1/private-b/wire_response.json"
    atomic_json(orphan, dict(unit_id="private-b", content="private"))
    source = raw / d.ROSTER
    atomic_json(source, dict(reproducibility_checksum=d.canonical_hash("mismatched_source")))
    observed = dict(
        changed_roles=roles,
        a_test_disclosure=bool(
            d.disclosures(dict(unit_id="private-test", answer="private"), {"private-test"})
        ),
        b_orphan_response=bool(d.attempt_inventory(raw, {"private-b"})["attempts"]),
        changed_tokenizer=tokenizer,
        rehashed_mismatched_source=d.p.reference(source)["sha256"] != d.PINS[d.ROSTER],
    )
    return [
        dict(name=k, passed=v, fixture_only=True, claim_scope="private_custody_controls")
        for k, v in observed.items()
    ]


def freeze(root: Path, raw: Path, fixture: bool) -> Json:
    """Check private storage and current dependencies before authenticated metadata reduction."""
    start = time.monotonic()
    d.progress("preconditions_before")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    environment = d.p.environment(raw)
    environment["checks"].append(
        d.p.gate(raw, "private_storage_mode", 0o700, raw.stat().st_mode & 0o777, "private_storage")
    )
    evidence = d.audit(root, raw)
    health = json.loads(HEALTH.read_bytes()) if HEALTH.is_file() else {}
    d.progress("preconditions_and_audit_after", len(evidence["rows"]), 0)
    controls = mutation_controls(raw / "controls")
    evidence["checks"].append(
        d.p.gate(
            raw / "controls",
            "all_mutation_controls_passed",
            True,
            all(r["passed"] for r in controls),
            "custody_mutations",
        )
    )
    return dict(
        raw=str(raw),
        input_root=str(root),
        fixture_only=fixture,
        environment=environment,
        evidence=evidence,
        controls=controls,
        repository_health=dict(
            passed=health.get("passed", False),
            current_suite=health,
            current_suite_reference=d.p.reference(HEALTH),
            full_suite_repeated=False,
            affects_owned_readiness=False,
        ),
        duration_s=time.monotonic() - start,
        phase_spans=[dict(phase="authentication_and_custody", duration_s=time.monotonic() - start)],
        code_references=[d.p.reference(d.ROOT / name) for name in [*d.OWNED, d.TEST]],
    )


def replay(path: Path) -> Json:
    """Rebuild the terminal from byte-bound primitives; rehashed readiness is insufficient."""
    try:
        value = json.loads(path.read_bytes())
        for ref in (
            value["raw_shard_hashes"]
            + value["source_artifact_hashes"]
            + value["code_config_hashes"]
        ):
            if d.p.reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
                raise ValueError("primitive_bytes")
        work = json.loads(Path(value["raw_shard_hashes"][0]["path"]).read_bytes())
        source_checks = [
            d.p.gate(
                Path(work["input_root"]) / name,
                "sha256",
                pin,
                d.p.reference(Path(work["input_root"]) / name)["sha256"] or "missing_file",
                "declared_source_bytes",
            )
            for name, pin in d.PINS.items()
        ]
        if work["evidence"]["checks"][: len(source_checks)] != source_checks or any(
            ref["sha256"] != d.PINS[str(Path(ref["source_path"]).relative_to(work["input_root"]))]
            for ref in value["source_artifact_hashes"]
            if ref.get("source_path") in {str(Path(work["input_root"]) / name) for name in d.PINS}
        ):
            raise ValueError("independent_source_binding")
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
        inventory = work["evidence"]["b_attempt_inventory"]
        if inventory:
            b_ids = {r["unit_id"] for r in work["evidence"]["rows"] if r["arm"] == "B"}
            if d.attempt_inventory(Path(work["input_root"]), b_ids) != inventory:
                raise ValueError("invocation_inventory_drift")
        work["raw_references"] = value["raw_shard_hashes"]
        if d.build(work, validation, work["accepted"]) != value:
            raise ValueError("aggregate_drift")
        return dict(passed=True, reason="primitive_replay_passed")
    except (OSError, KeyError, ValueError, TypeError, IndexError) as error:
        return dict(passed=False, reason=str(error))


def main(argv: list[str] | None = None) -> int:
    """Retain failed candidates privately and publish a stable checked terminal once."""
    os.environ.update(PYTHONUNBUFFERED="1", JAX_PLATFORMS="cpu")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261112"], default="20261112")
    parser.add_argument("--input-root", type=Path, default=d.ROOT)
    parser.add_argument("--fixture-e2e", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    if args.cold_replay:
        report = replay(args.cold_replay)
        print(json.dumps(report), flush=True)
        return int(not report["passed"])
    output = args.fixture_e2e or d.ROOT / "results" / (d.NAME + ".json")
    if args.fixture_e2e and output.resolve().is_relative_to(d.ROOT):
        parser.error("fixture writes require private scratch")
    private = Path(tempfile.mkdtemp(prefix="carnot-9069-", dir="/var/tmp"))
    raw = private / "evidence"
    commands, scope = plan(private)
    scope["frozen_ns"] = time.monotonic_ns()
    atomic_json(private / "validation_scope_manifest.json", scope)
    start = time.monotonic()
    work = freeze(args.input_root, raw, args.fixture_e2e is not None)
    validation: Json = dict(scope=scope, receipts=[], coverage={})
    accepted = False
    if not args.fixture_e2e:
        with patch.dict(os.environ, COVERAGE_RCFILE=str(private / "coverage.ini")):
            for command in commands:
                validation["receipts"].append(checks.run_check(command, raw / "validation", scope))
        validation["coverage"] = measured([private / ".coverage"], d.OWNED, raw / "coverage")
        validation["coverage"].pop("rows", None)
        accepted = checks.receipts_pass(
            dict(
                validation_scope_manifest=scope, validation=validation, frozen_ns=scope["frozen_ns"]
            )
        ) and verify_measured(validation["coverage"], d.OWNED)
    work.update(accepted=accepted, duration_s=time.monotonic() - start)
    work["phase_spans"].append(
        dict(
            phase="validation", duration_s=work["duration_s"] - work["phase_spans"][0]["duration_s"]
        )
    )
    atomic_json(raw / "work.json", work)
    atomic_json(raw / "validation.json", validation)
    work["raw_references"] = [
        d.p.reference(raw / "work.json"),
        d.p.reference(raw / "validation.json"),
    ]
    value = d.build(work, validation, accepted)
    atomic_json(raw / "candidate.json", value)
    if not args.fixture_e2e and not accepted:
        d.progress("owned_checks_failed_primary_preserved")
        return 1
    d.progress("publication_before")
    with patch.object(terminal, "d", d):
        receipt = publish_primary(
            output,
            value,
            replay
            if args.fixture_e2e
            else lambda path: terminal.scanners(path, raw / "terminal", scope),
        )
    atomic_json(raw / "publication.json", receipt)
    d.progress("publication_after", value["completed_count"], value["censored_count"])
    print(
        json.dumps(
            dict(
                path=str(output),
                honest_verdict=value["honest_verdict"],
                a_custody_ready_score=value["a_custody_ready_score"],
                b_collection_ready_score=value["b_collection_ready_score"],
            )
        ),
        flush=True,
    )
    return 0
