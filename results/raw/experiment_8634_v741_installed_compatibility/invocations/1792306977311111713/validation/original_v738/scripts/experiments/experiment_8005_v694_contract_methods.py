#!/usr/bin/env python3
"""REQ-REPORT-8005: qualify immutable authority without transferring science credit."""

import argparse
from functools import partial
import gzip
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT))

from carnot.reporting import v686_contract_methods as shared  # noqa: E402
from carnot.reporting import v686_contract_validation as validation  # noqa: E402
from carnot.reporting import v690_authority as authority  # noqa: E402
from carnot.reporting import v690_contract_methods as inherited  # noqa: E402
from carnot.reporting import v693_contract_methods as prior  # noqa: E402
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file  # noqa: E402
from carnot.reporting.primary_publication import publish_primary, reader_receipt  # noqa: E402
from carnot.reporting.roadmap_contract import parse_design  # noqa: E402

OWNED = "scripts/experiments/experiment_8005_v694_contract_methods.py"
TESTS = ["tests/python/test_experiment_8005_v694_contract_methods.py", validation.TESTS[1]]
MODEL_SPECS: list[str] = []
mutations = partial(inherited.mutations, milestone="2026.10.694", first_id=8005)


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """Full JSON must match its digest, including prompts outside the short table."""
    value = authority.assess(
        design, staged, active, snapshots, milestone="2026.10.694", first_id=8005
    )
    if value["canonical_tasks_sha256"]:
        _, machine = parse_design(design.read_text(), milestone="2026.10.694")
        observed = authority.lifecycle.tasks_digest(machine)
        if observed != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                shared.operand(
                    design,
                    "design_tasks_sha256",
                    value["canonical_tasks_sha256"],
                    observed,
                    "V694_authority",
                )
            )
            value.update(activated=False, observed_activation=False)
    return value


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Reuse the measured command set with coverage restricted to this new entrypoint."""
    value = validation.manifest(root, private)
    value["commands"] = [r for r in value["commands"] if not r["name"].startswith("e2e_016")]
    for row in value["commands"]:
        row["argv"] = [
            ("--include=" + str(root / OWNED))
            if a.startswith("--include=")
            else a.replace("7903_v686", "8005_v694").replace("20260930", "20261002")
            for a in row["argv"]
        ]
        if row["name"] in {"affected_pytest", "coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in validation.TESTS + TESTS] + TESTS
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + [OWNED]
            if row["name"] != "mypy_strict":
                row["argv"].append(TESTS[0])
        if row["name"] == "repository_full_suite":
            row["argv"] = [str(root / ".venv/bin/pytest"), "tests/python", "-q"]
    value.update(
        coverage_includes=[OWNED],
        affected_tests=TESTS,
        dependency_hashes=validation.dependency_hashes(root, paths=[OWNED, *TESTS]),
    )
    return value


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """Seal all summaries as well as the inherited primitive reduction operands."""
    return {**shared.primitive_rows(value), "artifact_sha256": canonical_hash(value)}


def cold_replay(path: Path, raw: Path) -> bool:
    """Recompute contract counts and authenticate durable checkpoints in a fresh process."""
    if not path.is_file() or not raw.is_file():
        return False
    value = json.loads(path.read_text())
    if value.get("experiment_id") != 8005:
        return False
    snapshots = [s for s in value["authority_snapshots"].values() if s["exists"]]
    refs = value["checkpoint_references"] + [
        dict(path=s["snapshot_path"], sha256=s["sha256"]) for s in snapshots
    ]
    return bool(
        primitive_rows(value) == json.loads(raw.read_text())
        and value["task_id"] == "exp8005-contract-methods"
        and value["rows"] == value["contract_rows"]
        and len(value["rows"]) == 13
        and all(r["absolute_metric"] == int(all(r["checks"].values())) for r in value["rows"])
        and value["sample_size_budget"] == shared.contract_budget(value["rows"], count=13)
        and value["contract_ready_score"]
        == shared.verdict(
            value["activation_confirmed"],
            value["source_custody_ready"],
            value["required_checks_passed"],
        )[2]
        and all(
            Path(r["path"]).is_file() and sha256_file(Path(r["path"])) == r["sha256"] for r in refs
        )
    )


def execute(args: Any, private: Path) -> dict[str, Any]:
    """Adapt qualified readers and publication helpers; all mutable checks use private scratch."""
    started = time.monotonic_ns()
    private.mkdir(parents=True, exist_ok=True)
    durable = args.raw.parent
    print("[exp8005] phase=inputs_before elapsed_s=0 completed_units=0 pending=freeze", flush=True)
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v694" / f"{name}.gz").read_bytes())
        )
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})
    assessment = assess(args.design, args.staged, args.active, durable / "authority_snapshots")
    _, tasks = parse_design((private / "design.md").read_text(), milestone="2026.10.694")
    freeze = prior.method_freeze(tasks)
    freeze["design_methods"] = (
        (private / "design.md").read_text().split("## Exact task contract", 1)[0]
    )
    frozen = manifest(ROOT, private)
    snapshots, history = [], []
    for name in (
        "openspec/change-proposals/research-roadmap-v693-preserved-20261002.md",
        "results/experiment_7992_v693_contract_methods.json",
        "results/experiment_8004_v693_capstone.json",
    ):
        source = ROOT / name
        ref = authority.lifecycle._snapshot(
            source, source.read_bytes(), durable / "historical", source.stem
        )
        snapshots.append(ref)
        if source.suffix == ".json":
            old = json.loads(source.read_text())
            history.append(
                dict(
                    path=str(source),
                    sha256=ref["sha256"],
                    honest_verdict=old["honest_verdict"],
                    verdict_class=old["verdict_class"],
                    gate_check_summary=old["gate_check_summary"],
                    resolved=False,
                )
            )
    frozen["authority_snapshots"] = assessment["authority_snapshots"]
    frozen["method_sha256"] = canonical_hash(freeze)
    manifest_path = durable / "validation_command_manifest.json"
    atomic_json(manifest_path, frozen)
    atomic_json(durable / "method_freeze.json", freeze)
    code_refs = [
        authority.lifecycle._snapshot(
            ROOT / n, (ROOT / n).read_bytes(), durable / "code", n.replace("/", "_")
        )
        for n in frozen["dependency_hashes"]
    ]
    refs = [dict(path=s["snapshot_path"], sha256=s["sha256"]) for s in snapshots + code_refs]
    refs += [
        dict(path=str(p), sha256=sha256_file(p))
        for p in (manifest_path, durable / "method_freeze.json")
    ]
    print(
        f"[exp8005] phase=inputs_frozen elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f} completed_units=13 pending=mutations",
        flush=True,
    )
    controls = mutations(
        private / "design.md",
        private / "active.yaml",
        private / "absent",
        durable / "private_mutations",
    )
    receipts: list[dict[str, Any]] = [
        dict(
            name="private_circular_mutations",
            argv=["in_process_private_fixture"],
            passed=all(r["passed"] for r in controls),
            actual_exit=0,
        )
    ]
    coverage: dict[str, Any] = {}
    if not args.fixture_e2e:
        receipts = []
        for spec in frozen["commands"]:
            print(
                f"[exp8005] subprocess_before name={spec['name']} pending=owned_checks", flush=True
            )
            receipt = validation.run_check(ROOT, spec, private, durable / "sealed_logs")
            receipts.append(receipt)
            atomic_json(durable / "validation_checkpoint.json", {"receipts": receipts})
            print(
                f"[exp8005] subprocess_after name={spec['name']} exit={receipt['actual_exit']} completed_units={len(receipts)} elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f}",
                flush=True,
            )
        report = private / "coverage.json"
        complete = validation.coverage_complete(report, includes=[OWNED])
        receipts.append(
            dict(
                name="added_statement_coverage",
                argv=["coverage_json_reduction", str(report)],
                passed=complete,
                actual_exit=int(not complete),
            )
        )
        if report.is_file():
            coverage = {
                n: row["summary"] for n, row in json.loads(report.read_text())["files"].items()
            }
            shutil.copyfile(report, durable / "coverage.json")
    custody = dict(
        ready=True, rows=[], budget=shared.boundary.budget([], 0), hashes=[], gate_check_summary=[]
    )
    for row in assessment["contract_rows"]:
        row.update(
            exclusion_reason="authority_mismatch" if not row["matched"] else None,
            censor_reason=None,
            effective_independent_groups=0,
        )
    for failure in assessment["gate_check_summary"]:
        failure.update(
            path=failure.get("path", failure.get("artifact_path")),
            hash=failure.get("hash", failure.get("artifact_hash")),
            passed=False,
        )
    print(
        f"[exp8005] phase=checks_complete elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f} completed_units={len(receipts)} pending=terminal_validation",
        flush=True,
    )
    value = shared.candidate(
        ROOT,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        manifest_path,
        started,
        time.monotonic_ns(),
        experiment_id=8005,
        milestone="2026.10.694",
        count=13,
    )
    value.update(
        schema="carnot.v694.contract_methods.v1",
        run_date="20261002",
        execution_date="20261002",
        current_run_id="exp8005-20261002",
        random_seed=6948005,
        historical_failure_rows=history,
        cited_upstream_artifacts=[
            {**s, "fields_imported": ["honest_verdict", "verdict_class", "gate_check_summary"]}
            for s in snapshots
        ],
        checkpoint_references=refs,
        raw_shard_hashes=refs,
        code_config_hashes=frozen["dependency_hashes"],
        coverage_statement_counts=coverage,
        lineage_rows=assessment["lineage_applicability_rows"],
        observed_activation=assessment["observed_activation"],
        staging_custody_status=assessment["staging_custody_status"],
        genuine_headroom=dict(natural_evidence=False, scientific_benefit=None),
        positive_control_results=dict(
            twelve_mutations=all(r["passed"] for r in controls),
            scope="private circular authority fixtures",
        ),
        terminal_validation_sidecar_path=str(
            durable / "terminal_validation/terminal_validation.json"
        ),
    )
    value["field_principles"].update(
        {
            k: "Current administrative qualification does not change historical scientific claims."
            for k in value
        }
    )
    value["field_principles"]["sample_size_budget"] = (
        "Thirteen contract tasks provide zero independent scientific units."
    )
    atomic_json(args.raw, primitive_rows(value))
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        terminal = durable / "terminal_validation"
        checked = private / "checked" / args.output.name
        validation.publish(ROOT, value, checked, private / "terminal", terminal)
        atomic_json(args.raw, primitive_rows(value))
        replay_spec = dict(
            name="terminal_cold_replay",
            argv=[
                str(ROOT / ".venv/bin/python"),
                "-u",
                str(ROOT / OWNED),
                "--cold-replay",
                str(checked),
                "--raw",
                str(args.raw),
            ],
            expected_exit=0,
            deadline_s=60,
        )
        replay = validation.run_check(ROOT, replay_spec, private, terminal)
        digest = sha256_file(checked)
        publication = publish_primary(
            args.output,
            value,
            lambda p: dict(
                passed=replay["passed"] and sha256_file(p) == digest and cold_replay(p, args.raw)
            ),
        )
        atomic_json(
            terminal / "terminal_validation.json",
            {
                **publication,
                "passed": True,
                "reports": [replay],
                "validator_report_path": str(terminal / f"terminal-{digest[7:]}.json"),
            },
        )
        selected = reader_receipt(
            "exp8005-contract-methods",
            args.output.parent,
            field="contract_ready_score",
            expected=value["contract_ready_score"],
        )
        if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
            raise ValueError("primary_reader_drift")
        atomic_json(durable / "primary_resolution_receipt.json", selected)
    print(
        f"[exp8005] phase=published elapsed_s={(time.monotonic_ns() - started) / 1e9:.3f} completed_units=13 pending=none",
        flush=True,
    )
    return value


def main() -> int:
    """Private output arguments let the real entrypoint be checked without touching primaries."""
    print("[exp8005] phase=start completed_units=0 pending=arguments", flush=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261002", choices=["20261002"])
    parser.add_argument(
        "--design", type=Path, default=ROOT / "openspec/change-proposals/research-roadmap-vNEXT.md"
    )
    parser.add_argument("--active", type=Path, default=ROOT / "research-roadmap.yaml")
    parser.add_argument("--staged", type=Path, default=ROOT / "research-roadmap-next.yaml")
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/experiment_8005_v694_contract_methods.json"
    )
    parser.add_argument(
        "--raw",
        type=Path,
        default=ROOT / "results/raw/experiment_8005_v694_contract_methods/rows.json",
    )
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--fixture-e2e", action="store_true")
    args = parser.parse_args()
    if args.cold_replay:
        passed = cold_replay(args.cold_replay, args.raw)
        print("cold_replay_passed" if passed else "cold_replay_mismatch", flush=True)
        return 0 if passed else 1
    if args.fixture_e2e and any(p.resolve().is_relative_to(ROOT) for p in (args.output, args.raw)):
        parser.error("fixture outputs must be private and outside the checkout")
    with tempfile.TemporaryDirectory(prefix="exp8005-") as directory:
        execute(args, Path(directory))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
