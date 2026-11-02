"""REQ-REPORT-8031: freeze current authority without converting fixtures into science.

Private controls demonstrate reader mechanics. A missing external design stays
blocked even when every control works, so downstream science keeps its own gates.
"""

from functools import partial
import gzip
import json
from pathlib import Path
import shutil
import time
from typing import Any

import yaml

from carnot.reporting import v686_contract_methods as shared
from carnot.reporting import v686_contract_validation as validation
from carnot.reporting import v690_authority as authority
from carnot.reporting import v690_contract_methods as inherited
from carnot.reporting import v693_contract_methods as prior
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.roadmap_contract import parse_design

ROOT = Path(__file__).resolve().parents[3]
CLI = "scripts/experiments/experiment_8031_v696_contract_methods.py"
OWNED = ["python/carnot/reporting/v696_contract_methods.py", CLI]
TEST = "tests/python/test_experiment_8031_v696_contract_methods.py"
MILESTONE = "2026.10.696"
MODEL_SPECS: list[str] = []
mutations = partial(inherited.mutations, milestone=MILESTONE, first_id=8031)
START = time.monotonic()


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Real work boundaries keep the conductor informed while bounded children run."""
    print(
        f"[exp8031] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> dict[str, Any]:
    """The existing reader compares thirteen complete tasks, including their prompts."""
    value = authority.assess(design, staged, active, snapshots, milestone=MILESTONE, first_id=8031)
    if value["canonical_tasks_sha256"]:
        _, machine = parse_design(design.read_text(), milestone=MILESTONE)
        observed = authority.lifecycle.tasks_digest(machine)
        if observed != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                shared.operand(
                    design,
                    "design_tasks_sha256",
                    value["canonical_tasks_sha256"],
                    observed,
                    "V696_authority",
                )
            )
            value.update(activated=False, observed_activation=False)
    return value


def manifest(root: Path, private: Path) -> dict[str, Any]:
    """Reuse bounded supervision and measure only this invocation's added statements."""
    value = validation.manifest(root, private)
    tests = [TEST, validation.TESTS[1]]
    value["commands"] = [
        r
        for r in value["commands"]
        if r["name"] != "affected_pytest" and not r["name"].startswith("e2e_016")
    ]
    for row in value["commands"]:
        row["argv"] = [
            ("--include=" + ",".join(str(root / p) for p in OWNED))
            if a.startswith("--include=")
            else a.replace("7903_v686", "8031_v696").replace("20260930", "20261002")
            for a in row["argv"]
        ]
        if row["name"] in {"coverage_unit", "scoped_spec_coverage"}:
            row["argv"] = [a for a in row["argv"] if a not in validation.TESTS + tests] + tests
        if row["name"] in {"ruff_check", "ruff_format", "mypy_strict"}:
            row["argv"] = row["argv"][: 3 if row["name"] == "ruff_format" else 2] + OWNED
            row["argv"] += ["--follow-imports=silent"] if row["name"] == "mypy_strict" else [TEST]
        if row["name"] == "repository_full_suite":
            row.update(argv=[str(root / ".venv/bin/pytest"), "tests/python", "-q"], deadline_s=90)
    py = str(root / ".venv/bin/python")
    for name, argv in (
        (
            "roadmap_schema",
            [
                py,
                "-c",
                "import yaml; from scripts.roadmap_schema import Roadmap; Roadmap.model_validate(yaml.safe_load(open('research-roadmap.yaml'))); print('schema_passed', flush=True)",
            ],
        ),
        ("roadmap_gate_audit", [py, "scripts/audit_roadmap_gates.py", "research-roadmap.yaml"]),
        ("exclusion_manifest", [py, "scripts/exclusion_manifest_lint.py", "research-roadmap.yaml"]),
    ):
        value["commands"].append(
            dict(name=name, argv=argv, expected_exit=0, deadline_s=90, classification="required")
        )
    value.update(
        coverage_includes=OWNED,
        affected_tests=tests,
        dependency_hashes=validation.dependency_hashes(
            root, paths=OWNED + tests + validation.CONSUMERS
        ),
    )
    value["dependency_hashes"]["ops/exclusion_manifest.yaml"] = sha256_file(
        root / "ops/exclusion_manifest.yaml"
    )
    for name in ("design.md.gz", "active.yaml.gz"):
        path = "tests/fixtures/v696/" + name
        value["dependency_hashes"][path] = sha256_file(root / path)
    return value


def seal(value: dict[str, Any], durable: Path, role: str) -> dict[str, str]:
    """Content addressed names prevent a later invocation from replacing old operands."""
    path = durable / f"{role}-{canonical_hash(value)[7:]}.json"
    atomic_json(path, value)
    return dict(path=str(path), sha256=sha256_file(path))


def historical(durable: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep original blocked authority operands and their immutable source bytes."""
    refs, rows = [], []
    for identity, slug in ((8018, "contract_methods"), (8030, "capstone")):
        source = ROOT / f"results/experiment_{identity}_v695_{slug}.json"
        original = json.loads(source.read_text())
        saved = authority.lifecycle._snapshot(
            source, source.read_bytes(), durable / "historical", source.stem
        )
        ref = dict(path=saved["snapshot_path"], sha256=saved["sha256"], original_path=str(source))
        refs.append(ref)
        rows.append(
            dict(
                experiment_id=identity,
                honest_verdict=original["honest_verdict"],
                verdict_class=original["verdict_class"],
                gate_check_summary=original["gate_check_summary"],
                original_primary_sha256=saved["sha256"],
                resolved=False,
            )
        )
        for role, snapshot in original["authority_snapshots"].items():
            if snapshot["exists"]:
                path = Path(snapshot["snapshot_path"])
                frozen = authority.lifecycle._snapshot(
                    path, path.read_bytes(), durable / "historical", f"{identity}_{role}"
                )
                refs.append(dict(path=frozen["snapshot_path"], sha256=frozen["sha256"]))
    return refs, rows


def method_freeze(design: Path, tasks: list[dict[str, Any]]) -> dict[str, Any]:
    """Freeze this design's statistics and complete branch protocols before checks."""
    freeze = prior.method_freeze(tasks)
    text = design.read_text() if design.is_file() else ""
    freeze.update(
        design_methods=text.split("## Exact task contract", 1)[0],
        branch_protocol_hashes={t["id"]: canonical_hash(t) for t in tasks},
        statistical_choices=dict(
            draws=10000,
            multiplicity="one-sided Holm H1/H2/H3 at 0.05",
            absent_p=1,
            source_unit="source group",
            online_unit="moving block length32; sensitivity16/64",
            seed_reduction="mean within source or slot",
            target="probability of unsupportedness",
            action_thresholds=dict(accept_below=0.1, reject_above=0.5, ties="escalate"),
        ),
        primary_comparisons=[
            dict(
                hypothesis="H1",
                task="exp8037",
                comparator="two-feature logistic vs full-NLL-only logistic",
                endpoint="Brier reduction",
                minimum=0.01,
                cost_increase_max=0.01,
                extra_false_accepts=0,
            ),
            dict(
                hypothesis="H2",
                task="exp8037",
                comparator="two-feature spline energy vs two-feature logistic",
                endpoint="typed cost reduction",
                minimum=0.02,
                brier_increase_max=0.01,
                extra_false_accepts=0,
                beneficial_changed_decisions_min=5,
            ),
            dict(
                hypothesis="H3",
                task="exp8039",
                comparator="window64 vs cumulative replay",
                endpoint="later typed cost reduction",
                minimum=0.02,
                retention_brier_increase_max=0.01,
                retention_cost_increase_max=0.02,
                extra_false_accepts=0,
                beneficial_changed_decisions_min=5,
            ),
        ],
        science_pre_gate=False,
    )
    freeze["gate_producer_rows"] = [
        dict(
            task_id=t["id"],
            upstream_id=g["upstream"],
            artifact_field=g["artifact_field"],
            passed=g["artifact_field"]
            in producer["prompt"].split("REQUIRED ARTIFACT FIELDS:", 1)[-1],
            producer_protocol_sha256=canonical_hash(producer),
        )
        for i, t in enumerate(tasks)
        for g in t["gated_on"]
        for producer in tasks[:i]
        if producer["id"] == g["upstream"]
    ]
    return freeze


def primitive_rows(value: dict[str, Any]) -> dict[str, Any]:
    """An aggregate seal supplements primitive reduction so edited claims cannot pass."""
    return {**shared.primitive_rows(value), "artifact_sha256": canonical_hash(value)}


def cold_replay(path: Path, raw: Path) -> bool:
    """Reconstruct administrative counts and authenticate immutable invocation operands."""
    if not path.is_file() or not raw.is_file():
        return False
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or value.get("experiment_id") != 8031:
            return False
        refs = value["checkpoint_references"] + [
            dict(path=s["snapshot_path"], sha256=s["sha256"])
            for s in value["authority_snapshots"].values()
            if s["exists"]
        ]
        return bool(
            primitive_rows(value) == json.loads(raw.read_text())
            and value["task_id"] == "exp8031-contract-methods"
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
                Path(r["path"]).is_file() and sha256_file(Path(r["path"])) == r["sha256"]
                for r in refs
            )
        )
    except (ValueError, KeyError, TypeError, OSError):
        return False


def execute(args: Any, private: Path) -> dict[str, Any]:
    """Freeze invocation bytes before private checks; outside prerequisites remain terminal."""
    started = time.monotonic_ns()
    private.mkdir(parents=True, exist_ok=True)
    durable = args.raw.parent
    progress("inputs_before", pending="immutable authority and original failures")
    assessment = assess(args.design, args.staged, args.active, durable / "authority_snapshots")
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v696" / (name + ".gz")).read_bytes())
        )
    atomic_json(private / "negative.json", {"experiment_id": 0})
    atomic_json(private / "negative-rows.json", {"rows": []})
    if assessment["canonical_tasks_sha256"]:
        _, tasks = parse_design(args.design.read_text(), milestone=MILESTONE)
    else:
        tasks = yaml.safe_load(args.active.read_bytes())["tasks"] if args.active.is_file() else []
    freeze = method_freeze(args.design, tasks)
    frozen = manifest(ROOT, private)
    frozen.update(
        authority_snapshots=assessment["authority_snapshots"], method_sha256=canonical_hash(freeze)
    )
    refs, history = historical(durable)
    method_ref = seal(freeze, durable, "method_freeze")
    manifest_ref = seal(frozen, durable, "validation_command_manifest")
    refs += [method_ref, manifest_ref]
    code_refs = [
        authority.lifecycle._snapshot(
            ROOT / n, (ROOT / n).read_bytes(), durable / "code", n.replace("/", "_")
        )
        for n in frozen["dependency_hashes"]
    ]
    refs += [dict(path=s["snapshot_path"], sha256=s["sha256"]) for s in code_refs]
    progress("inputs_frozen", 13, "private mutations and terminal reader checks")
    controls = mutations(
        private / "design.md",
        private / "active.yaml",
        private / "absent",
        private / "mutations",
    )
    for row in controls:
        row.update(
            raw_numerator=int(row["passed"]),
            raw_denominator=1,
            excluded=False,
            censored=False,
            exclusion_reason=None,
            censor_reason=None,
            effective_independent_groups=0,
        )
    receipts = [
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
            progress("subprocess_before_" + spec["name"], len(receipts), "owned checks")
            receipts.append(
                validation.run_check(ROOT, spec, private, durable / "sealed_logs", heartbeat_s=30)
            )
            refs.append(seal(dict(receipts=receipts), durable / "checkpoints", "validation"))
            progress("subprocess_after_" + spec["name"], len(receipts), "remaining checks")
        report = private / "coverage.json"
        complete = validation.coverage_complete(report, includes=OWNED)
        receipts.append(
            dict(
                name="added_statement_coverage",
                argv=["coverage_json_reduction", str(report)],
                passed=complete,
                actual_exit=int(not complete),
            )
        )
        if report.is_file():
            coverage = {n: r["summary"] for n, r in json.loads(report.read_text())["files"].items()}
            refs.append(seal(dict(statement_counts=coverage), durable, "coverage_statement_counts"))
    custody = dict(
        ready=True, rows=[], budget=shared.boundary.budget([], 0), hashes=[], gate_check_summary=[]
    )
    refs += [dict(path=r["log_path"], sha256=r["log_sha256"]) for r in receipts if "log_path" in r]
    refs.append(seal(dict(rows=controls), durable, "mutation_rows"))
    value = shared.candidate(
        ROOT,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        Path(manifest_ref["path"]),
        started,
        time.monotonic_ns(),
        experiment_id=8031,
        milestone=MILESTONE,
        count=13,
    )
    for row in value["rows"]:
        row.update(
            exclusion_reason="authority_mismatch" if not row["matched"] else None,
            censor_reason=None,
            effective_independent_groups=0,
        )
    for failure in value["gate_check_summary"]:
        failure.update(
            path=failure.get("path", failure.get("artifact_path")),
            hash=failure.get("hash", failure.get("artifact_hash")),
            passed=False,
            sha256=failure.get("hash", failure.get("artifact_hash")),
            check_name=failure["artifact_field"],
        )
    value.update(
        schema="carnot.v696.contract_methods.v1",
        run_date="20261002",
        execution_date="20261002",
        current_run_id="exp8031-20261002",
        random_seed=6968031,
        historical_reader_failure_rows=history,
        cited_upstream_artifacts=[r for r in refs if "original_path" in r],
        checkpoint_references=refs,
        raw_shard_hashes=refs,
        code_config_hashes=frozen["dependency_hashes"],
        coverage_statement_counts=coverage,
        lineage_rows=assessment["lineage_applicability_rows"],
        observed_activation=assessment["observed_activation"],
        staging_custody_status=assessment["staging_custody_status"],
        historical_active_milestone="2026.10.695",
        staged_v696_status="not_observed_consumed" if not args.staged.is_file() else "observed",
        genuine_headroom=dict(natural_evidence=False, scientific_benefit=None),
        positive_control_results=dict(
            twelve_mutations=all(r["passed"] for r in controls),
            scope="private circular fixtures; terminal readers measured by owned tests",
        ),
        generalized_learning_benefit_score=0,
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
    )
    value.update(
        {
            f"{key}_count": number
            for key, number in value["sample_size_budget"].items()
            if key != "started"
        }
    )
    value["substrate_declaration"] = dict(
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        pretrained_model_calls=0,
    )
    value["planning_ready_score"] = assessment["planning_ready_score"]
    value["statistical_plan_sha256"] = canonical_hash(freeze)
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    value["field_principles"].update(
        {
            k: "Bind immutable current administrative evidence; historical failures and science stay separate."
            for k in value
        }
    )
    value["field_principles"]["sample_size_budget"] = (
        "Thirteen authority tasks give zero independent scientific units."
    )
    value["field_principles"]["contract_ready_score"] = (
        "Exact authority plus complete owned checks; no scientific gate or benefit transfer."
    )
    atomic_json(args.raw, primitive_rows(value))
    progress("checks_complete", len(receipts), "exact terminal bytes")
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        publish(value, args.output, args.raw, private, durable)
    progress("published", 13, "none")
    return value


def publish(value: dict[str, Any], output: Path, raw: Path, private: Path, durable: Path) -> None:
    """Existing validators and publication lock accept only the exact cold-replayed bytes."""
    checked = private / "checked" / output.name
    validation.publish(ROOT, value, checked, private / "terminal", durable / "terminal_validation")
    atomic_json(raw, primitive_rows(value))
    spec = dict(
        name="terminal_cold_replay",
        argv=[
            str(ROOT / ".venv/bin/python"),
            "-u",
            str(ROOT / CLI),
            "--cold-replay",
            str(checked),
            "--raw",
            str(raw),
        ],
        expected_exit=0,
        deadline_s=60,
    )
    replay = validation.run_check(ROOT, spec, private, durable / "terminal_validation")
    digest = sha256_file(checked)
    publication = publish_primary(
        output,
        value,
        lambda p: dict(
            passed=replay["passed"] and sha256_file(p) == digest and cold_replay(p, raw)
        ),
    )
    selected = reader_receipt(
        "exp8031-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(
        durable / "terminal_validation.json",
        dict(**publication, passed=True, reports=[replay], readers=selected),
    )
    rechecks = [
        validation.run_check(
            ROOT,
            dict(name=name, argv=argv, expected_exit=0, deadline_s=60),
            private,
            durable / "terminal_validation",
        )
        for name, argv in (
            (
                "published_cold_replay",
                [
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--cold-replay",
                    str(output),
                    "--raw",
                    str(raw),
                ],
            ),
            (
                "published_adversarial",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/adversarial_verify.py",
                    "--json",
                    str(output),
                ],
            ),
            (
                "published_strict_rows",
                [
                    str(ROOT / ".venv/bin/python"),
                    "scripts/verdict_row_consistency_lint.py",
                    "--strict",
                    str(output),
                ],
            ),
        )
    ]
    atomic_json(
        durable / "published_recheck.json",
        dict(primary_sha256=sha256_file(output), receipts=rechecks),
    )
    if sha256_file(output) != digest or not all(r["passed"] for r in rechecks):
        raise ValueError("published_validation_failed")
