"""REQ-REPORT-8044: qualify an administrative reader without scientific credit."""

import gzip
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

import yaml

from carnot.reporting import v696_contract_methods as prior
from carnot.reporting import v697_evidence as evidence
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT = prior.ROOT
CLI = "scripts/experiments/experiment_8044_v697_contract_methods.py"
TEST = "tests/python/test_experiment_8044_v697_contract_methods.py"
OWNED = [
    "python/carnot/reporting/v697_contract_methods.py",
    "python/carnot/reporting/v697_evidence.py",
    CLI,
]
MILESTONE = "2026.10.697"
MODEL_SPECS: list[str] = []
validation = prior.validation
history = evidence.history
START = time.monotonic()


def progress(phase: str, units: int = 0, pending: str = "") -> None:
    """Report real phase boundaries so bounded children cannot appear stalled."""
    print(
        f"[exp8044] phase={phase} elapsed_s={time.monotonic() - START:.3f} completed_units={units} pending={pending}",
        flush=True,
    )


def assess(design: Path, staged: Path, active: Path, snapshots: Path) -> Json:
    """Reuse parameterized readers and authenticate the full embedded task list."""
    value = prior.authority.assess(
        design, staged, active, snapshots, milestone=MILESTONE, first_id=8044
    )
    if value["canonical_tasks_sha256"]:
        _, tasks = parse_design(design.read_text(), milestone=MILESTONE)
        actual = prior.authority.lifecycle.tasks_digest(tasks)
        if actual != value["canonical_tasks_sha256"]:
            value["gate_check_summary"].append(
                evidence.operand(
                    design,
                    "V697_authority",
                    "design_tasks_sha256",
                    value["canonical_tasks_sha256"],
                    actual,
                )
            )
            value.update(activated=False, observed_activation=False)
    return value


def mutations(design: Path, active: Path, staged: Path, private: Path) -> list[Json]:
    """Use the twelve shipped adversarial controls with current task identities."""
    return prior.inherited.mutations(
        design, active, staged, private, milestone=MILESTONE, first_id=8044
    )


def method_freeze(design: Path, tasks: list[Json]) -> Json:
    """Keep complete task protocols so later work inherits actual budgets and rules."""
    mappings = [
        dict(
            method="GASP",
            source="https://arxiv.org/abs/2607.04223",
            tasks=[8047, 8048, 8049, 8050],
            code_steps="full_A/full_B/no_source_A/no_source_B fixed-answer scoring; fit matched logistic and spline energy; seal predictions; audit human full-source targets",
            non_transferring_assumptions="source sensitivity is not correctness; generator domain results and annotation accuracy do not transfer",
        ),
        dict(
            method="LILAC+",
            source="https://arxiv.org/html/2605.18842v1",
            tasks=[8051, 8052, 8053],
            code_steps="propose small-head gradient; check disjoint released-feedback guard; backtrack or rollback; audit future loss and charge guard scans",
            non_transferring_assumptions="driving constraints and conditional safety theorem do not transfer to factual verification",
        ),
        dict(
            method="FedProTIP",
            source="https://arxiv.org/html/2509.21606v1",
            tasks=[8051, 8052],
            code_steps="bound candidate parameter motion with finite backtracking; freeze rejected updates; independently test retention",
            non_transferring_assumptions="no federated clients, learned projector or inferred task identity; this is a local adaptation",
        ),
    ]
    return dict(
        task_contracts=tasks,
        branch_protocol_hashes={t["id"]: canonical_hash(t) for t in tasks},
        design_methods=design.read_text().split("## Exact task contract", 1)[0]
        if design.is_file()
        else None,
        statistical_choices=dict(
            draws=10000,
            multiplicity="one-sided Holm H1/H2/H3 at 0.05",
            absent_p=1,
            margins=[0.01, 0.02, 0.02],
            source_unit="source group",
            online_unit="chronological moving blocks32; sensitivity16/64",
            seed_reduction="within source or slot",
        ),
        method_source_map=mappings,
        literature_adoption_decisions=mappings,
        deferred_directions=[
            "EBT generator training",
            "ETS decoder changes",
            "new constraint solver",
            "hardware speed or acquisition before compatible complete-service measurements",
        ],
        science_pre_gate=False,
        current_pretrained_calls=0,
    )


def manifest(root: Path, private: Path) -> Json:
    """Freeze exact commands; inherited historical coverage targets stay excluded."""
    value = prior.manifest(root, private)
    for spec in value["commands"]:
        spec["argv"] = [
            a.replace("8031_v696", "8044_v697")
            .replace("20261002", "20261003")
            .replace(prior.OWNED[0], OWNED[0])
            for a in spec["argv"]
        ]
        if spec["name"] in ("ruff_check", "ruff_format", "mypy_strict"):
            spec["argv"].insert(2, OWNED[1])
        if spec["name"] == "repository_full_suite":
            spec["deadline_s"] = 90
        spec["argv"] = [
            "--include=" + ",".join(str(root / p) for p in OWNED)
            if a.startswith("--include=")
            else a
            for a in spec["argv"]
        ]
    value.update(
        coverage_includes=OWNED,
        affected_tests=[TEST, validation.TESTS[1]],
        dependency_hashes=validation.dependency_hashes(
            root, paths=OWNED + [TEST, validation.TESTS[1]] + validation.CONSUMERS
        ),
    )
    for name in ("design.md.gz", "active.yaml.gz"):
        path = "tests/fixtures/v697/" + name
        value["dependency_hashes"][path] = sha256_file(root / path)
    return value


def primitive_rows(value: Json) -> Json:
    """Seal claims as well as rows because custody alone cannot detect edited verdicts."""
    return dict(
        rows=value["rows"],
        historical_disposition_rows=value["historical_disposition_rows"],
        artifact_sha256=canonical_hash(value),
    )


def cold_replay(path: Path, raw: Path) -> bool:
    """Recompute authority equations in a fresh process using durable source copies."""
    try:
        value = json.loads(path.read_text())
        rows = value["rows"]
        refs = value["checkpoint_references"]
        snapshots = value["authority_snapshots"]
        design = next((r for r in refs if r.get("role") == "current_design"), None)
        active = next((r for r in refs if r.get("role") == "current_active"), None)
        matched = False
        tasks = []
        if design and active:
            table, tasks = parse_design(Path(design["path"]).read_text(), milestone=MILESTONE)
            observed = yaml.safe_load(Path(active["path"]).read_bytes())
            shown = [
                dict(order=i + 1, **{k: t[k] for k in ("id", "title", "phase", "deliverable")})
                for i, t in enumerate(tasks)
            ]
            matched = (
                observed["milestone"] == MILESTONE
                and tasks == observed["tasks"]
                and table == shown
                and len(tasks) == 13
            )
        return bool(
            value["experiment_id"] == 8044
            and value["task_id"] == "exp8044-contract-methods"
            and primitive_rows(value) == json.loads(raw.read_text())
            and len(rows) == 13
            and all(r["absolute_metric"] == int(all(r["checks"].values())) for r in rows)
            and value["sample_size_budget"] == prior.shared.contract_budget(rows, count=13)
            and evidence.replay_history(value["historical_disposition_rows"], refs)
            and (not value["activation_confirmed"] or matched)
            and (
                not value["activation_confirmed"]
                or prior.authority.lifecycle.tasks_digest(tasks) == value["canonical_tasks_sha256"]
            )
            and all(
                Path(r["path"]).is_file() and sha256_file(Path(r["path"])) == r["sha256"]
                for r in refs
            )
            and all(
                r.get("role") != "current_code"
                or sha256_file(Path(r["original_path"])) == r["sha256"]
                for r in refs
            )
            and all(
                sha256_file(Path(s["snapshot_path"])) == s["sha256"]
                for s in snapshots.values()
                if s["exists"]
            )
        )
    except (OSError, ValueError, KeyError, TypeError, StopIteration):
        return False


def execute(args: Any, private: Path) -> Json:
    """Freeze owned decisions before checks and keep external failures terminal."""
    started = time.monotonic_ns()
    private.mkdir(parents=True, exist_ok=True)
    durable = args.raw.parent
    progress("preconditions", pending="local resources, terminal sidecars and invocation authority")
    custody = history(ROOT, [], durable / "historical")
    assessment = assess(args.design, args.staged, args.active, durable / "authority_snapshots")
    tasks = (
        parse_design(args.design.read_text(), milestone=MILESTONE)[1]
        if assessment["canonical_tasks_sha256"]
        else []
    )
    custody = history(ROOT, tasks, durable / "historical")
    freeze = method_freeze(args.design, tasks)
    frozen = manifest(ROOT, private)
    if getattr(args, "owned_checks_only", False):
        frozen["commands"] = [s for s in frozen["commands"] if s["classification"] != "diagnostic"]
        frozen["repository_health_scope"] = (
            "Earlier bounded invocation; preserved separately from current owned acceptance."
        )
    frozen.update(
        authority_snapshots=assessment["authority_snapshots"], method_sha256=canonical_hash(freeze)
    )
    method_ref = prior.seal(freeze, durable, "method_freeze")
    manifest_ref = prior.seal(frozen, durable, "validation_command_manifest")
    refs = custody["hashes"] + [method_ref, manifest_ref]
    refs += [
        dict(path=str(p), sha256=sha256_file(p), role="development_log")
        for p in sorted((durable / "development_logs").glob("*"))
        if p.is_file()
    ]
    for name, digest in frozen["dependency_hashes"].items():
        saved = prior.authority.lifecycle._snapshot(
            ROOT / name, (ROOT / name).read_bytes(), durable / "code", name.replace("/", "_")
        )
        refs.append(
            dict(
                path=saved["snapshot_path"],
                sha256=digest,
                original_path=str(ROOT / name),
                role="current_code",
            )
        )
    for role, snapshot in assessment["authority_snapshots"].items():
        if snapshot["exists"]:
            refs.append(
                dict(
                    path=snapshot["snapshot_path"],
                    sha256=snapshot["sha256"],
                    role="current_" + role,
                )
            )
    for name in ("design.md", "active.yaml"):
        (private / name).write_bytes(
            gzip.decompress((ROOT / "tests/fixtures/v697" / (name + ".gz")).read_bytes())
        )
    atomic_json(private / "negative.json", dict(experiment_id=0))
    atomic_json(private / "negative-rows.json", dict(rows=[]))
    progress("inputs_frozen", 13, "twelve private mutations and exact validation manifest")
    controls = mutations(
        private / "design.md", private / "active.yaml", private / "absent", private / "mutations"
    )
    receipts = [
        dict(
            name="private_circular_mutations",
            argv=["in_process_private_fixture"],
            expected_exit=0,
            actual_exit=0,
            passed=all(r["passed"] for r in controls),
        )
    ]
    coverage: Json = {}
    if not args.fixture_e2e:
        receipts = []
        for spec in frozen["commands"]:
            progress("subprocess_before_" + spec["name"], len(receipts), "owned checks")
            receipts.append(
                validation.run_check(
                    ROOT, spec, private, durable / "validation_logs", heartbeat_s=30
                )
            )
            progress("subprocess_after_" + spec["name"], len(receipts), "remaining checks")
        report = private / "coverage.json"
        complete = validation.coverage_complete(report, includes=OWNED)
        receipts.append(
            dict(
                name="added_statement_coverage",
                argv=["coverage_json_reduction"],
                expected_exit=0,
                actual_exit=int(not complete),
                passed=complete,
            )
        )
        if report.is_file():
            coverage = {n: r["summary"] for n, r in json.loads(report.read_text())["files"].items()}
            refs.append(prior.seal(json.loads(report.read_text()), durable, "coverage_report"))
    refs += [dict(path=r["log_path"], sha256=r["log_sha256"]) for r in receipts if "log_path" in r]
    value = prior.shared.candidate(
        ROOT,
        assessment,
        custody,
        freeze,
        controls,
        receipts,
        Path(manifest_ref["path"]),
        started,
        time.monotonic_ns(),
        experiment_id=8044,
        milestone=MILESTONE,
        count=13,
    )
    for row in value["rows"]:
        row.update(
            effective_independent_groups=0,
            exclusion_reason="authority_mismatch" if not row["matched"] else None,
            censor_reason=None,
        )
    for gate in value["gate_check_summary"]:
        gate.update(
            path=gate.get("path", gate.get("artifact_path")),
            sha256=gate.get("sha256", gate.get("hash", gate.get("artifact_hash"))),
            passed=False,
            check_name=gate["artifact_field"],
        )
    value.update(
        schema="carnot.v697.contract_methods.v1",
        run_date=args.date,
        current_run_id="exp8044-" + args.date,
        random_seed=6978044,
        checkpoint_references=refs,
        raw_shard_hashes=refs,
        cited_upstream_artifacts=custody["hashes"],
        code_config_hashes=frozen["dependency_hashes"],
        coverage_statement_counts=coverage,
        historical_disposition_rows=custody["historical_disposition_rows"],
        prior_failure_checks=custody["prior_failure_checks"],
        method_source_map=freeze["method_source_map"],
        preconditions_checked=custody["preconditions_checked"],
        python_environment=dict(version=sys.version, executable=sys.executable),
        staging_custody_status=assessment["staging_custody_status"],
        activation_observation=dict(
            observed=assessment["observed_activation"], staging=assessment["staging_custody_status"]
        ),
        terminal_validation_sidecar_path=str(durable / "terminal_validation.json"),
        MODEL_SPECS=[],
        genuine_headroom=dict(natural_evidence=False, scientific_benefit=None),
        positive_control_results=dict(twelve_mutations=all(r["passed"] for r in controls)),
        generalized_learning_benefit_score=0,
        substrate_declaration=dict(
            inference_substrate="aggregation_from_upstream_artifacts",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            pretrained_model_calls=0,
        ),
    )
    value["repository_health"]["earlier_bounded_runs"] = [
        dict(
            path=str(p),
            sha256=sha256_file(p),
            observation=json.loads(p.read_text())["repository_health"],
        )
        for p in (durable / "development_logs").glob("initial_primary.json")
    ]
    value.update(
        {k + "_count": v for k, v in value["sample_size_budget"].items() if k != "started"}
    )
    value["started_monotonic_timestamp_ns"] = value.pop("started_monotonic_ns")
    value["ended_monotonic_timestamp_ns"] = value.pop("ended_monotonic_ns")
    if value["verdict_class"] == "blocked" and custody["gate_check_summary"]:
        value["honest_verdict"] = "complete_blocked_" + re.sub(
            "[^a-z0-9_]", "_", Path(custody["gate_check_summary"][0]["path"]).name.lower()
        )
    value["field_principles"].update(
        {
            k: "Bind actual current administrative evidence; imported historical measurements give no scientific credit."
            for k in value
        }
    )
    value["field_principles"].update(
        historical_disposition_rows="Final byte identities and earlier log claims stay separate; absence has no measured verdict.",
        contract_ready_score="Qualified authority and all owned checks; scientific benefit is separate.",
        model_invocation_counts="Zero current pretrained calls; imported work is historical.",
    )
    atomic_json(args.raw, primitive_rows(value))
    if args.fixture_e2e:
        atomic_json(args.output, value)
    else:
        publish(value, args.output, args.raw, private, durable)
    progress("published", 13, "none")
    return value


def publish(value: Json, output: Path, raw: Path, private: Path, durable: Path) -> None:
    """Check candidate and published bytes through the existing terminal consumers."""
    candidate = private / output.name
    atomic_json(candidate, value)
    digest = sha256_file(candidate)
    reports = []
    py = str(ROOT / ".venv/bin/python")
    for phase, target in (("candidate", candidate), ("published", output)):
        for name, argv in (
            (
                "cold_replay",
                [py, "-u", str(ROOT / CLI), "--cold-replay", str(target), "--raw", str(raw)],
            ),
            ("adversarial", [py, "scripts/adversarial_verify.py", "--json", str(target)]),
            (
                "strict_rows",
                [py, "scripts/verdict_row_consistency_lint.py", "--strict", str(target)],
            ),
        ):
            progress("subprocess_before_" + phase + "_" + name, len(reports), "terminal validation")
            spec = dict(name=phase + "_" + name, argv=argv, expected_exit=0, deadline_s=120)
            reports.append(validation.run_check(ROOT, spec, private, durable / "terminal_logs"))
            progress("subprocess_after_" + phase + "_" + name, len(reports), "terminal validation")
        if not all(r["passed"] for r in reports):
            prior.seal(dict(receipts=reports), durable, "failed_terminal")
            raise ValueError("terminal_validation_failed")
        if phase == "candidate":
            publication = publish_primary(
                output,
                value,
                lambda p: dict(passed=sha256_file(p) == digest and cold_replay(p, raw)),
            )
    selected = reader_receipt(
        "exp8044-contract-methods",
        output.parent,
        field="contract_ready_score",
        expected=value["contract_ready_score"],
    )
    if selected["gate_sha256"] != digest or selected["document_sha256"] != digest:
        raise ValueError("primary_reader_drift")
    atomic_json(
        durable / "terminal_validation.json",
        dict(**publication, passed=True, reports=reports, readers=selected),
    )
