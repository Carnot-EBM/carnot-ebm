"""REQ-REPORT-8051: publish persistent feedback-constrained CPU trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8038_v696_windowed_online_learning as prior
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify import feedback_constrained_8051 as m

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8051_v697_feedback_constrained_learning"
TASK = "exp8051-feedback-constrained-learning"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_feedback_constrained_8051.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/feedback_constrained_8051.py", SCRIPT]
MODEL_SPECS: list[str] = []


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Copy the CPU branch's actual operands without importing unrelated archived work."""
    data: Json = dict(references=[], gate_checks=[])
    failures = []
    values: Json = {}
    resources = [
        ROOT / p
        for p in (
            "AGENTS.md",
            "CODEX.md",
            "CLAUDE.md",
            "ops/e2e-test-plan.md",
            "openspec/capabilities/research-reporting/spec.md",
            "scripts/experiment_template.py",
            "python/carnot/reporting/current_work_receipt.py",
            "python/carnot/reporting/primary_publication.py",
            "python/carnot/verify/windowed_online_8038.py",
            "python/carnot/verify/causal_online_8025.py",
            "python/carnot/verify/sparse_energy_7996.py",
            "python/carnot/experiment_8038_v696_windowed_online_learning.py",
            "research-references.md",
        )
    ]
    resources += [
        ROOT / ".venv/bin" / tool for tool in ("python", "pytest", "coverage", "ruff", "mypy")
    ]
    for resource in resources:
        present = resource.is_file()
        check = dict(
            upstream_id="local_resource",
            path=str(resource),
            sha256=sha256_file(resource) if present else None,
            check_name="resource_exists",
            artifact_field="resource_exists",
            expected=True,
            observed=present,
            passed=present,
        )
        data["gate_checks"].append(check)
        if not present:
            failures.append(check)
    path = root / "results"

    def require(field: str, expected: Any, observed: Any) -> None:
        prior.upstream.require(path, field, expected, observed)
        data["gate_checks"].append(
            dict(
                upstream_id=path.stem,
                path=str(path),
                sha256=sha256_file(path),
                check_name=field,
                artifact_field=field,
                expected=expected,
                observed=observed,
                passed=True,
            )
        )

    for identity, name, readiness in [
        (8032, prior.upstream.NAME, "learning_inputs_ready_score"),
        (8020, "experiment_8020_v695_qualified_energy_fit", "energy_fit_ready_score"),
        (8046, m.protocol.NAME, "learning_protocol_ready_score"),
    ]:
        path = root / "results" / (name + ".json")
        try:
            require("primary_exists", True, path.is_file())
            value = json.loads(path.read_text())
            sidecar = Path(value["terminal_validation_sidecar_path"])
            binding = json.loads(sidecar.read_text())
            binding = binding.get("publication", binding)
            report_path = Path(binding["sidecar_path"])
            report = json.loads(report_path.read_text())
            for field, expected, observed in [
                ("experiment_id", identity, value.get("experiment_id")),
                (readiness, 1, value.get(readiness, "MISSING_CONTRACT_FIELD")),
                ("flagged_adversarial", False, value.get("flagged_adversarial")),
                ("terminal_primary_sha256", sha256_file(path), binding.get("primary_sha256")),
                ("report_primary_sha256", sha256_file(path), report.get("primary_sha256")),
                ("terminal_primary_path", str(path), binding.get("primary_path")),
                ("report_primary_path", str(path), report.get("primary_path")),
                ("report.passed", True, report.get("report", {}).get("passed")),
            ]:
                require(field, expected, observed)
            data["references"].extend(
                prior.upstream.copy_bound(reference(p), raw) for p in (path, sidecar, report_path)
            )
            values[identity] = value
        except (prior.upstream.Contract, KeyError, ValueError, OSError) as error:
            failures.append(
                error.gate
                if isinstance(error, prior.upstream.Contract)
                else prior.upstream.Contract(
                    path, "input_contract", "complete byte-bound fields", str(error)
                ).gate
            )
    if not failures:
        try:
            m.progress("qualified_small_head_load_before")
            h = values[8046]["qualified_head"]
            head_refs = [
                prior.upstream.copy_bound(ref, raw) for ref in values[8020]["head_checkpoints"]
            ]
            data["references"].extend(head_refs)
            originals = [json.loads(checked(ref).read_text()) for ref in head_refs]
            require(
                "original_head_parameters",
                True,
                any(r["parameters"] == h["parameters"] for r in originals),
            )
            methods_ref = prior.upstream.copy_bound(values[8032]["methods_reference"], raw)
            data["references"].append(methods_ref)
            methods = json.loads(checked(methods_ref).read_text())
            require("frozen_methods", True, methods["frozen"])
            require("sealed_methods", prior.upstream.METHODS, methods["methods"])
            require("converged_head", True, h["converged"] and len(h["parameters"]) == 110)
            head = {k: h[k] for k in ("arm", "parameters", "geometry", "calibration")}
            head["decay_scale"] = 1.0
            for field in ("parameters", "geometry", "calibration"):
                require("head." + field, methods["head"][field], h[field])
            public_ref = prior.upstream.copy_bound(methods["public"]["stream"], raw)
            target_ref = prior.upstream.copy_bound(
                values[8032]["role_manifests"]["evaluator"]["stream"], raw
            )
            data["references"].extend([public_ref, target_ref])
            sources = json.loads(checked(public_ref).read_text())["rows"]
            for row in sources:
                row["source_cluster_id"] = m.protocol.normalized(bytes.fromhex(row["source_bytes"]))
            require("original_slots", list(range(256)), [r["slot"] for r in sources])
            require(
                "guard_partition_rows",
                m.partition_rows(sources),
                values[8046]["guard_partition_rows"],
            )
            data.update(
                head=head, sources=sources, seeds=m.CONFIG["seeds"], target_reference=target_ref
            )
            m.progress("qualified_small_head_load_after", 256, 0)
        except (prior.upstream.Contract, KeyError, ValueError, OSError) as error:
            failures.append(
                error.gate
                if isinstance(error, prior.upstream.Contract)
                else prior.upstream.Contract(
                    path, "input_contract", "complete byte-bound fields", str(error)
                ).gate
            )
    return data, failures


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Use the existing bounded manifest with coverage limited to this task's code."""
    commands = prior.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    replacements = list(zip(prior.OWNED, OWNED, strict=True)) + [(prior.TEST, TEST)]

    def argument(value: str) -> str:
        for before, after in replacements:
            value = value.replace(before, after)
        return value

    return [replace(c, argv=tuple(argument(a) for a in c.argv)) for c in commands]


def replay(path: Path) -> Json:
    """Reject altered primitive bytes and unsafe readiness in a fresh reader."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["acceptance_gate_results"]["trajectory"]:
        for key, observed in m.reduce(Path(value["trajectory_directory"])).items():
            m.equal(value[key], observed, "reduction_drift:" + key)
    m.equal(
        not value["learning_trajectory_ready_score"]
        or (value["verdict_class"] == "null" and all(value["acceptance_gate_results"].values())),
        True,
        "unsafe_readiness",
    )
    return dict(passed=True, sha256=sha256_file(path))


def terminal(path: Path) -> Json:
    """Run cold reduction and both repository artifact linters on exact bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / SCRIPT), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    log_root = path.parent / "raw" / NAME if path.parent.name == "results" else path.parent
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=log_root / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def build(data: Json, failures: list[Json]) -> Json:
    """A blocked result retains empty measurements instead of inventing evidence."""
    value = prior.base(failures)
    value.update(
        experiment_id=8051,
        task_id=TASK,
        milestone="2026.10.697",
        schema="carnot.v697.feedback_constrained_learning.v1",
        run_date="20261003",
        honest_verdict="complete_blocked_feedback_inputs"
        if failures
        else "complete_null_feedback_constrained_learning",
        claim_scope="This invocation measures persistent CPU small-head updates on one historically exposed development stream. Empirical guard checks establish no future safety, retention or independent learning benefit.",
        config=m.CONFIG,
        gate_check_summary=failures or data.get("gate_checks", []),
        preconditions_checked=failures + data.get("gate_checks", []),
        cited_upstream_artifacts=data.get("references", []),
        methodology_note="Frozen calibrated BCE geometry; cumulative update-only replay, step .01 and ridge .001; four attempts per16 eligible update releases, cap64, seeds101-120. Every prediction precedes original slot+20 feedback. All released guard rows check Brier, typed cost and individual new false accepts against the initial head. All five alpha diagnostics run; rejected work counts. Descriptive exposed-stream metrics give no benefit or deployment credit.",
        future_label_mutation_results=[],
        guard_partition_rows=[],
        issue_release_rows=[],
        candidate_update_rows=[],
        accepted_update_rows=[],
        rejection_rows=[],
        reset_rows=[],
        guard_check_rows=[],
        attempted_gradient_counts=[],
        committed_update_counts=[],
        head_checkpoints=[],
    )
    if failures:
        value["honest_verdict"] = "complete_blocked_" + str(failures[0]["check_name"]).replace(
            ".", "_"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    """Freeze inputs, run bounded numerical work, then qualify one primary artifact."""
    m.progress("start_preconditions")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261003"], default="20261003")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = artifact_output_root(root=args.root) / (NAME + ".json")
        raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8051-") as temporary:
            scratch = Path(temporary)
            commands = validation_plan(scratch)
            health_path = output.parent / "raw" / NAME / "repository_health.json"
            historical_health = (
                json.loads(health_path.read_text()) if health_path.exists() else None
            )
            if historical_health is not None:
                commands = [c for c in commands if c.scope != "repository_health"]
                prior.upstream.copy_bound(reference(health_path), raw, "historical_health")
            m.progress("inputs_before")
            data, failures = (
                (json.loads(args.fixture_input.read_text()), [])
                if args.fixture_input
                else load_inputs(args.root, raw)
            )
            value = build(data, failures)
            deps = [
                Path(m.old.__file__),
                Path(m.protocol.__file__),
                Path(prior.__file__),
                Path(prior.upstream.__file__),
                ROOT / "python/carnot/verify/sparse_energy_7996.py",
                ROOT / "python/carnot/reporting/primary_publication.py",
                ROOT / "python/carnot/reporting/current_work_receipt.py",
            ]
            value["code_config_hashes"] = [
                prior.upstream.copy_bound(reference(p), raw, "code")
                for p in [*(ROOT / p for p in OWNED + [TEST]), *deps]
            ]
            atomic_json(
                raw / "configuration.json",
                dict(
                    config=m.CONFIG,
                    guard=m.protocol.METHODS["guard"],
                    code=value["code_config_hashes"],
                    inputs=data.get("references", []),
                    commands=[
                        dict(name=c.name, argv=c.argv, scope=c.scope, timeout_s=c.timeout_s)
                        for c in commands
                    ],
                    artifact_guard_enabled=True,
                    statistics="descriptive source and arm metrics; seeds do not increase independent n",
                ),
            )
            frozen = time.monotonic()
            m.progress("configuration_frozen")
            if not failures:
                value.update(m.measure(data, raw / "trajectory"))
                value["trajectory_directory"] = str(raw / "trajectory")
                value["acceptance_gate_results"]["trajectory"] = True
                value["positive_control_results"] = m.controls()
                value["acceptance_gate_results"]["controls"] = value["positive_control_results"][
                    "passed"
                ]
                value["checkpoint_references"] = [r["head"] for r in value["head_checkpoints"]]
                value["trained_head_specs"] = [
                    dict(
                        r,
                        parameters=len(data["head"]["parameters"]),
                        device="cpu",
                        pretrained=False,
                    )
                    for r in value["attempted_gradient_counts"]
                ]
            measured = time.monotonic()
            if args.fixture_input:
                value.update(
                    verifier_is_oracle=True,
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_feedback_fixture",
                    claim_scope="Artificial private CPU control; no natural learning benefit credit.",
                )
            if not args.validation_worker:
                m.progress("validation_before")
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        CARNOT_8051_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage-health"),
                        PYTHONUNBUFFERED="1",
                        JAX_PLATFORMS="cpu",
                        OPENBLAS_NUM_THREADS="1",
                    ),
                )
                for receipt_row in receipts:
                    receipt_row.update(
                        expected_exit_code=0, actual_exit_code=receipt_row["exit_code"]
                    )
                value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
                value["repository_health"] = [
                    r for r in receipts if r["scope"] == "repository_health"
                ]
                if historical_health is None:
                    atomic_json(
                        health_path,
                        dict(
                            receipts=value["repository_health"],
                            scope="One bounded repository-wide diagnostic; unrelated failures do not qualify or invalidate owned science.",
                        ),
                    )
                else:
                    value["repository_health"] = historical_health["receipts"]
                report = (
                    json.loads((scratch / "coverage.json").read_text())
                    if (scratch / "coverage.json").exists()
                    else dict(files={})
                )
                value["coverage_statement_counts"] = {
                    k: v["summary"] for k, v in report["files"].items()
                }
                passed = (
                    len(value["coverage_statement_counts"]) == len(OWNED)
                    and all(
                        r["missing_lines"] == 0 for r in value["coverage_statement_counts"].values()
                    )
                    and all(r["passed"] for r in value["validation_receipts"])
                )
                value["acceptance_gate_results"]["owned_checks"] = passed
                value["future_label_mutation_results"] = [
                    dict(
                        passed=passed,
                        test_path=TEST,
                        scope="private future-label mutation and CLI checks; not a future safety guarantee",
                    )
                ]
                if not passed and not failures:
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_feedback_checks",
                    )
                value["learning_trajectory_ready_score"] = int(
                    value["verdict_class"] == "null"
                    and all(value["acceptance_gate_results"].values())
                )
                m.progress("validation_after")
            value.update(
                duration_s=time.monotonic() - began,
                phase_spans=[
                    dict(phase="freeze", duration_s=frozen - began),
                    dict(phase="numerical", duration_s=measured - frozen),
                    dict(phase="validation", duration_s=time.monotonic() - measured),
                ],
                terminal_validation_sidecar_path=str(
                    output.parent / "raw" / NAME / "terminal_validation.json"
                ),
            )
            for key in (
                "intended",
                "eligible",
                "completed",
                "excluded",
                "failed",
                "censored",
                "independent",
            ):
                value[key + "_count"] = value["sample_size_budget"][key]
            value["raw_shard_hashes"] = [
                reference(p) for p in sorted(raw.rglob("*")) if p.is_file()
            ]
            history = output.parent / "raw" / NAME / "preflight"
            value["raw_shard_hashes"] += [
                reference(p) for p in sorted(history.rglob("*")) if p.is_file()
            ]
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=m.CONFIG, code=value["code_config_hashes"], raw=value["raw_shard_hashes"]
                )
            )
            value["field_principles"] = {
                k: "Bind this invocation to actual durable evidence. Exposed guard checks and artificial controls give no future safety or deployment credit."
                for k in value
            }
            m.progress("publication_before")
            receipt = publish_primary(output, value, replay if args.validation_worker else terminal)
            post = replay(output) if args.validation_worker else terminal(output)
            readers = reader_receipt(
                TASK,
                output.parent,
                field="learning_trajectory_ready_score",
                expected=value["learning_trajectory_ready_score"],
            )
            atomic_json(
                Path(value["terminal_validation_sidecar_path"]),
                dict(publication=receipt, published=post, readers=readers),
            )
            m.equal(post["passed"] and readers["passed"], True, "terminal_validation")
            m.progress("publication_after")
        return 0
    except (ValueError, KeyError, OSError, TimeoutError) as error:
        print(f"[exp8051] owned_failure {type(error).__name__}: {error}", flush=True)
        return 1
