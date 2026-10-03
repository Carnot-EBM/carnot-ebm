"""REQ-REPORT-8038: seal equal-budget learning for independent retention work."""

from __future__ import annotations

import argparse
from dataclasses import replace
import importlib
import importlib.metadata
import json
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot import experiment_8032_v696_sealed_methods as upstream
from carnot import experiment_8019_v695_eligible_targets as prior
from carnot.experiment_artifacts import artifact_output_root
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import CommandSpec, run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify.evidence_features_7980 import normalized
from carnot.verify import windowed_online_8038 as w

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[2]
NAME = "experiment_8038_v696_windowed_online_learning"
TASK = "exp8038-windowed-online-learning"
SCRIPT = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_windowed_online_8038.py"
OWNED = [f"python/carnot/{NAME}.py", "python/carnot/verify/windowed_online_8038.py", SCRIPT]
MODEL_SPECS: list[str] = []


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Authenticate the frozen protocol and its starting head before any replay."""
    refs: list[Json] = []
    path = root / "results" / (upstream.NAME + ".json")
    try:
        upstream.require(path, "primary_exists", True, path.is_file())
        value = json.loads(path.read_text())
        terminal = json.loads(Path(value["terminal_validation_sidecar_path"]).read_text())
        binding = terminal.get("publication", terminal)
        report = json.loads(Path(binding["sidecar_path"]).read_text())
        for obj in (binding, report):
            upstream.require(path, "primary_sha256", sha256_file(path), obj.get("primary_sha256"))
            upstream.require(path, "primary_path", str(path), obj.get("primary_path"))
        upstream.require(path, "report.passed", True, report.get("report", {}).get("passed"))
        for field, expected in (
            ("experiment_id", 8032),
            ("learning_inputs_ready_score", 1),
            ("flagged_adversarial", False),
            ("verdict_class", "null"),
        ):
            upstream.require(path, field, expected, value.get(field, "MISSING_CONTRACT_FIELD"))
        for p in (
            path,
            Path(value["terminal_validation_sidecar_path"]),
            Path(binding["sidecar_path"]),
        ):
            refs.append(upstream.copy_bound(reference(p), raw))
        protocol_ref = upstream.copy_bound(value["methods_reference"], raw)
        refs.append(protocol_ref)
        protocol = json.loads(checked(protocol_ref).read_text())
        upstream.require(path, "methods", upstream.METHODS, protocol["methods"])
        upstream.require(path, "frozen", True, protocol["frozen"])
        w.progress("qualified_small_head_load_before")
        h = protocol["head"]
        upstream.require(
            path, "head.converged", True, h["converged"] and len(h["parameters"]) == 110
        )
        head = {k: h[k] for k in ("arm", "parameters", "geometry", "calibration")}
        head["decay_scale"] = 1.0
        w.progress("qualified_small_head_load_after")
        public_ref = upstream.copy_bound(protocol["public"]["stream"], raw)
        target_ref = upstream.copy_bound(value["role_manifests"]["evaluator"]["stream"], raw)
        refs += [public_ref, target_ref]
        sources = json.loads(checked(public_ref).read_text())["rows"]
        upstream.require(path, "original_slots", list(range(256)), [r["slot"] for r in sources])
        for r in sources:
            r["source_cluster_id"] = normalized(bytes.fromhex(r["source_bytes"]))
        for ref in value["cited_upstream_artifacts"]:
            refs.append(upstream.copy_bound(ref, raw, "historical"))
        return dict(
            head=head,
            sources=sources,
            target_reference=target_ref,
            seeds=w.CONFIG["seeds"],
            references=refs,
            historical_exposure=value["historical_exposure"],
            gate_checks=[
                dict(
                    upstream_id=value.get("task_id", "exp8032"),
                    path=str(path),
                    sha256=sha256_file(path),
                    artifact_field=field,
                    expected=expected,
                    observed=observed,
                    passed=expected == observed,
                    check_name=field,
                )
                for field, expected, observed in (
                    ("learning_inputs_ready_score", 1, value["learning_inputs_ready_score"]),
                    ("terminal_primary_sha256", sha256_file(path), report["primary_sha256"]),
                    ("report.passed", True, report["report"]["passed"]),
                    (
                        "methods_reference.sha256",
                        value["methods_reference"]["sha256"],
                        protocol_ref["sha256"],
                    ),
                    ("head.converged", True, h["converged"]),
                )
            ],
        ), []
    except upstream.Contract as error:
        return dict(references=refs), [error.gate]
    except (KeyError, ValueError, OSError) as error:
        return dict(references=refs), [
            upstream.Contract(
                path,
                str(error.args[0]) if isinstance(error, KeyError) else "frozen_input_contract",
                "complete byte-bound fields",
                "MISSING_CONTRACT_FIELD" if isinstance(error, KeyError) else str(error),
            ).gate
        ]


def validation_plan(scratch: Path) -> list[CommandSpec]:
    """Reuse bounded commands, limiting coverage to the new producer and CLI."""
    commands = prior.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(
        "[run]\nparallel = True\ndata_file = "
        + str(scratch / ".coverage")
        + "\ninclude =\n    "
        + "\n    ".join(str(ROOT / p) for p in OWNED)
        + "\n"
    )
    return [
        replace(
            c,
            argv=tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv)
            + ((OWNED[1],) if c.name in {"ruff_check", "ruff_format", "strict_mypy"} else ()),
            timeout_s=min(c.timeout_s, 180),
        )
        for c in commands
    ]


def replay(path: Path) -> Json:
    """Cold reduction rejects changed aggregates, code custody or unsafe readiness."""
    value = json.loads(path.read_text())
    for ref in value["raw_shard_hashes"] + value["code_config_hashes"]:
        checked(ref)
    if value["acceptance_gate_results"]["trajectory"]:
        for key, observed in w.reduce(Path(value["trajectory_directory"])).items():
            w.equal(value[key], observed, "reduction_drift:" + key)
    w.equal(
        not value["learning_trajectory_ready_score"]
        or (value["verdict_class"] == "null" and all(value["acceptance_gate_results"].values())),
        True,
        "unsafe_readiness",
    )
    return dict(passed=True, sha256=sha256_file(path))


def terminal(path: Path) -> Json:
    """Fresh processes validate both candidate and published bytes with old readers."""
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


def base(failures: list[Json]) -> Json:
    """Terminal null and blocked records retain the same required field contract."""
    value = dict(
        experiment_id=8038,
        task_id=TASK,
        milestone="2026.10.696",
        schema="carnot.v696.windowed_online_learning.v1",
        run_date="20261002",
        honest_verdict="complete_blocked_windowed_inputs"
        if failures
        else "complete_null_windowed_trajectories",
        verdict_class="blocked" if failures else "null",
        claim_scope="This invocation replays one exposed development stream under equal adaptive budgets. Temporal support changes; generator and importance weights stay fixed. No retention, independent learning benefit or deployment claim.",
        gate_check_summary=failures,
        random_seed=101,
        algorithm_seeds=w.CONFIG["seeds"],
        acceptance_gate_results=dict(trajectory=False, owned_checks=False, controls=False),
        verifier_is_oracle=False,
        genuine_headroom=dict(measured=False, benefit_unassessed=True),
        positive_control_results={},
        generalized_learning_benefit_score=0,
        learning_trajectory_ready_score=0,
        inference_substrate="verifier_scoring",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_specs=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        trained_head_specs=[],
        substrate_declaration=dict(
            custody="aggregation_from_upstream_artifacts",
            numerical="verifier_scoring",
            pretrained="no_model_load",
        ),
        methodology_note="Uniform temporal replay support changes with fixed calibrated BCE geometry, rate 0.01 and ridge 0.001. Issue before release at slot+20; four gradients per 16 eligible labels, cap64; no tail flush or retention selection. CPU and synchronous journal timing are distinct. The 100x arithmetic scenario is prospective only.",
        config=w.CONFIG,
        historical_exposure={},
        future_access_tests=[],
        checkpoint_references=[],
        code_config_hashes=[],
        raw_shard_hashes=[],
        cited_upstream_artifacts=[],
        validation_receipts=[],
        repository_health=[],
        coverage_statement_counts={},
        flagged_adversarial=False,
        phase_spans=[],
        retained_labels_opened=False,
        sample_size_budget=dict(
            intended=256,
            eligible=0,
            started=0,
            completed=0,
            excluded=0,
            failed=0,
            censored=256,
            independent=0,
            seeds_are_independent=False,
            independent_datasets=1,
        ),
    )
    for key in (
        "rows",
        "issued_prediction_rows",
        "feedback_release_rows",
        "replay_buffer_rows",
        "gradient_rows",
        "update_budget_rows",
        "durable_commit_rows",
        "cpu_update_costs",
    ):
        value[key] = []
    return value


def main(argv: list[str] | None = None) -> int:
    """Freeze bytes, measure once, validate, then publish one terminal primary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    started = time.monotonic()
    w.progress("begin_no_pretrained_loads_or_generations")
    try:
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        output = artifact_output_root(root=args.root) / (NAME + ".json")
        raw = output.parent / "raw" / output.stem / "invocations" / str(time.time_ns())
        raw.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="carnot-8038-") as temporary:
            scratch = Path(temporary)
            commands = validation_plan(scratch)
            w.progress("input_contracts_before")
            data, failures = (
                (json.loads(args.fixture_input.read_text()), [])
                if args.fixture_input
                else load_inputs(args.root, raw)
            )
            value = base(failures)
            value["gate_check_summary"] = failures or data.get("gate_checks", [])
            value["cited_upstream_artifacts"] = data.get("references", [])
            value["historical_exposure"] = data.get(
                "historical_exposure", dict(artificial=bool(args.fixture_input))
            )
            dependencies = [
                Path(w.old.__file__),
                Path(w.sealed.__file__),
                Path(w.old.conditioned.__file__),
            ]
            dependencies += [
                Path(importlib.import_module(name).__file__)
                for name in (
                    "carnot.verify.sparse_energy_7996",
                    "carnot.experiment_8021_v695_typed_decision_test",
                    "carnot.experiment_8019_v695_eligible_targets",
                    "carnot.reporting.current_work_receipt",
                    "carnot.reporting.primary_publication",
                    "carnot.reporting.experiment_7303_validation_scope",
                )
            ]
            value["code_config_hashes"] = [
                upstream.copy_bound(reference(p), raw, "code")
                for p in [*(ROOT / p for p in OWNED + [TEST]), *dependencies]
            ]
            atomic_json(
                raw / "configuration.json",
                dict(
                    config=w.CONFIG,
                    code=value["code_config_hashes"],
                    commands=[
                        dict(name=c.name, argv=c.argv, timeout_s=c.timeout_s) for c in commands
                    ],
                    numerical_acceptance="parity<=1e-10; equal actual adaptive gradients; no future-label influence; exact journal replay",
                    data_access="due stream labels only; retention unopened",
                    input_references=data.get("references", []),
                    artifact_guard_enabled=True,
                    library_versions={
                        name: importlib.metadata.version(name) for name in ("numpy", "scipy")
                    },
                ),
            )
            frozen = time.monotonic()
            w.progress("methods_code_budgets_identity_frozen")
            if not failures:
                value.update(w.measure(data, raw / "trajectory"))
                value["trajectory_directory"] = str(raw / "trajectory")
                value["acceptance_gate_results"]["trajectory"] = True
                value["positive_control_results"] = w.controls()
                value["acceptance_gate_results"]["controls"] = value["positive_control_results"][
                    "passed"
                ]
                value["trained_head_specs"] = [
                    dict(
                        arm=r["arm"],
                        seed=r["seed"],
                        parameters=len(data["head"]["parameters"]),
                        device="cpu",
                        pretrained=False,
                        actual_updates=r["actual_gradient_count"],
                    )
                    for r in value["update_budget_rows"]
                ]
            measured = time.monotonic()
            if args.fixture_input:
                value.update(
                    verifier_is_oracle=True,
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_windowed_fixture",
                    claim_scope="Artificial CPU causal CLI fixture; no natural learning credit.",
                )
            if not args.validation_worker:
                w.progress("owned_validation_before")
                receipts = run_commands(
                    ROOT,
                    commands,
                    log_dir=raw / "validation_logs",
                    heartbeat_s=30,
                    extra_env=dict(
                        CARNOT_8038_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                        COVERAGE_FILE=str(scratch / ".coverage-health"),
                        JAX_PLATFORMS="cpu",
                        OPENBLAS_NUM_THREADS="1",
                    ),
                )
                value["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
                value["repository_health"] = [
                    r for r in receipts if r["scope"] == "repository_health"
                ]
                report = (
                    json.loads((scratch / "coverage.json").read_text())
                    if (scratch / "coverage.json").exists()
                    else dict(files={})
                )
                counts = {k: v["summary"] for k, v in report["files"].items()}
                value["coverage_statement_counts"] = counts
                passed = (
                    len(counts) == len(OWNED)
                    and all(r["missing_lines"] == 0 for r in counts.values())
                    and all(r["passed"] for r in value["validation_receipts"])
                )
                value["acceptance_gate_results"]["owned_checks"] = passed
                value["future_access_tests"] = [
                    dict(
                        check_name="future_label_mutation_and_private_cli",
                        passed=passed,
                        test_path=TEST,
                    )
                ]
                if not passed and not failures:
                    value.update(
                        verdict_class="disqualified",
                        honest_verdict="complete_disqualified_windowed_checks",
                    )
                value["learning_trajectory_ready_score"] = int(
                    value["verdict_class"] == "null"
                    and all(value["acceptance_gate_results"].values())
                )
                w.progress("owned_validation_after")
            value.update(
                duration_s=time.monotonic() - started,
                phase_spans=[
                    dict(phase="freeze", duration_s=frozen - started),
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
            value["reproducibility_checksum"] = canonical_hash(
                dict(
                    config=w.CONFIG, code=value["code_config_hashes"], raw=value["raw_shard_hashes"]
                )
            )
            value["field_principles"] = {
                k: "Bind this invocation to its actual evidence and durable bytes; imported development and synthetic controls give no deployment credit."
                for k in value
            }
            value["field_principles"].update(
                learning_trajectory_ready_score="One requires valid sealed trajectories and all owned checks, independently of benefit.",
                sample_size_budget="Count original source groups once; seeds and replayed IDs add no independent samples.",
                cpu_update_costs="Measure CPU updates, written bytes and defined active operations; transaction wall time is separate.",
                substrate_declaration="No current pretrained calls; custody and CPU small-head arithmetic have separate scopes.",
                gradient_rows="Use only eligible released labels, preserve issued losses and count every reused ID.",
                honest_verdict="Owned work is terminal; missing external prerequisites are blocked rather than partial.",
            )
            w.progress("publication_before")
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
            w.equal(post["passed"] and readers["passed"], True, "terminal_validation")
            w.progress("publication_after")
        return 0
    except (ValueError, KeyError, OSError, TimeoutError) as error:
        print(f"[exp8038] failed={error}", flush=True)
        return 1
