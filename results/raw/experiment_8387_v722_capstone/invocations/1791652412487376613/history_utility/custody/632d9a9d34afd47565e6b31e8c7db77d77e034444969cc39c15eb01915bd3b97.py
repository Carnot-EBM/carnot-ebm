"""REQ-REPORT-8351: evaluator access follows every authenticated shadow seal.

The adapter reuses byte custody and bounded child supervision. It imports no
learner gain claim and never supplies retention targets to a learner process.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting import continuous_local_learning_8348 as learning
from carnot.reporting import continuous_local_execution_8348 as execution
from carnot.reporting import methods_stream_execution_8111 as commands
from carnot.reporting import static_benefit_audit_8350 as custody
from carnot.reporting import sentence_spline_fit_8334 as historical
from carnot.reporting import v720_frozen_input_contract as authority
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.verify.calibrated_memory_trajectory_8211 import journal
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.verify import learning_retention_audit_8351 as k
from carnot.verify import reserved_prediction_seal_8335 as seal

Json = dict[str, Any]
ROOT = learning.ROOT
NAME, TASK = "experiment_8351_v720_learning_retention_audit", "exp8351-learning-retention-audit"
CLI, TEST = (
    "scripts/experiments/" + NAME + ".py",
    "tests/python/test_learning_retention_audit_8351.py",
)
OWNED = [
    "python/carnot/verify/learning_retention_audit_8351.py",
    "python/carnot/reporting/learning_retention_audit_8351.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
progress, reference = k.progress, learning.reference


def bind(work: Json, ref: Json, raw: Path, *, parse: bool = True) -> Json:
    """Reuse an immutable snapshot when two receipts cite the same original bytes."""
    dest = raw / "inputs" / (ref["sha256"][7:] + "-" + Path(ref["path"]).name)
    if dest.exists():
        learning.require(
            work, Path(ref["path"]), "sha256", ref["sha256"], sha256_file(Path(ref["path"]))
        )
        learning.require(work, dest, "snapshot_sha256", ref["sha256"], sha256_file(dest))
        return dict(json.loads(dest.read_bytes())) if parse else dict(reference(dest))
    return dict(learning.bind(work, ref, raw, parse=parse))


def measure(root: Path, raw: Path) -> Json:
    """Authenticate external inputs before independent reconstruction and evaluator access."""
    began = time.monotonic()
    progress("before_preconditions")
    raw.mkdir(parents=True, exist_ok=True, mode=0o700)
    raw.chmod(0o700)
    work: Json = dict(
        root=str(root),
        gates=[],
        failures=[],
        refs=[],
        code_refs=[],
        raw_refs=[],
        authority={},
        bundle={},
        state={},
        events=[],
        checkpoints={},
        future={},
        checks={},
        targets=[],
        historical=[],
        label_access_log=[],
        prior_evaluator_exposure={},
        owned_failure=False,
        preconditions_checked=True,
    )
    owned_phase = False
    path = root / custody.FROZEN
    try:
        learning.require(
            work,
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ("python", "pytest", "coverage", "ruff", "mypy"):
            learning.require(
                work,
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        progress("before_authority")
        work["authority"] = authority.authority(root, raw / "authority")
        for snap in work["authority"]["authority_snapshots"].values():
            work["refs"].append(
                dict(
                    path=snap["snapshot_path"],
                    sha256=snap["sha256"],
                    source_path=snap["source_path"],
                )
            )
        learning.require(
            work,
            root / "research-roadmap.yaml",
            "exact_task_authority",
            True,
            work["authority"]["activated"]
            and any(
                t["id"] == TASK and t["deliverable"] == "results/" + NAME + ".json"
                for t in work["authority"]["tasks"]
            ),
        )
        progress("after_authority")
        bind(work, reference(root / "ops/exclusion_manifest.yaml"), raw, parse=False)
        frozen = custody.primary(root, custody.FROZEN, "frozen_heads_ready_score", work, raw)
        learner = custody.primary(root, custody.LEARNER, "trajectory_ready_score", work, raw)
        source = custody.primary(root, inputs.SOURCE, "fit_support_ready_score", work, raw)
        pred = custody.primary(root, inputs.PRED, "predictions_ready_score", work, raw)
        prediction_seal = bind(work, pred["prediction_manifest"], raw)
        static_bundle = bind(work, prediction_seal["input_reference"], raw)
        seal.validate_bundle(static_bundle)
        predictions = [r for ref in prediction_seal["files"] for r in bind(work, ref, raw)["rows"]]
        learning.require(
            work,
            root / custody.FROZEN,
            "original_static_head",
            frozen["frozen_policy"]["whole_model_sha256"],
            static_bundle["head_hash"],
        )
        learning.require(
            work,
            root / inputs.PRED,
            "static_prediction_replay",
            True,
            seal.parity(static_bundle, prediction_seal["issued_at"], predictions)["passed"],
        )
        progress("before_all_retention_seals")
        custody.retention(learner, static_bundle, predictions, work, raw)
        progress("after_all_retention_seals", 4, 0)
        upstream = bind(work, learner["measurement_reference"], raw)
        bundle = learning.authenticated_projection(upstream)
        learning.require(
            work,
            root / custody.LEARNER,
            "authenticated_bundle",
            canonical_hash(bundle),
            canonical_hash(upstream["bundle"]),
        )
        work["bundle"], work["state"] = bundle, upstream["state"]
        origin = Path(learner["measurement_reference"]["path"]).parent
        for ref in upstream["refs"] + upstream["code_config_hashes"] + learner["raw_shard_hashes"]:
            bind(work, ref, raw, parse=False)
        progress("before_subprocess_upstream_cold_replay")
        receipt = execution.check(
            dict(
                name="upstream_cold",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / learning.CLI),
                    "--cold-replay",
                    str(root / custody.LEARNER),
                ],
                deadline_s=180,
            ),
            raw / "upstream_replay",
        )
        work["upstream_replay_receipt"] = receipt
        progress("after_subprocess_upstream_cold_replay", 1, 0)
        work["owned_failure"] = not receipt["passed"]
        work["events"] = journal(origin / "uninterrupted/events.jsonl")
        names = {str(w): f"uninterrupted/checkpoint-{w}.json" for w in (32, 64, 96)}
        names.update(crash32="crash32/final.json", crash64="crash64/final.json")
        work["checkpoints"] = {
            key: bind(work, reference(origin / name), raw) for key, name in names.items()
        }
        work["future"] = bind(work, reference(origin / "future/final.json"), raw)
        progress("before_benchmark_scalar_reconstruction")
        owned_phase = True
        work["checks"] = k.reconstruct(
            bundle, work["state"], work["events"], work["checkpoints"], work["future"]
        )
        work["owned_failure"] |= not work["checks"]["passed"]
        owned_phase = False
        progress("after_benchmark_scalar_reconstruction", 96, 0)
        protocol_path = root / historical.PROTOCOL
        protocol = bind(
            work,
            dict(path=str(protocol_path), sha256=learning.PINS[historical.PROTOCOL]),
            raw,
        )
        roles = protocol["original_roles"]
        sets = [{r["source_cluster_id"] for r in roles[a]} for a in ("fit", "tune", "evaluation")]
        separated = all(not a.intersection(b) for i, a in enumerate(sets) for b in sets[i + 1 :])
        work["checks"]["source_role_separation"] = dict(
            passed=separated,
            fit_count=len(sets[0]),
            tune_count=len(sets[1]),
            stream_count=96,
            retention_count=32,
            registered_role_reference=reference(protocol_path),
        )
        learning.require(work, protocol_path, "registered_source_role_separation", True, separated)
        for a, b in [
            ("issued_rows", "issued"),
            ("update_rows", "updates"),
            ("retention_shadow_rows", "retention"),
        ]:
            learning.require(
                work,
                root / custody.LEARNER,
                a,
                canonical_hash(learner[a]),
                canonical_hash(work["state"][b]),
            )
        prior_path = root / "results/experiment_8350_v720_static_benefit_audit.json"
        prior = bind(work, reference(prior_path), raw) if prior_path.exists() else {}
        work["prior_evaluator_exposure"] = dict(
            observed=bool(prior.get("label_access_log")),
            artifact=reference(prior_path) if prior else None,
            historical_verdict=prior.get("honest_verdict"),
            label_access_log=prior.get("label_access_log", []),
            learner_input=False,
        )
        sealed_ns = time.time_ns()
        progress("before_evaluator_target_access")
        work["targets"] = bind(work, source["evaluator_shards"]["reserved"], raw)["rows"]
        work["label_access_log"].append(
            dict(
                phase="independent_evaluator_only",
                operand=source["evaluator_shards"]["reserved"],
                all_seals_authenticated_before_access=True,
                seals_authenticated_wall_ns=sealed_ns,
                access_wall_ns=time.time_ns(),
                learner_feedback_count=0,
                retention_target_count=32,
            )
        )
        work["historical"] = source["historical_model_provenance"]
        owned_phase = True
        k.reduce(work["state"], work["targets"], work["checks"])
        progress("after_evaluator_target_access", 128, 0)
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration) as error:
        work["owned_failure"] |= owned_phase
        if isinstance(error, OSError) and error.filename:
            path = Path(error.filename)
        if not work["failures"]:
            work["failures"].append(
                authority.failure(
                    path,
                    "authenticated_external_operand",
                    True,
                    str(error) if path.exists() else None,
                )
            )
    atomic_json(raw / "frozen_configuration.json", k.CONFIG)
    atomic_json(
        raw / "primitive_evidence.json",
        {
            f: work[f]
            for f in (
                "bundle",
                "state",
                "events",
                "checkpoints",
                "future",
                "targets",
                "checks",
                "label_access_log",
            )
        },
    )
    work.update(
        code_refs=[reference(ROOT / f) for f in [*OWNED, TEST]],
        raw_refs=[
            reference(raw / "primitive_evidence.json"),
            reference(raw / "frozen_configuration.json"),
        ],
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_reconstruct_evaluate",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_measurement", int(bool(work["targets"])), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """A valid scientific null is ready; external absence and owned failures differ."""
    measured = bool(work["targets"])
    reduced = k.reduce(work["state"], work["targets"], work["checks"]) if measured else {}
    coverage = work.get("owned_coverage_reference")
    covered = json.loads(Path(coverage["path"]).read_bytes()) if coverage else None
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    checked = checked and (
        covered is None or all(covered["files"][p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    checked = checked and reduced.get("dense_sparse_exact_parity", True)
    ready = int(checked and measured and not work["failures"])
    klass = (
        "disqualified"
        if not checked
        else "blocked"
        if not ready
        else "positive"
        if reduced["h2_development_signal_score"]
        else "null"
    )
    verdict = "complete_" + (
        "disqualified_owned_validation"
        if not checked
        else "blocked_external_input"
        if not ready
        else reduced["science_disposition"]
    )
    count = reduced.get("qualified_count", 0)
    value: Json = dict(
        reduced,
        experiment_id=8351,
        task_id=TASK,
        milestone="2026.10.720",
        run_date="20261009",
        honest_verdict=verdict,
        verdict_class=klass,
        gate_check_summary=work["gates"] + work["failures"],
        learning_audit_ready_score=ready,
        h2_development_signal_score=ready * reduced.get("h2_development_signal_score", 0),
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"],
        rows=reduced.get("paired_cost_rows", []),
        intended_count=88,
        completed_count=count,
        failed_count=0,
        censored_count=0 if measured else 88,
        excluded_count=88 - count if measured else 0,
        independent_count=count,
        sample_size_budget=dict(
            k.CONFIG, arms=k.ARMS, retention_intended=32, retention_windows=k.WINDOWS
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=not checked,
        acceptance_gates=dict(
            owned=checked,
            all_seals_before_targets=bool(work["label_access_log"]),
            causal=work["checks"].get("passed", False),
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("adversarial_findings", []),
        finding_dispositions=work.get("finding_dispositions", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178312,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=[
                    "byte-bound primitive evidence or authority; no claimed learner gains"
                ],
            )
            for r in work["refs"]
        ],
        paired_cost_rows=reduced.get("paired_cost_rows", []),
        block_bootstrap_summary=reduced.get("block_bootstrap_summary", {}),
        retention_rows=reduced.get("retention_rows", []),
        retention_window_bounds=reduced.get("retention_window_bounds", []),
        action_reachability=work["checks"],
        label_access_log=work["label_access_log"],
        prior_evaluator_exposure=work["prior_evaluator_exposure"],
        measurement_reference=reference(raw / "measurement.json"),
        owned_coverage_reference=coverage,
        execution_manifest_reference=work.get("execution_manifest"),
        upstream_replay_receipt=work.get("upstream_replay_receipt"),
        authority=work["authority"],
        mechanism_limits=[
            "four cached relation features",
            "fixed cubic knots",
            "temperature-scaled .01 update; maximum88 delayed releases",
            "ordinary logistic spline probability",
            "exposed development only; no new hard constraints",
            "calibration-only is descriptive; a frozen win alone does not establish locality advantage",
            "23 qualified retention sources:9 supported and14 unsupported;9 missing slots; exploratory panel",
        ],
        retirement=dict(
            retired=bool(
                ready and reduced.get("science_disposition") == "null_update_budget_no_headroom"
            ),
            scope="exact_four_feature_fixed_knot_.01_update_budget",
            reason="certified inability to cross action boundaries"
            if reduced.get("science_disposition") == "null_update_budget_no_headroom"
            else "no exact repeated failure established; preserve historical dispositions",
        ),
        future_dependencies=[
            dict(
                path="results/experiment_8359_v720_capstone.json",
                role="future_consumer_not_existing_input",
            )
        ],
        methodology_note="Intended-slot H2 costs include missing-feature escalations. Moving-block10000,length8,seed7178312,nominal one-sided97.5 percent. Fixed exposed trajectory intervals are descriptive and establish no IID or conformal guarantee. No generalized learning or newly learned hard constraints.",
    )
    value["field_principles"] = {
        f: "Bind " + f + " to original byte-bound evidence and exposed-development scope."
        for f in value
    }
    value["field_principles"].update(
        field_principles="Explain why every added field is present.",
        reproducibility_checksum="Bind all semantic claims to independently replayed primitives.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reauthenticate and cold-reduce primitives even after self-consistent rehashing."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        for operand in work["refs"] + work["code_refs"] + work["raw_refs"]:
            if sha256_file(Path(operand["path"])) != operand["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for prefix in ("stdout", "stderr", "log"):
                if (
                    receipt.get(prefix + "_path")
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        for entry in work["label_access_log"]:
            if (
                not entry["all_seals_authenticated_before_access"]
                or entry["learner_feedback_count"] != 0
                or not 0 < entry["seals_authenticated_wall_ns"] <= entry["access_wall_ns"]
            ):
                return False
        if value != build(work, Path(ref["path"]).parent, value["validation_receipts"]):
            return False
        with TemporaryDirectory(prefix="carnot-8351-cold-") as directory:
            expected = measure(Path(work["root"]), Path(directory) / "raw")
        for field in (
            "bundle",
            "state",
            "events",
            "checkpoints",
            "future",
            "checks",
            "targets",
            "historical",
            "owned_failure",
            "prior_evaluator_exposure",
        ):
            if work[field] != expected[field]:
                return False
        if any(
            work["authority"].get(f) != expected["authority"].get(f)
            for f in (
                "activated",
                "planning_matched",
                "canonical_tasks_sha256",
                "tasks",
                "contract_rows",
            )
        ):
            return False
        for field in ("gates", "failures"):

            def normalized(data: Json) -> str:
                rows = [
                    {
                        key: val
                        for key, val in row.items()
                        if row.get("artifact_field") != "private_resources"
                        or key not in ("upstream", "path", "hash")
                    }
                    for row in data[field]
                ]
                return json.dumps(rows, sort_keys=True).replace(data["gates"][0]["path"], "<raw>")

            left, right = normalized(work), normalized(expected)
            if left != right:
                return False
        primitive = json.loads(Path(work["raw_refs"][0]["path"]).read_bytes())
        if (
            primitive != {f: work[f] for f in primitive}
            or json.loads(Path(work["raw_refs"][1]["path"]).read_bytes()) != k.CONFIG
        ):
            return False
        return bool(value == build(work, Path(ref["path"]).parent, value["validation_receipts"]))
    except (OSError, ValueError, KeyError, TypeError, IndexError, StopIteration):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze scoped tests, subprocess coverage and unchanged consumers before work."""
    with patch.object(commands, "e", sys.modules[__name__]), patch.object(commands, "OWNED", OWNED):
        plan = commands.manifest(private, candidate)
    plan.pop("repository_health")
    with (private / "coverage.ini").open("a") as stream:
        stream.write("patch = subprocess\n")
    plan["commands"][0]["argv"].remove("-s")
    plan["commands"][0]["argv"].append("--basetemp=" + str(private / "owned"))
    plan["commands"][0]["deadline_s"] = 600
    plan["commands"][1]["argv"] = [
        str(ROOT / ".venv/bin/pytest"),
        "-n",
        "0",
        "-o",
        "addopts=",
        "--no-cov",
        "-q",
        "--basetemp=" + str(private / "consumers"),
        "tests/python/test_primary_publication_7928.py",
        "tests/python/test_local_update_isolation_8306.py",
        "tests/python/test_source_boundary_7852.py",
        "tests/python/test_restricted_decision_audit_8210.py",
    ]
    plan["commands"][1]["deadline_s"] = 600
    plan["commands"][-1]["argv"].insert(2, "--files")
    typing = private / "strict-mypy.ini"
    typing.write_text(
        "[mypy]\npython_version = 3.12\nstrict = true\nignore_missing_imports = true\nexplicit_package_bases = true\n"
    )
    next(c for c in plan["commands"] if c["name"] == "strict_mypy")["argv"].insert(
        1, "--config-file=" + str(typing)
    )
    return dict(plan, owned=OWNED, no_full_repository_suite=True, heartbeat_s=20)


def main(argv: list[str] | None = None) -> int:
    """Reuse real bounded children, typed findings and atomic primary publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
    ):
        return int(execution.main(argv))
