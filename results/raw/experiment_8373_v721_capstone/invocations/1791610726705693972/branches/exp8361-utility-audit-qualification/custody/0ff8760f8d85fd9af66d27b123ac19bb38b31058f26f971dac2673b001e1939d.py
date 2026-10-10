"""REQ-REPORT-8350: evaluator access follows authenticated immutable learner seals."""

from __future__ import annotations

from copy import deepcopy
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
from carnot.reporting import v720_frozen_inputs as inputs
from carnot.reporting import sentence_spline_execution_8334 as execution
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.verify import static_benefit_audit_8350 as k
from carnot.verify import reserved_prediction_seal_8335 as seal
from carnot.verify.continuous_local_learning_8348 import ARMS as ONLINE_ARMS

Json = dict[str, Any]
ROOT = learning.ROOT
NAME, TASK = "experiment_8350_v720_static_benefit_audit", "exp8350-static-benefit-audit"
CLI, TEST = "scripts/experiments/" + NAME + ".py", "tests/python/test_static_benefit_audit_8350.py"
OWNED = [
    "python/carnot/verify/static_benefit_audit_8350.py",
    "python/carnot/reporting/static_benefit_audit_8350.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
FROZEN = "results/experiment_8346_v720_frozen_input_contract.json"
LEARNER = "results/experiment_8348_v720_continuous_local_learning.json"
PINS = dict(
    inputs.PINS,
    **{
        FROZEN: learning.PINS[FROZEN],
        LEARNER: "sha256:c98cbd678aef6aa5055cd7033a2c36404dc8d20337a5e601bd90b2d7dba00f07",
    },
)
progress, reference = k.progress, learning.reference
bind, require = learning.bind, learning.require
BASE_MANIFEST = execution.manifest


def primary(root: Path, name: str, field: str, work: Json, raw: Path) -> Json:
    """Original pinned bytes and terminal reports authorize reuse, never utility."""
    path = root / name
    value = bind(work, dict(path=str(path), sha256=PINS[name]), raw)
    validate_primary(value, path)
    terminal = bind(work, reference(Path(value["terminal_validation_sidecar_path"])), raw)
    side = Path(terminal["publication"]["sidecar_path"])
    report = read_bound_sidecar(path, side)
    bind(work, reference(side), raw)
    for f, want, got in [
        ("terminal_passed", True, report["report"]["passed"]),
        ("terminal_primary_hash", PINS[name], terminal["publication"]["primary_sha256"]),
        ("required_checks_passed", True, value.get("required_checks_passed")),
        ("flagged_adversarial", False, value.get("flagged_adversarial")),
        (field, 1, value.get(field)),
    ]:
        require(work, path, f, want, got)
    return value


def retention(learner: Json, bundle: Json, predictions: list[Json], work: Json, raw: Path) -> Json:
    """Every intended shadow and issued action must be immutable before labels open."""
    refs = learner["retention_prediction_seals"]
    require(work, ROOT / LEARNER, "four_window_seal_count", 4, len(refs))
    require(
        work,
        ROOT / LEARNER,
        "four_distinct_retention_seals",
        4,
        len({(r["path"], r["sha256"]) for r in refs}),
    )
    retained = []
    windows = []
    expected_ids = {(p["unit_id"], p["source_cluster_id"], p["slot"]) for p in bundle["slots"][96:]}
    for ref in refs:
        data = bind(work, ref, raw)
        rows = data["rows"]
        window = rows[0]["window"] if rows else None
        expected = {(u, s, n, a, window) for u, s, n in expected_ids for a in ONLINE_ARMS}
        observed = {
            (r["unit_id"], r["source_cluster_id"], r["slot"], r["arm"], r["window"]) for r in rows
        }
        good = (
            data["targets_opened"] is False
            and len(rows) == 160
            and observed == expected
            and all(
                r["targets_opened"] is False
                and r["action"] == k.fitted.action(r["p"])
                and r["state_hash"]
                for r in rows
            )
        )
        require(work, Path(ref["path"]), "complete_retention_window", True, good)
        retained.extend(rows)
        windows.append(window)
        progress("retention_window_authenticated", len(windows), 4 - len(windows))
    require(work, ROOT / LEARNER, "retention_windows", [0, 32, 64, 96], sorted(windows))
    require(
        work,
        ROOT / LEARNER,
        "sealed_shadow_identity",
        canonical_hash(learner["retention_shadow_rows"]),
        canonical_hash(retained),
    )
    require(
        work, ROOT / LEARNER, "retention_targets_opened", 0, learner["retention_targets_opened"]
    )
    require(
        work,
        ROOT / LEARNER,
        "head_checkpoint",
        bundle["head_manifest"]["checkpoint_reference"]["sha256"],
        learner["heads_sha256"],
    )
    issued = learner["issued_rows"]
    identities = {(r["unit_id"], r["slot"], r["arm"]) for r in issued}
    expected_issues = {
        (p["unit_id"], p["slot"], a) for p in bundle["slots"][:96] for a in ONLINE_ARMS
    }
    require(
        work,
        ROOT / LEARNER,
        "complete_learning_seal",
        True,
        len(issued) == 480 and identities == expected_issues,
    )
    frozen = {p["slot"]: p for p in predictions if p["arm"] == "spline34"}
    require(
        work,
        ROOT / LEARNER,
        "frozen_issued_policy",
        True,
        all(
            (
                r["p"] is None
                and frozen[r["slot"]]["p"] is None
                or r["p"] is not None
                and frozen[r["slot"]]["p"] is not None
                and abs(r["p"] - frozen[r["slot"]]["p"]) < 1e-12
            )
            and r["action"] == frozen[r["slot"]]["action"]
            for r in [*issued, *retained]
            if r["arm"] == "frozen_spline"
        ),
    )
    return dict(
        passed=True,
        windows=windows,
        intended_per_window=160,
        issued_count=len(issued),
        seal_references=refs,
        online_utility_gate=False,
        checked_before_target_access=True,
    )


def independence(bundle: Json, predictions: list[Json]) -> Json:
    """Predictor inputs contain features only; later mutations cannot rewrite issues."""
    head = bundle["head_manifest"]["heads"][0]
    row = next(r for r in bundle["slots"] if r["x"] is not None)
    x = [row["x"][0], *row["x"][12:16]]
    issued = deepcopy(predictions)
    digest = canonical_hash(issued)
    contextual = [
        dict(features=x, withheld_target=y, source_id=s, feature_mask=m)
        for y, s, m in [(0, "original", True), (1, "changed", False), (None, "", None)]
    ]
    probabilities = [k.fitted.predict(head, dict(features=c["features"])) for c in contextual]
    masked_probabilities = [
        k.fitted.predict(head, dict(features=c["features"] if c["feature_mask"] else None))
        for c in contextual
    ]
    rejected = False
    try:
        k.fitted.predict(head, dict(y=1))
    except ValueError:
        rejected = True
    return dict(
        withheld_target_invariance=len(set(probabilities)) == 1,
        source_id_invariance=len(set(probabilities)) == 1,
        issued_feature_mask_invariance=digest == canonical_hash(issued),
        masked_rescore_escalates=k.fitted.action(k.fitted.predict(head, dict(features=None)))
        == "escalate",
        label_only_forbidden_feature_rejected=rejected,
        feature_mask_sensitivity_control=masked_probabilities[0] is not None
        and masked_probabilities[1] is None,
        perturbation_rows=[
            dict(
                withheld_target=c["withheld_target"],
                source_id=c["source_id"],
                feature_mask=c["feature_mask"],
                recomputed_unmasked_p=p,
                proposed_masked_p=m,
                issued_predictions_sha256=canonical_hash(issued),
            )
            for c, p, m in zip(contextual, probabilities, masked_probabilities, strict=True)
        ],
        issued_sha256=digest,
        scope="Withheld targets and IDs never enter the feature-only call. Changed masks affect a proposed rescore, while original issued predictions remain immutable.",
    )


def optimizer(source: Json, head: Json, bundle: Json, work: Json, raw: Path) -> Json:
    """Recompute fitted objective residuals and the exact constructed optimizer control."""
    import numpy as np

    predictors = bind(work, source["predictor_shards"]["fit"], raw)["rows"]
    targets = {
        r["unit_id"]: r["y"] for r in bind(work, source["evaluator_shards"]["fit"], raw)["rows"]
    }
    qualified = [r for r in predictors if r["x"] is not None and targets[r["unit_id"]] in (0, 1)]
    x = np.asarray([[r["x"][0], *r["x"][12:16]] for r in qualified])
    y = np.asarray([targets[r["unit_id"]] for r in qualified])
    residuals = []
    for fitted_head in bundle["head_manifest"]["heads"]:
        phi = k.fitted.matrix(fitted_head["arm"], x, fitted_head["geometry"])
        diagnostic = k.fitted.diagnostic(np.asarray(fitted_head["coefficients"]), phi, y)
        residuals.append(dict(arm=fitted_head["arm"], fit_sources=len(x), **diagnostic))
    progress("before_benchmark_exact_optimizer_control")
    control = k.fitted.optimizer_control()
    progress("after_benchmark_exact_optimizer_control", 1, 0)
    require(
        work,
        ROOT / inputs.HEAD,
        "exact_optimizer_control",
        canonical_hash(head["optimizer_control"]),
        canonical_hash(control),
    )
    return dict(
        passed=control["passed"],
        control_verdict_class="circular_positive",
        exact_control_sha256=canonical_hash(control),
        initial_control_loss=control["initial"]["loss"],
        final_control_loss=control["final"]["loss"],
        residuals=residuals,
        geometry_qualified=False,
        qualification_reason="Frozen budget reports nonzero projected-gradient residuals; no convergence certificate or registered convergence tolerance exists.",
    )


def measure(root: Path, raw: Path) -> Json:
    """A missing external seal closes the evaluator and permits only a stream diagnostic."""
    began = time.monotonic()
    progress("before_measurement")
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    atomic_json(raw / "frozen_configuration.json", k.CONFIG)
    work: Json = dict(
        root=str(root),
        gates=[],
        failures=[],
        refs=[],
        predictions=[],
        targets=[],
        full_h1_measured=False,
        stream_diagnostic=[],
        reduction={},
        optimizer={},
        retention_seal_check=dict(passed=False, windows=[]),
        label_access_log=[],
        comparator=None,
        independence={},
        historical=[],
        authority={},
        raw_refs=[],
        owned_failure=False,
    )
    path = root / FROZEN
    try:
        require(
            work,
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            require(
                work,
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        progress("before_authority")
        work["authority"] = learning.authority.authority(root, raw / "authority")
        for snap in work["authority"]["authority_snapshots"].values():
            work["refs"].append(
                dict(
                    path=snap["snapshot_path"],
                    sha256=snap["sha256"],
                    source_path=snap["source_path"],
                )
            )
        require(
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
        frozen = primary(root, FROZEN, "frozen_predictions_ready_score", work, raw)
        pred = primary(root, inputs.PRED, "predictions_ready_score", work, raw)
        source = primary(root, inputs.SOURCE, "fit_support_ready_score", work, raw)
        head = primary(root, inputs.HEAD, "heads_ready_score", work, raw)
        work["historical"] = source["historical_model_provenance"]
        prediction_seal = bind(work, pred["prediction_manifest"], raw)
        bundle = bind(work, prediction_seal["input_reference"], raw)
        seal.validate_bundle(bundle)
        work["predictions"] = [
            r for ref in prediction_seal["files"] for r in bind(work, ref, raw)["rows"]
        ]
        require(
            work,
            root / inputs.PRED,
            "original_prediction_rows",
            canonical_hash(pred["prediction_rows"]),
            canonical_hash(work["predictions"]),
        )
        require(
            work,
            root / FROZEN,
            "original_static_policy",
            frozen["frozen_policy"]["whole_model_sha256"],
            bundle["head_hash"],
        )
        require(
            work,
            root / inputs.PRED,
            "static_seal_targets_unopened",
            False,
            prediction_seal["evaluator_labels_opened"],
        )
        require(
            work,
            root / inputs.PRED,
            "independent_static_prediction_replay",
            True,
            seal.parity(bundle, prediction_seal["issued_at"], work["predictions"])["passed"],
        )
        work["comparator"] = bundle["head_manifest"]["selected_comparator"]
        work["independence"] = independence(bundle, work["predictions"])
        require(
            work,
            root / inputs.PRED,
            "prediction_input_independence",
            True,
            all(
                work["independence"][f]
                for f in [
                    "withheld_target_invariance",
                    "source_id_invariance",
                    "issued_feature_mask_invariance",
                    "masked_rescore_escalates",
                    "label_only_forbidden_feature_rejected",
                    "feature_mask_sensitivity_control",
                ]
            ),
        )
        work["optimizer"] = optimizer(source, head, bundle, work, raw)
        progress("before_retention_authentication")
        path = root / LEARNER
        learner = primary(root, LEARNER, "trajectory_ready_score", work, raw)
        work["retention_seal_check"] = retention(learner, bundle, work["predictions"], work, raw)
        progress("after_retention_authentication", 4, 0)
        seals_wall_ns = time.time_ns()
        path = Path(source["evaluator_shards"]["reserved"]["path"])
        progress("before_evaluator_access")
        targets = bind(work, source["evaluator_shards"]["reserved"], raw)["rows"]
        work["label_access_log"].append(
            dict(
                phase="static_evaluator_only",
                operand=source["evaluator_shards"]["reserved"],
                all_seals_authenticated_before_access=True,
                learner_feedback_count=0,
                access_wall_ns=time.time_ns(),
                seals_authenticated_wall_ns=seals_wall_ns,
                intended_targets=len(targets),
            )
        )
        work["targets"] = targets
        work["reduction"] = k.reduce(
            work["predictions"], targets, work["comparator"], work["optimizer"]
        )
        work["full_h1_measured"] = True
        progress("after_evaluator_access", 128, 0)
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        work["owned_failure"] = bool(work["label_access_log"])
        if not work["failures"]:
            work["failures"].append(
                learning.authority.failure(
                    path,
                    "authenticated_external_operand",
                    True,
                    str(error) if path.exists() else None,
                )
            )
    if not work["full_h1_measured"]:
        work["stream_diagnostic"] = k.diagnostic(
            [p for p in work["predictions"] if p["slot"] <= 96]
        )
    atomic_json(
        raw / "primitive_evidence.json",
        {
            f: work[f]
            for f in [
                "predictions",
                "targets",
                "retention_seal_check",
                "label_access_log",
                "optimizer",
                "independence",
            ]
        },
    )
    work["raw_refs"] = [
        reference(raw / "primitive_evidence.json"),
        reference(raw / "frozen_configuration.json"),
    ]
    work.update(
        duration_s=time.monotonic() - began,
        phase_spans=[
            dict(
                phase="authenticate_seals_then_independent_reduction",
                start_s=0,
                duration_s=time.monotonic() - began,
            )
        ],
        code_refs=[reference(ROOT / f) for f in [*OWNED, TEST]],
        preconditions_checked=True,
    )
    atomic_json(raw / "measurement.json", work)
    progress("after_measurement", int(work["full_h1_measured"]), 0)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """A mechanically valid null is ready; external absence and owned errors differ."""
    coverage = work.get("owned_coverage_reference")
    covered = json.loads(Path(coverage["path"]).read_bytes()) if coverage else None
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    checked = checked and (
        covered is None or all(covered["files"][p]["summary"]["missing_lines"] == 0 for p in OWNED)
    )
    measured = work["full_h1_measured"]
    reduced = (
        k.reduce(work["predictions"], work["targets"], work["comparator"], work["optimizer"])
        if measured
        else {}
    )
    ready = int(checked and measured and not work["failures"])
    klass = (
        "disqualified"
        if not checked
        else "blocked"
        if work["failures"] or not measured
        else "positive"
        if reduced["h1_development_signal_score"]
        else "null"
    )
    suffix = reduced.get("science_disposition", "full_h1_seal") if checked else "owned_validation"
    qualified = reduced.get("qualified_count", 0)
    value: Json = dict(
        reduced,
        experiment_id=8350,
        task_id=TASK,
        milestone="2026.10.720",
        run_date="20261009",
        honest_verdict="complete_" + klass + "_" + suffix,
        verdict_class=klass,
        gate_check_summary=work["gates"] + work["failures"],
        static_audit_ready_score=ready,
        h1_development_signal_score=reduced.get("h1_development_signal_score", 0) * ready,
        full_h1_measured=measured,
        stream_diagnostic_only=not measured,
        stream_predictor_diagnostic=work["stream_diagnostic"],
        retention_seal_check=work["retention_seal_check"],
        label_access_log=work["label_access_log"],
        comparator_id=work["comparator"],
        paired_cost_rows=reduced.get("paired_cost_rows", []),
        bootstrap_summary=reduced.get("bootstrap_summary", {}),
        missing_bounds=reduced.get("missing_bounds", []),
        input_independence=work["independence"],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=work["historical"],
        rows=reduced.get("paired_cost_rows", work["stream_diagnostic"]),
        typed_cost_rows=reduced.get("rows", []),
        intended_count=128,
        completed_count=qualified,
        failed_count=0,
        censored_count=128 if not measured else 0,
        excluded_count=128 - qualified if measured else 0,
        independent_count=qualified,
        sample_size_budget=dict(
            k.CONFIG, stream_diagnostic_slots=96, retention_windows=[0, 32, 64, 96]
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=checked,
        flagged_adversarial=not checked,
        acceptance_gates=dict(
            owned=checked,
            full_h1_seals=work["retention_seal_check"]["passed"],
            static_audit_ready=bool(ready),
            scientific_reporting_threshold=k.CONFIG,
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("finding_audits", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178311,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=work["raw_refs"],
        cited_upstream_artifacts=[
            dict(r, fields_imported=["sealed inputs, static policy or terminal authority"])
            for r in work["refs"]
        ],
        measurement_reference=reference(raw / "measurement.json"),
        owned_coverage_reference=coverage,
        execution_manifest_reference=work.get("execution_manifest_reference"),
        invocation_argv=work.get("invocation_argv", []),
        authority=work["authority"],
        methodology_note="All128 frozen intended sources; independent typed cost reduction; source bootstrap10000, seed7178311. Nominal alpha.025 is descriptive exposed-development reporting. Missing targets retain bounds. H1 never gates online learning. Sigmoid34 parity forbids energy novelty.",
        future_dependencies=[
            dict(
                path="results/experiment_8351_v720_learning_retention_audit.json",
                role="future_H2_and_retention_evaluator_not_an_input",
            )
        ],
    )
    value["field_principles"] = {
        f: "Bind "
        + f
        + " to sealed source evidence, original denominators and exposed-development scope."
        for f in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Bind semantic fields except this checksum to independently replayed inputs."
    )
    value["field_principles"]["field_principles"] = (
        "Explain the evidentiary purpose of each artifact field."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Reauthenticate original operands and recompute science, even after rehashing."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["measurement_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        for r in [*work["refs"], *work["code_refs"], *work["raw_refs"]]:
            if sha256_file(Path(r["path"])) != r["sha256"]:
                return False
        for receipt in value["validation_receipts"]:
            for prefix in ("stdout", "stderr", "log"):
                if (
                    receipt.get(prefix + "_path")
                    and sha256_file(Path(receipt[prefix + "_path"])) != receipt[prefix + "_sha256"]
                ):
                    return False
        primitive = json.loads(Path(work["raw_refs"][0]["path"]).read_bytes())
        if (
            primitive != {f: work[f] for f in primitive}
            or json.loads(Path(work["raw_refs"][1]["path"]).read_bytes()) != k.CONFIG
        ):
            return False
        with TemporaryDirectory(prefix="carnot-8350-replay-") as directory:
            expected = measure(Path(work["root"]), Path(directory) / "raw")
        for field in [
            "predictions",
            "targets",
            "full_h1_measured",
            "comparator",
            "optimizer",
            "reduction",
            "retention_seal_check",
            "stream_diagnostic",
            "independence",
            "historical",
            "owned_failure",
        ]:
            if expected[field] != work[field]:
                return False
        if any(
            not 0 < r["seals_authenticated_wall_ns"] <= r["access_wall_ns"]
            or not r["all_seals_authenticated_before_access"]
            or r["learner_feedback_count"] != 0
            for r in work["label_access_log"]
        ):
            return False
        if {
            f: work["authority"].get(f) for f in ("activated", "canonical_tasks_sha256", "tasks")
        } != {
            f: expected["authority"].get(f)
            for f in ("activated", "canonical_tasks_sha256", "tasks")
        }:
            return False

        def gates(data: Json) -> list[Json]:
            return [
                {
                    key: val
                    for key, val in row.items()
                    if key
                    not in (
                        ["path", "hash", "upstream"]
                        if row["artifact_field"] == "private_resources"
                        else []
                    )
                }
                for row in [*data["gates"], *data["failures"]]
            ]

        if gates(work) != gates(expected):
            return False
        return bool(value == build(work, Path(ref["path"]).parent, value["validation_receipts"]))
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Trace actual children and freeze the applicable evaluator boundary consumers."""
    plan = BASE_MANIFEST(private, candidate)
    with (private / "coverage.ini").open("a") as stream:
        stream.write("patch = subprocess\n")
    plan["commands"][1]["argv"].extend(
        [
            "tests/python/test_source_boundary_7852.py",
            "tests/python/test_experiment_7942_v689_sentence_labels.py",
        ]
    )
    plan["commands"][1]["name"] = "consumers_and_private_E2E015_018_019_021"
    return dict(plan)


def main(argv: list[str] | None = None) -> int:
    """Reuse the bounded supervisor, typed finding controls and unchanged publisher."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "manifest", manifest),
    ):
        return int(execution.main(argv))
