"""REQ-REPORT-8021: reduce frozen policies on exposed development sources.

Prediction custody precedes evaluator access. A valid null can prepare later
work, but these cached sources cannot establish generalized learning benefit.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from carnot import experiment_8020_v695_qualified_energy_fit as prior
from carnot.reporting.current_work_receipt import (
    ZERO_INVOCATION_COUNTS,
    atomic_json,
    canonical_hash,
    sha256_file,
)
from carnot.reporting.evidence_features_custody_7980 import checked, reference
from carnot.reporting.experiment_7303_validation_scope import run_commands
from carnot.reporting.primary_publication import publish_primary, reader_receipt
from carnot.verify.qwen_energy_calibration_7972 import holm

Json = dict[str, Any]
ROOT = prior.ROOT
NAME = "experiment_8021_v695_typed_decision_test"
TASK = "exp8021-typed-decision-test"
OWNED = [f"python/carnot/{NAME}.py", f"scripts/experiments/{NAME}.py"]
TEST = "tests/python/test_typed_decision_8021.py"
CONFIG = dict(
    seed=69521,
    draws=10000,
    primary="conditioned_energy/posthoc",
    primary_seed=17,
    minimum_groups=192,
    per_class=20,
    comparator="minimum tune cost; fixed lexical tie order; no sigmoid identity",
    hypotheses=["static_policy", "source_intervention", "online_update"],
    alpha=0.05,
    minimum_gain=0.02,
    maximum_brier_degradation=0.005,
    minimum_headroom_errors=4,
    costs=dict(unsupported_accept=5, supported_reject=1, escalate=0.5, correct=0),
)
BEGAN = time.monotonic()


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual elapsed work so an operator can detect a stalled phase."""
    print(
        f"[exp8021] phase={phase} elapsed_s={time.monotonic() - BEGAN:.3f} completed={completed} pending={pending}",
        flush=True,
    )


def action(p: float | None) -> str:
    """Escalation wins exact expected-cost ties, including missing predictions."""
    if p is None:
        return "escalate"
    if not math.isfinite(p) or not 0 <= p <= 1:
        raise ValueError("probability")
    return "accept" if 5 * p < 0.5 else "reject" if 1 - p < 0.5 else "escalate"


def loss(decision: str, y: int | None) -> float | None:
    """Unknown targets contribute no cost; one denotes an unsupported answer."""
    return (
        None
        if y is None
        else 0.5
        if decision == "escalate"
        else float(5 * y if decision == "accept" else 1 - y)
    )


def key(row: Json) -> str:
    """Keep calibration conditions distinct while seeds remain repeat evidence."""
    return f"{row['arm']}/{row['condition']}"


def verify_seal(rows: list[Json], seal: str) -> bool:
    """The original seal binds row order and bytes before any new label access."""
    if canonical_hash(rows) != seal or any(r["y"] is not None for r in rows):
        raise ValueError("prediction_seal")
    return True


def select_comparator(tune: list[Json]) -> str:
    """Choose a nonduplicate baseline using tune cost, never stream outcomes."""
    candidates = {}
    for r in tune:
        if (
            r["seed"] == 17
            and r["arm"] not in {"conditioned_energy", "sigmoid_identity"}
            and r["p"] is not None
            and r["y"] is not None
        ):
            candidates.setdefault(key(r), []).append(loss(action(r["p"]), r["y"]))
    if not candidates:
        raise ValueError("comparator_support")
    return min(candidates, key=lambda k: (float(np.mean(candidates[k])), k))


def reduce(predictions: list[Json], targets: list[Json], comparator: str) -> Json:
    """Keep each intended slot, then bootstrap paired means of source groups."""
    labels = {r["family_id"]: r["eligible_y"] for r in targets}
    if len(labels) != len(targets) or any(
        y is not None and (type(y) is not int or y not in (0, 1)) for y in labels.values()
    ):
        raise ValueError("target_contract")
    if set(labels) != {r["family_id"] for r in predictions}:
        raise ValueError("target_roster")
    rows = []
    for r in predictions:
        y, p = labels[r["family_id"]], r["p"]
        known = y is not None and p is not None
        d = action(p)
        clipped = min(1 - 1e-12, max(1e-12, p)) if p is not None else None
        rows.append(
            dict(
                r,
                y=y,
                decision=d,
                actual_cost=loss(d, y) if known else None,
                brier=(p - y) ** 2 if known else None,
                log_loss=-(y * math.log(clipped) + (1 - y) * math.log(1 - clipped))
                if known
                else None,
                numerator=loss(d, y) if known else None,
                denominator=int(known),
                eligibility=known,
                exclusion_reason=r.get("exclusion_reason")
                or ("unknown_target" if y is None else "missing_prediction" if p is None else None),
            )
        )
    primary = [r for r in rows if r["seed"] == 17]
    lookup = {(key(r), r["family_id"]): r for r in primary}
    paired, headroom = [], []
    for r in primary:
        if key(r) != CONFIG["primary"]:
            continue
        b = lookup[(comparator, r["family_id"])]
        if r["eligibility"] and b["eligibility"]:
            paired.append(
                dict(
                    source_cluster_id=r["source_cluster_id"],
                    family_id=r["family_id"],
                    y=r["y"],
                    gain=b["actual_cost"] - r["actual_cost"],
                    brier_degradation=r["brier"] - b["brier"],
                    changed=r["decision"] != b["decision"],
                )
            )
            if b["actual_cost"] > 0:
                headroom.append(
                    dict(
                        b,
                        reducible_cost=b["actual_cost"],
                        baseline_error=b["decision"] != "escalate",
                    )
                )
    groups = sorted({r["source_cluster_id"] for r in paired})
    delta = np.array(
        [
            [
                np.mean([r[k] for r in paired if r["source_cluster_id"] == g])
                for k in ("gain", "brier_degradation")
            ]
            for g in groups
        ]
    )
    rng, draws = np.random.default_rng(CONFIG["seed"]), []
    progress("bootstrap_before", 0, CONFIG["draws"])
    for start in range(0, CONFIG["draws"], 1000):
        draws.extend(
            delta[rng.integers(0, len(groups), (1000, len(groups)))].mean(axis=1).tolist()
            if groups
            else [[0, 0]] * 1000
        )
        progress("bootstrap_draws", start + 1000, CONFIG["draws"] - start - 1000)
    boot = np.array(draws)
    mean = float(delta[:, 0].mean()) if groups else 0.0
    raw_p = (1 + int(np.sum(boot[:, 0] - mean >= mean))) / (CONFIG["draws"] + 1)
    adjusted = holm(dict(static_policy=raw_p, source_intervention=1.0, online_update=1.0))
    interval = np.quantile(boot[:, 0], [0.05 / 6, 1 - 0.05 / 6]).tolist()
    brier = float(delta[:, 1].mean()) if groups else None
    selected = [r for r in primary if key(r) == CONFIG["primary"]]
    baseline = [r for r in primary if key(r) == comparator]
    class_counts = {
        str(y): len({r["source_cluster_id"] for r in paired if r["y"] == y}) for y in (0, 1)
    }
    gates = dict(
        support=len(groups) >= 192 and min(class_counts.values()) >= 20,
        headroom=len({r["source_cluster_id"] for r in headroom if r["baseline_error"]}) >= 4,
        changed_actions=any(r["changed"] for r in paired),
        mean_cost_reduction=mean >= 0.02,
        adjusted_ci=bool(interval[0] > 0 and adjusted["static_policy"] < 0.05),
        brier_noninferiority=brier is not None and brier <= 0.005,
        no_additional_false_accepts=sum(r["decision"] == "accept" and r["y"] == 1 for r in selected)
        <= sum(r["decision"] == "accept" and r["y"] == 1 for r in baseline),
    )
    metrics = {}
    for arm, seed, condition in sorted({(r["arm"], r["seed"], r["condition"]) for r in rows}):
        chosen = [
            r for r in rows if (r["arm"], r["seed"], r["condition"]) == (arm, seed, condition)
        ]
        usable = [r for r in chosen if r["eligibility"]]
        accepted = [r for r in usable if r["decision"] == "accept"]
        false = sum(r["y"] == 1 for r in accepted)
        stats = dict(
            intended=len(chosen),
            eligible=len(usable),
            started=len(chosen),
            completed=len(usable),
            excluded=len(chosen) - len(usable),
            failed=sum(bool(r.get("failure_reason")) for r in chosen),
            censored=sum(bool(r.get("censor_reason")) for r in chosen),
            independent=len({r["source_cluster_id"] for r in usable}),
            false_accepts=false,
            accepted_risk=false / len(accepted) if accepted else None,
            accepted_risk_numerator=false,
            accepted_risk_denominator=len(accepted),
            coverage=sum(r["decision"] != "escalate" for r in usable) / len(usable)
            if usable
            else None,
            changed_actions=sum(
                r["decision"] != lookup[(comparator, r["family_id"])]["decision"] for r in chosen
            ),
        )
        for metric in ("actual_cost", "brier", "log_loss"):
            numerator = sum(r[metric] for r in usable)
            stats[metric] = dict(
                numerator=numerator,
                denominator=len(usable),
                mean=numerator / len(usable) if usable else None,
            )
        stats["calibration_bins"] = [
            dict(
                bin=i,
                denominator=len(members),
                probability_sum=sum(r["p"] for r in members),
                unsupported_sum=sum(r["y"] for r in members),
            )
            for i in range(10)
            for members in [[r for r in usable if min(9, int(r["p"] * 10)) == i]]
        ]
        metrics[f"{arm}/{condition}/{seed}"] = stats
    valid = [r for r in selected if r["eligibility"]]
    return dict(
        rows=rows,
        arm_metrics=metrics,
        paired_source_rows=paired,
        headroom_rows=headroom,
        selected_comparator=comparator,
        acceptance_gate_results=gates,
        decision_benefit_score=int(all(gates.values())),
        genuine_headroom=dict(
            groups=len({r["source_cluster_id"] for r in headroom}),
            baseline_excess_cost=sum(r["reducible_cost"] for r in headroom),
        ),
        adjusted_intervals=dict(
            mean_cost_reduction=mean,
            confidence_interval_95=np.quantile(boot[:, 0], [0.025, 0.975]).tolist(),
            holm_family_interval=interval,
            interval_method="Holm first-rank alpha/3 with two pending p=1 hypotheses",
            raw_p=raw_p,
            holm_adjusted_p=adjusted["static_policy"],
            adjusted_p_values=adjusted,
            pending_hypotheses=["source_intervention", "online_update"],
            draws=10000,
            independent=len(groups),
            brier_degradation=brier,
            brier_interval_95=np.quantile(boot[:, 1], [0.025, 0.975]).tolist(),
        ),
        sample_size_budget=dict(
            intended=len(selected),
            eligible=len(valid),
            started=len(selected),
            completed=len(valid),
            excluded=len(selected) - len(valid),
            failed=0,
            censored=0,
            independent=len(groups),
            class_counts=class_counts,
            seeds_are_independent=False,
        ),
    )


def load_inputs(root: Path, raw: Path) -> tuple[Json, list[Json]]:
    """Verify receipts and freeze public evidence before opening evaluator labels."""
    values, refs, failures = {}, [], []
    copy = prior.old.prior.copy_evidence
    for name, field in (
        (prior.NAME, "energy_fit_ready_score"),
        (prior.eligible.NAME, "stream_targets_ready_score"),
    ):
        path = root / "results" / (name + ".json")
        value = json.loads(path.read_text()) if path.is_file() else {}
        check = dict(
            upstream_id=value.get("task_id", name),
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            expected=1,
            observed=value.get(field, "MISSING_CONTRACT_FIELD"),
            passed=value.get(field) == 1
            and value.get("verdict_class") not in {"blocked", "disqualified"},
        )
        values[name] = value
        if not check["passed"]:
            failures.append(check)
        else:
            refs.append(copy(reference(path), raw))
    if failures:
        return dict(references=refs), failures
    fitted, eligible = values[prior.NAME], values[prior.eligible.NAME]
    try:
        progress("frozen_head_checkpoint_replay_before")
        prior.replay(root / "results" / (prior.NAME + ".json"))
        progress("frozen_head_checkpoint_replay_after")
        measurement_ref = copy(fitted["measurement_checkpoint"], raw)
        measurement = json.loads(checked(measurement_ref).read_text())
        predictions = [r for r in measurement["primitive_rows"] if r["role"] == "stream"]
        verify_seal(predictions, fitted["public_prediction_seals"]["stream"])
        if (
            measurement["public_prediction_seals"]["stream"]
            != fitted["public_prediction_seals"]["stream"]
        ):
            raise ValueError("prediction_seal_receipt")
        durable_heads = [copy(r, raw) for r in fitted["head_checkpoints"]]
        refs += [measurement_ref, *durable_heads, copy(fitted["calibration_checkpoint"], raw)]
        public_ref = copy(eligible["public_manifests"]["stream"], raw)
        public = json.loads(checked(public_ref).read_text())["rows"]
        if len(public) != 256 or len({r["family_id"] for r in public}) != 256:
            raise ValueError("original_stream_roster")
        indexed = {r["family_id"]: r for r in public}
        for r in predictions:
            p = indexed[r["family_id"]]
            if (
                r["source_cluster_id"] != p["source_cluster_id"]
                or (r["p"] is not None) != p["public_eligible"]
            ):
                raise ValueError("original_mask_or_id")
        refs += [
            public_ref,
            copy(eligible["exclusion_manifest"], raw),
            copy(eligible["schema_manifests"]["evaluator"], raw),
        ]
        target_schema = copy(
            reference(root / "results" / "experiment_7955_v690_response_targets.json"), raw
        )
        schema = json.loads(checked(target_schema).read_text())["target_definition"]
        if (
            schema["primary"]
            != "any authenticated human source-unsupported span anywhere in the complete response"
        ):
            raise ValueError("source_target_schema")
        refs.append(target_schema)
        tune = [r for r in measurement["primitive_rows"] if r["role"] == "tune"]
        comparator = select_comparator(tune)
        seal_ref = prior.eligible.shard(
            raw,
            "prediction_seal",
            dict(
                original_seal=fitted["public_prediction_seals"]["stream"],
                measurement=measurement_ref,
                heads=durable_heads,
                comparator=comparator,
                public=public_ref,
                invocation_seal_checked_monotonic_ns=time.monotonic_ns(),
                historical_exposure=True,
            ),
        )
        refs.append(seal_ref)
        progress("prediction_seal_checked_before_stream_labels", 256, 256)
        label_ref = copy(eligible["evaluator_manifests"]["stream"], raw)
        targets = json.loads(checked(label_ref).read_text())
        spans = {r["family_id"] for r in targets["annotation_rows"]}
        if (
            targets["access_policy"] != "evaluator_only"
            or targets["role"] != "stream"
            or any(
                r["eligible_y"] is not None and r["eligible_y"] != int(r["family_id"] in spans)
                for r in targets["rows"]
            )
        ):
            raise ValueError("source_target_orientation")
        refs.append(label_ref)
        access_ref = prior.eligible.shard(
            raw,
            "label_access",
            dict(
                prediction_receipt=seal_ref,
                target_reference=label_ref,
                stream_access_monotonic_ns=time.monotonic_ns(),
                opened_roles=["stream"],
                retention_labels_opened=False,
                original_annotation_semantics="authenticated unsupported spans: y=int(bool(spans)); one means unsupported",
                source_schema=target_schema,
                historical_exposure=eligible["exposure_scope"],
            ),
        )
        refs += [access_ref, *[copy(r, raw) for r in eligible["exposure_rows"]]]
        refs.append(
            copy(reference(root / "results" / "experiment_8006_v694_independent_replay.json"), raw)
        )
        refs += [
            copy(dict(path=r["path"], sha256=r["hash"]), raw)
            for r in eligible["historical_failure_logs"]
        ]
        return dict(
            predictions=predictions,
            targets=targets["rows"],
            comparator=comparator,
            references=refs,
            prediction_seal=seal_ref,
            label_access_receipt=access_ref,
            exposure_scope=eligible["exposure_scope"],
            trained_head_specs=[
                dict(r, scope="imported_exp8020", current_training_steps=0)
                for r in fitted["trained_head_specs"]
            ],
        ), []
    except (ValueError, KeyError, OSError) as error:
        return dict(references=refs), [
            dict(
                check,
                artifact_field="immutable_prediction_and_source_contract",
                expected="sealed original heads, IDs, masks and unsupported orientation",
                observed=str(error),
                passed=False,
            )
        ]


def controls(raw: Path) -> Json:
    """Separate oracle fixtures test sensitivity and cannot earn natural benefit."""
    outcomes = {}
    for name, room in (("known_headroom", True), ("zero_headroom", False)):
        predictions, targets = [], []
        for i in range(256):
            y = i % 2
            targets.append(dict(family_id=str(i), eligible_y=y))
            for arm in ("conditioned_energy", "linear"):
                predictions.append(
                    dict(
                        family_id=str(i),
                        source_cluster_id=str(i),
                        role="stream",
                        arm=arm,
                        condition="posthoc",
                        seed=17,
                        p=float(1 - y if room and arm == "linear" else y),
                        y=None,
                    )
                )
        ref = prior.eligible.shard(
            raw,
            "controls",
            dict(predictions=predictions, targets=targets, comparator="linear/posthoc"),
        )
        result = reduce(predictions, targets, "linear/posthoc")
        outcomes[name] = dict(
            decision_benefit_score=result["decision_benefit_score"],
            acceptance_gate_results=result["acceptance_gate_results"],
            adjusted_intervals=result["adjusted_intervals"],
            checkpoint=ref,
            verifier_is_oracle=True,
            verdict_class="circular_positive" if result["decision_benefit_score"] else "null",
            natural_benefit_credit=False,
        )
    outcomes["passed"] = (
        outcomes["known_headroom"]["decision_benefit_score"] == 1
        and outcomes["zero_headroom"]["decision_benefit_score"] == 0
    )
    return outcomes


def coverage_counts(scratch: Path) -> Json:
    """The denominator includes only added producer and direct CLI statements."""
    path = scratch / "coverage.json"
    report = json.loads(path.read_text()) if path.exists() else dict(files={})
    return {k: r["summary"] for k, r in report["files"].items()}


def validation_plan(scratch: Path) -> list[prior.CommandSpec]:
    """Reuse the tested runner's private coverage, consumer and E2E manifest."""
    commands = prior.validation_plan(scratch)
    config = scratch / "coverage.ini"
    config.write_text(config.read_text().replace(prior.NAME, NAME))
    return [
        replace(
            c, argv=tuple(a.replace(prior.NAME, NAME).replace(prior.TEST, TEST) for a in c.argv)
        )
        for c in commands
    ]


def replay(path: Path) -> Json:
    """A fresh reader recomputes every decision claim from durable raw rows."""
    v = json.loads(path.read_text())
    for ref in v["raw_shard_hashes"] + v["code_config_hashes"]:
        checked(ref)
    for name in ("known_headroom", "zero_headroom"):
        control = v["positive_control_results"][name]
        data = json.loads(checked(control["checkpoint"]).read_text())
        outcome = reduce(data["predictions"], data["targets"], data["comparator"])
        if any(
            outcome[k] != control[k]
            for k in ("decision_benefit_score", "acceptance_gate_results", "adjusted_intervals")
        ):
            raise ValueError("control_reduction")
    if v["checkpoint_references"]:
        inputs = json.loads(checked(v["checkpoint_references"][0]).read_text())
        result = reduce(inputs["predictions"], inputs["targets"], inputs["comparator"])
        for field in result:
            if field == "acceptance_gate_results":
                if any(v[field][k] != value for k, value in result[field].items()):
                    raise ValueError("reduction:" + field)
            elif field == "decision_benefit_score" and v["verdict_class"] == "disqualified":
                if v[field] != 0:
                    raise ValueError("unsafe_benefit")
            elif v[field] != result[field]:
                raise ValueError("reduction:" + field)
    if v["decision_measurement_ready_score"] and (
        v["verdict_class"] in {"blocked", "disqualified", "circular_positive"}
        or not v["acceptance_gate_results"]["owned_checks"]
        or not all(r["passed"] for r in v["validation_receipts"])
    ):
        raise ValueError("unsafe_readiness")
    return dict(passed=True, sha256=sha256_file(path))


def terminal(path: Path) -> Json:
    """Use unchanged adversarial and strict row readers on exact candidate bytes."""
    py = str(ROOT / ".venv/bin/python")
    commands = [
        prior.CommandSpec(
            "cold_reduction",
            (py, "-u", str(ROOT / OWNED[-1]), "--cold-replay", str(path)),
            "terminal",
            120,
        ),
        prior.CommandSpec(
            "adversarial",
            (py, "scripts/adversarial_verify.py", str(path), "--json"),
            "terminal",
            120,
        ),
        prior.CommandSpec(
            "strict_rows",
            (py, "scripts/verdict_row_consistency_lint.py", "--strict", str(path)),
            "terminal",
            120,
        ),
    ]
    progress("terminal_subprocesses_before", 0, 3)
    receipts = run_commands(
        ROOT,
        commands,
        log_dir=path.parent / "terminal_logs" / sha256_file(path).split(":")[-1],
        heartbeat_s=30,
    )
    progress("terminal_subprocesses_after", 3, 0)
    return dict(passed=all(r["passed"] for r in receipts), receipts=receipts)


def main(argv: list[str] | None = None) -> int:
    """Freeze methods, reduce once, preserve checks and publish validated bytes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", choices=["20261002"], default="20261002")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / (NAME + ".json"))
    parser.add_argument("--fixture-input", type=Path)
    parser.add_argument("--validation-worker", action="store_true")
    parser.add_argument("--reuse-repository-health", type=Path)
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    began = time.monotonic()
    progress("begin_no_model_load_no_generation")
    try:
        if args.cold_replay:
            replay(args.cold_replay)
            progress("cold_reduction_passed")
            return 0
        output = args.output.absolute()
        raw = output.parent / "raw" / output.stem
        scratch = Path(tempfile.mkdtemp(prefix="carnot-8021-"))
        commands = validation_plan(scratch)
        historical_health = []
        if args.reuse_repository_health:
            historical_health = json.loads(args.reuse_repository_health.read_text())[
                "repository_health"
            ]
            commands = [c for c in commands if c.scope != "repository_health"]
        copy = prior.old.prior.copy_evidence
        code = [copy(reference(ROOT / p), raw) for p in [*OWNED, TEST]]
        code += [copy(reference(Path(p)), raw) for p in (prior.__file__, prior.eligible.__file__)]
        history = [
            copy(reference(p), raw) for p in sorted((raw / "validation_history").glob("*.log"))
        ]
        if args.reuse_repository_health:
            history.append(copy(reference(args.reuse_repository_health), raw))
        config = prior.eligible.shard(
            raw,
            "configuration",
            dict(
                methods=CONFIG,
                code=code,
                commands=[asdict(c) for c in commands],
                roles="tune selects; stream evaluates; retention sealed",
                budgets=dict(stream_slots=256, retention_unopened=64),
                exposure="historically exposed development",
            ),
        )
        progress("methods_roles_budgets_commands_frozen")
        frozen = time.monotonic()
        positive_controls = controls(raw)
        if args.fixture_input:
            inputs, failures = json.loads(args.fixture_input.read_text()), []
            inputs["references"] = []
        else:
            inputs, failures = load_inputs(args.root, raw)
        v: Json = dict(
            experiment_id=8021,
            task_id=TASK,
            milestone="2026.10.695",
            run_date=20261002,
            schema="carnot.v695.typed_decision_test.v1",
            claim_scope="Frozen typed decisions on historically exposed cached development. No unseen, deployment or generalized learning claim.",
            honest_verdict="complete_blocked_typed_decision_prerequisites"
            if failures
            else "complete_null_typed_decisions",
            verdict_class="blocked" if failures else "null",
            gate_check_summary=failures,
            rows=[],
            arm_metrics={},
            paired_source_rows=[],
            headroom_rows=[],
            adjusted_intervals={},
            selected_comparator=None,
            sample_size_budget=dict(
                intended=256,
                eligible=0,
                started=0,
                completed=0,
                excluded=0,
                failed=0,
                censored=256,
                independent=0,
            ),
            random_seed=CONFIG["seed"],
            verifier_is_oracle=False,
            genuine_headroom=False,
            positive_control_results={},
            generalized_learning_benefit_score=0,
            decision_measurement_ready_score=0,
            decision_benefit_score=0,
            acceptance_gate_results=dict(measurement=False, owned_checks=False),
            cost_matrix=CONFIG["costs"],
            inference_substrate="verifier_ensemble_against_cached_candidates",
            inference_substrate_class="no_model_load",
            MODEL_SPECS=[],
            model_specs=[],
            model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
            trained_head_specs=inputs.get("trained_head_specs", []),
            prediction_seal=inputs.get("prediction_seal", {}),
            label_access_receipt=inputs.get("label_access_receipt", {}),
            exposure_scope=inputs.get(
                "exposure_scope", "historically exposed development; retention labels unopened"
            ),
            cited_upstream_artifacts=inputs["references"],
            raw_shard_hashes=[
                config,
                *history,
                *inputs["references"],
                *[positive_controls[k]["checkpoint"] for k in ("known_headroom", "zero_headroom")],
            ],
            code_config_hashes=code,
            checkpoint_references=[],
            validation_receipts=[],
            coverage_statement_counts={},
            repository_health=historical_health,
            preserved_validation_failures=history,
            flagged_adversarial=False,
        )
        if not failures:
            data_ref = prior.eligible.shard(
                raw,
                "decision_inputs",
                {k: inputs[k] for k in ("predictions", "targets", "comparator")},
            )
            v["checkpoint_references"] = [data_ref]
            v["raw_shard_hashes"].append(data_ref)
            v.update(reduce(inputs["predictions"], inputs["targets"], inputs["comparator"]))
            v["acceptance_gate_results"].update(
                measurement=positive_controls["passed"], owned_checks=False
            )
            if args.fixture_input:
                v.update(
                    verdict_class="circular_positive",
                    honest_verdict="complete_circular_positive_typed_decision_fixture",
                    verifier_is_oracle=True,
                )
            elif v["decision_benefit_score"]:
                v.update(
                    verdict_class="positive",
                    honest_verdict="complete_positive_exposed_typed_decisions",
                )
        measured = time.monotonic()
        progress("measurement_complete", len(v["rows"]), len(commands))
        if not args.validation_worker:
            progress("validation_subprocesses_before", 0, len(commands))
            receipts = run_commands(
                ROOT,
                commands,
                log_dir=raw / "validation_logs" / config["sha256"].split(":")[-1],
                heartbeat_s=30,
                extra_env=dict(
                    PYTHONUNBUFFERED="1",
                    JAX_PLATFORMS="cpu",
                    OPENBLAS_NUM_THREADS="1",
                    CARNOT_8021_COVERAGE_CONFIG=str(scratch / "coverage.ini"),
                    COVERAGE_FILE=str(scratch / ".coverage-health"),
                ),
            )
            progress("validation_subprocesses_after", len(receipts), 0)
            v["validation_receipts"] = [r for r in receipts if r["scope"] == "owned"]
            v["repository_health"] += [r for r in receipts if r["scope"] == "repository_health"]
            counts = coverage_counts(scratch)
            v["coverage_statement_counts"] = counts
            passed = (
                bool(counts)
                and all(
                    r["num_statements"] > 0 and r["missing_lines"] == 0 for r in counts.values()
                )
                and all(r["passed"] for r in v["validation_receipts"])
            )
            v["acceptance_gate_results"]["owned_checks"] = passed
            v["decision_measurement_ready_score"] = int(
                passed
                and not failures
                and v["acceptance_gate_results"].get("support", False)
                and not args.fixture_input
                and positive_controls["passed"]
            )
            if not passed and not failures:
                v.update(
                    verdict_class="disqualified",
                    honest_verdict="complete_disqualified_typed_decision_checks",
                    decision_benefit_score=0,
                )
        ended = time.monotonic()
        v.update(
            duration_s=ended - began,
            phase_spans=[
                dict(phase="freeze", duration_s=frozen - began),
                dict(phase="reduce", duration_s=measured - frozen),
                dict(phase="validation", duration_s=ended - measured),
            ],
            terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
            positive_control_results=positive_controls,
        )
        v["reproducibility_checksum"] = canonical_hash(
            dict(config=CONFIG, code=code, raw=v["raw_shard_hashes"])
        )
        v["field_principles"] = {
            k: "Bind current exposed-development work to durable exact bytes; preserve failed gates and source denominators without natural learning credit."
            for k in v
        }
        v["field_principles"].update(
            decision_measurement_ready_score="Integer downstream contract: complete eligible measurement and all owned checks; a valid null qualifies.",
            decision_benefit_score="One only when every useful-decision gate and owned check passes; fixtures remain circular.",
            prediction_seal="Original Exp8020 probabilities and immutable head bytes checked before this invocation opens stream labels.",
            label_access_receipt="Ordered evaluator access follows public seal verification; retention labels remain unopened.",
            adjusted_intervals="10000 paired source-group bootstrap draws; Holm family includes static plus two unmeasured hypotheses at p=1.",
            rows="Every original source, arm, seed and condition retains scored numerator, denominator, exclusion, failure and censor reasons.",
            model_invocation_counts="Only this invocation counts: no pretrained loads or generations; imported small heads are historical.",
            repository_health="Single bounded full-suite diagnostic retains unrelated failures separately from owned checks.",
        )
        progress("publication_before")
        receipt = publish_primary(output, v, replay if args.validation_worker else terminal)
        atomic_json(raw / "terminal_validation.json", receipt)
        reader = reader_receipt(
            TASK,
            output.parent,
            field="decision_measurement_ready_score",
            expected=v["decision_measurement_ready_score"],
        )
        atomic_json(raw / "reader_receipt.json", reader)
        if not reader["passed"] or reader["gate_sha256"] != sha256_file(output):
            raise ValueError("primary_reader")
        replay(output)
        progress("publication_after")
        return 0
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"[exp8021] failed={error}", flush=True)
        return 1
