"""REQ-REPORT-8488: evaluate frozen source controls under complete current authority.

The shipped audit owns prediction sealing, evaluator custody and cold replay.
Small adapters change only independently pinned operands and registered metrics.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator
from unittest.mock import patch

from carnot.reporting import v726_criterion_decision_audit as shipped
from carnot.reporting import v730_contract_methods as methods
from carnot.reporting.current_work_receipt import atomic_json
from carnot.reporting.primary_publication import read_bound_sidecar, reader_receipt
from carnot.verify import v730_information_metrics as n
from carnot.verify import v730_source_control_fit as fit

Json = dict[str, Any]
ROOT = methods.ROOT
NAME, TASK = "experiment_8488_v730_source_information_audit", "exp8488-source-information-audit"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_v730_source_information_8488.py"
OWNED = [
    "python/carnot/verify/v730_information_metrics.py",
    "python/carnot/reporting/v730_source_information_audit.py",
    "python/carnot/reporting/v730_source_information_runner.py",
    CLI,
]
UPSTREAM = "results/experiment_8487_v730_source_control_fit.json"
UPSTREAM_PIN = "sha256:094f758e2046efec7a93d22fcc297a56ac2af388a574fef78a638b15e61dcb4d"
HISTORICAL = "results/experiment_8432_v726_criterion_decision_audit.json"
HISTORICAL_PIN = "sha256:100efbb17df04e60271df4f8c7e574e1ad03948d1b0d4888625fd6b35993e4b2"
MODEL_SPECS: list[Json] = []
read, reference, checksum, sha256_file = (
    methods.read,
    methods.reference,
    methods.checksum,
    methods.sha256_file,
)
_METHODS = SimpleNamespace(**dict(vars(methods), legacy=shipped.methods.legacy))
_BUILD = shipped.build


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual progress because this CPU audit has no model activity to monitor."""
    print(f"[exp8488] phase={phase} completed={completed} pending={pending}", flush=True)


def operands(root: Path) -> tuple[Json, Json, Json, Json]:
    """Independent producer and original authority pins reject locally repaired evidence."""
    upstream = read(dict(path=str(root / UPSTREAM), sha256=UPSTREAM_PIN))
    receipt = reader_receipt(fit.TASK, root / "results", field="head_controls_ready_score")
    terminal = json.loads(Path(upstream["terminal_validation_sidecar_path"]).read_bytes())
    bound = read_bound_sidecar(root / UPSTREAM, Path(terminal["publication"]["sidecar_path"]))
    if (
        not upstream["required_checks_passed"]
        or upstream["flagged_adversarial"]
        or upstream["head_controls_ready_score"] != 1
        or not receipt["passed"]
        or not bound["report"]["passed"]
    ):
        raise ValueError("head_controls_not_qualified")
    for ref in upstream["code_config_hashes"]:
        if "path" in ref and sha256_file(Path(ref["path"])) != ref["sha256"]:
            raise ValueError("head_predictor_method_drift")
    controls, _ = fit.operands(root)
    original = read(dict(path=str(root / shipped.fit.UPSTREAM), sha256=shipped.fit.UPSTREAM_PIN))
    frozen = read(upstream["frozen_heads_reference"])
    features = read(controls["feature_manifest_reference"])
    custody = read(controls["evaluator_custody"])
    panel = dict(
        original,
        feature_panel_path=controls["feature_manifest_reference"]["path"],
        feature_panel_sha256=controls["feature_manifest_reference"]["sha256"],
        evaluator_custody=custody["reference"],
    )
    return upstream, panel, frozen, features


def predict_rows(frozen: Json, features: list[Json], strata: list[Json]) -> list[Json]:
    """Only numeric vectors enter fitted heads; metadata enters after probabilities exist."""
    metadata = {r["unit_id"]: r for r in strata}
    if not frozen:
        return [
            dict(
                unit_id=r["unit_id"],
                question_hash=r["question_hash"],
                dataset=metadata[r["unit_id"]]["dataset"],
                generator=metadata[r["unit_id"]]["generator"],
                arm=a,
                seed=s,
                probability=None,
                missing_reason="missing_required_upstream_controls",
            )
            for r in features
            for a in n.ARMS
            for s in n.SEEDS
        ]
    predictions = fit.predictions(frozen, dict(rows=features))
    return [
        dict(
            unit_id=r["answer_id"],
            question_hash=r["question_hash"],
            dataset=metadata[r["answer_id"]]["dataset"],
            generator=metadata[r["answer_id"]]["generator"],
            arm=r["arm"],
            seed=r["seed"],
            probability=r["probability"],
            missing_reason=None,
        )
        for r in predictions
    ]


@contextmanager
def binding() -> Iterator[None]:
    """Keep the shipped seal order, schema and rejection paths with explicit scoped bindings."""
    with patch.multiple(
        shipped,
        NAME=NAME,
        TASK=TASK,
        CLI=CLI,
        TEST=TEST,
        OWNED=OWNED,
        UPSTREAM=UPSTREAM,
        UPSTREAM_PIN=UPSTREAM_PIN,
        methods=_METHODS,
        operands=operands,
        predict_rows=predict_rows,
        n=n,
        progress=progress,
    ):
        yield


def measure(root: Path, raw: Path) -> Json:
    """Resource checks and current authority precede sealing; optional devices never gate CPU work."""
    with binding():
        work = dict(shipped.measure(root, raw))
    work["preconditions"].update(measurement_cap_s=3600, optional_cuda_and_boards_gate_cpu=False)
    for failure in work["failures"]:
        if failure.get("field") == "head_ready_score_and_authenticated_operands":
            failure.update(
                check="head_controls_ready_score_and_authenticated_operands",
                field="head_controls_ready_score_and_authenticated_operands",
                upstream=fit.TASK,
            )
    historical = read(dict(path=str(ROOT / HISTORICAL), sha256=HISTORICAL_PIN))
    work["historical_v726_gain"] = dict(
        gain=historical["brier_gain_vs_logistic"],
        ci95=historical["brier_gain_ci95"],
        decision_loss_delta=historical["decision_loss_delta"],
        comparison="original full spline versus nine-input logistic; exposed-history",
    )
    work["source_refs"].append(dict(path=str(ROOT / HISTORICAL), sha256=HISTORICAL_PIN))
    if work["result"]:
        controls = read(dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN))
        work["source_refs"].append(dict(path=str(root / fit.UPSTREAM), sha256=fit.UPSTREAM_PIN))
        work["source_refs"].extend(methods.protocol()["historical_authorities"]["V726"])
        work["source_refs"].append(controls["feature_manifest_reference"])
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Qualification and measured benefit stay separate even when an owned check fails."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    result = work["result"] or {}
    value.update(
        experiment_id=8488,
        task_id=TASK,
        milestone="2026.10.730",
        run_date="20261013",
        honest_verdict=f"complete_{value['verdict_class']}_v730_source_information_audit",
        source_audit_ready_score=value["decision_audit_ready_score"],
        source_information_benefit_score=value["exploratory_benefit_score"],
        historical_v726_gain=work["historical_v726_gain"],
        claim_retired=bool(result) and not bool(result["exploratory_benefit_score"]),
        external_new_label_condition="Externally collected, previously unexposed questions with new independently adjudicated labels and frozen generators/heads before access; never a new split of this panel.",
        population_grouping_loss_estimate=None,
        methodology_note="Two registered paired Brier contrasts on exposed fixed questions; 10000 whole-question draws at seed7278446 and Bonferroni97.5 percent intervals. Fixed typed decisions cost1/0/0.2. Repeated deterministic fits are not independent samples. No test-label curve fit, population grouping-loss estimate, SIFT implementation or theorem transfer.",
    )
    for key in [
        "paired_question_rows",
        "contrasts",
        "familywise_intervals",
        "decision_loss_deltas",
        "repeat_fit_audit",
        "matched_coverage_risk",
    ]:
        value[key] = result.get(key)
    value["brier_gain_vs_q_only_spline"] = value.pop("brier_gain_vs_logistic")
    value["familywise_confidence_level"] = 0.975
    value["acceptance_gates"].update(
        both_source_contrasts_positive=bool(result)
        and all(
            interval and interval[0] > 0 for interval in result["familywise_intervals"].values()
        )
    )
    value["historical_model_provenance"][0]["original_producer_hashes"] = work["source_refs"]
    value["field_principles"].update(
        {
            k: "Bound to pinned exposed operands and fixed question-level comparisons; qualification and benefit remain separate."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        source_audit_ready_score="Qualified numerical audit, including a measured null.",
        source_information_benefit_score="Both adjusted lower bounds exceed zero with typed-loss nonincrease versus q-only.",
        historical_v726_gain="Unchanged original nine-input logistic contrast; separate from current scalar/permutation tests.",
        repeat_fit_audit="Unique numerical prediction vectors and all pairwise seed decisions; frozen fit repeats have no independence credit.",
        matched_coverage_risk="Label-blind confidence ranking at the smallest fixed definitive count; fractional ties remove metadata selection.",
    )
    value["field_principles"].pop("brier_gain_vs_logistic")
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Shipped cold replay recomputes predictions and losses from independent original seals."""
    try:
        value = json.loads(path.read_bytes())
        work = read(value["primitive_reference"])
        original = read(dict(path=str(ROOT / HISTORICAL), sha256=HISTORICAL_PIN))
        if work["historical_v726_gain"] != dict(
            gain=original["brier_gain_vs_logistic"],
            ci95=original["brier_gain_ci95"],
            decision_loss_delta=original["decision_loss_delta"],
            comparison="original full spline versus nine-input logistic; exposed-history",
        ):
            return False
    except (OSError, ValueError, KeyError, TypeError):
        return False
    with binding(), patch.object(shipped, "build", build):
        return bool(shipped.replay(path))
