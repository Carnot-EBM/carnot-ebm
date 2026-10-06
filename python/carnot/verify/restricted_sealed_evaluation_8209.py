"""REQ-REPORT-8209: seal original slots before current evaluator access.

The boundary controls this invocation only. These repeatedly used sources
remain exposed development, even when prediction custody passes every check.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import sys
import time
from typing import Any
from unittest.mock import patch

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.reporting.primary_publication import read_bound_sidecar
from carnot.verify import restricted_energy_fit_8208 as upstream
from carnot.verify import restricted_sealed_rule_8209 as n
from carnot.verify import selective_sealed_evaluation_8196 as previous
from scripts.experiment_template import normalize_artifact_for_template_write

Json = dict[str, Any]
ROOT, fit, reference, execution = (
    upstream.ROOT,
    upstream.fit,
    upstream.reference,
    upstream.execution,
)
NAME = "experiment_8209_v709_restricted_sealed_evaluation"
TASK = "exp8209-restricted-sealed-evaluation"
MODULE = "python/carnot/verify/restricted_sealed_evaluation_8209.py"
NUMERIC = "python/carnot/verify/restricted_sealed_rule_8209.py"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_restricted_sealed_evaluation_8209.py"
OWNED = [MODULE, NUMERIC, CLI]
RUN_DATE = "20261006"
MODEL_SPECS: list[Json] = []
UPSTREAM = "results/experiment_8208_v709_restricted_energy_fit.json"
PINS = {
    UPSTREAM: "sha256:0063bc79bf9be8d3a7739603f105a263a83d9d9e599c57e4f9ff427263262aaa",
    previous.CAPTURE: previous.PINS[previous.CAPTURE],
}
CONFIG = dict(seed=7098209, intended=128, arms=n.ARMS, labels_opened=False)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counters distinguish completed scoring from a waiting child."""
    print(f"[exp8209] phase={phase} completed={completed} pending={pending}", flush=True)


def fixture() -> Json:
    """Zero-weight public fixtures qualify sealing without fitting or model work."""
    old = previous.fixture()
    for row in old["comparator"]:
        row["action"] = n.rule.base.action(row["p"])
    geometry = n.rule.base.geometry(n.np.zeros((16, 16)), [f"fit-{i}" for i in range(16)])
    heads = [
        dict(arm=a, geometry=geometry, weights=[0.0] * 17, temperature=1.0) for a in n.rule.ARMS
    ]
    frozen = dict(
        heads=heads,
        baseline=old["frozen"]["controls"],
        roles=old["roles"],
        selected_simple_control=dict(selected="logistic"),
    )
    roster = [
        dict(
            (k, r[k])
            for k in ("unit_id", "source_cluster_id", "slot", "status", "exclusion_reason")
        )
        | dict(numerator=1)
        for r in old["features"]
    ]
    return dict(
        frozen=frozen,
        frozen_content_sha256=canonical_hash(frozen),
        roster=roster,
        features=old["features"],
        comparator=old["comparator"],
    )


def measure(
    root: Path,
    raw: Path,
    *,
    fixture: bool = False,
    stream_path: Path | None = None,
    mutation: str = "",
) -> Json:
    """Authenticate branch custody before opening feature-only original evidence."""
    began, wall = time.monotonic_ns(), time.time_ns()
    raw.mkdir(parents=True, exist_ok=True)
    work: Json = dict(
        checks=[],
        refs=[],
        raw_shard_hashes=[],
        evidence={},
        owned_failure="",
        precondition_receipts=[],
        historical_model_provenance={},
        label_access_ledger=[],
    )
    gate = lambda path, key, expected, observed: fit.gate(work, path, key, expected, observed)
    read = lambda ref: fit.bind(work, ref, raw)
    progress("before_preconditions")
    try:
        gate(Path(sys.executable), "python_supported", True, sys.version_info >= (3, 11))
        probe = raw / ".probe"
        probe.write_bytes(b"private writable scratch")
        gate(
            raw, "private_scratch_writable", True, probe.read_bytes() == b"private writable scratch"
        )
        probe.unlink()
        if mutation:
            gate(root / UPSTREAM, "action_fit_ready_score", 1, None)
        if fixture:
            data = globals()["fixture"]()
        else:
            receipt = upstream.methods.supervisor.child(
                "python_environment",
                [sys.executable, "--version"],
                raw / "preconditions",
                deadline=30,
            )
            work["precondition_receipts"].append(receipt)
            gate(Path(sys.executable), "python_environment_exit", 0, receipt["exit_code"])
            values = {}
            for name, pin in PINS.items():
                value = read(dict(path=str(root / name), sha256=pin))
                gate(root / name, "input_json_object", True, isinstance(value, dict))
                for key, expected in [
                    ("required_checks_passed", True),
                    ("flagged_adversarial", False),
                    (
                        "action_fit_ready_score"
                        if name == UPSTREAM
                        else "evaluation_capture_ready_score",
                        1,
                    ),
                ]:
                    gate(root / name, key, expected, value.get(key))
                report = read_bound_sidecar(root / name, fit.historical.publication_sidecar(value))
                gate(root / name, "terminal_publication_passed", True, report["report"]["passed"])
                values[name] = value
            trained, capture = values[UPSTREAM], values[previous.CAPTURE]
            frozen = read(
                dict(path=trained["frozen_heads_path"], sha256=trained["frozen_heads_sha256"])
            )
            protocol = read(
                dict(path=str(ROOT / upstream.methods.PROTOCOL), sha256=upstream.methods.PIN)
            )
            gate(
                ROOT / upstream.methods.PROTOCOL,
                "frozen_reserved_roles",
                sorted(protocol["role_manifest"]["reserved"], key=lambda r: r["source_cluster_id"]),
                sorted(frozen["roles"]["reserved"], key=lambda r: r["source_cluster_id"]),
            )
            roster = deepcopy(capture["rows"])
            n.roster_check(roster, frozen["roles"])
            atomic_json(raw / "original_roster.json", dict(rows=roster))
            work["label_access_ledger"].append(
                dict(
                    event="roster_frozen",
                    monotonic_ns=time.monotonic_ns(),
                    labels_opened=False,
                    sha256=sha256_file(raw / "original_roster.json"),
                )
            )
            progress("original_roster_frozen", 128, 0)
            shards = {Path(r["path"]).stem: r for r in capture["raw_shard_hashes"]}
            features = read(shards["complete_features"])["feature_rows"]
            prior = read(shards["sealed_predictions"])["rows"]
            n.reject_labels([roster, features, prior])
            data = dict(
                frozen=frozen,
                frozen_content_sha256=canonical_hash(frozen),
                roster=roster,
                features=features,
                comparator=[r for r in prior if r["arm"] == "radial16"],
            )
            work["historical_model_provenance"] = dict(
                experiment_id=8184,
                model_invocation_counts=capture["model_invocation_counts"],
                primitive_call_manifest=shards["primitive_calls"],
                charged_current_model_work=0,
            )
            gate(
                Path(shards["primitive_calls"]["path"]),
                "historical_call_manifest_sha256",
                shards["primitive_calls"]["sha256"],
                sha256_file(Path(shards["primitive_calls"]["path"])),
            )
        work["label_access_ledger"].append(
            dict(
                event="target_free_features_opened",
                monotonic_ns=time.monotonic_ns(),
                labels_opened=False,
            )
        )
        progress("after_preconditions", 128, 0)
        work["owned_failure"] = "scoring_incomplete"
        progress("before_benchmark_frozen_scoring", 0, 128)
        atomic_json(raw / "adversarial_permissions.json", dict(rows=n.subset_probe()))
        reduced = n.reduce(data)
        progress("after_benchmark_frozen_scoring", 128, 0)
        for name, value in [
            ("primitive_evidence", data),
            ("original_roster", dict(rows=data["roster"])),
            ("frozen_heads", data["frozen"]),
            ("sealed_predictions", dict(rows=reduced["prediction_rows"], labels_opened=False)),
            ("independent_reduction", reduced),
        ]:
            atomic_json(raw / (name + ".json"), value)
            (raw / (name + ".json")).chmod(0o444)
        work["label_access_ledger"].append(
            dict(
                event="predictions_sealed",
                monotonic_ns=time.monotonic_ns(),
                labels_opened=False,
                predictions_sha256=sha256_file(raw / "sealed_predictions.json"),
            )
        )
        atomic_json(
            raw / "label_access_ledger.json",
            dict(rows=work["label_access_ledger"], evaluator_targets_opened=False),
        )
        (raw / "label_access_ledger.json").chmod(0o444)
        work["raw_shard_hashes"] = [
            reference(raw / (p + ".json"))
            for p in [
                "primitive_evidence",
                "original_roster",
                "frozen_heads",
                "sealed_predictions",
                "independent_reduction",
                "label_access_ledger",
            ]
        ]
        work["raw_shard_hashes"].append(reference(raw / "adversarial_permissions.json"))
        work.update(evidence=data, owned_failure="")
        progress("predictions_sealed", len(reduced["prediction_rows"]), 0)
    except (OSError, ValueError, KeyError, TypeError, StopIteration) as error:
        if work["owned_failure"]:
            work["owned_failure"] = str(error)
        elif all(c["passed"] for c in work["checks"]):
            work["checks"].append(
                dict(
                    check="input_structure",
                    path=str(root / UPSTREAM),
                    upstream=UPSTREAM,
                    hash=None,
                    artifact_field="input_structure",
                    op="==",
                    expected="authenticated_public_reserved_evidence",
                    observed=str(error),
                    passed=False,
                )
            )
        progress("terminal_operand_or_scoring_failure", len(work["checks"]), 0)
    ended = time.monotonic_ns()
    work.update(
        duration_s=(ended - began) / 1e9,
        measurement_clocks=dict(
            started_monotonic_ns=began, ended_monotonic_ns=ended, started_wall_ns=wall
        ),
        code_config_hashes=[
            reference(ROOT / p)
            for p in [
                *OWNED,
                TEST,
                upstream.methods.PROTOCOL,
                "python/carnot/verify/restricted_action_rule_8207.py",
                "python/carnot/verify/evidence_energy_8154.py",
            ]
        ],
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json], *, fixture: bool = False) -> Json:
    """Readiness certifies custody and validation; it grants no utility claim."""
    checked = bool(receipts) and all(r["passed"] for r in receipts) and not work["owned_failure"]
    failures = [c for c in work["checks"] if not c["passed"]]
    reduced = (
        n.reduce(work["evidence"])
        if work["evidence"]
        else dict(
            rows=[],
            prediction_rows=[],
            intended_count=128,
            independent_count=0,
            completed_count=0,
            failed_count=0,
            censored_count=0,
            excluded_count=128,
            acceptance_subset_violations=[],
            equivalent_logistic_parity=dict(passed=False, rows=[]),
        )
    )
    ready = int(checked and not failures and len(reduced["prediction_rows"]) == 768)
    verdict = (
        "disqualified"
        if not checked
        else "blocked"
        if failures
        else "circular_positive"
        if fixture
        else "null"
    )
    predictions = raw / "sealed_predictions.json"
    value: Json = dict(
        experiment_id=8209,
        task_id=TASK,
        milestone="2026.10.709",
        run_date=RUN_DATE,
        honest_verdict="complete_" + verdict + "_restricted_predictions_sealed",
        verdict_class=verdict,
        gate_check_summary=work["checks"],
        preconditions_checked=[
            dict(resource=c["check"], available=c["passed"], path=c["path"]) for c in work["checks"]
        ],
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=dict(
            model_loads=0, generate_calls=0, forward_calls=0, live_model_calls=0, model_count=0
        ),
        call_ledger=[],
        duration_s=work["duration_s"],
        random_seed=CONFIG["seed"],
        source_artifact_hashes=work["refs"],
        cited_upstream_artifacts=[
            dict(
                path=p,
                sha256=s,
                fields_imported=[
                    "readiness",
                    "frozen_heads",
                    "original_public_roster_and_features",
                ],
            )
            for p, s in PINS.items()
        ]
        if not fixture
        else [],
        verifier_is_oracle=fixture,
        exposure_scope="oracle_fixture_only"
        if fixture
        else "exposed_development_within_run_prediction_label_boundary",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        acceptance_gates=dict(
            external_inputs=not failures,
            all_slots_sealed=bool(ready),
            owned_validation=bool(checked),
            acceptance_subset=not reduced["acceptance_subset_violations"],
            evaluator_targets_unopened=True,
        ),
        required_checks_passed=bool(checked),
        flagged_adversarial=False,
        validation_receipts=receipts,
        precondition_receipts=work["precondition_receipts"],
        raw_shard_hashes=work["raw_shard_hashes"],
        code_config_hashes=work["code_config_hashes"],
        sealed_action_ready_score=ready,
        predictions_path=str(predictions) if predictions.is_file() else None,
        predictions_sha256=sha256_file(predictions) if predictions.is_file() else None,
        evaluator_targets_opened=False,
        label_access_ledger=work["label_access_ledger"],
        historical_model_provenance=work["historical_model_provenance"],
        selected_simple_control=work["evidence"].get("frozen", {}).get("selected_simple_control"),
        trained_head_specs=[
            dict(
                arm=h["arm"],
                head_sha256=canonical_hash(h),
                trained_in_current_run=False,
                imported_from_experiment=8208,
            )
            for h in work["evidence"].get("frozen", {}).get("heads", [])
        ],
        optional_sibling_disposition="8196_is_historical_context_not_a_readiness_gate",
        measurement_reference=reference(raw / "measurement.json"),
        measurement_clocks=work["measurement_clocks"],
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        repository_health=work.get("global_health", {}),
        sample_size_budget=CONFIG,
        methodology_note="Frozen CPU energy and matched additive/logistic heads on cached target-free features. Tune-selected logistic is the primary simple control; equivalent logistic certifies arithmetic only. Original baseline and always-escalate references are descriptive. Original97 complete,30 failed,1 excluded slots remain; missing evidence escalates in every arm. No evaluator labels or current LLM calls; cached Qwen acquisition is historical. Independent utility belongs to8210.",
        claim_scope="Within-run immutable prediction custody on exposed development; scientific benefit unmeasured.",
        **reduced,
    )
    value = normalize_artifact_for_template_write(value)
    value["field_principles"] = {
        k: "Bind exact invocation, authentic primitive rows, frozen inputs and checked exits; exposed development cannot establish independent benefit."
        for k in value
    }
    value["field_principles"].update(
        sealed_action_ready_score="All128 slots represented and sealed before current evaluator access, with passed owned validation.",
        label_access_ledger="Monotonic read/seal events include no evaluator access; future evaluators must bind sealed bytes.",
        independent_count="Unique complete sources; arms and repeated calculations are never independent sources.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Rehash custody and reconstruct sealed rows in a fresh evaluator-free process."""
    try:
        value = json.loads(path.read_bytes())
        checksum = value.pop("reproducibility_checksum")
        if canonical_hash(value) != checksum:
            return False
        for ref in [
            value["measurement_reference"],
            *value["source_artifact_hashes"],
            *value["raw_shard_hashes"],
            *value["code_config_hashes"],
        ]:
            if sha256_file(Path(ref["path"])) != ref["sha256"]:
                return False
        historical = value["historical_model_provenance"].get("primitive_call_manifest")
        if historical and sha256_file(Path(historical["path"])) != historical["sha256"]:
            return False
        for receipt in [*value["validation_receipts"], *value["precondition_receipts"]]:
            for label in ("stdout", "stderr"):
                if (
                    label + "_path" in receipt
                    and sha256_file(Path(receipt[label + "_path"])) != receipt[label + "_sha256"]
                ):
                    return False
        work = json.loads(Path(value["measurement_reference"]["path"]).read_bytes())
        raw = Path(value["terminal_validation_sidecar_path"]).parent
        if work["evidence"]:
            if not value["verifier_is_oracle"]:
                copies = {Path(r["path"]).name.split("-", 1)[1]: r for r in work["refs"]}
                if any(copies[Path(p).name]["sha256"] != pin for p, pin in PINS.items()):
                    return False
                load = lambda name: json.loads(Path(copies[name]["path"]).read_bytes())
                capture = load(Path(previous.CAPTURE).name)
                trained = load(Path(UPSTREAM).name)
                frozen = copies[Path(trained["frozen_heads_path"]).name]
                if frozen["sha256"] != trained["frozen_heads_sha256"] or work["evidence"][
                    "frozen"
                ] != load(Path(trained["frozen_heads_path"]).name):
                    return False
                if (
                    work["evidence"]["roster"] != capture["rows"]
                    or work["evidence"]["features"]
                    != load("complete_features.json")["feature_rows"]
                    or work["evidence"]["comparator"]
                    != [
                        r for r in load("sealed_predictions.json")["rows"] if r["arm"] == "radial16"
                    ]
                ):
                    return False
            reduced = n.reduce(work["evidence"])
            for name, expected in [
                ("primitive_evidence", work["evidence"]),
                ("original_roster", dict(rows=work["evidence"]["roster"])),
                ("frozen_heads", work["evidence"]["frozen"]),
                ("independent_reduction", reduced),
                ("sealed_predictions", dict(rows=reduced["prediction_rows"], labels_opened=False)),
                (
                    "label_access_ledger",
                    dict(rows=work["label_access_ledger"], evaluator_targets_opened=False),
                ),
                ("adversarial_permissions", dict(rows=n.subset_probe())),
            ]:
                if json.loads((raw / (name + ".json")).read_bytes()) != expected:
                    return False
        return build(
            work, raw, value["validation_receipts"], fixture=value["verifier_is_oracle"]
        ) == dict(value, reproducibility_checksum=checksum)
    except (OSError, ValueError, KeyError, TypeError):
        return False


def manifest(private: Path, candidate: Path) -> Json:
    """Freeze qualified checks, exact paths and full-suite health before scoring."""
    with (
        patch.object(upstream, "OWNED", OWNED),
        patch.object(upstream, "TEST", TEST),
        patch.object(upstream, "CLI", CLI),
        patch.object(execution, "e", sys.modules[__name__]),
    ):
        specs = upstream.manifest(private, candidate)
    for spec in specs["commands"]:
        if spec["name"] == "consumer_and_E2E015_019":
            spec["argv"].insert(-1, "tests/python/test_restricted_energy_fit_8208.py")
    return specs


def main(argv: list[str] | None = None) -> int:
    """Qualified supervision and unchanged terminal validators own publication."""
    with (
        patch.object(execution, "e", sys.modules[__name__]),
        patch.object(execution, "OWNED", OWNED),
        patch.object(execution, "manifest", manifest),
        patch.object(execution, "run_check", upstream.methods.run_check),
    ):
        return int(execution.main(argv))
