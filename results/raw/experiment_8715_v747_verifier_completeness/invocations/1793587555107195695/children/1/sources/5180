"""REQ-REPORT-8335: seal cached development decisions before evaluator use."""

from __future__ import annotations

from datetime import datetime, timezone, UTC
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any
import yaml

from carnot.reporting import sentence_spline_fit_8334 as upstream
from carnot.reporting.current_work_receipt import (
    atomic_json,
    canonical_hash,
    sha256_file,
    ZERO_INVOCATION_COUNTS,
)
from carnot.reporting.primary_publication import read_bound_sidecar, validate_primary
from carnot.reporting.roadmap_contract import compare_contract
from carnot.verify.reserved_prediction_seal_8335 import (
    ALLOWLIST,
    ARMS,
    validate_bundle,
    score,
    parity,
)

Json = dict[str, Any]
ROOT = upstream.ROOT
NAME = "experiment_8335_v719_reserved_prediction_seal"
TASK = "exp8335-reserved-prediction-seal"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_reserved_prediction_seal_8335.py"
OWNED = [
    "python/carnot/verify/reserved_prediction_seal_8335.py",
    "python/carnot/reporting/reserved_prediction_seal_8335.py",
    "python/carnot/reporting/reserved_prediction_execution_8335.py",
    CLI,
]
HEAD = "results/experiment_8334_v719_sentence_spline_fit.json"
SOURCE, PROTOCOL = upstream.SOURCE, upstream.PROTOCOL
PINS = dict(
    upstream.PINS,
    **{HEAD: "sha256:a63db57cb7f735a1171ce16b87d09aa02abe69c7f7abb6fc6ca5fa6240d59208"},
)
reference = upstream.reference
MODEL_SPECS: list[Json] = []


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts let the conductor observe bounded real work."""
    print(f"[exp8335] phase={phase} completed={completed} pending={pending}", flush=True)


def immutable(path: Path, value: Json) -> Json:
    """Exclusive creation prevents a later invocation from rewriting a seal."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    path.chmod(0o444)
    return reference(path)


def measure(root: Path, raw: Path) -> Json:
    """Authenticate public operands, then dispatch a child with no target paths."""
    from carnot.reporting import reserved_prediction_execution_8335 as runner

    began = time.monotonic()
    raw.mkdir(parents=True, exist_ok=True)
    raw.chmod(0o700)
    work: Json = dict(
        failures=[],
        gates=[],
        refs=[],
        bundle={},
        predictions=[],
        child_receipts=[],
        issued_at=datetime.now(UTC).isoformat(),
        parity=dict(passed=False),
    )

    def require(path: Path, field: str, expected: Any, observed: Any) -> None:
        gate = dict(
            upstream=path.stem,
            path=str(path),
            hash=sha256_file(path) if path.is_file() else None,
            artifact_field=field,
            op="==",
            expected=expected,
            observed=observed,
            passed=expected == observed,
        )
        work["gates"].append(gate)
        if not gate["passed"]:
            work["failures"].append(gate)
            raise ValueError(field)

    def bind(ref: Json) -> Json:
        path = Path(ref["path"])
        require(path, "sha256", ref["sha256"], sha256_file(path) if path.is_file() else None)
        dest = raw / "inputs" / (ref["sha256"][7:] + "-" + path.name)
        dest.parent.mkdir(exist_ok=True)
        dest.write_bytes(path.read_bytes())
        dest.chmod(0o400)
        work["refs"].append(reference(dest))
        return dict(json.loads(dest.read_bytes()))

    progress("before_authentication")
    try:
        require(
            raw,
            "private_resources",
            True,
            raw.stat().st_mode & 0o077 == 0 and shutil.disk_usage(raw).free > 1_000_000_000,
        )
        for tool in ["python", "pytest", "coverage", "ruff", "mypy"]:
            require(
                ROOT / ".venv/bin" / tool,
                "executable",
                True,
                os.access(ROOT / ".venv/bin" / tool, os.X_OK),
            )
        head, source, protocol = [
            bind(dict(path=str(root / p), sha256=PINS[p])) for p in [HEAD, SOURCE, PROTOCOL]
        ]
        for value, name in [(head, HEAD), (source, SOURCE)]:
            validate_primary(value, root / name)
            terminal = bind(reference(Path(value["terminal_validation_sidecar_path"])))
            side = Path(terminal["publication"]["sidecar_path"])
            report = read_bound_sidecar(root / name, side)
            bind(reference(side))
            require(side, "terminal_passed", True, report["report"]["passed"])
            require(
                root / name,
                "terminal_primary_sha256",
                PINS[name],
                terminal["publication"]["primary_sha256"],
            )
            require(root / name, "required_checks_passed", True, value["required_checks_passed"])
            require(root / name, "flagged_adversarial", False, value["flagged_adversarial"])
        require(root / HEAD, "heads_ready_score", 1, head["heads_ready_score"])
        seal = head["frozen_head_manifest"]
        require(
            root / SOURCE,
            "authenticated_by_head",
            PINS[SOURCE],
            seal["source_manifest_reference"]["sha256"],
        )
        design, active = (
            root / "openspec/change-proposals/research-roadmap-vNEXT.md",
            root / "research-roadmap.yaml",
        )
        activated = yaml.safe_load(active.read_bytes())
        authority = compare_contract(
            design.read_text(), activated, activated, milestone="2026.10.719", first_id=8332
        )
        require(active, "fourteen_task_agreement", True, authority["passed"])
        work["refs"] += [reference(design), reference(active)]
        slots = bind(source["predictor_shards"]["reserved"])["rows"]
        bundle = dict(
            slots=slots,
            head_manifest=seal,
            head_hash=seal["whole_model_sha256"],
            protocol_hash=PINS[PROTOCOL],
            roster=protocol["original_roles"]["evaluation"],
        )
        validate_bundle(bundle)
        work.update(bundle=bundle, historical=source["historical_model_provenance"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not work["failures"]:
            gate = dict(
                upstream="source_contract",
                path=str(root),
                hash=None,
                artifact_field="structure",
                op="==",
                expected="authenticated_predictor_only_operands",
                observed=str(error),
                passed=False,
            )
            work["gates"].append(gate)
            work["failures"].append(gate)
    progress("after_authentication")
    if not work["failures"]:
        input_ref = immutable(raw / "prediction_input.json", work["bundle"])
        work["input_reference"] = input_ref
        receipt = runner.check(
            dict(
                name="predictor_only_child",
                argv=[
                    str(ROOT / ".venv/bin/python"),
                    "-u",
                    str(ROOT / CLI),
                    "--score-child",
                    input_ref["path"],
                    str(raw / "child_predictions.json"),
                    work["issued_at"],
                ],
                deadline_s=60,
            ),
            raw / "logs",
        )
        work["child_receipts"].append(receipt)
        if receipt["passed"]:
            work["predictions"] = json.loads((raw / "child_predictions.json").read_bytes())["rows"]
            work["parity"] = parity(work["bundle"], work["issued_at"], work["predictions"])
    return finish(work, raw, began)


def finish(work: Json, raw: Path, began: float) -> Json:
    """Separate retention predictions from stream inputs before any audit."""
    progress("before_immutable_seal")
    work["prediction_refs"] = [
        immutable(
            raw / (role + "_predictors.json"),
            dict(
                rows=[r for r in work["predictions"] if r["role"] == role],
                evaluator_labels_opened=False,
            ),
        )
        for role in ["stream", "retention"]
    ]
    manifest = dict(
        issued_at=work["issued_at"],
        intended_slots=128,
        intended_rows=128 * len(ARMS),
        accounted_rows=len(work["predictions"]),
        files=work["prediction_refs"],
        input_reference=work.get("input_reference"),
        head_hash=work["bundle"].get("head_hash"),
        protocol_hash=PINS[PROTOCOL],
        evaluator_labels_opened=False,
    )
    work["seal_reference"] = immutable(raw / "prediction_seal.json", manifest)
    work["code_refs"] = [
        reference(ROOT / p)
        for p in [
            *OWNED,
            TEST,
            "python/carnot/verify/sentence_spline_fit_8334.py",
            "python/carnot/verify/local_update_isolation_8306.py",
            "python/carnot/reporting/sentence_spline_execution_8334.py",
            "python/carnot/reporting/v718_replay_history.py",
            "scripts/adversarial_verify.py",
        ]
    ]
    work["duration_s"] = time.monotonic() - began
    work["phase_spans"] = [
        dict(phase="authenticate_predict_and_seal", start_s=0.0, duration_s=work["duration_s"])
    ]
    atomic_json(raw / "measurement.json", work)
    progress(
        "after_immutable_seal", len(work["predictions"]), 128 * len(ARMS) - len(work["predictions"])
    )
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Prediction custody has no reserved science outcome and zero generalization."""
    coverage = work.get("owned_coverage_reference")
    coverage_ok = True
    if coverage:
        covered = json.loads(Path(coverage["path"]).read_bytes())
        coverage_ok = covered["totals"]["percent_covered"] == 100 and all(
            p in covered["files"] and covered["files"][p]["summary"]["missing_lines"] == 0
            for p in OWNED
        )
    checks = [*receipts, *work["child_receipts"]]
    owned = bool(receipts) and all(r["passed"] for r in checks) and coverage_ok
    ready = int(
        owned
        and not work["failures"]
        and work["parity"]["passed"]
        and len(work["predictions"]) == 768
    )
    klass = (
        "disqualified"
        if not owned or (not work["failures"] and not ready)
        else "blocked"
        if work["failures"]
        else "null"
    )
    slots = work["bundle"].get("slots", [])
    source_rows = [
        dict(
            slot=i + 1,
            unit_id=slots[i]["unit_id"] if slots else None,
            source_cluster_id=slots[i]["source_cluster_id"] if slots else None,
            status="completed" if work["predictions"] and slots[i]["x"] is not None else "excluded",
            prediction_count=6 if work["predictions"] else 0,
            p=work["predictions"][i * 6]["p"] if work["predictions"] else None,
        )
        for i in range(128)
    ]
    done = sum(r["status"] == "completed" for r in source_rows)
    value = dict(
        experiment_id=8335,
        task_id=TASK,
        milestone="2026.10.719",
        run_date="20261009",
        honest_verdict="complete_blocked_" + work["failures"][0]["upstream"]
        if klass == "blocked"
        else "complete_" + klass + "_reserved_prediction_seal",
        verdict_class=klass,
        gate_check_summary=work["gates"],
        predictions_ready_score=ready,
        inference_substrate="verifier_ensemble_against_cached_candidates",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=ZERO_INVOCATION_COUNTS,
        historical_model_provenance=work.get("historical", []),
        rows=source_rows,
        intended_count=128,
        completed_count=done,
        failed_count=0,
        censored_count=0,
        excluded_count=128 - done,
        independent_count=len(slots),
        sample_size_budget=dict(stream=96, retention=32, arms=6, intended_predictions=768),
        verifier_is_oracle=False,
        exposure_scope="exposed_cached_development",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=owned,
        flagged_adversarial=not owned,
        acceptance_gates=dict(
            custody=not work["failures"], independent_replay=work["parity"]["passed"], owned=owned
        ),
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        adversarial_findings=work.get("finding_audits", []),
        preconditions_checked=True,
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7178308,
        source_artifact_hashes=work["refs"],
        code_config_hashes=work["code_refs"],
        raw_shard_hashes=[*work["prediction_refs"], work["seal_reference"]],
        cited_upstream_artifacts=[
            dict(
                r,
                fields_imported=[
                    "frozen heads",
                    "public reserved features",
                    "protocol and custody",
                ],
            )
            for r in work["refs"]
        ],
        prediction_manifest=work["seal_reference"],
        prediction_rows=work["predictions"],
        label_access_ledger=[
            dict(
                phase="authenticate_predict_seal_replay",
                evaluator_access_count=0,
                retention_labels_available_to_online_learning=False,
            )
        ],
        stream_predictor_path=work["prediction_refs"][0]["path"],
        retention_predictor_path=work["prediction_refs"][1]["path"],
        sealed_head_hash=work["bundle"].get("head_hash"),
        sealed_protocol_hash=PINS[PROTOCOL],
        input_allowlist=ALLOWLIST,
        selected_comparator=work["bundle"].get("head_manifest", {}).get("selected_comparator"),
        prediction_parity=work["parity"],
        child_receipts=work["child_receipts"],
        future_audit_dependencies=[
            dict(
                path="results/experiment_8336_v719_continuous_local_learning.json",
                field="trajectory_ready_score",
                op="==",
                expected=1,
            )
        ],
        methodology_note="Reconstruct exposed cached development predictions from unchanged V717 heads and source roster. Missing features escalate with null probabilities. Six expressions share128 sources; sigmoid34 is exactly spline34, not independent evidence. No evaluator targets are opened. Static audit awaits both seals; no H1/H2 test occurs here.",
        work_reference=reference(raw / "measurement.json"),
        owned_coverage_reference=coverage,
        execution_manifest_reference=work.get("execution_manifest_reference"),
        invocation_argv=work.get("invocation_argv", []),
    )
    value["field_principles"] = {
        k: "Bind "
        + k
        + " to actual predictor custody, original source slots and exposed-development scope."
        for k in value
    }
    value["field_principles"]["reproducibility_checksum"] = (
        "Bind all semantic fields except this checksum."
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Cold replay rejects rehashed changes by rebuilding from pinned operands."""
    try:
        value = json.loads(path.read_bytes())
        ref = value["work_reference"]
        if sha256_file(Path(ref["path"])) != ref["sha256"]:
            return False
        work = json.loads(Path(ref["path"]).read_bytes())
        refs = [*work["refs"], *work["code_refs"], *work["prediction_refs"], work["seal_reference"]]
        if work.get("input_reference"):
            refs.append(work["input_reference"])
        if any(sha256_file(Path(r["path"])) != r["sha256"] for r in refs):
            return False
        if work["bundle"]:
            copied = {r["sha256"]: Path(r["path"]) for r in work["refs"]}
            head, source, protocol = [
                json.loads(copied[PINS[p]].read_bytes()) for p in [HEAD, SOURCE, PROTOCOL]
            ]
            seal = head["frozen_head_manifest"]
            expected = dict(
                slots=json.loads(
                    copied[source["predictor_shards"]["reserved"]["sha256"]].read_bytes()
                )["rows"],
                head_manifest=seal,
                head_hash=seal["whole_model_sha256"],
                protocol_hash=PINS[PROTOCOL],
                roster=protocol["original_roles"]["evaluation"],
            )
            if expected != work["bundle"]:
                return False
            validate_bundle(expected)
            if work["predictions"]:
                if score(expected, work["issued_at"]) != work["predictions"]:
                    return False
                if parity(expected, work["issued_at"], work["predictions"]) != work["parity"]:
                    return False
        for r, role in zip(work["prediction_refs"], ["stream", "retention"], strict=True):
            if json.loads(Path(r["path"]).read_bytes()) != dict(
                rows=[p for p in work["predictions"] if p["role"] == role],
                evaluator_labels_opened=False,
            ):
                return False
        manifest = json.loads(Path(work["seal_reference"]["path"]).read_bytes())
        if manifest != dict(
            issued_at=work["issued_at"],
            intended_slots=128,
            intended_rows=768,
            accounted_rows=len(work["predictions"]),
            files=work["prediction_refs"],
            input_reference=work.get("input_reference"),
            head_hash=work["bundle"].get("head_hash"),
            protocol_hash=PINS[PROTOCOL],
            evaluator_labels_opened=False,
        ):
            return False
        return bool(build(work, Path(ref["path"]).parent, value["validation_receipts"]) == value)
    except (OSError, ValueError, KeyError, TypeError, IndexError):
        return False
