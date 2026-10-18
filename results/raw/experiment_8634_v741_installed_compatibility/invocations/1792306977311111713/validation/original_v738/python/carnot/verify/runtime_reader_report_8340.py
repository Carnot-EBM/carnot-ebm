"""REQ-REPORT-8340: bind conclusions to primitives and unchanged terminal checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS
from carnot.reporting.evidence_features_custody_7980 import reference, checked
from carnot.reporting.v710_contract_replay import require_reference, snapshot
from carnot.verify import runtime_reader_8340 as q

Json = dict[str, Any]


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """All intended identity operands survive even when an external resource blocks."""
    passed = bool(receipts) and all(
        r["passed"] for r in receipts if r.get("scope") != "external_preconditions"
    )
    value = q.reduce(work, passed)
    value.update(
        experiment_id=q.EXPERIMENT_ID,
        task_id=q.TASK,
        milestone=q.MILESTONE,
        run_date="20261009",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        no_model_load=True,
        MODEL_SPECS=[],
        model_invocation_counts=dict(ZERO_INVOCATION_COUNTS),
        historical_model_provenance=dict(
            imported_from=q.UPSTREAM,
            current_calls_imported=False,
            model_invocation_counts=work["historical"].get("model_invocation_counts"),
            failure_layer=work["historical"].get("failure_layer", []),
        ),
        verifier_is_oracle=False,
        exposure_scope="exposed development; private controls are constructed mechanics",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        validation_receipts=receipts,
        preconditions_checked=True,
        duration_s=work.get("duration_s", 0),
        phase_spans=work.get("phase_spans", []),
        random_seed=7198340,
        source_artifact_hashes=work["refs"],
        code_config_hashes=[
            snapshot(q.ROOT / p, raw / "code", Path(p).stem)
            for p in q.OWNED
            + [
                q.TEST,
                "python/carnot/verify/runtime_localization_8290.py",
                "python/carnot/verify/runtime_change_boundary_8307.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v709_execution.py",
                "python/carnot/reporting/v718_replay_history.py",
                "python/carnot/reporting/sentence_spline_execution_8334.py",
                "python/carnot/reporting/v719_contract_replay.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        raw_shard_hashes=[reference(raw / "measurement.json"), work["current_reference"]],
        primitive_reference=reference(raw / "measurement.json"),
        terminal_validation_sidecar_path=str(raw / "terminal_validation.json"),
        execution_authority=work.get("authority", {}),
        historical_fixture_hashes=[reference(q.design(q.ROOT, "2026.10.716"))],
        adversarial_findings=work.get("finding_audits", []),
        cited_upstream_artifacts=[
            dict(
                path=r["path"],
                sha256=r.get("sha256"),
                fields_imported=r.get("fields_imported", ["identity and byte custody"]),
            )
            for r in work["refs"]
        ],
        methodology_note="Read full versioned authority, authenticate failed historical execution and compare current causal identity bytes. No unchanged CUDA matrix or scientific measurement runs.",
        fixture_mode=work.get("private_run", False),
        owned_statement_coverage=work.get("owned_statement_coverage", {}),
        prior_validation_attempts=work.get("prior_validation_attempts", []),
    )
    value["field_principles"] = {
        k: "Retain exact invocation, byte custody and measured execution; no scientific benefit follows."
        for k in value
    }
    value["field_principles"].update(
        runtime_reader_ready_score="Complete current reader checks and actual authority are separate from device health.",
        runtime_changed_score="Only available authenticated driver/library/device/lease identity deltas admit a probe.",
        cuda_context_ready_score="A changed environment needs actual leased context, allocation and copy parity.",
        change_evidence="Every intended causal operand retains old/current identities and explicit missingness.",
        historical_fixture_hashes="Preserved V716 design prevents mutable future plans from rewriting old authority.",
        adversarial_findings="Retain parsed findings and dispositions, including rejected false-zero controls.",
        reproducibility_checksum="Bind all candidate fields except this checksum to one canonical digest.",
    )
    value["reproducibility_checksum"] = q.checksum(value)
    return value


def replay(path: Path) -> bool:
    """Cold reduction rejects rehashed headlines and independently binds raw children."""
    try:
        value = json.loads(path.read_bytes())
        if value.get("reproducibility_checksum") != q.checksum(value):
            return False
        if [value.get(k) for k in ("experiment_id", "task_id", "milestone", "run_date")] != [
            q.EXPERIMENT_ID,
            q.TASK,
            q.MILESTONE,
            "20261009",
        ]:
            return False
        for ref in (
            value["source_artifact_hashes"]
            + value["code_config_hashes"]
            + value["raw_shard_hashes"]
            + value["historical_fixture_hashes"]
        ):
            require_reference(ref)
        work = json.loads(checked(value["primitive_reference"]).read_bytes())
        if work.get("baseline_reference"):
            require_reference(work["baseline_reference"])
            baseline = json.loads(Path(work["baseline_reference"]["snapshot_path"]).read_bytes())
            require_reference(work["baseline_source_reference"])
            source = json.loads(
                Path(work["baseline_source_reference"]["snapshot_path"]).read_bytes()
            )
            if (
                baseline != work["previous"]
                or work["baseline_reference"]["sha256"] != source["runtime_binding_sha256"]
            ):
                return False
        if json.loads(checked(work["current_reference"]).read_bytes()) != work["current"]:
            return False
        auth = work.get("authority", {})
        if auth.get("activated"):
            active = auth["authority_snapshots"]["active"]
            require_reference(dict(active, path=active["source_path"]))
            actual = yaml.safe_load(Path(active["snapshot_path"]).read_bytes())
            if actual["tasks"] != auth["tasks"] or actual["milestone"] != q.MILESTONE:
                return False
        for receipt in value["validation_receipts"] + work["observation_receipts"]:
            for stream in ("stdout", "stderr"):
                checked(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
        if work["observation_receipts"]:
            text = Path(work["observation_receipts"][0]["stdout_path"]).read_text()
            devices = [
                dict(
                    zip(
                        ("index", "uuid", "name", "driver_version"),
                        map(str.strip, cols),
                        strict=True,
                    )
                )
                for line in text.splitlines()
                if len(cols := line.split(",")) == 4
            ]
            if devices != work["current"].get("devices"):
                return False
        for row in work["diagnostic"]["rows"]:
            receipt = row["receipt"]
            for stream in ("stdout", "stderr"):
                checked(dict(path=receipt[stream + "_path"], sha256=receipt[stream + "_sha256"]))
            if (
                receipt["passed"]
                and json.loads(Path(receipt["stdout_path"]).read_text().splitlines()[-1])
                != row["primitive"]
            ):
                return False
        passed = bool(value["validation_receipts"]) and all(
            r["passed"]
            for r in value["validation_receipts"]
            if r.get("scope") != "external_preconditions"
        )
        return (
            value["required_checks_passed"] == passed
            and all(value[k] == v for k, v in q.reduce(work, passed).items())
            and not value["MODEL_SPECS"]
            and value["model_invocation_counts"] == ZERO_INVOCATION_COUNTS
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
