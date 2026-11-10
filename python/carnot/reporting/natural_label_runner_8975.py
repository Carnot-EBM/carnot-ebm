"""REQ-REPORT-8975: publish a reproducible access decision without loading a model."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from carnot.reporting import natural_label_eligibility_8975 as e
from carnot.reporting.current_work_receipt import (
    atomic_json,
    build_current_work_receipt,
    canonical_hash,
    sha256_file,
    validate_current_work_receipt,
)
from carnot.reporting.primary_publication import publish_primary
from carnot.reporting.v709_execution import child
from scripts.experiment_template import ExperimentTemplate

Json = dict[str, Any]
CACHE = Path.home() / ".cache/carnot8975-release"


def read_ref(reference: Json) -> Any:
    """Verify original bytes before trusting any label, permission or source field."""
    path = Path(reference["path"])
    if not path.is_file():
        raise FileNotFoundError("missing_file:" + str(path))
    if e.ref(path) != reference:
        raise ValueError("changed_bytes:" + str(path))
    return json.loads(path.read_bytes())


def prepare_manifest(cache: Path) -> Path:
    """Use only cached public metadata; an absent local index never triggers a corpus download."""
    cache.mkdir(parents=True, exist_ok=True, mode=0o700)
    roots = [Path.home() / ".cache/huggingface/hub", Path.home() / ".cache/huggingface/datasets"]
    found = [str(p) for root in roots for p in root.glob("*lmsys*")]
    manifest = cache / "input_manifest.json"
    atomic_json(
        manifest,
        dict(
            scope="author_release",
            release_revision=e.REVISION,
            files={key: e.ref(cache / (key + ".body")) for key in e.PINNED},
            source_access={},
            source_access_search=dict(
                roots=[str(p) for p in roots],
                matching_cache_entries=found,
                authorized_index=None,
                source_records_opened=0,
                status="no_authorized_index" if found else "missing_local_cache",
                new_agreements_accepted=False,
                dataset_download_attempted=False,
                access_token_contents_read=False,
                missing_fields=[
                    "authorized_source_index",
                    "local_use_authorization",
                    "model_specific_terms",
                    "conversation",
                    "model",
                    "source_offset",
                ],
            ),
            retrieval_receipt=e.ref(cache / "retrievals.json"),
            source_urls=dict(
                release="https://github.com/TAI-HAMBURG/AuthenHallu/tree/" + e.REVISION,
                paper="https://arxiv.org/html/2510.10539v2",
                terms="https://huggingface.co/datasets/lmsys/lmsys-chat-1m",
            ),
        ),
    )
    return manifest


def inspect(manifest: Path) -> Json:
    """A fixture may exercise publication but cannot authenticate a real natural corpus."""
    value = json.loads(manifest.read_bytes())
    if value["release_revision"] != e.REVISION:
        raise ValueError("release_revision")
    for key, reference in value["files"].items():
        path = Path(reference["path"])
        if not path.is_file():
            raise FileNotFoundError("missing_file:" + str(path))
        if e.ref(path) != reference:
            raise ValueError("changed_bytes:" + str(path))
        if value["scope"] != "private_fixture" and reference["sha256"] != "sha256:" + e.PINNED[key]:
            raise ValueError("pinned_release_hash:" + key)
    if value["scope"] != "private_fixture" and set(value["files"]) != set(e.PINNED):
        raise ValueError("missing_release_fields")
    labels = read_ref(value["files"]["labels"])
    access = value.get("source_access", {})
    if access.get("local_use_authorized") is True and "records_reference" in access:
        if Path(access["records_reference"]["path"]).stat().st_size > 10_000_000:
            raise ValueError("unindexed_source_exceeds_budget")
        access = dict(access, records=read_ref(access["records_reference"]))
    if value["scope"] == "private_fixture":
        access = {}
    reduced = e.audit(labels, access, e.ref(manifest))
    reduced.update(
        release_revision=e.REVISION,
        input_scope=value["scope"],
        source_artifact_hashes=[e.ref(manifest), *value["files"].values()],
        source_access_search=value.get("source_access_search", {}),
        license_reference=dict(
            author_release=value["files"].get("license"),
            source_terms=value["files"].get("terms"),
            author_license="MIT, copyright2026 yujie-ren",
            source_license="LMSYS-Chat-1M Dataset License Agreement; model-specific terms also apply.",
            original_text_redistribution_permitted=False,
            local_use_authorized=bool(access.get("local_use_authorized")),
        ),
        source_urls=value.get("source_urls", {}),
    )
    return reduced


def measure(manifest: Path, output: Path, run_date: str, start: int) -> Json:
    """Save immutable primitive operands and actual clocks before building the primary."""
    raw = output.parent / "raw" / e.NAME / "invocations" / str(time.time_ns())
    raw.mkdir(parents=True)
    disk = shutil.disk_usage(raw).free
    memory = os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    e.progress("authenticate_release_and_join")
    template = ExperimentTemplate(
        8975,
        "Natural label eligibility",
        "results/" + output.name,
        repo_root=output.parent,
        seed=8975,
    )
    with template.phase("source_access_audit"):
        try:
            if min(disk, memory) < 1_000_000:
                raise OSError("insufficient_disk_or_memory")
            reduced = inspect(manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reduced = e.audit([], {}, e.ref(manifest))
            reduced.update(
                input_scope="external_unavailable",
                release_revision=e.REVISION,
                source_artifact_hashes=[e.ref(manifest)],
                license_reference={},
                source_access_search={},
                source_urls={},
            )
            reduced["gate_check_summary"].append(
                e.gate(
                    "authenticated_inputs",
                    "pinned release and explicit source authority",
                    type(error).__name__ + ":" + str(error),
                    e.ref(manifest),
                )
            )
    end = time.monotonic_ns()
    receipt = build_current_work_receipt(
        run_id=raw.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        inference_substrate_details=dict(
            operation="release/annotation/source-access audit", current_LLM_calls=0
        ),
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=[dict(phase="preconditions_and_audit", start_s=0, end_s=(end - start) / 1e9)],
    )
    history = e.ROOT / "results/experiment_8908_v762_natural_criterion_audit.json"
    cited = []
    if history.is_file():
        saved = raw / history.name
        saved.write_bytes(history.read_bytes())
        old = json.loads(saved.read_bytes())
        cited.append(
            dict(
                experiment_id=8908,
                fields_imported=["honest_verdict", "verdict_class"],
                **e.ref(saved),
                imported_values={k: old[k] for k in ("honest_verdict", "verdict_class")},
                scope="historical_context_only; no runtime gate",
            )
        )
    primitive = dict(
        manifest=e.ref(manifest),
        reduction=reduced,
        run_date=run_date,
        current_work_receipt=receipt,
        cited_upstream_artifacts=cited,
        code_config_hashes=[
            e.ref(e.ROOT / p)
            for p in e.OWNED
            + [
                "tests/python/test_natural_label_eligibility_8975.py",
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/natural_criterion_audit_8908.py",
                "python/carnot/reporting/v709_execution.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        preconditions_checked=dict(
            python_version=os.sys.version,
            dependencies="stdlib and existing reporting helpers imported",
            writable_disk_bytes=disk,
            available_memory_bytes=memory,
            gpu_required=False,
            MODEL_SPECS=[],
            declared_upstream_gates=[],
            historical_inputs_required=False,
            request_timeout_s=30,
            network_retry_limit=1,
            network_total_cap_s=600,
            measurement_cap_s=3600,
            checks_reserve_s=900,
            invocation_cap_s=4800,
        ),
        template_fields=template.build_result({}, status="success"),
    )
    atomic_json(raw / "primitive.json", primitive)
    e.progress("measurement_complete", 800, 0)
    return e.ref(raw / "primitive.json")


def build(primitive_reference: Json, validation: Json) -> Json:
    """Readiness requires usable source joins; owned check failures independently disqualify."""
    primitive = read_ref(primitive_reference)
    reduced, receipt = primitive["reduction"], primitive["current_work_receipt"]
    checks = validation.get("checks", [])
    coverage = validation.get("owned_statement_coverage", {})
    passed = (
        e.CHECKS <= {c.get("name") for c in checks}
        and all(c.get("passed") is True for c in checks)
        and coverage.get("percent_covered") == 100
        and coverage.get("missing_lines") == []
        and coverage.get("excluded_lines") == []
    )
    verdict = (
        "disqualified"
        if not passed
        else "null"
        if reduced["natural_evidence_ready_score"]
        else "blocked"
    )
    value = dict(primitive["template_fields"], **reduced)
    value.update(
        experiment_id=8975,
        task_id=e.TASK,
        milestone="2026.11.767",
        run_date=primitive["run_date"],
        honest_verdict="complete_"
        + verdict
        + ("_source_access" if verdict == "blocked" else "_natural_label_eligibility"),
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=False,
        exposure_scope="Public development labels; generator pretraining exposure unknown.",
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        random_seed=8975,
        current_work_receipt=receipt,
        primitive_reference=primitive_reference,
        raw_shard_hashes=[primitive_reference],
        code_config_hashes=primitive["code_config_hashes"],
        cited_upstream_artifacts=primitive["cited_upstream_artifacts"],
        preconditions_checked=primitive["preconditions_checked"],
        validation_receipts=checks,
        owned_statement_coverage=coverage,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        repository_health=validation.get("repository_health", []),
        methodology="Authenticate pinned public release bytes. Audit 400 dialogue identities and800 pair slots against an already-authorized local source index. Separate input/context consistency from open-world factuality. No source text is reconstructed, no new agreement is accepted, no LLM is called and no model is fitted.",
    )
    if not passed:
        value["natural_evidence_ready_score"] = 0
        value["gate_check_summary"] = [
            *value["gate_check_summary"],
            e.gate("owned_required_checks", True, False, primitive_reference),
        ]
    value["acceptance_gates"] = [
        dict(
            criterion="natural_evidence_ready_score",
            expected=1,
            observed=value["natural_evidence_ready_score"],
            passed=value["natural_evidence_ready_score"] == 1,
            principle="All800 text/label pairs join; local use and model terms are authorized; task remains annotation-relative.",
        ),
        dict(
            criterion="owned_checks",
            expected=True,
            observed=passed,
            passed=passed,
            principle="Failed owned validation disqualifies even a scientific null or blocked input.",
        ),
    ]
    value["field_principles"] = {
        k: "Retain exact invocation identity, immutable operands and honest denominators; metadata access establishes no model benefit."
        for k in value
    }
    value["field_principles"].update(
        natural_evidence_ready_score="All intended text/label pairs must join under authorized local use; human labels remain fallible.",
        independent_count="Zero independent blind truth questions were evaluated; paired turns share a dialogue.",
        completed_count="Number of authorized, valid text/label joins; metadata_audited_count separately counts inspected labels.",
        proposed_group_splits="Empty while blocked; an eligible corpus uses deterministic dialogue groups and excludes label metadata from inference.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute rows from original operands, not from a checksum over claimed summaries."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    for reference in primitive["code_config_hashes"]:
        if e.ref(Path(reference["path"])) != reference:
            raise ValueError("executed_source_hash")
    if e.ref(Path(primitive["manifest"]["path"])) != primitive["manifest"]:
        raise ValueError("manifest_hash")
    reduced = primitive["reduction"]
    if reduced["input_scope"] != "external_unavailable":
        if inspect(Path(primitive["manifest"]["path"])) != reduced:
            raise ValueError("original_reduction_drift")
    else:
        fresh = e.audit([], {}, primitive["manifest"])
        if fresh["rows"] != reduced["rows"]:
            raise ValueError("blocked_reduction_drift")
    if validate_current_work_receipt(primitive["current_work_receipt"]):
        raise ValueError("current_work_receipt")
    for check in value["validation_receipts"]:
        for stream in ("stdout", "stderr"):
            if (
                stream + "_path" in check
                and sha256_file(Path(check[stream + "_path"])) != check[stream + "_sha256"]
            ):
                raise ValueError("validation_log_hash")
    expected = build(
        value["primitive_reference"],
        dict(
            checks=value["validation_receipts"],
            owned_statement_coverage=value["owned_statement_coverage"],
            repository_health=value["repository_health"],
        ),
    )
    if expected != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=sha256_file(path))


def validate(candidate: Path, logs: Path) -> Json:
    """Unchanged terminal validators inspect private bytes before atomic publication."""
    checks = []
    for name, args in [
        ("cold_replay", [e.CLI, "--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]:
        check = child(
            name, [str(e.ROOT / ".venv/bin/python"), "-u", *args], logs, deadline=60, heartbeat=20
        )
        for stream in ("stdout_path", "stderr_path"):
            print(Path(check[stream]).read_text(), end="", flush=True)
        checks.append(check)
    return dict(passed=all(c["passed"] for c in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """The thin CLI supports date-bound audit and cold replay from any working directory."""
    e.progress("start_no_model_load")
    start = time.monotonic_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261110")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--receipts",
        type=Path,
        default=e.ROOT / "results/raw" / e.NAME / "validation_receipts.json",
    )
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8:
            raise ValueError("run_date")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        manifest = args.inputs or prepare_manifest(CACHE)
        primitive = measure(manifest, args.output, args.date, start)
        validation = json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else {}
        raw = Path(primitive["path"]).parent
        candidate = raw / "preflight_candidate.json"
        atomic_json(candidate, build(primitive, validation))
        report = validate(candidate, raw / "preflight")
        validation.setdefault("checks", []).extend(report["checks"])
        result = build(primitive, validation)
        publication = publish_primary(args.output, result, lambda p: validate(p, raw / "terminal"))
        print(json.dumps(publication), flush=True)
        e.progress("complete", result["intended_count"], 0)
        return 0
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("refused:" + str(error))
        return 1
