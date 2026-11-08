"""REQ-REPORT-8908: retain private source bytes and publish only a validated audit manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

from carnot.reporting import natural_criterion_audit_8908 as e
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
PINNED = {
    "messages": "aa45c84735cb1c90ad07415cd12748f31d3fec7495c47b2a16a90ad9e0e5723e",
    "spans": "7b239d07608c2abde74e66f116f68012d75034bf672a77f72b62413ad07ebb21",
    "card": "d023c216f4b62d5d98e583c9d0c97e427aa9afa7b008b7cb2109f3c4819f4ec1",
    "responses": "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    "sources": "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    "license": "b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
}
UPSTREAMS = {
    "experiment_8597_v738_medexpert_evidence_audit.json": (
        8597,
        "exp8597-medexpert-evidence-audit",
        "c3d315ded484dcb219d2102ce4de810b8bcd6e8d42a581f4f5eb794e07440ea1",
    ),
    "experiment_8897_v761_source_bound_decisions.json": (
        8897,
        "exp8897-source-bound-decisions",
        "e0d969510b272ff2d81eb3027cf8f81111ac5cfefe3e984a789411a7d91cf7be",
    ),
}
URLS = {
    k: f"https://huggingface.co/datasets/blue-guardrails/ragtruth-plus-plus/resolve/{e.REVISION}/{name}"
    for k, name in [
        ("messages", "messages.csv"),
        ("spans", "hallucination_spans.csv"),
        ("card", "README.md"),
    ]
}
URLS.update(
    {
        k: f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{e.ORIGINAL_REVISION}/{name}"
        for k, name in [
            ("responses", "dataset/response.jsonl"),
            ("sources", "dataset/source_info.jsonl"),
            ("license", "LICENSE"),
        ]
    }
)
URLS["method"] = (
    "https://blueguardrails.com/en/blog/ragtruth-plus-plus-enhanced-hallucination-detection-benchmark"
)


def ref(path: Path) -> Json:
    """An absent file has no hash; absence must never look like a measured zero."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def read_ref(reference: Json) -> Any:
    """Authenticate original bytes before parsing any measurement or authority field."""
    path = Path(reference["path"])
    if ref(path)["sha256"] != reference["sha256"] or not path.is_file():
        raise ValueError("source_hash:" + str(path))
    return json.loads(path.read_bytes())


def fetch_inputs(directory: Path) -> Path:
    """Bound each public request and retain headers independently from dataset payloads."""
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    files, retrievals, upstreams = {}, [], []
    for i, (kind, url) in enumerate(URLS.items()):
        e.progress("before_retrieval_" + kind, i, len(URLS) - i)
        body, headers = directory / (kind + ".body"), directory / (kind + ".headers")
        receipt = child(
            "fetch_" + kind,
            [
                "curl",
                "-L",
                "--connect-timeout",
                "10",
                "--max-time",
                "55",
                "--max-filesize",
                "104857600",
                "-sS",
                "-D",
                str(headers),
                "-o",
                str(body),
                "-w",
                "%{http_code}",
                url,
            ],
            directory / "retrieval_logs",
            deadline=60,
            heartbeat=20,
        )
        status = Path(receipt["stdout_path"]).read_text().strip()
        retrievals.append(
            dict(url=url, http_status=status, body=ref(body), headers=ref(headers), receipt=receipt)
        )
        atomic_json(directory / "retrievals.json", dict(requests=retrievals))
        e.progress("after_retrieval_" + kind, i + 1, len(URLS) - i - 1)
        if not receipt["passed"] or status != "200":
            raise OSError("external_retrieval:" + kind + ":" + status)
        if kind in PINNED and ref(body)["sha256"] != "sha256:" + PINNED[kind]:
            raise ValueError("rejected_producer:" + kind)
        files[kind] = ref(body)
    for name, (ident, task, digest) in UPSTREAMS.items():
        path = e.ROOT / "results" / name
        upstream = read_ref(dict(path=str(path), sha256="sha256:" + digest))
        if (upstream["experiment_id"], upstream["task_id"]) != (ident, task):
            raise ValueError("upstream_identity")
        saved = directory / name
        saved.write_bytes(path.read_bytes())
        upstreams.append(ref(saved))
    manifest = directory / "input_manifest.json"
    atomic_json(
        manifest,
        dict(
            scope="author_release",
            release_revision=e.REVISION,
            original_revision=e.ORIGINAL_REVISION,
            files=files,
            retrievals=retrievals,
            upstreams=upstreams,
        ),
    )
    return manifest


def inspect(manifest: Path) -> Json:
    """Freeze rights and the crosswalk before any paired-label reduction."""
    data = json.loads(manifest.read_bytes())
    if data["release_revision"] != e.REVISION or data["original_revision"] != e.ORIGINAL_REVISION:
        raise ValueError("release_revision")
    files = data["files"]
    for kind, reference in files.items():
        if ref(Path(reference["path"])) != reference:
            raise ValueError("source_hash:" + kind)
        if (
            data["scope"] == "author_release"
            and kind in PINNED
            and reference["sha256"] != "sha256:" + PINNED[kind]
        ):
            raise ValueError("rejected_producer:" + kind)
    for retrieval in data["retrievals"]:
        for key in ("body", "headers"):
            reference = retrieval[key]
            if ref(Path(reference["path"])) != reference:
                raise ValueError("retrieval_hash:" + key)
    card, method = (
        Path(files["card"]["path"]).read_text(),
        Path(files["method"]["path"]).read_text(),
    )
    authority = "Use the dataset for benchmarking hallucination detection models." in card
    if not all(
        token in method
        for token in [
            "Original RAGTruth human annotations",
            "Blue Guardrails automatic predictions",
        ]
    ):
        raise ValueError("annotation_lineage")
    rights = [
        dict(
            component="revised_annotations",
            local_evaluation_permitted=authority,
            evaluation_authority=files["card"],
            evaluation_statement="Use the dataset for benchmarking hallucination detection models."
            if authority
            else None,
            redistribution_permission="unspecified",
            license=None,
        ),
        dict(
            component="original_RAGTruth",
            local_evaluation_permitted="MIT License" in Path(files["license"]["path"]).read_text(),
            evaluation_authority=files["license"],
            redistribution_permission="MIT conditions; third-party source rights remain separate",
            license="MIT",
        ),
    ]
    authority = all(r["local_evaluation_permitted"] for r in rights)
    crosswalk = dict(
        mapping=e.CROSSWALK,
        alias_reason="The released Subtle Introduction Of Baseless Info spelling denotes the same baseless-information category.",
        offsets="half-open Unicode code points; exact slice required",
        incorrect_refusal="Separate from a false factual statement; never merged into evidence-failure presence.",
        independent_blind_truth=False,
    )
    frozen = manifest.parent / "annotation_crosswalk.json"
    if frozen.exists() and json.loads(frozen.read_bytes()) != crosswalk:
        raise ValueError("crosswalk_drift")
    atomic_json(frozen, crosswalk)
    messages, spans = (
        e.read_csv(Path(files["messages"]["path"]), "messages"),
        e.read_csv(Path(files["spans"]["path"]), "spans"),
    )
    responses, sources = (
        [json.loads(line) for line in Path(files[k]["path"]).read_text().splitlines() if line]
        for k in ["responses", "sources"]
    )
    reduced = e.audit(messages, spans, responses, sources, authority=authority)
    source_tasks = {s["source_id"]: s["task_type"] for s in sources}
    preserved = [
        dict(
            reference=r,
            experiment_id=read_ref(r)["experiment_id"],
            task_id=read_ref(r)["task_id"],
            honest_verdict=read_ref(r)["honest_verdict"],
        )
        for r in data["upstreams"]
    ]
    reduced.update(
        release_revision=e.REVISION,
        original_revision=e.ORIGINAL_REVISION,
        source_urls=URLS,
        rights_ledger=rights,
        annotation_crosswalk=crosswalk,
        crosswalk_reference=ref(frozen),
        source_artifact_hashes=[ref(manifest), *files.values(), *data["upstreams"]],
        cited_upstream_artifacts=preserved,
        retrieval_receipts=data["retrievals"],
        input_scope=data["scope"],
        original_test_population_count=sum(r["split"] == "test" for r in responses),
        original_qa_summary_test_population_count=sum(
            r["split"] == "test" and source_tasks.get(r["source_id"]) in {"QA", "Summary"}
            for r in responses
        ),
        original_test_population_scope="Pinned original response records; sampled408 pairs do not estimate population prevalence.",
        gate_check_summary=[],
    )
    if not authority:
        reduced["gate_check_summary"].append(
            gate(manifest, "local_evaluation_authority", True, False)
        )
    return reduced


def gate(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name the exact missing or rejected operand without interpreting absence as zero."""
    return dict(
        check=field,
        upstream="author_release_or_owned_validation",
        artifact_field=field,
        expected=expected,
        observed=observed,
        **ref(path),
    )


def measure(manifest: Path, output: Path, run_date: str, *, started_ns: int | None = None) -> Json:
    """Save actual clocks and private full pairs before any reader-visible publication."""
    datetime.strptime(run_date, "%Y%m%d")
    start = started_ns if started_ns is not None else time.monotonic_ns()
    joined_at = time.monotonic_ns()
    template = ExperimentTemplate(
        8908,
        "Paired natural criterion audit",
        "results/" + output.name,
        repo_root=output.parent,
        seed=e.SEED,
    )
    e.progress("authenticate_and_join")
    with template.phase("authenticate_and_join"):
        try:
            reduced = inspect(manifest)
        except (OSError, ValueError, KeyError, TypeError) as error:
            reduced = e.audit([], [], [], [], authority=False)
            reduced.update(
                corpus_audit_ready_score=0,
                gate_check_summary=[
                    gate(
                        manifest,
                        "external_inputs",
                        "authenticated pinned operands",
                        "missing_file"
                        if isinstance(error, FileNotFoundError)
                        else "rejected_producer:" + str(error),
                    )
                ],
                source_artifact_hashes=[ref(manifest)],
                cited_upstream_artifacts=[],
                release_revision=e.REVISION,
                source_urls=URLS,
                rights_ledger=[],
                annotation_crosswalk=dict(mapping=e.CROSSWALK),
                input_scope="external_unavailable",
            )
    private = Path(tempfile.mkdtemp(prefix="carnot8908-", dir=str(Path.home() / ".cache")))
    private.chmod(0o700)
    atomic_json(private / "full_pairs.json", dict(pairs=reduced.pop("private_pairs")))
    reduced["local_payload_reference"] = ref(private / "full_pairs.json")
    end = time.monotonic_ns()
    spans = [
        dict(phase="input_acquisition", start_s=0, end_s=(joined_at - start) / 1e9),
        dict(
            phase="authenticate_and_join",
            start_s=(joined_at - start) / 1e9,
            end_s=(end - start) / 1e9,
        ),
    ]
    receipt = build_current_work_receipt(
        run_id=private.name,
        owner_pid=os.getpid(),
        events=[],
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_details=dict(
            operation="paired annotation custody audit", current_LLM_calls=0
        ),
        inference_substrate_class="no_model_load"
        if reduced["corpus_audit_ready_score"]
        else "blocked_no_run",
        execution_venue="host",
        started_monotonic_ns=start,
        ended_monotonic_ns=end,
        phase_spans=spans,
    )
    primitive = dict(
        manifest=ref(manifest),
        reduction=reduced,
        run_date=run_date,
        current_work_receipt=receipt,
        template_fields=template.build_result({}, status="success"),
        code_config_hashes=[
            ref(e.ROOT / n)
            for n in e.OWNED
            + [
                "scripts/experiment_template.py",
                "python/carnot/reporting/current_work_receipt.py",
                "python/carnot/reporting/primary_publication.py",
                "python/carnot/reporting/v709_execution.py",
                "tests/python/test_natural_criterion_audit_8908.py",
                "scripts/adversarial_verify.py",
                "scripts/verdict_row_consistency_lint.py",
            ]
        ],
        preconditions_checked=dict(
            cpu_only=True,
            gpu_required=False,
            bounded_requests=len(URLS),
            request_timeout_s=60,
            max_download_bytes_per_request=104857600,
            writable_private_directory=str(private),
            pilot_work_cap_s=4200,
            invocation_cap_s=4800,
        ),
    )
    atomic_json(private / "primitive.json", primitive)
    e.progress("measurement_complete", reduced["completed_count"], 408 - reduced["completed_count"])
    return ref(private / "primitive.json")


def build(primitive_reference: Json, validation: Json) -> Json:
    """Audit readiness is distinct from optional canary eligibility and scientific benefit."""
    primitive = read_ref(primitive_reference)
    reduced, receipt = primitive["reduction"], primitive["current_work_receipt"]
    checks, coverage = validation.get("checks", []), validation.get("owned_statement_coverage", {})
    passed = (
        bool(checks)
        and all(c.get("passed") is True for c in checks)
        and coverage.get("percent_covered") == 100
        and not coverage.get("missing_lines")
        and not coverage.get("excluded_lines")
    )
    required_names = {
        "owned_pytest",
        "owned_coverage",
        "ruff",
        "format",
        "mypy",
        "spec",
        "e2e018_021",
    }
    if reduced["input_scope"] == "author_release":
        passed = passed and required_names <= {c.get("name") for c in checks}
    verdict = (
        "disqualified"
        if not passed
        else "null"
        if reduced["corpus_audit_ready_score"]
        else "blocked"
    )
    value = dict(primitive["template_fields"], **reduced)
    value.update(
        experiment_id=8908,
        task_id=e.TASK,
        milestone="2026.11.762",
        run_date=primitive["run_date"],
        honest_verdict="complete_" + verdict + "_natural_criterion_audit",
        verdict_class=verdict,
        inference_substrate=receipt["inference_substrate"],
        inference_substrate_class=receipt["inference_substrate_class"],
        execution_venue="host",
        MODEL_SPECS=[],
        model_invocation_counts=receipt["invocation_counts"],
        numerical_training_calls=0,
        verifier_is_oracle=False,
        exposure_scope="Original annotations and automatic predictions were shown during revised annotation. Generator pretraining exposure is unknown; this is dependent criterion comparison.",
        duration_s=receipt["duration_s"],
        phase_spans=receipt["phase_spans"],
        current_work_receipt=receipt,
        random_seed=e.SEED,
        code_config_hashes=primitive["code_config_hashes"],
        raw_shard_hashes=[primitive_reference, reduced["local_payload_reference"]],
        primitive_reference=primitive_reference,
        preconditions_checked=primitive["preconditions_checked"],
        validation_receipts=checks,
        owned_statement_coverage=coverage,
        repository_health=validation.get("repository_health", []),
        duration_scope="Input acquisition and corpus audit; terminal validation has separate receipt clocks.",
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        methodology="Read separate author CSV schemas at a pinned revision. Freeze category crosswalk and explicit evaluation authority. Join exact prompts and responses to pinned original test records. Retain all408 slots. Compare annotation presence and exact spans; freeze24 source-task/ID-hash cases without model calls. Assisted revisions are dependent criteria and establish no independent natural-language truth.",
    )
    if not passed:
        value.update(corpus_audit_ready_score=0, natural_canary_ready_score=0)
        value["gate_check_summary"] = [
            *reduced["gate_check_summary"],
            gate(Path(primitive_reference["path"]), "owned_required_checks", True, False),
        ]
    value["acceptance_gates"] = [
        dict(criterion=k, expected=1, observed=value[k], passed=value[k] == 1)
        for k in ["corpus_audit_ready_score", "natural_canary_ready_score"]
    ]
    value["field_principles"] = {
        k: "Bind this invocation to original bytes and explicit denominators; completion establishes no independent semantic truth."
        for k in value
    }
    value["field_principles"].update(
        corpus_audit_ready_score="One means a complete auditable coverage and rights decision, even without24 eligible pairs.",
        natural_canary_ready_score="One means exactly24 complete, sound joined pairs with explicit local benchmarking authority; model/context feasibility is still downstream.",
        independent_count="Zero independent blind truth units. The408 paired responses can share sources and dependent annotation procedures.",
        rights_ledger="Local benchmarking permission is explicit. Missing redistribution terms do not authorize publishing payloads.",
        local_payload_reference="Private original prompts, responses and both span criteria; public fields contain only identifiers and hashes.",
        canary_manifest="Task-stratified selection uses only ID hashes; disagreement and model outcomes cannot select cases.",
        disagreement_rows="Three paired disagreements over all valid joins; refusal is separate from evidence-failure presence.",
        original_test_population_count="Original test response count is distinct from the408 sampled cases and from source clusters.",
        annotation_crosswalk="Five distinct categories and one explicit author spelling alias, frozen before reduction.",
    )
    value.pop("reproducibility_checksum", None)
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Rebuild derived claims from the original authority, even after a forged checksum repair."""
    value = json.loads(path.read_bytes())
    primitive = read_ref(value["primitive_reference"])
    if any(ref(Path(r["path"])) != r for r in primitive["code_config_hashes"]):
        raise ValueError("executed_source_hash")
    reduced = primitive["reduction"]
    if reduced["corpus_audit_ready_score"]:
        fresh = inspect(Path(primitive["manifest"]["path"]))
        payload = fresh.pop("private_pairs")
        saved = read_ref(reduced["local_payload_reference"])
        if saved != dict(pairs=payload):
            raise ValueError("private_payload_drift")
        fresh["local_payload_reference"] = reduced["local_payload_reference"]
        if fresh != reduced or ref(Path(primitive["manifest"]["path"])) != primitive["manifest"]:
            raise ValueError("original_reduction_drift")
    if validate_current_work_receipt(primitive["current_work_receipt"]):
        raise ValueError("current_work_receipt")
    for row in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if (
                stream + "_path" in row
                and sha256_file(Path(row[stream + "_path"])) != row[stream + "_sha256"]
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
    """Unchanged validators inspect prospective bytes before the primary is visible."""
    checks = []
    for name, argv in [
        ("cold_replay", [e.CLI, "--cold-replay", str(candidate)]),
        ("adversarial", ["scripts/adversarial_verify.py", "--json", str(candidate)]),
        ("strict_rows", ["scripts/verdict_row_consistency_lint.py", "--strict", str(candidate)]),
    ]:
        checks.append(
            child(
                name,
                [str(e.ROOT / ".venv/bin/python"), "-u", *argv],
                logs,
                deadline=60,
                heartbeat=20,
            )
        )
    return dict(passed=all(r["passed"] for r in checks), checks=checks)


def main(argv: list[str] | None = None) -> int:
    """Permit script-path audit and cold replay without loading a model or publishing payloads."""
    e.progress("start_no_model_load")
    started_ns = time.monotonic_ns()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--date", default="20261107")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--receipts",
        type=Path,
        default=e.ROOT / "results/raw" / e.NAME / "validation_receipts.json",
    )
    parser.add_argument("--output", type=Path, default=e.ROOT / "results" / (e.NAME + ".json"))
    parser.add_argument("--cold-replay", type=Path)
    parser.add_argument("--deliberate-error", action="store_true")
    args = parser.parse_args(argv)
    try:
        datetime.strptime(args.date, "%Y%m%d")
        if len(args.date) != 8 or args.deliberate_error:
            raise ValueError("owned_input_control")
        if args.cold_replay:
            print(json.dumps(replay(args.cold_replay)), flush=True)
            return 0
        manifest = args.inputs
        if manifest is None:
            directory = Path(
                tempfile.mkdtemp(prefix="carnot8908-release-", dir=str(Path.home() / ".cache"))
            )
            try:
                manifest = fetch_inputs(directory)
            except (OSError, ValueError) as error:
                e.progress("external_fetch_blocked:" + str(error))
                manifest = directory / "unavailable_input_manifest.json"
        source = measure(manifest, args.output, args.date, started_ns=started_ns)
        validation = (
            json.loads(args.receipts.read_bytes()) if args.receipts.is_file() else dict(checks=[])
        )
        value = build(source, validation)
        logs = Path(source["path"]).parent / "validation"
        candidate = logs.parent / (e.NAME + ".json")
        atomic_json(candidate, value)
        report = validate(candidate, logs / "prospective")
        validation.setdefault("checks", []).extend(report["checks"])
        if not report["passed"]:
            validation["checks"].append(dict(name="owned_terminal_checks", passed=False))
        value = build(source, validation)
        publication = publish_primary(
            args.output, value, lambda path: validate(path, logs / "terminal")
        )
        atomic_json(logs / "publication.json", publication)
        e.progress("published_" + value["verdict_class"], 1, 0)
        return int(not value["required_checks_passed"])
    except (OSError, ValueError, KeyError, TypeError) as error:
        e.progress("rejected:" + str(error))
        return 1
