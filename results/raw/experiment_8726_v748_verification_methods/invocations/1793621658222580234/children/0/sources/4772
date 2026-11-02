"""REQ-REPORT-8585: keep human label evidence separate from evaluation eligibility."""

from __future__ import annotations

from collections import Counter
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import time
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.hallmark_release_8553 import read as read, reference as reference

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8585_v737_hallutruth_release_audit"
TASK = "exp8585-hallutruth-release-audit"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/hallutruth_release_8585.py",
    "python/carnot/reporting/hallutruth_release_runner_8585.py",
    CLI,
]
MAX_ROWS, MAX_BYTES = 5000, 100 * 1024 * 1024
MODEL_SPECS: list[Json] = []
AUTHOR_COMMIT = "e6e2b14733effc8da7b87fa79230a20cc4b7336b"
AUTHOR_HASHES = {
    "train.jsonl": "fba55409166302bc2a58df7ce70f2e2c43c64afea96d5d71b123ca9d0f2ed063",
    "dev.jsonl": "588dc8f90654d765ca63a2ca5219818108dcc14361d0bff19ec4aeedfae323fa",
    "test.jsonl": "b68f179e951af131132f713b8786712088d3d6fbfde3e05080b8765078e50d33",
    "README.md": "46a68aea91fcd20b0c626f399c158a5f0fb536fae7371db563a2276d2dccbb26",
    "prepare.py": "1d0f46fcd4312d5108d240b9d13e9f0f47c716023716f584f2c88ba69ac8ca49",
}
EXCLUSION_MANIFEST = dict(
    allowed_fields=["question", "generated_answer"],
    permitted_evidence="Separately supplied, provenance-checked evidence only; no release gold.",
    excluded_fields=[
        "label",
        "gold_answer",
        "answer",
        "hallucinations",
        "options",
        "id",
        "generator_model",
        "split",
        "needs_human_review",
        "truncated",
        "level",
        "gold_match",
        "historical_note",
        "data_note",
        "quality_note",
    ],
    excluded_nested_fields=["hallucinations.*", "*.explanation", "*.review_notes"],
    policy="Strict allowlist also excludes unknown future fields. No label-based selection.",
)


def feature_reader(row: Json, manifest: Json, evidence: list[str] | None = None) -> Json:
    """Only explicit question/response fields survive, so hidden targets cannot leak."""
    if manifest["allowed_fields"] != ["question", "generated_answer"]:
        raise ValueError("feature_allowlist")
    return {**{key: row.get(key) for key in manifest["allowed_fields"]}, "evidence": evidence or []}


def missing(path: str, field: str, expected: Any, observed: Any, sha: str | None = None) -> Json:
    """An absent release is a blockage; an observed zero remains an observation."""
    return dict(
        check=field,
        upstream="Bekhouche/HalluTruthQA-4K",
        path=path,
        sha256=sha,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def load(path: Path) -> Json:
    """Read only bounded, hashed local copies of publicly acquired original bytes."""
    data: Json = dict(
        splits={},
        missing_resources=[],
        files=[],
        declared_counts={},
        annotation={},
        rights={},
        source_metadata={},
    )
    if not path.is_file():
        data["missing_resources"].append(missing(str(path), "release_manifest", "available", None))
        return data
    manifest = json.loads(path.read_bytes())
    if manifest["commit"] != "fixture":
        hashes = {Path(r["path"]).name: r["sha256"] for r in manifest["files"]}
        if manifest["commit"] != AUTHOR_COMMIT or any(
            hashes.get(name) != "sha256:" + digest for name, digest in AUTHOR_HASHES.items()
        ):
            raise ValueError("release_authority")
    data.update(manifest)
    total = 0
    for ref in manifest["files"]:
        source = Path(ref["path"])
        if not source.is_file():
            data["missing_resources"].append(
                missing(str(source), "source_available", True, None, ref["sha256"])
            )
            continue
        total += source.stat().st_size
        if total > MAX_BYTES:
            raise ValueError("byte_budget")
        if reference(source)["sha256"] != ref["sha256"]:
            raise ValueError("source_hash")
        if "split" in ref:
            data["splits"][ref["split"]] = [
                json.loads(line) for line in source.read_bytes().splitlines() if line.strip()
            ]
            if sum(map(len, data["splits"].values())) > MAX_ROWS:
                raise ValueError("row_budget")
    data["downloaded_bytes"] = total
    return data


def reduce(data: Json) -> Json:
    """Inspect each released instance, without repairing text or choosing features by label."""
    private, rows, overlap, cues, lineage = [], [], [], [], []
    failures = list(data["missing_resources"])
    questions: dict[str, Counter[str]] = {}
    span_count = invalid = byte_matches = review = missing_review = 0
    for split, records in sorted(data["splits"].items()):
        questions[split] = Counter(canonical_hash(r.get("question")) for r in records)
        labels = Counter(str(r.get("label")) for r in records)
        ids = [r.get("id") for r in records]
        expected = data["declared_counts"].get(split)
        if len(records) != expected or len(set(ids)) != len(ids):
            failures.append(
                missing(
                    split,
                    "complete_split_unique_ids",
                    dict(count=expected, unique_id_count=expected),
                    dict(count=len(records), unique_id_count=len(set(ids))),
                )
            )
        for index, record in enumerate(records):
            reasons = [
                "missing_" + key
                for key in ["id", "question", "generated_answer", "gold_answer", "generator_model"]
                if not record.get(key)
            ]
            if record.get("label") not in {"hallucination", "no_hallucination"}:
                reasons.append("missing_or_unknown_human_verdict")
            text = record.get("generated_answer") or ""
            spans = record.get("hallucinations") or []
            bad = 0
            for span in spans:
                span_count += 1
                start, end = span.get("span_start"), span.get("span_end")
                valid = (
                    type(start) is int
                    and type(end) is int
                    and 0 <= start < end <= len(text)
                    and text[start:end] == span.get("hallucinated_span")
                )
                bad += not valid
                invalid += not valid
                if type(start) is int and type(end) is int:
                    byte_matches += text.encode("utf-8")[start:end] == str(
                        span.get("hallucinated_span")
                    ).encode("utf-8")
            if record.get("label") == "hallucination" and not spans:
                reasons.append("hallucination_spans_missing")
            review += record.get("needs_human_review") is True
            missing_review += "needs_human_review" not in record
            private.append(
                dict(
                    split=split,
                    index=index,
                    record_sha256=canonical_hash(record),
                    question_sha256=canonical_hash(record.get("question")),
                    response_sha256=canonical_hash(text),
                    reference_sha256=canonical_hash(record.get("gold_answer")),
                    generator=record.get("generator_model"),
                    human_verdict=record.get("label"),
                    span_count=len(spans),
                    invalid_span_count=bad,
                    missing_reasons=reasons,
                    review_flag=record.get("needs_human_review"),
                    row_adjudication_receipt=None,
                )
            )
        rows.append(
            dict(
                split=split,
                intended_count=expected,
                completed_count=len(records),
                class_counts=dict(labels),
                missing_reason=None,
            )
        )
        lineage.append(
            dict(
                split=split,
                count=len(records),
                documented_expert_review_count=len(records)
                if data["annotation"].get("expert_review_documented")
                else 0,
                row_adjudication_receipt_count=0,
                missing_reason="No per-row original annotator, reviewer or adjudication logs released; false review flags can be defaults.",
            )
        )
        cues.append(
            dict(
                split=split,
                exact_duplicate_excess_count=len(records) - len(questions[split]),
                generator_counts=dict(Counter(str(r.get("generator_model")) for r in records)),
                nonempty_construction_fields={
                    key: sum(bool(r.get(key)) for r in records)
                    for key in [
                        "quality_note",
                        "data_note",
                        "historical_note",
                        "level",
                        "gold_match",
                        "truncated",
                    ]
                },
                label_span_presence_counts={
                    str(label): dict(
                        Counter(
                            str(bool(r.get("hallucinations")))
                            for r in records
                            if str(r.get("label")) == label
                        )
                    )
                    for label in labels
                },
                warning="IDs, options, span presence and review fields are construction cues excluded a priori. No detector trained.",
            )
        )
        print(
            f"[exp8585] inspected split={split} completed={len(private)} pending={max(0, sum(data['declared_counts'].values()) - len(private))}",
            flush=True,
        )
    for left, right in itertools.combinations(sorted(questions), 2):
        common = sorted(questions[left].keys() & questions[right].keys())
        overlap.append(
            dict(
                left_split=left,
                right_split=right,
                exact_question_overlap_count=len(common),
                question_sha256s=common,
                matching_left_count=sum(questions[left][q] for q in common),
                matching_right_count=sum(questions[right][q] for q in common),
            )
        )
    if set(data["declared_counts"]) != set(data["splits"]) or not data["splits"]:
        failures.append(
            missing(
                "release",
                "complete_public_splits",
                list(data["declared_counts"]),
                list(data["splits"]),
            )
        )
    total = len(private)
    return dict(
        private_rows=private,
        rows=rows,
        intended_count=sum(data["declared_counts"].values()),
        completed_count=total,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            max_rows=MAX_ROWS,
            max_download_bytes=MAX_BYTES,
            inspected_rows=total,
            independent_evaluation_units=0,
        ),
        release_audit_ready_score=int(not failures),
        human_label_evidence_score=int(
            bool(total)
            and data["annotation"].get("expert_review_documented", False)
            and data["annotation"].get("independent_of_carnot", False)
            and all(r["human_verdict"] in {"hallucination", "no_hallucination"} for r in private)
        ),
        unseen_question_eligible_score=0,
        unseen_generator_eligible_score=0,
        deployment_permission_score=0,
        # This is a closed eligibility gate, not a measured zero error rate.
        natural_error_evidence_score=False,
        gate_check_summary=failures,
        annotation_lineage_rows=lineage,
        split_overlap_rows=overlap,
        construction_cue_rows=cues,
        span_diagnostics=dict(
            span_count=span_count,
            invalid_character_span_count=invalid,
            matching_byte_offset_span_count=byte_matches,
            offset_unit="Unicode character indices into unchanged UTF-8-decoded response; not bytes",
        ),
        unresolved_review_count=review,
        missing_review_flag_count=missing_review,
        excluded_label_fields=EXCLUSION_MANIFEST,
        future_protocol_conditions=[
            "New unexposed question clusters and independent adjudication before predictor access.",
            "Arabic-qualified feature pipeline and source-grounded versus world-factual labels kept distinct.",
            "Explicit rights for intended use; multiple withheld generator families for generator holdout.",
            "Prospectively sampled deployment responses for natural-error prevalence; no controlled-release estimate.",
            "A future multilingual protocol requires language-specific expert evidence; no change to this milestone corpus.",
        ],
    )


def measure(manifest: Path, output: Path) -> Json:
    """Save immutable aggregate primitives and private rows before building a verdict."""
    start = time.monotonic_ns()
    wall = time.time_ns()
    scratch = Path(tempfile.mkdtemp(prefix="carnot8585-", dir="/var/tmp"))
    os.chmod(scratch, 0o700)
    probe = scratch / "probe"
    probe.write_text("disk-backed")
    probe.unlink()
    data = load(manifest)
    summary = reduce(data)
    private = scratch / "instance_rows.json"
    atomic_json(private, summary.pop("private_rows"))
    os.chmod(private, 0o600)
    raw = output.parent / "raw" / NAME
    raw.mkdir(parents=True, exist_ok=True)
    atomic_json(raw / "label_exclusion_manifest.json", EXCLUSION_MANIFEST)
    primitive = dict(
        reduction=summary,
        manifest_path=str(manifest.resolve()),
        source_manifest=reference(manifest) if manifest.is_file() else None,
        source_artifact_hashes=data["files"],
        source_metadata=data["source_metadata"],
        release_reference=dict(
            repository="Bekhouche/HalluTruthQA-4K",
            commit=data.get("commit"),
            downloaded_bytes=sum(ref.get("bytes", 0) for ref in data["files"]),
            authenticated_input_bytes=data.get("downloaded_bytes"),
            original_metadata=data.get("original_metadata"),
        ),
        rights_reference=data["rights"],
        annotation_process=data["annotation"],
        private_rows_reference=reference(private),
        code_config_hashes=[reference(ROOT / n) for n in OWNED],
        preconditions_checked=dict(
            started_wall_ns=wall,
            scratch=str(scratch),
            writable_disk_backed=True,
            filesystem=next(
                (
                    line.split()[2]
                    for line in Path("/proc/mounts").read_text().splitlines()
                    if line.split()[1] == "/"
                ),
                "unknown",
            ),
            memory=next(
                line
                for line in Path("/proc/meminfo").read_text().splitlines()
                if line.startswith("MemAvailable:")
            ),
            disk_free_bytes=shutil.disk_usage(scratch).free,
            tools={
                name: (ROOT / ".venv/bin" / name).is_file()
                for name in ["python", "pytest", "ruff", "mypy"]
            },
            public_read_only=True,
            optional_cuda_required=False,
            optional_board_required=False,
            measurement_deadline_s=3600,
            validation_reserve_s=900,
            invocation_cap_s=4800,
        ),
        phase_spans=[
            dict(
                phase="authenticated_cpu_release_inspection",
                started_monotonic_ns=start,
                ended_monotonic_ns=time.monotonic_ns(),
            )
        ],
        duration_s=(time.monotonic_ns() - start) / 1e9,
    )
    atomic_json(raw / "primitive_rows.json", primitive)
    return reference(raw / "primitive_rows.json")


def build(source: Json, receipts: list[Json], output: Path) -> Json:
    """Audit completion permits a null finding; owned validation failure disqualifies."""
    primitive = read(source)
    summary = primitive["reduction"]
    passed = bool(receipts) and all(r.get("passed") is True for r in receipts)
    verdict = (
        "disqualified"
        if not passed
        else "blocked"
        if not summary["release_audit_ready_score"]
        else "null"
    )
    value = dict(
        summary,
        experiment_id=8585,
        task_id=TASK,
        milestone="2026.10.737",
        run_date="20261016",
        honest_verdict="complete_" + verdict + "_hallutruth_release_audit",
        verdict_class=verdict,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(loads=0, generation=0, llm_calls=0),
        historical_model_provenance=dict(
            generator="QCRI/Fanar-1-9B-Instruct",
            invoked=False,
            scope="Author-controlled historical Arabic generation only; no current execution.",
        ),
        verifier_is_oracle=False,
        exposure_scope="Complete public Arabic release inspected; all official splits exposed to this audit.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=False,
        validation_receipts=receipts,
        acceptance_gates=[
            dict(criterion=key, passed=bool(summary[key]))
            for key in [
                "release_audit_ready_score",
                "human_label_evidence_score",
                "unseen_question_eligible_score",
                "unseen_generator_eligible_score",
                "deployment_permission_score",
                "natural_error_evidence_score",
            ]
        ],
        adversarial_findings=[read_log(r) for r in receipts if r.get("name") == "adversarial"],
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        primitive_reference=source,
        raw_shard_hashes=[source, primitive["private_rows_reference"]],
        random_seed=8585,
        methodology="Authenticate bounded release bytes; count labels, validate unchanged character spans, exact question overlap and construction cues; no model, detector or training.",
        cited_upstream_artifacts=primitive["source_artifact_hashes"],
        **{
            key: primitive[key]
            for key in [
                "preconditions_checked",
                "duration_s",
                "phase_spans",
                "source_artifact_hashes",
                "code_config_hashes",
                "release_reference",
                "rights_reference",
                "annotation_process",
                "private_rows_reference",
                "source_metadata",
            ]
        },
    )
    value["field_principles"] = {
        key: "Bind measured audit evidence to immutable original bytes; completion grants no generalization or deployment permission."
        for key in value
    }
    value["field_principles"].update(
        release_audit_ready_score="One means complete bounded inspection, including measured ineligibility; it is not independent evaluation readiness.",
        human_label_evidence_score="One means original expert/reviewer protocol and complete binary labels independent of Carnot scores; row adjudication logs remain missing.",
        unseen_question_eligible_score="Zero: observed question overlap and complete public exposure require a new question partition.",
        unseen_generator_eligible_score="Zero: one controlled historical generator cannot test generator holdout.",
        deployment_permission_score="Zero: the card's NC/ND terms do not establish permission for intended deployment.",
        natural_error_evidence_score="Zero: controlled question/response construction does not measure deployment prevalence.",
        annotation_lineage_rows="Protocol coverage and available per-row adjudication receipt coverage are different denominators.",
        span_diagnostics="Offsets index original decoded Unicode characters; matching byte offsets are a separate diagnostic.",
        split_overlap_rows="Exact unchanged question hashes identify overlap without reassigning official splits.",
        construction_cue_rows="Aggregate construction diagnostics are audit evidence only; labels never choose detector features.",
        independent_count="Zero independent evaluation units: released instance counts are inventory only.",
    )
    if not passed:
        value["release_audit_ready_score"] = 0
        value["gate_check_summary"] = [
            *summary["gate_check_summary"],
            missing(str(output), "owned_required_checks", True, False),
        ]
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def read_log(receipt: Json) -> Any:
    """Preserve scanner findings rather than discarding successful command output."""
    return json.loads(Path(receipt["stdout_path"]).read_bytes())


def replay(path: Path) -> Json:
    """Recompute from original sources so a repaired envelope cannot bless changed counts."""
    value = json.loads(path.read_bytes())
    source = value["primitive_reference"]
    primitive = read(source)
    if primitive["code_config_hashes"] != [reference(ROOT / name) for name in OWNED]:
        raise ValueError("code_hash")
    if primitive["source_manifest"] is not None:
        read(primitive["source_manifest"])
    data = load(Path(primitive["manifest_path"]))
    summary = reduce(data)
    if summary.pop("private_rows") != read(primitive["private_rows_reference"]):
        raise ValueError("private_row_drift")
    if summary != primitive["reduction"]:
        raise ValueError("primitive_reduction_drift")
    for receipt in value["validation_receipts"]:
        for stream in ["stdout", "stderr"]:
            if (
                stream + "_path" in receipt
                and reference(Path(receipt[stream + "_path"]))["sha256"]
                != receipt[stream + "_sha256"]
            ):
                raise ValueError("validation_log_hash")
    output = Path(value["terminal_validation_sidecar_path"]).parents[2] / (NAME + ".json")
    if build(source, value["validation_receipts"], output) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=reference(path)["sha256"])
