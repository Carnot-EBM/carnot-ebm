"""REQ-REPORT-8597: audit label authority without converting missing evidence into truth.

The original clinician protocol and later assisted review have different meanings.
No sentence detector or numerical head runs in this release-availability audit.
"""

from __future__ import annotations

from datetime import datetime
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
NAME = "experiment_8597_v738_medexpert_evidence_audit"
TASK = "exp8597-medexpert-evidence-audit"
CLI = "scripts/experiments/" + NAME + ".py"
RAW = ROOT / "results/raw" / NAME
MANIFEST = RAW / "input_manifest.json"
MANIFEST_SHA = "sha256:bfbd85a0abf5c0e435a271a22bee4c526c40db16bbe9930fe39c40f375ff79b5"
MODEL_SPECS: list[Json] = []
OWNED = [
    "python/carnot/reporting/medexpert_evidence_8597.py",
    "python/carnot/reporting/medexpert_evidence_runner_8597.py",
    CLI,
]


def gate(ref: Json, field: str, expected: Any, observed: Any, upstream: str) -> Json:
    """Keep absence explicit because an unavailable label is not an observed zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=ref["path"],
        sha256=ref["sha256"],
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
        passed=False,
    )


def inspect(path: Path) -> Json:
    """Rebuild the bounded audit from original bytes rather than trusting summary assertions."""
    if reference(path)["sha256"] != MANIFEST_SHA:
        raise ValueError("original_authority")
    manifest = json.loads(path.read_bytes())
    refs = list(manifest["sources"].values()) + manifest["upstreams"]
    retrievals = read(manifest["retrievals"])
    for row in retrievals["requests"]:
        refs.extend([row["body"], row["headers"]])
    for corpus in manifest["corpora"]:
        refs.append(corpus["reference"])
        refs.extend(corpus.get("source_files", []))
    for ref in refs:
        if reference(Path(ref["path"]))["sha256"] != ref["sha256"]:
            raise ValueError("source_hash")
    if (
        retrievals["total_requests"] > 10
        or sum(r["bytes"] for r in retrievals["requests"]) > 100 * 1024 * 1024
        or sum(r["duration_s"] for r in retrievals["requests"]) > 300
    ):
        raise ValueError("retrieval_budget")
    sources = manifest["sources"]
    hf = read(sources["hf_release"])
    tree = read(sources["original_tree"])
    paper = Path(sources["paper"]["path"]).read_text()
    if not all(token in paper for token in ["225 sentences", "six verifiers", "139 sentences"]):
        raise ValueError("paper_protocol")
    missing = manifest["absence_reason"]
    failures = [
        gate(
            sources["anonymous_tree"],
            "assisted_release_http_status",
            200,
            401,
            "2609.30467 Appendix A.2",
        ),
        gate(
            sources["hf_release"],
            "original_sentence_bytes",
            "authenticated sentence rows",
            None,
            hf["id"],
        ),
        gate(
            sources["anonymous_tree"],
            "assisted_sentence_bytes",
            "original adjudication rows",
            None,
            "MedExpert-gold",
        ),
        gate(
            sources["hf_release"],
            "question_overlap_inventory",
            "sealed normalized question/source hashes",
            None,
            hf["id"],
        ),
        gate(
            sources["anonymous_tree"],
            "assisted_use_rights",
            "explicit release terms",
            None,
            "MedExpert-gold",
        ),
    ]
    scores = dict(
        release_audit_ready_score=1,
        release_available_score=0,
        original_annotation_authority_score=1,
        assisted_label_independence_score=0,
        question_disjoint_score=0,
        rights_eligible_score=0,
        evaluation_eligible_score=0,
    )
    return dict(
        **scores,
        rows=[
            dict(
                unit=unit,
                intended_count=None,
                completed_count=0,
                sentence_labels_available=False,
                missing_reason=missing,
            )
            for unit in [
                "original_clinician_sentence_inventory",
                "assisted_gold_sentence_inventory",
            ]
        ],
        intended_count=None,
        completed_count=0,
        failed_count=0,
        censored_count=0,
        excluded_count=0,
        independent_count=0,
        sample_size_budget=dict(
            max_sentence_rows=15000,
            inspected_sentence_rows=0,
            independent_evaluation_units=0,
            max_requests=10,
            actual_requests=retrievals["total_requests"],
            max_download_bytes=104857600,
        ),
        gate_check_summary=failures,
        release_reference=dict(
            original_publication="https://proceedings.mlr.press/v297/yarmohammadi26a.html",
            original_repository="https://github.com/JHU-CLSP/MedExpert",
            original_git_tree_hash=tree["sha"],
            original_commit_hash=None,
            original_dataset=hf["id"],
            dataset_commit=hf["sha"],
            dataset_public=not hf["private"] and not hf["gated"],
            original_files=[r["rfilename"] for r in hf["siblings"]],
            assisted_repository="https://anonymous.4open.science/r/Medical_RAG_eval-4AB5",
            assisted_commit_hash=None,
            release_rows_acquired=False,
            original_release_metadata_available=True,
            retrievals=retrievals,
        ),
        annotation_lineage_rows=[
            dict(
                stage="original_clinician_annotation",
                authority="Yarmohammadi et al., PMLR 297",
                clinician_count=8,
                unit="highlighted span; omission and certainty at response level",
                target_criterion="Factual errors tied to clinical harm; omissions are separate.",
                evidence_shown="Generated response and clinician guidelines; original protocol.",
                blinding_verified=False,
                per_row_authority_available=False,
                unmarked_sentence_independently_correct=False,
            ),
            dict(
                stage="assisted_adjudication",
                authority="2609.30467 Appendix A.2",
                clinician_count=5,
                unit="sentence and associated claims",
                target_criterion="Any false associated claim marks the sentence has-error.",
                evidence_shown="Sentence-level claims, six verifier votes and their reasoning.",
                blinding_verified=False,
                per_row_authority_available=False,
                independent_of_verifier_outputs=False,
            ),
        ],
        selection_rows=[
            dict(
                paper_selected_count=225,
                paper_retained_errors=86,
                paper_reclassified_correct=139,
                verifier_votes_shown=6,
                selection_rule="All originally error-marked sentences only.",
                conditional_inclusion_probability=1,
                unmarked_inclusion_probability=0,
                population_sampling_probability=None,
                duplicate_question_count=None,
                population_precision_identifiable=False,
                prevalence_identifiable=False,
                general_domain_accuracy_identifiable=False,
            )
        ],
        count_reconciliation=dict(
            paper_qa_pairs=540,
            paper_original_table_errors=224,
            paper_appendix_selected_errors=225,
            observed_release_sentence_count=None,
            resolved=False,
            reason="Table 1 and Appendix A.2 differ; missing rows forbid reconciliation.",
        ),
        overlap_rows=[
            dict(
                corpus=c["name"],
                manifest=c["reference"],
                target_question_count=None,
                matched_count=None,
                status="not_computable",
                missing_reason=missing,
            )
            for c in manifest["corpora"]
        ],
        evidence_availability_rows=[
            dict(kind=kind, available=False, missing_reason=missing)
            for kind in [
                "sentence_labels",
                "gold_evidence_passages",
                "passage_relevance_labels",
                "span_offsets",
                "review_flags",
                "error_types",
                "row_generator_identity",
            ]
        ],
        rights_reference=dict(
            original_license=hf["cardData"]["license"],
            original_license_scope="Original release metadata only.",
            assisted_license=None,
            intended_evaluation_rights_confirmed=False,
        ),
        eligible_question_inventory=dict(sealed=False, question_hashes=[], missing_reason=missing),
        future_protocol_conditions=[
            "Acquire original and assisted release bytes under explicit terms, without credentials.",
            "Enumerate at most15000 rows with stable IDs, unchanged spans and hashes; keep text private.",
            "Reconcile224 versus225 and audit missing flags, error types, generators and label fields.",
            "Normalize questions/sources with NFKC, casefold and whitespace collapse before hashing.",
            "Compare every recorded exposure manifest, including LPHB, HalluTruthQA and the learning panel.",
            "A negative match only addresses recorded exposure; generator training contamination remains unknown.",
            "Seal a disjoint question inventory before any fitting or scoring; acquire blinded adjudication.",
            "Prospectively sample responses for population estimates; selected gold cannot supply them.",
        ],
        source_artifact_hashes=[manifest["retrievals"], *refs],
        preserved_history=[
            dict(reference=r, honest_verdict=read(r)["honest_verdict"])
            for r in manifest["upstreams"]
        ],
    )


def measure(manifest: Path, output: Path, run_date: str) -> Json:
    """Save immutable operands and real clocks before any validation or publication."""
    datetime.strptime(run_date, "%Y%m%d")
    start = time.monotonic_ns()
    scratch = Path(tempfile.mkdtemp(prefix="carnot8597-", dir="/var/tmp"))
    os.chmod(scratch, 0o700)
    probe = scratch / "disk_probe"
    probe.write_bytes(b"writable disk-backed scratch")
    with probe.open("rb") as stream:
        os.fsync(stream.fileno())
    reduced = inspect(manifest)
    preconditions = dict(
        scratch=str(scratch),
        writable_disk_backed=True,
        filesystem=next(
            line.split()[2]
            for line in Path("/proc/mounts").read_text().splitlines()
            if line.split()[1] == "/"
        ),
        memory=next(
            line
            for line in Path("/proc/meminfo").read_text().splitlines()
            if line.startswith("MemAvailable:")
        ),
        disk_free_bytes=shutil.disk_usage(scratch).free,
        tools={n: (ROOT / ".venv/bin" / n).is_file() for n in ["python", "pytest", "ruff", "mypy"]},
        optional_cuda_required=False,
        optional_board_required=False,
        measurement_deadline_s=3600,
        validation_reserve_s=900,
        invocation_cap_s=4800,
    )
    for tool, available in preconditions["tools"].items():
        if not available:
            reduced["gate_check_summary"].append(
                gate(reference(manifest), "tool_" + tool, True, False, "local_environment")
            )
    end = time.monotonic_ns()
    primitive = dict(
        manifest=reference(manifest),
        reduction=reduced,
        run_date=run_date,
        preconditions_checked=preconditions,
        duration_s=(end - start) / 1e9,
        started_wall_ns=time.time_ns(),
        phase_spans=[
            dict(phase="authenticate_and_audit", started_monotonic_ns=start, ended_monotonic_ns=end)
        ],
        code_config_hashes=[reference(ROOT / n) for n in OWNED],
    )
    raw = output.parent / "raw" / NAME / "invocations" / str(time.time_ns())
    atomic_json(raw / "primitive.json", primitive)
    return reference(raw / "primitive.json")


def build(source: Json, receipts: list[Json], output: Path) -> Json:
    """A finished audit can be blocked; a failed owned check must disqualify it."""
    primitive = read(source)
    reduced = primitive["reduction"]
    passed = bool(receipts) and all(r.get("passed") is True for r in receipts)
    findings = [
        json.loads(Path(r["stdout_path"]).read_bytes())
        for r in receipts
        if r["name"] == "adversarial"
    ]
    value = dict(
        reduced,
        experiment_id=8597,
        task_id=TASK,
        milestone="2026.10.738",
        run_date=primitive["run_date"],
        verdict_class="blocked" if passed else "disqualified",
        honest_verdict="complete_blocked_medexpert_release_unavailable"
        if passed
        else "complete_disqualified_medexpert_owned_validation",
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=MODEL_SPECS,
        model_invocation_counts=dict(loads=0, generation=0, llm_calls=0),
        historical_model_provenance=dict(
            invoked=False,
            scope="Original publication response generators only.",
            generators=[
                "Llama-2 Chat 7B",
                "Llama3.3 70B Instruct",
                "OLMo-2 13B",
                "Gemma-2 Instruct 27B",
                "OpenBioLLM-70B",
            ],
        ),
        actual_numerical_inference=dict(
            performed=False,
            energy_head=None,
            checkpoint=None,
            reason="No detector or head is evaluated.",
        ),
        verifier_is_oracle=False,
        exposure_scope="Only recorded corpus manifests are authenticated; target questions unavailable. Unknown training contamination is unresolved.",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        acceptance_gates=[
            dict(criterion=k, passed=bool(v)) for k, v in reduced.items() if k.endswith("_score")
        ],
        validation_receipts=receipts,
        terminal_validation_sidecar_path=str(
            output.parent / "raw" / NAME / "terminal_validation.json"
        ),
        adversarial_findings=findings,
        primitive_reference=source,
        raw_shard_hashes=[source],
        random_seed=8597,
        methodology="Authenticate author-linked documents and release metadata. Separate original labels, assisted adjudication, selection and evidence availability. No model, head, benchmark scoring or training.",
        cited_upstream_artifacts=reduced["source_artifact_hashes"],
        **{
            k: primitive[k]
            for k in ["preconditions_checked", "duration_s", "phase_spans", "code_config_hashes"]
        },
    )
    if not passed:
        value["release_audit_ready_score"] = 0
        value["gate_check_summary"] = [
            *reduced["gate_check_summary"],
            gate(source, "owned_required_checks", True, False, TASK),
        ]
    value["field_principles"] = {
        key: "Bind this invocation to authenticated evidence; missing differs from zero. Completion grants no detector benefit."
        for key in value
    }
    value["field_principles"].update(
        original_annotation_authority_score="One documents original clinician protocol; row authority and blinding remain unverified.",
        release_audit_ready_score="One means completed bounded audit, including explicit release blockage. It does not permit evaluation.",
        release_available_score="Zero means required original and assisted row bytes were not acquired; public original metadata is separate.",
        assisted_label_independence_score="Zero because six verifier votes and reasoning were shown during adjudication.",
        question_disjoint_score="Zero means unqualified; missing target questions cannot produce a negative overlap match.",
        rights_eligible_score="Zero means original CC BY4.0 metadata does not establish terms for the unavailable assisted release.",
        evaluation_eligible_score="Zero keeps selected, assisted and unavailable labels out of independent evaluation.",
        intended_count="Unknown sentence denominator; paper counts are claims rather than measured released row counts.",
        completed_count="Observed sentence rows acquired in this invocation, not documentary paper counts.",
        independent_count="Zero eligible independent evaluation units; repeated claims do not create units.",
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> Json:
    """Recompute from original authority so a repaired checksum cannot bless changed results."""
    value = json.loads(path.read_bytes())
    primitive = read(value["primitive_reference"])
    if primitive["code_config_hashes"] != [reference(ROOT / n) for n in OWNED]:
        raise ValueError("code_hash")
    read(primitive["manifest"])
    if inspect(Path(primitive["manifest"]["path"])) != primitive["reduction"]:
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
    if build(value["primitive_reference"], value["validation_receipts"], output) != value:
        raise ValueError("reduction_drift")
    return dict(passed=True, replay_passed=True, primary_sha256=reference(path)["sha256"])
