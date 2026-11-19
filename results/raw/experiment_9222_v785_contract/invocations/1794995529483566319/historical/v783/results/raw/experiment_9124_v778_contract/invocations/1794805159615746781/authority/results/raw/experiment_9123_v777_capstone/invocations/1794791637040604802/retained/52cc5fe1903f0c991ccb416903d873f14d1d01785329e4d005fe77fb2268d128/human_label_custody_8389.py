"""REQ-REPORT-8389: released human labels need byte and question custody."""

from __future__ import annotations

from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import time
from typing import Any, cast
import unicodedata
import urllib.request

from carnot.reporting.current_work_receipt import canonical_hash, atomic_json
from carnot.reporting.external_evidence_8375 import (
    reference as reference,
    read_reference as read_reference,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8389_v723_human_label_custody"
TASK = "exp8389-human-label-custody"
CLI = f"scripts/experiments/{NAME}.py"
TEST = "tests/python/test_human_label_custody_8389.py"
COMMIT = "c2cf053567d5ae458656ee0e420509e35d73406d"
MASTER = "data/human_judge_release_master.csv"
AUDIT = "audits/factorial_prompt_run/gpt5mini_factorial_v3/labels.csv"
GENERATORS = ("gemma-2-9b", "llama3-8b", "mistral-7b")
MODEL_SPECS: list[Json] = []
OWNED = [
    "python/carnot/reporting/human_label_custody_8389.py",
    "python/carnot/reporting/human_label_custody_runner_8389.py",
    CLI,
]
TARGET = dict(
    primary="annotator1_label",
    sensitivity="annotator2_label",
    encoding={"0": "correct", "1": "hallucination", "-1": "missing_or_uncertain"},
    criterion="factual_correctness",
    cluster="normalized_question",
    support_floor=80,
    per_class_floor=8,
    advertised_questions=300,
    advertised_responses=900,
)
# Reviewed git blobs remain independent of any subsequently rehashed manifest.
RELEASE_FILES = {
    "README.md": "d5328d87d0780e4a3754cc4e0f69725ba5fdd8e4",
    "LICENSE-DATA": "7173e81e912d641f5f67a5186dbb7d04826ca905",
    "THIRD_PARTY_NOTICES.md": "68c54ebf887b551df102948170b20696da9a355c",
    "data/README.md": "cdc50b26705f7a4a31d0eb686ebcd540425bb3ef",
    "data/human_judge_release_data_dictionary.csv": "f822798f8f5b0d69909780bc1f1ed7fc90576943",
    MASTER: "a5540b1121358aee3f26655a6275948408a5d4ac",
    AUDIT: "fdd45c8474957ec1c493c7314ecb9815c7c9acc6",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts expose unfinished work before a supervisor deadline."""
    print(f"[exp8389] phase={phase} completed={completed} pending={pending}", flush=True)


def normalized(text: str) -> str:
    """Question formatting differences must not create independent examples."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def text_hash(text: str) -> str:
    """Hash exact response text while keeping the generated text private."""
    return "sha256:" + hashlib.sha256(text.encode()).hexdigest()


def gate(
    path: str,
    field: str,
    expected: Any,
    observed: Any,
    digest: Any = None,
    operator: str = "==",
    upstream: str = TASK,
) -> Json:
    """Name unavailable operands explicitly so absence cannot be mistaken for zero."""
    return dict(
        check=field,
        upstream=upstream,
        path=path,
        sha256=digest,
        field=field,
        operator=operator,
        expected_value=expected,
        observed_value=observed,
        passed=False,
    )


def authenticate(name: str, ref: Json) -> bytes:
    """A local SHA-256 cannot replace the independently reviewed author git blob."""
    data = read_reference(ref)
    blob = hashlib.sha1(
        b"blob " + str(len(data)).encode() + b"\0" + data, usedforsecurity=False
    ).hexdigest()
    if blob != RELEASE_FILES[name]:
        raise ValueError("reviewed_blob:" + name)
    return cast(bytes, data)


def acquire(raw: Path) -> Json:
    """Fetch only pinned public bytes; never run author installation or regeneration."""
    started = time.monotonic()
    refs: Json = {}
    gates: list[Json] = []
    total = 0
    for index, name in enumerate(RELEASE_FILES):
        progress("before_retrieval_" + name, index, len(RELEASE_FILES) - index)
        url = f"https://raw.githubusercontent.com/jova486/LPHB/{COMMIT}/{name}"
        try:
            remaining = 600 - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("acquisition_deadline")
            with urllib.request.urlopen(url, timeout=min(30, remaining)) as response:
                data = response.read(50 * 1024 * 1024 - total + 1)
            total += len(data)
            if total > 50 * 1024 * 1024:
                raise ValueError("acquisition_byte_cap")
            path = raw / "release" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            ref = reference(path)
            authenticate(name, ref)
            path.chmod(0o400)
            refs[name] = dict(ref, url=url, git_blob=RELEASE_FILES[name])
        except (OSError, ValueError) as error:
            gates.append(gate(url, "released_bytes_available", RELEASE_FILES[name], str(error)))
        progress("after_retrieval_" + name, index + 1, len(RELEASE_FILES) - index - 1)
    return dict(
        release_commit=COMMIT,
        release_files=refs,
        acquisition_gates=gates,
        acquisition_bytes=total,
        acquisition_duration_s=time.monotonic() - started,
        target_mapping=dict(TARGET),
        license_permitted=len(refs) == len(RELEASE_FILES),
        dataverse_limit=dict(
            doi="10.7910/DVN/PCHISZ",
            status=403,
            scope="planning observation; no current retry or access bypass",
        ),
    )


def overlap_inventory(root: Path, raw: Path) -> list[Json]:
    """Opaque old source hashes cannot establish question-level disjointness."""
    paths = [
        root / "results/experiment_8305_v717_cached_sentence_custody.json",
        root / "results/raw/experiment_8375_v722_external_evidence_readiness/release_manifest.json",
    ]
    inventory: list[Json] = []
    for index, path in enumerate(paths):
        if not path.is_file():
            inventory.append(dict(path=str(path), sha256=None, comparable=False, rows=[]))
            continue
        value = json.loads(path.read_bytes())
        copy = raw / "prior_sources" / f"{index}.json"
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
        rows = value.get("source_role_manifest", value.get("source_overlap_rows", []))
        comparable = bool(rows) and all("question" in row or "question_hash" in row for row in rows)
        inventory.append(
            dict(
                reference(copy),
                source_path=str(path),
                comparable=comparable,
                rows=rows,
                reason=None if comparable else "no_comparable_question_identity",
            )
        )
    return inventory


def validate_manifest(manifest: Json) -> Json:
    """Only this validator opens targets after checking the frozen mapping.

    Human labels describe factual correctness. Reference matching and released
    model judgments are retained as separate signals, never replacement targets.
    """
    if manifest["target_mapping"] != TARGET or manifest["release_commit"] != COMMIT:
        raise ValueError("target_mapping_or_release_commit")
    files = manifest["release_files"]
    operands = {name: authenticate(name, ref) for name, ref in files.items()}
    master = list(csv.DictReader(io.StringIO(operands.get(MASTER, b"").decode("utf-8-sig"))))
    audit = list(csv.DictReader(io.StringIO(operands.get(AUDIT, b"").decode("utf-8-sig"))))
    gates = list(manifest["acquisition_gates"])
    audit_index = {(r["dataset"], normalized(r["qid"]), r["model"]): r for r in audit}
    alignment = len(audit_index) == len(audit) == len(master) == 900
    prior = manifest["overlap_inventory"]
    for item in prior:
        if not item["comparable"]:
            gates.append(
                gate(
                    item.get("source_path", item["path"]),
                    "normalized_question_identity_available",
                    True,
                    None,
                    item.get("sha256"),
                    upstream=item.get("upstream", "prior_Carnot_source_manifest"),
                )
            )
        if item.get("sha256"):
            original = json.loads(read_reference(item))
            read_reference(dict(path=item["source_path"], sha256=item["sha256"]))
            original_rows = original.get(
                "source_role_manifest", original.get("source_overlap_rows", [])
            )
            comparable = bool(original_rows) and all(
                "question" in r or "question_hash" in r for r in original_rows
            )
            if item["rows"] != original_rows or item["comparable"] != comparable:
                raise ValueError("source_overlap_custody")
    known = bool(prior) and all(p["comparable"] for p in prior)
    prior_hashes = {
        r.get("question_hash", text_hash(normalized(r.get("question", ""))))
        for p in prior
        for r in p["rows"]
    }
    prior_ids = {normalized(r["qid"]) for p in prior for r in p["rows"] if r.get("qid")}
    seen: set[tuple[str, str, str]] = set()
    question_ids: dict[str, set[tuple[str, str]]] = defaultdict(set)
    identities: dict[tuple[str, str], set[str]] = defaultdict(set)
    classes: dict[int, set[str]] = {0: set(), 1: set()}
    independent: set[str] = set()
    rows: list[Json] = []
    counts: Counter[str] = Counter(
        {
            k: 0
            for k in (
                "A1_valid",
                "A1_missing",
                "A2_valid",
                "A2_missing",
                "A1_correct",
                "A1_hallucination",
                "A1_A2_disagreement",
            )
        }
    )
    for index, rec in enumerate(master):
        if index % 100 == 0:
            progress("manifest_validation", index, len(master) - index)
        key = (rec["dataset"], normalized(rec["qid"]), rec["model"])
        cluster = text_hash(normalized(rec["question"]))
        answer_hash = text_hash(rec["answer_text"])
        if key in seen or rec["model"] not in GENERATORS or not rec["question"].strip():
            alignment = False
        seen.add(key)
        question_ids[cluster].add(key[:2])
        identities[key[:2]].add(cluster)
        peer = audit_index.get(key)
        joined = bool(
            peer
            and peer["item_id"] == rec["item_id"]
            and peer["question"] == rec["question"]
            and text_hash(peer["answer_text"]) == answer_hash
            and peer["annotator1_label"] == rec["annotator1_label"]
        )
        alignment = alignment and joined
        a1, a2 = int(rec[TARGET["primary"]]), int(rec[TARGET["sensitivity"]])
        if a1 not in (-1, 0, 1) or a2 not in (-1, 0, 1):
            raise ValueError("human_label_encoding")
        counts["A1_valid" if a1 >= 0 else "A1_missing"] += 1
        if a1 >= 0:
            counts["A1_correct" if a1 == 0 else "A1_hallucination"] += 1
        counts["A2_valid" if a2 >= 0 else "A2_missing"] += 1
        disagreement = a1 >= 0 and a2 >= 0 and a1 != a2
        counts["A1_A2_disagreement"] += int(disagreement)
        overlap = (
            "overlap"
            if cluster in prior_hashes or key[1] in prior_ids
            else "disjoint"
            if known
            else "unknown"
        )
        eligible = a1 >= 0 and joined and overlap == "disjoint"
        if eligible:
            independent.add(cluster)
            classes[a1].add(cluster)
        automatic = {k: int(v) for k, v in rec.items() if k.startswith("label_")}
        reference_scores = {
            k: v for k, v in rec.items() if k.startswith(("rouge", "bert", "meteor", "bart", "nli"))
        }
        rows.append(
            dict(
                unit_id=rec["item_id"],
                arm="A1_factual_correctness",
                family=cluster,
                seed=None,
                cluster=cluster,
                qid=rec["qid"],
                normalized_qid=key[1],
                dataset=key[0],
                generator=key[2],
                question_hash=cluster,
                response_hash=answer_hash,
                human_primary=a1 if a1 >= 0 else None,
                human_sensitivity=a2 if a2 >= 0 else None,
                human_provenance=dict(
                    A1=dict(file=MASTER, column=TARGET["primary"], row=index + 2),
                    A2=dict(file=MASTER, column=TARGET["sensitivity"], row=index + 2),
                ),
                automatic_judgments=automatic,
                reference_faithfulness_scores=reference_scores,
                audit_aligned=joined,
                disagreement=disagreement,
                overlap=overlap,
                status="completed",
                absolute_metric=int(eligible),
                raw_numerator=int(eligible),
                raw_denominator=1,
                censored=False,
                excluded=not eligible,
                missing_reason=None
                if eligible
                else "A1_missing"
                if a1 < 0
                else "audit_join"
                if not joined
                else overlap + "_overlap",
            )
        )
    cluster_identity = all(len(v) == 1 for v in question_ids.values()) and all(
        len(v) == 1 for v in identities.values()
    )
    alignment = alignment and cluster_identity and len(identities) == len(question_ids) == 300
    missing: list[Json] = []
    for dataset, qid in sorted(identities):
        for model in GENERATORS:
            if (dataset, qid, model) not in seen:
                missing.append(
                    dict(qid=qid, dataset=dataset, generator=model, reason="missing_generator")
                )
    for i in range(max(0, 300 - len(identities))):
        missing.extend(
            dict(qid=None, question_slot=i, generator=m, reason="question_identity_unavailable")
            for m in GENERATORS
        )
    for index, slot in enumerate(missing):
        rows.append(
            dict(
                unit_id=f"missing:{index}",
                arm="A1_factual_correctness",
                family=None,
                seed=None,
                status="censored",
                absolute_metric=0,
                raw_numerator=0,
                raw_denominator=1,
                censored=True,
                excluded=False,
                missing_reason=slot["reason"],
                **slot,
            )
        )
    checks = dict(
        release_identity=len(files) == len(RELEASE_FILES),
        license_permitted=manifest["license_permitted"],
        response_label_alignment=alignment,
        complete_slots=len(master) == 900 and not missing,
        known_question_overlap=known,
        disjoint_question_floor=len(independent) >= 80,
        each_class_question_floor=all(len(c) >= 8 for c in classes.values()),
    )
    for field, passed in checks.items():
        if not passed:
            observed: Any = (
                len(independent)
                if field == "disjoint_question_floor"
                else {str(k): len(v) for k, v in classes.items()}
                if field == "each_class_question_floor"
                else None
                if field in ("known_question_overlap", "release_identity")
                else passed
            )
            gates.append(
                gate(
                    files.get(MASTER, {}).get("path", MASTER),
                    field,
                    80
                    if field == "disjoint_question_floor"
                    else 8
                    if field == "each_class_question_floor"
                    else True,
                    observed,
                    files.get(MASTER, {}).get("sha256"),
                    ">=" if field.endswith("floor") else "==",
                )
            )
    progress("manifest_validation_complete", len(master), len(missing))
    if "validated_rows" in manifest and manifest["validated_rows"] != rows:
        raise ValueError("manifest_rows_drift")
    return dict(
        rows=rows,
        intended_count=900,
        completed_count=len(master),
        failed_count=0,
        censored_count=len(missing),
        excluded_count=sum(r["excluded"] for r in rows),
        independent_count=len(independent),
        question_cluster_count=len(question_ids),
        response_count=len(master),
        label_authority_counts=dict(counts),
        overlap_rows=[
            dict(
                qid=r["qid"],
                generator=r["generator"],
                question_hash=r["question_hash"],
                status=r["overlap"],
            )
            for r in rows
            if r["status"] == "completed"
        ],
        missing_slots=missing,
        class_cluster_support={str(k): len(v) for k, v in classes.items()},
        audit_response_count=len(audit),
        audit_alignment=alignment,
        cluster_identity_valid=cluster_identity,
        acceptance_gates=checks,
        gate_check_summary=gates,
        released_label_panel_ready_score=int(all(checks.values()) and not gates),
    )


def build(work: Json, receipts: list[Json]) -> Json:
    """Successful execution establishes custody only, never detector benefit."""
    manifest = json.loads(read_reference(work["manifest_reference"]))
    value = validate_manifest(manifest)
    gates = value["gate_check_summary"] + work["gate_check_summary"]
    passed = bool(receipts) and all(r["passed"] for r in receipts if r.get("scope") != "global")
    kind = "disqualified" if not passed else "blocked" if gates else "positive"
    value.update(
        experiment_id=8389,
        task_id=TASK,
        milestone="2026.10.723",
        run_date="20261010",
        honest_verdict=f"complete_{kind}_human_label_custody",
        verdict_class=kind,
        gate_check_summary=gates,
        inference_substrate="aggregation_from_upstream_artifacts",
        inference_substrate_class="no_model_load",
        MODEL_SPECS=[],
        model_invocation_counts=work["model_invocation_counts"],
        historical_model_provenance=[
            dict(
                generator=m,
                kind="cached_third_party_answers",
                source=MASTER,
                source_sha256=manifest["release_files"].get(MASTER, {}).get("sha256"),
                current_calls=0,
            )
            for m in GENERATORS
        ],
        sample_size_budget=dict(
            intended_responses=900,
            intended_questions=300,
            minimum_disjoint_questions=80,
            each_class_question_floor=8,
            repeats_are_independent=False,
        ),
        verifier_is_oracle=False,
        exposure_scope="released human labels; read-only custody; no detector fit",
        independent_generalization_score=0,
        generalized_learning_benefit_score=0,
        required_checks_passed=passed,
        flagged_adversarial=not passed,
        validation_receipts=receipts,
        terminal_validation_sidecar_path=work["terminal_validation_sidecar_path"],
        adversarial_findings=work.get("adversarial_findings", []),
        preconditions_checked=work["preconditions_checked"],
        duration_s=work["duration_s"],
        phase_spans=work["phase_spans"],
        random_seed=7238389,
        source_artifact_hashes=work["source_artifact_hashes"],
        code_config_hashes=work["code_config_hashes"],
        raw_shard_hashes=[work["manifest_reference"], work["primitive_reference"]],
        cited_upstream_artifacts=[
            dict(r, imported_fields="task authority, source identities or release bytes")
            for r in work["source_artifact_hashes"]
        ],
        reproducibility_checksum=canonical_hash(
            [
                work["manifest_reference"],
                work["source_artifact_hashes"],
                work["code_config_hashes"],
                7238389,
            ]
        ),
        release_commit=COMMIT,
        release_file_hashes=manifest["release_files"],
        source_manifest_path=work["manifest_reference"]["path"],
        license_manifest=dict(
            author="Jorma Valjakka et al.; LPHB",
            annotations="CC BY-SA 4.0",
            generated_answers={
                "gemma-2-9b": "Gemma Terms of Use",
                "llama3-8b": "Meta Llama 3 Community License",
                "mistral-7b": "Apache 2.0",
            },
            source_datasets={
                "hotpotqa": "CC BY-SA 4.0",
                "truthfulqa": "Apache 2.0",
                "triviaqa": "research use; underlying rights retained; code Apache 2.0",
            },
            permitted_scope="local read-only research audit; no redistribution or training",
            notice_operands=[
                manifest["release_files"].get(n)
                for n in ("LICENSE-DATA", "THIRD_PARTY_NOTICES.md", "data/README.md")
            ],
        ),
        target_mapping=manifest["target_mapping"],
        dataverse_limit=manifest.get("dataverse_limit"),
        acquisition_bytes=manifest.get("acquisition_bytes", 0),
        acquisition_duration_s=manifest.get("acquisition_duration_s", 0),
        manifest_reference=work["manifest_reference"],
        primitive_reference=work["primitive_reference"],
        work_reference=reference(Path(work["work_path"])),
        invocation_id=work["invocation_id"],
        authority=work["authority"],
        repository_health=work["repository_health"],
        methodology_note="A1 factual correctness is primary; A2 is partial sensitivity only. Automatic labels and reference scores are cached signals. Question clusters count once. Unknown overlap blocks custody readiness. No semantic benefit, detector fitting or current model execution. H1=-0.00390625 and H2=0 remain closed exposed-development findings; the dyadic-logit certificate is separate numerical policy.",
    )
    value["acceptance_gates"].update(
        owned_validation=passed, current_authority=not work["gate_check_summary"]
    )
    value["released_label_panel_ready_score"] = int(
        value["released_label_panel_ready_score"] and passed and not gates
    )
    principles = {}
    for fields, why in [
        (
            "rows intended_count completed_count failed_count censored_count excluded_count independent_count sample_size_budget question_cluster_count response_count label_authority_counts overlap_rows missing_slots class_cluster_support audit_response_count audit_alignment cluster_identity_valid",
            "Count responses and independent questions separately; preserve absence and human label provenance.",
        ),
        (
            "honest_verdict verdict_class gate_check_summary acceptance_gates released_label_panel_ready_score",
            "Custody readiness requires actual authorized aligned bytes and disjoint questions; absence blocks.",
        ),
        (
            "inference_substrate inference_substrate_class MODEL_SPECS model_invocation_counts historical_model_provenance",
            "Cached historical answers never become current mandated-model calls.",
        ),
        (
            "verifier_is_oracle exposure_scope independent_generalization_score generalized_learning_benefit_score methodology_note",
            "Custody feasibility establishes no learned detector or semantic benefit.",
        ),
        (
            "required_checks_passed flagged_adversarial validation_receipts terminal_validation_sidecar_path adversarial_findings repository_health",
            "Bound owned validation and every finding to exact candidate bytes; global health is separate.",
        ),
        (
            "release_commit release_file_hashes source_manifest_path license_manifest target_mapping dataverse_limit acquisition_bytes acquisition_duration_s",
            "Use reviewed public author bytes and frozen human targets within permitted read-only use and acquisition caps.",
        ),
        (
            "experiment_id task_id milestone run_date invocation_id authority",
            "Bind the actual invocation to its complete current task authority.",
        ),
        (
            "preconditions_checked duration_s phase_spans random_seed reproducibility_checksum source_artifact_hashes code_config_hashes raw_shard_hashes cited_upstream_artifacts manifest_reference primitive_reference work_reference",
            "Recompute from sealed source operands, exact code and bounded invocation evidence.",
        ),
    ]:
        principles.update({field: why for field in fields.split()})
    value["field_principles"] = dict(
        principles, field_principles="Explain every added field while preserving typed values."
    )
    return value


def replay(path: Path) -> bool:
    """Recompute primitive rows from reviewed release bytes, not rehashed summaries."""
    try:
        value = json.loads(path.read_bytes())
        work = json.loads(read_reference(value["work_reference"]))
        for ref in work["source_artifact_hashes"] + work["code_config_hashes"]:
            read_reference(ref)
        manifest = json.loads(read_reference(work["manifest_reference"]))
        if json.loads(read_reference(work["primitive_reference"])) != validate_manifest(manifest):
            raise ValueError("primitive_reduction_drift")
        if value != build(work, value["validation_receipts"]):
            raise ValueError("reduction_drift")
        from carnot.reporting.primary_publication import validate_primary

        validate_primary(value, ROOT / "results" / (NAME + ".json"))
        return True
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        progress("replay_rejected_" + type(error).__name__)
        return False
