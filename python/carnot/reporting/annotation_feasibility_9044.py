"""REQ-REPORT-9044-V772: establish annotation structure without opening human outcomes.

Original annotator IDs and a span schema can establish availability independently
of source closure. They cannot qualify a cohort, label support or factual truth.
"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.human_disagreement_source_9015 import (
    COUNTS,
    PINNED,
    REVISION as REVISION,
    SALT,
    gate,
    ref as ref,
)

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9044_v772_annotation_feasibility"
CLI = "scripts/experiments/" + NAME + ".py"
OWNED = [
    "python/carnot/reporting/annotation_feasibility_9044.py",
    "python/carnot/reporting/annotation_feasibility_runner_9044.py",
    CLI,
]
TEST = "tests/python/test_annotation_feasibility_9044.py"
DEFAULT_INPUT = (
    ROOT / "results/raw/experiment_9015_v770_human_disagreement_source/release/manifest.json"
)
MODEL_SPECS: list[Json] = []
COLUMNS = ["id", "lang", "annotations.list.element.annotator_id"]
CRITERION_PHRASE = "not supported by the provided context"
AUTHORITY = {
    "openspec/change-proposals/research-roadmap-v771-preserved-20261112.md": "645327851ea953d7cfd0961838dcc36eb7e10dd8dbabd32a81c52adadb76d3fd",
    "openspec/change-proposals/research-roadmap-v771-queue-preserved-20261112.yaml": "2f18f190413146ad123228723a2e9d0780981a5c69a471473b5d06b757041d42",
    "results/experiment_9015_v770_human_disagreement_source.json": "94fefec6e66dbbca4974fbfdab9f5c470bfc515a62d1d0314f1793169872d96a",
    "results/experiment_9029_v771_source_identity_resolution.json": "a9ca396d7bc115c221a948bb835a3da3e6221feba16dbdf65bf0779fc19a2120",
    "results/experiment_9033_human_label_methods.json": "e8a1bd9080155673562425c1ab97bbf1bd0ab52fa273f25c8fd97f96b2614dc0",
    "results/raw/experiment_8388_v723_contract_methods/methods/2610.08026v1.html": "466a1bed106aebc8015203d724e11a6373f219378273a820dc07c2bd9e1c00c8",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush completed counts so supervisors can distinguish progress from silence."""
    print(f"[exp9044] phase={phase} completed={completed} pending={pending}", flush=True)


def structural_row(row: Json, split: str, span_schema: bool) -> Json:
    """Inspect people only; span contents and even span-list lengths remain unread."""
    annotations = row.get("annotations") or []
    ids = [a.get("annotator_id") for a in annotations]
    valid = [i for i in ids if isinstance(i, str) and i.strip()]
    # The retained FAQ bounds participants; oversized rosters need completion authority.
    limit = {"en": 12, "zh": 12, "de": 4}.get(row["lang"].lower(), 3)
    mismatch = len(ids) > limit
    ready = span_schema and 3 <= len(ids) <= limit and len(set(valid)) == len(ids)
    return dict(
        unit_id=row["id"],
        group_id="unassigned:" + row["id"],
        official_split=split,
        language=row["lang"],
        annotator_ids=ids,
        annotator_count=len(ids),
        distinct_nonempty_annotator_count=len(set(valid)),
        documented_annotator_limit=limit,
        annotator_count_exceeds_documented_limit=mismatch,
        original_judgment_participation="consistent_with_documented_count"
        if ready
        else "unqualified",
        original_span_schema_present=span_schema,
        structure_ready=ready,
        labels_value_validity="withheld",
        outcome_status="withheld",
        source_group_eligibility="unknown",
    )


def inspect(manifest: Path) -> Json:
    """Authenticate author bytes before projecting IDs from every official record."""
    gates: list[Json] = []
    references = [ref(manifest)]
    rows: list[Json] = []
    schemas: list[Json] = []
    author = False

    def check(name: str, expected: Any, observed: Any, reference: Json, field: str) -> None:
        operand = gate(name, expected, observed, reference)
        operand["artifact_field"] = field
        gates.append(operand)
        if not operand["passed"]:
            raise ValueError(name)

    try:
        data = json.loads(manifest.read_bytes())
        author = data["scope"] == "author_release"
        check("release_revision", REVISION, data["revision"], ref(manifest), "revision")
        for key in [*PINNED, "metadata"]:
            reference = data["metadata"] if key == "metadata" else data["files"][key]
            references.append(reference)
            actual = ref(Path(reference["path"]))["sha256"]
            check("retained_" + key, reference["sha256"], actual, reference, "sha256")
            if author and key in PINNED:
                check("pinned_" + key, "sha256:" + PINNED[key], actual, reference, "sha256")
        metadata = json.loads(Path(data["metadata"]["path"]).read_bytes())
        check("metadata_revision", REVISION, metadata["sha"], data["metadata"], "sha")
        check(
            "license_metadata",
            "cc-by-4.0",
            metadata["cardData"]["license"],
            data["metadata"],
            "cardData.license",
        )
        for key, phrase in [
            ("policy", "Raw annotations"),
            ("policy", "annotator_id"),
            ("guidelines", CRITERION_PHRASE),
            ("license", "Creative Commons"),
        ]:
            text = Path(data["files"][key]["path"]).read_text()
            if not author and key == "license":
                phrase = "CC BY 4.0"
            check("original_" + key, True, phrase in text, data["files"][key], phrase)
        for split in ["validation", "test"]:
            progress("before_id_projection_" + split)
            file = pq.ParquetFile(data["files"][split]["path"])
            schema = file.schema_arrow
            expected = pa.list_(
                pa.struct(
                    [
                        pa.field("annotator_id", pa.string()),
                        pa.field("labels", pa.list_(pa.list_(pa.int64()))),
                    ]
                )
            )
            span_schema = (
                "annotations" in schema.names and schema.field("annotations").type == expected
            )
            leaves = {file.schema.column(i).path for i in range(len(file.schema))}
            columns = COLUMNS if COLUMNS[-1] in leaves else ["id", "lang"]
            public = file.read(columns=columns).to_pylist()
            rows.extend(structural_row(row, split, span_schema) for row in public)
            schemas.append(
                dict(
                    split=split,
                    reference=data["files"][split],
                    record_count=file.metadata.num_rows,
                    columns_read=columns,
                    span_schema_present=span_schema,
                    schema=str(schema),
                    label_values_read=False,
                )
            )
            progress("after_id_projection_" + split, len(public), 0)
        check(
            "original_participation_authority",
            0,
            sum(r["annotator_count_exceeds_documented_limit"] for r in rows),
            data["files"]["guidelines"],
            "all_records.annotator_count_exceeds_documented_limit",
        )
        complete = bool(rows) and all(r["structure_ready"] for r in rows)
        check(
            "original_individual_judgments",
            True,
            complete,
            ref(manifest),
            "all_records.distinct_nonempty_original_annotators>=3_and_span_schema",
        )
    except (OSError, ValueError, KeyError, TypeError) as error:
        if not gates or gates[-1]["passed"]:
            observed = (
                "missing_field:" + str(error)
                if isinstance(error, KeyError)
                else "missing_file:" + str(error)
                if isinstance(error, FileNotFoundError)
                else str(error)
            )
            gates.append(
                gate("authenticated_inputs", "retained original authority", observed, ref(manifest))
            )
    ready = int(author and bool(rows) and all(g["passed"] for g in gates))
    units = [
        dict(
            unit_id=r["unit_id"],
            group_id=r["group_id"],
            arm="original_human",
            condition=r["official_split"],
            metric="annotation_structure_available",
            value=int(r["structure_ready"]),
            numerator=int(r["structure_ready"]),
            denominator=1,
            status="completed",
            exclusion_reason=None,
        )
        for r in rows
    ]
    missing = [g for g in gates if not g["passed"]]
    return dict(
        annotation_schema_rows=rows,
        rows=units,
        intended_count=len(units),
        completed_count=len(units),
        independent_count=0,
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        annotation_structure_ready_score=ready,
        natural_pilot_ready_score=0,
        gate_check_summary=gates,
        missing_authority=missing,
        source_artifact_hashes=references,
        access_receipts=schemas,
        test_outcome_values_accessed=False,
        annotation_count_histograms={
            s: dict(Counter(str(r["annotator_count"]) for r in rows if r["official_split"] == s))
            for s in ["validation", "test"]
        },
        participation_mismatch_counts={
            s: sum(
                r["annotator_count_exceeds_documented_limit"]
                for r in rows
                if r["official_split"] == s
            )
            for s in ["validation", "test"]
        },
        original_criterion=dict(
            definition=CRITERION_PHRASE,
            provenance="Author dataset card raw annotations and retained official FAQ instructions",
            instructions="Minimal character spans; conservative content-word marking; NO HALLUCINATION or UNSURE comments",
            reference="Wikipedia reference used by original annotators; reference support is not factual truth",
            full_instruction_document="not retained; FAQ supplies original public definition and approximate instructions",
            uncertainty_comments="not present in released annotation struct; unknown, never fabricated",
        ),
        criterion_mapping=dict(
            y_soft="fraction of original annotators marking any nonempty span",
            y_hard="y_soft >= 0.5",
            disagreement="person-level response votes, never majority-character union",
            claim="fallible original human reference-support risk; not factual correctness",
            ties="retained under >=0.5 rule",
            offset_validity="unknown; no actual label values read",
        ),
        pilot_contract=dict(
            salt=SALT,
            role_counts=COUNTS,
            class_minima=dict(FIT=5, CAL=5, TEST=20),
            source_disjoint_unseen_groups=True,
            original_annotators_per_record=3,
            cohort_selected=False,
            source_closure="unknown here; read separate Exp9029 gate",
            frozen_cohort="unqualified",
            class_support="unknown; outcomes withheld",
            assessable_without_outcomes=[
                "release authority",
                "record counts",
                "annotator IDs",
                "span schema",
            ],
            unknown_without_further_authority=[
                "unseen source closure",
                "frozen400-group cohort",
                "per-class minima",
                "span validity",
                "original uncertainty",
            ],
        ),
        structural_checksum=canonical_hash(rows),
    )
