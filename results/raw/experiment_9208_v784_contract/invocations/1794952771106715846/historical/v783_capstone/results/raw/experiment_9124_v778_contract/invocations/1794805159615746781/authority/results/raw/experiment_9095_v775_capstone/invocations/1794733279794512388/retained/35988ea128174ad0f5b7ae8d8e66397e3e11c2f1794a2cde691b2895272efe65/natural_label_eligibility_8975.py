"""REQ-REPORT-8975: labels alone cannot establish an accessible natural evaluation corpus."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import canonical_hash, sha256_file
from carnot.reporting.natural_criterion_audit_8908 import index

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_8975_v767_natural_label_eligibility"
TASK = "exp8975-natural-label-eligibility"
CLI = "scripts/experiments/" + NAME + ".py"
REVISION = "a62008a1bc1224610f691012f35f65eab054ce3d"
OWNED = [
    "python/carnot/reporting/natural_label_eligibility_8975.py",
    "python/carnot/reporting/natural_label_runner_8975.py",
    CLI,
]
CHECKS = {"owned_pytest", "owned_coverage", "ruff", "format", "mypy", "spec", "e2e018_021"}
PINNED = dict(
    labels="3bd0e8ec7b682c8092e36d204e47437cc5d0bcd1d402e21f8257962b206e57d8",
    license="578a4c00aae524f5fddacd40b040521ff60e728fe89867a5984769344965f440",
    readme="3f45bf1d7b96507786633a5d64fec7ad696a0023d6f2752de621c97e9b66999e",
    paper="063bf46a1d31dcaa582d50361beef23336bba27d9e53906e7f740bc236886d04",
    terms="651ed02ef953895318704dd95f365cd85a179d2d2a2c64ad117a33e8485c49f6",
    tree="0829bf4c9dd61cf2e92af558e1ab75747344adc3990e9f7bc5db6a2438ce1614",
)
EXCLUDED = [
    "occurrence1",
    "occurrence2",
    "category1",
    "category2",
    "explanation",
    "annotation_explanations",
    "detector_outputs",
    "adjudication",
    "evidence_labels",
]
CRITERIA = [
    dict(
        category=c,
        intended_task=t,
        independent_cited_evidence=False,
        evidence_basis="Human judgment; release provides no item-level evidence citations.",
    )
    for c, t in [
        ("Input-conflicting", "input adherence; can include instruction mismatch"),
        ("Context-conflicting", "internal/context consistency"),
        ("Fact-conflicting", "open-world factuality; external evidence unresolved"),
    ]
]


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report actual progress even when all source joins are blocked."""
    print(f"[exp8975] phase={phase} completed={completed} pending={pending}", flush=True)


def ref(path: Path) -> Json:
    """An absent file has no hash, which distinguishes absence from changed bytes."""
    return dict(path=str(path.absolute()), sha256=sha256_file(path) if path.is_file() else None)


def gate(field: str, expected: Any, observed: Any, reference: Json) -> Json:
    """Each failed condition names the exact upstream operand and its byte identity."""
    return dict(
        upstream="AuthenHallu/LMSYS-Chat-1M",
        check="eligibility:" + field,
        **reference,
        field=field,
        operator="==",
        expected=expected,
        observed=observed,
    )


def audit(labels: list[Json], access: Json, reference: Json | None = None) -> Json:
    """Keep every intended pair and distinguish annotation auditing from usable text joins."""
    reference = reference or dict(path=None, sha256=None)
    identities = Counter(r.get("conversation_id") for r in labels)
    permitted = (
        access.get("local_use_authorized") is True
        and access.get("model_terms_verified") is True
        and bool(access.get("license_reference"))
    )
    sources = index(access.get("records", []) if permitted else [], "conversation_id")
    joins, pairs = [], []
    for slot in range(max(400, len(labels))):
        label = labels[slot] if slot < len(labels) else {}
        ident = label.get("conversation_id", f"missing-slot-{slot}")
        candidates = sources.get(ident, [])
        source = candidates[0] if len(candidates) == 1 else {}
        messages = source.get("conversation", [])
        missing = []
        if not permitted:
            missing.append("accepted_local_use_and_model_terms")
        if len(candidates) != 1:
            missing.append("unique_source_record")
        if not isinstance(source.get("model"), str) or not source["model"]:
            missing.append("model")
        if type(source.get("source_offset")) is not int or source["source_offset"] < 0:
            missing.append("source_offset")
        roles = ["user", "assistant", "user", "assistant"]
        if len(messages) != 4 or any(
            m.get("role") != roles[j]
            or not isinstance(m.get("content"), str)
            or not m["content"].strip()
            for j, m in enumerate(messages[:4])
        ):
            missing.extend([f"conversation[{j}].content/role" for j in range(4)])
        if identities[ident] != 1 or not isinstance(ident, str) or not ident:
            missing.append("unique_dialogue_identity")
        for turn in (1, 2):
            occurrence, category = label.get(f"occurrence{turn}"), label.get(f"category{turn}")
            valid = (occurrence == "No Hallucination" and category is None) or (
                occurrence == "Hallucination"
                and category in {"Input-conflicting", "Context-conflicting", "Fact-conflicting"}
            )
            if not valid:
                missing.append(f"pair{turn}.valid_label")
        joined = not missing
        joins.append(
            dict(
                conversation_id=ident,
                group_id=ident,
                release_row_offset=slot,
                source_offset=source.get("source_offset"),
                source_model=source.get("model"),
                source_message_offsets=[0, 1, 2, 3] if joined else None,
                missing_fields=missing,
                joined_pair_count=2 if joined else 0,
                duplicate_identity_count=identities[ident] - 1,
                source_match_count=len(candidates),
            )
        )
        for turn in (1, 2):
            pairs.append(
                dict(
                    unit_id=f"{ident}:pair{turn}:slot{slot}",
                    conversation_id=ident,
                    group_id=ident,
                    pair_index=turn,
                    arm="source_label_join",
                    occurrence=label.get(f"occurrence{turn}"),
                    category=label.get(f"category{turn}"),
                    annotation_present=f"occurrence{turn}" in label,
                    status="completed" if joined else "censored",
                    source_joined=int(joined),
                    missing_fields=missing,
                    source_message_offsets=[2 * turn - 2, 2 * turn - 1],
                    text_sha256=canonical_hash(messages[: 2 * turn]) if joined else None,
                )
            )
        if slot % 100 == 99:
            progress("dialogue_join", slot + 1, max(400, len(labels)) - slot - 1)
    completed = sum(r["source_joined"] for r in pairs)
    ready = int(len(labels) == 400 and len(identities) == 400 and completed == 800)
    gates = [
        gate(k, wanted, got, reference)
        for k, wanted, got in [
            ("dialogue_count", 400, len(labels)),
            ("unique_dialogue_count", 400, len(identities)),
            ("authorized_local_use", True, permitted),
            ("joined_pairs", 800, completed),
        ]
        if wanted != got
    ]
    for field in ("local_use_authorized", "model_terms_verified", "license_reference"):
        if not access.get(field):
            gates.append(
                gate(
                    "source_access." + field,
                    "explicit existing authorization",
                    access.get(field, "missing_field"),
                    reference,
                )
            )
    groups = sorted((r["group_id"] for r in joins), key=lambda g: canonical_hash([8975, g]))
    splits = [
        dict(split=n, group_ids=groups[a:b], pair_count=2 * (b - a))
        for n, a, b in [("train", 0, 240), ("validation", 240, 320), ("test", 320, 400)]
    ]
    return dict(
        rows=pairs,
        dialogue_join_rows=joins,
        intended_count=len(pairs),
        completed_count=completed,
        failed_count=0,
        censored_count=len(pairs) - completed,
        excluded_count=0,
        independent_count=0,
        independent_generalization_score=0,
        metadata_audited_count=sum(r["annotation_present"] for r in pairs),
        annotation_criterion_rows=CRITERIA,
        natural_evidence_ready_score=ready,
        source_access_status="joined_authorized" if ready else "blocked_source_access",
        gate_check_summary=gates,
        duplicate_dialogue_count=sum(v - 1 for v in identities.values()),
        category_counts=dict(Counter(str(r["category"]) for r in pairs)),
        eligibility_matrix=[
            dict(task="annotation_relative_detection", eligible=True, independent_truth=False)
        ]
        if ready
        else [],
        proposed_group_splits=splits if ready else [],
        excluded_feature_keys=EXCLUDED,
        split_protocol="Hash(seed8975, conversation_id), 240/80/80 groups, both turns together; deduplicate text/source clusters before any future fit."
        if ready
        else None,
        annotation_protocol=dict(
            fully_manual=True,
            first_stage_dialogues=200,
            first_stage_annotators=3,
            adjudication="Lead annotator resolves disagreements.",
            second_stage_dialogues=200,
            second_stage_annotators=1,
            occurrence_fleiss_kappa=0.591,
            mean_pairwise_f1=0.738,
            per_item_annotator_or_stage_mapping="not released",
            ambiguity="No per-item uncertainty, disagreement votes, evidence URLs or detailed ambiguity rules are released.",
        ),
        evidence_limitations=[
            "Human labels are annotation-relative judgments, not infallible truth.",
            "Input mismatch includes instruction adherence and must not be pooled as factual falsity.",
            "No item-level independent evidence citations or source-model metadata are released.",
            "Both turns share a dialogue; published labels are exposed development evidence.",
            "Historical labels cannot label newly generated Qwen responses.",
            "No model fitting, ranking, energy benefit or no-cheap-oracle claim is measured.",
        ],
    )
