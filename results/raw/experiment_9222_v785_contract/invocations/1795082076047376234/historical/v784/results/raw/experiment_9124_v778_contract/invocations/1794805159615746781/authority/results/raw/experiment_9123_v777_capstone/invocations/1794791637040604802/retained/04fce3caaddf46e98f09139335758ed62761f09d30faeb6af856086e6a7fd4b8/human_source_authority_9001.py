"""REQ-REPORT-9001: qualify original human decisions without recycling exposed groups."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.natural_criterion_audit_8908 import normalize_spans
from carnot.reporting.natural_label_eligibility_8975 import ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9001_v769_human_source_authority"
TASK = "exp9001-human-source-authority"
CLI = "scripts/experiments/" + NAME + ".py"
REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
SALT = "v769-human-v1"
MANUAL_POLICY = "manual annotations at both the individual cases and word levels"
MODEL_SPECS: list[Json] = []
COUNTS = dict(fit=320, calibration=80, test=200)
PINNED = dict(
    sources="0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b",
    responses="e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073",
    policy="3686178d505a38cb20a94827917369bcb9d0aa44cf5154d7ddbac740ee06628d",
    license="b7fd7d6bdfe0cbba63c63a310914beb4a4acb8bf08da73849219f45385f5b244",
)
OWNED = [
    "python/carnot/reporting/human_source_authority_9001.py",
    "python/carnot/reporting/human_source_runner_9001.py",
    CLI,
]
CHECKS = {
    "owned_pytest",
    "owned_coverage",
    "ruff",
    "format",
    "mypy",
    "spec",
    "e2e018_021",
    "full_pytest",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush real counts so release processing remains observable."""
    print(f"[exp9001] phase={phase} completed={completed} pending={pending}", flush=True)


def normalized(text: str) -> str:
    """Ignore Unicode presentation and whitespace when identifying duplicate documents."""
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def rank(ident: str) -> str:
    """Choose responses and groups without consulting their annotations."""
    return hashlib.sha256((SALT + ":" + ident).encode()).hexdigest()


def grouped(sources: list[Json], responses: list[Json]) -> tuple[list[Json], list[Json]]:
    """Merge shared document passages transitively before official splits or labels."""
    indexed = {s["source_id"]: s for s in sources}
    if len(indexed) != len(sources) or len({r["id"] for r in responses}) != len(responses):
        raise ValueError("duplicate_identity")
    selected = {k: s for k, s in indexed.items() if s["task_type"] in {"QA", "Summary"}}
    parents = {k: k for k in selected}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen: dict[str, str] = {}
    for number, (sid, source) in enumerate(sorted(selected.items())):
        content = source["source_info"]
        text = content if source["task_type"] == "Summary" else content["passages"]
        if not text or not source["prompt"]:
            raise ValueError("missing_context")
        docs = re.split(r"(?:^|\n\s*\n)passage \d+:", text)
        keys = [canonical_hash(normalized(text))]
        for doc in docs:
            words = normalized(doc).split()
            keys.extend(canonical_hash(words[i : i + 32]) for i in range(max(0, len(words) - 31)))
            if words:
                keys.append(canonical_hash(words))
        for key in keys:
            if key in seen:
                parents[find(sid)] = find(seen[key])
            else:
                seen[key] = sid
        if number % 100 == 0:
            progress("group_documents", number, len(selected) - number)
    members: dict[str, list[Json]] = defaultdict(list)
    excluded = []
    for response in responses:
        if response["source_id"] not in indexed or response["split"] not in {"train", "test"}:
            raise ValueError("response_join_or_split")
        if response["source_id"] not in selected:
            continue
        if response["quality"] != "good":
            excluded.append(
                dict(unit_id=response["id"], quality=response["quality"], reason="quality_failed")
            )
            continue
        if not response["response"]:
            raise ValueError("empty_response")
        members[find(response["source_id"])].append(response)
    components: dict[str, list[str]] = defaultdict(list)
    for sid in selected:
        components[find(sid)].append(sid)
    groups = []
    for root, ids in sorted(components.items()):
        candidates = members[root]
        groups.append(
            dict(
                group_id=canonical_hash(sorted(ids)),
                source_ids=sorted(ids),
                source_content_hashes=[
                    canonical_hash(
                        normalized(
                            selected[s]["source_info"]
                            if selected[s]["task_type"] == "Summary"
                            else selected[s]["source_info"]["passages"]
                        )
                    )
                    for s in sorted(ids)
                ],
                candidates=candidates,
                splits=sorted({r["split"] for r in candidates}),
            )
        )
    return groups, excluded


def audit(
    sources: list[Json], responses: list[Json], exposed: set[str], manual: bool, raw: Path
) -> Json:
    """Freeze public roles first; original spans remain evaluator-only evidence."""
    progress("group_and_select")
    raw.mkdir(parents=True, exist_ok=True)
    groups, quality = grouped(sources, responses)
    eligible, group_rows, roles = [], [], []
    by_source = {s["source_id"]: s for s in sources}
    by_response = {r["id"]: r for r in responses}
    for group in groups:
        prior = sorted(set(group["source_ids"]) & exposed)
        reason = (
            "prior_exposure"
            if prior
            else "no_good_response"
            if not group["candidates"]
            else "cross_split"
            if len(group["splits"]) != 1
            else None
        )
        public = {k: v for k, v in group.items() if k != "candidates"}
        group_rows.append(dict(public, exposed_source_ids=prior, exclusion_reason=reason))
        if reason is None:
            winner = min(group["candidates"], key=lambda r: rank(r["id"]))
            eligible.append(
                dict(
                    public,
                    unit_id=winner["id"],
                    source_id=winner["source_id"],
                    official_split=winner["split"],
                )
            )
    for split in ("train", "test"):
        ranked = sorted(
            (g for g in eligible if g["official_split"] == split), key=lambda g: rank(g["group_id"])
        )
        offset = 0
        for role, count in COUNTS.items():
            if (role == "test") == (split == "test"):
                for group in ranked[offset : offset + count]:
                    response, source = by_response[group["unit_id"]], by_source[group["source_id"]]
                    roles.append(
                        dict(
                            group,
                            role=role,
                            source_sha256=canonical_hash(source),
                            response_sha256=canonical_hash(response["response"]),
                        )
                    )
                offset += count
    atomic_json(raw / "role_manifest.json", dict(salt=SALT, roles=roles, labels_accessible=False))
    public_views = [
        dict(
            unit_id=r["unit_id"],
            group_id=r["group_id"],
            role=r["role"],
            source=by_source[r["source_id"]],
            response=by_response[r["unit_id"]]["response"],
        )
        for r in roles
    ]
    atomic_json(raw / "inference_views.json", dict(rows=public_views))
    progress("roles_sealed_before_labels", len(roles), 600 - len(roles))
    annotations = []
    natural_ids = {sid for g in groups for sid in g["source_ids"]}
    for index, response in enumerate(responses):
        if response["source_id"] not in natural_ids:
            continue
        normalized_spans = normalize_spans(response["labels"], response["response"], revised=False)
        annotations.append(
            dict(
                unit_id=response["id"],
                source_id=response["source_id"],
                annotation_sha256=canonical_hash(response["labels"]),
                response_record_sha256=canonical_hash(response),
                original_spans=response["labels"],
                quality=response["quality"],
                primary_error=int(bool(normalized_spans)),
                excluding_implicit_true=int(
                    any(not s.get("implicit_true", False) for s in response["labels"])
                ),
                implicit_true_presence=int(
                    any(s.get("implicit_true", False) for s in response["labels"])
                ),
                categories=sorted({s["label_type"] for s in response["labels"]}),
            )
        )
        if index % 1000 == 0:
            progress("original_span_audit", index, len(responses) - index)
    sidecar = raw / "original_labels.json"
    atomic_json(sidecar, dict(rows=annotations))
    sidecar.chmod(0o600)
    label_index = {r["unit_id"]: r for r in annotations}
    classes = {
        role: dict(
            Counter(
                str(label_index[r["unit_id"]]["primary_error"]) for r in roles if r["role"] == role
            )
        )
        for role in COUNTS
    }
    pilot = len(roles) == 600 and all(
        classes[role].get(str(label), 0) >= (20 if role == "test" else 5)
        for role in COUNTS
        for label in (0, 1)
    )
    selected_ids = {r["group_id"] for r in roles}
    rows = [
        dict(
            unit_id=g["group_id"],
            group_id=g["group_id"],
            arm="original_human",
            condition="unseen_source_pool",
            metric="eligible_source_group",
            value=int(g["exclusion_reason"] is None),
            numerator=int(g["exclusion_reason"] is None),
            denominator=1,
            status="completed" if g["group_id"] in selected_ids else "excluded",
            exclusion_reason=g["exclusion_reason"]
            or (None if g["group_id"] in selected_ids else "not_scheduled"),
        )
        for g in group_rows
    ]
    for role, expected in COUNTS.items():
        for slot in range(sum(r["role"] == role for r in roles), expected):
            rows.append(
                dict(
                    unit_id=f"missing:{role}:{slot}",
                    group_id=f"missing:{role}:{slot}",
                    arm="original_human",
                    condition=role,
                    metric="eligible_source_group",
                    value=None,
                    numerator=0,
                    denominator=1,
                    status="censored",
                    exclusion_reason="unseen_source_shortage",
                )
            )
    gates = [
        dict(
            check="unseen_source_groups",
            upstream="original_RAGTruth",
            path="",
            sha256=None,
            artifact_field="unseen_source_group_count",
            op=">=",
            expected=600,
            observed=len(eligible),
            passed=len(eligible) >= 600,
        ),
        dict(
            check="full_role_class_contract",
            upstream="original_RAGTruth",
            path="",
            sha256=None,
            artifact_field="natural_pilot_ready_score",
            op="==",
            expected=1,
            observed=int(manual and pilot),
            passed=manual and pilot,
        ),
    ]
    gates.extend(
        dict(
            check=field,
            upstream="original_RAGTruth",
            path="",
            sha256=None,
            artifact_field=field,
            op=op,
            expected=expected,
            observed=observed,
            passed=passed,
        )
        for field, op, expected, observed, passed in [
            ("documented_manual", "==", True, manual, manual),
            (
                "human_authority_group_count",
                ">=",
                200,
                len(groups) if manual else 0,
                manual and len(groups) >= 200,
            ),
        ]
    )
    return dict(
        rows=rows,
        intended_count=len(rows),
        independent_count=len(eligible),
        completed_count=len(roles),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        censored_count=600 - len(roles),
        failed_count=0,
        pilot_intended_count=600,
        human_authority_group_count=len(groups) if manual else 0,
        source_group_rows=group_rows,
        role_manifest=roles,
        class_counts=classes,
        annotation_origin_rows=[
            dict(
                origin="original_manual_span_decisions" if manual else "unsupported_authority",
                author_policy_phrase=MANUAL_POLICY,
                documented_manual=manual,
                annotator_independence="not_recorded",
                span_count=sum(len(a["original_spans"]) for a in annotations),
                response_count=len(annotations),
                response_error_count=sum(a["primary_error"] for a in annotations),
                implicit_true_response_count=sum(a["implicit_true_presence"] for a in annotations),
                excluding_implicit_true_error_count=sum(
                    a["excluding_implicit_true"] for a in annotations
                ),
                category_response_counts=dict(
                    Counter(c for a in annotations for c in a["categories"])
                ),
            )
        ],
        label_sidecar_hashes=[ref(sidecar)],
        public_view_hashes=[ref(raw / "role_manifest.json"), ref(raw / "inference_views.json")],
        independent_label_ready_score=int(manual and len(groups) >= 200),
        natural_pilot_ready_score=int(manual and pilot),
        independent_generalization_score=0,
        gate_check_summary=gates,
        quality_exclusions=quality,
        criterion_policy=dict(
            primary="any original human unsupported span, including implicit_true",
            sensitivity="retain original categories, implicit_true, due_to_null, comments and quality; exclude implicit_true only in sensitivity",
            selection_salt=SALT,
            role_counts=COUNTS,
            overlap="NFKC/casefold/whitespace; exact documents or shared contiguous32-word window; transitive union",
            ambiguity="Human source support is fallible. implicit_true can be factually true but absent from context. No individual annotator IDs or independent adjudication records.",
        ),
    )


def inspect(manifest: Path, raw: Path) -> Json:
    """Reject changed release or exposure bytes before any label can grant eligibility."""
    if not manifest.is_file():
        raise FileNotFoundError("missing_file:" + str(manifest))
    data = json.loads(manifest.read_bytes())
    if data["revision"] != REVISION:
        raise ValueError("release_revision")
    references = (
        list(data["files"].values())
        + data["exposure_witnesses"]
        + data.get("custody_references", [])
    )
    for reference in references:
        if ref(Path(reference["path"])) != reference:
            raise ValueError("changed_bytes:" + reference["path"])
    if data["scope"] == "author_release" and any(
        data["files"][k]["sha256"] != "sha256:" + v for k, v in PINNED.items()
    ):
        raise ValueError("pinned_release")
    files = {k: Path(v["path"]) for k, v in data["files"].items()}
    manual = (
        MANUAL_POLICY in files["policy"].read_text()
        and "MIT License" in files["license"].read_text()
    )
    exposed: set[str] = set()
    ledger = []
    for reference in data["exposure_witnesses"]:
        ids = {
            str(json.loads(line)["source_id"])
            for line in Path(reference["path"]).read_text().splitlines()
        }
        exposed.update(ids)
        ledger.append(
            dict(reference, source_ids=sorted(ids), scope="historical_development_exposure")
        )
    sources, responses = (
        [json.loads(line) for line in files[k].read_text().splitlines()]
        for k in ("sources", "responses")
    )
    value = audit(sources, responses, exposed, manual, raw)
    if data["scope"] != "author_release":
        value.update(independent_label_ready_score=0, natural_pilot_ready_score=0)
    for gate in value["gate_check_summary"]:
        gate.update(ref(manifest))
    value.update(
        input_scope=data["scope"],
        exposure_ledger=ledger,
        source_artifact_hashes=[ref(manifest), *references],
        cited_upstream_artifacts=data.get("cited_upstream_artifacts", []),
        source_access_receipts=data["source_access_receipts"],
        exposure_inventory=data.get("exposure_inventory"),
    )
    return value
