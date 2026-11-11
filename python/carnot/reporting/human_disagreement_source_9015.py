"""REQ-REPORT-9015-V770: qualify original human disagreement without model inference."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash
from carnot.reporting.human_source_authority_9001 import normalized
from carnot.reporting.natural_label_eligibility_8975 import ref as ref

Json = dict[str, Any]
ROOT = Path(__file__).resolve().parents[3]
NAME = "experiment_9015_v770_human_disagreement_source"
TASK = "exp9015-human-disagreement-source"
CLI = "scripts/experiments/" + NAME + ".py"
REVISION = "3f99df059b001c174b4443e2c41840a5ff1320d7"
SALT = "v770-human-v1"
LANGUAGES = ["ar", "ca", "cs", "de", "en", "es", "eu", "fa", "fi", "fr", "hi", "it", "sv", "zh"]
COUNTS = dict(FIT=160, CAL=40, TEST=200)
MODEL_SPECS: list[Json] = []
PINNED = {
    "policy": "119d89a826bf7ca3802082082168335c5a37d2f46660eca1d1f337c771a0054a",
    "validation": "049650111c8948ddeac579d00396f117d9b441463b4b964544d5e57e9619572a",
    "test": "6d5a2bffc0cdc753ce7f98b48609706f7cb911b846c87a71f75a5cdade7229f7",
    "guidelines": "bea3b862c385ca9b7816300cb5c1ae1fe8bd9c340d3fd0139073018814e5366e",
    "license": "9ba9550ad48438d0836ddab3da480b3b69ffa0aac7b7878b5a0039e7ab429411",
}
OWNED = [
    "python/carnot/reporting/human_disagreement_source_9015.py",
    "python/carnot/reporting/human_disagreement_runner_9015.py",
    CLI,
]
TEST = "tests/python/test_human_disagreement_source_9015.py"


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flush actual counts so stalled source access cannot look like completed work."""
    print(f"[exp9015] phase={phase} completed={completed} pending={pending}", flush=True)


def rank(ident: str) -> str:
    """Choose a single response and role without consulting outcome annotations."""
    return hashlib.sha256((SALT + ":" + ident).encode()).hexdigest()


def labels(row: Json) -> Json:
    """Count people marking any span; character-majority unions lose this disagreement."""
    text, annotations = row["model_output_text"], row["annotations"]
    ids = [a["annotator_id"] for a in annotations]
    if not text or len(ids) < 3 or len(set(ids)) != len(ids) or not all(ids):
        raise ValueError("three_distinct_original_annotators")
    votes, spans = [], []
    for annotation in annotations:
        if annotation.get("automatic", False):
            raise ValueError("automatic_annotation")
        for start, end in annotation["labels"]:
            if type(start) is not int or type(end) is not int or not 0 <= start < end <= len(text):
                raise ValueError("original_unicode_offset")
            spans.append(
                dict(
                    annotator_id=annotation["annotator_id"],
                    start=start,
                    end=end,
                    text=text[start:end],
                )
            )
        votes.append(int(bool(annotation["labels"])))
    numerator, denominator = sum(votes), len(votes)
    return dict(
        unit_id=row["id"],
        y_soft=numerator / denominator,
        y_hard=int(2 * numerator >= denominator),
        numerator=numerator,
        denominator=denominator,
        annotator_count=denominator,
        tie=2 * numerator == denominator,
        original_annotations=annotations,
        original_spans=spans,
        uncertain_annotators=[a["annotator_id"] for a in annotations if a.get("uncertain", False)],
        disagreement=0 < numerator < denominator,
    )


def gate(check: str, expected: Any, observed: Any, reference: Json | None = None) -> Json:
    """Name the exact retained operand so a terminal external block is actionable."""
    return dict(
        check=check,
        upstream="author_Mu_SHROOM",
        path="",
        sha256=None,
        artifact_field=check,
        op="==",
        expected=expected,
        observed=observed,
        passed=observed == expected,
    ) | (reference or {})


def audit(records: list[Json], exposed: set[str], resolutions: dict[str, str], raw: Path) -> Json:
    """Merge source identities and freeze roles before opening any human judgment."""
    raw.mkdir(parents=True, exist_ok=True)
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("duplicate_response_id")
    parents = {r["id"]: r["id"] for r in records}

    def find(key: str) -> str:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    seen: dict[str, str] = {}
    public = {}
    for number, row in enumerate(records):
        qid = resolutions.get(row.get("wikipedia_url", ""), "")
        keys = ["prompt:" + canonical_hash(normalized(row["model_input"]))]
        keys += ["url:" + row["wikipedia_url"]] if row.get("wikipedia_url") else []
        keys += ["qid:" + qid] if re.fullmatch(r"Q[1-9][0-9]*", qid) else []
        keys += ["topic:" + str(row["topic_id"])] if row.get("topic_id") else []
        keys += (
            ["source:" + canonical_hash(normalized(row["source_text"]))]
            if row.get("source_text")
            else []
        )
        public[row["id"]] = dict(
            unit_id=row["id"],
            language=row["lang"].lower(),
            split=row["split"],
            keys=keys,
            resolved=bool(re.fullmatch(r"Q[1-9][0-9]*", qid)),
            prompt_present=bool(row["model_input"]),
        )
        for key in keys:
            if key in seen:
                parents[find(row["id"])] = find(seen[key])
            else:
                seen[key] = row["id"]
        if number % 100 == 0:
            progress("group_public_metadata", number, len(records) - number)
    members: dict[str, list[Json]] = defaultdict(list)
    for ident, row in public.items():
        members[find(ident)].append(row)
    groups, eligible = [], []
    for rows in members.values():
        keys = sorted({k for r in rows for k in r["keys"]})
        splits = {r["split"] for r in rows}
        reason = (
            "prior_exposure"
            if set(keys) & exposed
            else "unresolved_source_identity"
            if not all(r["resolved"] and r["prompt_present"] for r in rows)
            else "unlabeled_or_automatic_split"
            if not splits <= {"validation", "test"}
            else "cross_split"
            if len(splits) != 1
            else None
        )
        group = dict(
            group_id=canonical_hash(sorted(r["unit_id"] for r in rows)),
            source_memberships=rows,
            source_keys=keys,
            exclusion_reason=reason,
        )
        groups.append(group)
        if reason is None:
            winner = min(rows, key=lambda r: rank(r["unit_id"]))
            eligible.append(
                dict(
                    group_id=group["group_id"],
                    unit_id=winner["unit_id"],
                    official_split=winner["split"],
                )
            )
    roles: list[Json] = []
    for split in ["validation", "test"]:
        ranked = sorted(
            (g for g in eligible if g["official_split"] == split), key=lambda g: rank(g["group_id"])
        )
        offset = 0
        for role, count in COUNTS.items():
            if (role == "TEST") == (split == "test"):
                roles.extend(dict(g, role=role) for g in ranked[offset : offset + count])
                offset += count
    atomic_json(raw / "role_manifest.json", dict(salt=SALT, roles=roles, labels_accessible=False))
    progress("roles_frozen_before_labels", len(roles), 400 - len(roles))
    indexed = {r["id"]: r for r in records}
    annotators = []
    failures = []
    for selected_role in roles:
        try:
            annotators.append(
                dict(
                    labels(indexed[selected_role["unit_id"]]),
                    group_id=selected_role["group_id"],
                    role=selected_role["role"],
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            failures.append(dict(unit_id=selected_role["unit_id"], reason=str(error)))
    atomic_json(raw / "labels.json", dict(rows=annotators, failures=failures))
    (raw / "labels.json").chmod(0o600)
    classes = {
        role: dict(Counter(str(r["y_hard"]) for r in annotators if r["role"] == role))
        for role in COUNTS
    }
    ready = (
        len(roles) == 400
        and not failures
        and all(
            classes[r].get(str(y), 0) >= (20 if r == "TEST" else 5) for r in COUNTS for y in [0, 1]
        )
    )
    selected = {r["group_id"] for r in roles}
    rows = [
        dict(
            unit_id=g["group_id"],
            group_id=g["group_id"],
            arm="original_human",
            condition="source_inventory",
            metric="eligible_source_group",
            value=int(g["exclusion_reason"] is None),
            numerator=int(g["exclusion_reason"] is None),
            denominator=1,
            status="completed" if g["group_id"] in selected else "excluded",
            exclusion_reason=g["exclusion_reason"]
            or (None if g["group_id"] in selected else "not_scheduled"),
        )
        for g in groups
    ]
    for role, count in COUNTS.items():
        for slot in range(sum(r["role"] == role for r in roles), count):
            ident = f"missing:{role}:{slot}"
            rows.append(
                dict(
                    unit_id=ident,
                    group_id=ident,
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
    return dict(
        rows=rows,
        intended_count=len(rows),
        independent_count=len(eligible),
        completed_count=len(roles),
        excluded_count=sum(r["status"] == "excluded" for r in rows),
        censored_count=400 - len(roles),
        failed_count=0,
        source_group_rows=groups,
        role_manifest=roles,
        class_counts=classes,
        raw_annotator_rows=annotators,
        annotation_disagreement_rows=[
            r for r in annotators if r["disagreement"] or r["tie"] or r["uncertain_annotators"]
        ],
        annotation_failures=failures,
        unresolved_source_groups=[
            g["group_id"] for g in groups if g["exclusion_reason"] == "unresolved_source_identity"
        ],
        label_sidecar_hashes=[ref(raw / "labels.json")],
        public_view_hashes=[ref(raw / "role_manifest.json")],
        independent_generalization_score=0,
        natural_pilot_ready_score=int(ready),
        independent_label_ready_score=int(ready),
        criterion_policy=dict(
            y_soft="fraction of original annotators marking any nonempty unsupported span",
            y_hard="y_soft>=0.5",
            sensitivity="Exact ties and original uncertain annotations remain visible; no union-of-character-majorities substitution.",
            claim="Agreement with fallible original human judgments; no factual oracle or span leaderboard claim.",
            salt=SALT,
            role_counts=COUNTS,
        ),
        gate_check_summary=[
            gate("full_role_class_label_contract", 1, int(ready)),
            gate("all_fourteen_languages", LANGUAGES, sorted({r["lang"].lower() for r in records})),
        ],
    )


def inspect(manifest: Path, raw: Path) -> Json:
    """Authenticate retained author bytes and terminal retrieval failures before label access."""
    data = json.loads(manifest.read_bytes())
    if data["revision"] != REVISION:
        raise ValueError("release_revision")
    if data["scope"] == "author_release" and any(
        data["files"][key]["sha256"] != "sha256:" + pin for key, pin in PINNED.items()
    ):
        raise ValueError("pinned_release")
    refs = [
        *data["files"].values(),
        data["resolutions"],
        data["exposure_inventory"],
        data["metadata"],
        *data["resolution_receipts"],
        *data.get("custody_references", []),
    ]
    refs += [dict(path=r["path"], sha256=r["sha256"]) for r in data["source_access_receipts"]]
    for reference in refs:
        if ref(Path(reference["path"])) != reference:
            raise ValueError("changed_bytes:" + reference["path"])
    metadata = json.loads(Path(data["metadata"]["path"]).read_bytes())
    policy = Path(data["files"]["policy"]["path"]).read_text()
    authority = (
        metadata["sha"] == REVISION
        and metadata["cardData"]["license"] == "cc-by-4.0"
        and "annotator_id" in policy
        and "Raw annotations" in policy
    )
    failures = [r for r in data["source_access_receipts"] if r["status"] != 200 or r["error"]]
    source_resolution = json.loads(Path(data["resolutions"]["path"]).read_bytes())
    inventory = json.loads(Path(data["exposure_inventory"]["path"]).read_bytes())
    exposure_complete = inventory.get("cross_language_exposure_resolution_complete") is True
    import pyarrow.parquet as pq

    reader: Any = pq.read_table
    records: list[Json] = []
    for split in ["validation", "test"]:
        public = reader(
            data["files"][split]["path"],
            columns=["id", "lang", "model_input", "model_output_text", "wikipedia_url"],
        )
        records.extend(dict(r, split=split) for r in public.to_pylist())
    # Access failure is terminal: do not select partial-language cohorts or inspect their outcomes.
    resolved = (
        {}
        if failures or not exposure_complete
        else {u: r["wikidata_id"] for u, r in source_resolution.items() if r["wikidata_id"]}
    )
    if not failures and exposure_complete:
        annotations = {
            r["id"]: r["annotations"]
            for split in ["validation", "test"]
            for r in reader(data["files"][split]["path"], columns=["id", "annotations"]).to_pylist()
        }
        for row in records:
            row["annotations"] = annotations[row["id"]]
    reduced = audit(records, set(inventory.get("exposed_keys", [])), resolved, raw)
    gates = [
        gate("release_authority", True, authority),
        gate("exposure_source_integrity", True, exposure_complete, data["exposure_inventory"]),
        gate(
            "network_budget",
            True,
            data["network_bytes"] <= 268435456
            and data.get("network_total_elapsed_upper_bound_s", data["network_duration_s"]) <= 600,
        ),
    ]
    if failures:
        rejected = failures[0]
        operand = gate("source_resolution_access", 200, rejected["status"], ref(manifest))
        operand.update(
            artifact_field=f"source_access_receipts[{data['source_access_receipts'].index(rejected)}].status",
            request_url=rejected["url"],
            access_error=rejected["error"],
            rejected_payload_reference=dict(path=rejected["path"], sha256=rejected["sha256"]),
        )
        gates.insert(0, operand)
    reduced["gate_check_summary"] = gates + reduced["gate_check_summary"]
    for operand in reduced["gate_check_summary"]:
        if not operand["path"]:
            operand.update(ref(manifest))
    qualified = (
        authority
        and data["scope"] == "author_release"
        and all(g["passed"] for g in reduced["gate_check_summary"])
    )
    if not qualified:
        reduced.update(natural_pilot_ready_score=0, independent_label_ready_score=0)
    reduced.update(
        input_scope=data["scope"],
        release_revision=REVISION,
        license_receipt=data["files"]["license"],
        annotation_policy_hash=data["files"]["guidelines"],
        source_artifact_hashes=[ref(manifest), *refs],
        exposure_inventory=data["exposure_inventory"],
        source_resolution_receipts=data["resolution_receipts"],
        source_access_receipts=data["source_access_receipts"],
        labels_deferred_due_to_external_block=bool(failures or not exposure_complete),
    )
    return reduced
