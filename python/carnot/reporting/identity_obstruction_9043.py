"""REQ-REPORT-9043-V772: explain retained identity failures without changing eligibility."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re
from typing import Any
from urllib.parse import parse_qs, quote, urlsplit

import yaml

from carnot.reporting import source_identity_9029 as old
from carnot.reporting.current_work_receipt import canonical_hash
from carnot.reporting.roadmap_contract import parse_design

Json = dict[str, Any]
ROOT = old.ROOT
NAME = "experiment_9043_v772_identity_obstruction_audit"
CLI = "scripts/experiments/" + NAME + ".py"
TEST = "tests/python/test_identity_obstruction_9043.py"
OWNED = [
    "python/carnot/reporting/identity_obstruction_9043.py",
    "python/carnot/reporting/identity_obstruction_runner_9043.py",
    CLI,
]
UPSTREAM = ROOT / "results/experiment_9029_v771_source_identity_resolution.json"
UPSTREAM_PIN = "sha256:a9ca396d7bc115c221a948bb835a3da3e6221feba16dbdf65bf0779fc19a2120"
METADATA_PIN = "sha256:f6f6aeea7373b800af4208aceceda1f895d32053679e71f18ed3602c80778e6e"
DESIGN = "openspec/change-proposals/research-roadmap-v771-preserved-20261112.md"
QUEUE = "openspec/change-proposals/research-roadmap-v771-queue-preserved-20261112.yaml"
AUTHORITY_PINS = {
    DESIGN: "sha256:645327851ea953d7cfd0961838dcc36eb7e10dd8dbabd32a81c52adadb76d3fd",
    QUEUE: "sha256:2f18f190413146ad123228723a2e9d0780981a5c69a471473b5d06b757041d42",
}
FULL_TASK_DIGEST = "sha256:a353be3a3351cc47a11d5c93a3a36a057653d9851e17da6896e33a7737e9ce1c"
MODEL_SPECS: list[Json] = []
CATEGORIES = [
    "malformed_or_missing_receipt",
    "alias_or_parser_omission",
    "missing_page",
    "existing_page_without_qid",
    "unresolved_other",
]
CALLERS = [
    "source_identity_9029.load_inputs",
    "source_identity_9029.acquire",
    "source_identity_runner_9029.replay",
    "tests/python/test_source_identity_9029.py",
]
EVIDENCE = {
    "malformed_or_missing_receipt": "Recover the original request and complete response with independently pinned hashes; author release revision must match.",
    "alias_or_parser_omission": "Bind the retained valid QID through a tested alias chain, then recompute every release and historical join under the original gate.",
    "missing_page": "Obtain author-confirmed original URL, revision or move/deletion history and an authenticated cross-language QID or complete equivalent source mapping for every historical exposure.",
    "existing_page_without_qid": "Obtain author-confirmed source revision and authenticated Wikidata sitelinks or an equivalent complete cross-language source mapping; a local pageid is insufficient.",
    "unresolved_other": "Resolve the recorded API error, invalid marker, ambiguous alias or invalid QID using author-confirmed source metadata and authenticated cross-language mappings.",
}


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Counts show actual offline work; metadata content must stay out of logs."""
    print(f"[exp9043] phase={phase} completed={completed} pending={pending}", flush=True)


class Blocked(ValueError):
    """Carry the failed operand so absence cannot be mistaken for a measured zero."""

    def __init__(self, operand: Json):
        self.operand = operand
        super().__init__(operand["check"])


def require(path: Path, field: str, expected: Any, observed: Any) -> Json:
    """Name exact custody failures before any response classification starts."""
    operand = old.gate(
        field, expected, observed, dict(old.ref(path), upstream="Exp9029", artifact_field=field)
    )
    if not operand["passed"]:
        raise Blocked(operand)
    return operand


def pinned(path: Path, digest: str) -> Json:
    """An independent digest prevents rehashed metadata from replacing authority."""
    reference = old.ref(path)
    require(path, "input_sha256", digest, reference["sha256"] or "missing_file")
    return reference


def load(path: Path) -> Json:
    """Open only retained public metadata; author parquet annotations stay unopened."""
    progress("authenticate_before")
    refs = [pinned(path, UPSTREAM_PIN)]
    primary = json.loads(path.read_bytes())
    metadata_ref = primary["metadata_manifest"]
    refs.append(pinned(Path(metadata_ref["path"]), METADATA_PIN))
    require(
        path, "metadata_manifest.sha256", METADATA_PIN, metadata_ref.get("sha256", "missing_field")
    )
    metadata = json.loads(Path(metadata_ref["path"]).read_bytes())
    checks = [
        require(path, field, expected, primary.get(field, "missing_field"))
        for field, expected in [
            ("experiment_id", 9029),
            ("source_identity_ready_score", 0),
            ("unresolved_request_count", 215),
            ("required_checks_passed", True),
        ]
    ]
    for name, digest in AUTHORITY_PINS.items():
        refs.append(pinned(ROOT / name, digest))
    tasks = parse_design((ROOT / DESIGN).read_text(), milestone="2026.11.771")[1]
    checks.append(
        require(ROOT / DESIGN, "full_task_digest", FULL_TASK_DIGEST, canonical_hash(tasks))
    )
    checks.append(
        require(
            ROOT / QUEUE,
            "full_task_digest",
            FULL_TASK_DIGEST,
            canonical_hash(yaml.safe_load((ROOT / QUEUE).read_bytes())["tasks"]),
        )
    )
    refs += metadata["source_artifact_hashes"] + primary["repository_health"]["references"]
    refs += [dict(path=r["path"], sha256=r["sha256"]) for r in metadata["request_rows"]]
    for index, reference in enumerate(refs):
        pinned(Path(reference["path"]), reference["sha256"])
        if index % 50 == 0:
            progress("authenticate", index + 1, len(refs) - index - 1)
    progress("authenticate_after", len(refs), 0)
    return dict(primary=primary, metadata=metadata, references=refs, gate_check_summary=checks)


def observe(receipt: Json, key: str) -> Json:
    """Distinguish transport, page existence and QID loss using original response bytes."""
    row: Json = dict(
        category="unresolved_other",
        normalized_title=None,
        redirected_title=None,
        namespace=None,
        pageid_present=False,
        pageid=None,
        missing_marker=False,
        invalid_marker=False,
        markers={},
        wikibase_item_present=False,
        wikibase_item=None,
        parser_result={},
        alias_chain=[],
        response_reference={k: receipt.get(k) for k in ["path", "sha256"]},
        request_sha256=canonical_hash(receipt["url"]),
        status=receipt["status"],
    )
    try:
        parsed = old.parse_receipt(receipt)
        url = urlsplit(receipt["url"])
        titles = parse_qs(url.query)["titles"][0].split("|")
        original = next(
            t
            for t in titles
            if old.identity("https://" + str(url.hostname) + "/wiki/" + quote(t, safe="")) == key
        )
        row["parser_result"] = parsed.get(key, {})
        if receipt["status"] != 200 or receipt.get("error"):
            return row
        data = json.loads(Path(receipt["path"]).read_bytes())
        if "error" in data:
            return row
        public = data["query"]
        aliases = {
            a["from"]: a["to"]
            for kind in ["normalized", "converted", "redirects"]
            for a in public.get(kind, [])
        }
        normalized = {a["from"]: a["to"] for a in public.get("normalized", [])}
        row["normalized_title"] = normalized.get(original, original)
        title, seen = original, set()
        while title in aliases and title not in seen:
            seen.add(title)
            row["alias_chain"].append(dict(from_title=title, to_title=aliases[title]))
            title = aliases[title]
        row["redirected_title"] = title
        pages = {p["title"]: p for p in public["pages"].values()}
        page = pages.get(title, {})
        props = page.get("pageprops", {})
        qid = props.get("wikibase_item")
        row.update(
            namespace=page.get("ns"),
            pageid_present="pageid" in page,
            pageid=page.get("pageid"),
            missing_marker="missing" in page,
            invalid_marker="invalid" in page,
            markers={k: page[k] for k in ["missing", "invalid", "invalidreason"] if k in page},
            wikibase_item_present="wikibase_item" in props,
            wikibase_item=qid,
        )
        valid = isinstance(qid, str) and re.fullmatch(r"Q[1-9][0-9]*", qid)
        if valid and parsed.get(key, {}).get("wikidata_id") != qid:
            row["category"] = "alias_or_parser_omission"
        elif "missing" in page:
            row["category"] = "missing_page"
        elif (
            type(page.get("pageid")) is int
            and page["pageid"] > 0
            and "invalid" not in page
            and "wikibase_item" not in props
        ):
            row["category"] = "existing_page_without_qid"
    except (OSError, ValueError, KeyError, TypeError, AttributeError, StopIteration):
        row["category"] = "malformed_or_missing_receipt"
    return row


def diagnose(metadata: Json) -> Json:
    """Reconstruct the original denominator before explaining each unresolved identity."""
    progress("reconstruct_before")
    resolutions: Json = {}
    receipts: dict[str, list[Json]] = {}
    for rec in metadata["request_rows"]:
        resolutions.update(old.parse_receipt(rec))
        request = urlsplit(rec["url"])
        for title in parse_qs(request.query)["titles"][0].split("|"):
            key = old.identity(
                "https://" + str(request.hostname) + "/wiki/" + quote(title, safe="")
            )
            receipts.setdefault(key, []).append(rec)
    historical = {old.identity(url) for url in metadata["inventory"]["historical_reference_urls"]}
    required = historical | {old.identity(row["wikipedia_url"]) for row in metadata["records"]}
    unresolved = sorted(k for k in required if not resolutions.get(k, {}).get("wikidata_id"))
    if unresolved != metadata["unresolved_requests"] or len(unresolved) != 215:
        raise ValueError("unresolved_denominator_drift")
    reduced = old.reduce_metadata(metadata["records"], metadata["inventory"], resolutions)
    for field in ["source_group_rows", "unresolved_source_groups", "source_identity_ready_score"]:
        if reduced[field] != metadata[field]:
            raise ValueError("strict_closure_drift:" + field)
    groups = [g for g in reduced["source_group_rows"] if not g["identity_resolved"]]
    membership = {
        k[4:]: g["group_id"] for g in groups for k in g["source_keys"] if k.startswith("url:")
    }
    obstructions, rows = [], []
    for index, key in enumerate(unresolved):
        observations = [observe(rec, key) for rec in receipts.get(key, [])]
        selected = next(
            (o for o in reversed(observations) if o["status"] == 200),
            observations[-1] if observations else dict(category="malformed_or_missing_receipt"),
        )
        category = selected["category"]
        obstructions.append(
            dict(
                selected,
                unit_id=key,
                group_id=membership.get(key, key),
                historical=key in historical,
                release=key in membership,
                category=category,
                observations=observations,
                additional_evidence=EVIDENCE[category],
                identical_retained_replay_can_change=False,
                identical_live_retry="Only changed remote metadata or transport can change a new response; identical retained bytes cannot change this diagnosis.",
            )
        )
        rows.append(
            dict(
                unit_id=key,
                group_id=membership.get(key, key),
                arm="retained_metadata",
                condition=category,
                metric="identity_obstruction_classified",
                value=1,
                numerator=1,
                denominator=1,
                status="completed",
                exclusion_reason=None,
            )
        )
        if index % 50 == 0:
            progress("classify", index + 1, len(unresolved) - index - 1)
    counts = dict.fromkeys(CATEGORIES, 0) | dict(Counter(o["category"] for o in obstructions))
    omissions = [o["unit_id"] for o in obstructions if o["category"] == "alias_or_parser_omission"]
    progress("reconstruct_after", len(rows), 0)
    return dict(
        obstruction_rows=obstructions,
        rows=rows,
        category_counts=counts,
        request_rows=metadata["request_rows"],
        release_group_rows=groups,
        gate_check_summary=reduced["gate_check_summary"],
        intended_count=len(rows),
        completed_count=len(rows),
        independent_count=len(rows),
        excluded_count=0,
        censored_count=0,
        failed_count=0,
        independent_unit_definition="Unique requested host/title; not independent scientific outcomes.",
        identity_diagnosis_ready_score=int(sum(counts.values()) == 215),
        source_identity_ready_score=reduced["source_identity_ready_score"],
        parser_regression_evidence=dict(
            proven_omissions=omissions,
            callers=CALLERS,
            proposed_change="No production change is supported by retained bytes."
            if not omissions
            else "Add only the demonstrated alias link to the original parser and test retained bytes before recomputing all joins.",
            private_control="test_alias_omission_is_only_a_private_control; converted-title QID, not a measured omission",
        ),
        reopening_evidence=dict(
            by_category=EVIDENCE,
            criterion="Every release and historical identity has an authenticated cross-language mapping; original release_identity_closure=0, historical_identity_closure=0, unknown_exposure=[], and all fourteen languages pass unchanged.",
            identical_request_sufficient=False,
            downstream_dependency_scheduled=False,
        ),
    )
