"""REQ-VERIFY-8084: seal public roles before complete-response annotation access.

Recorded source declarations are conservative exclusions. They cannot reveal
unknown model pretraining or unrecorded external access to this public dataset.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import development_cohort_7994 as reader
from carnot.verify import response_targets_7955 as targets
from carnot.verify.evidence_features_7980 import normalized as normalized

Json = dict[str, Any]
ROLES = dict(fit=128, tune=64, evaluation=192, stream=256, retention=128)
FIELDS = re.compile(
    rb'"(source_id|source_ids|source_bytes|source_hash|source_hashes|source_cluster_id|normalized_source_hash)"\s*:\s*("(?:\\.|[^"\\])*"|\[[^\]]*\])'
)


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Report finished units, because silent scans cannot prove liveness."""
    print(f"[exp8084] {phase} completed={completed} pending={pending}", flush=True)


def text_bytes(value: Any) -> bytes:
    """Use the official public serialization so targets can authenticate bytes."""
    return (
        value
        if isinstance(value, str)
        else json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    ).encode()


def normalize(value: bytes) -> bytes:
    """Formatting differences must not make an old source appear fresh."""
    return (
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value.decode()).casefold())
        .strip()
        .encode()
    )


def shingles(value: bytes) -> set[tuple[str, ...]]:
    """Compare five-token neighborhoods, retaining short complete sources too."""
    words = normalize(value).decode().split()
    return {tuple(words[i : i + 5]) for i in range(max(1, len(words) - 4))}


def immutable(path: Path, value: Json) -> Json:
    """Refuse a changed seal instead of silently replacing primitive evidence."""
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError("immutable_drift")
    atomic_json(path, value)
    path.chmod(0o444)
    return dict(path=str(path.absolute()), sha256=sha256_file(path))


def operand(path: Path, field: str, expected: Any, observed: Any, op: str = "==") -> Json:
    """Carry the actual failed operand so a block is not mistaken for zero benefit."""
    passed = observed == expected if op == "==" else observed >= expected
    return dict(
        check=field,
        upstream="recorded_history",
        path=str(path),
        hash=sha256_file(path) if path.is_file() else None,
        field=field,
        op=op,
        expected=expected,
        observed=observed,
        passed=passed,
    )


def inventory(root: Path, sources: list[Json], raw: Path, sealed: Path | None = None) -> Json:
    """Stream prior results and checkpoint inputs without decoding human labels.

    Each file binds exact bytes and its individual declarations. Overlapping
    chunks preserve long public fields; malformed or unavailable records remain
    unknown. Opaque binary checkpoints are explicit coverage limitations.
    """
    if sealed is not None:
        previous: Json = json.loads(sealed.read_text())
        for index, ref in enumerate(previous["files"]):
            path = Path(ref["path"])
            observed = sha256_file(path) if path.is_file() else None
            if observed != ref["sha256"]:
                previous["unknown_history"].append(
                    dict(
                        path=str(path),
                        reason="history_input_hash",
                        expected=ref["sha256"],
                        observed=observed,
                        blocking=True,
                    )
                )
            if index % 256 == 0:
                progress(
                    "authenticate_completed_inventory",
                    index + 1,
                    len(previous["files"]) - index - 1,
                )
        previous["inventory_reuse"] = dict(
            path=str(sealed.absolute()),
            sha256=sha256_file(sealed),
            scope="same task completed inventory; declarations retain original exposure, not fresh data",
        )
        return previous
    catalog = {s["source_id"]: text_bytes(s["source_info"]) for s in sources}
    hashes = {
        h: sid
        for sid, blob in catalog.items()
        for h in (normalized(blob), hashlib.sha256(blob).hexdigest())
    }
    files: list[Json] = []
    declarations: list[Json] = []
    unknown: list[Json] = []
    contents: dict[str, str] = {}
    roots = [root / p for p in ("results", "checkpoints", ".checkpoints")]
    paths = sorted(
        {
            p
            for base in roots
            for p in base.rglob("*")
            if p.is_file()
            and not p.resolve().is_relative_to(raw.resolve())
            and "experiment_8084_v700_fresh_cohort_methods" not in str(p)
        }
    )
    for index, path in enumerate(paths):
        if path.suffix not in {".json", ".jsonl", ".bin"} and ".part" not in path.name:
            if "checkpoint" in str(path) and path.suffix not in {".log", ".lock"}:
                unknown.append(dict(path=str(path), reason="opaque_checkpoint", blocking=False))
            continue
        digest, ids, tail = hashlib.sha256(), set(), b""
        try:
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                    block = tail + chunk
                    for key, token in FIELDS.findall(block):
                        value = json.loads(token)
                        if key == b"source_bytes":
                            try:
                                blob = bytes.fromhex(value)
                                norm = normalized(blob)
                                contents[norm] = normalize(blob).hex()
                                if norm in hashes:
                                    ids.add(hashes[norm])
                            except (ValueError, UnicodeError):
                                unknown.append(
                                    dict(
                                        path=str(path),
                                        reason="malformed_source_bytes",
                                        blocking=True,
                                    )
                                )
                        elif key in {b"source_id", b"source_ids"}:
                            ids.update(
                                v
                                for v in (value if isinstance(value, list) else [value])
                                if v in catalog
                            )
                        else:
                            ids.update(
                                hashes[v.removeprefix("sha256:")]
                                for v in (value if isinstance(value, list) else [value])
                                if v.removeprefix("sha256:") in hashes
                            )
                    tail = block[-262144:]
            files.append(
                dict(
                    path=str(path), sha256="sha256:" + digest.hexdigest(), bytes=path.stat().st_size
                )
            )
            if ids:
                declarations.append(
                    dict(
                        path=str(path),
                        sha256=files[-1]["sha256"],
                        source_ids=sorted(ids),
                        exposure="prior_declared_source_access",
                    )
                )
        except (OSError, ValueError, TypeError, AttributeError) as error:
            unknown.append(dict(path=str(path), reason=str(error), blocking=True))
        if index % 256 == 0:
            progress("history_scan", index + 1, len(paths) - index - 1)
    known_ids: list[str] = sorted({sid for row in declarations for sid in row["source_ids"]})
    for sid in known_ids:
        contents[normalized(catalog[sid])] = normalize(catalog[sid]).hex()
    progress("inventory_complete", len(files), 0)
    return dict(
        files=files,
        declarations=declarations,
        known_source_ids=known_ids,
        normalized_contents=contents,
        unknown_history=unknown,
        unknown_coverage=[
            "unrecorded external history",
            "model pretraining",
            "undeclared source access",
        ],
        whole_jsonl_access_is_not_unseen_label_proof=True,
    )


def select(
    sources: list[Json], responses: list[Json], inv: Json
) -> tuple[list[Json], list[Json], list[Json], list[Json]]:
    """Select stable public identities; outcomes cannot change ordering or allocation."""
    catalog = {s["source_id"]: text_bytes(s["source_info"]) for s in sources}
    groups: dict[str, list[Json]] = {}
    for response in responses:
        if response["split"] == "train" and response["source_id"] in catalog:
            blob = catalog[response["source_id"]]
            groups.setdefault(normalized(blob), []).append(dict(response, blob=blob))
    blobs = {key: rows[0]["blob"] for key, rows in groups.items()}
    blobs.update(
        {key: bytes.fromhex(value) for key, value in inv.get("normalized_contents", {}).items()}
    )
    parents = {key: key for key in blobs}
    tokens = {key: shingles(blob) for key, blob in blobs.items()}
    inverted: dict[tuple[str, ...], set[str]] = {}

    def find(key: str) -> str:
        while parents[key] != key:
            key = parents[key]
        return key

    for index, key in enumerate(sorted(blobs)):
        neighbors = {other for token in tokens[key] for other in inverted.get(token, set())}
        for other in neighbors:
            left, right = tokens[key], tokens[other]
            if len(left & right) / len(left | right) >= 0.8:
                parents[find(key)] = find(other)
        for token in tokens[key]:
            inverted.setdefault(token, set()).add(key)
        if index % 256 == 0:
            progress("near_duplicate_scan", index + 1, len(blobs) - index - 1)
    components: dict[str, list[str]] = {}
    for key in blobs:
        components.setdefault(find(key), []).append(key)
    clusters = [
        dict(cluster_id=canonical_hash(sorted(keys)), normalized_hashes=sorted(keys))
        for keys in components.values()
        if len(keys) > 1
    ]
    known = set(inv["known_source_ids"])
    exposed = set(inv.get("normalized_contents", {})) | {
        key for key, rows in groups.items() if any(r["source_id"] in known for r in rows)
    }
    blocked_components = {find(key) for key in exposed}
    excluded, candidates, seen = [], [], set()
    order = sorted(
        groups, key=lambda key: hashlib.sha256(b"V700" + normalize(blobs[key])).hexdigest()
    )
    for key in order:
        rows = groups[key]
        reason = (
            "known_prior_exposure"
            if find(key) in blocked_components
            else "near_duplicate_group"
            if find(key) in seen
            else None
        )
        if reason:
            excluded.append(
                dict(
                    source_ids=sorted({r["source_id"] for r in rows}),
                    source_hash=key,
                    exclusion_reason=reason,
                )
            )
            continue
        seen.add(find(key))
        chosen = min(
            rows,
            key=lambda r: (0, int(r["id"]), r["id"]) if r["id"].isdecimal() else (1, 0, r["id"]),
        )
        candidates.append(
            dict(
                chosen,
                source_hash=key,
                selection_hash=hashlib.sha256(b"V700" + normalize(blobs[key])).hexdigest(),
            )
        )
    roster, public, offset = [], [], 0
    for role, count in ROLES.items():
        for slot, row in enumerate(candidates[offset : offset + count]):
            fid = canonical_hash(dict(source_hash=row["source_hash"], response_id=row["id"]))
            roster.append(
                dict(
                    family_id=fid,
                    source_id=row["source_id"],
                    response_id=row["id"],
                    role=role,
                    slot=slot,
                    selection_hash=row["selection_hash"],
                    source_cluster_id="sha256:" + hashlib.sha256(row["blob"]).hexdigest(),
                    normalized_source_hash=row["source_hash"],
                    status="completed",
                )
            )
            public.append(
                dict(
                    family_id=fid,
                    source_bytes=row["blob"].hex(),
                    answer_bytes=row["response"].encode().hex(),
                )
            )
        offset += count
    return roster, public, excluded, clusters


def separation(public: list[Json], roster: list[Json]) -> None:
    """Reject labels and all duplicate content before the evaluator can open targets."""
    seen = set()
    for row, role in zip(public, roster, strict=True):
        if set(row) != {"family_id", "source_bytes", "answer_bytes"}:
            raise ValueError("public_fields")
        identity = normalized(bytes.fromhex(row["source_bytes"]))
        if identity in seen:
            raise ValueError("cross_role_duplicate")
        seen.add(identity)


def seal(root: Path, raw: Path, mutation: str = "", inventory_path: Path | None = None) -> Json:
    """Persist inventory, algorithm and selection before reading selected labels."""
    raw.mkdir(parents=True, exist_ok=True)
    plan: Json = dict(
        raw=str(raw.absolute()),
        rows=[],
        failures=[],
        preconditions_checked=[],
        role_manifests={},
        evaluator_label_manifests={},
        raw_shard_hashes=[],
        exposure_inventory={},
        near_duplicate_clusters=[],
        exclusion_rows=[],
        selection_sealed_before_labels=False,
    )
    for name in ("source_info.jsonl", "response.jsonl"):
        path = root / "data/ragtruth" / name
        check = operand(path, "exists", True, path.is_file())
        plan["preconditions_checked"].append(check)
        if not check["passed"]:
            plan["failures"].append(check)
    if plan["failures"]:
        return plan
    sources, responses = reader.public_training(root)
    inv = inventory(root, sources, raw, inventory_path)
    if mutation == "unavailable_history":
        inv["unknown_history"].append(
            dict(path=str(root / "results/missing-public.json"), reason="missing", blocking=True)
        )
    plan["exposure_inventory"] = immutable(raw / "exposure_inventory.json", inv)
    plan["raw_shard_hashes"].append(plan["exposure_inventory"])
    plan["raw_shard_hashes"].append(
        immutable(
            raw / "selection_algorithm.json",
            dict(
                roles=ROLES,
                normalization="NFKC/casefold/collapsed whitespace",
                salt="V700",
                response_order="numeric then lexical lowest ID",
                shingle_tokens=5,
                jaccard_threshold=0.8,
                inventory_sha256=plan["exposure_inventory"]["sha256"],
                replacement=False,
            ),
        )
    )
    roster, public, excluded, clusters = select(sources, responses, inv)
    plan.update(
        exclusion_rows=excluded,
        near_duplicate_clusters=clusters,
        known_excluded_count=len({sid for row in excluded for sid in row["source_ids"]}),
        unknown_history=inv["unknown_history"],
        public_pool_selected_count=len(roster),
    )
    checks = [
        operand(
            raw / "exposure_inventory.json",
            "available_history",
            [],
            [r for r in inv["unknown_history"] if r["blocking"]],
        ),
        operand(
            root / "data/ragtruth/source_info.jsonl",
            "disjoint_public_groups",
            768,
            len(roster),
            ">=",
        ),
    ]
    plan["preconditions_checked"].extend(checks)
    plan["failures"].extend(r for r in checks if not r["passed"])
    if plan["failures"]:
        return plan
    if mutation == "contamination":
        public[0]["y"] = 0
    if mutation == "duplicate":
        public[128]["source_bytes"] = public[0]["source_bytes"]
    try:
        separation(public, roster)
    except ValueError as error:
        plan["failures"].append(operand(raw / "selection.json", "separation", "passed", str(error)))
        return plan
    frozen, audit = targets.freeze(public)
    plan["raw_shard_hashes"].append(
        immutable(raw / "selection.json", dict(roster=roster, public=frozen, audit=audit))
    )
    for role in ROLES:
        manifest = [
            dict(
                r,
                **p,
                public_eligible=a["public_eligible"],
                public_exclusion_reason=a["exclusion_reason"],
            )
            for r, p, a in zip(roster, frozen, audit, strict=True)
            if r["role"] == role
        ]
        plan["role_manifests"][role] = immutable(
            raw / "public" / (role + ".json"), dict(rows=manifest)
        )
    plan["selection_sealed_before_labels"] = True
    progress("selection_sealed_before_evaluator", len(roster), 0)
    selected_ids = {r["response_id"] for r in roster}
    evaluator_responses = []
    with (root / "data/ragtruth/response.jsonl").open() as stream:
        for line in stream:
            match = re.search(r'"id"\s*:\s*("(?:\\.|[^"\\])*")', line)
            if match and json.loads(match[1]) in selected_ids:
                evaluator_responses.append(json.loads(line))
    evaluators = [
        dict(family_id=r["family_id"], role=r["role"], response_id=r["response_id"]) for r in roster
    ]
    labels, spans = targets.join(
        frozen,
        audit,
        dict(roles=roster, evaluators=evaluators, responses=evaluator_responses, sources=sources),
    )
    for role in ROLES:
        ref = immutable(
            raw / "evaluator" / (role + ".json"),
            dict(
                rows=[r for r in labels if r["role"] == role],
                original_span_lineage=[
                    s
                    for s in spans
                    if s["family_id"] in {r["family_id"] for r in roster if r["role"] == role}
                ],
            ),
        )
        Path(ref["path"]).chmod(0o400)
        plan["evaluator_label_manifests"][role] = ref
    plan["rows"] = [
        dict(
            source=r["normalized_source_hash"],
            source_id=r["source_id"],
            unit=r["family_id"],
            role=r["role"],
            arm="cohort_custody",
            condition=r["role"],
            slot=r["slot"],
            metric="eligible_complete_response",
            numerator=int(label["status"] == "completed"),
            denominator=1,
            eligible=label["status"] == "completed",
            status=label["status"],
            exclusion_reason=label["exclusion_reason"],
            exposure="recorded_history_disjoint; unknown external/pretraining history",
        )
        for r, label in zip(roster, labels, strict=True)
    ]
    plan["class_support"] = {
        role: {str(y): sum(r["role"] == role and r["y"] == y for r in labels) for y in (0, 1)}
        for role in ROLES
    }
    plan["raw_shard_hashes"].extend(
        [*plan["role_manifests"].values(), *plan["evaluator_label_manifests"].values()]
    )
    progress("evaluator_complete", len(labels), 0)
    return plan
