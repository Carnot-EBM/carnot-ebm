"""REQ-VERIFY-7994: public training selection precedes evaluator annotation access.

This is bounded recorded source disjointness. Public pretraining and unknown
historical access remain unknown. No pretrained model or fitted head is used.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
from typing import Any
import unicodedata

from carnot.reporting import evidence_features_custody_7980 as c
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import evidence_features_7980 as f
from carnot.verify import response_targets_7955 as targets

Json = dict[str, Any]
PIN = "sha256:dc06fadccb5a0bfce0b545256a9e8133e61f002df642b2d71d829753322438ed"
SALT = b"V693-development"
ROLES = {"calibration": 64, "stream": 256, "retention": 64}
FIELDS = re.compile(r'"(id|source_id|split|response)"\s*:\s*("(?:\\.|[^"\\])*")')


def progress(phase: str) -> None:
    """Flush progress so a supervisor can distinguish work from a stuck child."""
    print("[exp7994] " + phase, flush=True)


def public_training(root: Path) -> tuple[list[Json], list[Json]]:
    """Decode only public string fields; never parse labels or TEST records."""
    responses = []
    with (root / "data/ragtruth/response.jsonl").open() as stream:
        for index, line in enumerate(stream):
            if re.search(r'"split"\s*:\s*"train"', line):
                responses.append({k: json.loads(v) for k, v in FIELDS.findall(line)})
            if index % 512 == 0:
                progress(f"public_projection scanned={index}")
    return c.selected_sources(
        root / "data/ragtruth/source_info.jsonl", {r["source_id"] for r in responses}
    ), responses


def authenticate(root: Path, *, fixture: bool = False) -> Json:
    """Missing contract fields are explicit failures rather than observed zeros."""
    path = root / "results/experiment_7980_v692_evidence_features.json"
    exists = path.is_file()
    value = json.loads(path.read_text()) if exists else {}
    expected = dict(
        experiment_id=7980,
        task_id="exp7980-evidence-features",
        run_date="20261001",
        feature_views_ready_score=1,
        flagged_adversarial=False,
    )
    checks = [c.operand(7980, path, "exists", True, exists)]
    if not fixture:
        checks.append(c.operand(7980, path, "sha256", PIN, sha256_file(path) if exists else None))
    checks.extend(
        c.operand(7980, path, k, v, value.get(k, "MISSING_CONTRACT_FIELD"))
        for k, v in expected.items()
    )
    if exists:
        for key in ("original_public", "rows", "exposure_audit"):
            checks.append(
                c.operand(
                    7980,
                    path,
                    key + "_contract",
                    "present",
                    "present" if key in value else "MISSING_CONTRACT_FIELD",
                )
            )
        manifests = [value["original_public"]] if "original_public" in value else []
        manifests += list(value.get("public_role_manifests", {}).values())
        for item in manifests:
            p = Path(item["path"])
            checks.append(
                c.operand(
                    7980, p, "sha256", item["sha256"], sha256_file(p) if p.is_file() else None
                )
            )
    for name in ("source_info.jsonl", "response.jsonl"):
        p = root / "data/ragtruth" / name
        checks.append(c.operand("official_train", p, "exists", True, p.is_file()))
    references = (
        [
            dict(
                c.reference(path),
                producer_invocation_date=value.get("run_date", "MISSING_CONTRACT_FIELD"),
                imported_fields=["original_public", "rows", "exposure_audit"],
            )
        ]
        if exists
        else []
    )
    return dict(
        upstream=value,
        checks=checks,
        references=references,
        failures=[r for r in checks if not r["passed"]],
    )


def exclusions(plan: Json, *, fixture: bool = False) -> tuple[set[str], set[str], list[Json]]:
    """Separate explicit evaluation exclusion from known general training exposure."""
    upstream = plan["upstream"]
    original = json.loads(c.checked(upstream["original_public"]).read_text())["request_rows"]
    if not fixture and (
        len(original) != 640
        or len(upstream["rows"]) != 640
        or {r["family_id"] for r in original} != {r["family_id"] for r in upstream["rows"]}
    ):
        raise ValueError("original_roster_contract")
    ids = {r["source_id"] for r in upstream["rows"]}
    hashes = {f.normalized(bytes.fromhex(r["source_bytes"])) for r in original}
    hashes.update(hashlib.sha256(bytes.fromhex(r["source_bytes"])).hexdigest() for r in original)
    history = upstream["exposure_audit"]["discovered_exclusions"]
    for row in history:
        if re.search(r"(?:^|[/_])(?:evaluation|eval|test|retention)(?:[/_.-]|$)", row["path"]):
            ids.update(row["source_ids"])
            hashes.update(row["source_hashes"])
    return ids, hashes, history


def select(
    sources: list[Json], responses: list[Json], ids: set[str], hashes: set[str]
) -> tuple[list[Json], list[Json]]:
    """Choose stable identities without inspecting quality, annotations or models."""
    indexed = {s["source_id"]: s["source_info"] for s in sources}
    groups: dict[str, list[Json]] = {}
    for r in responses:
        if r["split"] != "train" or r["source_id"] not in indexed:
            continue
        s = indexed[r["source_id"]]
        blob = (
            s
            if isinstance(s, str)
            else json.dumps(s, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
        ).encode()
        key = f.normalized(blob)
        groups.setdefault(key, []).append(dict(r, blob=blob))
    ordered = []
    for key, candidates in groups.items():
        if key in hashes or any(
            r["source_id"] in ids or hashlib.sha256(r["blob"]).hexdigest() in hashes
            for r in candidates
        ):
            continue
        r = min(
            candidates,
            key=lambda r: (0, int(r["id"]), r["id"]) if r["id"].isdecimal() else (1, 0, r["id"]),
        )
        normalized = (
            re.sub(r"\s+", " ", unicodedata.normalize("NFKC", r["blob"].decode()).casefold())
            .strip()
            .encode()
        )
        ordered.append((hashlib.sha256(SALT + normalized).hexdigest(), key, r))
    roster, public = [], []
    for i, (order, key, r) in enumerate(sorted(ordered)[:384]):
        role = "calibration" if i < 64 else "stream" if i < 320 else "retention"
        fid = canonical_hash(dict(source_hash=key, response_id=r["id"]))
        roster.append(
            dict(
                family_id=fid,
                source_id=r["source_id"],
                response_id=r["id"],
                normalized_source_hash=key,
                selection_hash=order,
                role=role,
                slot=i - 64 if role == "stream" else None,
                order=i,
            )
        )
        public.append(
            dict(
                family_id=fid,
                source_bytes=r["blob"].hex(),
                answer_bytes=r["response"].encode().hex(),
            )
        )
    check_roles(roster)
    return roster, public


def check_roles(roster: list[Json]) -> None:
    """Reject duplicate groups even when a mutation moves them across roles."""
    if len({r["normalized_source_hash"] for r in roster}) != len(roster) or len(
        {r["family_id"] for r in roster}
    ) != len(roster):
        raise ValueError("duplicate_role")


def schedule(roster: list[Json], available: set[str]) -> list[Json]:
    """Missing labels and late feedback keep their original time slots."""
    stream = {r["slot"]: r for r in roster if r["role"] == "stream"}
    return [
        dict(
            slot=i,
            family_id=stream.get(i, {}).get("family_id"),
            reveal_at_slot=i + 20,
            status="available"
            if stream.get(i, {}).get("family_id") in available
            else "unavailable",
            end_censored=i + 20 >= 256,
        )
        for i in range(256)
    ]


def authorize(row: Json, purpose: str, slot: int, audit: Json | None) -> None:
    """Only delayed stream feedback and sealed independent retention audits open labels."""
    if row["role"] == "calibration" and purpose == "calibrate":
        return
    if row["role"] == "stream" and purpose == "feedback":
        if not row["slot"] + 20 <= slot < 256:
            raise ValueError("future_label")
        return
    if row["role"] == "retention" and purpose == "audit":
        if (
            audit is None
            or json.loads(c.checked(audit).read_text()).get("independent_learning_audit")
            is not True
        ):
            raise ValueError("audit_seal")
        return
    raise ValueError("role_access")


def seal(raw: Path, roster: list[Json], public: list[Json]) -> Path:
    """All public bytes and role orders reach storage before annotation access."""
    check_roles(roster)
    features = []
    for i, row in enumerate(public):
        features.append(f.extract(row))
        if i % 32 == 0:
            progress(f"features completed={i + 1}/{len(public)}")
    documents = {
        "public": dict(request_rows=public),
        "features": dict(rows=features),
        "roster": dict(rows=roster),
        "schedule": dict(rows=schedule(roster, set())),
    }
    refs = {}
    for name, data in documents.items():
        path = raw / "public" / (name + ".json")
        atomic_json(path, data)
        refs[name] = c.reference(path)
    refs["roles"] = {}
    for role in ROLES:
        selected = {r["family_id"] for r in roster if r["role"] == role}
        path = raw / "public" / (role + ".json")
        atomic_json(
            path,
            dict(
                request_rows=[r for r in public if r["family_id"] in selected],
                features=[r for r in features if r["family_id"] in selected],
            ),
        )
        refs["roles"][role] = dict(c.reference(path), count=len(selected))
    path = raw / "public/seal.json"
    atomic_json(path, refs)
    return path


def evaluate(root: Path, seal_path: Path, output: Path) -> Json:
    """The evaluator alone opens annotations and emits label-free terminal states."""
    progress("evaluator_after_public_seal")
    seal_data = json.loads(seal_path.read_text())
    for key in ("public", "features", "roster", "schedule"):
        c.checked(seal_data[key])
    roster = json.loads(c.checked(seal_data["roster"]).read_text())["rows"]
    c.reserved_join(root, seal_path, output / "all_labels.json")
    labels = json.loads((output / "all_labels.json").read_text())
    by_id = {r["family_id"]: r for r in roster}
    features = {
        r["family_id"]: r for r in json.loads(c.checked(seal_data["features"]).read_text())["rows"]
    }
    for row in labels["rows"]:
        row.update(role=by_id[row["family_id"]]["role"], slot=by_id[row["family_id"]]["slot"])
        if features[row["family_id"]]["abstention"]:
            row.update(
                y=None, status="excluded", exclusion_reason=features[row["family_id"]]["abstention"]
            )
    refs = {}
    for role in ROLES:
        rows = [r for r in labels["rows"] if r["role"] == role]
        ids = {r["family_id"] for r in rows}
        path = output / (role + ".json")
        atomic_json(
            path,
            dict(
                rows=rows,
                annotation_rows=[r for r in labels["annotation_rows"] if r["family_id"] in ids],
                access_policy="evaluator_only",
                public_seal_sha256=sha256_file(seal_path),
            ),
        )
        path.chmod(0o600)
        refs[role] = dict(c.reference(path), count=len(rows))
    primitive = [
        dict(
            family_id=r["family_id"],
            role=r["role"],
            status=r["status"],
            eligibility=r["y"] is not None,
            numerator=int(r["y"] is not None),
            denominator=1,
            failure=r["exclusion_reason"],
            censor_status=False,
            arm="cohort_preparation",
            seed=69394,
        )
        for r in labels["rows"]
    ]
    text = "Unsupported."
    control = dict(
        response=text,
        quality="good",
        labels=[
            dict(
                start=0,
                end=len(text),
                text=text,
                label_type="unsupported",
                implicit_true=False,
                due_to_null=False,
                meta=None,
            )
        ],
    )
    from carnot.verify.sentence_labels_7942 import checked_spans

    controls = dict(
        positive=int(bool(checked_spans(control, text.encode()))),
        negative=int(bool(checked_spans(dict(control, labels=[]), text.encode()))),
    )
    summary = dict(
        rows=primitive,
        evaluator_role_manifests=refs,
        positive_control_results=dict(
            controls,
            passed=controls == {"positive": 1, "negative": 0},
            scope="synthetic_annotation_transport_only",
        ),
        headroom=dict(
            eligible=sum(r["y"] is not None for r in labels["rows"]),
            positive=sum(r["y"] == 1 for r in labels["rows"]),
            negative=sum(r["y"] == 0 for r in labels["rows"]),
        ),
    )
    atomic_json(output / "summary.json", summary)
    return summary


def read_label(
    path: Path, family_id: str, purpose: str, slot: int, audit: Json | None = None
) -> int | None:
    """This evaluator API rejects early access before returning a selected target."""
    row = next(r for r in json.loads(path.read_text())["rows"] if r["family_id"] == family_id)
    authorize(row, purpose, slot, audit)
    return row["y"]
