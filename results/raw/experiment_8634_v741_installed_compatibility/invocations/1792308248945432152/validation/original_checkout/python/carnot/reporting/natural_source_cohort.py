"""Select and seal natural source families for REQ-REPORT-7715.

Every choice uses public source, response, split and quality columns only.
The annotation callback runs after the public manifest reaches disk.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from carnot.experiment_7423_v651_annotated_protocol import serialize_source
from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import fresh_relation_cohort as base


PUBLIC_FIELDS = frozenset(
    {
        "family_id",
        "role",
        "official_split",
        "source_id",
        "response_id",
        "complete_source",
        "complete_response",
        "source_sha256",
        "response_sha256",
    }
)


def build_public_families(
    source_rows: Sequence[Mapping[str, Any]],
    response_rows: Sequence[Mapping[str, Any]],
    salt: str,
) -> tuple[list[dict], list[dict]]:
    """Group source copies and sibling responses without opening labels."""
    sources: dict[str, str] = {}
    for source in source_rows:
        source_id = source["source_id"]
        if not isinstance(source_id, str) or not source_id or source_id in sources:
            raise ValueError("source_identity_invalid")
        sources[source_id] = serialize_source(source["task_type"], source["source_info"])
    rows: list[dict] = []
    excluded: list[dict] = []
    seen_responses: set[str] = set()
    for response in response_rows:
        response_id = response["id"]
        source_id = response["source_id"]
        split = response["split"]
        if not isinstance(response_id, str) or not response_id or response_id in seen_responses:
            raise ValueError("response_identity_invalid")
        seen_responses.add(response_id)
        if source_id not in sources or split not in {"train", "test"}:
            raise ValueError("response_source_or_split_invalid")
        if response["quality"] != "good":
            excluded.append({"response_id": response_id, "reason": "quality_excluded"})
            continue
        answer = response["response"]
        if not isinstance(answer, str) or not answer:
            raise ValueError("response_text_invalid")
        rows.append(
            {
                "source_id": source_id,
                "response_id": response_id,
                "official_split": split,
                "complete_source": sources[source_id],
                "complete_response": answer,
            }
        )
    parents = list(range(len(rows)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    seen: dict[tuple[str, str], int] = {}
    for index, row in enumerate(rows):
        for key in (
            ("id", row["source_id"]),
            ("source", base.normalized(row["complete_source"])),
            ("answer", base.normalized(row["complete_response"])),
        ):
            if key in seen:
                parents[find(index)] = find(seen[key])
            else:
                seen[key] = index
    groups: dict[int, list[dict]] = defaultdict(list)
    for index, row in enumerate(rows):
        groups[find(index)].append(row)
    families: list[dict] = []
    for members in groups.values():
        source_ids = sorted({row["source_id"] for row in members})
        source_hashes = sorted(
            {base.digest(base.normalized(row["complete_source"])) for row in members}
        )
        answer_hashes = sorted(
            {base.digest(base.normalized(row["complete_response"])) for row in members}
        )
        family_id = base.stable_hash(source_hashes)
        splits = {row["official_split"] for row in members}
        if len(splits) != 1:
            excluded.append({"family_id": family_id, "reason": "cross_official_split_family"})
            continue
        winner = min(members, key=lambda row: base.digest(salt + ":" + row["response_id"]))
        families.append(
            {
                "family_id": family_id,
                "source_ids": source_ids,
                "instance_ids": source_ids,
                "source_hashes": source_hashes,
                "answer_hashes": answer_hashes,
                "member_count": len(members),
                "official_split": splits.pop(),
                "view": winner,
            }
        )
    return sorted(families, key=lambda row: row["family_id"]), excluded


def subtract_and_assign(
    families: Sequence[Mapping[str, Any]],
    exposed_source_ids: set[str],
    counts: Mapping[str, int],
    salt: str,
) -> dict:
    """Subtract exposed source IDs, then assign fixed roles by public hash."""
    if len({row["family_id"] for row in families}) != len(families):
        raise ValueError("duplicate_family")
    eligible = [row for row in families if not set(row["source_ids"]) & exposed_source_ids]
    candidate_counts = {
        split: sum(row["official_split"] == split for row in eligible)
        for split in ("train", "test")
    }
    needed = {
        split: sum(
            count for role, count in counts.items() if (role == "evaluation") == (split == "test")
        )
        for split in ("train", "test")
    }
    shortages = {split: max(0, needed[split] - candidate_counts[split]) for split in needed}
    selected: list[dict] = []
    if not any(shortages.values()):
        for split in ("train", "test"):
            ranked = sorted(
                (row for row in eligible if row["official_split"] == split),
                key=lambda row: base.digest(salt + ":" + row["family_id"]),
            )
            offset = 0
            for role, count in counts.items():
                if (role == "evaluation") == (split == "test"):
                    selected.extend(
                        {**row, "role": role} for row in ranked[offset : offset + count]
                    )
                    offset += count
    return {
        "selected": selected,
        "candidate_counts": candidate_counts,
        "shortages": shortages,
        "excluded_prior_exposure_count": len(families) - len(eligible),
        "selected_prior_exposure_count": 0,
    }


def validate_predictor(row: Mapping[str, Any], role: str) -> None:
    """Reject a changed byte, wrong role, or any evaluator-only column."""
    if set(row) != PUBLIC_FIELDS:
        raise ValueError("public_field_mismatch")
    if row["role"] != role:
        raise ValueError("role_mismatch")
    if (role == "evaluation") != (row["official_split"] == "test"):
        raise ValueError("split_mismatch")
    if row["source_sha256"] != base.digest(row["complete_source"]):
        raise ValueError("source_hash_mismatch")
    if row["response_sha256"] != base.digest(row["complete_response"]):
        raise ValueError("response_hash_mismatch")


def predictor_inputs(path: Path, role: str) -> list[dict]:
    """Open only a named public role file; never accept an evaluator path."""
    if path.name != f"{role}_public.jsonl" or ".." in path.parts:
        raise ValueError("public_role_path")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for row in rows:
        validate_predictor(row, role)
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], mode: int) -> str:
    """Write one complete role file and return its exact byte hash."""
    payload = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(payload, encoding="utf-8")
    os.chmod(path, mode)
    return sha256_file(path)


def seal(
    raw_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    salt: str,
    label_reader: Callable[[Mapping[str, Any]], int],
    exposure_receipt: Mapping[str, Any],
    windows_link: Mapping[str, Any],
) -> dict:
    """Freeze public roles and source hashes before reading any human label."""
    if len({row["family_id"] for row in selected}) != len(selected):
        raise ValueError("duplicate_family")
    if len(selected) != sum(counts.values()):
        raise ValueError("role_count_mismatch")
    raw_dir.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict] = {}
    for role, expected in counts.items():
        members = [row for row in selected if row["role"] == role]
        if len(members) != expected:
            raise ValueError("role_count_mismatch")
        public_rows = []
        for member in members:
            view = member["view"]
            public = {
                "family_id": member["family_id"],
                "role": role,
                "official_split": member["official_split"],
                "source_id": view["source_id"],
                "response_id": view["response_id"],
                "complete_source": view["complete_source"],
                "complete_response": view["complete_response"],
                "source_sha256": base.digest(view["complete_source"]),
                "response_sha256": base.digest(view["complete_response"]),
            }
            validate_predictor(public, role)
            public_rows.append(public)
        name = f"{role}_public.jsonl"
        roles[role] = {
            "families": [row["family_id"] for row in public_rows],
            "source_hashes": [row["source_sha256"] for row in public_rows],
            "response_hashes": [row["response_sha256"] for row in public_rows],
            "public_path": name,
            "public_sha256": _write_jsonl(raw_dir / name, public_rows, 0o644),
            "count": expected,
        }
    public_manifest = {
        "schema": "carnot.exp7715.v672.public.v1",
        "salt": salt,
        "counts": dict(counts),
        "roles": roles,
        "exposure_receipt": dict(exposure_receipt),
        "windows_protocol": dict(windows_link),
        "label_definition": "1 iff at least one human-annotated unsupported response span",
        "labels_accessible": False,
    }
    atomic_json(raw_dir / "public_manifest.json", public_manifest)
    for role in counts:
        members = [row for row in selected if row["role"] == role]
        labels = []
        for member in members:
            label = label_reader(member)
            if label not in (0, 1):
                raise ValueError("human_label_invalid")
            labels.append(
                {
                    "family_id": member["family_id"],
                    "response_id": member["view"]["response_id"],
                    "label": label,
                    "authority": "RAGTruth human unsupported spans",
                }
            )
        name = f"{role}_evaluator.jsonl"
        roles[role]["evaluator_path"] = name
        roles[role]["evaluator_sha256"] = _write_jsonl(raw_dir / name, labels, 0o600)
    manifest = {
        **public_manifest,
        "public_manifest_sha256": sha256_file(raw_dir / "public_manifest.json"),
    }
    atomic_json(raw_dir / "manifest.json", manifest)
    return manifest


def cold_reduce(manifest_path: Path, counts: Mapping[str, int]) -> dict:
    """Reopen every public and evaluator byte and reject changed rosters."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw_dir = manifest_path.parent
    public_path = raw_dir / "public_manifest.json"
    if sha256_file(public_path) != manifest["public_manifest_sha256"]:
        raise ValueError("public_manifest_hash_mismatch")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if public["roles"] != {
        role: {key: value for key, value in data.items() if not key.startswith("evaluator_")}
        for role, data in manifest["roles"].items()
    } or public["counts"] != dict(counts):
        raise ValueError("public_manifest_mismatch")
    families: set[str] = set()
    source_hashes: set[str] = set()
    for role, expected in counts.items():
        info = manifest["roles"][role]
        public_file = raw_dir / info["public_path"]
        evaluator_file = raw_dir / info["evaluator_path"]
        if sha256_file(public_file) != info["public_sha256"]:
            raise ValueError("public_hash_mismatch")
        if sha256_file(evaluator_file) != info["evaluator_sha256"]:
            raise ValueError("evaluator_hash_mismatch")
        rows = predictor_inputs(public_file, role)
        labels = [
            json.loads(line) for line in evaluator_file.read_text(encoding="utf-8").splitlines()
        ]
        if len(rows) != expected or len(labels) != expected:
            raise ValueError("role_count_mismatch")
        if [row["family_id"] for row in rows] != info["families"]:
            raise ValueError("family_roster_mismatch")
        if [row["source_sha256"] for row in rows] != info["source_hashes"]:
            raise ValueError("source_roster_mismatch")
        if [row["response_sha256"] for row in rows] != info["response_hashes"]:
            raise ValueError("response_roster_mismatch")
        for row, label in zip(rows, labels, strict=True):
            if row["family_id"] in families or row["source_sha256"] in source_hashes:
                raise ValueError("duplicate_family")
            families.add(row["family_id"])
            source_hashes.add(row["source_sha256"])
            if (
                set(label) != {"family_id", "response_id", "label", "authority"}
                or label["family_id"] != row["family_id"]
                or label["response_id"] != row["response_id"]
                or label["label"] not in (0, 1)
                or label["authority"] != "RAGTruth human unsupported spans"
            ):
                raise ValueError("label_join_mismatch")
    return {"families": len(families), "role_counts": dict(counts), "isolated": True}
