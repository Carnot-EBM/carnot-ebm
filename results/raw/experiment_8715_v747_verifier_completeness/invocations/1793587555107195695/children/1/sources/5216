"""Select and seal public LettuceDetect families for REQ-REPORT-7701.

Only the label callback may read annotation bytes. The public protocol reaches
disk before that callback runs. Cold reduction checks exact sealed bytes.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
import json
import os
from pathlib import Path
from typing import Any

from carnot.reporting.current_work_receipt import atomic_json, sha256_file
from carnot.reporting import fresh_relation_cohort as base


def plan(
    rows: Sequence[Mapping[str, Any]],
    ledger: Mapping[str, set[str]],
    prior_selected_ids: set[str],
    counts: Mapping[str, int],
    salt: str,
) -> dict:
    """Cluster public rows and subtract all known prior exposure before roles."""
    families, collisions = base.cluster_public_rows(rows)
    all_exposure = {name: set(values) for name, values in ledger.items()}
    all_exposure.setdefault("source_hashes", set())
    all_exposure.setdefault("answer_hashes", set())
    all_exposure.setdefault("family_ids", set()).update(prior_selected_ids)
    eligible, exposed = base.subtract_exposure(families, all_exposure)
    selected = base.assign_roles(eligible, counts, salt)
    chosen = {item["family_id"] for item in selected}
    if len(chosen) != sum(counts.values()) or chosen & prior_selected_ids:
        raise ValueError("selected_prior_exposure")
    return {
        "selected": selected,
        "collisions": collisions,
        "exclusions": exposed,
        "selected_prior_exposure_count": len(chosen & prior_selected_ids),
        "excluded_prior_exposure_count": len(exposed),
        "inventory": {
            "public_rows": len(rows),
            "clustered_families": len(families),
            "split_or_question_exclusions": len(collisions),
            "eligible_families": len(eligible),
        },
    }


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]], mode: int) -> str:
    """Write one complete role store with explicit public or evaluator mode."""
    data = "".join(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n" for row in rows)
    path.write_text(data, encoding="utf-8")
    os.chmod(path, mode)
    return sha256_file(path)


def _read_jsonl(path: Path) -> list[dict]:
    """Read only a named role store, with invalid bytes failing closed."""
    try:
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("jsonl_corrupt") from exc


def predictor_inputs(path: Path, role: str) -> list[dict]:
    """Give a predictor only one public role store, with no evaluator handle."""
    rows = _read_jsonl(path)
    for row in rows:
        base.validate_predictor(row, role)
    return rows


def seal(
    raw_dir: Path,
    selected: Sequence[Mapping[str, Any]],
    counts: Mapping[str, int],
    salt: str,
    label_reader: Callable[[Mapping[str, Any]], int],
) -> dict:
    """Freeze label-free roles, then write separately permissioned labels."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    roles: dict[str, dict] = {}
    for role, expected in counts.items():
        members = [item for item in selected if item["role"] == role]
        if len(members) != expected:
            raise ValueError(f"underfilled_{role}")
        inputs = [base.predictor_view(item["view"], item["family_id"], role) for item in members]
        for item in inputs:
            base.validate_predictor(item, role)
        name = f"{role}_model_inputs.jsonl"
        digest = _write_jsonl(raw_dir / name, inputs, 0o644)
        roster = [item["family_id"] for item in members]
        roles[role] = {
            "families": roster,
            "role_hash": base.stable_hash(roster),
            "model_inputs": name,
            "model_inputs_sha256": digest,
            "source_hashes": [item["source_sha256"] for item in inputs],
            "group_count": len(members),
        }
    if len({item["family_id"] for item in selected}) != sum(counts.values()):
        raise ValueError("cross_role_family")
    public_protocol = {
        "schema": "carnot.exp7701.v671.public.v1",
        "salt": salt,
        "counts": dict(counts),
        "roles": roles,
        "selection_rule": "public source-question-answer families; subtract prior exposure; salted hash order",
        "claim_scope": "data validity only; no source informativeness or natural hallucination claim",
        "label_access": "evaluator only after public protocol freeze",
    }
    atomic_json(raw_dir / "public_protocol.json", public_protocol)
    evaluator_stores: dict[str, dict] = {}
    for role in counts:
        members = [item for item in selected if item["role"] == role]
        labels = []
        for item in members:
            label = label_reader(item)
            if label not in (0, 1):
                raise ValueError("label_invalid")
            labels.append(
                {
                    "family_id": item["family_id"],
                    "answer_sha256": base.digest(item["view"]["answer"]),
                    "role": role,
                    "label": label,
                    "provenance": "LettuceDetect injected-error annotation spans",
                }
            )
        name = f"{role}_evaluator_store.jsonl"
        evaluator_stores[role] = {
            "path": name,
            "sha256": _write_jsonl(raw_dir / name, labels, 0o600),
            "count": len(labels),
            "access": "evaluator_only_mode_0600; no predictor handle",
        }
    protocol = {
        **public_protocol,
        "public_protocol_sha256": sha256_file(raw_dir / "public_protocol.json"),
        "evaluator_stores": evaluator_stores,
    }
    atomic_json(raw_dir / "protocol.json", protocol)
    return protocol


def cold_reduce(protocol_path: Path, counts: Mapping[str, int]) -> dict:
    """Reconstruct a sealed roster from files rather than producer memory."""
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    raw_dir = protocol_path.parent
    public_path = raw_dir / "public_protocol.json"
    if sha256_file(public_path) != protocol["public_protocol_sha256"]:
        raise ValueError("public_protocol_hash_mismatch")
    public = json.loads(public_path.read_text(encoding="utf-8"))
    if public["roles"] != protocol["roles"] or public["counts"] != dict(counts):
        raise ValueError("role_protocol_mismatch")
    observed: Counter[str] = Counter()
    families: set[str] = set()
    selected_prior = 0
    for role, expected in counts.items():
        info = public["roles"][role]
        evaluator = protocol["evaluator_stores"][role]
        model_path = raw_dir / info["model_inputs"]
        label_path = raw_dir / evaluator["path"]
        if sha256_file(model_path) != info["model_inputs_sha256"]:
            raise ValueError("model_inputs_hash_mismatch")
        if sha256_file(label_path) != evaluator["sha256"]:
            raise ValueError("evaluator_hash_mismatch")
        models = _read_jsonl(model_path)
        labels = _read_jsonl(label_path)
        if len(models) != expected or len(labels) != expected:
            raise ValueError("role_count_mismatch")
        if [item["component_hash"] for item in models] != info["families"]:
            raise ValueError("role_hash_mismatch")
        if base.stable_hash(info["families"]) != info["role_hash"]:
            raise ValueError("role_hash_mismatch")
        if [item["source_sha256"] for item in models] != info["source_hashes"]:
            raise ValueError("source_hash_mismatch")
        for model, label in zip(models, labels, strict=True):
            base.validate_predictor(model, role)
            if set(label) != {"family_id", "answer_sha256", "role", "label", "provenance"}:
                raise ValueError("label_schema_mismatch")
            family_id = model["component_hash"]
            if family_id in families:
                raise ValueError("cross_role_family")
            families.add(family_id)
            if model["historically_exposed"]:
                selected_prior += 1
            if (
                label["family_id"] != family_id
                or label["role"] != role
                or label["answer_sha256"] != model["answer_sha256"]
                or label["label"] not in (0, 1)
            ):
                raise ValueError("label_join_mismatch")
            if (role == "evaluation") != (model["official_split"] == "test"):
                raise ValueError("role_split_mismatch")
            observed[role] += 1
    if len(families) != sum(counts.values()) or selected_prior:
        raise ValueError("selected_prior_exposure_or_count")
    return {
        "families": len(families),
        "role_counts": dict(observed),
        "selected_prior_exposure_count": selected_prior,
        "evaluator_access": "joined only in cold evaluator reduction",
    }
