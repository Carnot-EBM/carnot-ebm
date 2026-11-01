"""Keep original development text and evaluator labels in a sealed cohort.

Source copies remain one family. This prevents a repeated answer from crossing
roles and looking like a new independent observation (REQ-REPORT-7866).
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Callable
import unicodedata

from carnot.reporting.current_work_receipt import sha256_file
from carnot.verify import source_alignment, source_projection


ROLES = {
    "fit": 256,
    "tune": 64,
    "policy": 64,
    "online_update": 96,
    "online_admission": 64,
    "evaluation": 64,
    "retention": 32,
}
IMPORTS = ("carnot.verify.source_projection", "carnot.verify.source_alignment")


def qualified_imports() -> dict[str, str]:
    """Name the exact Python files that supplied the public projection."""
    return {
        IMPORTS[0]: str(Path(source_projection.__file__).resolve()),
        IMPORTS[1]: str(Path(source_alignment.__file__).resolve()),
    }


def imports_valid(resolved: dict[str, str]) -> bool:
    """Reject the short keys that hid the previous import gate failure."""
    root = Path(__file__).resolve().parents[3]
    return set(resolved) == set(IMPORTS) and all(
        Path(resolved[name]).resolve()
        == root / "python" / Path(*name.split(".")).with_suffix(".py")
        for name in IMPORTS
    )


def group_key(source: str, answer: str) -> str:
    """Group whitespace and case copies before considering role separation."""
    normalized = [
        re.sub(r"\s+", " ", unicodedata.normalize("NFKC", value).casefold()).strip()
        for value in (source, answer)
    ]
    return (
        "sha256:" + hashlib.sha256(json.dumps(normalized, ensure_ascii=False).encode()).hexdigest()
    )


def check_role_groups(rows: list[dict[str, Any]]) -> None:
    """A repeated source-answer pair cannot provide an independent role."""
    groups: dict[str, str] = {}
    for row in rows:
        group, role = row["source_group"], row["role"]
        if group in groups and groups[group] != role:
            raise ValueError("duplicate_group_role_leakage")
        groups[group] = role


def reduce_budget(rows: list[dict[str, Any]], intended: int) -> dict[str, int]:
    """Count families from primitive rows, including every exclusion."""
    states = Counter(row["status"] for row in rows)
    return {
        "intended": intended,
        "eligible": states["completed"],
        "started": len(rows),
        "completed": states["completed"],
        "censored": states["censored"],
        "excluded": states["excluded"],
        "independent": len({row["source_group"] for row in rows}),
    }


def acquire(
    manifest: dict[str, Any], progress: Callable[[int], None]
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Reopen authenticated originals; prior V682 output is only a diagnostic."""
    development_path = Path(manifest["development_manifest_path"])
    development = json.loads(development_path.read_text())
    canonical = source_projection.read_jsonl(Path(manifest["rows_path"]))
    targets = source_projection.read_jsonl(Path(manifest["rows_path"]).with_name("targets.jsonl"))
    rows: list[dict[str, Any]] = []
    public: list[dict[str, str]] = []
    for role, count in ROLES.items():
        entry = development["roles"][role]
        if entry["count"] != count or len(entry["families"]) != count:
            raise ValueError(f"role_count_drift:{role}")
        originals = source_projection.read_jsonl(development_path.parent / entry["public_path"])
        labels = source_projection.read_jsonl(development_path.parent / entry["evaluator_path"])
        if len(originals) != count or len(labels) != count:
            raise ValueError(f"role_shard_drift:{role}")
        for original, label, family in zip(originals, labels, entry["families"], strict=True):
            index = len(rows)
            source, answer = original["complete_source"], original["complete_response"]
            canonical_row, target = canonical[index], targets[index]
            if (
                any(
                    item["family_id"] != family for item in (original, label, canonical_row, target)
                )
                or canonical_row["role"] != role
                or target["role"] != role
                or label["label"] != target["response_label"]
                or original["source_sha256"] != canonical_row["source_sha256"]
                or original["response_sha256"] != canonical_row["response_sha256"]
            ):
                raise ValueError(f"family_join_drift:{index}")
            offsets = [
                [len(answer[: item["start"]].encode()), len(answer[: item["end"]].encode())]
                for item in label["annotations"]
            ]
            if offsets != target["annotation_byte_offsets"] or any(
                answer[item["start"] : item["end"]] != item["text"] for item in label["annotations"]
            ):
                raise ValueError(f"label_offset_drift:{index}")
            if any(
                bytes.fromhex(canonical_row[f"view_{arm}"]["source_bytes"]) != source.encode()
                or bytes.fromhex(canonical_row[f"view_{arm}"]["answer_bytes"]) != answer.encode()
                for arm in ("a", "b")
            ):
                raise ValueError(f"original_text_drift:{index}")
            rows.append(
                {
                    "family_id": family,
                    "role": role,
                    "source_group": group_key(source, answer),
                    "source_sha256": original["source_sha256"],
                    "answer_sha256": original["response_sha256"],
                    "complete_source": source,
                    "complete_response": answer,
                    "label_provenance": {
                        "response_id": label.get("response_id"),
                        "human_label": label["label"],
                        "annotation_byte_offsets": offsets,
                    },
                    "exclusion_reasons": [],
                    "arm": "public_projection",
                    "seed": 68366,
                    "status": "completed",
                }
            )
            public.append(
                {
                    "family_id": family,
                    "source_bytes": source.encode().hex(),
                    "answer_bytes": answer.encode().hex(),
                }
            )
            if len(rows) % 32 == 0:
                progress(len(rows))
    if len(rows) != 640 or len(canonical) != 640 or len(targets) != 640:
        raise ValueError("canonical_roster_drift")
    if len({row["family_id"] for row in rows}) != 640:
        raise ValueError("duplicate_family")
    check_role_groups(rows)
    return rows, public
