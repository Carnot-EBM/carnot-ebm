"""REQ-REPORT-8591: authenticate affected evidence without replaying old controllers.

The original canary receipt keeps its rejected dependency. A separate descriptor
allows a new consumer to qualify an explicit revision without rewriting history.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from carnot.reporting import v737_contract_methods as old
from carnot.reporting import v738_contract_methods as e
from carnot.reporting import causal_accounting_8581 as accounting

Json = dict[str, Any]


def historical(p: Json) -> Json:
    """A changed validator affects learning admission, while resident parity stands alone."""
    h = dict(old.historical(p))
    extra = []
    for n, field in [(8580, "current_validator_ready_score"), (8581, "online_adapter_ready_score")]:
        e.progress(f"family_{n}_before")
        row = old.family(p["original_producers"][str(n)], n)
        try:
            value = e.read(p["original_producers"][str(n)])
            if value.get(field) != 1:
                raise ValueError(field)
        except (OSError, ValueError, KeyError, TypeError) as error:
            row["qualified"] = False
            row["gate_check_summary"].append(
                e.gate(
                    Path(p["original_producers"][str(n)]["path"]), field, 1, str(error), f"exp{n}"
                )
            )
        extra.append(row)
        e.progress(f"family_{n}_after", 1, 0)
    h["families"] += extra
    h["references"] += [r for row in extra for r in row["references"]]
    h["gate_check_summary"] += [g for row in extra for g in row["gate_check_summary"]]
    h["learning_inputs_ready_score"] = int(
        h["learning_inputs_ready_score"] and all(r["qualified"] for r in extra)
    )
    return h


def canary_descriptor(p: Json) -> Json:
    """Keep the old hash and new evidence side by side, with identical numeric operands."""
    old_ref = p["original_producers"]["8552"]
    original = e.read(old_ref)
    dependency = next(
        r
        for r in original["code_config_hashes"]
        if r.get("path", "").endswith("/scripts/adversarial_verify.py")
    )
    current = e.read(p["original_producers"]["8580"])
    return dict(
        schema="carnot.v738.canary_revision.v1",
        original_canary_reference=old_ref,
        rejected_original_dependency=dependency,
        current_dependency=e.reference(e.ROOT / "scripts/adversarial_verify.py"),
        current_validator_primary=p["original_producers"]["8580"],
        current_validator_reference=current["current_validator_reference"],
        evidence_canary=p["evidence_canary"],
        evidence_canary_sha256=e.canonical_hash(p["evidence_canary"]),
        source_reference=p["original_producers"]["8529"],
        head_reference=p["original_producers"]["8431"],
        original_rejection_controls_preserved=True,
    )


def accept_canary(path: Path) -> bool:
    """A fresh CLI must authenticate exact revision evidence and reject unexplained drift."""
    try:
        candidate = e.read(e.reference(path))
        p = e.protocol()
        if candidate != canary_descriptor(p):
            return False
        original = old.family(candidate["original_canary_reference"], 8552)
        failures = original["gate_check_summary"]
        if (
            len(failures) != 1
            or failures[0]["path"] != candidate["rejected_original_dependency"]["path"]
        ):
            return False
        revision = accounting.accept_revision(candidate["current_validator_reference"])
        return bool(
            revision["ready"]
            and candidate["current_dependency"] in revision["code"]
            and all(
                old.family(candidate[key], n)["qualified"]
                for key, n in [("head_reference", 8431), ("source_reference", 8529)]
            )
        )
    except (OSError, ValueError, KeyError, TypeError):
        return False
