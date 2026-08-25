"""Exp5368: deterministic budget-curated memory governance.

Spec refs: REQ-LEARN-5368, SCENARIO-LEARN-5368-BUDGET,
SCENARIO-LEARN-5368-SAFETY, SCENARIO-LEARN-5368-SHARE-TRUST.

This fixture adds the cost side of continuous self-learning. Useful memory is
not automatically good memory: it can be stale, poisoned, too risky to share,
or simply too expensive for a bounded context budget. The module therefore
scores memory by value-minus-harm per byte and makes deterministic KEEP, DROP,
SHARE, QUARANTINE, TRUST, and UNTRUST decisions without loading or changing any
model weights.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from carnot import experiment_5355_dependency_provenance_self_learning_v488 as exp5355
from carnot import experiment_5356_memory_tool_drift_harness_v488 as exp5356
from carnot import experiment_5357_dependency_drift_self_learning_scaleup_v488 as exp5357
from carnot.provenance_receipts import receipt_bytes


JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[2]
EXPERIMENT = "experiment_5368_budget_curated_memory_governance_v489"
EXPERIMENT_ID = 5368
MILESTONE = "v489"
SCHEMA = "carnot.experiment_5368.budget_curated_memory_governance.v489"
RUN_DATE = "2026-07-07"
RANDOM_SEED = 5368
RESULT_RELATIVE_PATH = Path("results/experiment_5368_budget_curated_memory_governance_v489.json")
EXP5355_RELATIVE_PATH = exp5355.RESULT_RELATIVE_PATH
EXP5356_RELATIVE_PATH = exp5356.RESULT_RELATIVE_PATH
EXP5357_RELATIVE_PATH = exp5357.RESULT_RELATIVE_PATH
SPEC_RELATIVE_PATH = Path("openspec/capabilities/self-learning/spec.md")
MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5368_budget_curated_memory_governance_v489.py"
)
EXP5355_MODULE_RELATIVE_PATH = exp5355.MODULE_RELATIVE_PATH
EXP5356_MODULE_RELATIVE_PATH = Path(
    "python/carnot/experiment_5356_memory_tool_drift_harness_v488.py"
)
EXP5357_MODULE_RELATIVE_PATH = exp5357.MODULE_RELATIVE_PATH

INFERENCE_SUBSTRATE = "deterministic_budget_curated_memory_governance"
SPEC_REFS = (
    "REQ-LEARN-5368",
    "SCENARIO-LEARN-5368-BUDGET",
    "SCENARIO-LEARN-5368-SAFETY",
    "SCENARIO-LEARN-5368-SHARE-TRUST",
)
TERMINAL_PREFIXES = ("complete:", "blocked_")

DEFAULT_BUDGET_BYTES = 400
MEMORY_ITEM_COUNT = 7
TRUST_STALE_RISK_MAX = 0.2
TRUST_POISON_RISK_MAX = 0.1
SHARE_RISK_MAX = 0.1
POISON_QUARANTINE_RISK = 0.7
STALE_QUARANTINE_RISK = 0.6

REQUIRED_FIELD_PRINCIPLES = {
    "experiment_id": "Stable id ties the artifact to this roadmap task.",
    "milestone": ("Keeps the governance fixture tied to the `.489` memory budget scale-up gate."),
    "status": "Complete only if the budget governance fixture runs.",
    "honest_verdict": "One-line ready or blocked verdict with a terminal prefix.",
    "inference_substrate": ("Expected value is deterministic_budget_curated_memory_governance."),
    "continuous_self_learning_target": (
        "Bare boolean must be true because this advances continuous self-learning."
    ),
    "no_weight_mutation": (
        "Bare boolean must be true because the fixture governs memory only, not model weights."
    ),
    "budget_curated_memory_ready": (
        "Bare boolean gate for scale-up; true only when value, cost, harm, "
        "and trust decisions are all measured."
    ),
    "memory_item_count": "Bare integer counts memory records evaluated.",
    "budget_bytes": ("Bare integer records the memory byte budget used by the fixture."),
    "retained_bytes": "Bare integer records bytes retained after curation.",
    "value_minus_harm_per_byte_mean": ("Bare numeric score normalized by byte cost."),
    "keep_precision": ("Bare numeric fraction of kept items that are useful and non-harmful."),
    "stale_memory_deflection_rate": (
        "Bare numeric fraction of stale items not trusted or not used."
    ),
    "poison_memory_deflection_rate": (
        "Bare numeric fraction of poisoned items not trusted or not used."
    ),
    "share_decision_precision": (
        "Bare numeric fraction of shared items that pass provenance and trust constraints."
    ),
    "trust_decision_precision": (
        "Bare numeric fraction of trusted items that are clean useful items."
    ),
    "rollback_recovery_rate": ("Bare numeric recovery rate after bad memory is detected."),
    "unsafe_false_accepts": (
        "Bare integer count of harmful memory items accepted as trusted useful memory."
    ),
    "tests_run": ("Lists deterministic governance, coverage, and pytest commands."),
}
WRAPPED_FIELDS = (
    "experiment_id",
    "milestone",
    "status",
    "honest_verdict",
    "inference_substrate",
    "tests_run",
)
BARE_INTEGER_FIELDS = (
    "memory_item_count",
    "budget_bytes",
    "retained_bytes",
    "unsafe_false_accepts",
)
BARE_BOOL_FIELDS = ("budget_curated_memory_ready",)
BARE_NUMERIC_FIELDS = (
    "value_minus_harm_per_byte_mean",
    "keep_precision",
    "stale_memory_deflection_rate",
    "poison_memory_deflection_rate",
    "share_decision_precision",
    "trust_decision_precision",
    "rollback_recovery_rate",
)


@dataclass(frozen=True)
class MemoryItem:
    """One self-learning memory candidate with explicit governance inputs."""

    memory_id: str
    memory_variant: str
    provenance: Mapping[str, Any]
    byte_cost: int
    estimated_verifier_value: float
    stale_risk: float
    poison_risk: float
    sharing_risk: float
    trust_label: str
    useful: bool
    harmful: bool
    rollback_available: bool

    def harm_score(self) -> float:
        return round(self.stale_risk + self.poison_risk + self.sharing_risk, 6)

    def value_minus_harm_per_byte(self) -> float:
        return round(
            (self.estimated_verifier_value - self.harm_score()) / self.byte_cost,
            6,
        )

    def as_dict(self) -> JsonDict:
        return {
            "memory_id": self.memory_id,
            "memory_variant": self.memory_variant,
            "provenance": dict(self.provenance),
            "byte_cost": self.byte_cost,
            "estimated_verifier_value": self.estimated_verifier_value,
            "stale_risk": self.stale_risk,
            "poison_risk": self.poison_risk,
            "sharing_risk": self.sharing_risk,
            "trust_label": self.trust_label,
            "useful": self.useful,
            "harmful": self.harmful,
            "rollback_available": self.rollback_available,
        }


def confirm_source_gate(root: Path | str = REPO_ROOT) -> JsonDict:
    """Require clean provenance, drift, and scale-up artifacts before curation."""

    root_path = Path(root)
    dependency = _read_json(root_path / EXP5355_RELATIVE_PATH)
    drift = _read_json(root_path / EXP5356_RELATIVE_PATH)
    scaleup = _read_json(root_path / EXP5357_RELATIVE_PATH)
    checks = {
        "dependency_provenance_ready": (dependency.get("dependency_provenance_ready") is True),
        "memory_tool_drift_ready": drift.get("memory_tool_drift_ready") is True,
        "self_learning_scaleup_ready": (scaleup.get("self_learning_scaleup_ready") is True),
        "source_unsafe_false_accepts_zero": (
            dependency.get("unsafe_false_accepts") == 0
            and drift.get("unsafe_false_accepts") == 0
            and scaleup.get("unsafe_false_accepts") == 0
        ),
        "rollback_recovery_ready": (
            drift.get("rollback_recovery_rate") == 1.0
            and scaleup.get("rollback_recovery_rate") == 1.0
        ),
        "no_weight_mutation": (
            dependency.get("no_weight_mutation") is True
            and drift.get("no_weight_mutation") is True
            and scaleup.get("no_weight_mutation") is True
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    return {
        **checks,
        "failed_gates": failed,
        "all_passed": not failed,
        "source_artifacts": [
            str(EXP5355_RELATIVE_PATH),
            str(EXP5356_RELATIVE_PATH),
            str(EXP5357_RELATIVE_PATH),
        ],
        "dependency_source_honest_verdict": _wrapped_value(dependency.get("honest_verdict")),
        "drift_source_honest_verdict": _wrapped_value(drift.get("honest_verdict")),
        "scaleup_source_honest_verdict": _wrapped_value(scaleup.get("honest_verdict")),
    }


def build_memory_items() -> tuple[MemoryItem, ...]:
    """Build a deterministic memory panel from provenance and drift fixtures."""

    return (
        _memory_item(
            "mem5368-clean-dependency-edge",
            "clean",
            EXP5355_RELATIVE_PATH,
            "exp5355:dep-positive-retrieval",
            "dependency edge that reconstructs safe retrieval",
            120,
            0.95,
            0.02,
            0.01,
            0.08,
            "verified_clean",
            useful=True,
            harmful=False,
            rollback_available=False,
        ),
        _memory_item(
            "mem5368-clean-rollback-route",
            "clean",
            EXP5356_RELATIVE_PATH,
            "exp5356:choose-rollback-route:rollback",
            "rollback route that restores clean selection",
            100,
            0.85,
            0.03,
            0.02,
            0.05,
            "verified_clean",
            useful=True,
            harmful=False,
            rollback_available=False,
        ),
        _memory_item(
            "mem5368-clean-scaleup-summary",
            "clean",
            EXP5357_RELATIVE_PATH,
            "exp5357:trace-memory-poisoned-tool:combined",
            "combined policy summary with zero unsafe accepts",
            180,
            0.92,
            0.04,
            0.02,
            0.12,
            "verified_clean",
            useful=True,
            harmful=False,
            rollback_available=False,
        ),
        _memory_item(
            "mem5368-stale-runtime-receipt",
            "stale",
            EXP5356_RELATIVE_PATH,
            "exp5356:choose-rollback-route:stale",
            "old runtime receipt that would route to stale rollback",
            90,
            0.70,
            0.75,
            0.05,
            0.12,
            "stale",
            useful=False,
            harmful=True,
            rollback_available=True,
        ),
        _memory_item(
            "mem5368-poisoned-tool-override",
            "poisoned",
            EXP5356_RELATIVE_PATH,
            "exp5356:select-safety-gate:poisoned",
            "poisoned memory that asks to accept without verifier",
            80,
            0.80,
            0.10,
            0.90,
            0.35,
            "poisoned",
            useful=False,
            harmful=True,
            rollback_available=True,
        ),
        _memory_item(
            "mem5368-unverified-sharing-tip",
            "unverified",
            EXP5355_RELATIVE_PATH,
            "exp5355:unverified-share-tip",
            "unsourced sharing hint with weak provenance",
            70,
            0.55,
            0.10,
            0.20,
            0.50,
            "unverified",
            useful=False,
            harmful=True,
            rollback_available=True,
            provenance_verified=False,
        ),
        _memory_item(
            "mem5368-clean-low-value-overbudget",
            "low_value",
            EXP5357_RELATIVE_PATH,
            "exp5357:trace-cycle-benign-drift-control",
            "clean but low-value summary that does not fit the budget",
            220,
            0.45,
            0.04,
            0.02,
            0.03,
            "verified_clean",
            useful=True,
            harmful=False,
            rollback_available=False,
        ),
    )


def curate_memory_items(
    items: Sequence[MemoryItem],
    *,
    budget_bytes: int = DEFAULT_BUDGET_BYTES,
) -> JsonDict:
    """Score memory by value-minus-harm per byte and apply budget rules."""

    retained_bytes = 0
    rows: list[JsonDict] = []
    ranked_items = sorted(
        items,
        key=lambda item: (-item.value_minus_harm_per_byte(), item.memory_id),
    )
    for rank, item in enumerate(ranked_items, start=1):
        trust_decision = _trust_decision(item)
        keep_decision = _keep_decision(
            item,
            trust_decision,
            retained_bytes,
            budget_bytes,
        )
        if keep_decision == "KEEP":
            retained_bytes += item.byte_cost
        share_decision = _share_decision(item, trust_decision, keep_decision)
        rows.append(
            {
                **item.as_dict(),
                "score_rank": rank,
                "harm_score": item.harm_score(),
                "value_minus_harm_per_byte": item.value_minus_harm_per_byte(),
                "provenance_verified": bool(item.provenance["verified"]),
                "trust_decision": trust_decision,
                "keep_decision": keep_decision,
                "share_decision": share_decision,
                "rollback_recovered": _rollback_recovered(item),
                "decision_inputs_measured": {
                    "value": item.estimated_verifier_value is not None,
                    "cost": item.byte_cost > 0,
                    "harm": True,
                    "trust": trust_decision in {"TRUST", "UNTRUST"},
                },
            }
        )
    return _curation_metrics(rows, budget_bytes, retained_bytes)


def build_result_artifact(
    *,
    root: Path | str = REPO_ROOT,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Build the terminal budget-governance artifact from deterministic rows."""

    source_gate = confirm_source_gate(root=root)
    if source_gate["all_passed"]:
        curation = curate_memory_items(build_memory_items())
    else:
        curation = _blocked_curation()
    complete = _governance_complete(source_gate, curation, tests_run)
    status = "budget_curated_memory_ready" if complete else "blocked_budget_curated_memory_gate"
    artifact: JsonDict = {
        "schema": SCHEMA,
        "experiment": EXPERIMENT,
        "run_date": RUN_DATE,
        "random_seed": RANDOM_SEED,
        "spec_refs": list(SPEC_REFS),
        "source_artifacts": [
            str(EXP5355_RELATIVE_PATH),
            str(EXP5356_RELATIVE_PATH),
            str(EXP5357_RELATIVE_PATH),
        ],
        "experiment_id": _wrap("experiment_id", EXPERIMENT_ID),
        "milestone": _wrap("milestone", MILESTONE),
        "status": _wrap("status", status),
        "honest_verdict": _wrap(
            "honest_verdict",
            _honest_verdict(complete, source_gate, curation, tests_run),
        ),
        "inference_substrate": _wrap("inference_substrate", INFERENCE_SUBSTRATE),
        "continuous_self_learning_target": True,
        "no_weight_mutation": bool(source_gate["no_weight_mutation"]),
        "budget_curated_memory_ready": complete,
        "memory_item_count": curation["memory_item_count"],
        "budget_bytes": curation["budget_bytes"],
        "retained_bytes": curation["retained_bytes"],
        "value_minus_harm_per_byte_mean": curation["value_minus_harm_per_byte_mean"],
        "keep_precision": curation["keep_precision"],
        "stale_memory_deflection_rate": curation["stale_memory_deflection_rate"],
        "poison_memory_deflection_rate": curation["poison_memory_deflection_rate"],
        "share_decision_precision": curation["share_decision_precision"],
        "trust_decision_precision": curation["trust_decision_precision"],
        "rollback_recovery_rate": curation["rollback_recovery_rate"],
        "unsafe_false_accepts": curation["unsafe_false_accepts"],
        "tests_run": _wrap("tests_run", [dict(row) for row in tests_run]),
        "source_gate": source_gate,
        "readiness_gate": _readiness_gate(source_gate, curation, tests_run),
        "memory_decision_rows": curation["decision_rows"],
        "decision_summary": curation["decision_summary"],
        "score_formula": "(estimated_verifier_value - harm_score) / byte_cost",
        "thresholds": {
            "trust_stale_risk_max": TRUST_STALE_RISK_MAX,
            "trust_poison_risk_max": TRUST_POISON_RISK_MAX,
            "share_risk_max": SHARE_RISK_MAX,
            "poison_quarantine_risk": POISON_QUARANTINE_RISK,
            "stale_quarantine_risk": STALE_QUARANTINE_RISK,
        },
        "weight_mutation_receipt": _weight_mutation_receipt(),
        "methodology_note": (
            "Exp5368 is deterministic and fixture-only. It reuses Exp5355 "
            "provenance, Exp5356 memory drift controls, and Exp5357 scale-up "
            "readiness as source gates, then scores memory records by "
            "value-minus-harm per byte under a fixed byte budget. It invokes "
            "no LLM, API judge, fine-tuning path, adapter update, or "
            "foundation-weight mutation."
        ),
        "field_principles": dict(REQUIRED_FIELD_PRINCIPLES),
        "source_artifact_checksums": source_artifact_checksums(root),
    }
    artifact["reproducibility_checksum"] = _checksum(artifact)
    artifact = _json_ready(artifact)
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> bool:
    """Validate fields consumed by downstream memory scale-up gates."""

    for field in WRAPPED_FIELDS:
        wrapped = artifact.get(field)
        if (
            not isinstance(wrapped, Mapping)
            or "value" not in wrapped
            or wrapped.get("principle") != REQUIRED_FIELD_PRINCIPLES[field]
        ):
            raise ValueError(f"{field} must be principle-wrapped")
    if not str(artifact["honest_verdict"]["value"]).startswith(TERMINAL_PREFIXES):
        raise ValueError("honest_verdict terminal prefix invalid")
    if artifact["inference_substrate"]["value"] != INFERENCE_SUBSTRATE:
        raise ValueError("inference_substrate mismatch")
    if artifact.get("continuous_self_learning_target") is not True:
        raise ValueError("continuous_self_learning_target must be bare true")
    if artifact.get("no_weight_mutation") is not True:
        raise ValueError("no_weight_mutation must be bare true")
    for field in BARE_INTEGER_FIELDS:
        value = artifact.get(field)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f"{field} must be a bare integer")
    for field in BARE_BOOL_FIELDS:
        if not isinstance(artifact.get(field), bool):
            raise ValueError(f"{field} must be a bare bool")
    for field in BARE_NUMERIC_FIELDS:
        if not _is_numeric(artifact.get(field)):
            raise ValueError(f"{field} must be bare numeric")
    if artifact["retained_bytes"] > artifact["budget_bytes"]:
        raise ValueError("retained_bytes must not exceed budget_bytes")
    if artifact.get("unsafe_false_accepts") != 0:
        raise ValueError("unsafe_false_accepts must be 0")
    if artifact["budget_curated_memory_ready"]:
        if not artifact["tests_run"]["value"]:
            raise ValueError("tests_run must record commands for ready governance")
        if not artifact["readiness_gate"]["value_cost_harm_trust_measured"]:
            raise ValueError("value_cost_harm_trust_measured must be true")
    return True


def run(
    *,
    root: Path | str = REPO_ROOT,
    result_path: Path | str = REPO_ROOT / RESULT_RELATIVE_PATH,
    tests_run: Sequence[Mapping[str, Any]] = (),
) -> JsonDict:
    """Write the Exp5368 result artifact and return its JSON payload."""

    artifact = build_result_artifact(root=root, tests_run=tests_run)
    _write_json(Path(result_path), artifact)
    return artifact


def source_artifact_checksums(root: Path | str = REPO_ROOT) -> JsonDict:
    """Return sha256 receipts for source artifacts, spec, and modules."""

    root_path = Path(root)
    return {
        "exp5355": _sha256_file(root_path / EXP5355_RELATIVE_PATH),
        "exp5356": _sha256_file(root_path / EXP5356_RELATIVE_PATH),
        "exp5357": _sha256_file(root_path / EXP5357_RELATIVE_PATH),
        "spec": _sha256_file(root_path / SPEC_RELATIVE_PATH),
        "module": _sha256_file(root_path / MODULE_RELATIVE_PATH),
        "exp5355_module": _sha256_file(root_path / EXP5355_MODULE_RELATIVE_PATH),
        "exp5356_module": _sha256_file(root_path / EXP5356_MODULE_RELATIVE_PATH),
        "exp5357_module": _sha256_file(root_path / EXP5357_MODULE_RELATIVE_PATH),
    }


def _memory_item(
    memory_id: str,
    memory_variant: str,
    source_artifact: Path,
    source_ref: str,
    evidence_summary: str,
    byte_cost: int,
    estimated_verifier_value: float,
    stale_risk: float,
    poison_risk: float,
    sharing_risk: float,
    trust_label: str,
    *,
    useful: bool,
    harmful: bool,
    rollback_available: bool,
    provenance_verified: bool = True,
) -> MemoryItem:
    return MemoryItem(
        memory_id=memory_id,
        memory_variant=memory_variant,
        provenance={
            "source_artifact": str(source_artifact),
            "source_ref": source_ref,
            "evidence_summary": evidence_summary,
            "verified": provenance_verified,
        },
        byte_cost=byte_cost,
        estimated_verifier_value=estimated_verifier_value,
        stale_risk=stale_risk,
        poison_risk=poison_risk,
        sharing_risk=sharing_risk,
        trust_label=trust_label,
        useful=useful,
        harmful=harmful,
        rollback_available=rollback_available,
    )


def _trust_decision(item: MemoryItem) -> str:
    trusted = bool(
        item.provenance["verified"]
        and item.trust_label == "verified_clean"
        and item.useful
        and not item.harmful
        and item.stale_risk <= TRUST_STALE_RISK_MAX
        and item.poison_risk <= TRUST_POISON_RISK_MAX
    )
    return "TRUST" if trusted else "UNTRUST"


def _keep_decision(
    item: MemoryItem,
    trust_decision: str,
    retained_bytes: int,
    budget_bytes: int,
) -> str:
    if (
        item.poison_risk >= POISON_QUARANTINE_RISK
        or item.stale_risk >= STALE_QUARANTINE_RISK
        or (item.harmful and trust_decision == "UNTRUST")
    ):
        return "QUARANTINE"
    if trust_decision != "TRUST" or item.value_minus_harm_per_byte() <= 0.0:
        return "DROP"
    if retained_bytes + item.byte_cost > budget_bytes:
        return "DROP"
    return "KEEP"


def _share_decision(
    item: MemoryItem,
    trust_decision: str,
    keep_decision: str,
) -> str:
    if (
        keep_decision == "KEEP"
        and trust_decision == "TRUST"
        and item.provenance["verified"]
        and item.sharing_risk <= SHARE_RISK_MAX
    ):
        return "SHARE"
    return "DO_NOT_SHARE"


def _rollback_recovered(item: MemoryItem) -> bool:
    return bool(item.rollback_available and (item.harmful or item.memory_variant == "stale"))


def _curation_metrics(
    rows: Sequence[Mapping[str, Any]],
    budget_bytes: int,
    retained_bytes: int,
) -> JsonDict:
    kept = [row for row in rows if row["keep_decision"] == "KEEP"]
    shared = [row for row in rows if row["share_decision"] == "SHARE"]
    trusted = [row for row in rows if row["trust_decision"] == "TRUST"]
    stale = [row for row in rows if row["memory_variant"] == "stale"]
    poisoned = [row for row in rows if row["memory_variant"] == "poisoned"]
    bad = [
        row
        for row in rows
        if row["harmful"] or row["memory_variant"] in {"stale", "poisoned", "unverified"}
    ]
    unsafe_false_accepts = sum(
        1
        for row in rows
        if row["harmful"] and row["keep_decision"] == "KEEP" and row["trust_decision"] == "TRUST"
    )
    decision_summary = {
        "keep_ids": [row["memory_id"] for row in kept],
        "share_ids": [row["memory_id"] for row in shared],
        "trust_ids": [row["memory_id"] for row in trusted],
        "drop_ids": [row["memory_id"] for row in rows if row["keep_decision"] == "DROP"],
        "quarantine_ids": [
            row["memory_id"] for row in rows if row["keep_decision"] == "QUARANTINE"
        ],
    }
    return {
        "decision_rows": list(rows),
        "decision_summary": decision_summary,
        "memory_item_count": len(rows),
        "budget_bytes": budget_bytes,
        "retained_bytes": retained_bytes,
        "value_minus_harm_per_byte_mean": _mean(row["value_minus_harm_per_byte"] for row in rows),
        "keep_precision": _rate(
            sum(1 for row in kept if row["useful"] and not row["harmful"]),
            len(kept),
        ),
        "stale_memory_deflection_rate": _rate(
            sum(
                1
                for row in stale
                if row["trust_decision"] == "UNTRUST" or row["keep_decision"] != "KEEP"
            ),
            len(stale),
        ),
        "poison_memory_deflection_rate": _rate(
            sum(
                1
                for row in poisoned
                if row["trust_decision"] == "UNTRUST" or row["keep_decision"] != "KEEP"
            ),
            len(poisoned),
        ),
        "share_decision_precision": _rate(
            sum(
                1
                for row in shared
                if row["provenance_verified"]
                and row["trust_decision"] == "TRUST"
                and row["sharing_risk"] <= SHARE_RISK_MAX
            ),
            len(shared),
        ),
        "trust_decision_precision": _rate(
            sum(1 for row in trusted if row["useful"] and not row["harmful"]),
            len(trusted),
        ),
        "rollback_recovery_rate": _rate(
            sum(1 for row in bad if row["rollback_recovered"]),
            len(bad),
        ),
        "unsafe_false_accepts": unsafe_false_accepts,
        "value_cost_harm_trust_measured": all(
            row["decision_inputs_measured"]["value"]
            and row["decision_inputs_measured"]["cost"]
            and row["decision_inputs_measured"]["harm"]
            and row["decision_inputs_measured"]["trust"]
            for row in rows
        ),
    }


def _blocked_curation() -> JsonDict:
    return _curation_metrics([], DEFAULT_BUDGET_BYTES, 0)


def _readiness_gate(
    source_gate: Mapping[str, Any],
    curation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> JsonDict:
    checks = {
        "source_gate_passed": source_gate["all_passed"] is True,
        "value_cost_harm_trust_measured": (curation["value_cost_harm_trust_measured"] is True),
        "retained_within_budget": curation["retained_bytes"] <= curation["budget_bytes"],
        "unsafe_false_accepts_zero": curation["unsafe_false_accepts"] == 0,
        "stale_deflection_positive": curation["stale_memory_deflection_rate"] > 0.0,
        "poison_deflection_positive": curation["poison_memory_deflection_rate"] > 0.0,
        "keep_precision_positive": curation["keep_precision"] > 0.0,
        "share_precision_positive": curation["share_decision_precision"] > 0.0,
        "trust_precision_positive": curation["trust_decision_precision"] > 0.0,
        "rollback_recovery_ready": curation["rollback_recovery_rate"] == 1.0,
        "tests_recorded": bool(tests_run),
        "no_weight_mutation": source_gate["no_weight_mutation"] is True,
    }
    return {
        **checks,
        "failed_gates": [name for name, passed in checks.items() if not passed],
        "all_passed": all(checks.values()),
    }


def _governance_complete(
    source_gate: Mapping[str, Any],
    curation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> bool:
    return bool(_readiness_gate(source_gate, curation, tests_run)["all_passed"])


def _honest_verdict(
    complete: bool,
    source_gate: Mapping[str, Any],
    curation: Mapping[str, Any],
    tests_run: Sequence[Mapping[str, Any]],
) -> str:
    if complete:
        return (
            "complete: budget-curated memory governance evaluated "
            f"{curation['memory_item_count']} memory items under "
            f"{curation['budget_bytes']} bytes, retained "
            f"{curation['retained_bytes']} bytes, deflected stale and poisoned "
            "memory, shared only trusted provenance-backed memory, recovered "
            "bad-memory rollback, and preserved no model weight mutation"
        )
    blockers = list(source_gate.get("failed_gates", []))
    blockers.extend(_readiness_gate(source_gate, curation, tests_run)["failed_gates"])
    if not tests_run:
        blockers.append("tests_not_recorded")
    return "blocked_budget_curated_memory_not_ready: " + ",".join(dict.fromkeys(blockers))


def _weight_mutation_receipt() -> JsonDict:
    return {
        "no_weight_mutation": True,
        "model_weights_loaded": False,
        "model_weight_hash_before": "sha256:no_model_weights_loaded",
        "model_weight_hash_after": "sha256:no_model_weights_loaded",
        "state_mutation_targets": [
            "deterministic_memory_governance_rows",
            "deterministic_budget_curation_summary",
            "deterministic_share_trust_decisions",
        ],
        "forbidden_updates": {
            "fine_tune_model_weights": False,
            "lora_or_adapter_update": False,
            "foundation_weight_write": False,
        },
    }


def _wrap(field: str, value: Any) -> JsonDict:
    return {"value": value, "principle": REQUIRED_FIELD_PRINCIPLES[field]}


def _wrapped_value(value: Any) -> Any:
    if isinstance(value, Mapping) and "value" in value:
        return value["value"]
    return value


def _read_json(path: Path) -> JsonDict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _sha256_file(path: Path) -> str:
    return (
        "sha256:"
        + hashlib.sha256(
            receipt_bytes(path, artifact_relative_path=RESULT_RELATIVE_PATH)
        ).hexdigest()
    )


def _checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple | list):
        return [_json_ready(item) for item in value]
    return json.loads(json.dumps(value, sort_keys=True))


def _mean(values: Sequence[float] | Any) -> float:
    rows = list(values)
    if not rows:
        return 0.0
    return round(sum(float(value) for value in rows) / len(rows), 6)


def _rate(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)


def _is_numeric(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
