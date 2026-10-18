"""REQ-REPORT-8453: receipt custody and resolved support are separate gates.

The original engineering rows remain useful after private custody recovery.
They cannot grant transfer support when the required outcomes are still pending.
"""

from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import time
from typing import Any, Iterator
from unittest.mock import patch

import yaml

from carnot.reporting import arc_custody_8453 as custody
from carnot.reporting import arc_supervisor_support_8439 as old
from carnot.reporting import v726_contract_methods as contract
from carnot.reporting.current_work_receipt import atomic_json, canonical_hash

Json = dict[str, Any]
ROOT = old.ROOT
NAME, TASK = "experiment_8453_v727_arc_seal_and_support", "exp8453-arc-seal-and-support"
MILESTONE = "2026.10.727"
TASK_PIN = "sha256:ea36eead121f3080eec72eda590df882b6d5f6a2ab65bbab8a167e1108723d51"
TASKS_PIN = "026228ee20a3160ad55ba35fe8286fab6db3a14d62a89b8d3d3d171e1acb1aa8"
CLI, TEST = f"scripts/experiments/{NAME}.py", "tests/python/test_arc_seal_and_support_8453.py"
OWNED = [
    "python/carnot/reporting/arc_custody_8453.py",
    "python/carnot/reporting/arc_seal_and_support_8453.py",
    "python/carnot/reporting/arc_seal_and_support_runner_8453.py",
    CLI,
]
MODEL_SPECS: list[Json] = []
reference, checksum, sha256_file = old.reference, old.checksum, old.sha256_file
_MEASURE, _BUILD, _REPLAY, _AUTHORITY = old.measure, old.build, old.replay, contract.authority
_ASSESS = contract.assess_authorities


def progress(phase: str, completed: int = 0, pending: int = 0) -> None:
    """Flushed counts show actual work without requiring a model or game load."""
    print(f"[exp8453] phase={phase} completed={completed} pending={pending}", flush=True)


def authority(root: Path, raw: Path) -> Json:
    """Full-object authentication needs independent design, not just the active YAML."""
    with patch.multiple(
        contract,
        NAME=NAME,
        TASK=TASK,
        MILESTONE=MILESTONE,
        TASKS_PIN=TASKS_PIN,
        assess_authorities=lambda *a, **kw: _ASSESS(*a, **dict(kw, first_id=8443)),
    ):
        result = _AUTHORITY(root, raw)
    active = root / contract.ACTIVE
    tasks = yaml.safe_load(active.read_bytes()).get("tasks", []) if active.is_file() else []
    task = next((t for t in tasks if t["id"] == TASK), None)
    result.update(
        tasks=tasks,
        observed_count=len(tasks),
        task_full_object_hash=canonical_hash(task) if task else None,
    )
    if result["task_full_object_hash"] != TASK_PIN:
        result["activated"] = False
    return dict(result)


@contextmanager
def binding() -> Iterator[None]:
    """Narrow aliases keep the historical reducers and acceptance thresholds intact."""
    with (
        patch.multiple(
            old,
            NAME=NAME,
            TASK=TASK,
            CLI=CLI,
            TEST=TEST,
            OWNED=OWNED,
            build=build,
            progress=progress,
        ),
        patch.object(contract, "authority", authority),
    ):
        yield


def measure(root: Path, raw: Path, private: Path, supplied: Path | None = None) -> Json:
    """Authenticate resources and original custody before assessing resolved support."""
    with binding():
        work = dict(_MEASURE(root, raw, private, supplied))
    for gate in work["failures"]:
        if gate["field"] == "activated_complete_V726_authority":
            gate["field"] = "activated_complete_V727_authority"
    for gate in work["authority"]["gate_check_summary"]:
        work["failures"].append(
            dict(
                check="independent_full_object_authority",
                upstream=TASK,
                path=gate.get("artifact_path", str(root / contract.DESIGN)),
                hash=work["authority"]["authority_snapshots"]["design"].get("sha256"),
                field=gate["artifact_field"],
                operator=gate.get("op", "=="),
                expected=gate["expected"],
                observed=gate["observed"],
                passed=False,
            )
        )
    progress("custody_before")
    work["custody"] = custody.recover(root, raw / "custody", private)
    progress("custody_after", int(work["custody"]["passed"]), int(not work["custody"]["passed"]))
    work["ended_monotonic_ns"] = time.monotonic_ns()
    work["phase_spans"].append(
        dict(
            phase="original_private_custody_recovery",
            duration_s=work["custody"].get("duration_s", 0),
        )
    )
    if work["custody"]["passed"]:
        work["failures"] = [g for g in work["failures"] if g["field"] != "validated_primary_sha256"]
    else:
        work["failures"].append(
            old.gate(
                root / old.evidence.CANDIDATE,
                "recovered_original_publication",
                True,
                work["custody"].get("reason", False),
            )
        )
    work["preconditions_checked"].update(
        measurement_cap_s=3600, optional_cuda_required=False, optional_board_required=False
    )
    work["raw_shard_hashes"].update(
        {str(p): sha256_file(p) for p in (raw / "custody").rglob("*") if p.is_file()}
    )
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, receipts: list[Json], raw: Path, output: Path) -> Json:
    """Custody recovery cannot create the missing third game or resolved redirect trials."""
    with binding():
        value = dict(_BUILD(work, receipts, raw, output))
    recovered = work["custody"]
    qualified = value["required_checks_passed"] and (
        recovered["passed"] or recovered["external_missing"]
    )
    receipt_ready = bool(qualified and recovered["passed"] and len(work["rows"]) == 8)
    summary = old.evidence.summarize(work["rows"])
    support = bool(summary["overlap_by_arm"]) and all(
        c["supported"] for c in summary["overlap_by_arm"].values()
    )
    gates = list(work["failures"])
    for arm, cell in summary["overlap_by_arm"].items():
        if not cell["supported"]:
            g = old.gate(
                raw / "measurement.json",
                "resolved_support:" + arm,
                dict(overlapping_games=3, resolved_per_shared_cell=5),
                dict(
                    overlapping_games=cell["overlapping_games"],
                    resolved_by_cell={
                        c["game"] + ":" + c["policy_arm"]: c["resolved_by_levelup"]
                        for c in cell["cells"]
                    },
                ),
            )
            g.update(
                check="supervisor_refinement_support",
                upstream="exp8412-arc-panel-qualification",
                operator=">=",
            )
            gates.append(g)
    ready = bool(receipt_ready and support and not work["failures"])
    verdict = "disqualified" if not qualified else "null" if ready else "blocked"
    value.update(
        experiment_id=8453,
        task_id=TASK,
        milestone=MILESTONE,
        honest_verdict=f"complete_{verdict}_v727_arc_seal_and_support",
        verdict_class=verdict,
        required_checks_passed=bool(qualified),
        flagged_adversarial=not qualified,
        gate_check_summary=gates,
        arc_receipt_ready_score=int(receipt_ready),
        arc_support_ready_score=int(ready),
        original_validated_hash=recovered.get("original_validated_hash", old.CANDIDATE_PIN),
        current_observed_hash=work["source_artifact_hashes"].get(
            str(Path(work["root"]) / old.evidence.UPSTREAM)
        ),
        historical_publication_flags=recovered.get("flags", {}),
        custody_recovery=recovered,
        new_level_solves=0,
        numeric_learning_substrate="verifier_ensemble_against_cached_candidates",
        temporal_context_availability=dict(
            primitive_action_traces=bool(work["rows"]),
            redirect_windows=bool(work["rows"]),
            resolved_redirect_windows=summary["censoring_counts"]["resolved_redirects"],
            future_lead="TeV",
            arc_transfer_proof=False,
        ),
        avo_refinement_data_needed=[
            "At least three overlapping adapter-withheld games",
            "At least five resolved firings per shared game/policy/supervisor-arm cell",
            "Time-ordered observations, actions, verifier decisions and redirect effects",
            "Authenticated continuation outcomes and censoring boundaries",
        ],
        unchanged_live_policy=True,
        future_dependencies=[
            dict(experiment_id=i, status="future_dependency_not_observation")
            for i in range(8443, 8453)
        ],
    )
    value["acceptance_gates"].update(
        receipt_custody=receipt_ready,
        refinement_support=support,
        current_full_authority=work["authority"]["activated"],
    )
    value["field_principles"].update(
        {
            k: "Bind this field to original sealed evidence; receipt qualification creates no resolved transfer support."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["field_principles"].update(
        arc_receipt_ready_score="Original checked bytes, primitives and fresh historical CLI controls qualify separately from current authority.",
        arc_support_ready_score="Receipt, full authority and unchanged three-game/five-resolved-cell support floors must all pass.",
        historical_publication_flags="Retain original and currently observed flags without changing either historical publication.",
    )
    value["reproducibility_checksum"] = checksum(value)
    return value


def replay(path: Path) -> bool:
    """Fresh primitive replay also rejects self-rehashed claims of recovered custody."""
    try:
        value = json.loads(path.read_bytes())
        work = old.evidence.read(value["work_reference"])
        if not custody.replay(work["custody"]):
            return False
        if work["custody"]["passed"]:
            source = json.loads(Path(work["custody"]["retained_path"]).read_bytes())
            observed_path = (
                Path(value["work_reference"]["path"]).parent / "observed_upstream_primary.json"
            )
            observed = json.loads(observed_path.read_bytes()) if observed_path.is_file() else {}
            for role, operand in [("original", source), ("current", observed)]:
                if work["custody"]["flags"][role] != {
                    k: operand.get(k) for k in ["flagged_adversarial", "adversarial_findings"]
                }:
                    return False
        with binding():
            return bool(_REPLAY(path))
    except (OSError, ValueError, KeyError, TypeError):
        return False
