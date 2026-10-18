"""REQ-VERIFY-8172: reuse scalar replay while binding the later V705 clock.

Saved predictions establish what changed. Independent equations and untouched
source masks establish whether that change improved later typed decisions.
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
import json
from pathlib import Path
from typing import Any, Iterator
from unittest.mock import patch

import numpy as np
import yaml

from carnot.reporting.current_work_receipt import atomic_json, canonical_hash, sha256_file
from carnot.verify import admission_horizon_methods_8152 as schedule
from carnot.verify import learning_audit_8144 as previous

Json = dict[str, Any]
ROOT = previous.ROOT
NAME = "experiment_8172_v706_learning_benefit_audit"
CLI = f"scripts/experiments/{NAME}.py"
MODULE = "python/carnot/verify/learning_benefit_audit_8172.py"
RUNNER = "python/carnot/reporting/learning_benefit_execution_8172.py"
TEST = "tests/python/test_learning_benefit_audit_8172.py"
UPSTREAM = "results/experiment_8171_v706_released_feedback_learning.json"
UPSTREAM_HASH = "sha256:7123a7bebb9963319855a4e27301bff27699105e83cb974cc6a6556d100f3a5c"
MODEL_SPECS: list[Json] = []
ARMS = previous.ARMS
BASE_BUILD = previous.build
BASE_STATISTICS = previous.statistics
BASE_CUSTODY = previous.upstream.engine.methods.Custody
reference = previous.reference
progress = schedule.progress


class Custody(BASE_CUSTODY):
    """Bind the new upstream identity without relaxing the qualified byte gates."""

    def require(self, path: Path, field: str, expected: Any, observed: Any) -> None:
        self.upstream = "exp8171-released-feedback-learning"
        if field == "experiment_id":
            expected = 8171
            for name in ["python", "pytest", "coverage", "ruff", "mypy"]:
                tool = ROOT / ".venv/bin" / name
                super().require(tool, "runtime_" + name, True, tool.is_file())
            self.bind(ROOT / schedule.PROTOCOL, schedule.PROTOCOL_HASH)
            exclusion = ROOT / "ops/exclusion_manifest.yaml"
            self.bind(exclusion)
            retired = yaml.safe_load(exclusion.read_text()).get("retired_experiments", [])
            super().require(
                exclusion,
                "experiment8172_not_retired",
                True,
                not any(r.get("experiment_id") == 8172 for r in retired),
            )
        elif field == "numerical_protocol":
            expected = schedule.protocol()
        super().require(path, field, expected, observed)


def reconstruct(state: Json, rows: list[Json], label_path: Path) -> Json:
    """The independent interpreter opens only the feedback slot it just released."""
    schedule.engine.historical.public_rows(rows, 256)
    previous.equal(32, state["capacity"])
    vault = schedule.engine.historical.LabelVault(label_path, rows)
    opened: Json = {}

    class ReleasedLabels(list[int | None]):
        def __getitem__(self, index: Any) -> Any:
            slot = int(index) + 1
            label = vault.release(slot, slot + 20, sealed=True)
            opened[str(slot)] = label
            return label["y"]

    result = schedule.reference(state, rows, ReleasedLabels())
    checkpoints, opportunities, installations = [], [], []
    events = state["events"]
    radial = schedule.engine.historical.radial
    for index, event in enumerate(events):
        slot = event["slot"]
        if event["kind"] == "durable_commit":
            checkpoints.append(
                dict(
                    slot=slot,
                    heads_hash=event["state_hash"],
                    pending=event["pending"],
                    released=[s for s in state["released"] if s + 20 < slot],
                    used_admission=[s for s in state["used_admission"] if s + 20 < slot],
                )
            )
        if event["kind"] in ["admit_once", "defer_candidate"]:
            opportunities.append(
                dict(
                    slot=slot,
                    disposition="admitted" if event["kind"] == "admit_once" else "deferred",
                    steps=event.get("steps", {}),
                )
            )
        if event["kind"] != "admit_once":
            continue
        candidate = next(e for e in reversed(events[:index]) if e["kind"] == "commit_candidate")
        stop = next((e["slot"] for e in events[index + 1 :] if e["kind"] == "admit_once"), 256)
        for arm, step in event["steps"].items():
            if not step:
                continue
            base = candidate["candidates"][arm]["base"]
            changed_probability = changed_decision = eligible = 0
            for row, issued in zip(rows[slot:stop], state["issued"][slot:stop], strict=True):
                if row["values"] is None:
                    continue
                eligible += 1
                before = schedule.engine.scalar_probability(base, state["geometry"], row["values"])
                after = issued["predictions"][arm]
                changed_probability += abs(after - before) > 1e-12
                changed_decision += radial.action(after) != radial.action(before)
            installations.append(
                dict(
                    seed=state["seed"],
                    arm=arm,
                    install_slot=slot,
                    remaining_original_slots=256 - slot,
                    observed_until_slot=stop,
                    eligible_later_predictions=eligible,
                    changed_later_probabilities=int(changed_probability),
                    changed_later_decisions=int(changed_decision),
                )
            )
    progress("installed_heads_replayed", len(installations))
    print(
        f"[exp8172] later_probabilities_changed={sum(r['changed_later_probabilities'] for r in installations)} "
        f"later_decisions_changed={sum(r['changed_later_decisions'] for r in installations)}",
        flush=True,
    )
    return dict(
        result,
        seed=state["seed"],
        heads=state["arms"],
        geometry=state["geometry"],
        opened=opened,
        pending=state["pending"],
        released_count=result["release_count"],
        opportunities=opportunities,
        checkpoints=checkpoints,
        installation_rows=installations,
        final_state_hash=canonical_hash(state),
    )


def statistics(rows: list[Json]) -> Json:
    """Every registered control gets a safety operand even when the gain fails."""
    identities: dict[tuple[str, str], int] = {}
    for row in rows:
        key = (row["condition"], row["source_cluster_id"])
        previous.equal(identities.setdefault(key, row["slot"]), row["slot"])
    result = BASE_STATISTICS(rows)
    panel = [r for r in result["per_source_results"] if r["gain"] is not None]
    controls = {
        arm: float(
            np.mean(
                [
                    r["arms"]["error_center"]["typed_cost"] - r["arms"][arm]["typed_cost"]
                    for r in panel
                ]
            )
        )
        if panel
        else None
        for arm in ARMS
        if arm != "error_center"
    }
    result["other_control_cost_increases"] = controls
    result["h2_passed"] = bool(
        result["h2_passed"] and all(v is not None and v <= 0.02 for v in controls.values())
    )
    result["paired_intervals"] = [result["paired_gain_interval"], *result["sensitivity_intervals"]]
    return result


def exposure(installations: list[Json]) -> Json:
    """Count heads that actually affected subsequent original slots before gain."""
    return dict(
        installed_head_count=len(installations),
        heads_changing_later_probabilities=sum(
            r["changed_later_probabilities"] > 0 for r in installations
        ),
        heads_changing_later_decisions=sum(r["changed_later_decisions"] > 0 for r in installations),
        changed_later_probabilities=sum(r["changed_later_probabilities"] for r in installations),
        changed_later_decisions=sum(r["changed_later_decisions"] for r in installations),
        installation_rows=installations,
        repeat_credit="seed/head counts describe exposure; independent sources are counted once",
    )


@contextmanager
def adapters(installations: list[Json]) -> Iterator[None]:
    """Scoped adapters retain qualified reducers without editing historical code."""

    def replay_head(state: Json, rows: list[Json], labels: Path) -> Json:
        rebuilt = reconstruct(state, rows, labels)
        installations.extend(rebuilt["installation_rows"])
        return rebuilt

    replacements = dict(
        UPSTREAM=UPSTREAM,
        UPSTREAM_HASH=UPSTREAM_HASH,
        MODULE=MODULE,
        RUNNER=RUNNER,
        CLI=CLI,
        TEST=TEST,
        reconstruct=replay_head,
        statistics=statistics,
        build=build,
    )
    with ExitStack() as stack:
        for key, value in replacements.items():
            stack.enter_context(patch.object(previous, key, value))
        stack.enter_context(patch.object(previous.upstream.engine.methods, "Custody", Custody))
        yield


def measure(root: Path, raw: Path, *, fixture: bool = False) -> Json:
    """Qualified custody and retention sealing precede independent reductions."""
    installations: list[Json] = []
    with adapters(installations):
        work = previous.measure(root, raw, fixture=fixture)
    work["installation_exposure_summary"] = exposure(installations)
    work["preconditions_checked"] = dict(
        gate="Exp8171.learning_trajectory_ready_score==1",
        model_loads=0,
        runtime_executable=str(ROOT / ".venv/bin/python"),
        protocol_sha256=schedule.PROTOCOL_HASH,
    )
    work["code_config_hashes"].update(
        {
            p: sha256_file(ROOT / p)
            for p in [
                previous.MODULE,
                schedule.MODULE,
                schedule.PROTOCOL,
                "ops/exclusion_manifest.yaml",
            ]
        }
    )
    if work["input_ready"]:
        upstream = json.loads((root / UPSTREAM).read_text())
        work["cited_upstream_artifacts"] = [
            dict(
                experiment_id=8171,
                fields_imported=[
                    "state_manifest",
                    "input_manifests",
                    "historical_model_provenance",
                ],
                sha256=sha256_file(root / UPSTREAM),
            )
        ]
        work["historical_model_provenance"] = upstream.get("historical_model_provenance", {})
    atomic_json(raw / "measurement.json", work)
    return work


def build(work: Json, raw: Path, receipts: list[Json]) -> Json:
    """Normal readiness, measurable benefit and retained safety stay separate."""
    checked = [dict(r, passed=r["passed"] and r.get("normal_exit", True)) for r in receipts]
    value = BASE_BUILD(work, raw, checked)
    value.pop("reproducibility_checksum")
    value.update(
        experiment_id=8172,
        task_id="exp8172-learning-benefit-audit",
        milestone="2026.10.706",
        retention_ready_score=int(
            value["learning_audit_ready_score"] and value["retention_passed"]
        ),
        causal_checks=work["causal_order_checks"],
        acceptance_gates=schedule.protocol()["statistical_plan"],
        schedule_mechanism_attempt_ended=bool(
            value["verdict_class"] == "null" and value["support_sufficient"]
        ),
        methodology_note="Independent scalar V705 event replay seals all final predictions before retained labels. Seed means preserve original source masks. Probability changes, typed decisions, H2 benefit and retention remain separate.",
    )
    value["field_principles"].update(
        {
            k: "Byte-bound original-source evidence; seed repeats supply no external generalization credit."
            for k in value
            if k not in value["field_principles"]
        }
    )
    value["reproducibility_checksum"] = canonical_hash(value)
    return value


def replay(path: Path) -> bool:
    """Recompute equations, exposure and headlines even after a forged rehash."""
    try:
        value = json.loads(path.read_text())
        previous.equal(
            [8172, "exp8172-learning-benefit-audit"], [value["experiment_id"], value["task_id"]]
        )
        installations: list[Json] = []
        with adapters(installations):
            if not previous.replay(path):
                return False
        previous.equal(exposure(installations), value["installation_exposure_summary"])
        rebuilt = build(
            value,
            Path(value["terminal_validation_sidecar_path"]).parent,
            value["validation_receipts"],
        )
        for key in [
            "retention_ready_score",
            "causal_checks",
            "acceptance_gates",
            "schedule_mechanism_attempt_ended",
        ]:
            previous.equal(rebuilt[key], value[key])
        return True
    except (OSError, ValueError, KeyError, TypeError):
        return False
