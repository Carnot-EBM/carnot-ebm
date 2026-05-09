"""CerCE-style certificate ledger for FR-11 policy promotion.

The ledger is deliberately boring bookkeeping.  FR-11 can propose query-time
policy updates, but a later promotion step needs a durable certificate that says
which constraints were checked and whether the promoted policy accepted anything
the deterministic contract said to reject.  This module stores those rows in a
stable JSON shape so the promotion gate can be audited without replaying a model.

Spec: REQ-LEARN-1594, SCENARIO-LEARN-1594, SCENARIO-LEARN-1595.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from carnot.pipeline.fr11_event_bus import FR11EventBus, ViolationEvent

JsonDict = dict[str, Any]

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN_DATE = "20260509"
OUTPUT_FILE = "experiment_1594_cerce_ledger.json"
SCHEMA = "cerce_certificate_ledger_v1"
DEFAULT_OUTPUT_PATH = REPO_ROOT / "results" / OUTPUT_FILE
DEFAULT_PROMOTION_MANIFEST_PATH = REPO_ROOT / "results" / "fr11_live_policy_promotion_1524.jsonl"

REQUIRED_ARTIFACT_FIELDS: tuple[str, ...] = (
    "status",
    "schema",
    "continuous_self_learning_task",
    "cerce_ledger_ready",
    "policy_certificates_evaluated",
    "constraint_violation_records",
    "fr11_events_recorded",
    "accepted_violation_count",
    "false_accept_delta",
    "nonforgetting_certificate_rate",
    "promotion_safe_policy_updates",
    "blocked_policy_updates",
    "ledger_rows",
    "blockers",
    "honest_verdict",
)


@dataclass(frozen=True)
class LedgerRow:
    """One constraint certificate row for one policy update.

    A policy can touch many contract cases.  Keeping one row per
    ``(policy_update_id, constraint_id)`` makes the promotion decision explainable:
    an auditor can point at the exact constraint whose promoted behavior accepted
    a violation instead of only seeing a scalar failure count.
    """

    policy_update_id: str
    constraint_id: str
    constraint_type: str
    source: str
    baseline_violation: bool
    promoted_violation: bool
    accepted_violation: bool
    false_accept_delta: int
    soundness_mistake: bool
    certificate_id: str

    def to_dict(self) -> JsonDict:
        """Return a deterministic JSON-compatible representation."""

        return {
            "accepted_violation": self.accepted_violation,
            "baseline_violation": self.baseline_violation,
            "certificate_id": self.certificate_id,
            "constraint_id": self.constraint_id,
            "constraint_type": self.constraint_type,
            "false_accept_delta": self.false_accept_delta,
            "policy_update_id": self.policy_update_id,
            "promoted_violation": self.promoted_violation,
            "soundness_mistake": self.soundness_mistake,
            "source": self.source,
        }


class CerCECertificateLedger:
    """Collect FR-11 events and promotion rows into policy certificates.

    The ledger has two separate data streams.  ``on_fr11_violation`` is the
    EventBus subscriber hook, proving the FR-11 update loop can feed the ledger
    without special-casing the bus.  ``record_constraint_case`` stores the
    promotion-facing certificate rows that decide whether a policy update is
    safe to promote.
    """

    def __init__(self, *, run_date: str = RUN_DATE) -> None:
        self.run_date = run_date
        self._rows: list[LedgerRow] = []
        self._fr11_events: list[JsonDict] = []

    @property
    def fr11_events_recorded(self) -> int:
        """Number of FR-11 update-loop events captured by this ledger."""

        return len(self._fr11_events)

    def on_fr11_violation(
        self,
        event: ViolationEvent,
        *,
        policy_update_id: str | None = None,
    ) -> None:
        """Record one FR-11 ``ViolationEvent`` delivered by the EventBus."""

        policy_id = str(policy_update_id or event.query_id)
        self._fr11_events.append(
            {
                "constraint_type": event.constraint_type,
                "energy_score": float(event.energy_score),
                "policy_update_id": policy_id,
                "probe_confidence": float(event.probe_confidence),
                "query_id": event.query_id,
                "question_domain": event.question_domain,
                "source": "fr11_event_bus",
                "step_index": int(event.step_index),
                "timestamp": event.timestamp,
            }
        )

    def record_constraint_case(
        self,
        *,
        policy_update_id: str,
        constraint_id: str,
        constraint_type: str,
        source: str,
        baseline_violation: bool,
        promoted_violation: bool,
        accepted_violation: bool,
        false_accept_delta: int,
        soundness_mistake: bool,
    ) -> LedgerRow:
        """Append one normalized REQ-LEARN-1594 certificate row."""

        row = LedgerRow(
            policy_update_id=str(policy_update_id),
            constraint_id=str(constraint_id),
            constraint_type=str(constraint_type),
            source=str(source),
            baseline_violation=bool(baseline_violation),
            promoted_violation=bool(promoted_violation),
            accepted_violation=bool(accepted_violation),
            false_accept_delta=int(false_accept_delta),
            soundness_mistake=bool(soundness_mistake),
            certificate_id=_certificate_id(
                policy_update_id=str(policy_update_id),
                constraint_id=str(constraint_id),
                constraint_type=str(constraint_type),
                source=str(source),
            ),
        )
        self._rows.append(row)
        return row

    def ledger_rows(self) -> list[JsonDict]:
        """Return certificate rows in insertion order."""

        return [row.to_dict() for row in self._rows]

    def fr11_event_rows(self) -> list[JsonDict]:
        """Return raw FR-11 event rows in insertion order."""

        return [dict(row) for row in self._fr11_events]

    def violation_counts_by_type(self) -> dict[str, int]:
        """Summarize captured FR-11 events by constraint type."""

        return dict(Counter(row["constraint_type"] for row in self._fr11_events))

    def violation_counts_by_policy(self) -> dict[str, int]:
        """Summarize captured FR-11 events by policy update id."""

        return dict(Counter(row["policy_update_id"] for row in self._fr11_events))

    def policy_certificates(self) -> list[JsonDict]:
        """Aggregate ledger rows into one promotion certificate per policy."""

        rows_by_policy: dict[str, list[LedgerRow]] = defaultdict(list)
        for row in self._rows:
            rows_by_policy[row.policy_update_id].append(row)

        certificates: list[JsonDict] = []
        event_counts = self.violation_counts_by_policy()
        for policy_id in sorted(rows_by_policy):
            rows = rows_by_policy[policy_id]
            false_delta = sum(row.false_accept_delta for row in rows)
            accepted_count = sum(int(row.accepted_violation) for row in rows)
            soundness_count = sum(int(row.soundness_mistake) for row in rows)
            promotion_safe = bool(
                rows and accepted_count == 0 and soundness_count == 0 and false_delta <= 0
            )
            certificates.append(
                {
                    "accepted_violation_count": accepted_count,
                    "certificate_id": _policy_certificate_id(policy_id, rows),
                    "constraint_count": len(rows),
                    "constraint_ids": sorted(row.constraint_id for row in rows),
                    "false_accept_delta": false_delta,
                    "fr11_events_recorded": int(event_counts.get(policy_id, 0)),
                    "policy_update_id": policy_id,
                    "promotion_safe": promotion_safe,
                    "soundness_mistakes": soundness_count,
                }
            )
        return certificates


def attach_fr11_event_bus(
    bus: FR11EventBus,
    ledger: CerCECertificateLedger,
    *,
    policy_update_id: str | None = None,
) -> None:
    """Subscribe the ledger to the FR-11 EventBus without changing bus semantics."""

    def _subscriber(event: ViolationEvent) -> None:
        ledger.on_fr11_violation(event, policy_update_id=policy_update_id)

    bus.subscribe(_subscriber)


def ingest_promotion_rows(
    ledger: CerCECertificateLedger,
    rows: Sequence[Mapping[str, Any]],
    *,
    source: str = "exp1524_promotion_manifest",
) -> int:
    """Normalize Exp 1524 promotion manifest rows into ledger rows."""

    inserted = 0
    for index, row in enumerate(rows):
        if row.get("row_type") != "policy_promotion_evaluation":
            continue
        policy_id = str(row.get("policy_update_id") or row.get("source_event_id") or "")
        if not policy_id:
            continue
        constraint_id = str(
            row.get("contract_case_id")
            or row.get("source_case_id")
            or row.get("prompt_or_case_id")
            or f"constraint-{index:04d}"
        )
        baseline_violation = _row_false_accept(row, "baseline")
        promoted_violation = _row_false_accept(row, "promoted")
        soundness_mistake = int(row.get("soundness_mistakes", 0) or 0) > 0
        accepted_violation = bool(promoted_violation or _accepted_rejected_contract(row))
        ledger.record_constraint_case(
            policy_update_id=policy_id,
            constraint_id=constraint_id,
            constraint_type=_constraint_type(row, constraint_id),
            source=source,
            baseline_violation=baseline_violation,
            promoted_violation=promoted_violation,
            accepted_violation=accepted_violation,
            false_accept_delta=int(row.get("false_accept_delta", 0) or 0),
            soundness_mistake=soundness_mistake,
        )
        inserted += 1
    return inserted


def write_in_progress_artifact(
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str = RUN_DATE,
) -> JsonDict:
    """REQ-LEARN-1594-1: write the bootstrap artifact before loading inputs."""

    artifact: JsonDict = {
        "status": "in_progress",
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1594", "SCENARIO-LEARN-1594", "SCENARIO-LEARN-1595"],
        "run_date": run_date,
        "project_root": str(project_root),
        "continuous_self_learning_task": True,
        "cerce_ledger_ready": False,
        "policy_certificates_evaluated": 0,
        "constraint_violation_records": 0,
        "fr11_events_recorded": 0,
        "accepted_violation_count": 0,
        "false_accept_delta": 0,
        "nonforgetting_certificate_rate": 0.0,
        "promotion_safe_policy_updates": [],
        "blocked_policy_updates": [],
        "policy_certificates": [],
        "ledger_rows": [],
        "fr11_event_rows": [],
        "fr11_violation_counts_by_type": {},
        "fr11_violation_counts_by_policy": {},
        "blockers": ["cerce_certificate_ledger_in_progress"],
        "honest_verdict": "in_progress",
        "tests_run": [],
    }
    validate_artifact(artifact)
    return _write_json(output_path, artifact)


def build_artifact(
    ledger: CerCECertificateLedger,
    *,
    project_root: Path | str = REPO_ROOT,
    run_date: str | None = None,
    source_artifacts: Sequence[str] | None = None,
    blockers: Sequence[str] | None = None,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Build the terminal CerCE ledger artifact from collected rows."""

    certificates = ledger.policy_certificates()
    safe_ids = sorted(
        str(cert["policy_update_id"]) for cert in certificates if cert["promotion_safe"]
    )
    blocked_ids = sorted(
        str(cert["policy_update_id"]) for cert in certificates if not cert["promotion_safe"]
    )
    accepted_count = sum(int(cert["accepted_violation_count"]) for cert in certificates)
    false_delta = sum(int(cert["false_accept_delta"]) for cert in certificates)
    soundness_count = sum(int(cert["soundness_mistakes"]) for cert in certificates)
    rate = round(len(safe_ids) / len(certificates), 6) if certificates else 0.0

    blocker_set = set(blockers or [])
    if not certificates:
        blocker_set.add("no_policy_certificates")
    if accepted_count:
        blocker_set.add("accepted_constraint_violation")
    if false_delta > 0:
        blocker_set.add("positive_false_accept_delta")
    if soundness_count:
        blocker_set.add("soundness_mistake")

    ready = bool(certificates and rate == 1.0 and not blocker_set)
    artifact: JsonDict = {
        "status": "complete" if ready else "blocked",
        "schema": SCHEMA,
        "spec": ["REQ-LEARN-1594", "SCENARIO-LEARN-1594", "SCENARIO-LEARN-1595"],
        "run_date": run_date or ledger.run_date,
        "project_root": str(project_root),
        "source_artifacts": list(source_artifacts or []),
        "continuous_self_learning_task": True,
        "cerce_ledger_ready": ready,
        "policy_certificates_evaluated": len(certificates),
        "constraint_violation_records": len(ledger.ledger_rows()),
        "fr11_events_recorded": ledger.fr11_events_recorded,
        "accepted_violation_count": accepted_count,
        "false_accept_delta": false_delta,
        "nonforgetting_certificate_rate": rate,
        "promotion_safe_policy_updates": safe_ids,
        "blocked_policy_updates": blocked_ids,
        "policy_certificates": certificates,
        "ledger_rows": ledger.ledger_rows(),
        "fr11_event_rows": ledger.fr11_event_rows(),
        "fr11_violation_counts_by_type": ledger.violation_counts_by_type(),
        "fr11_violation_counts_by_policy": ledger.violation_counts_by_policy(),
        "blockers": sorted(blocker_set),
        "honest_verdict": (
            "complete: cerce_certificate_ledger_ready"
            if ready
            else "complete: cerce_certificate_ledger_blocked"
        ),
        "tests_run": list(tests_run or []),
    }
    validate_artifact(artifact)
    return artifact


def validate_artifact(artifact: Mapping[str, Any]) -> None:
    """Validate the artifact fields used by the conductor and tests."""

    missing = [field for field in REQUIRED_ARTIFACT_FIELDS if field not in artifact]
    if missing:
        raise AssertionError(f"missing required fields: {missing}")
    if artifact["schema"] != SCHEMA:
        raise AssertionError(f"unsupported schema: {artifact['schema']}")
    if artifact["status"] not in {"in_progress", "complete", "blocked"}:
        raise AssertionError(f"unsupported status: {artifact['status']}")
    rate = float(artifact["nonforgetting_certificate_rate"])
    if not 0.0 <= rate <= 1.0:
        raise AssertionError("nonforgetting_certificate_rate must be between 0 and 1")
    if int(artifact["constraint_violation_records"]) != len(artifact["ledger_rows"]):
        raise AssertionError("constraint_violation_records must match ledger_rows")
    if artifact["cerce_ledger_ready"]:
        if artifact["status"] != "complete":
            raise AssertionError("ready ledger must have complete status")
        if artifact["blockers"]:
            raise AssertionError("ready ledger cannot have blockers")
        if int(artifact["accepted_violation_count"]) != 0:
            raise AssertionError("ready ledger requires zero accepted violations")
        if int(artifact["false_accept_delta"]) > 0:
            raise AssertionError("ready ledger cannot increase false accepts")
        if rate != 1.0:
            raise AssertionError("ready ledger requires nonforgetting rate of 1.0")


def run_experiment(
    *,
    project_root: Path | str = REPO_ROOT,
    promotion_manifest_path: Path | str = DEFAULT_PROMOTION_MANIFEST_PATH,
    output_path: Path | str = DEFAULT_OUTPUT_PATH,
    run_date: str = RUN_DATE,
    tests_run: Sequence[str] | None = None,
) -> JsonDict:
    """Run Exp 1594 from the live policy-promotion manifest and write JSON."""

    root = Path(project_root)
    output = _resolve_under_root(root, Path(output_path))
    manifest = _resolve_under_root(root, Path(promotion_manifest_path))
    write_in_progress_artifact(output, project_root=root, run_date=run_date)

    ledger = CerCECertificateLedger(run_date=run_date)
    blockers: list[str] = []
    source_artifacts = [_display_path(manifest, project_root=root)]
    if manifest.exists():
        ingest_promotion_rows(ledger, _read_jsonl(manifest))
    else:
        blockers.append("missing_promotion_manifest")

    artifact = build_artifact(
        ledger,
        project_root=root,
        run_date=run_date,
        source_artifacts=source_artifacts,
        blockers=blockers,
        tests_run=tests_run,
    )
    return _write_json(output, artifact)


def _certificate_id(
    *,
    policy_update_id: str,
    constraint_id: str,
    constraint_type: str,
    source: str,
) -> str:
    payload = json.dumps(
        {
            "constraint_id": constraint_id,
            "constraint_type": constraint_type,
            "policy_update_id": policy_update_id,
            "source": source,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _policy_certificate_id(policy_update_id: str, rows: Sequence[LedgerRow]) -> str:
    payload = json.dumps(
        {
            "policy_update_id": policy_update_id,
            "row_certificate_ids": sorted(row.certificate_id for row in rows),
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _row_false_accept(row: Mapping[str, Any], mode: str) -> bool:
    direct = row.get(f"{mode}_false_accept")
    nested = row.get("runtime_contract_validation")
    if isinstance(nested, Mapping):
        mode_row = nested.get(mode)
        if isinstance(mode_row, Mapping) and mode_row.get("false_accept") is True:
            return True
    return bool(direct)


def _accepted_rejected_contract(row: Mapping[str, Any]) -> bool:
    nested = row.get("runtime_contract_validation")
    if not isinstance(nested, Mapping):
        return False
    promoted = nested.get("promoted")
    if not isinstance(promoted, Mapping):
        return False
    return (
        promoted.get("expected_label") is False
        and promoted.get("proposed_final_deterministic_accept") is True
    )


def _constraint_type(row: Mapping[str, Any], constraint_id: str) -> str:
    source_family = str(row.get("source_family") or "").strip()
    if source_family:
        return source_family
    if ":" in constraint_id:
        return constraint_id.split(":", 1)[0]
    return "runtime_contract"


def _read_jsonl(path: Path | str) -> list[JsonDict]:
    rows: list[JsonDict] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, Mapping):
            rows.append(dict(payload))
    return rows


def _write_json(path: Path | str, payload: Mapping[str, Any]) -> JsonDict:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    destination.write_text(
        json.dumps(serializable, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return serializable


def _display_path(path: Path | str, *, project_root: Path | str = REPO_ROOT) -> str:
    target = Path(path)
    try:
        return target.relative_to(Path(project_root)).as_posix()
    except ValueError:
        return target.as_posix()


def _resolve_under_root(root: Path, path: Path) -> Path:
    if path.is_absolute():
        return path
    return root / path


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint used by the conductor to write the Exp 1594 artifact."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=str(REPO_ROOT))
    parser.add_argument("--promotion-manifest", default=str(DEFAULT_PROMOTION_MANIFEST_PATH))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args(argv)
    artifact = run_experiment(
        project_root=Path(args.project_root),
        promotion_manifest_path=Path(args.promotion_manifest),
        output_path=Path(args.output),
    )
    print(artifact["honest_verdict"])
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
