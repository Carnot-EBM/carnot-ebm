"""Deterministic self-learning policy compilation from accepted repairs and cases.

Spec: REQ-VERIFY-052, REQ-VERIFY-053,
SCENARIO-VERIFY-056, SCENARIO-VERIFY-057, SCENARIO-VERIFY-058,
SCENARIO-VERIFY-059
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from carnot.pipeline.case_memory import CaseEntry, CaseMemory, CaseQuery, CaseRecord

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from carnot.pipeline.case_memory import CaseMatch
    from carnot.pipeline.tracker import ConstraintTracker


VERSION = 1
RUN_DATE = "20260413"
POLICY_OUTPUT = Path("results/self_learning_policy_240.json")


def _clean_text(value: object) -> str:
    return " ".join(str(value).strip().split())


def _clean_multiline(value: object) -> str:
    lines = [str(line).strip() for line in str(value).splitlines() if str(line).strip()]
    return "\n".join(lines)


def _as_strings(value: object) -> tuple[str, ...]:
    if not isinstance(value, (list, tuple)):
        return ()
    items = [str(item).strip() for item in value if str(item).strip()]
    return tuple(dict.fromkeys(items))


def _id_fragment(value: str) -> str:
    return value.strip().lower()


def _primary_signal(entry: CaseEntry) -> str:
    if entry.key.violation_families:
        return entry.key.violation_families[0]
    if entry.violation_types:
        return entry.violation_types[0]
    return "generic"


def _weighted_mean(values: Iterable[tuple[int, float]]) -> float:
    total_weight = 0
    total_value = 0.0
    for weight, value in values:
        total_weight += weight
        total_value += weight * value
    if total_weight == 0:
        return 0.0
    return total_value / total_weight


@dataclass(frozen=True)
class PolicyProvenance:
    """One provenance record attached to a compiled policy update."""

    source_type: str
    source_experiment: int | None
    source_artifact: str | None
    case_id: str
    support: int
    confidence: float
    iteration: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_type": self.source_type,
            "source_experiment": self.source_experiment,
            "source_artifact": self.source_artifact,
            "case_id": self.case_id,
            "support": self.support,
            "confidence": self.confidence,
            "iteration": self.iteration,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PolicyProvenance:
        source_experiment = payload.get("source_experiment")
        iteration = payload.get("iteration")
        return cls(
            source_type=str(payload.get("source_type") or ""),
            source_experiment=int(source_experiment) if source_experiment is not None else None,
            source_artifact=(
                str(payload.get("source_artifact"))
                if payload.get("source_artifact") is not None
                else None
            ),
            case_id=str(payload.get("case_id") or ""),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            iteration=int(iteration) if iteration is not None else None,
        )


@dataclass(frozen=True)
class ThresholdOverride:
    """A deterministic verifier-threshold update compiled from case evidence."""

    update_id: str
    model_name: str
    benchmark_slice: str
    verifier_name: str
    threshold_name: str
    baseline_value: float
    threshold_value: float
    violation_signals: tuple[str, ...]
    support: int
    confidence: float
    provenance: tuple[PolicyProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "model_name": self.model_name,
            "benchmark_slice": self.benchmark_slice,
            "verifier_name": self.verifier_name,
            "threshold_name": self.threshold_name,
            "baseline_value": self.baseline_value,
            "threshold_value": self.threshold_value,
            "violation_signals": list(self.violation_signals),
            "support": self.support,
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ThresholdOverride:
        return cls(
            update_id=str(payload.get("update_id") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            verifier_name=str(payload.get("verifier_name") or ""),
            threshold_name=str(payload.get("threshold_name") or ""),
            baseline_value=float(payload.get("baseline_value") or 0.0),
            threshold_value=float(payload.get("threshold_value") or 0.0),
            violation_signals=tuple(str(item) for item in payload.get("violation_signals", [])),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=tuple(
                PolicyProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class PropertyBudgetUpdate:
    """A deterministic property-priority budget for code verification."""

    update_id: str
    model_name: str
    benchmark_slice: str
    budget: int
    property_names: tuple[str, ...]
    support: int
    confidence: float
    provenance: tuple[PolicyProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "model_name": self.model_name,
            "benchmark_slice": self.benchmark_slice,
            "budget": self.budget,
            "property_names": list(self.property_names),
            "support": self.support,
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> PropertyBudgetUpdate:
        return cls(
            update_id=str(payload.get("update_id") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            budget=int(payload.get("budget") or 0),
            property_names=tuple(str(item) for item in payload.get("property_names", [])),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=tuple(
                PolicyProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class RepairPromptPatch:
    """A reusable repair-prompt patch compiled from accepted repairs."""

    update_id: str
    model_names: tuple[str, ...]
    benchmark_slice: str
    trigger_error_type: str
    prompt_patch: str
    support: int
    success_rate: float
    provenance: tuple[PolicyProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "model_names": list(self.model_names),
            "benchmark_slice": self.benchmark_slice,
            "trigger_error_type": self.trigger_error_type,
            "prompt_patch": self.prompt_patch,
            "support": self.support,
            "success_rate": self.success_rate,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RepairPromptPatch:
        return cls(
            update_id=str(payload.get("update_id") or ""),
            model_names=tuple(str(item) for item in payload.get("model_names", [])),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            trigger_error_type=str(payload.get("trigger_error_type") or ""),
            prompt_patch=str(payload.get("prompt_patch") or ""),
            support=int(payload.get("support") or 0),
            success_rate=float(payload.get("success_rate") or 0.0),
            provenance=tuple(
                PolicyProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class RoutingHint:
    """A deterministic routing hint compiled from high-precision cases."""

    update_id: str
    model_name: str
    benchmark_slice: str
    violation_signals: tuple[str, ...]
    route_to: str
    reason: str
    support: int
    confidence: float
    provenance: tuple[PolicyProvenance, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "update_id": self.update_id,
            "model_name": self.model_name,
            "benchmark_slice": self.benchmark_slice,
            "violation_signals": list(self.violation_signals),
            "route_to": self.route_to,
            "reason": self.reason,
            "support": self.support,
            "confidence": self.confidence,
            "provenance": [item.to_dict() for item in self.provenance],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> RoutingHint:
        return cls(
            update_id=str(payload.get("update_id") or ""),
            model_name=str(payload.get("model_name") or ""),
            benchmark_slice=str(payload.get("benchmark_slice") or ""),
            violation_signals=tuple(str(item) for item in payload.get("violation_signals", [])),
            route_to=str(payload.get("route_to") or ""),
            reason=str(payload.get("reason") or ""),
            support=int(payload.get("support") or 0),
            confidence=float(payload.get("confidence") or 0.0),
            provenance=tuple(
                PolicyProvenance.from_dict(item)
                for item in payload.get("provenance", [])
                if isinstance(item, dict)
            ),
        )


@dataclass(frozen=True)
class PolicyQuery:
    """Query shape used to resolve matching policy updates."""

    model_name: str
    benchmark_slice: str
    violation_types: tuple[str, ...]
    violation_families: tuple[str, ...]
    prompt_sketch: str
    prompt_tokens: tuple[str, ...]
    property_names: tuple[str, ...]
    preferred_repair_outcome: str | None = None

    @classmethod
    def from_record(
        cls,
        record: CaseRecord,
        *,
        preferred_repair_outcome: str | None = None,
    ) -> PolicyQuery:
        return cls(
            model_name=record.model_name,
            benchmark_slice=record.benchmark_slice,
            violation_types=record.violation_types,
            violation_families=record.violation_families,
            prompt_sketch=record.prompt_sketch,
            prompt_tokens=record.prompt_tokens,
            property_names=record.property_names,
            preferred_repair_outcome=preferred_repair_outcome,
        )

    def to_case_query(self) -> CaseQuery:
        return CaseQuery(
            model_name=self.model_name,
            benchmark_slice=self.benchmark_slice,
            violation_types=self.violation_types,
            violation_families=self.violation_families,
            prompt_sketch=self.prompt_sketch,
            prompt_tokens=self.prompt_tokens,
            property_names=self.property_names,
            preferred_repair_outcome=self.preferred_repair_outcome,
        )


@dataclass(frozen=True)
class RuntimePolicyContext:
    """Additive runtime view over tracker stats, case memory, and policy hits."""

    tracker_stats: dict[str, dict[str, Any]]
    case_matches: tuple[CaseMatch, ...] = ()
    threshold_overrides: tuple[ThresholdOverride, ...] = ()
    property_budget_updates: tuple[PropertyBudgetUpdate, ...] = ()
    repair_prompt_patches: tuple[RepairPromptPatch, ...] = ()
    routing_hints: tuple[RoutingHint, ...] = ()


@dataclass(frozen=True)
class SelfLearningPolicy:
    """Compiled self-learning policy artifact."""

    run_date: str = RUN_DATE
    threshold_overrides: tuple[ThresholdOverride, ...] = ()
    property_budget_updates: tuple[PropertyBudgetUpdate, ...] = ()
    repair_prompt_patches: tuple[RepairPromptPatch, ...] = ()
    routing_hints: tuple[RoutingHint, ...] = ()

    def _summary(self) -> dict[str, int]:
        return {
            "n_threshold_overrides": len(self.threshold_overrides),
            "n_property_budget_updates": len(self.property_budget_updates),
            "n_repair_prompt_patches": len(self.repair_prompt_patches),
            "n_routing_hints": len(self.routing_hints),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": VERSION,
            "experiment": 240,
            "run_date": self.run_date,
            "title": "Compiled self-learning policy",
            "summary": self._summary(),
            "threshold_overrides": [item.to_dict() for item in self.threshold_overrides],
            "property_budget_updates": [item.to_dict() for item in self.property_budget_updates],
            "repair_prompt_patches": [item.to_dict() for item in self.repair_prompt_patches],
            "routing_hints": [item.to_dict() for item in self.routing_hints],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SelfLearningPolicy:
        version = payload.get("version")
        if version is not None and int(version) != VERSION:
            raise ValueError(
                f"Unsupported self-learning policy format (expected version={VERSION})"
            )
        return cls(
            run_date=str(payload.get("run_date") or RUN_DATE),
            threshold_overrides=tuple(
                ThresholdOverride.from_dict(item)
                for item in payload.get("threshold_overrides", [])
                if isinstance(item, dict)
            ),
            property_budget_updates=tuple(
                PropertyBudgetUpdate.from_dict(item)
                for item in payload.get("property_budget_updates", [])
                if isinstance(item, dict)
            ),
            repair_prompt_patches=tuple(
                RepairPromptPatch.from_dict(item)
                for item in payload.get("repair_prompt_patches", [])
                if isinstance(item, dict)
            ),
            routing_hints=tuple(
                RoutingHint.from_dict(item)
                for item in payload.get("routing_hints", [])
                if isinstance(item, dict)
            ),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, payload: str) -> SelfLearningPolicy:
        raw = json.loads(payload)
        if not isinstance(raw, dict):
            raise ValueError("Self-learning policy payload must be a JSON object")
        return cls.from_dict(raw)

    def save(self, path: str | Path = POLICY_OUTPUT) -> None:
        resolved = Path(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(self.to_json(), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path = POLICY_OUTPUT) -> SelfLearningPolicy:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("Self-learning policy payload must be a JSON object")
        return cls.from_dict(raw)

    def runtime_context(
        self,
        query: PolicyQuery,
        *,
        tracker: ConstraintTracker | None = None,
        case_memory: CaseMemory | None = None,
    ) -> RuntimePolicyContext:
        tracker_stats = tracker.stats() if tracker is not None else {}
        case_matches: tuple[CaseMatch, ...] = ()
        if case_memory is not None:
            case_matches = tuple(case_memory.retrieve(query.to_case_query(), limit=5))

        return RuntimePolicyContext(
            tracker_stats=tracker_stats,
            case_matches=case_matches,
            threshold_overrides=self._match_threshold_overrides(query),
            property_budget_updates=self._match_property_budgets(query),
            repair_prompt_patches=self._match_repair_patches(query),
            routing_hints=self._match_routing_hints(query),
        )

    def _match_threshold_overrides(self, query: PolicyQuery) -> tuple[ThresholdOverride, ...]:
        matches = [
            item
            for item in self.threshold_overrides
            if item.benchmark_slice == query.benchmark_slice
            and item.model_name == query.model_name
            and bool(
                set(item.violation_signals)
                & set((*query.violation_families, *query.violation_types))
            )
        ]
        return tuple(sorted(matches, key=lambda item: item.update_id))

    def _match_property_budgets(self, query: PolicyQuery) -> tuple[PropertyBudgetUpdate, ...]:
        matches = [
            item
            for item in self.property_budget_updates
            if item.benchmark_slice == query.benchmark_slice
            and item.model_name == query.model_name
            and (
                not query.property_names
                or bool(set(item.property_names) & set(query.property_names))
            )
        ]
        return tuple(sorted(matches, key=lambda item: item.update_id))

    def _match_repair_patches(self, query: PolicyQuery) -> tuple[RepairPromptPatch, ...]:
        query_signals = set((*query.violation_types, *query.violation_families))
        matches = [
            item
            for item in self.repair_prompt_patches
            if item.benchmark_slice == query.benchmark_slice
            and query.model_name in item.model_names
            and item.trigger_error_type in query_signals
        ]
        return tuple(sorted(matches, key=lambda item: item.update_id))

    def _match_routing_hints(self, query: PolicyQuery) -> tuple[RoutingHint, ...]:
        query_signals = set((*query.violation_families, *query.violation_types))
        matches = [
            item
            for item in self.routing_hints
            if item.benchmark_slice == query.benchmark_slice
            and item.model_name == query.model_name
            and bool(set(item.violation_signals) & query_signals)
        ]
        return tuple(sorted(matches, key=lambda item: item.update_id))


class SelfLearningPolicyCompiler:
    """Compile deterministic policy updates from case memory and accepted repairs."""

    def __init__(
        self,
        *,
        min_case_support: int = 2,
        min_case_confidence: float = 0.9,
        min_patch_support: int = 2,
    ) -> None:
        self._min_case_support = min_case_support
        self._min_case_confidence = min_case_confidence
        self._min_patch_support = min_patch_support

    def compile(
        self,
        *,
        case_memory: CaseMemory | None,
        accepted_repairs: Iterable[Mapping[str, Any] | dict[str, Any]] = (),
        tracker: ConstraintTracker | None = None,
    ) -> SelfLearningPolicy:
        tracker_stats = tracker.stats() if tracker is not None else {}
        entries = case_memory.entries() if case_memory is not None else ()
        qualifying_entries = tuple(
            entry
            for entry in entries
            if (
                entry.support >= self._min_case_support
                and entry.confidence >= self._min_case_confidence
            )
        )

        threshold_overrides = self._compile_threshold_overrides(qualifying_entries, tracker_stats)
        property_budget_updates = self._compile_property_budgets(qualifying_entries)
        repair_prompt_patches = self._compile_repair_patches(tuple(accepted_repairs))
        routing_hints = self._compile_routing_hints(qualifying_entries)

        return SelfLearningPolicy(
            threshold_overrides=threshold_overrides,
            property_budget_updates=property_budget_updates,
            repair_prompt_patches=repair_prompt_patches,
            routing_hints=routing_hints,
        )

    def _compile_threshold_overrides(
        self,
        entries: tuple[CaseEntry, ...],
        tracker_stats: dict[str, dict[str, Any]],
    ) -> tuple[ThresholdOverride, ...]:
        updates: list[ThresholdOverride] = []
        for entry in entries:
            if entry.key.repair_outcome not in {"improved", "regressed"}:
                continue

            signal = _primary_signal(entry)
            stats = tracker_stats.get(signal, {})
            tracker_precision = float(stats.get("precision") or 0.0)
            verifier_name = (
                "semantic_verifier_v2"
                if entry.case_kind == "semantic_verification"
                else "verify_repair"
            )
            baseline_value = 0.7 if entry.case_kind == "semantic_verification" else 0.6
            if entry.key.repair_outcome == "improved":
                adjustment = 0.05 if tracker_precision < 0.75 else 0.1
                threshold_value = max(0.0, baseline_value - adjustment)
            else:
                threshold_value = min(1.0, baseline_value + 0.05)

            updates.append(
                ThresholdOverride(
                    update_id=(
                        f"threshold:{_id_fragment(entry.key.model_name)}:"
                        f"{entry.key.benchmark_slice}:{signal}"
                    ),
                    model_name=entry.key.model_name,
                    benchmark_slice=entry.key.benchmark_slice,
                    verifier_name=verifier_name,
                    threshold_name="repair_trigger_threshold",
                    baseline_value=baseline_value,
                    threshold_value=threshold_value,
                    violation_signals=(signal,),
                    support=entry.support,
                    confidence=entry.confidence,
                    provenance=self._provenance_from_case_entry(entry),
                )
            )
        return tuple(sorted(updates, key=lambda item: item.update_id))

    def _compile_property_budgets(
        self,
        entries: tuple[CaseEntry, ...],
    ) -> tuple[PropertyBudgetUpdate, ...]:
        grouped: dict[tuple[str, str], list[CaseEntry]] = {}
        for entry in entries:
            if not entry.key.property_names or entry.key.repair_outcome != "improved":
                continue
            grouped.setdefault((entry.key.model_name, entry.key.benchmark_slice), []).append(entry)

        updates: list[PropertyBudgetUpdate] = []
        for key in sorted(grouped):
            model_name, benchmark_slice = key
            group = grouped[key]
            counts: dict[str, int] = {}
            for entry in group:
                for property_name in entry.key.property_names:
                    counts[property_name] = counts.get(property_name, 0) + entry.support
            ordered_properties = tuple(sorted(counts, key=lambda item: (-counts[item], item)))
            support = sum(entry.support for entry in group)
            confidence = _weighted_mean((entry.support, entry.confidence) for entry in group)
            provenance: list[PolicyProvenance] = []
            for entry in sorted(group, key=lambda item: item.key.fingerprint):
                provenance.extend(self._provenance_from_case_entry(entry))
            updates.append(
                PropertyBudgetUpdate(
                    update_id=(f"property_budget:{_id_fragment(model_name)}:{benchmark_slice}"),
                    model_name=model_name,
                    benchmark_slice=benchmark_slice,
                    budget=min(3, len(ordered_properties)),
                    property_names=ordered_properties,
                    support=support,
                    confidence=confidence,
                    provenance=tuple(provenance),
                )
            )
        return tuple(sorted(updates, key=lambda item: item.update_id))

    def _compile_repair_patches(
        self,
        accepted_repairs: tuple[Mapping[str, Any] | dict[str, Any], ...],
    ) -> tuple[RepairPromptPatch, ...]:
        updates: list[RepairPromptPatch] = []
        for row in accepted_repairs:
            support = int(row.get("support") or 0)
            template = _clean_multiline(row.get("template") or "")
            if support < self._min_patch_support or not template:
                continue

            benchmark = _clean_text(row.get("benchmark") or "")
            domain = _clean_text(row.get("domain") or "")
            benchmark_slice = benchmark if not domain else f"{benchmark}/{domain}"
            model_names = _as_strings(row.get("model_names"))
            if not model_names:
                continue
            successful_cases = int(row.get("successful_cases") or 0)
            failed_cases = int(row.get("failed_cases") or 0)
            total_cases = successful_cases + failed_cases
            success_rate = successful_cases / total_cases if total_cases else 0.0
            trigger_error_type = _clean_text(row.get("trigger_error_type") or "")
            provenance = self._provenance_from_repair_row(
                row,
                support=support,
                confidence=success_rate,
            )

            for model_name in model_names:
                updates.append(
                    RepairPromptPatch(
                        update_id=(
                            f"repair_patch:{_id_fragment(model_name)}:"
                            f"{benchmark_slice}:{trigger_error_type}"
                        ),
                        model_names=(model_name,),
                        benchmark_slice=benchmark_slice,
                        trigger_error_type=trigger_error_type,
                        prompt_patch=template,
                        support=support,
                        success_rate=success_rate,
                        provenance=provenance,
                    )
                )
        return tuple(sorted(updates, key=lambda item: item.update_id))

    def _compile_routing_hints(
        self,
        entries: tuple[CaseEntry, ...],
    ) -> tuple[RoutingHint, ...]:
        updates: list[RoutingHint] = []
        for entry in entries:
            signal = _primary_signal(entry)
            if entry.key.repair_outcome == "improved":
                route_to = "case_memory_then_repair"
                reason = "high_precision_improved_case"
            elif entry.key.repair_outcome == "regressed":
                route_to = "verify_only"
                reason = "repair_regression_case"
            else:
                continue
            updates.append(
                RoutingHint(
                    update_id=(
                        f"routing:{_id_fragment(entry.key.model_name)}:"
                        f"{entry.key.benchmark_slice}:{signal}"
                    ),
                    model_name=entry.key.model_name,
                    benchmark_slice=entry.key.benchmark_slice,
                    violation_signals=(signal,),
                    route_to=route_to,
                    reason=reason,
                    support=entry.support,
                    confidence=entry.confidence,
                    provenance=self._provenance_from_case_entry(entry),
                )
            )
        return tuple(sorted(updates, key=lambda item: item.update_id))

    @staticmethod
    def _provenance_from_case_entry(entry: CaseEntry) -> tuple[PolicyProvenance, ...]:
        return tuple(
            PolicyProvenance(
                source_type="case_memory",
                source_experiment=item.source_experiment,
                source_artifact=item.source_artifact,
                case_id=item.case_id,
                support=entry.support,
                confidence=entry.confidence,
            )
            for item in entry.provenance
        )

    @staticmethod
    def _provenance_from_repair_row(
        row: Mapping[str, Any] | dict[str, Any],
        *,
        support: int,
        confidence: float,
    ) -> tuple[PolicyProvenance, ...]:
        provenance: list[PolicyProvenance] = []
        raw_items = row.get("provenance")
        if not isinstance(raw_items, list):
            return ()
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            source_experiment = item.get("source_experiment")
            iteration = item.get("iteration")
            provenance.append(
                PolicyProvenance(
                    source_type="repair_snippet",
                    source_experiment=(
                        int(source_experiment) if source_experiment is not None else None
                    ),
                    source_artifact=(
                        str(item.get("source_artifact"))
                        if item.get("source_artifact") is not None
                        else None
                    ),
                    case_id=str(item.get("case_id") or ""),
                    support=support,
                    confidence=confidence,
                    iteration=int(iteration) if iteration is not None else None,
                )
            )
        return tuple(provenance)


__all__ = [
    "POLICY_OUTPUT",
    "RUN_DATE",
    "PolicyProvenance",
    "ThresholdOverride",
    "PropertyBudgetUpdate",
    "RepairPromptPatch",
    "RoutingHint",
    "PolicyQuery",
    "RuntimePolicyContext",
    "SelfLearningPolicy",
    "SelfLearningPolicyCompiler",
]
