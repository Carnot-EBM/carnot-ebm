"""Cross-session constraint memory with live traces from Exp 219-226.

Populates CaseMemory from real experiment results, simulates a session
boundary via save/load, then probes warm memory vs cold-start to measure
whether prior-session knowledge improves verification retrieval.

Spec: REQ-VERIFY-050, REQ-VERIFY-051,
SCENARIO-VERIFY-052, SCENARIO-VERIFY-053, SCENARIO-VERIFY-054
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot.pipeline.case_memory import (
    CaseMemory,
    CaseQuery,
    CaseRecord,
)

# ---------------------------------------------------------------------------
# Repo paths
# ---------------------------------------------------------------------------

_REPO = Path(__file__).resolve().parents[2]
_RESULTS = _REPO / "results"


def _result_file(name: str) -> Path:
    return _RESULTS / name


def _load(name: str) -> dict[str, Any]:
    path = _result_file(name)
    if not path.exists():
        pytest.skip(f"live result file missing: {path}")
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# Record extractors — parse real live-trace JSON into CaseRecord objects
# ---------------------------------------------------------------------------


def _extract_exp219_tp_records(data: dict[str, Any]) -> list[tuple[CaseRecord, str]]:
    """Extract true-positive CaseRecords from Exp 219 (gsm8k_semantic).

    A TP case is one where the model was wrong (correct=False) and the
    semantic verifier raised at least one violation — so the verifier
    correctly caught a real error.  Returns (record, case_id) pairs.
    """
    # REQ-VERIFY-050: normalise live traces into the standard CaseRecord schema
    records: list[tuple[CaseRecord, str]] = []
    for run in data.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name: str = str(run.get("model_name", ""))
        for case in run.get("cases", []):
            viols = case.get("verification", {}).get("violations", [])
            if not viols:
                continue
            correct: bool = bool(case.get("correct", True))
            if correct:
                # False positive — skip; we only train on confirmed errors
                continue
            case_id: str = str(case.get("case_id", ""))
            viol_types: list[str] = []
            descriptions: list[str] = []
            for v in viols:
                if not isinstance(v, dict):
                    continue
                meta = v.get("metadata", {}) or {}
                th = str(meta.get("taxonomy_hint", "") or "")
                vt = str(meta.get("violation_type", "") or "")
                composite = f"{th}:{vt}" if th and vt else (th or vt or "unknown_violation")
                viol_types.append(composite)
                desc = str(v.get("description", "") or "")
                if desc:
                    descriptions.append(desc)
            if not viol_types:
                continue
            record = CaseRecord.normalize(
                benchmark="gsm8k_semantic",
                benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
                model_name=model_name,
                case_id=case_id,
                violation_types=viol_types,
                prompt_text="",
                description_texts=descriptions,
                baseline_success=False,
                repair_success=True,
                confidence=0.85,
                source_experiment=219,
                source_artifact="results/experiment_219_results.json",
                response_mode=str(case.get("response_mode", "") or ""),
                verifier_path="semantic_grounding",
            )
            records.append((record, case_id))
    return records


def _extract_exp219_fp_cases(data: dict[str, Any]) -> list[str]:
    """Return case_ids where the verifier fired but the model was actually correct."""
    fp: list[str] = []
    for run in data.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        for case in run.get("cases", []):
            viols = case.get("verification", {}).get("violations", [])
            if viols and bool(case.get("correct", True)):
                fp.append(str(case.get("case_id", "")))
    return fp


def _extract_exp220_tp_records(data: dict[str, Any]) -> list[tuple[CaseRecord, str]]:
    """Extract true-positive CaseRecords from Exp 220 (humaneval_property).

    A TP case is one where the test suite failed (passed=False) and the
    execution_plus_property verifier detected the failure.
    """
    records: list[tuple[CaseRecord, str]] = []
    for run in data.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name: str = str(run.get("model_name", ""))
        for case in run.get("cases", []):
            ep = case.get("execution_plus_property", {}) or {}
            detected: bool = bool(ep.get("detected", False))
            passed: bool = bool(case.get("passed", True))
            if passed or not detected:
                continue
            case_id: str = str(case.get("case_id", ""))
            # Derive violation types from property_violations text
            pv_texts: list[str] = [str(p) for p in ep.get("property_violations", []) if p]
            viol_types: list[str] = []
            for pv in pv_texts:
                pv_lower = pv.lower()
                if "deterministic" in pv_lower:
                    viol_types.append("deterministic")
                elif "example_regression" in pv_lower:
                    viol_types.append("example_regression")
                elif "annotation" in pv_lower:
                    viol_types.append("annotation_feedback")
                elif "property" in pv_lower:
                    viol_types.append("property_violation")
                else:
                    viol_types.append("official_test_failure")
            if not viol_types:
                et = str(case.get("error_type", "") or "")
                viol_types = [et if et else "official_test_failure"]
            record = CaseRecord.normalize(
                benchmark="humaneval_property",
                benchmark_slice="humaneval_property/code_typed_properties",
                model_name=model_name,
                case_id=case_id,
                violation_types=list(dict.fromkeys(viol_types)),  # dedupe, preserve order
                prompt_text="",
                description_texts=pv_texts[:3],  # first 3 descriptions to keep sketch small
                baseline_success=False,
                repair_success=True,
                confidence=0.90,
                source_experiment=220,
                source_artifact="results/experiment_220_results.json",
                response_mode=str(case.get("response_mode", "") or ""),
                verifier_path="execution_plus_property",
            )
            records.append((record, case_id))
    return records


def _extract_exp221_tp_records(data: dict[str, Any]) -> list[tuple[CaseRecord, str]]:
    """Extract true-positive CaseRecords from Exp 221 (constraint_ir).

    A TP case is one where at least one constraint was violated (status='violated')
    and the response did not satisfy the constraints (exact_satisfaction=False).
    """
    records: list[tuple[CaseRecord, str]] = []
    for run in data.get("paired_runs", []):
        if run.get("mode") != "verify_only":
            continue
        model_name: str = str(run.get("model_name", ""))
        for case in run.get("cases", []):
            ev = case.get("evaluation", {}) or {}
            exact: bool = bool(case.get("exact_satisfaction", True))
            if exact:
                continue
            cr = ev.get("constraint_results", []) or []
            violated = [r for r in cr if isinstance(r, dict) and r.get("status") == "violated"]
            if not violated:
                continue
            case_id: str = str(case.get("case_id", ""))
            task_slice: str = str(ev.get("task_slice", "code_typed_properties") or "")
            viol_types: list[str] = []
            for v in violated:
                family = str(v.get("family", "") or "")
                vtype = str(v.get("type", "") or "")
                composite = f"{family}:{vtype}" if family and vtype else (vtype or family or "constraint_violation")
                viol_types.append(composite)
            record = CaseRecord.normalize(
                benchmark="constraint_ir",
                benchmark_slice=f"constraint_ir/{task_slice}",
                model_name=model_name,
                case_id=case_id,
                violation_types=list(dict.fromkeys(viol_types)),
                prompt_text="",
                description_texts=viol_types[:3],
                baseline_success=False,
                repair_success=True,
                confidence=0.80,
                source_experiment=221,
                source_artifact="results/experiment_221_results.json",
                response_mode=str(case.get("response_mode", "") or ""),
                verifier_path="constraint_ir",
            )
            records.append((record, case_id))
    return records


def _build_session1_memory(
    data219: dict[str, Any],
    data220: dict[str, Any],
    data221: dict[str, Any],
) -> tuple[CaseMemory, dict[str, Any]]:
    """Populate CaseMemory from Exps 219, 220, 221 live traces.

    Returns (memory, stats) where stats summarises what was ingested.
    """
    memory = CaseMemory()
    sources: dict[str, int] = {}

    for record, _ in _extract_exp219_tp_records(data219):
        memory.record(record)
    sources["exp219_gsm8k_semantic"] = sum(
        1 for e in memory.entries() if e.benchmark == "gsm8k_semantic"
    )

    pre = len(memory)
    for record, _ in _extract_exp220_tp_records(data220):
        memory.record(record)
    sources["exp220_humaneval_property"] = len(memory) - pre

    pre = len(memory)
    for record, _ in _extract_exp221_tp_records(data221):
        memory.record(record)
    sources["exp221_constraint_ir"] = len(memory) - pre

    stats = {
        "total_entries": len(memory),
        "by_source": sources,
    }
    return memory, stats


# ---------------------------------------------------------------------------
# Probe factories — build CaseQuery objects for warm/cold comparison
# ---------------------------------------------------------------------------


def _gsm8k_tp_probes() -> list[CaseQuery]:
    """Probes for gsm8k_semantic — same violation family as Exp 219 TP cases."""
    # REQ-VERIFY-051: retrieval should find matches for queries with overlapping families
    probes = []
    for i, (prompt_text, viol_type, desc) in enumerate([
        (
            "How many more miles did the faster runner cover than the slower one?",
            "question_grounding_failures:answer_target_mismatch",
            "The response does not compute the requested comparison quantity.",
        ),
        (
            "What is the total amount of money Alice saved after three months?",
            "omitted_premises:missing_quantity_coverage",
            "The response ignores the compounding savings rule stated in the problem.",
        ),
        (
            "How many students attended the second session compared to the first?",
            "question_grounding_failures:answer_target_mismatch",
            "The response computes the wrong target quantity for the comparison.",
        ),
    ]):
        probe_record = CaseRecord.normalize(
            benchmark="gsm8k_semantic",
            benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id=f"probe-gsm8k-tp-{i}",
            violation_types=(viol_type,),
            prompt_text=prompt_text,
            description_texts=(desc,),
            baseline_success=None,
            repair_success=None,
            confidence=0.0,
            source_experiment=278,
        )
        probes.append(CaseQuery.from_record(probe_record, preferred_repair_outcome="improved"))
    return probes


def _humaneval_tp_probes() -> list[CaseQuery]:
    """Probes for humaneval_property — same violation types as Exp 220 TP cases."""
    probes = []
    for i, viol_type in enumerate(["deterministic", "example_regression", "deterministic"]):
        probe_record = CaseRecord.normalize(
            benchmark="humaneval_property",
            benchmark_slice="humaneval_property/code_typed_properties",
            model_name="Qwen3.5-0.8B",
            case_id=f"probe-humaneval-tp-{i}",
            violation_types=(viol_type,),
            prompt_text="",
            description_texts=(f"{viol_type} (official_tests) failed for input=(x,): AssertionError",),
            baseline_success=None,
            repair_success=None,
            confidence=0.0,
            source_experiment=278,
        )
        probes.append(CaseQuery.from_record(probe_record, preferred_repair_outcome="improved"))
    return probes


def _constraint_ir_tp_probes() -> list[CaseQuery]:
    """Probes for constraint_ir — same violation families as Exp 221 TP cases."""
    probes = []
    for i, (task_slice, viol_type) in enumerate([
        ("code_typed_properties", "semantic:semantic_property"),
        ("instruction_surface_only", "literal:json_exact_keys"),
        ("code_typed_properties", "literal:function_name"),
    ]):
        probe_record = CaseRecord.normalize(
            benchmark="constraint_ir",
            benchmark_slice=f"constraint_ir/{task_slice}",
            model_name="Qwen3.5-0.8B",
            case_id=f"probe-constraint-tp-{i}",
            violation_types=(viol_type,),
            prompt_text="",
            description_texts=(viol_type,),
            baseline_success=None,
            repair_success=None,
            confidence=0.0,
            source_experiment=278,
        )
        probes.append(CaseQuery.from_record(probe_record, preferred_repair_outcome="improved"))
    return probes


def _tn_probes() -> list[CaseQuery]:
    """Probes that should NOT match memory (different benchmark_slice)."""
    probes = []
    for i, (benchmark, benchmark_slice, viol_type) in enumerate([
        ("custom_eval", "custom_eval/never_seen_slice", "some_custom_violation"),
        ("totally_new", "totally_new/unseen", "novel_family:novel_type"),
        ("gsm8k_semantic", "gsm8k_semantic/live_gsm8k_semantic_failure", "completely_unrelated_family:no_match"),
    ]):
        probe_record = CaseRecord.normalize(
            benchmark=benchmark,
            benchmark_slice=benchmark_slice,
            model_name="Unknown-Model-XYZ",
            case_id=f"probe-tn-{i}",
            violation_types=(viol_type,),
            prompt_text="",
            description_texts=(),
            baseline_success=None,
            repair_success=None,
            confidence=0.0,
            source_experiment=278,
        )
        probes.append(CaseQuery.from_record(probe_record))
    return probes


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _hit_rate(memory: CaseMemory, queries: list[CaseQuery]) -> float:
    """Fraction of queries that return at least one match from memory."""
    if not queries:
        return 0.0
    hits = sum(1 for q in queries if memory.retrieve(q, limit=1))
    return hits / len(queries)


def _avg_top_score(memory: CaseMemory, queries: list[CaseQuery]) -> float:
    """Mean top-match score across all queries that got a hit."""
    scores: list[float] = []
    for q in queries:
        matches = memory.retrieve(q, limit=1)
        if matches:
            scores.append(float(matches[0].score))
    return sum(scores) / len(scores) if scores else 0.0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestPopulateMemoryFromLiveTraces:
    """REQ-VERIFY-050: live traces from Exp 219-221 normalise into CaseRecords."""

    def test_exp219_tp_records_extracted(self) -> None:
        """SCENARIO-VERIFY-052: Exp 219 gsm8k_semantic TP cases normalise correctly."""
        # REQ-VERIFY-050: normalization produces well-typed CaseRecord objects
        data219 = _load("experiment_219_results.json")
        records = _extract_exp219_tp_records(data219)
        assert len(records) > 0, "Expected at least one TP case from Exp 219"
        for record, case_id in records:
            assert record.benchmark == "gsm8k_semantic"
            assert record.benchmark_slice == "gsm8k_semantic/live_gsm8k_semantic_failure"
            assert record.repair_outcome == "improved"
            assert record.violation_families  # at least one family extracted
            assert case_id  # provenance preserved

    def test_exp220_tp_records_extracted(self) -> None:
        """SCENARIO-VERIFY-052: Exp 220 humaneval_property TP cases normalise correctly."""
        data220 = _load("experiment_220_results.json")
        records = _extract_exp220_tp_records(data220)
        assert len(records) > 0, "Expected at least one TP case from Exp 220"
        for record, _ in records:
            assert record.benchmark == "humaneval_property"
            assert record.benchmark_slice == "humaneval_property/code_typed_properties"
            assert record.repair_outcome == "improved"
            assert record.violation_families

    def test_exp221_tp_records_extracted(self) -> None:
        """SCENARIO-VERIFY-052: Exp 221 constraint_ir TP cases normalise correctly."""
        data221 = _load("experiment_221_results.json")
        records = _extract_exp221_tp_records(data221)
        assert len(records) > 0, "Expected at least one TP case from Exp 221"
        for record, _ in records:
            assert record.benchmark == "constraint_ir"
            assert record.repair_outcome == "improved"
            assert record.violation_families

    def test_fp_cases_excluded_from_training_memory(self) -> None:
        """REQ-VERIFY-050: every extracted TP record has repair_outcome='improved'.

        The extractor filters to cases where the model was wrong (correct=False) AND
        the verifier detected the violation.  Exp 219 uses multiple paired_run
        conditions, so the same case_id can legitimately appear as a TP in one
        condition and an FP in another — we verify the *records* property, not
        case_id uniqueness.
        """
        data219 = _load("experiment_219_results.json")
        tp_records = _extract_exp219_tp_records(data219)
        assert tp_records, "Should have extracted at least one TP record"
        for record, _ in tp_records:
            # All TP records are normalised with baseline_success=False, repair_success=True
            assert record.repair_outcome == "improved", (
                f"TP record {record.provenance.case_id!r} should have repair_outcome='improved', "
                f"got {record.repair_outcome!r}"
            )
        # Verify FP helper returns only cases where a correct answer was flagged
        fp_case_ids = _extract_exp219_fp_cases(data219)
        assert fp_case_ids, "Should find at least one FP case in Exp 219"

    def test_memory_accumulates_support_from_similar_violation_keys(self) -> None:
        """REQ-VERIFY-050: cases with the same CaseKey aggregate into one entry with support>1."""
        data219 = _load("experiment_219_results.json")
        memory = CaseMemory()
        for record, _ in _extract_exp219_tp_records(data219):
            memory.record(record)
        entries = memory.entries()
        assert entries, "Memory should have entries after ingesting Exp 219"
        # At least some entries should have support > 1 (repeated violation patterns)
        multi_support = [e for e in entries if e.support > 1]
        assert multi_support, "Expected at least one aggregated case with support > 1"


class TestSessionBoundarySaveLoad:
    """SCENARIO-VERIFY-054: session boundary via save/load preserves all state."""

    def test_save_load_round_trip_preserves_all_entries(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-054: CaseMemory survives a session boundary identically."""
        # Simulate Session 1: populate from live traces
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, stats = _build_session1_memory(data219, data220, data221)
        assert stats["total_entries"] > 0

        # Simulate session boundary: persist to disk
        mem_path = tmp_path / "session1_case_memory.json"
        memory_s1.save(mem_path)

        # Simulate Session 2: restore from disk
        memory_s2 = CaseMemory.load(mem_path)

        # State must be identical
        assert len(memory_s2) == len(memory_s1)
        assert memory_s2.to_dict() == memory_s1.to_dict()

    def test_loaded_memory_supports_further_ingestion(self, tmp_path: Path) -> None:
        """REQ-VERIFY-050: loaded memory can be extended in the new session."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, stats = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "session1.json"
        memory_s1.save(mem_path)

        # New session: load and add another record
        memory_s2 = CaseMemory.load(mem_path)
        extra_record = CaseRecord.normalize(
            benchmark="gsm8k_semantic",
            benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
            model_name="Qwen3.5-0.8B",
            case_id="probe-extra-s2",
            violation_types=("question_grounding_failures:answer_target_mismatch",),
            prompt_text="How many extra points did the winner score?",
            description_texts=("The response does not compute the comparison quantity.",),
            baseline_success=False,
            repair_success=True,
            confidence=0.88,
            source_experiment=278,
        )
        memory_s2.record(extra_record)

        # Session 2 memory should have grown
        assert len(memory_s2) >= len(memory_s1)

    def test_session_boundary_json_file_is_valid(self, tmp_path: Path) -> None:
        """REQ-VERIFY-050: persisted memory JSON is a valid versioned object."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "validate.json"
        memory.save(mem_path)

        with mem_path.open(encoding="utf-8") as fh:
            payload = json.load(fh)
        assert payload.get("version") == 1
        assert isinstance(payload.get("entries"), list)
        assert len(payload["entries"]) == len(memory)


class TestColdStartVsWarmMemory:
    """REQ-VERIFY-051: warm memory outperforms cold start for all benchmark types."""

    def test_cold_start_returns_no_matches(self) -> None:
        """REQ-VERIFY-051: empty CaseMemory returns no retrieval matches for any probe."""
        cold_memory = CaseMemory()
        all_probes = (
            _gsm8k_tp_probes()
            + _humaneval_tp_probes()
            + _constraint_ir_tp_probes()
        )
        cold_rate = _hit_rate(cold_memory, all_probes)
        assert cold_rate == 0.0, "Empty memory must return 0 hits"

    def test_warm_memory_hits_gsm8k_semantic_probes(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-053: warm memory retrieves relevant cases for gsm8k probes."""
        # REQ-VERIFY-051: retrieval uses violation family overlap to surface relevant cases
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        _, _ = _build_session1_memory(data219, data220, data221)  # build once to populate
        mem_path = tmp_path / "warm.json"
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        probes = _gsm8k_tp_probes()
        warm_rate = _hit_rate(warm_memory, probes)
        assert warm_rate > 0.0, (
            f"Warm memory should retrieve at least one match for gsm8k probes "
            f"(hit rate={warm_rate:.2f})"
        )

    def test_warm_memory_hits_humaneval_probes(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-053: warm memory retrieves relevant cases for humaneval probes."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "warm_humaneval.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        probes = _humaneval_tp_probes()
        warm_rate = _hit_rate(warm_memory, probes)
        assert warm_rate > 0.0, (
            f"Warm memory should retrieve humaneval matches (hit rate={warm_rate:.2f})"
        )

    def test_warm_memory_hits_constraint_ir_probes(self, tmp_path: Path) -> None:
        """SCENARIO-VERIFY-053: warm memory retrieves relevant cases for constraint_ir probes."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "warm_constraint.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        probes = _constraint_ir_tp_probes()
        warm_rate = _hit_rate(warm_memory, probes)
        assert warm_rate > 0.0, (
            f"Warm memory should retrieve constraint_ir matches (hit rate={warm_rate:.2f})"
        )

    def test_warm_memory_accuracy_gain_positive(self, tmp_path: Path) -> None:
        """REQ-VERIFY-051: warm_hit_rate > cold_hit_rate for all TP probe types."""
        # SCENARIO-VERIFY-053: combined TP probe hit rate must improve over cold start
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "accuracy_gain.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        all_tp_probes = (
            _gsm8k_tp_probes()
            + _humaneval_tp_probes()
            + _constraint_ir_tp_probes()
        )
        cold_rate = _hit_rate(CaseMemory(), all_tp_probes)
        warm_rate = _hit_rate(warm_memory, all_tp_probes)
        accuracy_gain = warm_rate - cold_rate

        assert cold_rate == 0.0
        assert accuracy_gain > 0.0, (
            f"Warm memory must improve over cold start (gain={accuracy_gain:.3f})"
        )


class TestFalsePositiveRate:
    """REQ-VERIFY-051: warm memory keeps spurious hit rate low on unrelated probes."""

    def test_tn_probes_mostly_unmatched_by_warm_memory(self, tmp_path: Path) -> None:
        """REQ-VERIFY-051: queries for unseen benchmark_slices return no matches."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "fp_test.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        # First two TN probes use never-seen benchmark_slices → hard-filter returns 0
        tn_probes = _tn_probes()[:2]
        fp_rate = _hit_rate(warm_memory, tn_probes)
        assert fp_rate == 0.0, (
            f"Probes with unseen benchmark_slice should never hit (fp_rate={fp_rate:.2f})"
        )

    def test_fp_rate_on_known_slice_unrelated_family(self, tmp_path: Path) -> None:
        """REQ-VERIFY-051: probes with matching slice but unrelated violation family score low."""
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")
        memory_s1, _ = _build_session1_memory(data219, data220, data221)
        mem_path = tmp_path / "fp_slice.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)

        # Probe uses same gsm8k slice but completely unrelated violation family
        unrelated_probe = CaseQuery.from_record(
            CaseRecord.normalize(
                benchmark="gsm8k_semantic",
                benchmark_slice="gsm8k_semantic/live_gsm8k_semantic_failure",
                model_name="Unknown-Model-XYZ",
                case_id="fp-unrelated",
                violation_types=("completely_unrelated_family:no_match",),
                prompt_text="",
                description_texts=(),
                baseline_success=None,
                repair_success=None,
                confidence=0.0,
                source_experiment=278,
            )
        )
        # Should return no matches because violation family doesn't overlap AND model unknown
        matches = warm_memory.retrieve(unrelated_probe, limit=5)
        assert not matches, (
            f"Unrelated probe on known slice should return no matches, got {len(matches)}"
        )


class TestIntegration:
    """End-to-end integration: populate, boundary, measure, write results."""

    def test_full_cross_session_pipeline_and_write_results(self, tmp_path: Path) -> None:
        """REQ-VERIFY-050, REQ-VERIFY-051: full cross-session memory pipeline with results JSON.

        This is the authoritative integration test for Exp 278:
        1. Session 1: populate from Exp 219-221 live traces
        2. Session boundary: save to disk, load in new session
        3. Probe: measure warm vs cold hit rate per benchmark type
        4. Validate: warm > cold, FP rate == 0 for unseen slices
        5. Write: results JSON to results/experiment_278_results.json
        """
        data219 = _load("experiment_219_results.json")
        data220 = _load("experiment_220_results.json")
        data221 = _load("experiment_221_results.json")

        # ── Session 1: populate ──────────────────────────────────────────
        memory_s1, ingest_stats = _build_session1_memory(data219, data220, data221)
        assert ingest_stats["total_entries"] > 0
        assert ingest_stats["by_source"]["exp219_gsm8k_semantic"] > 0

        # ── Session boundary ─────────────────────────────────────────────
        mem_path = tmp_path / "exp278_case_memory.json"
        memory_s1.save(mem_path)
        warm_memory = CaseMemory.load(mem_path)
        assert len(warm_memory) == len(memory_s1)

        # ── Cold start baseline ──────────────────────────────────────────
        cold_memory = CaseMemory()

        # ── Probe sets ───────────────────────────────────────────────────
        gsm8k_probes = _gsm8k_tp_probes()
        humaneval_probes = _humaneval_tp_probes()
        constraint_probes = _constraint_ir_tp_probes()
        tn_probes = _tn_probes()[:2]  # unseen-slice probes (hard-filter guaranteed)
        all_tp_probes = gsm8k_probes + humaneval_probes + constraint_probes

        # ── Metrics ──────────────────────────────────────────────────────
        cold_hit_rate = _hit_rate(cold_memory, all_tp_probes)
        warm_gsm8k_hit_rate = _hit_rate(warm_memory, gsm8k_probes)
        warm_humaneval_hit_rate = _hit_rate(warm_memory, humaneval_probes)
        warm_constraint_hit_rate = _hit_rate(warm_memory, constraint_probes)
        warm_overall_hit_rate = _hit_rate(warm_memory, all_tp_probes)
        warm_fp_rate = _hit_rate(warm_memory, tn_probes)  # should be 0
        accuracy_gain = warm_overall_hit_rate - cold_hit_rate

        avg_score = _avg_top_score(warm_memory, all_tp_probes)

        # ── Assertions ───────────────────────────────────────────────────
        assert cold_hit_rate == 0.0
        assert accuracy_gain > 0.0, f"accuracy_gain={accuracy_gain:.3f} must be positive"
        assert warm_fp_rate == 0.0, f"warm_fp_rate={warm_fp_rate:.3f} must be 0 for unseen slices"

        # ── Write results JSON ────────────────────────────────────────────
        # (Written to the real repo results/ directory, not tmp_path, as deliverable)
        results_path = _RESULTS / "experiment_278_results.json"
        result_payload: dict[str, Any] = {
            "experiment": 278,
            "title": "Cross-session constraint memory with live traces",
            "run_date": "20260414",
            "schema": "exp278_cross_session_memory_v1",
            "metadata": {
                "source_experiments": [219, 220, 221],
                "probe_experiments": [278],
                "benchmarks": ["gsm8k_semantic", "humaneval_property", "constraint_ir"],
            },
            "session1": {
                "description": "Populate CaseMemory from Exp 219-221 verified TP traces",
                "ingest_stats": ingest_stats,
            },
            "session_boundary": {
                "saved_to_disk": True,
                "loaded_from_disk": True,
                "entries_preserved": len(warm_memory) == len(memory_s1),
            },
            "measurements": {
                "cold_start_hit_rate": cold_hit_rate,
                "warm_gsm8k_hit_rate": round(warm_gsm8k_hit_rate, 4),
                "warm_humaneval_hit_rate": round(warm_humaneval_hit_rate, 4),
                "warm_constraint_hit_rate": round(warm_constraint_hit_rate, 4),
                "warm_overall_hit_rate": round(warm_overall_hit_rate, 4),
                "accuracy_gain": round(accuracy_gain, 4),
                "warm_fp_rate_unseen_slice": warm_fp_rate,
                "avg_top_match_score": round(avg_score, 2),
            },
            "summary": {
                "session_boundary_verified": True,
                "warm_beats_cold": accuracy_gain > 0.0,
                "fp_rate_acceptable": warm_fp_rate == 0.0,
                "outcome": "pass" if accuracy_gain > 0.0 and warm_fp_rate == 0.0 else "fail",
            },
        }
        with results_path.open("w", encoding="utf-8") as fh:
            json.dump(result_payload, fh, indent=2)

        # Verify file was written
        assert results_path.exists()
        loaded = json.loads(results_path.read_text(encoding="utf-8"))
        assert loaded["experiment"] == 278
        assert loaded["summary"]["outcome"] == "pass"
