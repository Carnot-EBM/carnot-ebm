"""Tests for carnot.pipeline.constraint_template_store.

Spec: REQ-LEARN-058, REQ-LEARN-059,
SCENARIO-LEARN-090, SCENARIO-LEARN-091, SCENARIO-LEARN-092
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from carnot.pipeline.constraint_template_store import (
    ConstraintTemplate,
    ConstraintTemplateStore,
)


# ---------------------------------------------------------------------------
# ConstraintTemplate serialisation
# ---------------------------------------------------------------------------


class TestConstraintTemplate:
    """REQ-LEARN-058: ConstraintTemplate dataclass serialises round-trip."""

    def test_to_dict_round_trip(self):
        # SCENARIO-LEARN-090: template serialises and deserialises cleanly
        tmpl = ConstraintTemplate(
            name="carry_guard",
            violation_type="carry",
            context_keywords=["arithmetic", "sum"],
            constraint_text="Guard against carry violations.",
            n_violations_observed=5,
        )
        d = tmpl.to_dict()
        assert d["name"] == "carry_guard"
        assert d["violation_type"] == "carry"
        assert d["context_keywords"] == ["arithmetic", "sum"]
        assert d["constraint_text"] == "Guard against carry violations."
        assert d["n_violations_observed"] == 5

        restored = ConstraintTemplate.from_dict(d)
        assert restored == tmpl

    def test_from_dict_with_missing_fields(self):
        # from_dict should tolerate missing optional keys gracefully
        restored = ConstraintTemplate.from_dict({})
        assert restored.name == ""
        assert restored.violation_type == ""
        assert restored.context_keywords == []
        assert restored.n_violations_observed == 0


# ---------------------------------------------------------------------------
# add_violation
# ---------------------------------------------------------------------------


class TestAddViolation:
    """REQ-LEARN-058 / SCENARIO-LEARN-090: add_violation accumulates counts."""

    def test_add_increments_count(self):
        store = ConstraintTemplateStore()
        store.add_violation("carry", "arithmetic question about sum")
        store.add_violation("carry", "another carry question")
        counts = store.violation_counts()
        assert counts["carry"] == 2

    def test_add_normalises_colon_prefix(self):
        # "carry:overflow" should accumulate under "carry"
        store = ConstraintTemplateStore()
        store.add_violation("carry:overflow", "arithmetic")
        store.add_violation("carry", "addition problem")
        counts = store.violation_counts()
        assert counts["carry"] == 2

    def test_add_ignores_empty_type(self):
        store = ConstraintTemplateStore()
        store.add_violation("", "some context")
        assert store.violation_counts() == {}

    def test_add_multiple_types(self):
        store = ConstraintTemplateStore()
        store.add_violation("carry", "x")
        store.add_violation("semantic", "y")
        store.add_violation("semantic", "z")
        counts = store.violation_counts()
        assert counts["carry"] == 1
        assert counts["semantic"] == 2

    def test_add_invalidates_distill_cache(self):
        # After add_violation, distill() must recompute.
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "ctx")
        t1 = store.distill(min_observations=3)
        assert len(t1) == 1
        # Add one more — cache should be invalidated
        store.add_violation("carry", "extra ctx")
        t2 = store.distill(min_observations=3)
        assert t2[0].n_violations_observed == 4


# ---------------------------------------------------------------------------
# distill
# ---------------------------------------------------------------------------


class TestDistill:
    """REQ-LEARN-058 / SCENARIO-LEARN-090 / SCENARIO-LEARN-091: distill threshold."""

    def test_below_threshold_not_distilled(self):
        store = ConstraintTemplateStore()
        store.add_violation("carry", "ctx")
        store.add_violation("carry", "ctx2")
        templates = store.distill(min_observations=3)
        assert templates == []

    def test_at_threshold_distilled(self):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "arithmetic carry sum")
        templates = store.distill(min_observations=3)
        assert len(templates) == 1
        tmpl = templates[0]
        assert tmpl.violation_type == "carry"
        assert tmpl.name == "carry_guard"
        assert tmpl.n_violations_observed == 3
        # Context keywords should include tokens from the context
        assert "arithmetic" in tmpl.context_keywords or "carry" in tmpl.context_keywords

    def test_above_threshold_distilled(self):
        store = ConstraintTemplateStore()
        for _ in range(5):
            store.add_violation("semantic", "semantic mismatch answer")
        templates = store.distill(min_observations=3)
        assert len(templates) == 1
        assert templates[0].n_violations_observed == 5

    def test_mixed_maturity(self):
        # Only mature patterns appear in distill output
        store = ConstraintTemplateStore()
        for _ in range(4):
            store.add_violation("carry", "ctx")
        store.add_violation("semantic", "ctx")  # only 1 — below threshold
        templates = store.distill(min_observations=3)
        vtypes = [t.violation_type for t in templates]
        assert "carry" in vtypes
        assert "semantic" not in vtypes

    def test_distill_sorted_by_violation_type(self):
        store = ConstraintTemplateStore()
        for vtype in ["zebra", "apple", "mango"]:
            for _ in range(3):
                store.add_violation(vtype, "ctx")
        templates = store.distill(min_observations=3)
        vtypes = [t.violation_type for t in templates]
        assert vtypes == sorted(vtypes)

    def test_distill_result_cached(self):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "ctx")
        t1 = store.distill()
        t2 = store.distill()
        # Same objects — cache hit (no recompute)
        assert t1 == t2

    def test_constraint_text_contains_violation_type(self):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("unit", "unit conversion error")
        templates = store.distill(min_observations=3)
        assert "unit" in templates[0].constraint_text


# ---------------------------------------------------------------------------
# retrieve
# ---------------------------------------------------------------------------


class TestRetrieve:
    """REQ-LEARN-059 / SCENARIO-LEARN-092: retrieve by keyword overlap."""

    def _make_store_with_patterns(self) -> ConstraintTemplateStore:
        store = ConstraintTemplateStore()
        for _ in range(4):
            store.add_violation("carry", "arithmetic carry sum overflow addition")
        for _ in range(4):
            store.add_violation("semantic", "semantic answer mismatch question grounding")
        for _ in range(4):
            store.add_violation("sign", "sign negative positive arithmetic")
        store.distill(min_observations=3)
        return store

    def test_retrieve_returns_top_k(self):
        store = self._make_store_with_patterns()
        results = store.retrieve("arithmetic carry sum", top_k=2)
        assert len(results) <= 2

    def test_retrieve_ranks_by_keyword_overlap(self):
        store = self._make_store_with_patterns()
        # "arithmetic carry" should rank carry_guard above semantic_guard
        results = store.retrieve("arithmetic carry overflow", top_k=3)
        vtypes = [r.violation_type for r in results]
        assert vtypes[0] == "carry"

    def test_retrieve_semantic_context(self):
        store = self._make_store_with_patterns()
        results = store.retrieve("semantic answer mismatch grounding", top_k=3)
        assert results[0].violation_type == "semantic"

    def test_retrieve_no_context_match_returns_any(self):
        store = self._make_store_with_patterns()
        # A completely unrelated query still returns templates (all score 0)
        results = store.retrieve("xyzzy quux", top_k=3)
        assert len(results) > 0

    def test_retrieve_fewer_than_top_k_templates(self):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "ctx")
        store.distill(min_observations=3)
        results = store.retrieve("ctx", top_k=5)
        assert len(results) == 1

    def test_retrieve_calls_distill_if_needed(self):
        # retrieve() should work without an explicit prior distill() call
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "arithmetic")
        results = store.retrieve("arithmetic", top_k=3)
        # At min_observations=1, carry (3 obs) should be present
        assert any(r.violation_type == "carry" for r in results)

    def test_retrieve_empty_store_returns_empty(self):
        store = ConstraintTemplateStore()
        results = store.retrieve("anything", top_k=3)
        assert results == []


# ---------------------------------------------------------------------------
# save / load
# ---------------------------------------------------------------------------


class TestSaveLoad:
    """REQ-LEARN-058 / SCENARIO-LEARN-090: store persists and reloads correctly."""

    def test_save_load_round_trip(self, tmp_path):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "arithmetic carry")
        store.distill(min_observations=3)
        path = tmp_path / "store.json"
        store.save(path)
        assert path.exists()

        loaded = ConstraintTemplateStore.load(path)
        counts = loaded.violation_counts()
        assert counts["carry"] == 3
        # Distilled cache should be restored
        templates = loaded.distill(min_observations=3)
        assert len(templates) == 1
        assert templates[0].violation_type == "carry"

    def test_save_json_has_schema_version(self, tmp_path):
        store = ConstraintTemplateStore()
        path = tmp_path / "store.json"
        store.save(path)
        raw = json.loads(path.read_text())
        assert "schema_version" in raw
        assert raw["schema_version"] == 1

    def test_load_wrong_schema_version_raises(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text(json.dumps({"schema_version": 999, "accumulators": {}}))
        with pytest.raises(ValueError, match="schema_version"):
            ConstraintTemplateStore.load(path)

    def test_loaded_store_can_add_new_violations(self, tmp_path):
        store = ConstraintTemplateStore()
        for _ in range(3):
            store.add_violation("carry", "ctx")
        path = tmp_path / "store.json"
        store.save(path)

        loaded = ConstraintTemplateStore.load(path)
        loaded.add_violation("semantic", "sem ctx")
        loaded.add_violation("semantic", "sem ctx2")
        loaded.add_violation("semantic", "sem ctx3")
        templates = loaded.distill(min_observations=3)
        vtypes = [t.violation_type for t in templates]
        assert "carry" in vtypes
        assert "semantic" in vtypes

    def test_save_load_without_prior_distill(self, tmp_path):
        # save() should not crash when distill() was never called
        store = ConstraintTemplateStore()
        store.add_violation("carry", "ctx")
        path = tmp_path / "store.json"
        store.save(path)
        loaded = ConstraintTemplateStore.load(path)
        assert loaded.violation_counts()["carry"] == 1
