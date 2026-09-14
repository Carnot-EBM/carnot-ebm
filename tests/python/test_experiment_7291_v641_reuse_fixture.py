"""Tests for REQ-VERIFY-7291 and SCENARIO-VERIFY-7291-*.

The tests keep labels in a scorer-only value. Prediction functions receive
only the public fixture, so a passing test cannot hide authority leakage.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7291_v641_reuse_fixture as mod


ROOT = Path(__file__).resolve().parents[2]


def _source(source_id: str, version: int, text: str) -> dict[str, object]:
    """Build one public source document with complete cache provenance."""

    document = mod.make_document(f"{source_id}-v{version}", text)
    return {
        "source_id": source_id,
        "source_version": version,
        "document": document,
        "completion": mod.extract_completion(document, "source"),
        "parser_schema_hash": mod.PARSER_SCHEMA_HASH,
        "model_configuration": deepcopy(mod.INJECTED_MODEL_CONFIGURATION),
    }


def test_scenario_verify_7291_cache_rejects_stale_collision_and_missing_provenance() -> None:
    """SCENARIO-VERIFY-7291-CACHE invalidates changes and fails closed."""

    cache = mod.SourceCompilationCache(enabled=True, capacity=2)
    first = _source("source-a", 1, "Aster precedes Brin.")
    changed = _source("source-a", 2, "Brin precedes Aster.")

    inserted = cache.compile_or_get(first)
    hit = cache.compile_or_get(first)
    replacement = cache.compile_or_get(changed)
    stale = cache.compile_or_get(first)
    collision = cache.compile_or_get(_source("source-a", 2, "Cora precedes Daro."))
    missing = deepcopy(changed)
    del missing["parser_schema_hash"]

    assert inserted["ok"] is True and inserted["cache_hit"] is False
    assert hit["ok"] is True and hit["cache_hit"] is True
    assert replacement["ok"] is True and replacement["invalidated_entries"] == 1
    assert stale == mod.cache_error("stale_source_version")
    assert collision == mod.cache_error("source_id_collision", invalidated_entries=1)
    assert cache.compile_or_get(missing) == mod.cache_error("missing_provenance")
    assert cache.served_stale_constraints == 0


def test_scenario_verify_7291_cache_stores_sources_only_and_evicts_lru() -> None:
    """SCENARIO-VERIFY-7291-AUTHORITY keeps answers out and bounds capacity."""

    cache = mod.SourceCompilationCache(enabled=True, capacity=1)
    assert cache.compile_or_get(_source("source-a", 1, "Aster precedes Brin."))["ok"]
    second = cache.compile_or_get(_source("source-b", 1, "Cora precedes Daro."))

    assert second["evicted_entries"] == 1
    assert len(cache.snapshot()) == 1
    encoded = mod.canonical_json(cache.snapshot())
    assert "expected_label" not in encoded
    assert "claim" not in encoded
    assert "answer" not in encoded
    assert set(cache.snapshot()[0]) == {
        "cache_key",
        "source_id",
        "source_version",
        "compiled_source",
    }
    disabled = mod.SourceCompilationCache(enabled=False, capacity=1)
    assert disabled.compile_or_get(
        _source("source-c", 1, "Eris precedes Fenn.")
    ) == mod.cache_error("cache_disabled")
    with pytest.raises(ValueError, match="capacity"):
        mod.SourceCompilationCache(enabled=True, capacity=0)


def test_scenario_verify_7291_compiler_reconstructs_spans_and_rejects_schema() -> None:
    """SCENARIO-VERIFY-7291-CONTROLS uses the shipped pointer compiler."""

    document = mod.make_document("doc", "Aster starts before Brin by four ticks.")
    completion = mod.extract_completion(document, "source")
    compiled = mod.pointer.compile_pointer_completion(document, completion, "source")

    assert compiled["outcome"] == "known"
    relation = compiled["relations"][0]
    encoded = document["text"].encode("utf-8")
    assert encoded[relation["subject_start"] : relation["subject_end"]] == b"Aster"
    assert encoded[relation["object_start"] : relation["object_end"]] == b"Brin"
    extra = deepcopy(completion)
    extra["unknown_field"] = True
    assert mod.pointer.compile_pointer_completion(document, extra, "source")["errors"] == [
        "completion_shape"
    ]
    assert mod.extract_completion(mod.make_document("bad", "no relation here"), "claim") == {
        "outcome": "unknown",
        "relations": [],
    }
    assert mod.extract_completion(document, "invalid-call-type")["outcome"] == "unknown"
    duplicate = mod.make_document("duplicate", "Aster precedes Aster.")
    assert mod.extract_completion(duplicate, "source")["outcome"] == "unknown"


def test_scenario_verify_7291_cache_rejects_invalid_provenance_types() -> None:
    """SCENARIO-VERIFY-7291-CACHE rejects malformed identity and source text."""

    cache = mod.SourceCompilationCache(enabled=True, capacity=2)
    invalid_identity = _source("source-a", 1, "Aster precedes Brin.")
    invalid_identity["source_version"] = False
    invalid_text = _source("source-b", 1, "Cora precedes Daro.")
    invalid_text["document"] = {"document_id": "bad", "text": None, "mentions": []}

    assert cache.compile_or_get(invalid_identity) == mod.cache_error("invalid_provenance")
    assert cache.compile_or_get(invalid_text) == mod.cache_error("invalid_provenance")


def test_scenario_verify_7291_manifest_is_disjoint_balanced_and_private() -> None:
    """SCENARIO-VERIFY-7291-AUTHORITY freezes separate public and scorer views."""

    public, scorer = mod.build_fixture()
    evaluation = public["evaluation_groups"]
    development = public["development_groups"]

    assert len(development) == 8 and len(evaluation) == 16
    assert all(len(group["claims"]) == 8 for group in [*development, *evaluation])
    assert all(
        [row["source_version"] for row in group["claims"]] == [1, 1, 1, 1, 2, 2, 2, 2]
        for group in evaluation
    )
    assert len(scorer["labels"]) == 24 * 8
    eval_labels = [
        row["expected_decision"] for row in scorer["labels"] if row["split"] == "evaluation"
    ]
    assert {label: eval_labels.count(label) for label in set(eval_labels)} == {
        "supported": 48,
        "contradicted": 48,
        "unknown": 32,
    }
    public_text = mod.canonical_json(public)
    for forbidden in ("expected_decision", "gold_span", "construction_label"):
        assert forbidden not in public_text
    dev_ids = {group["group_id"] for group in development}
    eval_ids = {group["group_id"] for group in evaluation}
    assert dev_ids.isdisjoint(eval_ids)
    assert set(public["development_seeds"]).isdisjoint(public["evaluation_seeds"])
    assert public["collision_probes"]


def test_scenario_verify_7291_comparison_preserves_parity_costs_and_call_shapes() -> None:
    """SCENARIO-VERIFY-7291-COMPARISON runs all arms on each public claim."""

    public, scorer = mod.build_fixture()
    raw = mod.execute_public_fixture(public)
    reduced = mod.reduce_rows(raw["prediction_rows"], scorer)

    assert len(raw["prediction_rows"]) == 128 * 3
    assert len(reduced) == 128 * 3
    assert "expected_decision" not in mod.canonical_json(raw["prediction_rows"])
    assert {row["arm"] for row in reduced} == set(mod.ARMS)
    assert all(row["censored"] is False and "cost" in row for row in reduced)
    assert all(row["prediction"] == row["expected_decision"] for row in reduced)
    direct = [row for row in reduced if row["arm"] == "warm_prefix_direct"]
    reuse = [row for row in reduced if row["arm"] == "versioned_reuse_verifier"]
    assert all(len(row["draws"]) == 2 for row in direct)
    assert sum(row["cost"]["source_compilations"] for row in reuse) == 16 * 2
    assert sum(row["cost"]["claim_extractions"] for row in reuse) == 128
    assert raw["cache_stats"]["served_stale_constraints"] == 0
    assert mod.reduce_two_draws("supported", "contradicted") == "unknown"


def test_scenario_verify_7291_controls_cover_shuffle_rename_and_stale_negative() -> None:
    """SCENARIO-VERIFY-7291-CONTROLS keeps the stale arm out of primary rows."""

    public, _scorer = mod.build_fixture()
    raw = mod.execute_public_fixture(public)
    controls = raw["control_rows"]

    assert {row["control"] for row in controls} == set(mod.REQUIRED_CONTROLS)
    assert all(row["passed"] for row in controls)
    stale = next(row for row in controls if row["control"] == "deliberately_stale_cache")
    assert stale["labelled_negative_control"] is True
    assert stale["included_in_primary_rows"] is False
    assert all(row["arm"] in mod.ARMS for row in raw["prediction_rows"])


def test_scenario_verify_7291_bootstrap_resamples_groups_and_passes_frozen_gates() -> None:
    """SCENARIO-VERIFY-7291-BOOTSTRAP uses sixteen groups, not 128 claims."""

    public, scorer = mod.build_fixture()
    raw = mod.execute_public_fixture(public)
    rows = mod.reduce_rows(raw["prediction_rows"], scorer)
    first = mod.paired_group_bootstrap(rows, seed=mod.BOOTSTRAP_SEED, draws=10_000)
    second = mod.paired_group_bootstrap(rows, seed=mod.BOOTSTRAP_SEED, draws=10_000)

    assert first == second
    assert first["independent_groups"] == 16
    assert first["resampling_unit"] == "source_group"
    assert first["draws"] == 10_000
    assert first["accuracy_delta"]["one_sided_95_lower"] >= -0.02
    assert first["coverage_delta"]["one_sided_95_lower"] >= -0.02
    assert first["false_accept_counts"]["reuse"] <= first["false_accept_counts"]["direct"]
    assert [row["claims"] for row in first["amortization"]] == [1, 2, 4, 8]
    assert first["amortization"][-1]["one_sided_95_lower_speedup"] >= 1.5


def test_scenario_verify_7291_e2e_raw_reduction_and_terminal_validation(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7291-E2E binds raw evidence before terminal output."""

    raw_dir = tmp_path / "raw"
    output = tmp_path / "result.json"
    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        raw_dir=raw_dir,
        output_path=output,
        validation_receipts=mod.fixture_validation_receipts(),
    )

    assert output.exists()
    assert json.loads(output.read_text(encoding="utf-8")) == artifact
    assert artifact["status"] == "complete"
    assert artifact["reuse_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    assert artifact["inference_substrate"] == "cpu_exact_solver_or_simulator"
    assert artifact["inference_substrate_class"] == "cpu_exact_solver_or_simulator"
    assert mod.validate_artifact(artifact, root=tmp_path) == []
    assert mod.independent_reduce_raw(raw_dir)["rows"] == artifact["rows"]
    assert mod.classify_terminal(artifact["validation_receipts"]) == {
        "status": "complete",
        "verdict_class": "circular_positive",
        "honest_verdict": "complete_circular_positive_versioned_reuse_fixture_ready",
        "validation_complete": True,
    }

    baseline_receipts = deepcopy(artifact["validation_receipts"])
    suite = next(row for row in baseline_receipts if row["name"] == "full_python_suite")
    suite.update({"exit_code": 2, "passed": False, "baseline_failure": True})
    terminal = mod.classify_terminal(baseline_receipts)
    assert terminal == {
        "status": "complete",
        "verdict_class": "disqualified",
        "honest_verdict": "complete_disqualified_reuse_fixture_ready_but_repository_full_suite_failed",
        "validation_complete": False,
    }
    assert mod.classify_terminal([]) == {
        "status": "partial",
        "verdict_class": "partial",
        "honest_verdict": "partial_exp7291_validation_failed",
        "validation_complete": False,
    }
    disqualified = deepcopy(artifact)
    disqualified.update(terminal)
    disqualified["validation_receipts"] = baseline_receipts
    disqualified["reproducibility_checksum"] = mod.artifact_checksum(disqualified)
    assert mod.validate_artifact(disqualified, root=tmp_path) == []

    changed = deepcopy(artifact)
    changed["reuse_fixture_ready_score"] = 0
    assert "reuse_fixture_ready_score" in mod.validate_artifact(changed, root=tmp_path)

    mutations = []
    missing = deepcopy(artifact)
    del missing["cache_key_contract"]
    mutations.append((missing, "cache_key_contract"))
    wrong_identity = deepcopy(artifact)
    wrong_identity["execution_venue"] = "gpu"
    mutations.append((wrong_identity, "execution_venue"))
    wrong_verdict = deepcopy(artifact)
    wrong_verdict["verdict_class"] = "positive"
    mutations.append((wrong_verdict, "verdict_class"))
    short_rows = deepcopy(artifact)
    short_rows["rows"] = short_rows["rows"][:-1]
    mutations.append((short_rows, "rows"))
    short_controls = deepcopy(artifact)
    short_controls["control_rows"] = short_controls["control_rows"][:-1]
    mutations.append((short_controls, "control_rows"))
    no_manifest = deepcopy(artifact)
    no_manifest["fixture_manifest_path"] = None
    mutations.append((no_manifest, "fixture_manifest_path"))
    bad_manifest_hash = deepcopy(artifact)
    bad_manifest_hash["fixture_manifest_sha256"] = "sha256:changed"
    mutations.append((bad_manifest_hash, "fixture_manifest_sha256"))
    failed_validation = deepcopy(artifact)
    failed_validation["validation_receipts"] = []
    mutations.append((failed_validation, "validation_receipts"))
    for mutated, expected_error in mutations:
        assert expected_error in mod.validate_artifact(mutated, root=tmp_path)


def test_scenario_verify_7291_e2e_detects_raw_hash_drift_and_immutable_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7291-E2E rejects changed raw bytes and invalid assembly."""

    raw_dir = tmp_path / "raw"
    output = tmp_path / "result.json"
    mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        raw_dir=raw_dir,
        output_path=output,
        validation_receipts=mod.fixture_validation_receipts(),
    )
    value = {"stable": True}
    frozen = tmp_path / "frozen.json"
    mod._write_json(frozen, value, immutable=True)
    mod._write_json(frozen, value, immutable=True)
    with pytest.raises(ValueError, match="immutable evidence changed"):
        mod._write_json(frozen, {"stable": False}, immutable=True)

    scorer_path = raw_dir / mod.SCORER_NAME
    analysis_path = raw_dir / mod.ANALYSIS_NAME
    scorer_path.write_text(scorer_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    analysis_path.write_text(analysis_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    assert mod.independent_reduce_raw(raw_dir)["errors"] == [
        "scorer_authority_hash",
        "analysis_contract_hash",
    ]

    failing_raw = tmp_path / "failing-raw"
    monkeypatch.setattr(mod, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="terminal artifact invalid"):
        mod.build_artifact(
            ROOT,
            mod.RUN_DATE,
            raw_dir=failing_raw,
            output_path=tmp_path / "invalid.json",
            validation_receipts=mod.fixture_validation_receipts(),
        )


def test_req_verify_7291_missing_input_is_terminal_blocked(tmp_path: Path) -> None:
    """REQ-VERIFY-7291 makes an absent external input a terminal block."""

    checks = mod.authenticate_inputs(tmp_path)
    artifact = mod.blocked_artifact(mod.RUN_DATE, checks, duration_s=0.0)

    assert artifact["status"] == "blocked"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["gate_check_summary"]["expected"] == "present regular file"
    assert artifact["gate_check_summary"]["observed"] == "missing"
    assert artifact["reuse_fixture_ready_score"] == 0
    assert artifact["invocation_counts"] == mod.ZERO_INVOCATION_COUNTS
    written = mod.build_artifact(
        tmp_path,
        mod.RUN_DATE,
        raw_dir=tmp_path / "raw",
        output_path=tmp_path / "blocked.json",
        validation_receipts=[],
    )
    assert written["status"] == "blocked"
    assert json.loads((tmp_path / "blocked.json").read_text(encoding="utf-8")) == written
    assert mod._gate_summary(mod.authenticate_inputs(ROOT)) is None
    assert mod._manifest_lists_experiment(
        {"nested": [{"id": mod.EXPERIMENT_ID}]}, mod.EXPERIMENT_ID
    )
    with pytest.raises(argparse.ArgumentTypeError, match="date must be"):
        mod._date_argument("20260913")
    assert mod._date_argument(mod.RUN_DATE) == mod.RUN_DATE
