"""Tests for Exp 5044 second-corpus candidate cache data product.

Spec refs: REQ-VERIFY-5044, SCENARIO-VERIFY-5044.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_5044_second_corpus_candidate_cache as mod
from carnot.eval import constraintbench_feasibility_objective_pilot_v1 as cb


REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "verification" / "spec.md"


def _source_rows(n: int = 3) -> list[dict[str, Any]]:
    return cb.build_fixture_rows()[:n]


def _source_loader(rows: list[dict[str, Any]], source_path: Path):
    def load(*, root: Path) -> tuple[list[dict[str, Any]], Path, list[mod.PreconditionCheck]]:
        return (
            rows,
            source_path,
            [
                mod.PreconditionCheck(
                    "constraintbench_exact_pilot",
                    True,
                    f"{len(rows)} exact solver-backed source rows",
                    source_path.as_posix(),
                )
            ],
        )

    return load


def test_req_verify_5044_spec_declares_cache_contract() -> None:
    """REQ-VERIFY-5044: OpenSpec anchors the cache schema and resume contract."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    for marker in (
        "REQ-VERIFY-5044",
        "SCENARIO-VERIFY-5044",
        "experiment_5044_second_corpus_candidate_cache.py",
        "results/experiment_5044_second_corpus_candidate_cache.json",
        "second_corpus_cache_built",
        "genuine_sc_accuracy",
        "oracle_at_k",
        "verifier_is_oracle",
        "ConstraintBench",
    ):
        assert marker in spec
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_verify_5044_candidate_rows_validate_and_expose_headroom() -> None:
    """REQ-VERIFY-5044: candidate rows preserve labels and oracle headroom."""

    row = mod.build_candidate_cache_row(
        _source_rows(1)[0],
        variant_index=0,
        source_path=Path("data/research/constraintbench_feasibility_objective_pilot_v1.jsonl"),
    )
    metrics = mod.compute_cache_metrics([row])

    assert mod.validate_cache_row(row) == []
    assert row["schema"] == mod.CACHE_ROW_SCHEMA
    assert row["verifier_provenance"]["verifier_is_oracle"] is False
    assert row["solver_provenance"]["authority"] == "local_exhaustive_enumeration"
    assert len(row["candidates"]) == mod.DEFAULT_CANDIDATES_PER_QUESTION
    assert any(candidate["answer"] == row["gold"] for candidate in row["candidates"])
    assert metrics["genuine_sc_accuracy"] == pytest.approx(0.0)
    assert metrics["oracle_at_k"] == pytest.approx(1.0)
    assert metrics["headroom_present"] is True
    assert metrics["n_flips_possible"] == 1


def test_scenario_verify_5044_run_writes_required_schema_fields(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5044: run writes the reusable cache and terminal artifact."""

    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    artifact_path = tmp_path / mod.RESULT_RELATIVE_PATH
    source_path = tmp_path / "constraintbench_source.jsonl"
    artifact = mod.run(
        root=tmp_path,
        artifact_path=artifact_path,
        cache_path=cache_path,
        source_loader=_source_loader(_source_rows(2), source_path),
        limit=4,
        min_questions=4,
        write=True,
        now=lambda: 100.0,
    )

    persisted = json.loads(artifact_path.read_text(encoding="utf-8"))
    cached_rows = mod.read_complete_candidate_rows(cache_path)

    assert persisted == artifact
    assert set(mod.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert mod.artifact_schema_errors(artifact) == []
    assert (
        artifact["honest_verdict"]
        == "complete_second_corpus_cache_ready_constraintbench_exact_v1_n4"
    )
    assert artifact["second_corpus_cache_built"] is True
    assert artifact["second_corpus_name"] == mod.CONSTRAINTBENCH_CORPUS_NAME
    assert artifact["n_questions"] == 4
    assert artifact["n_candidate_rows"] == 20
    assert artifact["genuine_sc_accuracy"] == pytest.approx(0.0)
    assert artifact["oracle_at_k"] == pytest.approx(1.0)
    assert artifact["headroom_present"] is True
    assert artifact["verifier_is_oracle"] is False
    assert artifact["ppbench_probe"]["available"] is False
    assert artifact["fallback_used"] is True
    assert artifact["candidate_cache_path"] == cache_path.as_posix()
    assert len(cached_rows) == 4


def test_scenario_verify_5044_resume_skips_complete_rows(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5044: resume appends missing rows without duplication."""

    source_path = tmp_path / "constraintbench_source.jsonl"
    cache_path = tmp_path / mod.CACHE_RELATIVE_PATH
    existing = mod.build_candidate_cache_row(
        _source_rows(1)[0],
        variant_index=0,
        source_path=source_path,
    )
    mod.append_jsonl_row(cache_path, existing)

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=cache_path,
        source_loader=_source_loader(_source_rows(2), source_path),
        limit=3,
        min_questions=3,
        write=True,
        now=lambda: 200.0,
    )
    cached_rows = mod.read_complete_candidate_rows(cache_path)

    assert artifact["resume_summary"]["existing_complete_rows"] == 1
    assert artifact["resume_summary"]["appended_rows"] == 2
    assert artifact["n_questions"] == 3
    assert len(cached_rows) == 3
    assert len({row["row_id"] for row in cached_rows}) == 3
    persisted_row_ids = [
        json.loads(line)["row_id"]
        for line in cache_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert persisted_row_ids.count(existing["row_id"]) == 1


def test_req_verify_5044_loader_and_candidate_edge_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-5044: local loading and fallback candidate paths stay deterministic."""

    source_rows = _source_rows(11)
    source_file = tmp_path / mod.CONSTRAINTBENCH_SOURCE_RELATIVE_PATH
    source_file.parent.mkdir(parents=True, exist_ok=True)
    source_file.write_text(
        "\n".join(["", json.dumps(source_rows[0], sort_keys=True), ""]),
        encoding="utf-8",
    )

    loaded_rows, loaded_path, checks = mod.default_source_loader(root=tmp_path)
    rebuilt_rows, rebuilt_path, rebuilt_checks = mod.default_source_loader(
        root=tmp_path / "missing"
    )
    raw_without_reference = {
        key: value for key, value in source_rows[0].items() if key != "exact_reference"
    }
    rebuilt_row = mod.build_candidate_cache_row(
        raw_without_reference,
        variant_index=9,
        source_path=rebuilt_path,
    )

    assert loaded_rows == [source_rows[0]]
    assert loaded_path == source_file
    assert checks[0].available is True
    assert len(rebuilt_rows) == 15
    assert rebuilt_checks[0].available is True
    assert mod.validate_cache_row(rebuilt_row) == []

    assert "selected_items" in mod._fallback_wrong_solution(source_rows[0])
    assert "assignment" in mod._fallback_wrong_solution(source_rows[5])
    assert "colors" in mod._fallback_wrong_solution(source_rows[10])

    def fail_nonoptimal(_row: dict[str, Any]) -> dict[str, Any]:
        raise ValueError("no nonoptimal")

    monkeypatch.setattr(mod.cb, "feasible_nonoptimal_solution", fail_nonoptimal)
    assert "selected_items" in mod._wrong_solution(source_rows[0])

    empty_rows, empty_resume = mod.ensure_candidate_cache(
        cache_path=tmp_path / "empty.jsonl",
        source_rows=[],
        source_path=source_file,
        limit=2,
    )
    assert empty_rows == []
    assert empty_resume["target_rows"] == 2


def test_req_verify_5044_schema_error_edges(tmp_path: Path) -> None:
    """REQ-VERIFY-5044: schema validators report malformed rows and artifacts."""

    artifact = mod.run(
        root=tmp_path,
        artifact_path=tmp_path / mod.RESULT_RELATIVE_PATH,
        cache_path=tmp_path / mod.CACHE_RELATIVE_PATH,
        source_loader=_source_loader(_source_rows(1), tmp_path / "source.jsonl"),
        limit=1,
        min_questions=1,
        write=True,
        now=lambda: 300.0,
    )

    bad_row_errors = mod.validate_cache_row(
        {
            "schema": "bad",
            "corpus": "bad",
            "candidates": [None, {}],
            "solver_provenance": {},
            "verifier_provenance": {"verifier_is_oracle": True},
        }
    )
    empty_candidate_errors = mod.validate_cache_row(
        {
            **mod.build_candidate_cache_row(
                _source_rows(1)[0],
                variant_index=0,
                source_path=tmp_path / "source.jsonl",
            ),
            "candidates": [],
        }
    )

    assert {
        "schema",
        "corpus",
        "row_id",
        "question",
        "gold",
        "candidate_object",
        "candidate_id",
        "candidate_answer",
        "label_correct",
        "solver_verdict",
        "gold_missing_from_candidates",
        "solver_provenance",
        "verifier_provenance",
    }.issubset(set(bad_row_errors))
    assert "candidates" in empty_candidate_errors

    mutations = (
        ({key: value for key, value in artifact.items() if key != "duration_s"}, "duration_s"),
        ({**artifact, "schema": "wrong"}, "schema"),
        ({**artifact, "verifier_is_oracle": True}, "verifier_is_oracle"),
        ({**artifact, "model_specs": []}, "model_specs"),
        ({**artifact, "second_corpus_cache_built": "yes"}, "second_corpus_cache_built"),
        ({**artifact, "headroom_present": "yes"}, "headroom_present"),
        ({**artifact, "n_questions": -1}, "n_questions"),
        ({**artifact, "n_candidate_rows": "many"}, "n_candidate_rows"),
        ({**artifact, "genuine_sc_accuracy": True}, "genuine_sc_accuracy"),
        ({**artifact, "oracle_at_k": "bad"}, "oracle_at_k"),
        ({**artifact, "candidate_cache_path": ""}, "candidate_cache_path"),
        ({**artifact, "field_principles": []}, "field_principles"),
        ({**artifact, "honest_verdict": "maybe"}, "honest_verdict"),
    )
    for mutated, field in mutations:
        assert field in mod.artifact_schema_errors(mutated)


def test_scenario_verify_5044_blocked_paths_write_artifacts(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-5044: unavailable fallback sources fail closed."""

    def raising_loader(
        *, root: Path
    ) -> tuple[list[dict[str, Any]], Path, list[mod.PreconditionCheck]]:
        raise RuntimeError(f"bad root {root}")

    raised = mod.run(
        root=tmp_path / "raised",
        artifact_path=tmp_path / "raised.json",
        cache_path=tmp_path / "raised.jsonl",
        source_loader=raising_loader,
        limit=1,
        min_questions=1,
        write=True,
        now=lambda: 400.0,
    )
    empty = mod.run(
        root=tmp_path / "empty",
        artifact_path=tmp_path / "empty.json",
        cache_path=tmp_path / "empty.jsonl",
        source_loader=lambda root: (
            [],
            Path(root) / "empty-source.jsonl",
            [mod.PreconditionCheck("empty_source", False, "no rows")],
        ),
        limit=1,
        min_questions=1,
        write=True,
        now=lambda: 500.0,
    )

    assert raised["honest_verdict"] == "blocked_second_corpus_source_unavailable"
    assert raised["second_corpus_cache_built"] is False
    assert "RuntimeError" in raised["preconditions_checked"][0]["detail"]
    assert json.loads((tmp_path / "raised.json").read_text(encoding="utf-8")) == raised
    assert empty["honest_verdict"] == "blocked_second_corpus_source_unavailable"
    assert empty["preconditions_checked"][0]["resource"] == "empty_source"
    assert mod.artifact_schema_errors(raised) == []
    assert mod.artifact_schema_errors(empty) == []
