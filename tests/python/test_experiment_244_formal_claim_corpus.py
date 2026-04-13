"""Spec: REQ-VERIFY-056, REQ-VERIFY-057, SCENARIO-VERIFY-063, SCENARIO-VERIFY-064."""

from __future__ import annotations

import importlib.util
import json
import os
import runpy
from pathlib import Path

import pytest


def load_module():
    repo_root = Path(__file__).resolve().parents[2]
    module_path = repo_root / "scripts" / "experiment_244_formal_claim_corpus.py"
    spec = importlib.util.spec_from_file_location(
        "experiment_244_formal_claim_corpus",
        module_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )


def write_source_fixtures(repo: Path) -> None:
    write_json(
        repo / "results" / "experiment_221_results.json",
        {
            "run_date": "20260412",
            "paired_runs": [
                {
                    "model_name": "Qwen3.5-0.8B",
                    "model_hf_id": "Qwen/Qwen3.5-0.8B",
                    "mode": "verify_only",
                    "cases": [
                        {
                            "case_id": "exp211-code-score-2",
                            "flagged": True,
                            "response_mode": "answer_only_terse",
                            "response": (
                                "def score_casefold_keywords(text: str, "
                                "weights: dict[str, int]) -> int:\n"
                                "    return 0"
                            ),
                            "evaluation": {
                                "example_id": "exp211-code-score-2",
                                "exact_satisfaction": False,
                                "parseable": True,
                                "semantic_violation_count": 0,
                                "source_family": "code_typed_properties",
                                "task_slice": "code_typed_properties",
                                "raw_response": (
                                    "def score_casefold_keywords(text: str, "
                                    "weights: dict[str, int]) -> int:\n"
                                    "    return 0"
                                ),
                                "constraint_results": [
                                    {
                                        "constraint_id": "c1",
                                        "type": "function_name",
                                        "family": "literal",
                                        "status": "satisfied",
                                        "judge": "deterministic",
                                        "details": {},
                                    },
                                    {
                                        "constraint_id": "c4",
                                        "type": "semantic_property",
                                        "family": "search_optimization_limited",
                                        "status": "violated",
                                        "judge": "deterministic",
                                        "details": {},
                                    },
                                ],
                            },
                        }
                    ],
                }
            ],
        },
    )
    write_json(
        repo / "results" / "experiment_235_results.json",
        {
            "run_date": "20260413",
            "paired_runs": [
                {
                    "model_name": "Qwen3.5-0.8B",
                    "model_hf_id": "Qwen/Qwen3.5-0.8B",
                    "mode": "verify_only",
                    "cases": [
                        {
                            "case_id": "gsm8k-178",
                            "correct": False,
                            "flagged": True,
                            "response_mode": "grammar_gated_json",
                            "response": '{"final_answer":122}',
                            "verification": {
                                "typed_reasoning": {
                                    "question": (
                                        "Rani has ten more crabs than Monic, "
                                        "who has 4 fewer crabs than Bo. If Bo "
                                        "has 40 crabs, calculate the total "
                                        "number of crabs the three have together."
                                    ),
                                    "user_constraints": [
                                        {
                                            "constraint_id": "uc1",
                                            "kind": "prompt_constraint",
                                            "text": "Rani has ten more crabs than Monic",
                                        }
                                    ],
                                    "reasoning_steps": [],
                                    "atomic_claims": [
                                        {
                                            "claim_id": "cl1",
                                            "kind": "statement",
                                            "text": "Monic has 4 fewer crabs than Bo",
                                        },
                                        {
                                            "claim_id": "cl2",
                                            "kind": "equation",
                                            "text": "40 + 36 + 46 = 122",
                                        },
                                    ],
                                    "final_answer": {
                                        "text": "122",
                                        "answer_type": "number",
                                        "normalized": 122,
                                    },
                                    "provenance": {
                                        "extraction_method": "direct_json",
                                        "source_format": "json",
                                        "parser_version": "20260412",
                                    },
                                },
                                "semantic_verifier_v2": {
                                    "question_profile": {
                                        "question": (
                                            "Rani has ten more crabs than Monic, "
                                            "who has 4 fewer crabs than Bo. If Bo "
                                            "has 40 crabs, calculate the total "
                                            "number of crabs the three have together."
                                        ),
                                        "prompt_clauses": [
                                            {
                                                "clause_id": "p1",
                                                "text": "Rani has ten more crabs than Monic",
                                            },
                                            {
                                                "clause_id": "p2",
                                                "text": "Monic has 4 fewer crabs than Bo",
                                            },
                                        ],
                                        "target_clause": {
                                            "clause_id": "target",
                                            "text": (
                                                "calculate the total number of crabs "
                                                "the three have together"
                                            ),
                                        },
                                    },
                                    "claims": [
                                        {
                                            "claim_id": "cl1",
                                            "text": "Monic has 4 fewer crabs than Bo",
                                            "is_final": False,
                                        },
                                        {
                                            "claim_id": "cl2",
                                            "text": "40 + 36 + 46 = 122",
                                            "is_final": True,
                                        },
                                    ],
                                    "claim_results": [
                                        {
                                            "claim_id": "cl1",
                                            "text": "Monic has 4 fewer crabs than Bo",
                                            "is_final": False,
                                            "status": "supported",
                                            "matched_clause_ids": ["p2"],
                                            "missing_clause_ids": [],
                                            "missing_target_keywords": [],
                                            "supporting_claim_ids": [],
                                            "legacy_violation_types": [],
                                        },
                                        {
                                            "claim_id": "cl2",
                                            "text": "40 + 36 + 46 = 122",
                                            "is_final": True,
                                            "status": "violated",
                                            "matched_clause_ids": [],
                                            "missing_clause_ids": ["p1"],
                                            "missing_target_keywords": ["together"],
                                            "supporting_claim_ids": [],
                                            "legacy_violation_types": ["missing_quantity_coverage"],
                                        },
                                    ],
                                    "focus_claim_id": "cl2",
                                    "verdict": "violated",
                                },
                            },
                        }
                    ],
                }
            ],
        },
    )
    write_jsonl(
        repo / "data" / "research" / "constraint_ir_benchmark_211.jsonl",
        [
            {
                "example_id": "exp211-code-score-2",
                "source_family": "code_typed_properties",
                "source_refs": ["formalbench-inspired"],
                "prompt": (
                    "Write `score_casefold_keywords(text: str, "
                    "weights: dict[str, int]) -> int` that performs the same "
                    "keyword scoring case-insensitively."
                ),
                "gold_atomic_constraints": [
                    {
                        "constraint_id": "c1",
                        "type": "function_name",
                        "target": "function_name",
                        "relation": "equals",
                        "value": "score_casefold_keywords",
                    },
                    {
                        "constraint_id": "c4",
                        "type": "semantic_property",
                        "target": "scoring_rule",
                        "relation": "equals",
                        "value": "casefold_keyword_presence",
                    },
                ],
                "expected_verifier_path": "code_ir.typed_contracts_plus_execution",
            }
        ],
    )
    write_jsonl(
        repo / "data" / "research" / "semantic_failure_corpus_214.jsonl",
        [
            {
                "example_id": "exp214-live-923",
                "source_type": "live_trace",
                "source_artifact": "exp203_live",
                "source_refs": ["exp203:923", "exp206:923", "exp207:923"],
                "domain": "word_problem",
                "prompt": (
                    "Lana has 27 cups. After using 15 cups for cinnamon tea, "
                    "the rest are split into 3 rows with equal chamomile and mint. "
                    "How many mint cups are in each row?"
                ),
                "response": "27 - 15 = 12\n12 / 3 = 4\nAnswer: 4",
                "gold_diagnosis": {
                    "taxonomy_label": "omitted_premises",
                    "failure_mechanism": "The response stops at cups per row.",
                    "expected_outcome": "2",
                    "observed_outcome": "4",
                },
                "expected_verifier_signal": {
                    "verifier_path": "question_grounding.quantity_graph",
                    "signal_summary": "Need the final mint-per-row split.",
                    "should_flag": True,
                },
            }
        ],
    )


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


# REQ-VERIFY-056, SCENARIO-VERIFY-063
def test_build_corpus_normalizes_live_claims_and_preserves_localization(
    tmp_path: Path,
) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)

    rows = module.build_corpus(repo)
    assert rows == module.build_corpus(repo)

    prompt_row = next(
        row
        for row in rows
        if row["provenance"]["source_artifact"] == "results/experiment_221_results.json"
        and row["claim"]["claim_id"] == "c4"
    )
    assert prompt_row["schema_version"] == "carnot.formal_claim_corpus.v1"
    assert prompt_row["run_date"] == "20260413"
    assert prompt_row["source_family"] == "prompt_side_live_trace"
    assert prompt_row["claim"]["relation_type"] == "equals"
    assert prompt_row["claim"]["candidate_solver_route"] == "execution_oracle"
    assert prompt_row["claim"]["formalization_status"] == "formalized"
    assert prompt_row["gold_verdict"] == "violated"
    assert prompt_row["localization"]["seed_constraint_ids"] == ["c4"]

    semantic_row = next(
        row
        for row in rows
        if row["provenance"]["source_artifact"] == "results/experiment_235_results.json"
        and row["claim"]["claim_id"] == "cl2"
    )
    assert semantic_row["source_family"] == "semantic_live_trace"
    assert semantic_row["claim"]["relation_type"] == "equation"
    assert semantic_row["claim"]["candidate_solver_route"] == "arithmetic"
    assert semantic_row["claim"]["operands"] == [40.0, 36.0, 46.0, 122.0]
    assert semantic_row["gold_verdict"] == "violated"
    assert semantic_row["localization"]["missing_clause_ids"] == ["p1"]
    assert semantic_row["localization"]["taxonomy_hint"] == "missing_quantity_coverage"

    abstain_row = next(
        row
        for row in rows
        if row["provenance"]["source_artifact"] == "data/research/semantic_failure_corpus_214.jsonl"
        and row["claim"]["formalization_status"] == "not_formalizable"
    )
    assert abstain_row["source_family"] == "semantic_failure_live_trace"
    assert abstain_row["gold_verdict"] == "abstain"
    assert abstain_row["localization"]["taxonomy_label"] == "omitted_premises"


# REQ-VERIFY-057, SCENARIO-VERIFY-064
def test_build_results_reports_route_formalization_and_source_breakdown(
    tmp_path: Path,
) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)

    rows = module.build_corpus(repo)
    results = module.build_results(rows)

    assert results["experiment"] == "Exp 244"
    assert results["run_date"] == "20260413"
    assert results["summary"]["n_rows"] == len(rows)
    assert results["summary"]["route_counts"]["arithmetic"] >= 1
    assert results["summary"]["route_counts"]["execution_oracle"] == 1
    assert results["summary"]["formalization_status_counts"]["formalized"] >= 3
    assert results["summary"]["formalization_status_counts"]["not_formalizable"] >= 1
    assert results["summary"]["gold_verdict_counts"]["violated"] >= 2
    assert results["summary"]["source_breakdown"] == {
        "prompt_side_live_trace": 2,
        "semantic_failure_live_trace": 3,
        "semantic_live_trace": 2,
    }
    assert results["summary"]["formalizable_rate"] == round(
        results["summary"]["formalization_status_counts"]["formalized"] / len(rows),
        6,
    )


# REQ-VERIFY-056, REQ-VERIFY-057, SCENARIO-VERIFY-064
def test_main_writes_corpus_and_summary_idempotently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    module = load_module()
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)

    monkeypatch.setattr(module, "REPO_ROOT", repo)
    monkeypatch.setattr(
        module, "CORPUS_PATH", repo / "data" / "research" / "formal_claim_corpus_244.jsonl"
    )
    monkeypatch.setattr(module, "RESULTS_PATH", repo / "results" / "experiment_244_results.json")

    assert module.main() == 0
    first_corpus = (repo / "data" / "research" / "formal_claim_corpus_244.jsonl").read_text(
        encoding="utf-8"
    )
    first_results = (repo / "results" / "experiment_244_results.json").read_text(encoding="utf-8")
    assert module.main() == 0

    assert (repo / "data" / "research" / "formal_claim_corpus_244.jsonl").read_text(
        encoding="utf-8"
    ) == first_corpus
    assert (repo / "results" / "experiment_244_results.json").read_text(
        encoding="utf-8"
    ) == first_results

    corpus = read_jsonl(repo / "data" / "research" / "formal_claim_corpus_244.jsonl")
    results = json.loads(
        (repo / "results" / "experiment_244_results.json").read_text(encoding="utf-8")
    )
    assert len(corpus) == results["summary"]["n_rows"]
    assert all(row["provenance"]["source_artifact"] for row in corpus)


# REQ-VERIFY-057, SCENARIO-VERIFY-064
def test_cli_entrypoint_honors_repo_override(tmp_path: Path, monkeypatch) -> None:
    repo = make_repo(tmp_path)
    write_source_fixtures(repo)
    module_path = (
        Path(__file__).resolve().parents[2] / "scripts" / "experiment_244_formal_claim_corpus.py"
    )

    monkeypatch.setenv("CARNOT_REPO_ROOT", str(repo))
    monkeypatch.setattr("sys.argv", [str(module_path)])

    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit as exc:
        assert exc.code == 0

    assert (repo / "data" / "research" / "formal_claim_corpus_244.jsonl").exists()
    assert (repo / "results" / "experiment_244_results.json").exists()
    assert os.environ["CARNOT_REPO_ROOT"] == str(repo)


# REQ-VERIFY-056, REQ-VERIFY-057
def test_helper_edge_paths_cover_conservative_fallbacks(tmp_path: Path) -> None:
    module = load_module()

    outside = Path("/tmp/exp244.json")
    assert module.resolve_path(tmp_path, outside) == outside

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected JSON object"):
        module.load_json(bad_json)

    mixed_jsonl = tmp_path / "mixed.jsonl"
    mixed_jsonl.write_text('{"ok": 1}\n[]\n{"ok": 2}\n', encoding="utf-8")
    assert module.load_jsonl(mixed_jsonl) == [{"ok": 1}, {"ok": 2}]

    json_path = tmp_path / "nested" / "artifact.json"
    module.write_json(json_path, {"ok": True})
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"ok": True}

    assert module.slugify("Qwen 3.5 / Test") == "qwen-3-5-test"
    assert module._ordered_unique(["a", "", "a", "b"]) == ["a", "b"]
    assert module._parse_number("") is None
    assert module._parse_number("3/2") == 1.5
    assert module._parse_number("1/0") is None
    assert module._parse_number("oops") is None
    assert module._extract_numbers("values 1 3/2 -2") == [1.0, 1.5, -2.0]
    assert module._extract_identifiers("Answer total_value and total_value") == ["total_value"]

    assert module._safe_eval_expression("") is None
    assert module._safe_eval_expression("a + 1") is None
    assert module._safe_eval_expression(")") is None
    assert module._safe_eval_expression("-2") == -2.0
    assert module._safe_eval_expression("2*3") == 6.0
    assert module._safe_eval_expression("2**3") is None
    assert module._safe_eval_expression("1/0") is None

    assert module._equation_details("no equation") is None
    assert module._equation_details("1 + 2 = x") is None
    assert module._equation_details("1 + 2 = 1/0") is None
    assert module._equation_details("1 + (2 = 3") is None
    assert module._comparison_details("At least 3 remain")["relation_type"] == "greater_or_equal"
    assert module._comparison_details("No comparator") is None
    assert module._attribute_details("Alpha has 1/0") is None
    assert module._attribute_details("No numeric attribute here") is None
    assert module.normalize_semantic_claim("", is_final=False)["relation_type"] == "empty_claim"
    assert module.normalize_semantic_claim("Answer: 5", is_final=True)["target"] == "final_answer"

    assert module._value_is_arithmetic(4) is True
    assert module._value_is_arithmetic(["x", 1]) is True
    assert module._value_is_arithmetic({"x": 1}) is False
    assert module._constraint_route("count_exact", "equals", 1) == "cardinality"
    assert module._constraint_route("enum_membership", "in", ["a"]) == "set_membership"
    assert module._constraint_route("function_name", "equals", "name") == "boolean_entailment"
    assert module._constraint_route("derived_quantity", "equals", "x + 1") == "arithmetic"

    constraint = {
        "constraint_id": "c1",
        "type": "enum_membership",
        "target": "status",
        "relation": "in",
        "value": {"allowed": ["green", "red"]},
    }
    normalized = module.normalize_prompt_constraint(constraint)
    assert normalized["candidate_solver_route"] == "set_membership"
    assert (
        module._claim_text_from_constraint(constraint) == "status in {'allowed': ['green', 'red']}"
    )
    assert module._gold_verdict_from_status("unknown") == "abstain"
    built = module.build_row(
        row_id="exp244-test",
        source_family="prompt_side_live_trace",
        prompt="Prompt",
        response="Response",
        claim_id="c1",
        claim_role="prompt_constraint",
        claim_text="status in green",
        normalized_claim=normalized,
        gold_verdict="abstain",
        localization={"seed_constraint_ids": ["c1"]},
        provenance={"source_artifact": "fixture"},
    )
    assert built["claim"]["formalization_status"] == "formalized"

    exp221_rows = module.build_exp221_rows(
        {
            "run_date": "20260412",
            "paired_runs": [
                {"mode": "baseline", "cases": []},
                {
                    "mode": "verify_only",
                    "model_name": "Model",
                    "model_hf_id": "hf/model",
                    "cases": [
                        {
                            "case_id": "case-a",
                            "response": "ok",
                            "evaluation": {"constraint_results": ["skip-me"]},
                        }
                    ],
                },
            ],
        },
        {"case-a": {"prompt": "Prompt", "gold_atomic_constraints": []}},
    )
    assert exp221_rows == []

    exp235_rows = module.build_exp235_rows(
        {
            "run_date": "20260413",
            "paired_runs": [
                {"mode": "baseline", "cases": []},
                {
                    "mode": "verify_only",
                    "model_name": "Model",
                    "model_hf_id": "hf/model",
                    "cases": [
                        {
                            "case_id": "case-b",
                            "response": "ok",
                            "verification": {
                                "semantic_verifier_v2": {
                                    "claims": [],
                                    "claim_results": ["skip-me"],
                                }
                            },
                        }
                    ],
                },
            ],
        }
    )
    assert exp235_rows == []

    exp214_rows = module.build_exp214_rows(
        [
            {"source_type": "targeted_follow_up"},
            {
                "source_type": "live_trace",
                "example_id": "exp214-live-test",
                "prompt": "Prompt",
                "response": "\nAnswer: 5\n",
                "gold_diagnosis": {"taxonomy_label": "omitted_premises"},
                "expected_verifier_signal": {"verifier_path": "question_grounding.quantity_graph"},
            },
        ]
    )
    assert len(exp214_rows) == 1
    assert exp214_rows[0]["gold_verdict"] == "abstain"

    assert module.build_results([])["summary"]["formalizable_rate"] == 0.0
