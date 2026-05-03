"""Tests for Exp 1169 FoVer SOTA expansion helpers.

Spec: REQ-VERIFY-1169, SCENARIO-VERIFY-1169
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import carnot.eval.fover_sota_expansion_v6 as exp1169
from carnot.eval.fover_sota_expansion_v6 import (
    REQUIRED_ARTIFACT_FIELDS,
    CaseSpec,
    append_rows_jsonl,
    assign_sc_energy_label,
    answer_matches,
    build_artifact,
    build_labeled_rows,
    build_prompt,
    build_source_plan,
    inject_adversarial_step,
    latest_fover_corpus_size,
)


class FakeZ3:
    def score(self, text: str) -> float:
        return 1.0 if "2 + 2 = 5" in text or "1 + 1 = 3" in text else 0.0


class FakeAST:
    def score(self, text: str) -> float:
        return 0.0 if text.strip() else 1.0


class FakeSemantic:
    def score(self, text: str) -> float:
        return 1.0 if "Contradiction:" in text else 0.0


class IndeterminateZ3:
    def score(self, text: str) -> float:
        return 0.5


def test_latest_fover_corpus_size_ignores_invalid_newer_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1169-1: discover latest valid FoVer n_pairs without trusting invalid JSON."""
    (tmp_path / "notes.json").write_text(json.dumps({"n_pairs_after": 9999}))
    (tmp_path / "fover_bad.json").write_text("{")
    (tmp_path / "fover_list.json").write_text(json.dumps([{}, {}, {}]))
    (tmp_path / "fover_pairs.json").write_text(json.dumps({"pairs": [{}, {}, {}, {}]}))
    (tmp_path / "fover_v2_combined.json").write_text(
        json.dumps({"n_total_pairs": 1400, "pairs": [{}] * 2})
    )
    prior = tmp_path / "experiment_1119_fover_sota_extension_v5.json"
    prior.write_text(json.dumps({"n_pairs_after": 7329, "honest_verdict": "ok"}))
    (tmp_path / "experiment_1169_fover_sota_expansion_v6.json").write_text(
        json.dumps({"honest_verdict": "blocked_gate_check_failed"})
    )

    path, n_pairs = latest_fover_corpus_size(tmp_path)

    assert path == prior
    assert n_pairs == 7329


def test_latest_fover_corpus_size_can_exclude_current_artifact(tmp_path: Path) -> None:
    """REQ-VERIFY-1169-1: reruns can exclude Exp 1169 from prior-corpus discovery."""
    prior = tmp_path / "experiment_1119_fover_sota_extension_v5.json"
    current = tmp_path / "experiment_1169_fover_sota_expansion_v6.json"
    prior.write_text(json.dumps({"n_pairs_after": 7329}))
    current.write_text(json.dumps({"total_corpus_size": 8329}))

    path, n_pairs = latest_fover_corpus_size(tmp_path, exclude_paths={current})

    assert path == prior
    assert n_pairs == 7329


def test_latest_fover_corpus_size_raises_without_valid_json(tmp_path: Path) -> None:
    """REQ-VERIFY-1169-1: missing valid FoVer JSON is reported explicitly."""
    (tmp_path / "fover_empty.json").write_text(json.dumps({"honest_verdict": "no_count"}))

    with pytest.raises(FileNotFoundError):
        latest_fover_corpus_size(tmp_path)


def test_pair_count_rejects_non_collection_shapes() -> None:
    """REQ-VERIFY-1169-1: non-dict/non-list JSON shapes are not valid corpus counts."""
    assert exp1169._pair_count("not a corpus") is None


def test_build_source_plan_has_required_source_counts() -> None:
    """REQ-VERIFY-1169-2: source plan is exactly 200/100/200."""
    cases = build_source_plan()
    counts = {
        source: sum(1 for case in cases if case.source == source)
        for source in {c.source for c in cases}
    }

    assert len(cases) == 500
    assert counts == {"gsm8k": 200, "humaneval": 100, "arc_challenge": 200}


def test_build_prompt_includes_gold_fields_for_labelable_generation() -> None:
    """REQ-VERIFY-1169-2: prompts carry source-specific gold targets for verification."""
    case = CaseSpec(
        case_id="arc_1",
        source="arc_challenge",
        question="Which choice is correct?",
        answer="B",
        choices=["A. cold", "B. hot"],
    )

    prompt = build_prompt(case)

    assert "arc_challenge" in prompt
    assert "Verified answer: B" in prompt
    assert "A. cold" in prompt
    assert "Step 2" in prompt


def test_build_prompt_includes_humaneval_reference_solution() -> None:
    """REQ-VERIFY-1169-2: HumanEval prompts can carry canonical code for label checks."""
    case = CaseSpec(
        case_id="HumanEval/0",
        source="humaneval",
        question="def add(a, b):",
        answer="add",
        canonical_solution="    return a + b",
    )

    prompt = build_prompt(case)

    assert "Reference implementation" in prompt
    assert "return a + b" in prompt


def test_assign_sc_energy_label_requires_all_coherent_checks() -> None:
    """REQ-VERIFY-1169-4: coherent means Z3 pass, AST valid, and no contradiction."""
    assert assign_sc_energy_label(True, True, True, True) == "coherent"
    assert assign_sc_energy_label(False, True, True, True) == "incoherent"
    assert assign_sc_energy_label(True, False, True, True) == "incoherent"
    assert assign_sc_energy_label(True, True, False, True) == "incoherent"
    assert assign_sc_energy_label(True, True, True, False) == "incoherent"


def test_inject_adversarial_step_adds_z3_and_semantic_failure() -> None:
    """REQ-VERIFY-1169-3: adversarial rows contain an injected step-2/step-3 error."""
    response = "Step 1: Set up.\nStep 2: 2 + 2 = 4.\nStep 3: Final answer: 4."

    adversarial = inject_adversarial_step(response)

    assert "Step 2" in adversarial
    assert "2 + 2 = 5" in adversarial
    assert "Contradiction:" in adversarial


def test_inject_adversarial_step_appends_when_steps_absent() -> None:
    """REQ-VERIFY-1169-3: unstructured model output still receives an adversarial step."""
    adversarial = inject_adversarial_step("Final answer: 4.")

    assert adversarial.startswith("Final answer: 4.")
    assert "Step 2: 2 + 2 = 5" in adversarial


def test_answer_matches_handles_empty_and_text_answers() -> None:
    """REQ-VERIFY-1169-4: answer matching supports empty, numeric, and text targets."""
    assert answer_matches("Any response", None) is True
    assert answer_matches("Final answer: B", "B") is True
    assert answer_matches("Final answer: C", "B") is False
    assert exp1169._parse_number("not-a-number") is None


def test_build_labeled_rows_emits_standard_and_adversarial_rows() -> None:
    """REQ-VERIFY-1169-3/4: one standard row and one incoherent adversarial row are emitted."""
    case = CaseSpec(
        case_id="gsm8k_1",
        source="gsm8k",
        question="What is 2 + 2?",
        answer="4",
    )
    response = "Step 1: Identify the sum.\nStep 2: 2 + 2 = 4.\nStep 3: Final answer: 4."

    rows = build_labeled_rows(
        case,
        response,
        "unsloth/Qwen3.6-35B-A3B-GGUF",
        z3_verifier=FakeZ3(),
        ast_verifier=FakeAST(),
        semantic_verifier=FakeSemantic(),
    )

    assert [row["response_kind"] for row in rows] == ["standard", "adversarial"]
    assert rows[0]["sc_energy_label"] == "coherent"
    assert rows[0]["z3_pass"] is True
    assert rows[0]["ast_valid"] is True
    assert rows[1]["sc_energy_label"] == "incoherent"
    assert rows[1]["z3_pass"] is False
    assert rows[1]["semantic_contradiction_detected"] is True


def test_build_labeled_rows_marks_indeterminate_z3_incoherent() -> None:
    """REQ-VERIFY-1169-4: indeterminate Z3 is not treated as coherent."""
    case = CaseSpec(
        case_id="gsm8k_2",
        source="gsm8k",
        question="What is 2 + 2?",
        answer="4",
    )

    rows = build_labeled_rows(
        case,
        "Step 1: Think.\nStep 2: 2 + 2 = 4.\nStep 3: Final answer: 4.",
        "model",
        z3_verifier=IndeterminateZ3(),
        ast_verifier=FakeAST(),
        semantic_verifier=FakeSemantic(),
    )

    assert rows[0]["z3_pass"] is None
    assert rows[0]["sc_energy_label"] == "incoherent"


def test_append_rows_jsonl_is_additive(tmp_path: Path) -> None:
    """REQ-VERIFY-1169-1: append path preserves existing corpus lines."""
    corpus = tmp_path / "fover_corpus.jsonl"
    corpus.write_text(json.dumps({"row_id": "old"}) + "\n")

    written = append_rows_jsonl(corpus, [{"row_id": "new", "sc_energy_label": "coherent"}])

    assert written == 1
    rows = [json.loads(line) for line in corpus.read_text().splitlines()]
    assert [row["row_id"] for row in rows] == ["old", "new"]


def test_build_artifact_has_required_schema_and_counts() -> None:
    """REQ-VERIFY-1169-5: artifact exposes required fields and label breakdown."""
    rows = [
        {"source": "gsm8k", "sc_energy_label": "coherent", "z3_pass": True},
        {"source": "gsm8k", "sc_energy_label": "incoherent", "z3_pass": False},
        {"source": "humaneval", "sc_energy_label": "incoherent", "z3_pass": None},
    ]

    artifact = build_artifact(
        rows,
        prior_n_pairs=7329,
        current_corpus_size=7332,
        latest_corpus_path=Path("results/experiment_1119_fover_sota_extension_v5.json"),
        models_used=["unsloth/Qwen3.6-35B-A3B-GGUF"],
        models_unavailable=["unsloth/gemma-4-26B-A4B-it-GGUF"],
        batch_log=[{"batch_id": 0, "batch_size": 2, "batch_time_s": 0.1}],
        duration_s=1.5,
    )

    assert REQUIRED_ARTIFACT_FIELDS <= set(artifact)
    assert artifact["n_new_pairs"] == 3
    assert artifact["n_coherent"] == 1
    assert artifact["n_incoherent"] == 2
    assert artifact["n_z3_labeled"] == 2
    assert artifact["n_sc_energy_labeled"] == 3
    assert artifact["label_breakdown"]["gsm8k"]["coherent"] == 1
    assert artifact["honest_verdict"] == "partial_500_not_reached"


def test_build_artifact_success_and_unavailable_verdicts() -> None:
    """REQ-VERIFY-1169-5: verdict enum distinguishes complete and unavailable runs."""
    rows = [{"source": "arc_challenge", "sc_energy_label": "incoherent", "z3_pass": False}]
    complete_rows = rows * 500

    complete = build_artifact(
        complete_rows,
        prior_n_pairs=7329,
        current_corpus_size=7829,
        latest_corpus_path=Path("results/experiment_1119_fover_sota_extension_v5.json"),
        models_used=["model"],
        models_unavailable=[],
        batch_log=[],
        duration_s=2.0,
    )
    unavailable = build_artifact(
        [],
        prior_n_pairs=7329,
        current_corpus_size=7329,
        latest_corpus_path=Path("results/experiment_1119_fover_sota_extension_v5.json"),
        models_used=[],
        models_unavailable=["model"],
        batch_log=[],
        duration_s=2.0,
    )

    assert complete["honest_verdict"] == "corpus_expanded_labels_complete"
    assert complete["status"] == "success"
    assert unavailable["honest_verdict"] == "gguf_model_unavailable"


def test_build_artifact_raises_if_required_field_contract_changes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-1169-5: artifact builder fails closed if required schema drifts."""
    monkeypatch.setattr(exp1169, "REQUIRED_ARTIFACT_FIELDS", {"missing_field"})

    with pytest.raises(ValueError):
        build_artifact(
            [],
            prior_n_pairs=0,
            current_corpus_size=0,
            latest_corpus_path=Path("results/fover.json"),
            models_used=[],
            models_unavailable=[],
            batch_log=[],
            duration_s=0.0,
        )
