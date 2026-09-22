"""Tests for REQ-JEV-TETRIS-001 local Tetris option readout."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from scripts.experiments.jev_tetris_semif.readout_server import (
    DELEGATE_OPTION_ID,
    Option,
    assign_single_token_labels,
    build_prompt,
    evaluate_request,
    select_readout_options,
)


class FakeTokenizer:
    def __init__(self, split_labels: set[str] | None = None) -> None:
        self.split_labels = split_labels or set()

    def __call__(self, text: str) -> list[int]:
        label = text.strip()
        if label in self.split_labels:
            return [1, 2]
        return [100 + ord(label)]


class FakeScorer:
    model_spec = {
        "hf_id": "unsloth/Qwen3.5-9B-GGUF",
        "model_path": "/cache/qwen.gguf",
        "sha256": "abc123",
    }

    def __init__(self, preferred: str) -> None:
        self.preferred = preferred
        self.calls: list[tuple[object, str, tuple[Option, ...]]] = []

    def score(self, state: object, question: str, options: Sequence[Option]) -> dict[str, float]:
        frozen = tuple(options)
        self.calls.append((state, question, frozen))
        weights = {option.option_id: 0.01 for option in frozen}
        weights[self.preferred] = 0.9
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}


def criteria(count: int) -> dict[str, str]:
    return {f"p{index}": f"Placement sentence {index}." for index in range(count)}


def placement_evaluations(count: int, *, best_id: str | None = None) -> dict[str, dict[str, int]]:
    evaluations = {
        f"p{index}": {
            "holesCreated": index,
            "bumpinessDelta": index,
            "maxHeight": index,
            "aggregateHeight": index,
            "cleared": 0,
        }
        for index in range(count)
    }
    if best_id is not None:
        evaluations[best_id] = {
            "holesCreated": 0,
            "bumpinessDelta": -1,
            "maxHeight": 0,
            "aggregateHeight": 0,
            "cleared": 4,
        }
    return evaluations


def test_single_token_labels_skip_split_tokens() -> None:
    """SCENARIO-JEV-TETRIS-001-A: labels are distinct single tokens."""

    labels = assign_single_token_labels(FakeTokenizer({"B", "D"}), 4)

    assert [label.display for label in labels] == ["A", "C", "E", "F"]
    assert len({label.token_id for label in labels}) == 4


def test_prompt_contains_state_instruction_and_labeled_options() -> None:
    """SCENARIO-JEV-TETRIS-001-A: prompt construction is explicit."""

    options = (Option("p0", "Keeps the stack low."), Option("p1", "Clears one line."))
    labels = assign_single_token_labels(FakeTokenizer(), 2)
    prompt = build_prompt(
        {"shape": "The surface is level.", "piece": "The piece is the T."},
        "Which option keeps the stack low",
        options,
        labels,
    )

    assert "The surface is level." in prompt
    assert "Which option keeps the stack low" in prompt
    assert "A: p0" in prompt and "Keeps the stack low." in prompt
    assert "B: p1" in prompt and "Clears one line." in prompt
    assert prompt.endswith("Answer:")


def test_overflow_ranks_evaluations_before_top_15_and_delegate() -> None:
    """SCENARIO-JEV-TETRIS-001-B: quality ranking precedes the option cap."""

    selection = select_readout_options(criteria(20), placement_evaluations(20, best_id="p19"))

    assert [option.option_id for option in selection.options[:15]] == [
        "p19",
        *(f"p{index}" for index in range(14)),
    ]
    assert selection.options[15].option_id == DELEGATE_OPTION_ID
    assert selection.delegate_target == "p14"
    assert selection.tail_ids == ("p14", "p15", "p16", "p17", "p18")


def test_overflow_delegate_probability_maps_to_incumbent_tail() -> None:
    """SCENARIO-JEV-TETRIS-001-B: delegate mass maps to one legal placement."""

    scorer = FakeScorer(DELEGATE_OPTION_ID)
    raw = evaluate_request(
        {
            "state": {"shape": "rough"},
            "questions": {
                "sealed": {
                    "type": "choice",
                    "instructions": "Avoid trapped space",
                    "criteria": criteria(20),
                    "placement_evaluations": placement_evaluations(20),
                }
            },
        },
        scorer,
    )

    answer = raw["answers"]["sealed"]
    assert answer["choice"] == "p15"
    assert set(answer["probabilities"]) == set(criteria(20))
    assert answer["probabilities"]["p15"] > 0.8
    assert all(answer["probabilities"][f"p{index}"] == 0.0 for index in range(16, 20))


def test_response_shape_skips_ambient_questions(tmp_path: Path) -> None:
    """SCENARIO-JEV-TETRIS-001-C: response matches readChoice and skips ambient."""

    marker = tmp_path / "writer-must-stay-here.txt"
    scorer = FakeScorer("p1")
    raw = evaluate_request(
        {
            "state": {"room": "There is room."},
            "questions": {
                "sealed": {
                    "type": "choice",
                    "instructions": "Avoid trapped space",
                    "criteria": criteria(3),
                },
                "danger": {"type": "score", "criteria": ["safe", "bad"]},
                "doomed": {"type": "noul"},
            },
        },
        scorer,
    )
    marker.write_text("test-only\n")

    answer = raw["answers"]["sealed"]
    assert set(answer) == {"choice", "probabilities", "confidence"}
    assert answer["choice"] == "p1"
    assert set(raw["answers"]) == {"sealed"}
    assert raw["provenance"]["model"]["hf_id"] == "unsloth/Qwen3.5-9B-GGUF"
    assert raw["provenance"]["ambient_questions"] == "skipped_not_game_relevant"
    assert raw["provenance"]["voice_execution"] == "sequential"


def test_invalid_request_is_rejected() -> None:
    """REQ-JEV-TETRIS-001: malformed requests fail before scoring."""

    with pytest.raises(ValueError, match="questions"):
        evaluate_request({"state": {}}, FakeScorer("p0"))
