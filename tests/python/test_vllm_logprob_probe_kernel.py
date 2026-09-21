"""Tests for the pure helpers in the REQ-INFRA-7089 Kaggle probe."""

from __future__ import annotations

import importlib.util
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PROBE_DIR = ROOT / "scripts" / "kaggle" / "vllm_logprob_probe_kernel"
SPEC = importlib.util.spec_from_file_location("vllm_logprob_probe", PROBE_DIR / "main.py")
assert SPEC is not None
assert SPEC.loader is not None
PROBE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROBE)


def test_softmax_uses_only_declared_options() -> None:
    """REQ-INFRA-7089: unrelated returned tokens must not change option probabilities."""
    probabilities = PROBE.softmax_declared_options(
        {" A": -2.0, " B": -1.0, " unrelated": 50.0}, [" A", " B"]
    )

    assert set(probabilities) == {" A", " B"}
    assert math.isclose(sum(probabilities.values()), 1.0)
    assert probabilities[" B"] > probabilities[" A"]


def test_schema_field_detection_resolves_completion_request() -> None:
    """REQ-INFRA-7089: the probe must detect fields from the served schema."""
    fake_openapi = {
        "paths": {
            "/v1/completions": {
                "post": {
                    "requestBody": {
                        "content": {
                            "application/json": {
                                "schema": {"$ref": "#/components/schemas/CompletionRequest"}
                            }
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "CompletionRequest": {
                    "type": "object",
                    "properties": {
                        "logprobs": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                        "allowed_token_ids": {
                            "type": "array",
                            "items": {"type": "integer"},
                        },
                        "structured_outputs": {"$ref": "#/components/schemas/Structured"},
                    },
                },
                "Structured": {"type": "object"},
            }
        },
    }

    detected = PROBE.detect_completion_schema_fields(fake_openapi)

    assert detected["properties"]["logprobs"] == "integer|null"
    assert detected["properties"]["allowed_token_ids"] == "array[integer]"
    assert detected["supported"]["logprobs"] is True
    assert detected["supported"]["prompt_logprobs"] is False
    assert detected["supported"]["structured_decoding_fields"] == ["structured_outputs"]


def test_prompt_fixtures_are_valid_and_bounded(monkeypatch) -> None:
    """SCENARIO-INFRA-7089-PARTIAL-EVIDENCE: frozen prompts remain self-checking."""
    fixtures = PROBE.load_prompt_fixtures(PROBE_DIR / "fixtures.json")
    validation = PROBE.validate_prompt_fixtures(fixtures)

    assert validation["valid"] is True
    assert validation["count"] >= 40
    assert validation["unique_ids"] is True
    assert validation["answers_in_options"] is True
    assert validation["option_count_range"] == [2, 6]
    assert validation["labels_within_limit"] is True
    assert PROBE.embedded_prompt_fixtures() == fixtures
    monkeypatch.setattr(PROBE, "FIXTURE_PATH", PROBE_DIR / "not-uploaded.json")
    assert PROBE.load_prompt_fixtures() == fixtures
