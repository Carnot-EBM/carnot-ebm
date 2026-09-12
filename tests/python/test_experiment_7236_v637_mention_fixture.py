"""Tests for REQ-VERIFY-7236 and SCENARIO-VERIFY-7236-*.

The artifact lifecycle tests write only to a private temporary output tree.
"""

from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path

import pytest

from carnot import experiment_7236_v637_mention_fixture as mod


ROOT = Path(__file__).resolve().parents[2]


def _known(subject: str, predicate: str, obj: str, polarity: str = "positive") -> dict:
    """Build one pointer completion without adding hidden evaluator fields."""

    return {
        "outcome": "known",
        "relations": [
            {
                "subject_pointer": subject,
                "predicate": predicate,
                "object_pointer": obj,
                "polarity": polarity,
            }
        ],
    }


def _document(document_id: str, text: str) -> dict:
    """Compile one public document with the same production function as the runner."""

    return {
        "document_id": document_id,
        "text": text,
        "mentions": mod.build_mention_table(document_id, text.encode("utf-8")),
    }


def test_req_verify_7236_spec_precedes_implementation() -> None:
    """REQ-VERIFY-7236 names all required artifact fields and scenarios."""

    spec = (ROOT / mod.SPEC_PATH).read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-VERIFY-7236") :]
    for field in mod.REQUIRED_FIELD_PRINCIPLES:
        assert f"`{field}`" in section
    for scenario in ("MENTIONS", "ADAPTER", "SPLITS", "ARMS", "EQUIVARIANCE", "ARTIFACT"):
        assert f"SCENARIO-VERIFY-7236-{scenario}" in section


def test_scenario_verify_7236_mentions_reconstruct_repeated_unicode_bytes() -> None:
    """SCENARIO-VERIFY-7236-MENTIONS uses byte offsets and distinct local IDs."""

    text = "Élan precedes Miro. Élan follows 李."
    encoded = text.encode("utf-8")
    table = mod.build_mention_table("doc-u", encoded)
    assert [row["mention_id"] for row in table] == ["m000", "m001", "m002", "m003"]
    assert [row["surface_text"] for row in table] == ["Élan", "Miro", "Élan", "李"]
    for row in table:
        assert encoded[row["byte_start"] : row["byte_end"]].decode("utf-8") == row["surface_text"]
    first, second = (
        mod.resolve_pointer(encoded, table, "m000"),
        mod.resolve_pointer(encoded, table, "m002"),
    )
    assert first["byte_start"] != second["byte_start"]


def test_scenario_verify_7236_mentions_fail_closed_on_ambiguity_and_range() -> None:
    """SCENARIO-VERIFY-7236-MENTIONS makes every malformed pointer unknown."""

    encoded = b"Aster precedes Brin."
    table = mod.build_mention_table("doc", encoded)
    assert mod.resolve_pointer(encoded, table, "missing") is None
    duplicate = [*table, deepcopy(table[0])]
    assert mod.resolve_pointer(encoded, duplicate, "m000") is None
    changed = deepcopy(table)
    changed[0]["surface_text"] = "Other"
    assert mod.resolve_pointer(encoded, changed, "m000") is None
    outside = deepcopy(table)
    outside[0]["byte_end"] = len(encoded) + 1
    assert mod.resolve_pointer(encoded, outside, "m000") is None
    boolean = deepcopy(table)
    boolean[0]["byte_start"] = False
    assert mod.resolve_pointer(encoded, boolean, "m000") is None
    assert (
        mod.resolve_pointer(
            b"\xc3",
            [{"mention_id": "m000", "byte_start": 0, "byte_end": 1, "surface_text": "x"}],
            "m000",
        )
        is None
    )


def test_scenario_verify_7236_adapter_preserves_direction_polarity_and_unknown() -> None:
    """SCENARIO-VERIFY-7236-ADAPTER sends predicted semantics to the typed executor."""

    source = _document("source", "Aster precedes Brin.")
    claim = _document("claim", "Aster precedes Brin.")
    supported = mod.execute_pointer_pair(
        source,
        claim,
        _known("m000", "precedes", "m001"),
        _known("m000", "precedes", "m001"),
    )
    assert supported["decision"] == "supported" and supported["abstention"] is False
    reversed_claim = _document("claim-r", "Brin precedes Aster.")
    reversed_result = mod.execute_pointer_pair(
        source,
        reversed_claim,
        _known("m000", "precedes", "m001"),
        _known("m000", "precedes", "m001"),
    )
    assert reversed_result["decision"] == "contradicted"
    negative = mod.execute_pointer_pair(
        source,
        claim,
        _known("m000", "precedes", "m001"),
        _known("m000", "precedes", "m001", "negative"),
    )
    assert negative["decision"] == "contradicted"
    dangling = mod.execute_pointer_pair(
        source,
        claim,
        _known("missing", "precedes", "m001"),
        _known("m000", "precedes", "m001"),
    )
    assert dangling["decision"] == "unknown" and dangling["abstention"] is True


def test_scenario_verify_7236_adapter_rejects_bad_completion_shapes() -> None:
    """SCENARIO-VERIFY-7236-ADAPTER rejects invalid outcomes and semantic fields."""

    document = _document("doc", "Aster precedes Brin.")
    unknown = {"outcome": "unknown", "relations": []}
    assert mod.compile_pointer_completion(document, unknown, "source")["outcome"] == "unknown"
    cases = [
        None,
        {"outcome": "known"},
        {"outcome": "bad", "relations": []},
        {"outcome": "unknown", "relations": [{}]},
        {"outcome": "known", "relations": []},
        {"outcome": "known", "relations": [None]},
        _known("m000", "touches", "m001"),
        _known("m000", "precedes", "m001", "maybe"),
    ]
    for value in cases:
        assert mod.compile_pointer_completion(document, value, "source")["outcome"] == "unknown"
    assert (
        mod.compile_pointer_completion(document, _known("m000", "precedes", "m001"), "bad")[
            "outcome"
        ]
        == "unknown"
    )
    assert mod.compile_pointer_completion({}, _known("m000", "precedes", "m001"), "source")[
        "errors"
    ] == ["document_shape"]
    cross_document = deepcopy(document)
    cross_document["mentions"][0]["document_id"] = "other"
    assert mod.compile_pointer_completion(
        cross_document, _known("m000", "precedes", "m001"), "source"
    )["errors"] == ["cross_document_pointer"]
    two_sentences = _document("two", "Aster precedes Brin. Cora precedes Daro.")
    assert mod.compile_pointer_completion(
        two_sentences, _known("m000", "precedes", "m003"), "source"
    )["errors"] == ["pointer_sentence_mismatch"]
    with pytest.raises(ValueError, match="not unique"):
        mod._surface_pointer(document, "Missing", 0)


def test_scenario_verify_7236_splits_are_balanced_private_and_disjoint() -> None:
    """SCENARIO-VERIFY-7236-SPLITS seals balanced bases and hides authority fields."""

    public, authority = mod.build_fixture()
    assert len(public["rows"]) == len(authority["rows"]) == 72
    assert public["split_base_counts"] == {"calibration": 8, "held_out": 64}
    counts = authority["condition_counts"]
    assert counts["calibration"] == {key: 2 for key in mod.CONDITIONS}
    assert counts["held_out"] == {key: 16 for key in mod.CONDITIONS}
    public_text = mod.canonical_json(public)
    for forbidden in ("condition_key", "exact_label", "generation_seed", "gold_relation"):
        assert forbidden not in public_text
    calibration = [row for row in authority["rows"] if row["split"] == "calibration"]
    held_out = [row for row in authority["rows"] if row["split"] == "held_out"]
    assert {row["relation_phrase"] for row in calibration}.isdisjoint(
        {row["relation_phrase"] for row in held_out}
    )
    assert {name for row in calibration for name in row["entity_vocabulary"]}.isdisjoint(
        {name for row in held_out for name in row["entity_vocabulary"]}
    )
    assert all(len(row["variants"]) == 2 for row in public["rows"])


def test_scenario_verify_7236_arms_have_equal_calibration_contracts() -> None:
    """SCENARIO-VERIFY-7236-ARMS changes only the representation interface."""

    contract = mod.arm_contract()
    assert list(contract) == list(mod.ARMS)
    shared = [
        (row["calibration_unit_ids"], row["examples"], row["temperature"], row["token_caps"])
        for row in contract.values()
    ]
    assert shared.count(shared[0]) == 3
    assert all(row["unknown_allowed"] for row in contract.values())
    assert contract["explicit_schema_offset_control"]["reprompt_allowed"] is False
    assert contract["mention_pointer"]["model_predicts_numeric_offsets"] is False


def test_scenario_verify_7236_equivariance_and_positive_controls() -> None:
    """SCENARIO-VERIFY-7236-EQUIVARIANCE preserves all 72 base decisions."""

    public, authority = mod.build_fixture()
    rows = mod.execute_fixture(public, authority)
    assert len(rows) == 72 * 3
    assert all(row["metric"] == 1 and row["permutation_equivariant"] for row in rows)
    assert all(row["twin_prediction"] == row["prediction"] for row in rows)
    assert len({row["base_id"] for row in rows}) == 72
    assert sum(row["split"] == "held_out" for row in rows) == 64 * 3


def test_scenario_verify_7236_long_loop_emits_truthful_heartbeat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-VERIFY-7236 reports completed bases only after monotonic time advances."""

    public, authority = mod.build_fixture()
    times = iter([0.0, 61.0, *([61.0] * 71)])
    events: list[tuple[int, str, str]] = []
    monkeypatch.setattr(mod.time, "monotonic", lambda: next(times))
    monkeypatch.setattr(
        mod, "_progress", lambda phase, event, detail: events.append((phase, event, detail))
    )
    rows = mod.execute_fixture(public, authority)
    assert len(rows) == 216
    assert events == [(4, "heartbeat", "completed=1/72 elapsed_s=61.0")]


def test_scenario_verify_7236_mutations_fail_as_declared() -> None:
    """SCENARIO-VERIFY-7236-ADAPTER keeps structural and semantic negatives."""

    rows = mod.mutation_rows()
    assert {row["mutation"] for row in rows} == {
        "wrong_direction",
        "reversed_polarity",
        "dangling_pointer",
        "ambiguous_pointer",
        "out_of_range_pointer",
        "missing_support",
        "mention_permutation",
        "exact_offset_reconstruction",
    }
    assert all(row["passed"] for row in rows)
    assert {row["actual_decision"] for row in rows[:6]} >= {"contradicted", "unknown"}


def test_req_verify_7236_unwrap_and_quarantine_are_typed() -> None:
    """REQ-VERIFY-7236 unwraps only annotations and rejects quarantine before gates."""

    wrapped = {"principle": "why", "value": 1}
    ordinary = {"value": 1, "domain": "data"}
    assert mod.unwrap_principle_value(wrapped) == 1
    assert mod.unwrap_principle_value(ordinary) is ordinary
    assert mod.upstream_acceptable({"mention_fixture_ready_score": wrapped}) is True
    assert (
        mod.upstream_acceptable(
            {"mention_fixture_ready_score": wrapped, "flagged_adversarial": True}
        )
        is False
    )


def test_scenario_verify_7236_artifact_builds_and_cold_replay_rejects_drift(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7236-ARTIFACT seals raw bytes and rejects changed evidence."""

    artifact = mod.build_artifact(ROOT, mod.RUN_DATE, output_root=tmp_path)
    assert artifact["status"] == "complete"
    assert artifact["mention_fixture_ready_score"] == 1
    assert artifact["verdict_class"] == "circular_positive"
    assert artifact["MODEL_SPECS"] == [] and artifact["model_invoked"] is False
    assert mod.validate_artifact(artifact, ROOT, output_root=tmp_path) == []
    public_path = tmp_path / mod.PUBLIC_MANIFEST_PATH
    public = json.loads(public_path.read_text(encoding="utf-8"))
    public["rows"][0]["variants"][0]["source"]["text"] = "changed"
    public_path.write_text(json.dumps(public), encoding="utf-8")
    assert "public_manifest" in mod.validate_artifact(artifact, ROOT, output_root=tmp_path)

    changed = deepcopy(artifact)
    changed["rows"][0]["metric"] = 0
    assert "reproducibility_checksum" in mod.validate_artifact(changed, ROOT, output_root=tmp_path)


def test_scenario_verify_7236_artifact_writes_exact_external_block(tmp_path: Path) -> None:
    """SCENARIO-VERIFY-7236-ARTIFACT preserves the first missing source gate."""

    artifact = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_root=tmp_path,
        path_overrides={"exp7223_artifact": tmp_path / "missing.json"},
    )
    assert artifact["status"] == "blocked"
    assert artifact["inference_substrate"] == "blocked_no_run"
    assert artifact["gate_check_summary"]["passed"] is False
    assert artifact["mention_fixture_ready_score"] == 0
    assert mod.validate_artifact(artifact, ROOT, output_root=tmp_path) == []


def test_req_verify_7236_date_entrypoint_and_defensive_validation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """REQ-VERIFY-7236 keeps the CLI date fixed and the entrypoint thin."""

    assert mod._date_argument(mod.RUN_DATE) == mod.RUN_DATE
    with pytest.raises(argparse.ArgumentTypeError):
        mod._date_argument("20260911")
    assert mod.validate_artifact(None, ROOT) == ["artifact_mapping"]
    assert mod.validate_artifact({}, ROOT)[0].startswith("missing_required_field:")

    built = mod.build_artifact(ROOT, mod.RUN_DATE, output_root=tmp_path)
    monkeypatch.setattr(mod, "find_repo_root", lambda **_kwargs: ROOT)
    monkeypatch.setattr(mod, "build_artifact", lambda *_args, **_kwargs: built)
    monkeypatch.setattr(mod, "validate_artifact", lambda *_args, **_kwargs: [])
    assert mod.main(["--date", mod.RUN_DATE]) == 0
    monkeypatch.setattr(mod, "validate_artifact", lambda *_args, **_kwargs: ["bad"])
    assert mod.main(["--date", mod.RUN_DATE]) == 1

    wrapper = (ROOT / mod.WRAPPER_PATH).read_text(encoding="utf-8")
    assert "experiment_7236_v637_mention_fixture import main" in wrapper


def test_scenario_verify_7236_cold_validation_names_each_terminal_boundary(
    tmp_path: Path,
) -> None:
    """SCENARIO-VERIFY-7236-ARTIFACT names malformed terminal fields and raw bytes."""

    artifact = mod.build_artifact(ROOT, mod.RUN_DATE, output_root=tmp_path)
    mutations = {
        "field_principles": {},
        "run_date": "bad",
        "MODEL_SPECS": [{}],
        "execution_venue": "bad",
        "verifier_is_oracle": False,
        "duration_s": -1,
        "random_seed": 0,
        "inference_substrate": "bad",
        "inference_substrate_class": "bad",
        "verdict_class": "positive",
        "sample_size_budget": {},
        "mutation_rows": [],
        "arm_contract": {},
    }
    expected = {
        "MODEL_SPECS": "model_contract",
        "execution_venue": "execution_identity",
        "verdict_class": "readiness_terminal_state",
    }
    for field, value in mutations.items():
        changed = deepcopy(artifact)
        changed[field] = value
        assert expected.get(field, field) in mod.validate_artifact(
            changed, ROOT, output_root=tmp_path
        )

    partial = deepcopy(artifact)
    partial["status"] = "running"
    assert "status" in mod.validate_artifact(partial, ROOT, output_root=tmp_path)
    blocked = deepcopy(artifact)
    blocked.update(
        {"status": "blocked", "gate_check_summary": {}, "mention_fixture_ready_score": 1}
    )
    blocked_errors = mod.validate_artifact(blocked, ROOT, output_root=tmp_path)
    assert {"blocked_terminal_state", "gate_check_summary"} <= set(blocked_errors)

    authority_path = tmp_path / mod.AUTHORITY_MANIFEST_PATH
    authority = json.loads(authority_path.read_text(encoding="utf-8"))
    authority["rows"][0]["condition_key"] = "changed"
    authority_path.write_text(json.dumps(authority), encoding="utf-8")
    errors = mod.validate_artifact(artifact, ROOT, output_root=tmp_path)
    assert {"authority_manifest", "source_artifact_hashes"} <= set(errors)
    authority_path.unlink()
    assert "sealed_fixture_files" in mod.validate_artifact(artifact, ROOT, output_root=tmp_path)


def test_scenario_verify_7236_parse_and_internal_validation_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SCENARIO-VERIFY-7236-ARTIFACT refuses malformed sources and failed cold replay."""

    malformed = tmp_path / "malformed.json"
    malformed.write_text("not-json", encoding="utf-8")
    blocked = mod.build_artifact(
        ROOT,
        mod.RUN_DATE,
        output_root=tmp_path / "parse",
        path_overrides={"exp7223_artifact": malformed},
    )
    assert blocked["gate_check_summary"]["failed_check"] == "source_parse"

    original_validate = mod.validate_artifact
    monkeypatch.setattr(mod, "validate_artifact", lambda *_args, **_kwargs: ["forced"])
    with pytest.raises(ValueError, match="invalid blocked"):
        mod.build_artifact(
            ROOT,
            mod.RUN_DATE,
            output_root=tmp_path / "blocked-invalid",
            path_overrides={"exp7223_artifact": tmp_path / "missing.json"},
        )
    with pytest.raises(ValueError, match="invalid Exp7236"):
        mod.build_artifact(ROOT, mod.RUN_DATE, output_root=tmp_path / "complete-invalid")
    monkeypatch.setattr(mod, "validate_artifact", original_validate)
