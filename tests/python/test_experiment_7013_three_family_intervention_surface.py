"""Tests for the three-family exact intervention response surface.

Spec refs: REQ-ENERGY-7013 and SCENARIO-ENERGY-7013-*.
"""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

from carnot import experiment_7013_three_family_intervention_surface as exp


class FakeLlama:
    """Provide deterministic tokens and logits without loading a real GGUF."""

    def __init__(self, *, null_logits: bool = False) -> None:
        self.null_logits = null_logits
        self.scores: object = []

    def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
        del special
        text = value.decode("utf-8")
        tokens = [3 + (sum(word.encode("utf-8")) % 11) for word in text.split()]
        return ([1] if add_bos else []) + tokens

    def eval(self, tokens: list[int]) -> None:
        if self.null_logits:
            self.scores = [None for _token in tokens]
            return
        rows = np.zeros((len(tokens), 32), dtype=np.float64)
        for index, token in enumerate(tokens):
            rows[index, token] = 1.0 + index / 100.0
        self.scores = rows

    def reset(self) -> None:
        return None

    def detokenize(self, tokens: list[int]) -> bytes:
        return (" ".join(str(token) for token in tokens)).encode("utf-8")


def _digest(tag: str) -> str:
    return exp.sha256_text(tag)


def _model_specs(tmp_path: Path) -> list[dict[str, object]]:
    rows = []
    for index, repository in enumerate(exp.REQUIRED_MODEL_IDS):
        path = tmp_path / f"model-{index}-Q4_K_M.gguf"
        path.write_bytes(repository.encode("utf-8"))
        rows.append(
            {
                "model_repository": repository,
                "model_path": str(path),
                "filename": path.name,
                "quantization": "Q4_K_M",
                "headline_eligible": True,
                "cpu_smoke_only": False,
            }
        )
    return rows


def _server_row(repository: str, *, ordinal: int = 0) -> dict[str, object]:
    return {
        "model_repository": repository,
        "pid": 2000 + ordinal,
        "owned_by_task": True,
        "process_alive_at_health": True,
        "health_pid": 2000 + ordinal,
        "health_model_repository": repository,
        "task_id": exp.EXPERIMENT_ID,
        "health_task_id": exp.EXPERIMENT_ID,
        "backend": "cuda",
        "live_cuda": True,
        "offloaded_layers": 40,
        "gpu_uuid": f"GPU-{ordinal}",
        "cuda_device": ordinal,
        "n_ctx": exp.N_CTX,
        "port": 18000 + ordinal,
        "started_monotonic_ns": 100,
        "health_monotonic_ns": 200,
        "command_hash": _digest(f"command-{ordinal}"),
        "terminal": True,
    }


def _raw_and_sidecar(
    *,
    pair_count: int = 2,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    raw_rows: list[dict[str, object]] = []
    token_rows: list[dict[str, object]] = []
    sidecar_rows: list[dict[str, object]] = []
    for pair_index in range(pair_count):
        direction = "violation" if pair_index % 2 == 0 else "repair"
        split = "held_source" if pair_index < 8 else "train"
        for surface, role, ordinal in (
            ("primary", "clean", 0),
            ("primary", "changed", 1),
            ("isomorphic", "clean", 2),
            ("isomorphic", "changed", 3),
        ):
            semantic_key = _digest(f"semantic-{pair_index}-{ordinal}")
            if role == "clean":
                normalized = -1.0 if surface == "primary" else -1.05
            elif direction == "violation":
                normalized = -2.0 if surface == "primary" else -2.05
            else:
                normalized = -0.5 if surface == "primary" else -0.55
            token_logprobs = [normalized, normalized]
            row = {
                "semantic_key": semantic_key,
                "model_repository": "",
                "prompt_hash": _digest(f"prompt-{pair_index}-{ordinal}"),
                "prompt_char_count": 100,
                "prompt_word_count": 20,
                "scoring_preamble_hash": _digest("preamble"),
                "fixed_response_hash": _digest(exp.FIXED_RESPONSE_TEXT),
                "response_token_ids": [7, 8],
                "response_positions": [0, 1],
                "response_token_count": 2,
                "sequence_log_likelihood": sum(token_logprobs),
                "normalized_sequence_log_likelihood": normalized,
                "request_id": _digest(f"request-{pair_index}-{ordinal}"),
                "terminal": True,
                "status": "success",
                "error": None,
                "row_hash": "",
            }
            for model_index, repository in enumerate(exp.REQUIRED_MODEL_IDS):
                model_row = deepcopy(row)
                model_row["model_repository"] = repository
                model_row["request_id"] = _digest(f"request-{model_index}-{pair_index}-{ordinal}")
                model_row["row_hash"] = exp.response_row_hash(model_row)
                raw_rows.append(model_row)
                for position, (token_id, logprob) in enumerate(
                    zip([7, 8], token_logprobs, strict=True)
                ):
                    token_row = {
                        "semantic_key": semantic_key,
                        "model_repository": repository,
                        "relative_position": position,
                        "token_id": token_id,
                        "token_log_probability": logprob,
                        "logit_vector_hash": _digest(
                            f"logits-{model_index}-{pair_index}-{ordinal}-{position}"
                        ),
                        "terminal": True,
                    }
                    token_row["row_hash"] = exp.token_position_row_hash(token_row)
                    token_rows.append(token_row)
            clean_label = "equivalent"
            changed_label = "non_equivalent"
            sidecar_rows.append(
                {
                    "semantic_key": semantic_key,
                    "block_id": _digest(f"block-{pair_index}"),
                    "pair_role": role,
                    "surface_variant": surface,
                    "intervention_direction": direction,
                    "exact_label": clean_label if role == "clean" else changed_label,
                    "source_group_id": _digest(f"source-group-{pair_index}"),
                    "source_family": f"family-{pair_index % 4}",
                    "split": split,
                    "mutation_kind": f"mutation-{pair_index % 4}",
                }
            )
    return raw_rows, token_rows, sidecar_rows


def _runtime_rows() -> dict[str, list[dict[str, object]]]:
    servers = [
        _server_row(model, ordinal=index) for index, model in enumerate(exp.REQUIRED_MODEL_IDS)
    ]
    return {
        "model_rows": [
            {
                "model_repository": model,
                "filename": f"model-{index}-Q4_K_M.gguf",
                "quantization": "Q4_K_M",
                "terminal": True,
            }
            for index, model in enumerate(exp.REQUIRED_MODEL_IDS)
        ],
        "gpu_identity_rows": [
            {
                "model_repository": model,
                "gpu_uuid": f"GPU-{index}",
                "cuda_device": index,
                "live_cuda": True,
                "foreign_processes_before": [],
                "foreign_processes_resident": [],
                "foreign_processes_after": [],
                "terminal": True,
            }
            for index, model in enumerate(exp.REQUIRED_MODEL_IDS)
        ],
        "gpu_lease_rows": [
            {
                "model_repository": model,
                "owner_verified": True,
                "released": True,
                "terminal_phase": "terminal_complete",
                "terminal": True,
            }
            for model in exp.REQUIRED_MODEL_IDS
        ],
        "server_rows": servers,
        "request_counter_rows": [
            {
                "model_repository": model,
                "expected_requests": exp.EXPECTED_PROMPT_COUNT,
                "observed_requests": exp.EXPECTED_PROMPT_COUNT,
                "terminal": True,
            }
            for model in exp.REQUIRED_MODEL_IDS
        ],
        "completion_counter_rows": [
            {
                "model_repository": model,
                "expected_completions": exp.EXPECTED_PROMPT_COUNT,
                "observed_completions": exp.EXPECTED_PROMPT_COUNT,
                "error_count": 0,
                "terminal": True,
            }
            for model in exp.REQUIRED_MODEL_IDS
        ],
        "teardown_rows": [
            {
                "model_repository": model,
                "process_exit_confirmed": True,
                "port_release_confirmed": True,
                "model_close_called": True,
                "lease_released": True,
                "unrelated_process_kill_count_delta": 0,
                "passed": True,
                "terminal": True,
            }
            for model in exp.REQUIRED_MODEL_IDS
        ],
    }


def _complete_artifact(tmp_path: Path) -> dict[str, object]:
    raw, token_rows, sidecar = _raw_and_sidecar(pair_count=exp.EXPECTED_PAIR_COUNT)
    freeze = exp.build_response_freeze(
        raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    condition_rows, signed_rows = exp.join_frozen_conditions(
        raw,
        sidecar,
        freeze_manifest=freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    family, held, ties = exp.summarize_signed_responses(signed_rows)
    runtime = _runtime_rows()
    specs = _model_specs(tmp_path)
    return exp.build_artifact(
        duration_s=61.0,
        preconditions={"all_passed": True, "checks": []},
        model_specs=specs,
        model_file_hashes={model: _digest(model) for model in exp.REQUIRED_MODEL_IDS},
        llama_binary_hash=_digest("llama-binary"),
        command_hash=_digest("all-commands"),
        source_artifact_hashes={"exp7012": exp.EXPECTED_EXP7012_HASH},
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        response_freeze_hash=str(freeze["response_freeze_hash"]),
        response_frozen_at="2026-09-05T12:00:00Z",
        label_opened_at="2026-09-05T12:00:01Z",
        rows=raw,
        token_position_rows=token_rows,
        condition_rows=condition_rows,
        signed_response_rows=signed_rows,
        family_effect_rows=family,
        held_source_effect_rows=held,
        tie_rows=ties,
        prohibited_feature_rows=exp.runtime_prohibited_feature_rows(),
        per_pair_results=exp.per_pair_results(condition_rows, signed_rows),
        **runtime,
    )


def test_req_energy_7013_spec_precedes_implementation() -> None:
    """REQ-ENERGY-7013: the spec and scenario anchors exist before code."""

    root = Path(__file__).resolve().parents[2]
    spec = (root / "openspec/capabilities/energy-verification/spec.md").read_text()
    source = (
        root / "python/carnot/experiment_7013_three_family_intervention_surface.py"
    ).read_text()
    assert "REQ-ENERGY-7013" in spec
    assert "SCENARIO-ENERGY-7013-FREEZE" in spec
    assert "REQ-ENERGY-7013" in source


def test_scenario_energy_7013_preflight_rejects_missing_family_legacy_and_headline_cpu(
    tmp_path: Path,
) -> None:
    """SCENARIO-ENERGY-7013-PREFLIGHT: model substitutions fail closed."""

    specs = _model_specs(tmp_path)
    assert exp.model_spec_errors(specs) == []
    assert "model_family_roster" in exp.model_spec_errors(specs[:-1])
    legacy = deepcopy(specs)
    legacy[0]["model_repository"] = "Qwen/Qwen3.5-0.8B"
    assert "model_family_roster" in exp.model_spec_errors(legacy)
    cpu = deepcopy(specs)
    cpu[0]["cpu_smoke_only"] = True
    assert "headline_cpu_smoke" in exp.model_spec_errors(cpu)
    smoke = {
        "model_repository": "Qwen/Qwen3.5-0.8B",
        "cpu_smoke_only": True,
        "headline_eligible": False,
        "result_row_count": 1,
    }
    assert exp.cpu_smoke_errors(smoke) == ["cpu_smoke_populated_result_rows"]


def test_scenario_energy_7013_runtime_rejects_cpu_stale_and_non_owned_server() -> None:
    """SCENARIO-ENERGY-7013-RUNTIME: stale or foreign CPU servers fail."""

    valid = _server_row(exp.REQUIRED_MODEL_IDS[0])
    assert exp.server_receipt_errors(valid) == []
    cpu = valid | {"backend": "cpu", "live_cuda": False, "offloaded_layers": 0}
    assert "cpu_fallback" in exp.server_receipt_errors(cpu)
    stale = valid | {"health_pid": 9999}
    assert "stale_server" in exp.server_receipt_errors(stale)
    stale_time = valid | {"health_monotonic_ns": 99}
    assert "stale_server" in exp.server_receipt_errors(stale_time)
    foreign = valid | {"owned_by_task": False}
    assert "non_owned_process" in exp.server_receipt_errors(foreign)


def test_scenario_energy_7013_scoring_records_finite_positions_and_rejects_null_logits() -> None:
    """SCENARIO-ENERGY-7013-SCORING: null logits never become a score."""

    row, positions = exp.score_teacher_forced_response(
        FakeLlama(),
        prompt_text="Assess this mapping exactly.",
        semantic_key=_digest("semantic"),
        model_repository=exp.REQUIRED_MODEL_IDS[0],
        request_id=_digest("request"),
    )
    assert row["terminal"] is True
    assert row["status"] == "success"
    assert row["response_token_count"] == len(positions) > 0
    assert row["response_positions"] == list(range(len(positions)))
    assert np.isfinite(row["sequence_log_likelihood"])
    assert all(np.isfinite(item["token_log_probability"]) for item in positions)
    with pytest.raises(exp.ScoringError, match="null_or_malformed_logits"):
        exp.score_teacher_forced_response(
            FakeLlama(null_logits=True),
            prompt_text="Assess this mapping exactly.",
            semantic_key=_digest("semantic"),
            model_repository=exp.REQUIRED_MODEL_IDS[0],
            request_id=_digest("request"),
        )


def test_scenario_energy_7013_freeze_blocks_early_labels_and_changed_rows() -> None:
    """SCENARIO-ENERGY-7013-FREEZE: labels follow an immutable response hash."""

    raw, token_rows, sidecar = _raw_and_sidecar()
    freeze = exp.build_response_freeze(
        raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    exp.assert_label_access_allowed(freeze, "2026-09-05T12:00:01Z")
    with pytest.raises(exp.LabelAccessError, match="response_freeze_missing"):
        exp.assert_label_access_allowed({}, "2026-09-05T12:00:01Z")
    with pytest.raises(exp.LabelAccessError, match="label_open_not_after_freeze"):
        exp.assert_label_access_allowed(freeze, "2026-09-05T11:59:59Z")
    changed = deepcopy(raw)
    changed[0]["sequence_log_likelihood"] = 99.0
    with pytest.raises(exp.LabelAccessError, match="response_freeze_hash_mismatch"):
        exp.join_frozen_conditions(
            changed,
            sidecar,
            freeze_manifest=freeze,
            label_opened_at="2026-09-05T12:00:01Z",
        )


def test_scenario_energy_7013_cells_rejects_duplicates_prompt_mismatch_and_position_drift() -> None:
    """SCENARIO-ENERGY-7013-CELLS: exact cell and alignment checks fail closed."""

    raw, token_rows, sidecar = _raw_and_sidecar()
    assert "duplicate_response_row" in exp.response_row_errors(raw + [deepcopy(raw[0])])
    freeze = exp.build_response_freeze(
        raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    conditions, _signed = exp.join_frozen_conditions(
        raw,
        sidecar,
        freeze_manifest=freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    assert exp.alignment_errors(conditions) == []
    prompt_mismatch = deepcopy(conditions)
    prompt_mismatch[0]["prompt_char_count"] = 101
    assert "prompt_mismatch" in exp.alignment_errors(prompt_mismatch)
    position_drift = deepcopy(conditions)
    position_drift[0]["response_positions"] = [0, 2]
    assert "token_position_drift" in exp.alignment_errors(position_drift)


def test_scenario_energy_7013_effects_preserve_family_reversal_held_groups_and_ties() -> None:
    """SCENARIO-ENERGY-7013-EFFECTS: families and held groups stay separate."""

    raw, token_rows, sidecar = _raw_and_sidecar(pair_count=8)
    freeze = exp.build_response_freeze(
        raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    conditions, signed = exp.join_frozen_conditions(
        raw,
        sidecar,
        freeze_manifest=freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    by_pair = {}
    for row in signed:
        by_pair.setdefault(row["pair_id"], row)
    direction_to_delta = {
        next(
            item["intervention_direction"] for item in conditions if item["pair_id"] == pair_id
        ): row["signed_primary_delta"]
        for pair_id, row in by_pair.items()
    }
    assert direction_to_delta == {"violation": 1.0, "repair": -0.5}
    reversed_rows = deepcopy(signed)
    reversed_rows[0]["signed_primary_delta"] = -1.0
    reversed_rows[1]["signed_primary_delta"] = 0.0
    family, held, ties = exp.summarize_signed_responses(reversed_rows, draws=100)
    assert {row["model_repository"] for row in family} == set(exp.REQUIRED_MODEL_IDS)
    assert all(row["pooled_families"] is False for row in family)
    assert held and all(row["split"] == "held_source" for row in held)
    assert any(row["tie_kind"] == "primary_signed_delta" for row in ties)
    assert len(exp.per_pair_results(conditions, reversed_rows)) == 8


def test_scenario_energy_7013_artifact_blocks_teardown_counts_and_short_duration(
    tmp_path: Path,
) -> None:
    """SCENARIO-ENERGY-7013-ARTIFACT: runtime defects prevent completion."""

    artifact = _complete_artifact(tmp_path)
    assert artifact["intervention_surface_complete_score"] == 1
    assert exp.validate_artifact(artifact) == []
    teardown = deepcopy(artifact)
    teardown["teardown_rows"][0]["passed"] = False
    teardown["reproducibility_checksum"] = exp.artifact_checksum(teardown)
    assert "completion_score_mismatch" in exp.validate_artifact(teardown)
    counters = deepcopy(artifact)
    counters["completion_counter_rows"][0]["observed_completions"] -= 1
    counters["reproducibility_checksum"] = exp.artifact_checksum(counters)
    assert "completion_score_mismatch" in exp.validate_artifact(counters)
    short = deepcopy(artifact)
    short["duration_s"] = 59.9
    short["reproducibility_checksum"] = exp.artifact_checksum(short)
    assert "implausible_live_duration" in exp.validate_artifact(short)


def test_req_energy_7013_blocked_artifact_has_all_required_principles(tmp_path: Path) -> None:
    """REQ-ENERGY-7013: blocked preflight still writes the complete schema."""

    check = exp.gate_check("cached_models", 3, 2)
    artifact = exp.build_artifact(
        duration_s=0.2,
        preconditions={"all_passed": False, "checks": [check]},
        model_specs=_model_specs(tmp_path),
        model_file_hashes={},
        llama_binary_hash=None,
        command_hash=None,
        source_artifact_hashes={},
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
    )
    assert set(exp.REQUIRED_ARTIFACT_FIELDS) <= set(artifact)
    assert set(artifact["field_principles"]) == set(exp.REQUIRED_ARTIFACT_FIELDS)
    assert type(artifact["intervention_surface_complete_score"]) is int
    assert artifact["intervention_surface_complete_score"] == 0
    assert artifact["gate_check_summary"]["failed_check"] == "cached_models"
    assert artifact["verdict_class"] == "blocked"
    assert artifact["honest_verdict"] == "blocked_intervention_surface"
    assert exp.validate_artifact(artifact) == []


def test_req_energy_7013_validator_rejects_fields_hashes_features_and_verdict(
    tmp_path: Path,
) -> None:
    """REQ-ENERGY-7013: cold validation recomputes the complete evidence contract."""

    artifact = _complete_artifact(tmp_path)
    mutations = []
    missing = deepcopy(artifact)
    missing.pop("model_rows")
    mutations.append((missing, "required_fields_missing"))
    principles = deepcopy(artifact)
    principles["field_principles"].pop("rows")
    mutations.append((principles, "field_principles_mismatch"))
    checksum = deepcopy(artifact)
    checksum["reproducibility_checksum"] = "sha256:bad"
    mutations.append((checksum, "reproducibility_checksum_mismatch"))
    feature = deepcopy(artifact)
    feature["rows"][0]["exact_label"] = "equivalent"
    feature["rows"][0]["row_hash"] = exp.response_row_hash(feature["rows"][0])
    feature["reproducibility_checksum"] = exp.artifact_checksum(feature)
    mutations.append((feature, "prohibited_response_feature"))
    verdict = deepcopy(artifact)
    verdict["verdict_class"] = "blocked"
    verdict["honest_verdict"] = "complete_null_wrong"
    verdict["reproducibility_checksum"] = exp.artifact_checksum(verdict)
    mutations.append((verdict, "verdict_prefix_mismatch"))
    for mutated, expected in mutations:
        assert any(expected in error for error in exp.validate_artifact(mutated))


def test_req_energy_7013_wrapper_targets_module() -> None:
    """REQ-ENERGY-7013: the requested command invokes the package module."""

    root = Path(__file__).resolve().parents[2]
    wrapper = (
        root / "scripts/experiments/experiment_7013_three_family_intervention_surface.py"
    ).read_text()
    assert "carnot.experiment_7013_three_family_intervention_surface import main" in wrapper
    assert "research_conductor" not in wrapper


def test_req_energy_7013_defensive_scoring_and_row_validation_branches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ENERGY-7013: malformed scores and terminal rows fail closed."""

    specs = _model_specs(tmp_path)
    bad_path = deepcopy(specs)
    bad_path[0]["model_path"] = ""
    bad_path[0]["filename"] = "mmproj-Q4_K_M.gguf"
    assert "primary_gguf_path" in exp.model_spec_errors(bad_path)
    non_headline = deepcopy(specs)
    non_headline[0]["headline_eligible"] = False
    assert "headline_eligibility" in exp.model_spec_errors(non_headline)
    assert exp.cpu_smoke_errors({"result_row_count": 0}) == ["cpu_smoke_declaration"]

    with pytest.raises(exp.ScoringError, match="response_token_out_of_range"):
        exp._log_probability(np.zeros(2), 3)

    class BoundarySensitiveLlama(FakeLlama):
        def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
            tokens = super().tokenize(value, add_bos=add_bos, special=special)
            if add_bos and value.decode("utf-8").endswith(exp.FIXED_RESPONSE_TEXT):
                tokens.append(31)
            return tokens

    class EmptyResponseLlama(FakeLlama):
        def tokenize(self, value: bytes, *, add_bos: bool, special: bool) -> list[int]:
            if not add_bos:
                return []
            return super().tokenize(value, add_bos=add_bos, special=special)

    class MissingScoresLlama(FakeLlama):
        def eval(self, tokens: list[int]) -> None:
            del tokens
            self.scores = None

    class ShortScoresLlama(FakeLlama):
        def eval(self, tokens: list[int]) -> None:
            del tokens
            self.scores = []

    scoring_kwargs = {
        "prompt_text": "Assess this mapping exactly.",
        "semantic_key": _digest("guard-semantic"),
        "model_repository": exp.REQUIRED_MODEL_IDS[0],
        "request_id": _digest("guard-request"),
    }
    boundary_row, boundary_positions = exp.score_teacher_forced_response(
        BoundarySensitiveLlama(), **scoring_kwargs
    )
    assert boundary_row["status"] == "success"
    assert boundary_positions
    with pytest.raises(exp.ScoringError, match="teacher_forced_token_alignment"):
        exp.score_teacher_forced_response(EmptyResponseLlama(), **scoring_kwargs)
    monkeypatch.setattr(exp, "N_CTX", 1)
    with pytest.raises(exp.ScoringError, match="context_overflow"):
        exp.score_teacher_forced_response(FakeLlama(), **scoring_kwargs)
    monkeypatch.setattr(exp, "N_CTX", 4096)
    with pytest.raises(exp.ScoringError, match="null_or_malformed_logits"):
        exp.score_teacher_forced_response(MissingScoresLlama(), **scoring_kwargs)
    with pytest.raises(exp.ScoringError, match="null_or_malformed_logits"):
        exp.score_teacher_forced_response(ShortScoresLlama(), **scoring_kwargs)

    raw, token_rows, _sidecar = _raw_and_sidecar(pair_count=1)
    failed = exp.failed_response_row(
        semantic_key=_digest("failed-semantic"),
        model_repository=exp.REQUIRED_MODEL_IDS[0],
        prompt_text="failed prompt",
        request_id=_digest("failed-request"),
        error_text="scoring_failed",
    )
    assert exp.response_row_errors([failed]) == []
    assert "response_row_cross_product" in exp.response_row_errors(
        raw, expected_semantic_keys=[str(raw[0]["semantic_key"])]
    )
    row_mutations = (
        ({"row_hash": "bad"}, "response_row_hash"),
        ({"terminal": False}, "response_row_terminal"),
        ({"response_positions": [1, 0]}, "response_token_alignment"),
        ({"sequence_log_likelihood": float("nan")}, "response_score_non_finite"),
        ({"status": "failed", "error": ""}, "failed_response_without_error"),
    )
    for changes, expected in row_mutations:
        row = deepcopy(raw[0])
        row.update(changes)
        if "row_hash" not in changes:
            row["row_hash"] = exp.response_row_hash(row)
        assert expected in exp.response_row_errors([row])

    assert "duplicate_token_position_row" in exp.token_position_errors(
        token_rows + [deepcopy(token_rows[0])]
    )
    bad_position_hash = deepcopy(token_rows[0])
    bad_position_hash["row_hash"] = "bad"
    assert "token_position_row_hash" in exp.token_position_errors([bad_position_hash])
    invalid_position = deepcopy(token_rows[0])
    invalid_position["terminal"] = False
    invalid_position["row_hash"] = exp.token_position_row_hash(invalid_position)
    assert "token_position_invalid" in exp.token_position_errors([invalid_position])
    with pytest.raises(exp.FreezeError, match="response_rows_invalid"):
        exp.build_response_freeze(
            raw + [deepcopy(raw[0])],
            token_rows,
            prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
            frozen_at="2026-09-05T12:00:00Z",
        )


def test_scenario_energy_7013_freeze_and_sidecar_defensive_branches() -> None:
    """SCENARIO-ENERGY-7013-FREEZE: malformed clocks and sidecars stay sealed."""

    raw, token_rows, sidecar = _raw_and_sidecar(pair_count=1)
    freeze = exp.build_response_freeze(
        raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    for bad_time in (None, "not-a-time", "2026-09-05T12:00:00"):
        with pytest.raises(exp.LabelAccessError, match="response_freeze_missing"):
            exp.assert_label_access_allowed(freeze, bad_time)  # type: ignore[arg-type]
    with pytest.raises(exp.LabelAccessError, match="condition_role_invalid"):
        exp._condition_role({"surface_variant": "wrong", "pair_role": "clean"})
    assert exp._labels_match_direction(sidecar[:-1]) is False
    bad_direction = deepcopy(sidecar)
    for row in bad_direction:
        row["intervention_direction"] = "unknown"
    assert exp._labels_match_direction(bad_direction) is False

    duplicate = sidecar + [deepcopy(sidecar[0])]
    with pytest.raises(exp.LabelAccessError, match="duplicate_sidecar_key"):
        exp.join_frozen_conditions(
            raw,
            duplicate,
            freeze_manifest=freeze,
            label_opened_at="2026-09-05T12:00:01Z",
        )
    mismatched = deepcopy(sidecar)
    mismatched[0]["semantic_key"] = _digest("unknown-sidecar-key")
    with pytest.raises(exp.LabelAccessError, match="sidecar_key_mismatch"):
        exp.join_frozen_conditions(
            raw,
            mismatched,
            freeze_manifest=freeze,
            label_opened_at="2026-09-05T12:00:01Z",
        )
    invalid_block = deepcopy(sidecar)
    invalid_block[0]["exact_label"] = "wrong"
    with pytest.raises(exp.LabelAccessError, match="sidecar_block_invalid"):
        exp.join_frozen_conditions(
            raw,
            invalid_block,
            freeze_manifest=freeze,
            label_opened_at="2026-09-05T12:00:01Z",
        )

    missing_cell_raw = [
        row
        for index, row in enumerate(raw)
        if not (index == 0 and row["model_repository"] == exp.REQUIRED_MODEL_IDS[0])
    ]
    missing_freeze = exp.build_response_freeze(
        missing_cell_raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    _conditions, signed = exp.join_frozen_conditions(
        missing_cell_raw,
        sidecar,
        freeze_manifest=missing_freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    assert len(signed) == 2

    failed_raw = deepcopy(raw)
    replaced = failed_raw[0]
    failed_raw[0] = exp.failed_response_row(
        semantic_key=str(replaced["semantic_key"]),
        model_repository=str(replaced["model_repository"]),
        prompt_text="failed prompt",
        request_id=str(replaced["request_id"]),
        error_text="scoring_failed",
    )
    failed_freeze = exp.build_response_freeze(
        failed_raw,
        token_rows,
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
        frozen_at="2026-09-05T12:00:00Z",
    )
    _conditions, signed = exp.join_frozen_conditions(
        failed_raw,
        sidecar,
        freeze_manifest=failed_freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    assert len(signed) == 2

    conditions, _signed = exp.join_frozen_conditions(
        raw,
        sidecar,
        freeze_manifest=freeze,
        label_opened_at="2026-09-05T12:00:01Z",
    )
    assert "condition_cell_shape" in exp.alignment_errors(conditions[:-1])
    assert exp.bootstrap_interval([], seed=exp.RANDOM_SEED)["pair_count"] == 0


def test_req_energy_7013_completion_and_cold_validator_recompute_every_gate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """REQ-ENERGY-7013: every required completion receipt is independently checked."""

    artifact = _complete_artifact(tmp_path)
    receipt = _server_row(exp.REQUIRED_MODEL_IDS[0]) | {"process_alive_at_health": False}
    assert "server_receipt_incomplete" in exp.server_receipt_errors(receipt)
    process = {"gpu_uuid": "GPU-0", "pid": 123}
    assert exp._device_processes({"processes": [process]}, "GPU-0") == [process]

    mutations = (
        ("preconditions", lambda row: row["preconditions_checked"].update(all_passed=False)),
        ("model_specs", lambda row: row["MODEL_SPECS"].pop()),
        ("token_position", lambda row: row["token_position_rows"][0].update(terminal=False)),
        ("condition_cells", lambda row: row["condition_rows"].pop()),
        ("signed", lambda row: row["signed_response_rows"].pop()),
        ("model_rows", lambda row: row["model_rows"].pop()),
        ("gpu", lambda row: row["gpu_identity_rows"][0].update(live_cuda=False)),
        ("lease", lambda row: row["gpu_lease_rows"][0].update(released=False)),
        ("server", lambda row: row["server_rows"][0].update(health_pid=-1)),
        ("request", lambda row: row["request_counter_rows"][0].update(observed_requests=0)),
        ("freeze", lambda row: row.update(response_freeze_hash=_digest("other-freeze"))),
        ("model_hashes", lambda row: row["model_file_hashes"].pop(exp.REQUIRED_MODEL_IDS[0])),
        ("runtime_hashes", lambda row: row.update(command_hash=None)),
        ("feature_audit", lambda row: row.update(prohibited_feature_rows=[])),
    )
    expected_errors = (
        "preconditions_incomplete",
        "model_specs_incomplete",
        "token_position_rows_incomplete",
        "condition_cells_incomplete",
        "signed_response_rows_incomplete",
        "model_rows_incomplete",
        "gpu_identity_incomplete",
        "gpu_lease_incomplete",
        "server_rows_incomplete",
        "request_counters_incomplete",
        "response_freeze_mismatch",
        "model_file_hashes_incomplete",
        "runtime_hashes_incomplete",
        "prohibited_feature_audit_incomplete",
    )
    for (_name, mutate), expected in zip(mutations, expected_errors, strict=True):
        changed = deepcopy(artifact)
        mutate(changed)
        assert expected in exp.completion_errors(changed)

    partial = exp.build_artifact(
        duration_s=61.0,
        preconditions={"all_passed": True, "checks": []},
        model_specs=_model_specs(tmp_path),
        model_file_hashes={},
        llama_binary_hash=None,
        command_hash=None,
        source_artifact_hashes={},
        prompt_freeze_hash=exp.EXPECTED_LEARNER_PROMPT_HASH,
    )
    assert partial["verdict_class"] == "partial"
    monkeypatch.setattr(exp, "_science_positive", lambda _rows: False)
    null_artifact = _complete_artifact(tmp_path)
    assert null_artifact["verdict_class"] == "null"

    validator_mutations = (
        ({"inference_substrate": "offline"}, "inference_substrate_mismatch"),
        ({"expected_family_count": 2}, "fixed_count_mismatch:expected_family_count"),
        ({"intervention_surface_complete_score": True}, "completion_score_not_bare_int"),
        ({"verifier_is_oracle": True}, "verifier_is_oracle_mismatch"),
    )
    for changes, expected in validator_mutations:
        changed = deepcopy(artifact)
        changed.update(changes)
        changed["reproducibility_checksum"] = exp.artifact_checksum(changed)
        assert expected in exp.validate_artifact(changed)
    invalid_token_artifact = deepcopy(artifact)
    invalid_token_artifact["token_position_rows"][0]["terminal"] = False
    invalid_token_artifact["token_position_rows"][0]["row_hash"] = exp.token_position_row_hash(
        invalid_token_artifact["token_position_rows"][0]
    )
    invalid_token_artifact["reproducibility_checksum"] = exp.artifact_checksum(
        invalid_token_artifact
    )
    assert "token_position_rows_invalid" in exp.validate_artifact(invalid_token_artifact)
