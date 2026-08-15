"""Tests for Exp6450 SOTA fixed-policy candidate corpus.

Spec refs: REQ-INFRA-6450, SCENARIO-INFRA-6450-1,
SCENARIO-INFRA-6450-2, SCENARIO-INFRA-6450-3,
SCENARIO-INFRA-6450-4, SCENARIO-INFRA-6450-5.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6450_sota_fixed_policy_candidate_corpus as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH


def _model_paths(tmp_path: Path) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for model_id in mod.MANDATED_MODEL_IDS:
        path = tmp_path / (mod.model_slug(model_id) + "-Q4_K_M.gguf")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((model_id + "\nExp6450 fixture weights\n").encode("utf-8"))
        paths[model_id] = path
    return paths


def _cached_pair(paths: dict[str, Path], calls: list[dict[str, Any]]):
    def resolve(
        *,
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        calls.append(
            {
                "gpu_indices": gpu_indices,
                "preferred_quant": preferred_quant,
                "model_indices": model_indices,
            }
        )
        ordered = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(paths[model_id]),
            }
            for gpu, model_id in zip(gpu_indices, ordered, strict=True)
        ]

    return resolve


def _tokenizer(path: str) -> tuple[bool, str]:
    return True, f"fixture embedded tokenizer for {Path(path).name}"


def _host_ok(
    *,
    result_path: Path,
    data_dir: Path,
    model_specs: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {"resource": "rtx_3090_gpu_count", "available": True, "detail": "2 fixture GPUs"},
        {"resource": "free_vram", "available": True, "detail": "fixture VRAM OK"},
        {"resource": "mandatory_model_files", "available": True, "detail": str(len(model_specs))},
        {"resource": "embedded_gguf_tokenizers", "available": True, "detail": "fixture OK"},
        {"resource": "exact_simulator_imports", "available": True, "detail": "fixture OK"},
        {"resource": "disk_space", "available": True, "detail": "fixture OK"},
        {"resource": "monotonic_clock", "available": True, "detail": "fixture OK"},
        {
            "resource": "new_raw_output_paths",
            "available": not (data_dir / "raw_outputs").exists() and not result_path.exists(),
            "detail": "fresh fixture paths",
        },
    ]


def _fixture_generation(
    *,
    model: dict[str, Any],
    problems: list[dict[str, Any]],
    candidate_seeds: tuple[int, ...],
    prompts: dict[int, str],
    decoding_settings: dict[str, Any],
    output_dir: Path,
) -> dict[str, Any]:
    seed_outputs = []
    modes = ("success", "protected_violation", "illegal")
    for seed_index, seed in enumerate(candidate_seeds):
        lines = []
        for problem in problems:
            mode = modes[(int(problem["row_index"]) + seed_index) % len(modes)]
            payload = {
                "model_hf_id": model["hf_id"],
                "problem_id": problem["problem_id"],
                "candidate_seed": seed,
                "actions": mod.fixture_action_plan(problem, mode),
            }
            lines.append(json.dumps(payload, ensure_ascii=True, sort_keys=True))
        raw = "\n".join(lines) + "\n"
        seed_outputs.append(
            {
                "candidate_seed": seed,
                "raw_batch_text": raw,
                "prompt_sha256": mod.sha256_text(prompts[seed]),
                "decoding_settings": dict(decoding_settings),
                "runtime_receipt": {
                    "pid": 9000 + seed_index,
                    "parent_pid": 8000,
                    "device_uuid": f"GPU-fixture-{model['gpu']}",
                    "gpu_index": model["gpu"],
                    "cuda_offload": True,
                    "cpu_fallback": False,
                    "completion_tokens": 64,
                    "first_token_observed": True,
                },
                "timing": {
                    "started_monotonic_ns": 1000 + seed_index,
                    "ended_monotonic_ns": 2000 + seed_index,
                    "duration_s": 1.0,
                },
            }
        )
    return {
        "model_hf_id": model["hf_id"],
        "seed_outputs": seed_outputs,
        "model_runtime_receipt": {
            "runner": "fixture_llama_cpp_cuda",
            "model_hf_id": model["hf_id"],
            "cuda_offload": True,
            "cpu_fallback": False,
            "output_dir": str(output_dir),
        },
    }


def _test_exit_codes() -> dict[str, int]:
    return {command: 0 for command in mod.DEFAULT_TEST_COMMANDS}


def _artifact(tmp_path: Path, *, write: bool = False) -> dict[str, Any]:
    paths = _model_paths(tmp_path / "models")
    return mod.run(
        date="20260815",
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        data_dir=tmp_path / "exp6450-data",
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=_fixture_generation,
        test_exit_codes=_test_exit_codes(),
        duration_s=125.0,
        write=write,
    )


def test_req_infra_6450_spec_declares_fields_and_scenarios() -> None:
    """REQ-INFRA-6450: OpenSpec owns the Exp6450 corpus contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-INFRA-6450") : text.index("REQ-INFRA-6351")]
    for marker in (
        "SCENARIO-INFRA-6450-1",
        "SCENARIO-INFRA-6450-2",
        "SCENARIO-INFRA-6450-3",
        "SCENARIO-INFRA-6450-4",
        "SCENARIO-INFRA-6450-5",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        "typed action plans",
    ):
        assert marker in section
    for field in mod.REQUIRED_ARTIFACT_FIELDS:
        assert f"`{field}`" in section
        assert field in mod.FIELD_PRINCIPLES
    for condition in mod.READINESS_CONDITIONS:
        assert f"sota_corpus_ready_score:{condition}" in mod.FIELD_PRINCIPLES


def test_scenario_infra_6450_model_specs_use_cached_sota_and_embedded_tokenizers(
    tmp_path: Path,
) -> None:
    """SCENARIO-INFRA-6450-1: model rows use cached SOTA and embedded tokenizers."""

    calls: list[dict[str, Any]] = []
    resolved = mod.build_model_specs(
        cached_pair_func=_cached_pair(_model_paths(tmp_path), calls),
        tokenizer_func=_tokenizer,
    )

    assert calls == [
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": None},
        {"gpu_indices": (0, 1), "preferred_quant": "Q4_K_M", "model_indices": (0, 2)},
    ]
    assert [row["hf_id"] for row in resolved["MODEL_SPECS"]] == list(mod.MANDATED_MODEL_IDS)
    assert resolved["all_resolved"] is True
    assert resolved["autotokenizer_usage_count"] == 0
    assert all(row["tokenizer_source"] == mod.TOKENIZER_SOURCE for row in resolved["MODEL_SPECS"])


def test_scenario_infra_6450_manifest_seals_three_partitions(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6450-1: problems and partitions are sealed before generation."""

    problems = mod.build_policy_problems()
    manifest = mod.sealed_problem_and_partition_manifest(tmp_path / "data", problems, write=False)

    assert manifest["problem_count"] == 36
    assert manifest["sealed_before_inference"] is True
    assert manifest["partition_counts"] == {
        "allocation_held": 12,
        "development": 12,
        "selection_held": 12,
    }
    assert manifest["held_label_visible_before_generation_count"] == 0
    assert all(problem["typed_action_schema"] == mod.ACTION_SCHEMA for problem in problems)
    assert len({problem["problem_hash"] for problem in problems}) == 36


def test_scenario_infra_6450_parser_and_simulator_are_exact() -> None:
    """SCENARIO-INFRA-6450-2 and SCENARIO-INFRA-6450-3: parsing is fixed and exact."""

    problem = mod.build_policy_problems()[0]
    good_line = json.dumps(
        {
            "problem_id": problem["problem_id"],
            "candidate_seed": mod.CANDIDATE_SEEDS[0],
            "actions": mod.fixture_action_plan(problem, "success"),
        },
        sort_keys=True,
    )
    parsed = mod.parse_candidate_line(good_line.encode("utf-8"), problem, mod.CANDIDATE_SEEDS[0])
    exact = mod.simulate_action_plan(problem, parsed)
    assert parsed["parse_valid"] is True
    assert exact["legal"] is True
    assert exact["protected_ok"] is True
    assert exact["goal_ok"] is True
    assert exact["exact_success"] is True

    protected_line = json.dumps(
        {
            "problem_id": problem["problem_id"],
            "candidate_seed": mod.CANDIDATE_SEEDS[1],
            "actions": mod.fixture_action_plan(problem, "protected_violation"),
        },
        sort_keys=True,
    )
    protected = mod.simulate_action_plan(
        problem,
        mod.parse_candidate_line(
            protected_line.encode("utf-8"), problem, mod.CANDIDATE_SEEDS[1]
        ),
    )
    assert protected["legal"] is True
    assert protected["protected_ok"] is False
    assert protected["exact_success"] is False

    bad = mod.parse_candidate_line(b"not json", problem, mod.CANDIDATE_SEEDS[2])
    assert bad["parse_valid"] is False
    assert mod.simulate_action_plan(problem, bad)["exact_success"] is False

    illegal_line = json.dumps(
        {
            "problem_id": problem["problem_id"],
            "candidate_seed": mod.CANDIDATE_SEEDS[2],
            "actions": mod.fixture_action_plan(problem, "illegal"),
        },
        sort_keys=True,
    )
    illegal = mod.simulate_action_plan(
        problem,
        mod.parse_candidate_line(illegal_line.encode("utf-8"), problem, mod.CANDIDATE_SEEDS[2]),
    )
    assert illegal["legal"] is False
    assert illegal["exact_success"] is False


def test_scenario_infra_6450_artifact_rows_recompute_headroom(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6450-4: rows own aggregates, outcomes, and headroom."""

    artifact = _artifact(tmp_path, write=True)
    rows = artifact["per_unit_rows"]["rows"]
    written = tmp_path / mod.RESULT_RELATIVE_PATH.name
    recomputed = mod.recompute_aggregates_from_rows(rows)

    assert written.is_file()
    assert json.loads(written.read_text(encoding="utf-8")) == artifact
    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "success"
    assert artifact["sota_corpus_ready_score"] == 1.0
    assert artifact["models_used"] == list(mod.MANDATED_MODEL_IDS)
    assert artifact["autotokenizer_usage_count"] == 0
    assert artifact["verifier_is_oracle"] is True
    assert artifact["blocked_reason"] == ""
    assert len(rows) == 36 * len(mod.MANDATED_MODEL_IDS) * len(mod.CANDIDATE_SEEDS)
    assert len({row["raw_hash"] for row in rows}) == len(rows)
    assert all(row["raw_output_stored_before_parse"] for row in rows)
    assert all(row["finite_id_generated_answer_experiment"] is False for row in rows)
    assert all(row["parser_retry_count"] == 0 for row in rows)
    assert all(row["path_stage_hashes"] for row in rows)
    assert artifact["eligible_rows_by_model_and_partition"] == recomputed[
        "eligible_rows_by_model_and_partition"
    ]
    assert artifact["parse_failures_by_model"] == recomputed["parse_failures_by_model"]
    assert artifact["exact_outcomes_by_model_and_partition"] == recomputed[
        "exact_outcomes_by_model_and_partition"
    ]
    assert artifact["candidate_headroom_by_partition"] == recomputed[
        "candidate_headroom_by_partition"
    ]
    assert artifact["aggregate_row_recomputation"]["matches_reported"] is True
    assert artifact["aggregate_row_recomputation"]["model_ranking_claim_made"] is False
    for row in artifact["candidate_headroom_by_partition"].values():
        assert row["has_headroom"] is True
        assert row["mixed_exact_outcomes"] is True
    assert artifact["raw_output_uniqueness_and_reuse_count"]["reuse_count"] == 0
    assert artifact["reproducibility_checksum"] == mod.payload_checksum(artifact)


def test_scenario_infra_6450_attacks_and_blockers_fail_closed(tmp_path: Path) -> None:
    """SCENARIO-INFRA-6450-5: attacks and precondition blockers close the gate."""

    artifact = _artifact(tmp_path)
    attacks = artifact["attack_matrix"]
    assert {row["attack_id"] for row in attacks["rows"]} == set(mod.ATTACK_IDS)
    assert attacks["all_critical_fail_closed"] is True
    assert attacks["false_accept_count"] == 0

    bad = deepcopy(artifact)
    bad["attack_matrix"]["rows"][0]["fail_closed"] = False
    mod.refresh_terminal_fields(bad)
    assert bad["sota_corpus_ready_score"] == 0.0

    bad = deepcopy(artifact)
    bad["raw_output_uniqueness_and_reuse_count"]["reuse_count"] = 1
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "raw output reuse count must be zero" in mod.validate_artifact(bad)

    bad = deepcopy(artifact)
    bad["candidate_headroom_by_partition"]["development"]["has_headroom"] = False
    bad["reproducibility_checksum"] = mod.payload_checksum(bad)
    assert "each partition must have candidate headroom" in mod.validate_artifact(bad)

    raw_present_dir = tmp_path / "raw-present"
    (raw_present_dir / "raw_outputs").mkdir(parents=True)
    calls = {"generation": 0}

    def never_generate(**_kwargs: Any) -> dict[str, Any]:
        calls["generation"] += 1
        raise AssertionError("generation must not run after blocked preconditions")

    def raw_path_blocked(**kwargs: Any) -> list[dict[str, Any]]:
        rows = _host_ok(**kwargs)
        rows[-1]["available"] = False
        rows[-1]["detail"] = "raw_output_directory_preexisted"
        return rows

    paths = _model_paths(tmp_path / "blocked-models")
    blocked = mod.run(
        date="20260815",
        result_path=tmp_path / "blocked.json",
        data_dir=raw_present_dir,
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=_tokenizer,
        host_preflight_func=raw_path_blocked,
        generation_func=never_generate,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=True,
    )
    assert blocked["status"] == "blocked"
    assert blocked["sota_corpus_ready_score"] == 0.0
    assert "new_raw_output_paths" in blocked["blocked_reason"]
    assert blocked["honest_verdict"].startswith("blocked_")
    assert calls["generation"] == 0
    assert (tmp_path / "blocked.json").is_file()


def test_req_infra_6450_defensive_validation_and_parser_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """REQ-INFRA-6450: defensive parser, simulator, and schema branches are covered."""

    artifact = _artifact(tmp_path / "base")
    problem = mod.build_policy_problems()[0]

    assert mod.sha256_file(tmp_path / "missing.bin") is None
    assert mod.read_json_object(tmp_path / "missing.json") == {}
    not_object = tmp_path / "not-object.json"
    not_object.write_text("[]", encoding="utf-8")
    assert mod.read_json_object(not_object) == {}
    assert mod.model_slug("!!!") == "model"
    snapshot = tmp_path / "snapshots" / "rev123" / "model-plain.gguf"
    snapshot.parent.mkdir(parents=True)
    snapshot.write_text("fixture", encoding="utf-8")
    assert mod._revision_from_path(snapshot) == "rev123"
    assert mod._quantization_from_path(snapshot) == "unknown"

    empty_specs = mod.build_model_specs(cached_pair_func=lambda **_kwargs: [], tokenizer_func=_tokenizer)
    assert empty_specs["all_resolved"] is False
    missing_path = tmp_path / "absent.gguf"

    def missing_cached_pair(**kwargs: Any) -> list[dict[str, Any]]:
        model_indices = kwargs.get("model_indices")
        ids = (
            (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[2])
            if model_indices is None
            else (mod.MANDATED_MODEL_IDS[0], mod.MANDATED_MODEL_IDS[1])
        )
        return [
            {
                "name": mod.MODEL_TEMPLATE_BY_ID[model_id]["name"],
                "hf_id": model_id,
                "gpu": gpu,
                "model_path": str(missing_path),
            }
            for gpu, model_id in zip((0, 1), ids, strict=True)
        ]

    missing_specs = mod.build_model_specs(cached_pair_func=missing_cached_pair, tokenizer_func=_tokenizer)
    assert any("model_path_missing" in reason for reason in missing_specs["blocked_reasons"])

    paths = _model_paths(tmp_path / "tokenizer-fail-models")

    def bad_tokenizer(_path: str) -> tuple[bool, str]:
        return False, "fixture tokenizer failure"

    bad_token_specs = mod.build_model_specs(
        cached_pair_func=_cached_pair(paths, []),
        tokenizer_func=bad_tokenizer,
    )
    assert any(
        "embedded_tokenizer_not_loadable" in reason
        for reason in bad_token_specs["blocked_reasons"]
    )

    with pytest.raises(ValueError, match="unknown fixture action plan mode"):
        mod.fixture_action_plan(problem, "unknown")

    parser_cases = [
        (b"\xff", "unicode_decode"),
        (b"[]", "candidate_not_object"),
        (
            json.dumps(
                {
                    "problem_id": "wrong",
                    "candidate_seed": mod.CANDIDATE_SEEDS[0],
                    "actions": [],
                }
            ).encode("utf-8"),
            "problem_id_mismatch",
        ),
        (
            json.dumps(
                {
                    "problem_id": problem["problem_id"],
                    "candidate_seed": 1,
                    "actions": [],
                }
            ).encode("utf-8"),
            "candidate_seed_mismatch",
        ),
        (
            json.dumps(
                {
                    "problem_id": problem["problem_id"],
                    "candidate_seed": mod.CANDIDATE_SEEDS[0],
                    "actions": {},
                }
            ).encode("utf-8"),
            "actions_not_list",
        ),
        (
            json.dumps(
                {
                    "problem_id": problem["problem_id"],
                    "candidate_seed": mod.CANDIDATE_SEEDS[0],
                    "actions": ["bad"],
                }
            ).encode("utf-8"),
            "action_0_not_object",
        ),
        (
            json.dumps(
                {
                    "problem_id": problem["problem_id"],
                    "candidate_seed": mod.CANDIDATE_SEEDS[0],
                    "actions": [{"action": "teleport", "args": {}}],
                }
            ).encode("utf-8"),
            "action_0_schema_mismatch",
        ),
    ]
    for raw, expected in parser_cases:
        parsed = mod.parse_candidate_line(raw, problem, mod.CANDIDATE_SEEDS[0])
        assert expected in parsed["parse_error"]

    assert mod._candidate_line_map("bad-json\n[]\n{}", mod.CANDIDATE_SEEDS[0]) == {}

    simulator_cases = [
        ([{"action": "move", "args": {"direction": "up"}}], "invalid_direction:up"),
        ([{"action": "move", "args": {"direction": "west"}}], "move_out_of_bounds"),
        ([{"action": "inspect", "args": {"entity": 7}}], "inspect_missing_entity"),
        ([{"action": "move", "args": {"direction": "east"}}, {"action": "move", "args": {"direction": "east"}}, {"action": "pickup", "args": {"item": problem["fixed_entities"]["payload_item"]}}], "payload_without_badge"),
        ([{"action": "deliver", "args": {"item": problem["fixed_entities"]["payload_item"]}}], "deliver_precondition_failed"),
        ([{"action": "unknown", "args": {}}], "unknown_action:unknown"),
    ]
    for actions, expected in simulator_cases:
        exact = mod.simulate_action_plan(problem, {"parse_valid": True, "actions": actions})
        assert expected in exact["legality_violations"]
        assert exact["exact_success"] is False

    protected_item = problem["fixed_entities"]["protected_item"]
    protected_pickup = [
        *mod._direction_steps(
            tuple(problem["observable_facts"]["start"]),
            tuple(problem["observable_facts"]["items"][protected_item]),
        ),
        {"action": "pickup", "args": {"item": protected_item}},
    ]
    exact = mod.simulate_action_plan(problem, {"parse_valid": True, "actions": protected_pickup})
    assert "protected_item_touched" in exact["protected_violations"]

    too_long = [{"action": "inspect", "args": {"entity": "objective"}}] * 17
    exact = mod.simulate_action_plan(problem, {"parse_valid": True, "actions": too_long})
    assert "action_budget_exceeded" in exact["protected_violations"]

    assert mod._gate_summary(
        {"blocked_reason": "", "sota_corpus_ready_score": 0.0, "current_adversarial_findings": []}
    ) == "readiness closed: non-critical gate failure"
    assert mod._gate_summary({"blocked_reason": "model_resolution"}) == "blocked: model_resolution"

    parse_failed_rows = [deepcopy(artifact["per_unit_rows"]["rows"][0])]
    parse_failed_rows[0]["parse_valid"] = False
    assert mod.recompute_aggregates_from_rows(parse_failed_rows)["parse_failures_by_model"][
        parse_failed_rows[0]["model_hf_id"]
    ] == 1

    blocked_refresh = deepcopy(artifact)
    blocked_refresh["blocked_reason"] = "manual block"
    mod.refresh_terminal_fields(blocked_refresh)
    assert blocked_refresh["status"] == "blocked"
    assert blocked_refresh["honest_verdict"] == "blocked_manual_block"

    validation_cases = [
        (lambda row: row.pop("status"), "missing required field: status"),
        (lambda row: row.update(MODEL_SPECS=[]), "MODEL_SPECS mandated ids mismatch"),
        (lambda row: row.update(models_used=["bad"]), "models_used must be empty or match mandated ids"),
        (lambda row: row.update(autotokenizer_usage_count=1), "autotokenizer_usage_count must be zero"),
        (lambda row: row.update(inference_substrate="wrong"), "inference_substrate mismatch"),
        (lambda row: row.update(verifier_is_oracle=False), "verifier_is_oracle must be true for simulator only"),
        (lambda row: row["per_unit_rows"].update(row_count=1), "per_unit_rows must contain every candidate"),
        (lambda row: row["sealed_problem_and_partition_manifest"].update(problem_count=35), "sealed manifest problem_count must be 36"),
        (lambda row: row["sealed_problem_and_partition_manifest"].update(partition_counts={}), "partition counts must be sealed 12/12/12"),
        (
            lambda row: row["raw_output_uniqueness_and_reuse_count"].update(missing_raw_hash_count=1),
            "ready artifact cannot have missing raw hashes",
        ),
        (lambda row: row.update(cpu_fallback_count=1), "cpu_fallback_count must be zero"),
        (lambda row: row["attack_matrix"].update(all_critical_fail_closed=False), "attack matrix must fail closed"),
        (lambda row: row["attack_matrix"].update(false_accept_count=1), "ready artifact cannot accept attacks"),
        (lambda row: row["aggregate_row_recomputation"].update(matches_reported=False), "reported aggregates must recompute from rows"),
        (lambda row: row["per_unit_rows"]["rows"][0].update(model_ranking_claim=True), "model ranking claim is forbidden"),
        (lambda row: row.update(candidate_headroom_by_partition={}), "each partition must have candidate headroom"),
        (
            lambda row: row["candidate_headroom_by_partition"]["development"].update(
                mixed_exact_outcomes=False
            ),
            "each partition must have mixed exact outcomes",
        ),
        (lambda row: row.update(field_principles={}), "missing field_principles entry: status"),
        (
            lambda row: row["field_principles"].pop(
                "sota_corpus_ready_score:raw_hashes_unique"
            ),
            "missing readiness field_principles entry: raw_hashes_unique",
        ),
        (lambda row: row.update(field_provenance={}), "field_provenance must cover exactly required fields"),
        (lambda row: row.update(honest_verdict="bad prefix"), "honest_verdict lacks required terminal prefix"),
        (lambda row: row.update(reproducibility_checksum="sha256:bad"), "reproducibility_checksum mismatch"),
        (lambda row: row.update(duration_s=0.0), "sota_corpus_ready_score does not recompute"),
    ]
    for mutate, expected in validation_cases:
        bad = deepcopy(artifact)
        mutate(bad)
        if expected not in {
            "reproducibility_checksum mismatch",
            "sota_corpus_ready_score does not recompute",
        }:
            bad["reproducibility_checksum"] = mod.payload_checksum(bad)
        assert expected in mod.validate_artifact(bad)

    with monkeypatch.context() as mp:
        mp.setattr(mod, "validate_artifact", lambda _payload: ["forced schema error"])
        failed = mod.run(
            date="20260815",
            result_path=tmp_path / "failed.json",
            data_dir=tmp_path / "failed-data",
            cached_pair_func=_cached_pair(_model_paths(tmp_path / "failed-models"), []),
            tokenizer_func=_tokenizer,
            host_preflight_func=_host_ok,
            generation_func=_fixture_generation,
            test_exit_codes=_test_exit_codes(),
            duration_s=125.0,
            write=False,
        )
    assert failed["status"] == "failed_schema"
    assert failed["honest_verdict"].startswith("complete_failed_schema:")

    model_blocked = mod.run(
        date="20260815",
        result_path=tmp_path / "model-blocked.json",
        data_dir=tmp_path / "model-blocked-data",
        cached_pair_func=lambda **_kwargs: [],
        tokenizer_func=_tokenizer,
        host_preflight_func=_host_ok,
        generation_func=_fixture_generation,
        test_exit_codes=_test_exit_codes(),
        duration_s=0.0,
        write=False,
    )
    assert model_blocked["status"] == "blocked"
    assert "model_resolution" in model_blocked["blocked_reason"]
