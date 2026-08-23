"""Tests for Exp6528 V565 source, model, and method contract.

Spec refs: REQ-REPORT-6528, SCENARIO-REPORT-6528-SOURCES,
SCENARIO-REPORT-6528-DRIFT, SCENARIO-REPORT-6528-CACHE,
SCENARIO-REPORT-6528-METHODS, SCENARIO-REPORT-6528-SCHEMA.
"""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any

import pytest

from carnot import experiment_6528_v565_source_model_method_contract as mod


REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / mod.SPEC_RELATIVE_PATH

FOCUSED_COMMAND = (
    ".venv/bin/pytest tests/python/test_experiment_6528_v565_source_model_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_RUN_COMMAND = (
    ".venv/bin/coverage run --rcfile=/dev/null "
    "--include=python/carnot/experiment_6528_v565_source_model_method_contract.py "
    "-m pytest tests/python/test_experiment_6528_v565_source_model_method_contract.py "
    "-q --no-cov -n 0"
)
COVERAGE_REPORT_COMMAND = (
    ".venv/bin/coverage report --rcfile=/dev/null "
    "--include=python/carnot/experiment_6528_v565_source_model_method_contract.py "
    "--fail-under=100 --show-missing"
)
FULL_PYTEST_COMMAND = ".venv/bin/pytest tests/python -q"
SPEC_COMMAND = (
    ".venv/bin/python scripts/check_spec_coverage.py "
    "tests/python/test_experiment_6528_v565_source_model_method_contract.py"
)
RUN_COMMAND = (
    "cd /home/ianblenke/github.com/ianblenke/carnot && "
    ".venv/bin/python -m carnot.experiment_6528_v565_source_model_method_contract "
    "--date 20260823"
)
ADVERSARIAL_COMMAND = (
    ".venv/bin/python scripts/adversarial_verify.py "
    "results/experiment_6528_v565_source_model_method_contract.json"
)
VALIDATE_COMMAND = (
    ".venv/bin/python -m carnot.experiment_6528_v565_source_model_method_contract --validate"
)
GIT_STATUS_COMMAND = "git status --short"

TESTS_RUN = [
    {"command": FOCUSED_COMMAND, "exit_code": 0},
    {"command": COVERAGE_RUN_COMMAND, "exit_code": 0},
    {"command": COVERAGE_REPORT_COMMAND, "exit_code": 0},
    {"command": SPEC_COMMAND, "exit_code": 0},
    {"command": RUN_COMMAND, "exit_code": 0},
    {"command": ADVERSARIAL_COMMAND, "exit_code": 0},
    {"command": VALIDATE_COMMAND, "exit_code": 0},
    {"command": GIT_STATUS_COMMAND, "exit_code": 0},
    {"command": FULL_PYTEST_COMMAND, "exit_code": 0},
]


def _receipt(
    url: str,
    body: str,
    *,
    status_code: int = 200,
    error: str | None = None,
) -> mod.JsonDict:
    return {
        "ok": error is None and 200 <= status_code < 400,
        "status_code": status_code,
        "url": url,
        "headers": {"content-type": "application/test"},
        "body": body,
        "error": error,
    }


def _paper_body(source_id: str) -> str:
    source = mod.SOURCE_BY_ID[source_id]
    date = source["expected_source_date"]
    title = source["title"]
    return (
        "<html>"
        f"<title>{title}</title>"
        f"<h1>Title:{title}</h1>"
        f"<div>[Submitted on {date}]</div>"
        f"<div>[v1] Fri, {date} 12:00:00 UTC</div>"
        "<blockquote>Abstract: fixture method claim.</blockquote>"
        "</html>"
    )


def _fake_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id == "network_probe":
        return _receipt(url, "ok")
    if source_id in mod.SOURCE_BY_ID:
        source = mod.SOURCE_BY_ID[source_id]
        if source["source_kind"] == "paper":
            return _receipt(url, _paper_body(source_id))
        if source_id in {"openreview_dc_energy", "openreview_linear_decision_rules"}:
            body = json.dumps(
                {"title": source["title"], "source_date": source["expected_source_date"]}
            )
            return _receipt(url, body)
        if source_id == "semantic_scholar_ebt":
            return _receipt(
                url,
                json.dumps(
                    {
                        "title": "Energy-Based Transformers are Scalable Learners and Thinkers",
                        "citationCount": 35,
                        "citations": [
                            {
                                "title": "Memoir: Should a Model Write to Its Memory While It Thinks?"
                            },
                            {"title": "Solver-Hard Is Not Model-Hard"},
                        ],
                    }
                ),
            )
        if source_id == "semantic_scholar_arm_ebm":
            return _receipt(
                url,
                json.dumps(
                    {
                        "title": "Autoregressive Language Models are Secretly Energy-Based Models",
                        "citationCount": 8,
                        "citations": [{"title": "Distributional Energy-Based Models"}],
                    }
                ),
            )
        if source_id == "huggingface_support_reshaping":
            return _receipt(
                url, json.dumps({"title": source["title"], "source_date": "2026-07-31"})
            )
        if source_id == "drift_bench_repo":
            return _receipt(
                url, "<html><title>GitHub - kaons-research/drift-bench</title>MIT license</html>"
            )
        if source_id == "extropic_z1_status":
            return _receipt(
                url, "<html><title>From One to One Billion</title>Z1 early access 2027</html>"
            )
        if source_id == "logical_intelligence_kona":
            return _receipt(
                url, "<html><title>Kona: Energy-Based Models for AI Reasoning</title></html>"
            )
    if source_id == "drift_repo_api":
        return _receipt(
            url,
            json.dumps(
                {
                    "license": {"spdx_id": "MIT"},
                    "default_branch": "main",
                    "full_name": "kaons-research/drift-bench",
                }
            ),
        )
    if source_id == "drift_commit_api":
        return _receipt(url, json.dumps({"sha": "d24cda4f59a6ee06bafe886f4724899a7ec94f1c"}))
    if source_id == "drift_readme_pinned":
        return _receipt(url, "The original run's SQLite databases suffered filesystem corruption.")
    if source_id == "drift_license_pinned":
        return _receipt(url, "MIT License")
    if source_id == "drift_schema_pinned":
        return _receipt(url, "JSON schema: domain, problem_id, turns, constraints, answer")
    raise AssertionError(f"unexpected fetch {source_id}: {url}")


def _blocked_fetcher(url: str, source_id: str) -> mod.JsonDict:
    if source_id == "openreview_dc_energy":
        return _receipt(url, "Forbidden", status_code=403, error="HTTP 403")
    return _fake_fetcher(url, source_id)


def _fake_runner(args: list[str], cwd: Path) -> tuple[int, str, str]:
    joined = " ".join(args)
    assert cwd == REPO
    if "sweep_clusters.py" in joined:
        return 0, "http://export.arxiv.org/api/query?search_query=fixture\n", ""
    if "sweep_semscholar.py" in joined:
        return 0, "2605.23940\n2607.20792\n2608.00220\n", "# semscholar: fixture\n"
    raise AssertionError(f"unexpected command: {joined}")


def _model_resolvers(
    tmp_path: Path,
) -> tuple[mod.ModelPairResolver, mod.GgufResolver, dict[str, int]]:
    calls = {"pair": 0, "gguf": 0}
    tmp_path.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for model in mod.SOTA_GGUF_MODELS:
        filename = f"{model['name']}-{model['quantization']}.gguf"
        path = tmp_path / filename
        path.write_bytes(f"fixture-{model['hf_id']}".encode())
        paths[model["hf_id"]] = path

    def pair_resolver(
        gpu_indices: tuple[int, int] = (0, 1),
        preferred_quant: str = "Q4_K_M",
        model_indices: tuple[int, int] | None = None,
    ) -> list[dict[str, Any]]:
        del preferred_quant, model_indices
        calls["pair"] += 1
        return [
            {
                "name": mod.SOTA_GGUF_MODELS[0]["name"],
                "hf_id": mod.SOTA_GGUF_MODELS[0]["hf_id"],
                "gpu": gpu_indices[0],
                "model_path": str(paths[mod.SOTA_GGUF_MODELS[0]["hf_id"]]),
            },
            {
                "name": mod.SOTA_GGUF_MODELS[1]["name"],
                "hf_id": mod.SOTA_GGUF_MODELS[1]["hf_id"],
                "gpu": gpu_indices[1],
                "model_path": str(paths[mod.SOTA_GGUF_MODELS[1]["hf_id"]]),
            },
        ]

    def gguf_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> str | None:
        del preferred_quant
        calls["gguf"] += 1
        return str(paths[hf_id])

    return pair_resolver, gguf_resolver, calls


def _missing_pair_resolver(
    gpu_indices: tuple[int, int] = (0, 1),
    preferred_quant: str = "Q4_K_M",
    model_indices: tuple[int, int] | None = None,
) -> None:
    del gpu_indices, preferred_quant, model_indices
    return None


def _missing_gguf_resolver(hf_id: str, preferred_quant: str = "Q4_K_M") -> None:
    del hf_id, preferred_quant
    return None


def _artifact(
    tmp_path: Path, *, blocked: bool = False, missing_models: bool = False
) -> dict[str, Any]:
    pair_resolver, gguf_resolver, _calls = _model_resolvers(tmp_path / "models")
    return mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        run_date="20260823",
        fetcher=_blocked_fetcher if blocked else _fake_fetcher,
        command_runner=_fake_runner,
        cached_pair_resolver=_missing_pair_resolver if missing_models else pair_resolver,
        gguf_resolver=_missing_gguf_resolver if missing_models else gguf_resolver,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )


def _with_checksum(artifact: dict[str, Any]) -> dict[str, Any]:
    artifact["reproducibility_checksum"] = mod.reproducibility_checksum(artifact)
    return artifact


def test_req_report_6528_spec_declares_contract() -> None:
    """REQ-REPORT-6528: OpenSpec owns the Exp6528 artifact contract."""

    text = SPEC.read_text(encoding="utf-8")
    section = text[text.index("REQ-REPORT-6528") :]
    normalized = " ".join(section.split())

    for token in (
        "SCENARIO-REPORT-6528-SOURCES",
        "SCENARIO-REPORT-6528-DRIFT",
        "SCENARIO-REPORT-6528-CACHE",
        "SCENARIO-REPORT-6528-METHODS",
        "SCENARIO-REPORT-6528-SCHEMA",
        mod.RESULT_RELATIVE_PATH.as_posix(),
        mod.INFERENCE_SUBSTRATE,
        "`verifier_is_oracle=false`",
        "`verdict_class` SHALL be `null` or `partial`",
        "A blocked `honest_verdict` SHALL name each unavailable required channel or cache contract.",
    ):
        assert token in section
    for field, principle in mod.FIELD_PRINCIPLES.items():
        assert f"`{field}`" in section
        assert " ".join(principle.split()) in normalized


def test_scenario_report_6528_sources_and_drift_contract(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6528-SOURCES/DRIFT: primary rows and DRIFT contract are pinned."""

    artifact = _artifact(tmp_path)
    rows = {row["source_id"]: row for row in artifact["source_rows"]}
    drift = artifact["drift_bench_provenance_contract"]

    assert len(rows) == len(mod.SOURCE_MANIFEST)
    assert rows["drift_bench_arxiv"]["source_date"] == "2026-04-28"
    assert rows["memoir"]["source_date_verified"] is True
    assert rows["support_reshaping"]["code_or_data_availability"] == "public_github_claimed"
    assert rows["distributional_ebm"]["method"] == "decomposed_energy_uncertainty_abstention"
    assert (
        rows["solver_hard"]["local_applicability"]
        == "surface_hardness_stratified_embedding_diagnostic"
    )
    assert rows["openreview_dc_energy"]["source_kind"] == "openreview_forum"
    assert (
        rows["openreview_linear_decision_rules"]["method_transfer_status"]
        == "future_architecture_control"
    )
    assert rows["drift_bench_repo"]["stable_url"] == mod.DRIFT_REPO_URL
    assert rows["drift_bench_repo"]["retrieval_url"] == mod.DRIFT_API_REPO_URL
    assert rows["drift_bench_repo"]["primary_url_verified"] is True
    assert (
        rows["extropic_z1_status"]["exact_authority_boundary"]
        == "Product page only; no local TSU runner or access."
    )
    assert (
        rows["logical_intelligence_kona"]["non_transfer_boundary"]
        == "No proprietary Kona weights, runner, or speed claim transfers."
    )

    for row in artifact["source_rows"]:
        assert row["source_hash"].startswith("sha256:")
        assert row["method"]
        assert row["method_claim"]
        assert row["code_or_data_availability"]
        assert row["local_applicability"]
        assert row["non_transfer_boundary"]
        assert row["exact_authority_boundary"]

    assert artifact["primary_source_hashes"]["drift_bench_arxiv"].startswith("sha256:")
    assert artifact["citation_trail_receipts"][0]["observed_citation_count"] == 35
    assert artifact["citation_trail_receipts"][1]["observed_citation_count"] == 8
    assert all(
        row["count_is_current_guarantee"] is False for row in artifact["citation_count_boundaries"]
    )

    assert drift["repo_url"] == "https://github.com/kaons-research/drift-bench"
    assert drift["immutable_revision"] == "d24cda4f59a6ee06bafe886f4724899a7ec94f1c"
    assert drift["license"] == "MIT"
    assert drift["data_schema_path"] == "data/problems/README.md"
    assert drift["upstream_corruption_warning_present"] is True
    assert drift["local_result_receipts_regenerated"] is True
    assert drift["contract_ready"] is True


def test_scenario_report_6528_cache_methods_authority_and_schema(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6528-CACHE/METHODS/SCHEMA: rows recompute readiness."""

    pair_resolver, gguf_resolver, calls = _model_resolvers(tmp_path / "models")
    artifact = mod.build_artifact(
        repo_root=REPO,
        result_path=tmp_path / mod.RESULT_RELATIVE_PATH.name,
        run_date="20260823",
        fetcher=_fake_fetcher,
        command_runner=_fake_runner,
        cached_pair_resolver=pair_resolver,
        gguf_resolver=gguf_resolver,
        write=True,
        duration_s=1.0,
        tests_run=TESTS_RUN,
        now_utc="2026-08-23T12:00:00Z",
    )
    written = json.loads((tmp_path / mod.RESULT_RELATIVE_PATH.name).read_text(encoding="utf-8"))

    assert calls["pair"] == 1
    assert written == artifact
    assert mod.validate_artifact(artifact) == []
    assert set(artifact) == set(mod.REQUIRED_ARTIFACT_FIELDS)
    assert artifact["status"] == "complete_v565_source_model_method_contract_ready"
    assert artifact["honest_verdict"].startswith("complete_v565_source_model_method_contract_ready")
    assert artifact["verdict_class"] is None
    assert artifact["v565_method_contract_ready_score"] == 1.0
    assert artifact["inference_substrate"] == mod.INFERENCE_SUBSTRATE
    assert artifact["verifier_is_oracle"] is False
    assert artifact["reproducibility_checksum"] == mod.reproducibility_checksum(artifact)

    cache_rows = artifact["model_cache_resolution_rows"]
    assert len(cache_rows) == len(mod.SOTA_GGUF_MODELS)
    assert {row["hf_id"] for row in cache_rows} == {
        model["hf_id"] for model in mod.SOTA_GGUF_MODELS
    }
    assert all(row["cache_hit"] is True for row in cache_rows)
    assert all(row["model_loaded_or_run"] is False for row in cache_rows)
    assert all(row["model_file_sha256"].startswith("sha256:") for row in cache_rows)
    assert cache_rows[0]["selected_by_cached_sota_pair"] is True

    assert artifact["frozen_external_split_contract"]["downstream_field_spelling"] == [
        "split_name",
        "base_problem_id",
        "domain",
        "turn_index",
        "source_row_hash",
        "chronology_index",
    ]
    assert artifact["frozen_router_contract"]["learned_advice_may_prune"] is False
    assert artifact["frozen_embedding_contract"]["answer_scoring_allowed"] is False
    assert (
        artifact["frozen_transactional_learning_contract"]["same_query_write_negative_control"]
        is True
    )
    assert (
        artifact["frozen_arc_parser_contract"]["qwen3_xml_stop_rule"]
        == "stop_after_one_bounded_live_tool_call_receipt"
    )
    assert (
        artifact["hardware_stop_contract"]["gatemate_command_allowed_without_new_receipt"] is False
    )

    aggregate = artifact["aggregate_row_recomputation"]
    assert aggregate["source_rows_cover_manifest"] is True
    assert aggregate["required_primary_sources_verified"] is True
    assert aggregate["drift_bench_contract_ready"] is True
    assert aggregate["model_cache_contract_ready"] is True
    assert aggregate["adopted_methods_with_source_hook_control_boundary"] == len(
        mod.ADOPTED_SOURCE_IDS
    )
    assert aggregate["frozen_downstream_field_spelling_complete"] is True
    assert aggregate["ready_score_from_rows"] == 1.0
    assert artifact["gate_check_summary"]["all_gates_passed"] is True
    assert artifact["protected_files_unchanged"]["all_protected_files_unchanged"] is True
    assert artifact["preconditions_checked"]["no_high_concurrency_research_harness"] is True
    assert {row["row_type"] for row in artifact["per_unit_rows"]} >= {
        "source",
        "model_cache",
        "non_transfer",
        "contract",
        "gate",
    }


def test_scenario_report_6528_blocked_cache_or_source_is_partial(tmp_path: Path) -> None:
    """SCENARIO-REPORT-6528-SCHEMA: source or cache failure forces partial verdict."""

    artifact = _artifact(tmp_path, blocked=True, missing_models=True)
    failures = artifact["gate_check_summary"]["failed_checks"]
    rows = {row["source_id"]: row for row in artifact["source_rows"]}

    assert mod.validate_artifact(artifact) == []
    assert artifact["status"] == "blocked_v565_source_model_method_contract"
    assert artifact["honest_verdict"].startswith("blocked_v565_source_model_method_contract")
    assert "openreview" in artifact["honest_verdict"]
    assert "model_cache_preflight" in artifact["honest_verdict"]
    assert artifact["verdict_class"] == "partial"
    assert artifact["v565_method_contract_ready_score"] == 0.0
    assert rows["openreview_dc_energy"]["primary_url_verified"] is False
    assert rows["openreview_dc_energy"]["observed_error"] == "HTTP 403"
    assert all(row["cache_hit"] is False for row in artifact["model_cache_resolution_rows"])
    assert any(item["channel"] == "openreview" for item in failures)
    assert any(item["channel"] == "model_cache_preflight" for item in failures)


def test_scenario_report_6528_validation_edges(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-REPORT-6528-SCHEMA: malformed artifacts fail closed."""

    artifact = _artifact(tmp_path)

    assert mod.tests_run_receipts(TESTS_RUN) == TESTS_RUN
    assert all(row["exit_code"] == 0 for row in mod.tests_run_receipts(None))
    assert mod.sha256_file(tmp_path / "missing.gguf") == "missing"
    assert mod._extract_title('{"title": "JSON Title"}', "fallback") == "JSON Title"
    assert (
        mod._extract_title(
            '{"notes": [{"content": {"title": {"value": "Note Title"}}}]}', "fallback"
        )
        == "Note Title"
    )
    assert mod._extract_title("<title>HTML Title</title>", "fallback") == "HTML Title"
    assert mod._extract_title("<h1>Title:Arxiv Title</h1>", "fallback") == "Arxiv Title"
    assert mod._extract_title("{bad", "fallback") == "fallback"
    assert mod._extract_source_date('{"source_date": "2026-08-23"}') == "2026-08-23"
    assert mod._extract_source_date("[Submitted on 31 Jul 2026]") == "2026-07-31"
    assert mod._extract_source_date("[v1] Sun, 19 Jul 2026 03:23:22 UTC") == "2026-07-19"
    assert mod._extract_source_date("no date") is None
    assert mod._retrieval_state(200, "") == "available"
    assert mod._retrieval_state(200, "Verifying your browser") == "blocked"
    assert mod._retrieval_state(429, "Too Many Requests") == "rate_limited"
    assert mod._retrieval_state(404, "not found") == "not_found"
    assert mod._retrieval_state(503, "server") == "blocked"
    assert mod._citation_count(json.dumps({"citationCount": 7})) == 7
    assert mod._citation_count("{bad") is None
    assert mod._citation_titles(json.dumps({"citations": [{"title": "A"}, {"title": ""}, {}]})) == [
        "A"
    ]
    assert mod._citation_titles("{bad") == []
    assert mod._network_probe(_fake_fetcher, "2026-08-23T12:00:00Z")["network_available"] is True
    assert (
        mod._network_probe(
            lambda url, sid: _receipt(url, "bad", status_code=503, error="boom"),
            "2026-08-23T12:00:00Z",
        )["network_available"]
        is False
    )
    assert mod.blocked_verdict_channels(
        {
            "failed_checks": [
                "malformed",
                {"channel": "local_gate", "check": "mandated_gguf_cache_resolved"},
            ]
        }
    ) == ["model_cache_preflight"]
    assert (
        mod.blocked_honest_verdict({"failed_checks": []})
        == "blocked_v565_source_model_method_contract: source or model-cache gates failed"
    )

    bad = deepcopy(artifact)
    del bad["status"]
    _with_checksum(bad)
    assert any("required field set mismatch" in error for error in mod.validate_artifact(bad))

    bad = deepcopy(artifact)
    bad["unexpected"] = True
    bad["field_principles"] = {}
    bad["field_provenance"] = {}
    bad["verdict_class"] = "blocked"
    bad["inference_substrate"] = "live_llm"
    bad["verifier_is_oracle"] = True
    bad["honest_verdict"] = "ok"
    _with_checksum(bad)
    errors = mod.validate_artifact(bad)
    assert any("required field set mismatch" in error for error in errors)
    assert "field_principles mismatch" in errors
    assert "field_provenance must cover required fields" in errors
    assert "verdict_class outside Exp6528 enum" in errors
    assert "honest_verdict terminal prefix mismatch" in errors
    assert "inference_substrate mismatch" in errors
    assert "verifier_is_oracle must be false" in errors

    mutations = [
        (
            "ready score mismatch",
            lambda item: item.__setitem__("v565_method_contract_ready_score", 0.0),
        ),
        ("source rows must cover manifest", lambda item: item.__setitem__("source_rows", [])),
        (
            "required primary sources must verify",
            lambda item: item["source_rows"][0].__setitem__("primary_url_verified", False),
        ),
        (
            "adopted methods must map to source hooks controls and boundaries",
            lambda item: item["source_rows"][0].__setitem__("negative_control", ""),
        ),
        (
            "drift bench provenance contract must be ready",
            lambda item: item["drift_bench_provenance_contract"].__setitem__(
                "contract_ready", False
            ),
        ),
        (
            "model cache contract must cover all mandated models",
            lambda item: item.__setitem__("model_cache_resolution_rows", []),
        ),
        (
            "frozen contracts must expose downstream field spelling",
            lambda item: item["frozen_router_contract"].__setitem__(
                "downstream_field_spelling", []
            ),
        ),
        (
            "non-transfer rows must forbid unsupported transfer",
            lambda item: item["non_transfer_rows"][0].__setitem__("non_transfer_boundary", ""),
        ),
        (
            "learned routing cannot prune candidates",
            lambda item: item["frozen_router_contract"].__setitem__(
                "learned_advice_may_prune", True
            ),
        ),
        (
            "protected files changed",
            lambda item: item["protected_files_unchanged"].__setitem__(
                "all_protected_files_unchanged", False
            ),
        ),
        (
            "query receipts must include all required sequential channels",
            lambda item: item.__setitem__("query_receipts", []),
        ),
        (
            "primary_source_hashes must cover manifest",
            lambda item: item.__setitem__("primary_source_hashes", {}),
        ),
    ]
    for expected, mutate in mutations:
        broken = deepcopy(artifact)
        mutate(broken)
        _with_checksum(broken)
        assert expected in mod.validate_artifact(broken)

    bad = deepcopy(artifact)
    bad["reproducibility_checksum"] = "sha256:bad"
    assert "reproducibility_checksum mismatch" in mod.validate_artifact(bad)

    bad = _artifact(tmp_path / "blocked_validation", blocked=True, missing_models=True)
    bad["honest_verdict"] = "blocked_v565_source_model_method_contract: source gates failed"
    _with_checksum(bad)
    assert "blocked verdict must name unavailable required channels" in mod.validate_artifact(bad)

    assert (
        mod.main(["--validate", "--result-path", str(tmp_path / mod.RESULT_RELATIVE_PATH.name)])
        == 0
    )
    assert (
        mod.main(
            [
                "--date",
                "20260823",
                "--result-path",
                str(tmp_path / "cli.json"),
                "--no-live-network",
            ]
        )
        == 0
    )
    assert "experiment_6528_v565_source_model_method_contract.json" in capsys.readouterr().out

    invalid_json_path = tmp_path / "invalid.json"
    invalid_json_path.write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="expected JSON object"):
        mod._load_json(invalid_json_path)

    bad_path = tmp_path / "bad_artifact.json"
    broken = deepcopy(artifact)
    del broken["status"]
    _with_checksum(broken)
    bad_path.write_text(json.dumps(broken), encoding="utf-8")
    assert mod.main(["--validate", "--result-path", str(bad_path)]) == 1

    monkeypatch.setattr(mod, "build_artifact", lambda **_kwargs: {"bad": True})
    assert mod.main(["--date", "20260823", "--result-path", str(tmp_path / "bad_cli.json")]) == 1
