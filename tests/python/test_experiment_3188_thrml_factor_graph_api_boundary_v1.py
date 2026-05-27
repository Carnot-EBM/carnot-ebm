"""Tests for Exp 3188 THRML factor-graph API boundary v1.

Spec refs: REQ-HW-099, SCENARIO-HW-099.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from carnot.reporting import thrml_factor_graph_api_boundary_3188 as mod


REQUIRED_FIELDS = {
    "thrml_factor_graph_api_boundary_v1_ready",
    "thrml_import_available",
    "thrml_version",
    "selected_exact_rows",
    "factor_graph_translation_records",
    "api_gap_records",
    "local_api_smoke_passed",
    "hardware_speedup_claim_allowed",
    "kona_or_tsu_execution_claimed",
    "inference_substrate",
    "honest_verdict",
}


def _write_json(root: Path, rel_path: Path, payload: dict[str, Any]) -> None:
    path = root / rel_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_common_sources(root: Path) -> None:
    (root / "AGENTS.md").write_text("Read CODEX.md\n", encoding="utf-8")
    (root / "CODEX.md").write_text("Spec First\nTests First\n", encoding="utf-8")
    (root / "CLAUDE.md").write_text("No speedup claims without hardware receipts\n", encoding="utf-8")
    spec = root / "openspec/capabilities/fpga/spec.md"
    spec.parent.mkdir(parents=True, exist_ok=True)
    spec.write_text(
        "REQ-HW-099\nSCENARIO-HW-099\n"
        "results/experiment_3188_thrml_factor_graph_api_boundary_v1.json\n",
        encoding="utf-8",
    )
    (root / "research-hardware-wishlist.md").write_text(
        "THRML import is software context only; no TSU speedup.\n",
        encoding="utf-8",
    )
    (root / "research-references.md").write_text(
        "Extropic THRML and Kona are architecture references only.\n",
        encoding="utf-8",
    )
    _write_json(
        root,
        mod.EXP3180_REL_PATH,
        {
            "controlled_invariance_executor_v2_ready": True,
            "exact_rows_evaluated": [
                {
                    "row_id": "resyn-3084-arith-000",
                    "exact_label": "VALID",
                    "candidate_answers": ["VALID"],
                    "exact_authority_decision": "accept",
                    "known_false_accept_regression": False,
                    "semantic_false_accept": False,
                    "acceptance_authority": True,
                },
                {
                    "row_id": "resyn-3084-arith-003",
                    "exact_label": "INVALID",
                    "candidate_answers": ["INVALID", "VALID"],
                    "exact_authority_decision": "reject",
                    "known_false_accept_regression": True,
                    "semantic_false_accept": False,
                    "acceptance_authority": True,
                },
            ],
            "inference_substrate": {"new_live_model_calls": 0},
        },
    )


def _fake_thrml_module() -> SimpleNamespace:
    class FakeNode:
        pass

    class FakeBlock:
        def __init__(self, nodes: list[Any]) -> None:
            self.nodes = nodes

    class FakeFactor:
        def __init__(self, node_groups: list[Any], weights: Any) -> None:
            self.node_groups = node_groups
            self.weights = weights

        def to_interaction_groups(self) -> list[str]:
            return ["interaction"]

    return SimpleNamespace(
        __version__="0.0-test",
        __file__="/tmp/fake-thrml/__init__.py",
        CategoricalNode=FakeNode,
        Block=FakeBlock,
        models=SimpleNamespace(discrete_ebm=SimpleNamespace(CategoricalEBMFactor=FakeFactor)),
    )


def test_req_hw_099_probe_thrml_records_import_metadata_and_failures() -> None:
    """REQ-HW-099: THRML probes record version/path or an import blocker."""
    fake_thrml = _fake_thrml_module()

    def fake_version(_distribution_name: str) -> str:
        raise mod.metadata.PackageNotFoundError("thrml")

    available = mod.probe_thrml(importer=lambda _name: fake_thrml, version=fake_version)
    blocked = mod.probe_thrml(
        importer=lambda _name: (_ for _ in ()).throw(ModuleNotFoundError("No module named 'thrml'"))
    )

    assert available.import_available is True
    assert available.version == "0.0-test"
    assert available.import_path == "/tmp/fake-thrml/__init__.py"
    assert available.missing_symbols == []
    assert blocked.import_available is False
    assert blocked.import_error == "ModuleNotFoundError: No module named 'thrml'"
    assert blocked.missing_symbols == list(mod.REQUIRED_THRML_SYMBOLS)
    assert mod.has_symbol(fake_thrml, "thrml.Block") is True
    assert mod.has_symbol(SimpleNamespace(), "thrml.Block") is False


def test_req_hw_099_spec_anchor_exists() -> None:
    """REQ-HW-099: OpenSpec declares the THRML factor-graph boundary artifact."""
    spec = (mod.REPO_ROOT / "openspec/capabilities/fpga/spec.md").read_text(encoding="utf-8")

    assert "REQ-HW-099" in spec
    assert "SCENARIO-HW-099" in spec
    assert mod.OUTPUT_REL_PATH.as_posix() in spec
    assert "thrml_factor_graph_api_boundary_v1_ready" in spec


def test_req_hw_099_fallback_row_selection_and_label_routing(tmp_path: Path) -> None:
    """REQ-HW-099: row selection falls back deterministically when preferred ids are absent."""
    _write_json(
        tmp_path,
        mod.EXP3180_REL_PATH,
        {
            "exact_rows_evaluated": [
                {
                    "row_id": "other-1",
                    "exact_label": "SAT",
                    "candidate_answers": ["SAT"],
                    "acceptance_authority": True,
                },
                {
                    "row_id": "other-2",
                    "exact_label": "UNSAT",
                    "candidate_answers": ["VALID"],
                    "acceptance_authority": True,
                },
            ]
        },
    )

    rows = mod.select_exact_rows(tmp_path)

    assert [row["row_id"] for row in rows] == ["other-1", "other-2"]
    assert rows[0]["expected_action"] == "accept"
    assert rows[1]["expected_action"] == "reject"


def test_scenario_hw_099_builds_boundary_without_hardware_claims(tmp_path: Path) -> None:
    """SCENARIO-HW-099: exact rows map to construction-only THRML records."""
    _write_common_sources(tmp_path)
    thrml_probe = mod.ThrmlProbe(
        import_available=True,
        version="0.0-test",
        import_path="/tmp/fake-thrml/__init__.py",
        import_error=None,
        module=_fake_thrml_module(),
        available_symbols={
            "thrml.CategoricalNode": True,
            "thrml.Block": True,
            "thrml.models.discrete_ebm.CategoricalEBMFactor": True,
        },
        missing_symbols=[],
    )

    artifact = mod.build_artifact(
        tmp_path,
        thrml_probe=thrml_probe,
        started_s=1.0,
        now_s=2.0,
        tests_run=("pytest targeted",),
    )

    assert REQUIRED_FIELDS <= artifact.keys()
    assert artifact["thrml_factor_graph_api_boundary_v1_ready"] is True
    assert artifact["thrml_import_available"] is True
    assert artifact["thrml_version"] == "0.0-test"
    assert artifact["local_api_smoke_passed"] is True
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["kona_or_tsu_execution_claimed"] is False
    assert [row["row_id"] for row in artifact["selected_exact_rows"]] == [
        "resyn-3084-arith-000",
        "resyn-3084-arith-003",
    ]
    assert all(row["deterministic_authority"] is True for row in artifact["selected_exact_rows"])
    assert all(
        record["thrml_mapping"]["constructed"] is True
        for record in artifact["factor_graph_translation_records"]
    )
    assert all(
        record["thrml_mapping"]["construction_check"] == "passed"
        for record in artifact["factor_graph_translation_records"]
    )
    assert artifact["api_gap_records"] == [
        {
            "gap_id": "thrml_semantic_metadata_externalized",
            "severity": "adapter_needed",
            "missing_symbols": [],
            "details": "THRML nodes/factors can be constructed, but exact row ids, authority labels, and state-label names remain Carnot-side metadata.",
            "next_adapter_steps": [
                "preserve row_id, exact_label, candidate_answers, and state_labels in the Carnot adapter wrapper",
                "add a round-trip metadata test before any sampler integration",
            ],
        }
    ]
    assert artifact["inference_substrate"] == {
        "kind": "local_thrml_factor_graph_api_construction",
        "local_repo_only": True,
        "executes_hardware": False,
        "hardware_commands_run": [],
        "board_commands_run": [],
        "retired_kv260_host_storage_checks_used": False,
        "executes_models": False,
        "no_live_model_inference": True,
        "installs_packages": False,
        "sampler_benchmark_run": False,
        "sampler_speedup_reported": False,
        "thrml_import_available": True,
        "local_api_smoke_only": True,
        "kona_or_tsu_execution_claimed": False,
    }
    assert artifact["honest_verdict"].startswith("complete:")


def test_req_hw_099_blocked_when_thrml_import_is_missing(tmp_path: Path) -> None:
    """REQ-HW-099: missing THRML import writes a blocked preflight artifact."""
    _write_common_sources(tmp_path)
    thrml_probe = mod.ThrmlProbe(
        import_available=False,
        version=None,
        import_path=None,
        import_error="ModuleNotFoundError: No module named 'thrml'",
        module=None,
        available_symbols={},
        missing_symbols=list(mod.REQUIRED_THRML_SYMBOLS),
    )

    artifact = mod.build_artifact(tmp_path, thrml_probe=thrml_probe)

    assert artifact["thrml_factor_graph_api_boundary_v1_ready"] is False
    assert artifact["thrml_import_available"] is False
    assert artifact["thrml_version"] is None
    assert artifact["local_api_smoke_passed"] is False
    assert artifact["factor_graph_translation_records"]
    assert all(
        record["thrml_mapping"]["constructed"] is False
        for record in artifact["factor_graph_translation_records"]
    )
    assert artifact["api_gap_records"][0] == {
        "gap_id": "thrml_import_unavailable",
        "severity": "blocked_precondition",
        "missing_symbols": list(mod.REQUIRED_THRML_SYMBOLS),
        "details": "ModuleNotFoundError: No module named 'thrml'",
        "next_adapter_steps": [
            "install or repair the local THRML package in the project environment",
            "rerun only the Exp 3188 construction smoke after import succeeds",
        ],
    }
    assert artifact["hardware_speedup_claim_allowed"] is False
    assert artifact["kona_or_tsu_execution_claimed"] is False
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_099_records_construction_failures_as_api_gaps(tmp_path: Path) -> None:
    """REQ-HW-099: construction errors are blocked API gaps, not speedup claims."""
    _write_common_sources(tmp_path)

    class FakeNode:
        pass

    class FakeBlock:
        def __init__(self, nodes: list[Any]) -> None:
            self.nodes = nodes

    class FailingFactor:
        def __init__(self, node_groups: list[Any], weights: Any) -> None:
            self.node_groups = node_groups
            self.weights = weights

        def to_interaction_groups(self) -> list[str]:
            raise RuntimeError("shape mismatch")

    failing_thrml = SimpleNamespace(
        __version__="0.0-test",
        __file__="/tmp/fake-thrml/__init__.py",
        CategoricalNode=FakeNode,
        Block=FakeBlock,
        models=SimpleNamespace(discrete_ebm=SimpleNamespace(CategoricalEBMFactor=FailingFactor)),
    )
    probe = mod.ThrmlProbe(
        import_available=True,
        version="0.0-test",
        import_path="/tmp/fake-thrml/__init__.py",
        import_error=None,
        module=failing_thrml,
        available_symbols={symbol: True for symbol in mod.REQUIRED_THRML_SYMBOLS},
        missing_symbols=[],
    )

    artifact = mod.build_artifact(tmp_path, thrml_probe=probe)

    assert artifact["local_api_smoke_passed"] is False
    assert artifact["api_gap_records"][0]["gap_id"] == "thrml_construction_failed"
    assert artifact["factor_graph_translation_records"][0]["thrml_mapping"]["error"] == (
        "RuntimeError: shape mismatch"
    )
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_099_records_missing_api_symbols(tmp_path: Path) -> None:
    """REQ-HW-099: missing construction symbols become actionable API gaps."""
    _write_common_sources(tmp_path)
    thrml_probe = mod.ThrmlProbe(
        import_available=True,
        version="0.0-test",
        import_path="/tmp/fake-thrml/__init__.py",
        import_error=None,
        module=SimpleNamespace(__version__="0.0-test", __file__="/tmp/fake-thrml/__init__.py"),
        available_symbols={"thrml.CategoricalNode": False},
        missing_symbols=["thrml.Block", "thrml.models.discrete_ebm.CategoricalEBMFactor"],
    )

    artifact = mod.build_artifact(tmp_path, thrml_probe=thrml_probe)

    assert artifact["thrml_factor_graph_api_boundary_v1_ready"] is False
    assert artifact["local_api_smoke_passed"] is False
    assert artifact["api_gap_records"][0]["gap_id"] == "thrml_factor_graph_symbols_missing"
    assert artifact["api_gap_records"][0]["missing_symbols"] == [
        "thrml.Block",
        "thrml.models.discrete_ebm.CategoricalEBMFactor",
    ]
    assert artifact["honest_verdict"].startswith("blocked_precondition:")


def test_req_hw_099_source_and_validation_fail_closed() -> None:
    """REQ-HW-099: source and schema errors fail closed."""
    source_verdict = mod.honest_verdict(
        {
            "source_errors": [{"path": "required.json"}],
            "selected_exact_rows": [{"row_id": "row"}],
            "thrml_import_available": True,
            "local_api_smoke_passed": True,
        }
    )
    no_rows_verdict = mod.honest_verdict(
        {
            "source_errors": [],
            "selected_exact_rows": [],
            "thrml_import_available": True,
            "local_api_smoke_passed": True,
        }
    )

    assert source_verdict.startswith("blocked_precondition: required exact-row")
    assert no_rows_verdict.startswith("blocked_precondition: no deterministic exact rows")
    try:
        mod.validate_artifact({})
    except ValueError as exc:
        assert "missing required Exp 3188 artifact fields" in str(exc)
    else:  # pragma: no cover - assertion guard.
        raise AssertionError("validate_artifact should reject missing fields")
    assert mod.required_source_errors(
        [
            {
                "path": "bad.json",
                "required": True,
                "present": True,
                "readable_json_object": False,
                "source_type": "json",
            }
        ]
    ) == [
        {
            "path": "bad.json",
            "error": "malformed_required_json_source",
            "source_type": "json",
        }
    ]
    assert mod.exception_text(ValueError()) == "ValueError"


def test_req_hw_099_writer_and_helpers_are_deterministic(tmp_path: Path) -> None:
    """REQ-HW-099: writer, JSON reader, and checksums are deterministic."""
    _write_common_sources(tmp_path)
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    scalar_json = tmp_path / "scalar.json"
    scalar_json.write_text("[]", encoding="utf-8")

    output = mod.write_artifact(
        tmp_path,
        thrml_probe=mod.ThrmlProbe(
            import_available=False,
            version=None,
            import_path=None,
            import_error="missing",
            module=None,
            available_symbols={},
            missing_symbols=list(mod.REQUIRED_THRML_SYMBOLS),
        ),
        started_s=1.0,
        now_s=1.25,
        tests_run=("pytest targeted",),
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert output == tmp_path / mod.OUTPUT_REL_PATH
    assert payload["duration_s"] == 0.25
    assert mod.read_json_object(bad_json) == {}
    assert mod.read_json_object(scalar_json) == {}
    assert mod.sha256_file(tmp_path / "missing.txt") is None
    assert mod.required_source_errors(
        [
            {
                "path": "required.json",
                "required": True,
                "present": False,
                "readable_json_object": False,
                "source_type": "json",
            }
        ]
    ) == [
        {
            "path": "required.json",
            "error": "missing_required_source",
            "source_type": "json",
        }
    ]
