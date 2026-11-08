"""REQ-REPORT-8918: shortest-distance labels and bounded pilot evidence."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from carnot.reporting import bfs_error_pilot_8918 as p


def test_bfs_hand_and_matrix():
    """SCENARIO-REPORT-8918-CONTRACT: a two-edge walk need not have distance two."""
    edges = [["AA", "BB"], ["BB", "CC"], ["AA", "CC"], ["CC", "DD"]]
    assert p.distance_two(["AA", "BB", "CC", "DD", "EE"], edges, "AA") == ["DD"]
    import numpy as np

    for seed in range(10):
        rng = np.random.default_rng(seed)
        matrix = np.triu((rng.random((9, 9)) < 0.3).astype(int), 1)
        matrix += matrix.T
        nodes = [f"A{chr(65 + i)}" for i in range(9)]
        pairs = [[nodes[i], nodes[j]] for i in range(9) for j in range(i + 1, 9) if matrix[i, j]]
        for i, node in enumerate(nodes):
            expected = [
                nodes[j]
                for j in range(9)
                if j != i and matrix[i, j] == 0 and (matrix @ matrix)[i, j] > 0
            ]
            assert p.distance_two(nodes, pairs, node) == expected


def test_frozen_graphs():
    """REQ-REPORT-8918: fifty independent graph seeds seal truth before generation."""
    graphs = p.graph_manifest()
    assert graphs == p.graph_manifest()
    assert len(graphs) == len({g["seed"] for g in graphs}) == 50
    for g in graphs:
        assert 50 <= len(g["nodes"]) <= 60
        assert len(g["nodes"]) == len(set(g["nodes"]))
        assert g["expected"] and g["expected"] == p.distance_two(g["nodes"], g["edges"], g["start"])
        assert g["edge_probability"] == 3 / (len(g["nodes"]) - 1)
        assert g["prompt_sha256"] == p.canonical_hash(g["prompt"])
        assert len({tuple(e) for e in g["edges"]}) == len(g["edges"])
        assert all(a != b for a, b in g["edges"])


@pytest.mark.parametrize(
    "answer,expected",
    [
        ("AA", "AA"),
        (" AA.\n", "AA"),
        ("AA!", "AA"),
        ("", None),
        ("AA BB", None),
        ("node AA", None),
        ("aa", None),
        ("ZZ", None),
    ],
)
def test_parser(answer, expected):
    """REQ-REPORT-8918: the frozen parser cannot repair or select ambiguous answers."""
    assert p.parse_answer(answer, ["AA", "BB"])[0] == expected


def completed(graphs, errors=10):
    """REQ-REPORT-8918: scripted labels remain separate from model activity."""
    return [
        dict(
            unit_id=g["unit_id"],
            status="completed",
            answer=g["start"] if i < errors else g["expected"][0],
            duration_s=0.1,
            usage=dict(prompt_tokens=10, completion_tokens=1),
        )
        for i, g in enumerate(graphs)
    ]


def test_reduction_gates_and_censoring():
    """REQ-REPORT-8918: runtime, completion and error-source gates are independent."""
    graphs = p.graph_manifest()
    rows = completed(graphs)
    result = p.reduce_answers(graphs, rows)
    assert result["n_errors"] == 10 and result["error_rate"] == 0.2
    assert result["completed_count"] == result["independent_count"] == 50
    assert result["error_interval"]["low"] < 0.2 < result["error_interval"]["high"]
    assert p.gates(result, live=True, owned=True) == (
        1,
        1,
        1,
        "circular_positive",
        "complete_pilot_error_source_gate_passed",
    )
    below = p.reduce_answers(graphs, completed(graphs, 9))
    assert p.gates(below, live=True, owned=True)[3:] == (
        "null",
        "complete_pilot_error_rate_below_gate",
    )
    rows[0] = dict(unit_id=graphs[0]["unit_id"], status="runtime_failed", missing_reason="deadline")
    censored = p.reduce_answers(graphs, rows[:-1])
    assert censored["n_errors"] == 9 and censored["failed_count"] == 1
    assert censored["censored_count"] == 2 and censored["error_interval"] is None
    assert p.gates(censored, live=True, owned=True)[:3] == (0, 0, 1)
    assert p.gates(result, live=True, owned=False)[3] == "disqualified"
    assert p.gates(result, live=False, owned=True)[3] == "blocked"
    for bad in [graphs[:-1], graphs[:-1] + [graphs[0]]]:
        with pytest.raises(ValueError):
            p.reduce_answers(bad, [])
    with pytest.raises(ValueError):
        p.reduce_answers(graphs, rows + [rows[0]])


def test_actual_private_cli_and_cold_replay(tmp_path):
    """SCENARIO-REPORT-8918-CONTRACT: real CLI replay rejects rehashed answer drift."""
    cli = p.ROOT / p.CLI
    target = tmp_path / "experiment_8918_v763_bfs_error_pilot.json"
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)

    def invoke(*args):
        return subprocess.run(
            [sys.executable, "-u", str(cli), *args],
            cwd=tmp_path,
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )

    result = invoke("--date", "20261108", "--fixture-e2e", str(target))
    assert result.returncode == 0, result.stdout + result.stderr
    value = json.loads(target.read_text())
    assert value["qwen_runtime_ready_score"] == value["pilot_ready_score"] == 0
    assert value["model_invocation_counts"]["generation_calls_attempted"] == 0
    assert invoke("--cold-replay", str(target)).returncode == 0
    value["n_errors"] = 1
    value["reproducibility_checksum"] = p.checksum(value)
    target.write_text(json.dumps(value))
    assert invoke("--cold-replay", str(target)).returncode == 1
    assert invoke("--date", "20261107", "--fixture-e2e", str(target)).returncode != 0
    historical = json.dumps(dict(experiment_id=123, task_id="exp123-history"))
    target.write_text(historical)
    assert invoke("--date", "20261108", "--fixture-e2e", str(target)).returncode != 0
    assert target.read_text() == historical


def test_runtime_external_and_owned_controls(tmp_path, monkeypatch):
    """REQ-REPORT-8918: absent resources block; observed offload is mandatory."""
    from carnot.inference import bfs_pilot_runtime_8918 as runtime

    monkeypatch.setattr(
        runtime, "preflight", lambda raw: (_ for _ in ()).throw(RuntimeError("cache_missing"))
    )
    value = runtime.run(p.graph_manifest(), tmp_path)
    assert value["blocker"] == "cache_missing"
    assert value["checks"][-1]["observed"] == "cache_missing"
    assert value["counts"]["generation_calls_attempted"] == 0
    assert not value["live"]


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """REQ-REPORT-8918: mocked devices exercise controls without claiming live work."""
    from types import SimpleNamespace
    from carnot.inference import bfs_pilot_runtime_8918 as r

    model = tmp_path / "Qwen3.8-27B-Q4_K_M.gguf"
    model.write_bytes(b"test-only weights")
    gpu = dict(index=0, uuid="test-device", used_mb=0, free_mb=24000)
    resources = dict(
        model=dict(hf_id=p.SETTINGS["model"], model_path=str(model), sha256=p.sha256_file(model)),
        gpu=gpu,
        gpu_offload_supported=True,
    )
    control = dict(
        load_error=False, call_error=False, malformed=False, bad_offload=False, busy=False
    )

    class Lease:
        def __init__(self):
            self.document = dict(phase="preflight")

        @classmethod
        def acquire(cls, **kwargs):
            return cls()

        def owner_receipt(self):
            return dict(test_only=True)

        def transition(self, phase, **kwargs):
            self.document["phase"] = phase

        def heartbeat(self):
            pass

        def release(self):
            return dict(released=True)

    class Runtime:
        def __init__(self, *args):
            self.worker = SimpleNamespace(post_json=self.call)

        def load(self):
            if control["load_error"]:
                raise RuntimeError("load_failed")
            return dict(
                authenticated=True,
                gguf_sha256=resources["model"]["sha256"],
                offload_layers=[0, 0] if control["bad_offload"] else [65, 65],
                props=dict(chat_template="test-template"),
            )

        def call(self, endpoint, payload, timeout):
            assert timeout == 30 and payload["max_tokens"] == 16
            assert payload["chat_template_kwargs"] == dict(enable_thinking=False)
            assert "grammar" not in payload
            if control["call_error"]:
                raise TimeoutError("call_deadline")
            return dict(
                choices=[dict(message=dict(content=123 if control["malformed"] else "AA"))],
                usage=dict(prompt_tokens=10, completion_tokens=1),
            )

        def close(self):
            return dict(leak_free=True)

    monkeypatch.setattr(r, "preflight", lambda raw: resources)
    monkeypatch.setattr(r, "gpu_state", lambda: dict(gpu, used_mb=2000 if control["busy"] else 0))
    monkeypatch.setattr(r, "GpuLease", Lease)
    monkeypatch.setattr(r, "QwenRuntime", Runtime)
    monkeypatch.setattr(r, "bounded", lambda call, *a, **kw: call())
    return r, control, resources


@pytest.mark.parametrize(
    "failure", [None, "busy", "load_error", "call_error", "malformed", "bad_offload"]
)
def test_runtime_retains_failures(fake_runtime, tmp_path, failure):
    """REQ-REPORT-8918: each gate retains calls, tokens, elapsed work and censored slots."""
    r, control, _ = fake_runtime
    if failure:
        control[failure] = True
    result = r.run(p.graph_manifest(), tmp_path / "raw")
    reduced = p.reduce_answers(p.graph_manifest(), result["responses"])
    if failure is None:
        assert result["counts"]["generation_calls_completed"] == 50
        assert reduced["completed_count"] == 50
        assert result["settings"]["chat_template"] == "test-template"
    else:
        assert result["blocker"]
        assert reduced["censored_count"] == 50
        assert result["counts"]["generation_calls_completed"] == 0
    assert result["resource_lease"]["release"]["released"]


@pytest.mark.parametrize("mode", ["ok", "missing", "quant", "offload"])
def test_preflight_authenticates_cache(fake_runtime, tmp_path, monkeypatch, mode):
    """REQ-REPORT-8918: cache fallback and absent GPU support cannot qualify runtime."""
    from types import SimpleNamespace

    r, _, resources = fake_runtime
    original = __import__("importlib").import_module(r.__name__)
    spec = dict(resources["model"])
    if mode == "quant":
        spec["hf_id"] = "other"
    monkeypatch.setattr(r, "cached_current_model", lambda **kw: None if mode == "missing" else spec)
    module_path = tmp_path / "llama_cpp.py"
    module_path.write_text("test fixture")
    monkeypatch.setitem(
        sys.modules,
        "llama_cpp",
        SimpleNamespace(
            __file__=str(module_path), llama_supports_gpu_offload=lambda: mode != "offload"
        ),
    )
    # Restore the real preflight that the general runtime fixture replaces.
    source = p.ROOT / "python/carnot/inference/bfs_pilot_runtime_8918.py"
    import types

    real = types.ModuleType("preflight_test")
    exec(compile(source.read_text(), str(source), "exec"), real.__dict__)
    monkeypatch.setattr(real, "cached_current_model", r.cached_current_model)
    monkeypatch.setattr(real, "gpu_state", r.gpu_state)
    if mode == "ok":
        assert real.preflight(tmp_path)["gpu_offload_supported"]
    else:
        with pytest.raises(RuntimeError):
            real.preflight(tmp_path)
    assert original is r


def test_gpu_query_is_device_zero(monkeypatch):
    """REQ-REPORT-8918: no query may enumerate detached device one."""
    from types import SimpleNamespace
    from carnot.inference import bfs_pilot_runtime_8918 as r

    def query(argv, **kwargs):
        assert "--id=0" in argv and kwargs["timeout"] == 10
        return SimpleNamespace(stdout="0, test-device, 4, 24000\n")

    monkeypatch.setattr(subprocess, "run", query)
    assert r.gpu_state()["index"] == 0


def test_response_status_and_source_hashes(tmp_path):
    """REQ-REPORT-8918: invalid states and changed retained bytes fail closed."""
    graphs = p.graph_manifest()
    with pytest.raises(ValueError, match="response_status"):
        p.reduce_answers(graphs, [dict(unit_id=graphs[0]["unit_id"], status="invented")])
    file = tmp_path / "bytes"
    file.write_text("original")
    ref = p.reference(file)
    assert ref["sha256"] == p.sha256_file(file)


@pytest.mark.parametrize("qualified", [True, False])
@pytest.mark.parametrize("terminal_failure", [True, False])
def test_main_owned_publication(tmp_path, monkeypatch, qualified, terminal_failure):
    """SCENARIO-REPORT-8918-CONTRACT: owned checks precede compute and publication."""
    from carnot.inference import bfs_pilot_runtime_8918 as r
    from carnot.reporting.current_work_receipt import ZERO_INVOCATION_COUNTS

    original = p.ROOT
    root = tmp_path / "checkout"
    root.mkdir()
    for name in [
        "python",
        "scripts",
        "openspec",
        "docs",
        "ops",
        "tests",
        "AGENTS.md",
        "CODEX.md",
        "CLAUDE.md",
        "research-program.md",
    ]:
        (root / name).symlink_to(original / name)
    monkeypatch.setattr(p, "ROOT", root)
    private = tmp_path / "checks"
    private.mkdir()
    monkeypatch.setattr(p.tempfile, "mkdtemp", lambda **kw: str(private))

    def execute(repo, commands, **kwargs):
        if commands[0].name == "owned_unit_child" and qualified:
            (private / "coverage.json").write_text(
                json.dumps(
                    dict(
                        totals=dict(
                            percent_covered=100,
                            excluded_lines=0,
                            num_statements=100,
                            covered_lines=100,
                            missing_lines=0,
                        )
                    )
                )
            )
        log = tmp_path / "terminal.log"
        log.write_text("test-only receipt")
        return [
            dict(
                name=c.name,
                passed=(
                    qualified
                    if c.scope == "owned"
                    else not terminal_failure
                    if c.scope == "terminal"
                    else True
                ),
                exit_code=2 if terminal_failure else 0,
                log_path=str(log),
            )
            for c in commands
        ]

    monkeypatch.setattr(p, "run_commands", execute)

    def measured(graphs, raw):
        counts = dict(ZERO_INVOCATION_COUNTS)
        counts.update(
            model_loads_attempted=1,
            model_loads_completed=1,
            generation_calls_attempted=50,
            generation_calls_completed=50,
        )
        result = dict(
            responses=completed(graphs),
            counts=counts,
            live=True,
            checks=[],
            model_receipt=dict(
                authenticated=True,
                gguf_sha256="sha256:test",
                preflight_gguf_sha256="sha256:test",
                offload_layers=[65, 65],
            ),
            settings=dict(p.SETTINGS),
            resource_lease={},
            phase_spans=[],
        )
        p.atomic_json(raw / "runtime.json", result)
        return result

    monkeypatch.setattr(r, "run", measured)
    assert p.main(["--date", "20261108"]) == 0
    output = root / "results" / (p.NAME + ".json")
    value = json.loads(output.read_text())
    assert value["required_checks_passed"] is (qualified and not terminal_failure)
    assert value["verdict_class"] == (
        "circular_positive" if qualified and not terminal_failure else "disqualified"
    )
    assert p.replay(output)["passed"]
    if qualified and not terminal_failure:
        for key, replacement, message in [
            ("pilot_ready_score", 0, "gate_drift"),
            ("required_checks_passed", False, "qualification_drift"),
            ("generator_settings", {}, "runtime_binding_drift"),
            ("graph_manifest", [], "graph_truth_drift"),
        ]:
            changed = deepcopy(value)
            changed[key] = replacement
            changed["reproducibility_checksum"] = p.checksum(changed)
            output.write_text(json.dumps(changed))
            with pytest.raises(ValueError, match=message):
                p.replay(output)
        output.write_text(json.dumps(value))
        assert p.main(["--cold-replay", str(output)]) == 0
        value["reproducibility_checksum"] = "wrong"
        output.write_text(json.dumps(value))
        assert p.main(["--cold-replay", str(output)]) == 1
        value["reproducibility_checksum"] = p.checksum(value)
        output.write_text(json.dumps(value))
        Path(value["source_artifact_hashes"][0]["path"]).write_text("changed")
        with pytest.raises(ValueError, match="operand_hash_drift"):
            p.replay(output)


def test_graph_resampling_is_recorded(monkeypatch):
    """REQ-REPORT-8918: a graph without any exact-distance-two start is resampled."""
    original = p.distance_two
    remaining = len(p.graph_manifest()[0]["nodes"])

    def no_initial_start(nodes, edges, start):
        nonlocal remaining
        if remaining:
            remaining -= 1
            return []
        return original(nodes, edges, start)

    monkeypatch.setattr(p, "distance_two", no_initial_start)
    assert p.graph_manifest()[0]["resampling_count"] == 1


def test_pilot_budget_keeps_unstarted_slots(fake_runtime, tmp_path, monkeypatch):
    """REQ-REPORT-8918: the total generation budget cannot create replacement slots."""
    from types import SimpleNamespace

    r, _, _ = fake_runtime
    ticks = iter([0, 0, 0, 0, 0, 1600])
    monkeypatch.setattr(r, "time", SimpleNamespace(monotonic=lambda: next(ticks)))
    result = r.run(p.graph_manifest(), tmp_path / "budget")
    assert result["blocker"] == "pilot_budget"
    assert result["counts"]["generation_calls_attempted"] == 0
    assert p.reduce_answers(p.graph_manifest(), result["responses"])["censored_count"] == 50


@pytest.mark.parametrize("severity,expected", [(0, True), (1, False), (2, False)])
def test_terminal_info_is_retained(tmp_path, severity, expected):
    """REQ-REPORT-8918: zero-gate information is retained; warnings fail acceptance."""
    log = tmp_path / "auditor.json"
    log.write_text(json.dumps(dict(reports=[dict(loaded=True, max_severity=severity)])))
    receipts = [
        dict(name="strict_rows", passed=True),
        dict(name="adversarial", passed=False, exit_code=1, log_path=str(log)),
    ]
    assert p.terminal_passed(receipts) is expected
    assert not p.terminal_passed([dict(name="strict_rows", passed=False)])
