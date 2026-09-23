"""CPU tests for REQ-ARC-WMTE-7530 B2 live measurement orchestration."""

from __future__ import annotations

import json
import urllib.request

import pytest

from carnot import experiment_7531_b2_induction_gate_measurement as exp7531
from carnot.agentic import arc_executable_world_model as awm


_VALID_BARE_CODE = (
    "import numpy as np\n"
    "def engine(grid, action, data):\n    return np.asarray(grid)\n"
    "def is_level_complete(grid):\n    return False\n"
)


class _FakeResponse:
    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *_args: object) -> bool:
        return False

    def read(self, *_args: object) -> bytes:
        return self._payload


def test_schedule_reuses_only_the_frozen_e6_panel() -> None:
    """REQ-ARC-WMTE-7530 keeps game selection frozen before outcomes."""

    frozen = json.loads((exp7531.REPO_ROOT / exp7531.FROZEN_PANEL_PATH).read_text())
    rows = exp7531.build_schedule(frozen)

    assert len(rows) == len(exp7531.PANEL_GAMES) * len(exp7531.EPISODE_SEEDS)
    assert tuple(dict.fromkeys(row["game"] for row in rows)) == exp7531.PANEL_GAMES
    assert {row["game"] for row in rows} == set(exp7531.PANEL_GAMES)
    assert all(row["adapter_disabled"] is True for row in rows)
    assert all(row["game_source_read"] is False for row in rows)
    assert {row["max_new_tokens_per_call"] for row in rows} == {4096}


def test_b2_induction_generation_receives_4096_without_changing_7471(monkeypatch) -> None:
    """REQ-ARC-WMTE-10009 keeps the budget and opts only this B2 child into codeonly."""

    observed: dict[str, object] = {}

    def mocked_generation_boundary(
        _args: object,
        *,
        induction_max_tokens: int,
        induction_codeonly: bool = False,
    ) -> int:
        observed["max_tokens"] = induction_max_tokens
        observed["codeonly"] = induction_codeonly
        return 0

    monkeypatch.setattr(exp7531.e6, "run_live_session", mocked_generation_boundary)

    assert exp7531.run_live_session(exp7531.argparse.Namespace()) == 0
    assert observed == {"max_tokens": 4096, "codeonly": True}
    assert exp7531.exp7471.MAX_NEW_TOKENS == 256


def test_e6_induction_proposer_constructor_receives_4096() -> None:
    """SCENARIO-ARC-WMTE-10008-INDUCTION-BUDGET reaches the generation constructor."""

    observed: dict[str, object] = {}

    class MockProposer:
        def __init__(self, **kwargs: object) -> None:
            observed.update(kwargs)

    proposer = exp7531.e6._construct_induction_proposer(
        exp7531.argparse.Namespace(
            model_path="/tmp/mock-model.gguf",
            model_revision="mock-revision",
            port=59999,
        ),
        max_tokens=4096,
        proposer_type=MockProposer,
    )

    assert isinstance(proposer, MockProposer)
    assert observed["max_tokens"] == 4096


def test_b2_codeonly_on_builds_raw_directive_fence_and_stop(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SCENARIO-ARC-WMTE-10009-CODEONLY-ON captures the mocked generation request."""

    captured: dict[str, object] = {}

    def fake_urlopen(
        request: urllib.request.Request, timeout: float | None = None
    ) -> _FakeResponse:
        del timeout
        captured["url"] = request.full_url
        captured["body"] = json.loads(bytes(request.data or b"").decode())
        response = {
            "content": _VALID_BARE_CODE,
            "stop_type": "stop",
            "truncated": False,
        }
        return _FakeResponse(json.dumps(response).encode())

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setenv("CARNOT_ARC_INDUCE_THINK", "1")
    monkeypatch.setenv("CARNOT_ARC_CODEONLY_INDUCE", "0")
    proposer = exp7531.e6._construct_induction_proposer(
        exp7531.argparse.Namespace(
            model_path="/tmp/mock-model.gguf",
            model_revision="mock-revision",
            port=59999,
        ),
        max_tokens=4096,
        proposer_type=awm.LocalGGUFProposer,
        induction_codeonly=True,
    )
    monkeypatch.setattr(proposer, "_ensure_server", lambda: True)

    assert proposer.use_chat_template is True
    ok, _code = proposer.generate(
        "BASE_PROMPT",
        ("engine", "is_level_complete"),
        tries=1,
        codeonly_eligible=True,
    )

    assert ok is True
    assert captured["url"] == "http://127.0.0.1:59999/completion"
    body = captured["body"]
    assert isinstance(body, dict)
    assert body["prompt"].startswith(awm._L2_CODEONLY_DIRECTIVE)
    assert body["prompt"].endswith("BASE_PROMPT\n```python\n")
    assert body["stop"] == ["```"]
    assert body["n_predict"] == 4096
    assert proposer.use_chat_template is True
    assert exp7531.os.environ["CARNOT_ARC_INDUCE_THINK"] == "1"
    assert exp7531.os.environ["CARNOT_ARC_CODEONLY_INDUCE"] == "0"


def test_b2_codeonly_off_preserves_prior_generation_call() -> None:
    """SCENARIO-ARC-WMTE-10009-CODEONLY-OFF is prior-round parity."""

    observed: dict[str, object] = {}

    class MockProposer:
        def __init__(self, **kwargs: object) -> None:
            observed["constructor"] = kwargs
            self.use_chat_template = kwargs["use_chat_template"]

        def generate(self, prompt: str, **kwargs: object) -> tuple[bool, str]:
            observed["prompt"] = prompt
            observed["generation"] = kwargs
            return True, "unchanged"

    proposer = exp7531.e6._construct_induction_proposer(
        exp7531.argparse.Namespace(
            model_path="/tmp/mock-model.gguf",
            model_revision="mock-revision",
            port=59999,
        ),
        max_tokens=4096,
        proposer_type=MockProposer,
    )
    result = proposer.generate("BASE_PROMPT", codeonly_eligible=True)

    assert result == (True, "unchanged")
    assert observed["prompt"] == "BASE_PROMPT"
    assert observed["generation"] == {"codeonly_eligible": True}
    assert observed["constructor"]["use_chat_template"] is True
    assert observed["constructor"]["max_tokens"] == 4096


def test_completion_token_distribution_exposes_nonuniform_lengths() -> None:
    """SCENARIO-ARC-WMTE-10008-TOKEN-DISTRIBUTION reports cap diagnostics."""

    rows = [
        {"completion_tokens": 4096},
        {"completion_tokens": 1024},
        {"completion_tokens": 3072},
        {"completion_tokens": 1024},
    ]

    distribution = exp7531.completion_token_distribution(rows)

    assert distribution == {
        "count": 4,
        "histogram": {"1024": 2, "3072": 1, "4096": 1},
        "minimum": 1024,
        "maximum": 4096,
        "mean": 2304.0,
        "median": 2048.0,
        "unique_count": 3,
        "uniform": False,
        "possible_remaining_hard_cap": False,
    }


def test_durable_completion_receipts_exclude_stale_attempt_usage(tmp_path) -> None:
    """REQ-ARC-WMTE-10008 publishes transport-backed completion lengths."""

    request_dir = tmp_path / "sp80__seed-1" / "requests"
    request_dir.mkdir(parents=True)
    (request_dir / "00_request.json").write_text(
        json.dumps(
            {
                "prompt": awm._L2_CODEONLY_DIRECTIVE + "PROMPT\n```python\n",
                "stop": ["```"],
            }
        )
    )
    (request_dir / "00_response.json").write_text(
        json.dumps({"usage": {"completion_tokens": 4096}})
    )
    attempts = [
        {
            "episode_id": "sp80:seed-1",
            "attempt_id": "sp80:seed-1:induction:1",
            "completion_tokens": 4096,
        },
        {
            "episode_id": "sp80:seed-1",
            "attempt_id": "sp80:seed-1:induction:2",
            "completion_tokens": 4096,
        },
    ]

    evidence = exp7531.durable_completion_token_evidence(attempts, tmp_path)

    assert evidence["distribution"]["histogram"] == {"4096": 1}
    assert evidence["per_attempt_reported_distribution"]["histogram"] == {"4096": 2}
    assert evidence["rows_without_distinct_completed_response"] == ["sp80:seed-1:induction:2"]


def test_raw_response_evidence_excludes_non_codeonly_refactor(tmp_path) -> None:
    """REQ-ARC-WMTE-10009 attributes raw outcomes only to codeonly induction calls."""

    request_dir = tmp_path / "sb26__seed-1" / "requests"
    request_dir.mkdir(parents=True)
    (request_dir / "00_request.json").write_text(
        json.dumps(
            {
                "prompt": awm._L2_CODEONLY_DIRECTIVE + "PROMPT\n```python\n",
                "stop": ["```"],
            }
        )
    )
    (request_dir / "00_response.json").write_text(
        json.dumps({"content": "def engine():\n    pass", "stop_type": "word"})
    )
    (request_dir / "01_request.json").write_text(
        json.dumps({"messages": [{"role": "user", "content": "refactor"}]})
    )
    (request_dir / "01_response.json").write_text(
        json.dumps(
            {
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": "", "reasoning_content": "hidden"},
                    }
                ]
            }
        )
    )

    evidence = exp7531.raw_response_evidence(tmp_path)

    assert evidence["response_count"] == 1
    assert evidence["content_nonempty_count"] == 1
    assert evidence["all_content_nonempty"] is True
    assert evidence["normalized_finish_reason_histogram"] == {"stop": 1}
    assert evidence["excluded_non_codeonly_responses"]["response_count"] == 1


def test_induction_model_spec_records_the_effective_budget() -> None:
    """REQ-ARC-WMTE-10008 does not retain E6's historical 256-token receipt."""

    original = {
        "decoding": {"max_new_tokens": 256, "retry_budget": 0},
        "runtime_settings": {"max_new_tokens_per_call": 256, "n_ctx": 49_152},
    }

    corrected = exp7531.induction_model_spec(original)

    assert corrected["decoding"]["max_new_tokens"] == 4096
    assert corrected["runtime_settings"]["max_new_tokens_per_call"] == 4096
    assert original["decoding"]["max_new_tokens"] == 256
    assert original["runtime_settings"]["max_new_tokens_per_call"] == 256


def test_positive_control_diagnostic_flags_a_saturated_progress_proxy() -> None:
    """REQ-ARC-WMTE-10008 preserves the known positive-control limitation."""

    attempts = [
        {
            "planned": False,
            "verifier_result": "not_observed",
            "progress_within_window": True,
        },
        {
            "planned": False,
            "verifier_result": "not_observed",
            "progress_within_window": True,
        },
    ]

    diagnostic = exp7531.positive_control_diagnostic(attempts)

    assert diagnostic["planned_true_count"] == 0
    assert diagnostic["verifier_observed_count"] == 0
    assert diagnostic["progress_within_window_true_count"] == 2
    assert diagnostic["progress_signal_saturated"] is True


def test_static_precondition_requires_gpu_one_already_selected(monkeypatch) -> None:
    """REQ-ARC-WMTE-7530 checks CUDA visibility before runtime initialization."""

    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(exp7531, "STAGE1_PATH", exp7531.FROZEN_PANEL_PATH)
    checks, _cited, _environment_dir = exp7531.static_preconditions()

    check = next(row for row in checks if row["check"] == "cuda_visible_devices_already_gpu_1")
    assert check["passed"] is False
    assert check["observed"] is None
    assert check["path"] == "process_environment"


def test_blocked_artifact_keeps_every_scheduled_unit() -> None:
    """SCENARIO-ARC-WMTE-7530-FEASIBILITY never drops blocked units."""

    schedule = [
        {"episode_id": "sb26:seed-1", "game": "sb26", "seed": 1},
        {"episode_id": "vc33:seed-1", "game": "vc33", "seed": 1},
    ]
    failed = {
        "check": "physical_gpu_1_idle_before_runtime_preflight",
        "passed": False,
        "observed": {"used_memory_mb": 700},
    }
    artifact = exp7531.blocked_artifact(
        failed_check=failed,
        checks=[failed],
        cited=[],
        schedule=schedule,
        duration_s=0.5,
    )

    assert artifact["honest_verdict"].startswith("blocked_")
    assert artifact["inference_substrate"] == "no_model_load"
    assert artifact["gate_opportunity_count"] == 0
    assert artifact["induction_attempt_count"] == 0
    assert artifact["sample_floor"]["met"] is False
    assert [row["disposition"] for row in artifact["episode_rows"]] == [
        "unstarted",
        "unstarted",
    ]
    assert artifact["gate_ready_to_ship"] is False
