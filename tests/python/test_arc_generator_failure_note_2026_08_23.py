"""Generator failure notes name the effective model and carry death evidence.

Spec refs: REQ-ARC-WMTE-6670, SCENARIO-ARC-WMTE-6670-1..3
(openspec/capabilities/arc-world-model-trust-energy/spec.md).

Origin (2026-08-23): every supab5 A/B row failed with "GPU llama-server failed
for Qwen3.5-9B-MTP" while the server had loaded and served Qwen3.8-27B (the
harness keeps repo_substr at its frozen 9B pin by design; model_path wins at
load). The mislabel misdirected the investigation; the true cause — an external
SIGTERM, visible in the server's own stderr tail — never reached the row note.
"""

from __future__ import annotations

from unittest.mock import patch


def _proposer(**kwargs):
    from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

    return LocalGGUFProposer(**kwargs)


class TestEffectiveModelLabel:
    """SCENARIO-ARC-WMTE-6670-1."""

    def test_failure_message_names_model_path_over_frozen_pin(self):
        """The supab5 shape: frozen 9B repo pin + 27B model_path override."""
        prop = _proposer(
            repo_substr="Qwen3.5-9B-MTP",
            model_path="/cache/hub/snap/Qwen3.8-27B-Q4_K_M.gguf",
        )
        with patch.object(type(prop), "_ensure_server", lambda self: False):
            ok, msg = prop.generate("prompt", ("engine",), tries=1)
        assert ok is False
        assert msg.startswith("GPU llama-server failed")  # prefix contract unchanged
        assert "Qwen3.8-27B-Q4_K_M.gguf" in msg
        assert "Qwen3.5-9B-MTP" not in msg

    def test_failure_message_falls_back_to_repo_substr(self):
        prop = _proposer(repo_substr="Qwen3.8-27B")
        with patch.object(type(prop), "_ensure_server", lambda self: False):
            ok, msg = prop.generate("prompt", ("engine",), tries=1)
        assert ok is False
        assert "Qwen3.8-27B" in msg

    def test_complete_text_uses_the_same_label(self):
        prop = _proposer(
            repo_substr="Qwen3.5-9B-MTP",
            model_path="/cache/hub/snap/Qwen3.8-27B-Q4_K_M.gguf",
        )
        with patch.object(type(prop), "_ensure_server", lambda self: False):
            ok, msg = prop.complete_text("prompt")
        assert ok is False
        assert "Qwen3.8-27B-Q4_K_M.gguf" in msg
        assert "Qwen3.5-9B-MTP" not in msg


class TestDeathSignatureHint:
    """SCENARIO-ARC-WMTE-6670-2 and -3."""

    def test_external_kill_tail_reaches_the_note(self, tmp_path):
        log = tmp_path / "llama_server_p8993_1.log"
        log.write_text(
            "62.39.877.233 I slot print_timing: id 3 | n_decoded = 43664\n"
            "62.41.269.627 I srv operator(): operator(): cleaning up before exit...\n"
            "Received second interrupt, terminating immediately.\n"
        )
        prop = _proposer(repo_substr="Qwen3.8-27B")
        prop._stderr_log_path = log
        prop._note_server_failure("GPU llama-server failed for X")
        assert len(prop.server_failure_diagnostics) == 1
        note = prop.server_failure_diagnostics[0]
        assert "external termination signal" in note
        assert str(log) in note

    def test_clean_tail_adds_no_hint(self, tmp_path):
        log = tmp_path / "llama_server_p8993_2.log"
        log.write_text("I slot print_timing: id 3 | n_decoded = 100\n")
        prop = _proposer(repo_substr="Qwen3.8-27B")
        prop._stderr_log_path = log
        prop._note_server_failure("timeout")
        assert prop.server_failure_diagnostics == ["timeout"]

    def test_no_log_and_unreadable_log_never_raise(self, tmp_path):
        prop = _proposer(repo_substr="Qwen3.8-27B")
        # attribute never set (server never launched by this instance)
        prop._note_server_failure("a")
        # explicitly unset
        prop._stderr_log_path = None
        prop._note_server_failure("b")
        # points at a path that does not exist
        prop._stderr_log_path = tmp_path / "absent.log"
        prop._note_server_failure("c")
        assert prop.server_failure_diagnostics == ["a", "b", "c"]
        assert prop.n_server_failures == 3
