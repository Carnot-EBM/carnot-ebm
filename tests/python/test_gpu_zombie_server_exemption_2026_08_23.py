"""GPU zombie sweeps must spare inference servers and gate utilization per-GPU.

Spec refs: REQ-INFRA-079, SCENARIO-INFRA-6560, SCENARIO-INFRA-6561,
SCENARIO-INFRA-6562 (openspec/capabilities/pipeline/spec.md).

Origin (2026-08-23): the standing unsolved "llama-server reaper"
(ops/known-issues.md, five entries dated 2026-08-09) was
ExperimentTemplate.kill_gpu_zombies()'s nvidia-smi fallback. It took the
MINIMUM utilization across ALL GPUs as its idle gate, so an idle GPU 0
dragged the gate to 0% and the sweep SIGTERMed a llama-server on GPU 1 that
was decoding at 34 tok/s (live-reproduced the same day at 81 tok/s, GPU 1 at
97% utilization, sweep logging gpu_util=0.0%). Every test here encodes one
input the old code killed and the fixed code must spare.
"""

from __future__ import annotations

import signal
import sys
from unittest.mock import MagicMock, patch

# Match the import style of the sibling template tests (test_exclusion_manifest.py).
sys.path.insert(0, "scripts")


def _fake_smi(vram_output: str, util_output: str):
    """A subprocess.run stand-in serving canned nvidia-smi CSV for both queries."""

    def fake_subprocess_run(cmd, **kwargs):
        result = MagicMock()
        result.returncode = 0
        if any("--query-compute-apps" in c for c in cmd):
            result.stdout = vram_output
        else:
            result.stdout = util_output
        return result

    return fake_subprocess_run


class TestPerGpuUtilizationGate:
    """SCENARIO-INFRA-6560: the exact supab5 incident shape."""

    def test_busy_gpu1_process_survives_while_gpu0_idles(self):
        """A process on a 97%-busy GPU is spared even when another GPU reads 0%.

        The old code computed min(0, 97) = 0 < 5 and killed it. The fixed code
        joins the process to GPU-bbb (97%) and leaves it alone.
        """
        from scripts.experiment_template import ExperimentTemplate

        vram = "43210, 8782, GPU-bbb\n"  # the server, on GPU 1
        util = "GPU-aaa, 0\nGPU-bbb, 97\n"  # GPU 0 idle, GPU 1 busy

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    result = ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000, util_threshold_pct=5.0
                    )
                    mock_kill.assert_not_called()
        assert result["killed_pids"] == []

    def test_idle_own_gpu_python_process_is_still_killed(self):
        """The sweep keeps its purpose: a non-server process on ITS OWN idle GPU dies.

        Guards against 'fixing' the reaper by neutering it entirely.
        """
        from scripts.experiment_template import ExperimentTemplate

        vram = "43211, 5000, GPU-aaa\n"
        util = "GPU-aaa, 0\nGPU-bbb, 97\n"

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch(
                    "scripts.experiment_template._pid_cmdline",
                    return_value="python3 stale_pytest_worker.py",
                ):
                    with patch("scripts.experiment_template.os.kill") as mock_kill:
                        result = ExperimentTemplate.kill_gpu_zombies(
                            vram_threshold_mb=1000, util_threshold_pct=5.0
                        )
                        mock_kill.assert_called_once_with(43211, signal.SIGTERM)
        assert result["killed_pids"] == [43211]


class TestServerExemptionTemplateFallback:
    """SCENARIO-INFRA-6561, template nvidia-smi fallback."""

    def test_idle_llama_server_is_spared_by_name(self):
        """A llama-server on an idle GPU is skipped: idling between requests is
        a server's normal healthy state, not a zombie signature."""
        from scripts.experiment_template import ExperimentTemplate

        vram = "43212, 12000, GPU-aaa\n"
        util = "GPU-aaa, 0\n"

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch(
                    "scripts.experiment_template._pid_cmdline",
                    return_value="/home/u/.cache/llama.cpp/bin/llama-server -m model.gguf --port 8993",
                ):
                    with patch("scripts.experiment_template.os.kill") as mock_kill:
                        result = ExperimentTemplate.kill_gpu_zombies(
                            vram_threshold_mb=1000, util_threshold_pct=5.0
                        )
                        mock_kill.assert_not_called()
        assert result["killed_pids"] == []

    def test_idle_vllm_server_is_spared_by_name(self):
        from scripts.experiment_template import ExperimentTemplate

        vram = "43213, 20000, GPU-aaa\n"
        util = "GPU-aaa, 0\n"

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch(
                    "scripts.experiment_template._pid_cmdline",
                    return_value="python -m vllm.entrypoints.openai.api_server --model m",
                ):
                    with patch("scripts.experiment_template.os.kill") as mock_kill:
                        ExperimentTemplate.kill_gpu_zombies(
                            vram_threshold_mb=1000, util_threshold_pct=5.0
                        )
                        mock_kill.assert_not_called()


class TestMissingAttributionFailsClosed:
    """SCENARIO-INFRA-6562: no per-GPU attribution -> skip, never aggregate."""

    def test_two_column_output_is_skipped_not_killed(self):
        """Old-format output (pid, used_memory, no gpu_uuid) must not be killed
        under any aggregate gate — the exact old behaviour this fix removes."""
        from scripts.experiment_template import ExperimentTemplate

        vram = "43214, 5000\n"  # no gpu_uuid column
        util = "GPU-aaa, 0\n"

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    result = ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000, util_threshold_pct=5.0
                    )
                    mock_kill.assert_not_called()
        assert result["killed_pids"] == []

    def test_unknown_uuid_is_skipped_not_killed(self):
        from scripts.experiment_template import ExperimentTemplate

        vram = "43215, 5000, GPU-zzz\n"  # uuid absent from the util table
        util = "GPU-aaa, 0\n"

        with patch.dict(sys.modules, {"pynvml": None}):
            with patch(
                "scripts.experiment_template.subprocess.run",
                side_effect=_fake_smi(vram, util),
            ):
                with patch("scripts.experiment_template.os.kill") as mock_kill:
                    ExperimentTemplate.kill_gpu_zombies(
                        vram_threshold_mb=1000, util_threshold_pct=5.0
                    )
                    mock_kill.assert_not_called()


class TestServerMarkerHelper:
    """The /proc-reading helper itself, against a REAL child process."""

    def test_real_child_with_server_marker_in_cmdline_is_protected(self):
        """Spawn a real process whose cmdline carries the llama-server marker
        and confirm the helper reads it through /proc."""
        import subprocess

        from scripts.experiment_template import _pid_is_protected_server_proc

        # argv[3] is an inert bash positional parameter; it only has to appear
        # in /proc/<pid>/cmdline. The compound command stops bash's single-
        # command exec optimization from replacing the argv we assert on.
        child = subprocess.Popen(["bash", "-c", "sleep 20; exit 0", "llama-server"])
        try:
            assert _pid_is_protected_server_proc(child.pid) is True
        finally:
            child.terminate()
            child.wait(timeout=10)

    def test_dead_pid_is_not_protected(self):
        from scripts.experiment_template import _pid_is_protected_server_proc

        # PID 2 is kthreadd (unreadable cmdline) on Linux; a huge PID is absent.
        assert _pid_is_protected_server_proc(2**22 + 12345) is False


class TestPipelineZombieKillerExemption:
    """SCENARIO-INFRA-6561, python/carnot/pipeline/gpu_zombie_killer.py."""

    def test_server_pid_never_receives_sigkill(self):
        from carnot.pipeline import gpu_zombie_killer as gzk

        with patch.object(gzk, "_query_nvidia_smi", return_value="1000"):
            with patch.object(gzk, "get_gpu_memory_pids", return_value=[43216, 43217]):
                with patch.object(gzk, "_get_vram_used_mb", return_value=0.0):
                    with patch.object(
                        gzk,
                        "_pid_is_protected_server",
                        side_effect=lambda pid: pid == 43216,
                    ):
                        with patch.object(gzk.os, "kill") as mock_kill:
                            with patch.object(gzk.time, "sleep"):
                                result = gzk.kill_gpu_zombies(gpu_index=1)
        killed = [c.args[0] for c in mock_kill.call_args_list]
        assert 43216 not in killed
        assert 43217 in killed
        assert result.pids_skipped_protected == [43216]
        assert result.pids_killed == [43217]


class TestExpandedReaperExemption:
    """SCENARIO-INFRA-6561, python/carnot/pipeline/expanded_gpu_reaper.py."""

    def test_out_of_tree_old_server_is_skipped_with_reason(self):
        from carnot.pipeline import expanded_gpu_reaper as egr

        reaper = egr.ExpandedGPUReaper(
            egr.ExpandedGPUReaperConfig(min_vram_mb=1024, min_age_s=1800, dry_run=False)
        )
        fake_procs = [
            {"pid": 43218, "used_memory_mb": 20000, "process_name": "llama-server"}
        ]
        with patch.object(egr.shutil, "which", return_value="/usr/bin/nvidia-smi"):
            with patch.object(reaper, "_list_gpu_processes", return_value=fake_procs):
                with patch.object(reaper, "_in_our_subtree", return_value=False):
                    with patch.object(reaper, "_process_age_s", return_value=7200):
                        with patch.object(
                            egr, "_pid_is_protected_server", return_value=True
                        ):
                            with patch.object(egr.os, "kill") as mock_kill:
                                result = reaper.reap()
        mock_kill.assert_not_called()
        assert result.killed == []
        assert [s["reason"] for s in result.skipped] == ["protected_server"]
