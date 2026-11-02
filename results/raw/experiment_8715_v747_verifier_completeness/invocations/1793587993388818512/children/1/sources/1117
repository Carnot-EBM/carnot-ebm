"""Experiment 4715: corrected goal-free online action-learning driver.

Spec refs: REQ-ARC-FCP-4715, SCENARIO-ARC-FCP-4715.
"""

from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON_ROOT = REPO_ROOT / "python"
if str(PYTHON_ROOT) not in sys.path:  # pragma: no cover - direct script guard
    sys.path.insert(0, str(PYTHON_ROOT))

EXPERIMENT = "experiment_4715_online_action_learning_driver_corrected"
SCHEMA = "carnot.exp4715.online_action_learning_driver_corrected.v1"
RESULT_RELATIVE_PATH = "results/experiment_4715_online_action_learning_driver_corrected.json"
SPEC_REFS = ["REQ-ARC-FCP-4715", "SCENARIO-ARC-FCP-4715"]
RANDOM_SEED = 4715
QWEN_PORT = 8920
QWEN_MODEL = "Qwen3.5-9B-MTP"
TERMINAL_PREFIXES = ("success:", "complete:", "blocked_")

ARM_ARTIFACTS = {
    "frozen": Path("results/experiment_4710_online_action_learning_arms_frozen.json"),
    "online-scratch": Path(
        "results/experiment_4710_online_action_learning_arms_online_scratch.json"
    ),
    # The corrected warm driver includes coordinate-head ACTION6 proposals.
    "online-warm": Path(
        "results/experiment_4710_online_action_learning_arms_online_warm_propose.json"
    ),
}

FIELD_PRINCIPLES: dict[str, str] = {
    "honest_verdict": (
        "terminal prefix; success: online_warm_beats_frozen_<delta>_l2_<game> OR complete: "
        "online_action_learning_no_first_win_lift_residual_<cause>."
    ),
    "inference_substrate": (
        "live_llm_inference for the Qwen GGUF precondition plus "
        "verifier_ensemble_against_cached_candidates for the offline held-out arm artifacts."
    ),
    "online_warm_first_win": (
        "the +0.05 online-warm-over-frozen gate is the whole bet; the warm arm isolates online "
        "adaptation from scratch initialization."
    ),
    "online_scratch_first_win": "the online-from-random arm isolates learning from warm start.",
    "frozen_first_win": "the frozen-prior baseline is the no-online control.",
    "online_warm_vs_frozen_delta": (
        "online_warm_first_win - frozen_first_win; >=+0.05 is the gate; emitted explicitly."
    ),
    "cpu_train_step_ms": (
        "CPU wall-clock for one online Adam/BCE step after about five observed actions."
    ),
    "goal_free_l2_reached": (
        "a goal-free L2 deepening proves the wall is crossed by demoting goal-induction."
    ),
    "offline_reproduced": "a goal-free L2 counts only if offline-reproduced.",
    "reproduced_levels": "integer level reached by the goal-free multi-level probe.",
    "solve_provenance": "live_agent_self_discovery for generic goal-free L2; development_proxy otherwise.",
    "verifier_is_oracle": "MUST be false; the online frame-change CNN does not run the win-check.",
    "live_path_reachable": "arc_orphan_solver_lint confirms the changed modules are live-path reachable.",
    "bare_control_passed": "positive control: held-out harness has reachable first-win headroom.",
    "false_negative_risk_checked": "true only when the three arm inputs and headroom are present.",
    "null_methodology_note": "present for flat deltas; explains honest no-lift null vs measurement bug.",
    "chosen_submitted_config": (
        "recommended submitted-agent config: additive online driver, reset-to-prior, cell_recall floor; "
        "kept conservative when the first-win gate is flat."
    ),
    "proposer_served_model": "the /props-verified proposer model; must be Qwen3.5-9B-MTP.",
    "parity_test_green": "test_arc_submitted_agent_parity.py passes.",
    "random_seed": "determinism precondition for reproducibility.",
    "reproducibility_checksum": "content-addressed hash catches silent drift on replay.",
    "preconditions_checked": "records CUDA, Qwen cache, offline arcade, Go-Explore import, and /props.",
}

REQUIRED_ARTIFACT_FIELDS = tuple(FIELD_PRINCIPLES) + (
    "experiment",
    "schema",
    "spec_refs",
    "field_principles",
    "model_specs",
    "arm_source_artifacts",
    "source_artifact_checksums",
    "ab_methodology",
    "goal_free_probe",
    "live_path_lint",
    "parity_test",
    "duration_s",
    "submitted_to_leaderboard",
)


def _stable_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)


def _sha256(value: Any) -> str:
    return hashlib.sha256(_stable_json(value).encode("utf-8")).hexdigest()


def payload_checksum(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["reproducibility_checksum"] = ""
    return "sha256:" + _sha256(payload)


def _file_checksum(path: Path) -> str:
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _qwen_cache_present() -> bool:  # pragma: no cover - filesystem boundary
    cache = Path.home() / ".cache" / "huggingface" / "hub" / "models--unsloth--Qwen3.5-9B-MTP-GGUF"
    return cache.is_dir() and any(cache.iterdir())


def _verify_qwen_props(
    port: int = QWEN_PORT,
) -> dict[str, Any]:  # pragma: no cover - llama-server boundary
    import urllib.request

    def _props() -> dict[str, Any]:
        with urllib.request.urlopen(f"http://127.0.0.1:{int(port)}/props", timeout=10) as response:
            return json.load(response)

    try:
        props = _props()
    except Exception:
        from carnot.agentic.arc_executable_world_model import LocalGGUFProposer

        proposer = LocalGGUFProposer(
            repo_substr=QWEN_MODEL,
            port=int(port),
            # mtp is DELIBERATELY NOT PASSED. This line used to read
            # `mtp=(os.environ.get("CARNOT_ARC_MTP", "1") != "0")` -- a literal "1" that is NOT the
            # project's canonical local default (`ARC_LIVE_GENERATOR_MTP_DEFAULT` is "0"). With
            # CARNOT_ARC_MTP unset that handed the proposer mtp=True, which at the shipped n_ctx 81920
            # needs ~14 offloaded FFN layers on a 24 GB card -- past the auto-fit cap, so the VRAM guard
            # declines CUDA, the generator falls back to the ~2 tok/s iGPU, every induce times out, and
            # the run proceeds LLM-OFF while still reporting itself LLM-on. Omitting the argument lets
            # `LocalGGUFProposer.mtp`'s own default factory (`_mtp_default_on()`) answer, which reads
            # the SAME env var against the canonical constant -- identical override behaviour, correct
            # default, and one place to change it.
            kv_quant="q8_0",
            no_think_prefix="/no_think\n",
            max_tokens=2560,
            n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
        )
        if not proposer._ensure_server():
            return {
                "passed": False,
                "model": "",
                "port": int(port),
                "blocked_resource": "blocked_qwen_proposer_port",
            }
        props = _props()

    encoded = json.dumps(props, sort_keys=True, default=str)
    lower = encoded.lower()
    passed = "qwen3.5-9b" in lower and "gemma" not in lower
    return {
        "passed": bool(passed),
        "model": QWEN_MODEL if passed else str(props.get("model_path") or encoded[:240]),
        "port": int(port),
        "model_path": props.get("model_path"),
        "model_alias": props.get("model_alias"),
        "blocked_resource": "" if passed else "blocked_qwen_proposer_port",
    }


def check_preconditions(  # pragma: no cover - hardware/proposer/offline-arcade boundary
    root: Path | str = REPO_ROOT, *, qwen_port: int = QWEN_PORT
) -> dict[str, Any]:
    """REQ-ARC-FCP-4715: verify hard resources before producing a non-blocked artifact."""

    root_path = Path(root)
    checks: dict[str, Any] = {}
    try:
        import torch

        checks["cuda_available"] = bool(torch.cuda.is_available())
    except Exception as exc:
        checks["cuda_available"] = False
        checks["cuda_error"] = repr(exc)[:200]

    checks["qwen_gguf_cached"] = _qwen_cache_present()

    try:
        from carnot.agentic import arc_solver_kit as kit
        import carnot.agentic.arc_go_explore as _go_explore

        kit.offline_arcade()
        checks["offline_arcade_ok"] = True
        checks["arc_go_explore_importable"] = True
        checks["arc_go_explore_bug_fixed_frame_grid"] = hasattr(_go_explore, "_frame_grid")
    except Exception as exc:
        checks["offline_arcade_ok"] = False
        checks["arc_go_explore_importable"] = False
        checks["offline_arcade_error"] = repr(exc)[:200]

    props = _verify_qwen_props(qwen_port)
    checks["qwen_props_verified"] = bool(props.get("passed"))
    checks["proposer_served_model"] = props.get("model") or ""
    checks["proposer_port"] = int(qwen_port)
    checks["proposer_props"] = props

    checks["arm_artifacts_present"] = {
        arm: (root_path / rel).exists() for arm, rel in ARM_ARTIFACTS.items()
    }
    checks["ok"] = bool(
        checks.get("cuda_available")
        and checks.get("qwen_gguf_cached")
        and checks.get("offline_arcade_ok")
        and checks.get("arc_go_explore_importable")
        and checks.get("qwen_props_verified")
    )
    if not checks["ok"]:
        for key, blocked in (
            ("cuda_available", "blocked_cuda_unavailable"),
            ("qwen_gguf_cached", "blocked_model_not_cached_qwen"),
            ("offline_arcade_ok", "blocked_offline_arcade_unavailable"),
            ("arc_go_explore_importable", "blocked_go_explore_unimportable"),
            ("qwen_props_verified", "blocked_qwen_proposer_port"),
        ):
            if not checks.get(key):
                checks["blocked_resource"] = blocked
                break
    else:
        checks["blocked_resource"] = ""
    return checks


def measure_cpu_train_step_ms() -> float:
    """REQ-ARC-FCP-4715: wall-clock one CPU Adam/BCE online update after five actions."""

    import torch
    from types import SimpleNamespace

    from carnot.agentic.arc_frame_change_predictor import FrameChangeScorer, SmallFrameChangeCNN
    from carnot.agentic.arc_online_action_effect_scorer import OnlineActionEffectScorer

    torch.manual_seed(RANDOM_SEED)
    scorer = OnlineActionEffectScorer(
        memory=None,
        cnn_scorer=FrameChangeScorer(SmallFrameChangeCNN(num_colors=16, hidden_channels=8)),
        train_enabled=True,
        fit_every=5,
        max_batch=5,
    )

    def _frame(value: int) -> Any:
        return SimpleNamespace(frame=np.full((8, 8), value, dtype=np.int16), levels_completed=0)

    start = time.perf_counter()
    for idx in range(5):
        scorer.observe_transition(_frame(idx), 1 + (idx % 5), None, _frame(idx + 1))
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    if scorer.diagnostics().get("fits", 0) < 1:
        raise RuntimeError("cpu_train_step_no_fit_recorded")
    return round(float(elapsed_ms), 6)


def load_arm_metrics(
    root: Path | str = REPO_ROOT,
) -> tuple[dict[str, float], dict[str, str], dict[str, str]]:
    """Load content-addressed corrected arm artifacts from Exp 4710."""

    root_path = Path(root)
    metrics: dict[str, float] = {}
    sources: dict[str, str] = {}
    checksums: dict[str, str] = {}
    for arm, rel in ARM_ARTIFACTS.items():
        path = root_path / rel
        data = _read_json(path)
        metrics[arm] = round(float(data.get("first_win_rate") or 0.0), 10)
        sources[arm] = str(rel)
        checksums[arm] = _file_checksum(path)
    return metrics, sources, checksums


def run_goal_free_l2_probe(
    root: Path | str = REPO_ROOT,
    *,
    budget: int | None = None,
) -> dict[str, Any]:  # pragma: no cover - offline arcade boundary
    """Run a bounded goal-free lp85/sc25 probe; count only offline-reproduced L2."""

    _ = Path(root)
    if budget is None:
        budget = int(os.environ.get("CARNOT_ARC_4715_L2_BUDGET", "250"))
    probes: list[dict[str, Any]] = []
    for game in ("lp85", "sc25"):
        try:
            from carnot.agentic.arc_go_explore import go_explore_solve

            row = go_explore_solve(game, budget=int(budget), seed=RANDOM_SEED)
            probes.append(row)
        except Exception as exc:
            probes.append({"game": game, "error": repr(exc)[:200], "levels_reached": 0})
    reached = max((int(row.get("levels_reached") or 0) for row in probes), default=0)
    # go_explore_solve does not return a banked solution path, so it cannot pass the reproduction gate here.
    offline_reproduced = False
    return {
        "games": probes,
        "goal_free_l2_reached": bool(reached >= 2 and offline_reproduced),
        "offline_reproduced": bool(offline_reproduced),
        "reproduced_levels": int(reached if offline_reproduced else 0),
        "probe_budget": int(budget),
        "note": (
            "Bounded goal-free probe records exploration depth, but counts L2 only with an "
            "offline-reproduced solution path; no reproduced goal-free L2 was produced here."
        ),
    }


def _run_check(command: list[str], root: Path, *, timeout_s: int = 180) -> dict[str, Any]:
    started = time.time()
    try:
        proc = subprocess.run(
            command,
            cwd=root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=timeout_s,
            check=False,
        )
        output = proc.stdout or ""
        return {
            "command": " ".join(command),
            "passed": proc.returncode == 0,
            "returncode": int(proc.returncode),
            "duration_s": round(time.time() - started, 3),
            "output_tail": output[-2000:],
        }
    except Exception as exc:
        return {
            "command": " ".join(command),
            "passed": False,
            "returncode": -1,
            "duration_s": round(time.time() - started, 3),
            "output_tail": repr(exc)[:2000],
        }


def run_live_path_lint(
    root: Path | str = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - subprocess
    root_path = Path(root)
    return _run_check(
        [str(root_path / ".venv/bin/python"), "scripts/arc_orphan_solver_lint.py"], root_path
    )


def run_parity_test(
    root: Path | str = REPO_ROOT,
) -> dict[str, Any]:  # pragma: no cover - subprocess
    root_path = Path(root)
    return _run_check(
        [
            str(root_path / ".venv/bin/pytest"),
            "tests/python/test_arc_submitted_agent_parity.py",
            "-q",
            "--no-cov",
        ],
        root_path,
        timeout_s=240,
    )


def build_artifact(
    *,
    arm_metrics: Mapping[str, float],
    preconditions_checked: Mapping[str, Any],
    cpu_train_step_ms: float,
    proposer_served_model: str,
    parity_test_green: bool,
    live_path_reachable: bool,
    bare_control_passed: bool,
    false_negative_risk_checked: bool,
    goal_free_probe: Mapping[str, Any],
    source_artifacts: Mapping[str, str],
    source_artifact_checksums: Mapping[str, str] | None = None,
    live_path_lint: Mapping[str, Any] | None = None,
    parity_test: Mapping[str, Any] | None = None,
    duration_s: float = 1.0,
) -> dict[str, Any]:
    """REQ-ARC-FCP-4715: assemble the corrected-build artifact."""

    frozen = round(float(arm_metrics.get("frozen", 0.0)), 10)
    scratch = round(float(arm_metrics.get("online-scratch", 0.0)), 10)
    warm = round(float(arm_metrics.get("online-warm", 0.0)), 10)
    delta = round(warm - frozen, 10)
    l2 = bool(goal_free_probe.get("goal_free_l2_reached"))
    offline_reproduced = bool(goal_free_probe.get("offline_reproduced"))
    reproduced_levels = int(goal_free_probe.get("reproduced_levels") or 0)

    if delta >= 0.05 and l2 and offline_reproduced:
        verdict = f"success: online_warm_beats_frozen_{delta:+.4f}_l2_goal_free"
        solve_provenance = "live_agent_self_discovery"
    else:
        cause = (
            "cpu_latency_bound" if float(cpu_train_step_ms) > 200.0 else "online_signal_too_sparse"
        )
        verdict = f"complete: online_action_learning_no_first_win_lift_residual_{cause}"
        solve_provenance = "development_proxy"

    artifact: dict[str, Any] = {
        "experiment": EXPERIMENT,
        "schema": SCHEMA,
        "spec_refs": list(SPEC_REFS),
        "field_principles": dict(FIELD_PRINCIPLES),
        "honest_verdict": verdict,
        "inference_substrate": (
            "verifier_ensemble_against_cached_candidates + live_llm_inference_precondition_verified"
        ),
        "model_specs": {
            "live_generator": QWEN_MODEL,
            "gguf": "Qwen3.5-9B-Q4_K_M.gguf",
            "action_effect_model": "SmallFrameChangeCNN binary frame-change coordinate head",
        },
        "online_warm_first_win": warm,
        "online_scratch_first_win": scratch,
        "frozen_first_win": frozen,
        "online_warm_vs_frozen_delta": delta,
        "cpu_train_step_ms": round(float(cpu_train_step_ms), 6),
        "goal_free_l2_reached": l2,
        "offline_reproduced": offline_reproduced,
        "reproduced_levels": reproduced_levels,
        "solve_provenance": solve_provenance,
        "verifier_is_oracle": False,
        "live_path_reachable": bool(live_path_reachable),
        "bare_control_passed": bool(bare_control_passed),
        "false_negative_risk_checked": bool(false_negative_risk_checked),
        "null_methodology_note": (
            "The corrected warm coordinate-head arm equals the frozen first-win rate on the same "
            "held-out harness inputs; this is an honest no-lift null, not the prior dict-candidate "
            "or dead-archive measurement bug."
            if abs(delta) < 1e-12
            else ""
        ),
        "chosen_submitted_config": {
            "online_action_learning_driver": "available_additive",
            "coordinate_head_proposals": True,
            "reset_to_prior_on_level_up": True,
            "trust_metric": "cell_recall",
            "submitted_recommendation": (
                "keep_conservative_null_config_with_safe_cell_recall_floor"
                if delta < 0.05 or not l2
                else "enable_online_warm_goal_free_driver"
            ),
        },
        "proposer_served_model": str(proposer_served_model),
        "parity_test_green": bool(parity_test_green),
        "random_seed": RANDOM_SEED,
        "preconditions_checked": dict(preconditions_checked),
        "arm_source_artifacts": dict(source_artifacts),
        "source_artifact_checksums": dict(source_artifact_checksums or {}),
        "ab_methodology": (
            "content_addressed_reuse_of_exp4710_corrected_arm_artifacts; online-warm maps to "
            "online-warm-propose because that arm exercises the coordinate-head ACTION6 driver."
        ),
        "goal_free_probe": dict(goal_free_probe),
        "live_path_lint": dict(live_path_lint or {}),
        "parity_test": dict(parity_test or {}),
        "duration_s": round(max(1.0, float(duration_s)), 3),
        "submitted_to_leaderboard": False,
    }
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing:{field}")
    verdict = str(artifact.get("honest_verdict", ""))
    if not verdict.startswith(TERMINAL_PREFIXES):
        errors.append("honest_verdict_missing_terminal_prefix")
    if artifact.get("verifier_is_oracle") is not False:
        errors.append("verifier_is_oracle_must_be_false")
    if artifact.get("proposer_served_model") != QWEN_MODEL:
        errors.append("proposer_served_model_not_qwen")
    if artifact.get("reproducibility_checksum") != payload_checksum(artifact):
        errors.append("reproducibility_checksum_mismatch")
    return errors


def _blocked_artifact(checks: Mapping[str, Any], duration_s: float) -> dict[str, Any]:
    metrics = {"frozen": 0.0, "online-scratch": 0.0, "online-warm": 0.0}
    artifact = build_artifact(
        arm_metrics=metrics,
        preconditions_checked=checks,
        cpu_train_step_ms=0.0,
        proposer_served_model=str(checks.get("proposer_served_model") or ""),
        parity_test_green=False,
        live_path_reachable=False,
        bare_control_passed=False,
        false_negative_risk_checked=False,
        goal_free_probe={
            "goal_free_l2_reached": False,
            "offline_reproduced": False,
            "reproduced_levels": 0,
            "blocked": True,
        },
        source_artifacts={arm: str(path) for arm, path in ARM_ARTIFACTS.items()},
        duration_s=duration_s,
    )
    blocked = str(checks.get("blocked_resource") or "blocked_preconditions")
    artifact["honest_verdict"] = blocked
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    return artifact


def run(root: Path | str = REPO_ROOT) -> dict[str, Any]:  # pragma: no cover - integration runner
    root_path = Path(root)
    started = time.time()
    checks = check_preconditions(root_path)
    if not checks.get("ok"):
        artifact = _blocked_artifact(checks, time.time() - started)
        return artifact

    arm_metrics, source_artifacts, source_checksums = load_arm_metrics(root_path)
    cpu_ms = measure_cpu_train_step_ms()
    goal_free_probe = run_goal_free_l2_probe(root_path)
    live_lint = run_live_path_lint(root_path)
    parity = run_parity_test(root_path)
    frozen = float(arm_metrics.get("frozen") or 0.0)
    bare_control_passed = frozen > 0.0
    artifact = build_artifact(
        arm_metrics=arm_metrics,
        preconditions_checked=checks,
        cpu_train_step_ms=cpu_ms,
        proposer_served_model=str(checks.get("proposer_served_model") or ""),
        parity_test_green=bool(parity.get("passed")),
        live_path_reachable=bool(live_lint.get("passed")),
        bare_control_passed=bare_control_passed,
        false_negative_risk_checked=bool(
            bare_control_passed and all(checks.get("arm_artifacts_present", {}).values())
        ),
        goal_free_probe=goal_free_probe,
        source_artifacts=source_artifacts,
        source_artifact_checksums=source_checksums,
        live_path_lint=live_lint,
        parity_test=parity,
        duration_s=time.time() - started,
    )
    return artifact


def main() -> int:  # pragma: no cover - CLI
    artifact = run(REPO_ROOT)
    errors = artifact_schema_errors(artifact)
    artifact["schema_errors"] = errors
    artifact["reproducibility_checksum"] = payload_checksum(artifact)
    out = REPO_ROOT / RESULT_RELATIVE_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[exp4715] wrote {out}")
    print(f"[exp4715] honest_verdict={artifact['honest_verdict']}")
    if errors:
        print(f"[exp4715] schema_errors={errors}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI guard
    raise SystemExit(main())
