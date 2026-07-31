"""Exp 4100 conditional TRM verifier-RFT runner.

Spec refs: REQ-LEARN-4100, SCENARIO-LEARN-4100-SMOKE,
SCENARIO-LEARN-4100-RFT.
"""

from __future__ import annotations

from carnot.serialization_safety import safe_torch_load

import hashlib
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


try:  # pragma: no cover - exercised by native nano-trm Hydra process.
    from lightning import Callback
except Exception:  # pragma: no cover - keeps unit imports robust if lightning is absent.
    Callback = object  # type: ignore[assignment,misc]


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULT_FILENAME = "experiment_4100_trm_verifier_rft_conditional.json"
DEFAULT_OUTPUT = REPO_ROOT / "results" / RESULT_FILENAME
DEFAULT_TRM_WEIGHTS_DIR = (
    Path.home() / ".cache" / "huggingface" / "hub" / "models--arcprize--trm_arc_prize_verification"
)
EXP4099_PATTERN = "experiment_4099_*.json"
RANDOM_SEED = 4100
TERMINAL_PREFIXES = ("complete:", "success:", "passed:", "shipped:")
BLOCKED_PREFIX = "blocked_"
REQUIRED_ARTIFACT_FIELDS = (
    "honest_verdict",
    "branch_taken",
    "rft_vs_ablation_delta",
    "trm_native_trainer_checkpoint_ok",
    "preconditions_checked",
    "random_seed",
    "reproducibility_checksum",
)


@dataclass(frozen=True)
class PreconditionCheck:
    """One resource check recorded before Exp 4100 can train anything."""

    resource: str
    available: bool
    detail: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class SmokeRunResult:
    """Measured result from the native nano-trm checkpoint smoke."""

    checkpoint_ok: bool
    checkpoint_reload_ok: bool
    checkpoint_path: Path | None
    duration_s: float
    command: list[str]
    stdout_tail: list[str]


@dataclass(frozen=True)
class RftRunResult:
    """Measured result from the full verifier-certified RFT branch."""

    trm_native_trainer_checkpoint_ok: bool
    duration_s: float
    rft_vs_ablation_delta: dict[str, Any]
    arm_metrics: dict[str, Any]
    corpus_summary: dict[str, Any]


@dataclass(frozen=True)
class NativeSmokeConfig:
    """Native nano-trm trainer settings for the mechanism-smoke branch."""

    repo_root: Path | str = REPO_ROOT
    nano_trm_root: Path | str | None = None
    output_dir: Path | str | None = None
    python_executable: str | None = None
    random_seed: int = RANDOM_SEED
    max_steps: int = 200
    train_puzzles: int = 32
    val_puzzles: int = 8
    batch_size: int = 4
    hidden_size: int = 64
    num_heads: int = 4
    timeout_s: int = 1200

    def __post_init__(self) -> None:
        root = Path(self.repo_root)
        object.__setattr__(self, "repo_root", root)
        object.__setattr__(
            self,
            "nano_trm_root",
            Path(self.nano_trm_root) if self.nano_trm_root else root / "nano-trm",
        )
        output = (
            Path(self.output_dir)
            if self.output_dir
            else root / "results" / "experiment_4100_native_smoke"
        )
        object.__setattr__(self, "output_dir", output)
        python_executable = self.python_executable
        if python_executable is None:
            nano_python = root / "nano-trm" / ".venv" / "bin" / "python"
            python_executable = str(nano_python if nano_python.exists() else sys.executable)
        object.__setattr__(self, "python_executable", python_executable)

    @property
    def nano_trm_src(self) -> Path:
        return Path(self.nano_trm_root) / "src"

    @property
    def checkpoint_dir(self) -> Path:
        return Path(self.output_dir) / "checkpoints"

    def plan(self) -> dict[str, Any]:
        return {
            "native_trainer": "nano-trm/src/nn/train.py",
            "task": "generated_sudoku_4x4_mechanism_smoke",
            "max_steps": int(self.max_steps),
            "train_puzzles": int(self.train_puzzles),
            "val_puzzles": int(self.val_puzzles),
            "batch_size": int(self.batch_size),
            "hidden_size": int(self.hidden_size),
            "num_heads": int(self.num_heads),
            "random_seed": int(self.random_seed),
        }


@dataclass(frozen=True)
class RftConfig:
    """Inputs passed to the full RFT branch when Exp 4099 found signal."""

    repo_root: Path
    exp4099_artifact: Mapping[str, Any]
    exp4099_path: Path
    best_reranker: str
    random_seed: int = RANDOM_SEED


class NanoTrmProgressPrinter(Callback):  # pragma: no cover - used inside native trainer subprocess.
    """Lightning callback that prints one line per training step for Codex liveness."""

    def __init__(self, every_n_steps: int = 1) -> None:
        self.every_n_steps = max(int(every_n_steps), 1)

    def on_train_batch_end(
        self, trainer: Any, _pl_module: Any, _outputs: Any, _batch: Any, batch_idx: int
    ) -> None:
        step = int(getattr(trainer, "global_step", 0))
        if step == 0 or step % self.every_n_steps == 0:
            print(
                f"[exp4100:native-train] step={step} epoch={getattr(trainer, 'current_epoch', 0)} "
                f"batch_idx={batch_idx}",
                flush=True,
            )

    def on_train_epoch_end(self, trainer: Any, _pl_module: Any) -> None:
        print(
            f"[exp4100:native-train] epoch_end={getattr(trainer, 'current_epoch', 0)} "
            f"step={getattr(trainer, 'global_step', 0)}",
            flush=True,
        )


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable(item)
            for key, item in sorted(value.items(), key=lambda item: str(item[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _stable_checksum(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(_jsonable(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _default_cuda_checker() -> tuple[bool, str]:
    try:
        import torch  # pylint: disable=import-outside-toplevel
    except Exception as exc:  # pragma: no cover - environment dependent.
        return False, f"{type(exc).__name__}: {exc}"
    available = bool(torch.cuda.is_available())
    return available, f"torch.cuda.is_available()={available}"


def _check_trm_weights(path: Path) -> PreconditionCheck:
    if not path.exists() or not path.is_dir():
        return PreconditionCheck("trm_weights_cached", False, f"missing directory: {path}")
    if not any(path.iterdir()):
        return PreconditionCheck("trm_weights_cached", False, f"empty directory: {path}")
    return PreconditionCheck("trm_weights_cached", True, f"non-empty: {path}")


def _check_nano_trm(repo_root: Path) -> PreconditionCheck:
    required = [
        repo_root / "nano-trm" / "src" / name for name in ("arc_evaluator.py", "baseline.py")
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        return PreconditionCheck("nano_trm_substrate", False, "missing: " + ", ".join(missing))
    return PreconditionCheck("nano_trm_substrate", True, "found arc_evaluator.py and baseline.py")


def _check_cuda(cuda_checker: Callable[[], tuple[bool, str]]) -> PreconditionCheck:
    try:
        available, detail = cuda_checker()
    except Exception as exc:
        return PreconditionCheck("cuda_available", False, f"{type(exc).__name__}: {exc}")
    return PreconditionCheck("cuda_available", bool(available), str(detail))


def _load_exp4099(repo_root: Path) -> tuple[PreconditionCheck, Path | None, dict[str, Any] | None]:
    candidates = sorted(
        (repo_root / "results").glob(EXP4099_PATTERN),
        key=lambda path: (path.stat().st_mtime, path.name),
    )
    if not candidates:
        return (
            PreconditionCheck("exp4099_probe", False, f"missing results/{EXP4099_PATTERN}"),
            None,
            None,
        )
    path = candidates[-1]
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return PreconditionCheck("exp4099_probe", False, f"{type(exc).__name__}: {exc}"), path, None
    if not isinstance(payload.get("verifier_beats_trm_vote"), bool):
        return (
            PreconditionCheck("exp4099_probe", False, "verifier_beats_trm_vote is not a bare bool"),
            path,
            None,
        )
    return PreconditionCheck("exp4099_probe", True, f"loaded {path}"), path, payload


def check_preconditions(
    *,
    repo_root: str | Path = REPO_ROOT,
    trm_weights_dir: str | Path = DEFAULT_TRM_WEIGHTS_DIR,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
) -> tuple[list[PreconditionCheck], str | None, Path | None, dict[str, Any] | None]:
    """REQ-LEARN-4100: verify TRM, nano-trm, CUDA, and Exp 4099 before training."""

    root = Path(repo_root)
    checks = [
        _check_trm_weights(Path(trm_weights_dir)),
        _check_nano_trm(root),
        _check_cuda(cuda_checker),
    ]
    exp_check, exp4099_path, exp4099_artifact = _load_exp4099(root)
    checks.append(exp_check)

    blockers = {
        "trm_weights_cached": "blocked_trm_weights_not_cached",
        "nano_trm_substrate": "blocked_trm_substrate_missing",
        "cuda_available": "blocked_cuda_unavailable",
        "exp4099_probe": "blocked_exp4099_probe_missing",
    }
    for check in checks:
        if not check.available:
            return checks, blockers[check.resource], exp4099_path, exp4099_artifact
    return checks, None, exp4099_path, exp4099_artifact


def _field_principles() -> dict[str, str]:
    return {
        "honest_verdict": "Terminal-prefixed complete/success/passed/shipped verdict; blocked_* only for precondition stops.",
        "branch_taken": "Records rft, smoke, or blocked so a reader knows whether full RFT actually happened.",
        "rft_vs_ablation_delta": "De-confounded verifier-label A-vs-vote-label B held-out delta with bootstrap CI.",
        "trm_native_trainer_checkpoint_ok": "Bare bool for native nano-trm checkpoint save plus reload.",
        "preconditions_checked": "Resource checks for TRM weights, nano-trm substrate, CUDA, and Exp 4099.",
        "random_seed": "Determinism precondition for the training run.",
        "reproducibility_checksum": "Content hash of split/corpus or smoke substrate inputs.",
    }


def _precondition_dicts(
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for check in preconditions_checked:
        if isinstance(check, PreconditionCheck):
            rows.append(check.to_dict())
        else:
            rows.append(dict(check))
    return rows


def _not_run_delta(status: str) -> dict[str, Any]:
    return {
        "metric": "heldout_pass@2",
        "delta": 0.0,
        "ci95": [0.0, 0.0],
        "status": status,
    }


def build_blocked_artifact(
    reason: str,
    *,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    duration_s: float = 0.0,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    payload_for_checksum = {
        "reason": reason,
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "random_seed": int(random_seed),
    }
    artifact: dict[str, Any] = {
        "experiment": "experiment_4100_trm_verifier_rft_conditional",
        "schema": "carnot.experiment_4100_trm_verifier_rft_conditional.v1",
        "honest_verdict": reason,
        "branch_taken": "blocked",
        "rft_vs_ablation_delta": _not_run_delta("blocked_precondition"),
        "trm_native_trainer_checkpoint_ok": False,
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _stable_checksum(payload_for_checksum),
        "duration_s": round(float(duration_s), 3),
        "n_codex_calls": 0,
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-LEARN-4100"],
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def _exp4099_gap(exp4099_artifact: Mapping[str, Any]) -> dict[str, Any]:
    best = str(exp4099_artifact.get("best_reranker", "unknown"))
    per_reranker = exp4099_artifact.get("per_reranker")
    best_row = per_reranker.get(best, {}) if isinstance(per_reranker, Mapping) else {}
    return {
        "verifier_beats_trm_vote": bool(exp4099_artifact.get("verifier_beats_trm_vote", False)),
        "best_reranker": best,
        "captured_pp_directional": float(
            exp4099_artifact.get("captured_pp_directional", 0.0) or 0.0
        ),
        "best_captured_pp": best_row.get("captured_pp", 0.0)
        if isinstance(best_row, Mapping)
        else 0.0,
        "best_captured_pp_ci95": best_row.get("captured_pp_ci95", [0.0, 0.0])
        if isinstance(best_row, Mapping)
        else [0.0, 0.0],
        "bottleneck": "verifier_discrimination_on_trm_grids",
    }


def build_smoke_artifact(
    *,
    exp4099_artifact: Mapping[str, Any],
    exp4099_path: Path | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    smoke_result: SmokeRunResult,
    smoke_plan: Mapping[str, Any],
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4100-SMOKE: build artifact for the no-signal smoke branch."""

    checkpoint_ok = bool(smoke_result.checkpoint_ok and smoke_result.checkpoint_reload_ok)
    gap = _exp4099_gap(exp4099_artifact)
    if checkpoint_ok:
        verdict = (
            "complete: trm_native_trainer_checkpoint_ok_smoke_only_"
            f"no_verifier_grid_discrimination_best_{gap['best_reranker']}_"
            f"gap_{float(gap['captured_pp_directional']):.4f}"
        )
    else:
        verdict = "complete: trm_native_trainer_checkpoint_failed_smoke_only_no_verifier_grid_discrimination"
    checksum_payload = {
        "branch": "smoke",
        "exp4099_path": exp4099_path,
        "exp4099_checksum": exp4099_artifact.get("reproducibility_checksum"),
        "exp4099_verifier_beats_trm_vote": exp4099_artifact.get("verifier_beats_trm_vote"),
        "smoke_plan": dict(smoke_plan),
        "random_seed": int(random_seed),
    }
    artifact: dict[str, Any] = {
        "experiment": "experiment_4100_trm_verifier_rft_conditional",
        "schema": "carnot.experiment_4100_trm_verifier_rft_conditional.v1",
        "honest_verdict": verdict,
        "branch_taken": "smoke",
        "rft_vs_ablation_delta": _not_run_delta("not_run_no_verifier_signal"),
        "trm_native_trainer_checkpoint_ok": checkpoint_ok,
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(smoke_result.duration_s), 3),
        "n_codex_calls": 0,
        "inference_substrate": "native_nano_trm_trainer_checkpoint_smoke",
        "exp4099_path": str(exp4099_path) if exp4099_path else None,
        "verifier_gap": gap,
        "native_smoke": {
            "checkpoint_path": str(smoke_result.checkpoint_path)
            if smoke_result.checkpoint_path
            else None,
            "checkpoint_reload_ok": bool(smoke_result.checkpoint_reload_ok),
            "command": list(smoke_result.command),
            "stdout_tail": list(smoke_result.stdout_tail[-20:]),
            "plan": dict(smoke_plan),
        },
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-LEARN-4100", "SCENARIO-LEARN-4100-SMOKE"],
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def build_rft_artifact(
    *,
    exp4099_artifact: Mapping[str, Any],
    exp4099_path: Path | None,
    preconditions_checked: Sequence[PreconditionCheck | Mapping[str, Any]],
    rft_result: RftRunResult,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """SCENARIO-LEARN-4100-RFT: build artifact for the full RFT branch."""

    delta = dict(rft_result.rft_vs_ablation_delta)
    ci = delta.get("ci95", [0.0, 0.0])
    ci_low = float(ci[0]) if isinstance(ci, Sequence) and len(ci) == 2 else 0.0
    point_delta = float(delta.get("delta", 0.0) or 0.0)
    if rft_result.trm_native_trainer_checkpoint_ok and point_delta > 0.0 and ci_low > 0.0:
        verdict = f"success: verifier_rft_beats_vote_ablation_delta_{point_delta:.4f}_ci95_low_{ci_low:.4f}"
    elif rft_result.trm_native_trainer_checkpoint_ok:
        verdict = f"complete: verifier_rft_no_ci_separated_ablation_lift_delta_{point_delta:.4f}"
    else:
        verdict = "complete: verifier_rft_native_trainer_checkpoint_failed"
    checksum_payload = {
        "branch": "rft",
        "exp4099_path": exp4099_path,
        "exp4099_checksum": exp4099_artifact.get("reproducibility_checksum"),
        "corpus_summary": rft_result.corpus_summary,
        "arm_metrics": rft_result.arm_metrics,
        "delta": delta,
        "random_seed": int(random_seed),
    }
    artifact: dict[str, Any] = {
        "experiment": "experiment_4100_trm_verifier_rft_conditional",
        "schema": "carnot.experiment_4100_trm_verifier_rft_conditional.v1",
        "honest_verdict": verdict,
        "branch_taken": "rft",
        "rft_vs_ablation_delta": delta,
        "trm_native_trainer_checkpoint_ok": bool(rft_result.trm_native_trainer_checkpoint_ok),
        "preconditions_checked": _precondition_dicts(preconditions_checked),
        "random_seed": int(random_seed),
        "reproducibility_checksum": _stable_checksum(checksum_payload),
        "duration_s": round(float(rft_result.duration_s), 3),
        "n_codex_calls": 0,
        "inference_substrate": "native_nano_trm_verifier_rft",
        "exp4099_path": str(exp4099_path) if exp4099_path else None,
        "arm_metrics": dict(rft_result.arm_metrics),
        "corpus_summary": dict(rft_result.corpus_summary),
        "field_principles": _field_principles(),
        "spec_refs": ["REQ-LEARN-4100", "SCENARIO-LEARN-4100-RFT"],
    }
    errors = artifact_schema_errors(artifact)
    if errors:
        raise ValueError("; ".join(errors))
    return artifact


def artifact_schema_errors(artifact: Mapping[str, Any]) -> list[str]:
    errors: list[str] = []
    for field in REQUIRED_ARTIFACT_FIELDS:
        if field not in artifact:
            errors.append(f"missing required field {field}")
    verdict = artifact.get("honest_verdict")
    if not isinstance(verdict, str):
        errors.append("honest_verdict must be a string")
    elif not verdict.startswith((*TERMINAL_PREFIXES, BLOCKED_PREFIX)):
        errors.append("honest_verdict must be terminal-prefixed or blocked")
    if artifact.get("branch_taken") not in {"rft", "smoke", "blocked"}:
        errors.append("branch_taken must be one of rft, smoke, blocked")
    if not isinstance(artifact.get("trm_native_trainer_checkpoint_ok"), bool):
        errors.append("trm_native_trainer_checkpoint_ok must be a bare bool")
    if not isinstance(artifact.get("preconditions_checked"), list):
        errors.append("preconditions_checked must be a list")
    if not isinstance(artifact.get("random_seed"), int) or isinstance(
        artifact.get("random_seed"), bool
    ):
        errors.append("random_seed must be a bare int")
    if not isinstance(artifact.get("reproducibility_checksum"), str):
        errors.append("reproducibility_checksum must be a string")
    delta = artifact.get("rft_vs_ablation_delta")
    if not isinstance(delta, Mapping):
        errors.append("rft_vs_ablation_delta must be a dict")
    else:
        ci = delta.get("ci95")
        if not isinstance(ci, list) or len(ci) != 2:
            errors.append("rft_vs_ablation_delta must include two-element ci95")
        elif not all(
            isinstance(value, (int, float)) and not isinstance(value, bool) for value in ci
        ):
            errors.append("rft_vs_ablation_delta ci95 values must be numeric")
    return errors


def ensure_nano_trm_src_on_path(nano_trm_root: Path) -> Path:
    """Add nano-trm/src to sys.path because nano-trm is not pip-installed."""

    src = nano_trm_root / "src"
    src_text = str(src)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    return src


def build_native_smoke_command(config: NativeSmokeConfig) -> list[str]:
    train_py = Path(config.nano_trm_root) / "src" / "nn" / "train.py"
    epochs = max(
        (int(config.max_steps) // max(int(config.train_puzzles) // int(config.batch_size), 1)) + 1,
        1,
    )
    return [
        str(config.python_executable),
        str(train_py),
        "experiment=trm_sudoku_4x4",
        "logger=csv",
        f"hydra.run.dir={Path(config.output_dir)}",
        "save_dir=null",
        "append_wandb_name_to_save_dir=false",
        f"seed={int(config.random_seed)}",
        "data.data_dir=null",
        f"data.num_train_puzzles={int(config.train_puzzles)}",
        f"data.num_val_puzzles={int(config.val_puzzles)}",
        "data.num_test_puzzles=0",
        "data.max_grid_size=4",
        f"timekeeping.max_epochs={epochs}",
        f"timekeeping.batch_size={int(config.batch_size)}",
        "timekeeping.num_workers=0",
        f"trainer.max_epochs={epochs}",
        f"+trainer.max_steps={int(config.max_steps)}",
        "+trainer.num_sanity_val_steps=0",
        "+trainer.limit_val_batches=1",
        "+trainer.log_every_n_steps=1",
        "trainer.precision=32",
        "callbacks.model_checkpoint.monitor=null",
        "callbacks.model_checkpoint.save_last=true",
        "callbacks.model_checkpoint.save_top_k=-1",
        "callbacks.model_checkpoint.every_n_train_steps=50",
        f"callbacks.model_checkpoint.dirpath={Path(config.checkpoint_dir)}",
        "callbacks.model_checkpoint.filename=smoke_step",
        (
            "+callbacks.exp4100_progress._target_="
            "carnot.agentic.arc_exp4100_trm_verifier_rft_conditional.NanoTrmProgressPrinter"
        ),
        "+callbacks.exp4100_progress.every_n_steps=1",
        "model_tuning.hidden_size=64",
        f"model_tuning.num_heads={int(config.num_heads)}",
        "model_tuning.num_layers=1",
        "model_tuning.N_supervision=1",
        "model_tuning.N_supervision_val=1",
        "model_tuning.H_cycles=1",
        "model_tuning.L_cycles=1",
        "model_tuning.ffn_expansion=2",
        "model_tuning.puzzle_emb_dim=0",
        "model_tuning.puzzle_emb_len=0",
        "model_tuning.pos_emb_type=null",
        "model_tuning.use_mlp_t=true",
        "model_tuning.use_conv_swiglu=false",
        "model_tuning.use_board_swiglu=false",
        "model_tuning.use_muon=false",
    ]


def build_native_smoke_env(config: NativeSmokeConfig) -> dict[str, str]:
    repo_python = Path(config.repo_root) / "python"
    paths = [
        str(repo_python),
        str(config.repo_root),
        str(config.nano_trm_src),
        str(config.nano_trm_root),
    ]
    env = dict(os.environ)
    existing = env.get("PYTHONPATH")
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env["PYTHONUNBUFFERED"] = "1"
    env["WANDB_MODE"] = "disabled"
    env["WANDB_DISABLED"] = "true"
    return env


def _load_torch_checkpoint(path: Path) -> tuple[bool, str]:
    try:
        import torch  # pylint: disable=import-outside-toplevel

        try:
            payload = safe_torch_load(path, map_location="cpu", allow_unsafe_pickle=True)
        except TypeError:  # pragma: no cover - older torch.
            payload = torch.load(path, map_location="cpu")
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"
    if not isinstance(payload, Mapping):
        return False, f"unexpected checkpoint payload: {type(payload).__name__}"
    return True, "torch.load ok"


def _latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    last = checkpoint_dir / "last.ckpt"
    if last.exists():
        return last
    candidates = sorted(
        checkpoint_dir.glob("*.ckpt"), key=lambda path: (path.stat().st_mtime, path.name)
    )
    return candidates[-1] if candidates else None


def run_native_trm_smoke(
    config: NativeSmokeConfig,
) -> SmokeRunResult:  # pragma: no cover - launches external trainer.
    """Run a real native nano-trm training smoke and reload its checkpoint."""

    started = time.time()
    ensure_nano_trm_src_on_path(Path(config.nano_trm_root))
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    command = build_native_smoke_command(config)
    env = build_native_smoke_env(config)
    stdout_lines: list[str] = []
    print(f"[exp4100] launching native nano-trm smoke: max_steps={config.max_steps}", flush=True)
    try:
        proc = subprocess.Popen(  # noqa: S603 - command is constructed from local repo paths.
            command,
            cwd=str(config.nano_trm_root),
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
    except Exception as exc:
        return SmokeRunResult(
            False, False, None, time.time() - started, command, [f"{type(exc).__name__}: {exc}"]
        )

    assert proc.stdout is not None
    for line in proc.stdout:
        clean = line.rstrip()
        stdout_lines.append(clean)
        print(f"[exp4100:nano-trm] {clean}", flush=True)
        if time.time() - started > config.timeout_s:
            proc.kill()
            stdout_lines.append(f"timeout_s exceeded: {config.timeout_s}")
            break
    return_code = proc.wait()
    checkpoint_path = _latest_checkpoint(config.checkpoint_dir)
    reload_ok = False
    reload_detail = "checkpoint missing"
    if checkpoint_path is not None:
        reload_ok, reload_detail = _load_torch_checkpoint(checkpoint_path)
    stdout_lines.append(f"return_code={return_code}")
    stdout_lines.append(f"checkpoint={checkpoint_path}")
    stdout_lines.append(f"checkpoint_reload={reload_detail}")
    duration_s = time.time() - started
    return SmokeRunResult(
        checkpoint_ok=return_code == 0 and checkpoint_path is not None and reload_ok,
        checkpoint_reload_ok=reload_ok,
        checkpoint_path=checkpoint_path,
        duration_s=duration_s,
        command=command,
        stdout_tail=stdout_lines[-20:],
    )


def default_rft_runner(_config: RftConfig) -> RftRunResult:
    """Fail closed if a future Exp 4099 artifact asks for RFT without a runner."""

    return RftRunResult(
        trm_native_trainer_checkpoint_ok=False,
        duration_s=0.0,
        rft_vs_ablation_delta={
            "metric": "heldout_pass@2",
            "delta": 0.0,
            "ci95": [0.0, 0.0],
            "status": "not_run_default_rft_runner_unimplemented",
        },
        arm_metrics={},
        corpus_summary={"status": "not_run_default_rft_runner_unimplemented"},
    )


def run_experiment(
    *,
    repo_root: str | Path = REPO_ROOT,
    output_path: str | Path | None = None,
    trm_weights_dir: str | Path = DEFAULT_TRM_WEIGHTS_DIR,
    cuda_checker: Callable[[], tuple[bool, str]] = _default_cuda_checker,
    smoke_runner: Callable[[NativeSmokeConfig], SmokeRunResult] = run_native_trm_smoke,
    rft_runner: Callable[[RftConfig], RftRunResult] = default_rft_runner,
    random_seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    started = time.time()
    root = Path(repo_root)
    out_path = Path(output_path) if output_path is not None else root / "results" / RESULT_FILENAME
    checks, blocker, exp4099_path, exp4099_artifact = check_preconditions(
        repo_root=root,
        trm_weights_dir=trm_weights_dir,
        cuda_checker=cuda_checker,
    )
    if blocker is not None or exp4099_artifact is None:
        artifact = build_blocked_artifact(
            blocker or "blocked_exp4099_probe_missing",
            preconditions_checked=checks,
            duration_s=time.time() - started,
            random_seed=random_seed,
        )
        _write_json(out_path, artifact)
        return artifact

    if bool(exp4099_artifact.get("verifier_beats_trm_vote")):
        config = RftConfig(
            repo_root=root,
            exp4099_artifact=exp4099_artifact,
            exp4099_path=exp4099_path or root / "results" / EXP4099_PATTERN,
            best_reranker=str(exp4099_artifact.get("best_reranker", "unknown")),
            random_seed=random_seed,
        )
        try:
            rft_result = rft_runner(config)
        except Exception as exc:
            rft_result = RftRunResult(
                trm_native_trainer_checkpoint_ok=False,
                duration_s=time.time() - started,
                rft_vs_ablation_delta={
                    "metric": "heldout_pass@2",
                    "delta": 0.0,
                    "ci95": [0.0, 0.0],
                    "status": f"{type(exc).__name__}: {exc}",
                },
                arm_metrics={},
                corpus_summary={"error": f"{type(exc).__name__}: {exc}"},
            )
        artifact = build_rft_artifact(
            exp4099_artifact=exp4099_artifact,
            exp4099_path=exp4099_path,
            preconditions_checked=checks,
            rft_result=rft_result,
            random_seed=random_seed,
        )
    else:
        smoke_config = NativeSmokeConfig(repo_root=root, random_seed=random_seed)
        try:
            smoke_result = smoke_runner(smoke_config)
        except Exception as exc:
            smoke_result = SmokeRunResult(
                checkpoint_ok=False,
                checkpoint_reload_ok=False,
                checkpoint_path=None,
                duration_s=time.time() - started,
                command=[],
                stdout_tail=[f"{type(exc).__name__}: {exc}"],
            )
        artifact = build_smoke_artifact(
            exp4099_artifact=exp4099_artifact,
            exp4099_path=exp4099_path,
            preconditions_checked=checks,
            smoke_result=smoke_result,
            smoke_plan=smoke_config.plan(),
            random_seed=random_seed,
        )
    _write_json(out_path, artifact)
    return artifact


def main() -> None:  # pragma: no cover - thin CLI wrapper.
    artifact = run_experiment()
    print(
        json.dumps(
            {field: artifact.get(field) for field in REQUIRED_ARTIFACT_FIELDS},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
