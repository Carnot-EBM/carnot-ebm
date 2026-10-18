"""Run Exp5215: PAW compile-once amortization gate for ARC-AGI-3.

This is intentionally a pure analysis benchmark. It replays existing public
ARC action logs to recover level-up checkpoints, runs bounded local timing, and
writes the terminal JSON artifact. It does not alter the live agent or the ARC
solve registry.
"""

from __future__ import annotations

import hashlib
import json
import logging
import statistics
import time
from pathlib import Path
from typing import Any

from carnot.agentic import arc_paw_amortization_gate as gate


REPO = Path(__file__).resolve().parents[2]
RESULT_PATH = REPO / "results" / "experiment_5215_arc_paw_amortization_gate_v477.json"
REGISTRY_PATH = REPO / "ops" / "arc_solve_registry.yaml"
CURRENT_STEP_TOKEN_BUDGET = 64
PAW_NOTE_INTERPRETER_TOK_PER_S = 30.0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _walk_numbers(payload: Any, key: str) -> list[float]:
    if isinstance(payload, dict):
        values = [float(payload[key])] if key in payload and isinstance(payload[key], int | float) else []
        for value in payload.values():
            values.extend(_walk_numbers(value, key))
        return values
    if isinstance(payload, list):
        values: list[float] = []
        for item in payload:
            values.extend(_walk_numbers(item, key))
        return values
    return []


def _current_step_wall_clock(results_dir: Path) -> tuple[float, dict[str, Any]]:
    paths = [
        results_dir / "arc3_layerb_repeat_bench_qwen3_5-9b-mtp_mtp.json",
        results_dir / "arc3_layerb_ar_model_test_qwen3_5-9b-mtp_mtp.json",
    ]
    tok_per_s: list[float] = []
    read_paths: list[str] = []
    for path in paths:
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            tok_per_s.extend(_walk_numbers(payload, "tok_per_s"))
            read_paths.append(str(path))
    median_tok_per_s = statistics.median(tok_per_s) if tok_per_s else 8.0
    wall_s = CURRENT_STEP_TOKEN_BUDGET / float(median_tok_per_s)
    return wall_s, {
        "method": "current step = 64 generated action-decision tokens / median logged Qwen3.5-9B-MTP tok_per_s",
        "source_paths": read_paths,
        "tok_per_s_values": tok_per_s,
        "median_tok_per_s": median_tok_per_s,
        "token_budget": CURRENT_STEP_TOKEN_BUDGET,
    }


def _cheap_step_wall_clock() -> tuple[float, dict[str, Any]]:
    measured = 0.0
    backend = "unavailable"
    try:
        import torch

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        backend = f"torch_{device}"
        x = torch.randn(1, 512, device=device)
        w1 = torch.randn(512, 512, device=device)
        w2 = torch.randn(512, 8, device=device)
        for _ in range(10):
            y = torch.relu(x @ w1) @ w2
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        start = time.perf_counter()
        repeats = 200
        for _ in range(repeats):
            y = torch.relu(x @ w1) @ w2
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        measured = (time.perf_counter() - start) / repeats
        float(y.sum().detach().cpu())
    except Exception as exc:  # pragma: no cover - hardware fallback.
        backend = f"blocked_{type(exc).__name__}"
    conservative_floor = CURRENT_STEP_TOKEN_BUDGET / PAW_NOTE_INTERPRETER_TOK_PER_S
    return max(measured, conservative_floor), {
        "method": "max(local tiny interpreter timing, PAW-note 0.6B CPU 30 tok/s conservative floor)",
        "backend": backend,
        "measured_local_step_s": measured,
        "conservative_floor_s": conservative_floor,
        "token_budget": CURRENT_STEP_TOKEN_BUDGET,
    }


def _compile_wall_clock() -> tuple[float, dict[str, Any]]:
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("cuda_unavailable")
        device = "cuda:0"
        torch.manual_seed(5215)

        class LoraLinear(torch.nn.Module):
            def __init__(self, in_f: int, out_f: int, rank: int = 8) -> None:
                super().__init__()
                self.weight = torch.nn.Parameter(
                    torch.empty(out_f, in_f, device=device, dtype=torch.float16),
                    requires_grad=False,
                )
                torch.nn.init.normal_(self.weight, std=0.01)
                self.a = torch.nn.Parameter(
                    torch.randn(rank, in_f, device=device, dtype=torch.float16) * 0.01
                )
                self.b = torch.nn.Parameter(torch.zeros(out_f, rank, device=device, dtype=torch.float16))
                self.scale = 1.0 / rank

            def forward(self, x: Any) -> Any:
                return torch.nn.functional.linear(x, self.weight) + (
                    torch.nn.functional.linear(torch.nn.functional.linear(x, self.a), self.b)
                    * self.scale
                )

        class SyntheticBlock(torch.nn.Module):
            def __init__(self) -> None:
                super().__init__()
                hidden = 4096
                ff = 11008
                self.q = LoraLinear(hidden, hidden)
                self.k = LoraLinear(hidden, hidden)
                self.v = LoraLinear(hidden, hidden)
                self.o = LoraLinear(hidden, hidden)
                self.up = LoraLinear(hidden, ff)
                self.down = LoraLinear(ff, hidden)

            def forward(self, x: Any) -> Any:
                y = (self.q(x) + self.k(x) + self.v(x)) / 3
                y = self.o(torch.nn.functional.silu(y))
                return self.down(torch.nn.functional.silu(self.up(x + y)))

        block = SyntheticBlock()
        params = [param for param in block.parameters() if param.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=1e-3)
        x = torch.randn(128, 4096, device=device, dtype=torch.float16)
        target = torch.randn(128, 4096, device=device, dtype=torch.float16)
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(block(x).float(), target.float())
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        timed_steps = 10
        start = time.perf_counter()
        for _ in range(timed_steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(block(x).float(), target.float())
            loss.backward()
            optimizer.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        per_block_step_s = elapsed / timed_steps
        target_layers = 40
        compile_steps = 128
        safety_factor = 4.0
        estimate = per_block_step_s * target_layers * compile_steps * safety_factor
        evidence = {
            "method": (
                "synthetic Qwen-size LoRA block timing scaled to 40 layers * 128 steps * "
                "4x conservative safety for omitted attention, data, and adapter materialization costs"
            ),
            "device": torch.cuda.get_device_name(0),
            "elapsed_s": elapsed,
            "timed_steps": timed_steps,
            "per_block_step_s": per_block_step_s,
            "target_layers": target_layers,
            "compile_steps": compile_steps,
            "safety_factor": safety_factor,
            "trainable_lora_params_in_block": sum(param.numel() for param in params),
            "dataset": "synthetic Gaussian batch, no ARC hidden data",
        }
        return estimate, evidence
    except Exception as exc:  # pragma: no cover - hardware fallback.
        fallback = 600.0
        return fallback, {
            "method": "fallback conservative estimate because local LoRA timing failed",
            "fallback_compile_wall_clock_s": fallback,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _records_with_replay_boundaries(records: tuple[gate.ArcEpisodeRecord, ...]) -> tuple[gate.ArcEpisodeRecord, ...]:
    logging.getLogger("arc_agi").setLevel(logging.WARNING)
    from carnot.agentic import arc_game_adapters
    from carnot.agentic import arc_solver_kit

    arc = arc_solver_kit.offline_arcade()
    enriched: list[gate.ArcEpisodeRecord] = []
    for record in records:
        if record.level_up_action_indices or not record.solution_labels:
            enriched.append(record)
            continue
        adapter = arc_game_adapters.get_adapter(record.game)
        if adapter is None:
            enriched.append(record)
            continue
        try:
            env = arc.make(record.game, scorecard_id=arc.open_scorecard())
            indices = gate.replay_level_up_action_indices(
                labels=record.solution_labels,
                env=env,
                apply=adapter.apply,
                warmup_label=adapter.warmup_label,
                frame_level=arc_solver_kit.frame_level,
            )
            enriched.append(record.with_level_up_action_indices(indices))
        except Exception:
            enriched.append(record)
    return tuple(enriched)


def main() -> None:
    started = time.perf_counter()
    before = _sha256(REGISTRY_PATH)
    records = gate.load_arc_loop_records(REPO / "results")
    records = _records_with_replay_boundaries(records)
    compile_s, compile_evidence = _compile_wall_clock()
    current_s, current_evidence = _current_step_wall_clock(REPO / "results")
    cheap_s, cheap_evidence = _cheap_step_wall_clock()
    after = _sha256(REGISTRY_PATH)
    timing = gate.TimingEstimate(
        compile_wall_clock_s=compile_s,
        current_step_wall_clock_s=current_s,
        cheap_step_wall_clock_s=cheap_s,
        evidence={
            "compile": compile_evidence,
            "current_step": current_evidence,
            "cheap_step": cheap_evidence,
            "arc_registry_sha256_before": before,
            "arc_registry_sha256_after": after,
            "arc_registry_modified_by_runner": before != after,
        },
    )
    artifact = gate.build_artifact(
        records=records,
        timing=timing,
        duration_s=time.perf_counter() - started,
    )
    gate.write_artifact(RESULT_PATH, artifact)
    print(
        f"WROTE {RESULT_PATH} viable={artifact['paw_amortization_viable']['value']} "
        f"median={artifact['median_remaining_actions']['value']} "
        f"p75={artifact['p75_remaining_actions']['value']} "
        f"break_even={artifact['break_even_remaining_actions']['value']}"
    )


if __name__ == "__main__":
    main()
