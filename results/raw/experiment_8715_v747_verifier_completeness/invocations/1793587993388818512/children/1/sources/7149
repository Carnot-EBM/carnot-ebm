#!/usr/bin/env python3
"""Run Exp 3031 tiny DCCD-style structured repair panel.

Spec: REQ-CODE-3031, SCENARIO-CODE-3031.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "python"))

from carnot.eval import dccd_structured_repair_panel_3031 as panel  # noqa: E402


class LlamaCppPanelGenerator:
    """Small live-GGUF generator used by the Exp 3031 CLI."""

    def __init__(
        self,
        *,
        n_ctx: int = 4096,
        n_batch: int = 128,
        n_gpu_layers: int = -1,
        main_gpu: int = 0,
        temperature: float = 0.1,
        max_tokens: int = 320,
    ) -> None:
        self.n_ctx = n_ctx
        self.n_batch = n_batch
        self.n_gpu_layers = n_gpu_layers
        self.main_gpu = main_gpu
        self.temperature = temperature
        self.max_tokens = max_tokens
        self._llm: Any | None = None
        self._model_path: str | None = None

    def __call__(
        self,
        case: panel.PanelCase,
        mode: str,
        draft_text: str | None,
        model_spec: dict[str, Any],
    ) -> panel.GenerationResult:
        prompt = _prompt_for_case(case, mode, draft_text)
        started = time.monotonic()
        try:
            output = self._ensure_loaded(str(model_spec["model_path"]))(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                seed=_stable_seed(case.item_id, mode),
            )
        except Exception as exc:  # noqa: BLE001
            return panel.GenerationResult(
                raw_text="",
                duration_s=round(time.monotonic() - started, 6),
                tokens_generated=0,
                error=f"{type(exc).__name__}: {exc}",
            )
        choice = (output.get("choices") or [{}])[0]
        raw_text = str(choice.get("text") or "").strip()
        tokens = int(output.get("usage", {}).get("completion_tokens") or len(raw_text.split()))
        return panel.GenerationResult(
            raw_text=raw_text,
            duration_s=round(time.monotonic() - started, 6),
            tokens_generated=tokens,
        )

    def _ensure_loaded(self, model_path: str) -> Any:
        if self._llm is not None and self._model_path == model_path:
            return self._llm
        if self._llm is not None:
            self.close()
        from llama_cpp import Llama

        self._llm = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_batch=self.n_batch,
            n_ubatch=self.n_batch,
            n_gpu_layers=self.n_gpu_layers,
            main_gpu=self.main_gpu,
            verbose=False,
        )
        self._model_path = model_path
        return self._llm

    def close(self) -> None:
        if self._llm is not None and hasattr(self._llm, "close"):
            self._llm.close()
        self._llm = None
        self._model_path = None


def _prompt_for_case(case: panel.PanelCase, mode: str, draft_text: str | None) -> str:
    tests = "\n".join(str(row.get("code") or "") for row in case.tests)
    if mode == panel.UNCONSTRAINED_MODE:
        return (
            "Repair this Python function. First state the intended semantic fix in one "
            "sentence, then provide the corrected function.\n\n"
            f"Problem: {case.prompt}\n"
            f"Expected behavior: {case.expected_behavior}\n"
            f"Function name: {case.entry_point}\n"
            f"Buggy code:\n{case.baseline_candidate}\n"
            f"Failing validators:\n{tests}\n"
        )
    return (
        "Use the unconstrained draft to repair the Python function, but emit only one "
        "valid JSON object with string fields draft_intent and final_patch. final_patch "
        f"must be a complete Python function named {case.entry_point}.\n\n"
        f"Problem: {case.prompt}\n"
        f"Expected behavior: {case.expected_behavior}\n"
        f"Buggy code:\n{case.baseline_candidate}\n"
        f"Unconstrained draft:\n{draft_text or ''}\n"
        f"Validators:\n{tests}\n"
    )


def _stable_seed(item_id: str, mode: str) -> int:
    digest = hashlib.sha256(f"3031:{item_id}:{mode}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--test-run", action="append", default=[])
    parser.add_argument("--selected-model-path", type=Path, default=None)
    parser.add_argument("--selected-model-id", default=None)
    parser.add_argument("--main-gpu", type=int, default=0)
    parser.add_argument("--max-tokens", type=int, default=320)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    generator = LlamaCppPanelGenerator(
        main_gpu=args.main_gpu,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
    )
    config = panel.ExperimentConfig(
        output_path=args.output,
        selected_model_path=args.selected_model_path,
        selected_model_id=args.selected_model_id,
        tests_run=tuple(args.test_run),
    )
    try:
        artifact = panel.write_artifact(config, generator_fn=generator)
    finally:
    generator.close()
    print(json.dumps(artifact, indent=2, sort_keys=True))
    verdict = str(artifact.get("honest_verdict") or "")
    accepted_verdict_prefixes = (
        "complete:",
        "complete_flagged:",
        "blocked_sota_headline_model_unavailable",
    )
    return 0 if verdict.startswith(accepted_verdict_prefixes) else 1


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper.
    raise SystemExit(main())
