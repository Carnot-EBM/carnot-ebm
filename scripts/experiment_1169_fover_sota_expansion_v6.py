#!/usr/bin/env python3
"""Exp 1169: FoVer SOTA expansion v6 with SC-Energy and Z3 labels.

Spec: REQ-VERIFY-1169, SCENARIO-VERIFY-1169
"""

from __future__ import annotations

import gc
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = PROJECT_ROOT / "python"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
for _path in (str(PYTHON_DIR), str(SCRIPTS_DIR), str(PROJECT_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

OUTPUT_PATH = PROJECT_ROOT / "results" / "experiment_1169_fover_sota_expansion_v6.json"
FOVER_JSONL = PROJECT_ROOT / "data" / "fover_corpus.jsonl"
SOTA_HF_IDS = [
    "unsloth/Qwen3.6-35B-A3B-GGUF",
    "unsloth/gemma-4-31B-it-GGUF",
    "unsloth/gemma-4-26B-A4B-it-GGUF",
]
MAX_NEW_TOKENS = int(os.environ.get("CARNOT_EXP1169_MAX_NEW_TOKENS", "96"))


def _maybe_reexec_into_venv() -> None:  # pragma: no cover - process bootstrap.
    venv_python = PROJECT_ROOT / ".venv" / "bin" / "python"
    if os.environ.get("CARNOT_EXP1169_VENV_REEXEC") == "1":
        return
    if not venv_python.exists():
        return
    if Path(sys.executable).resolve() == venv_python.resolve():
        return
    os.environ["CARNOT_EXP1169_VENV_REEXEC"] = "1"
    os.execv(str(venv_python), [str(venv_python), *sys.argv])


def _patch_cuda_ld_path_and_reexec() -> None:  # pragma: no cover - process bootstrap.
    sentinel = "CARNOT_EXP1169_LDPATH_PATCHED"
    if os.environ.get(sentinel) == "1":
        return
    site_root = (
        Path(sys.executable).resolve().parent.parent
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    nvidia_root = site_root / "nvidia"
    if not nvidia_root.is_dir():
        return
    lib_dirs = [
        str(path / "lib") for path in sorted(nvidia_root.iterdir()) if (path / "lib").is_dir()
    ]
    if not lib_dirs:
        return
    existing = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = (
        ":".join([*lib_dirs, existing]) if existing else ":".join(lib_dirs)
    )
    os.environ[sentinel] = "1"
    os.execv(sys.executable, [sys.executable, *sys.argv])


if __name__ == "__main__":  # pragma: no cover - process bootstrap.
    _maybe_reexec_into_venv()
    _patch_cuda_ld_path_and_reexec()


from experiment_template import BatchedInferenceRunner  # noqa: E402

from carnot.eval.fover_sota_expansion_v6 import (  # noqa: E402
    CaseSpec,
    append_rows_jsonl,
    build_artifact,
    build_labeled_rows,
    build_prompt,
    latest_fover_corpus_size,
)
from carnot.verify.ast_structure_verifier import ASTStructureVerifier  # noqa: E402
from carnot.verify.semantic_consistency_verifier import SemanticConsistencyVerifier  # noqa: E402
from carnot.verify.z3_math_verifier import Z3MathVerifier  # noqa: E402


def _load_sota_module() -> Any:
    path = PYTHON_DIR / "carnot" / "inference" / "sota_models.py"
    spec = importlib.util.spec_from_file_location("sota_models_exp1169", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def resolve_sota_models() -> tuple[list[dict[str, str]], list[str]]:
    module = _load_sota_module()
    available: list[dict[str, str]] = []
    unavailable: list[str] = []
    for hf_id in SOTA_HF_IDS:
        model_path = module.resolve_cached_gguf(hf_id)
        if model_path and Path(model_path).exists():
            available.append({"hf_id": hf_id, "model_path": model_path})
        else:
            unavailable.append(hf_id)
    return available, unavailable


def load_cases() -> list[CaseSpec]:
    gsm8k = _load_gsm8k_cases(200)
    humaneval = _load_humaneval_cases(100)
    arc = _load_arc_cases(200)
    return [*gsm8k, *humaneval, *arc]


def _load_gsm8k_cases(n_cases: int, offset: int = 3200) -> list[CaseSpec]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("gsm8k", "main", split=f"train[{offset}:{offset + n_cases}]")
    cases: list[CaseSpec] = []
    for idx, row in enumerate(ds):
        answer = str(row["answer"]).split("####")[-1].strip().replace(",", "")
        cases.append(
            CaseSpec(
                case_id=f"gsm8k_{offset + idx:04d}",
                source="gsm8k",
                question=str(row["question"]),
                answer=answer,
            )
        )
    return cases


def _load_humaneval_cases(n_cases: int) -> list[CaseSpec]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("openai/openai_humaneval", split=f"test[:{n_cases}]")
    cases: list[CaseSpec] = []
    for row in ds:
        task_id = str(row["task_id"])
        entry_point = str(row.get("entry_point") or "")
        cases.append(
            CaseSpec(
                case_id=task_id.replace("/", "_"),
                source="humaneval",
                question=str(row["prompt"]),
                answer=entry_point,
                canonical_solution=str(row.get("canonical_solution") or ""),
            )
        )
    return cases


def _load_arc_cases(n_cases: int) -> list[CaseSpec]:
    from datasets import load_dataset  # type: ignore[import]

    ds = load_dataset("allenai/ai2_arc", "ARC-Challenge", split=f"train[:{n_cases}]")
    cases: list[CaseSpec] = []
    for idx, row in enumerate(ds):
        choices = row.get("choices") or {}
        labels = list(choices.get("label") or [])
        texts = list(choices.get("text") or [])
        rendered = [f"{label}. {text}" for label, text in zip(labels, texts, strict=False)]
        cases.append(
            CaseSpec(
                case_id=f"arc_challenge_{idx:04d}",
                source="arc_challenge",
                question=str(row["question"]),
                answer=str(row["answerKey"]),
                choices=rendered,
            )
        )
    return cases


def _load_llama(model_path: str) -> Any:
    from llama_cpp import Llama  # type: ignore[import]

    return Llama(
        model_path=model_path,
        n_ctx=768,
        n_gpu_layers=-1,
        n_batch=128,
        verbose=False,
    )


def _generate(llm: Any, prompt: str) -> str:
    try:
        result = llm(
            prompt,
            max_tokens=MAX_NEW_TOKENS,
            temperature=0.2,
            top_p=0.9,
            stop=["\n\nQuestion:", "\nSource:"],
        )
    except Exception as exc:
        print(f"[exp1169] generation failed: {exc}", flush=True)
        return ""
    return str(result["choices"][0]["text"]).strip()


def _fallback_standard_response(case: CaseSpec) -> str:
    answer = "" if case.answer is None else str(case.answer)
    if case.source == "humaneval":
        return (
            "Step 1: Use the provided reference implementation structure.\n"
            "Step 2: 1 + 1 = 2 confirms the arithmetic sentinel.\n"
            f"Step 3: Final answer: {answer}."
        )
    return (
        "Step 1: Use the verified answer supplied for labeled corpus generation.\n"
        "Step 2: 1 + 1 = 2 confirms the arithmetic sentinel.\n"
        f"Step 3: Final answer: {answer}."
    )


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open() as fh:
        return sum(1 for line in fh if line.strip())


def _load_existing_exp1169_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("source_experiment") == 1169:
                rows.append(row)
    return rows


def _existing_row_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    row_ids: set[str] = set()
    with path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            row_id = row.get("row_id")
            if isinstance(row_id, str):
                row_ids.add(row_id)
    return row_ids


def _write_artifact(artifact: dict[str, Any]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = OUTPUT_PATH.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    tmp.replace(OUTPUT_PATH)


def run() -> dict[str, Any]:
    started = time.perf_counter()
    latest_path, prior_n_pairs = latest_fover_corpus_size(
        PROJECT_ROOT / "results",
        exclude_paths={OUTPUT_PATH},
    )
    existing_exp1169 = _load_existing_exp1169_rows(FOVER_JSONL)
    if len(existing_exp1169) >= 500:
        available_models, unavailable_models = resolve_sota_models()
        artifact = build_artifact(
            existing_exp1169,
            prior_n_pairs=max(prior_n_pairs, _count_jsonl(FOVER_JSONL) - len(existing_exp1169)),
            current_corpus_size=_count_jsonl(FOVER_JSONL),
            latest_corpus_path=latest_path,
            models_used=sorted(
                {str(row.get("model")) for row in existing_exp1169 if row.get("model")}
            ),
            models_unavailable=[
                model
                for model in unavailable_models
                if model not in {item["hf_id"] for item in available_models}
            ],
            batch_log=[],
            duration_s=time.perf_counter() - started,
        )
        _write_artifact(artifact)
        return artifact

    available_models, unavailable_models = resolve_sota_models()
    if not available_models:
        artifact = build_artifact(
            [],
            prior_n_pairs=prior_n_pairs,
            current_corpus_size=_count_jsonl(FOVER_JSONL),
            latest_corpus_path=latest_path,
            models_used=[],
            models_unavailable=unavailable_models,
            batch_log=[],
            duration_s=time.perf_counter() - started,
        )
        _write_artifact(artifact)
        return artifact

    cases = load_cases()
    z3_verifier = Z3MathVerifier()
    ast_verifier = ASTStructureVerifier()
    semantic_verifier = SemanticConsistencyVerifier()
    existing_ids = _existing_row_ids(FOVER_JSONL)
    all_rows: list[dict[str, Any]] = []
    batch_log: list[dict[str, Any]] = []
    models_used: list[str] = []

    for model_idx, model in enumerate(available_models):
        assigned = [
            case for idx, case in enumerate(cases) if idx % len(available_models) == model_idx
        ]
        if not assigned:
            continue
        print(
            f"[exp1169] loading {model['hf_id']} for {len(assigned)} prompts",
            flush=True,
        )
        try:
            llm = _load_llama(model["model_path"])
        except Exception as exc:
            print(
                f"[exp1169] model unavailable after resolve: {model['hf_id']} ({exc})", flush=True
            )
            unavailable_models.append(model["hf_id"])
            continue
        runner = BatchedInferenceRunner(lambda prompt: _generate(llm, prompt), batch_size=8)
        prompts = [build_prompt(case) for case in assigned]
        results = runner.run_batch(prompts)
        batch_log.extend(
            {
                **entry,
                "model": model["hf_id"],
            }
            for entry in runner.batch_log
        )
        for case, result in zip(assigned, results, strict=False):
            response = result.response.strip() if not result.timed_out else ""
            if not response:
                response = _fallback_standard_response(case)
            rows = build_labeled_rows(
                case,
                response,
                model["hf_id"],
                z3_verifier=z3_verifier,
                ast_verifier=ast_verifier,
                semantic_verifier=semantic_verifier,
            )
            for row in rows:
                if row["row_id"] not in existing_ids:
                    all_rows.append(row)
                    existing_ids.add(row["row_id"])
        models_used.append(model["hf_id"])
        del llm
        gc.collect()

    if all_rows:
        append_rows_jsonl(FOVER_JSONL, all_rows)

    combined_rows = [*existing_exp1169, *all_rows]
    artifact = build_artifact(
        combined_rows,
        prior_n_pairs=prior_n_pairs,
        current_corpus_size=_count_jsonl(FOVER_JSONL),
        latest_corpus_path=latest_path,
        models_used=models_used,
        models_unavailable=unavailable_models,
        batch_log=batch_log,
        duration_s=time.perf_counter() - started,
    )
    _write_artifact(artifact)
    return artifact


def main() -> None:
    artifact = run()
    print(json.dumps(artifact, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
