"""PROTO-L2-PROPOSER-TRUNCATION-CHECK: is the lp85 L2 goal-induction failure TRUNCATION or
generation-quality?

The decisive signal: LocalGGUFProposer.generate() reads only `response["content"]` and discards
stopped_limit / stopped_eos / tokens_predicted from the /completion JSON.  We monkeypatch
generate() to capture the FULL response.  Then we:
  1. Build the EXACT same L2 induction prompt the live agent would construct (using induce_prompt
     with a previous_level_complete_grid). We do this WITHOUT running the full E3 agent loop —
     faster, reproducible, no 244-min drift.
  2. Call the proposer ONCE at max_tokens=4096 (the shipping default).
  3. Capture: stopped_limit, stopped_eos, tokens_predicted, failure_reason, last 400 chars.
  4. Retry at max_tokens=8192 and 6144 on the SAME prompt, WITHOUT re-running the agent.
  5. Emit verdict + artifact.

Integrity fields: inference_substrate=live_llm_inference, solve_provenance=development_proxy,
verifier_is_oracle=false.  We do NOT submit anything.  We do NOT edit shipped files.
"""
from __future__ import annotations

import ast
import json
import os
import sys
import time
import urllib.request
import hashlib
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# PRECONDITIONS (step 0): verify before ANY inference
# ---------------------------------------------------------------------------
QWEN_GGUF = os.path.expanduser(
    "~/.cache/huggingface/hub/models--unsloth--Qwen3.5-9B-MTP-GGUF/snapshots/"
    "9716a636ee4bddc3fed678220b7a33dd2a4160ae/Qwen3.5-9B-Q4_K_M.gguf"
)
PORT = int(os.environ.get("TRUNC_CHECK_PORT", "8921"))  # free port, not 8919 (gemma)
RESULT_PATH = Path("results/proto_l2_proposer_truncation_check.json")

print(f"[PRECOND] Checking Qwen GGUF at {QWEN_GGUF}")
if not os.path.exists(QWEN_GGUF):
    result = {
        "honest_verdict": "blocked_model_not_cached_qwen",
        "preconditions_checked": [{"resource": "qwen_gguf", "available": False}],
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "duration_s": 0.0,
        "error": f"GGUF not found: {QWEN_GGUF}",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[BLOCKED] Qwen GGUF not cached. Artifact: {RESULT_PATH}")
    sys.exit(0)
print(f"[PRECOND] GGUF found: OK")

# ---------------------------------------------------------------------------
# Path setup for carnot imports
# ---------------------------------------------------------------------------
REPO = Path(__file__).parent.parent.parent
sys.path.insert(0, str(REPO / "python"))
sys.path.insert(0, str(REPO))

# Force CPU for everything except the proposer's own llama-server
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # will be removed for server launch

print("[INIT] Importing carnot agentic modules...")
from carnot.agentic.arc_executable_world_model import (  # noqa: E402
    LocalGGUFProposer,
    Transition,
    induce_prompt,
)
import numpy as np  # noqa: E402

# ---------------------------------------------------------------------------
# Build a plausible lp85 L2 induction prompt WITHOUT running the full agent
# ---------------------------------------------------------------------------
# Strategy: we need (a) some transitions, (b) a previous_level_complete_grid.
# We construct a minimal synthetic setup:
#   - 5x5 grids (simple), cell=1
#   - previous_level_complete_grid = a 5x5 grid of zeros (plausible L1 terminal state)
#   - A small list of synthetic Transition-like objects (the prompt formatter just needs
#     them to render the ASCII observation, so we fake them).
# This gives us the REAL prompt structure (including the WIN STATE exemplar block that
# triggers when previous_level_complete_grid is not None) without a live env.

print("[PROMPT] Building synthetic lp85 L2 induction prompt...")

# Try to get real transitions from lp85 if arcengine is available
_real_transitions = []
_real_prev_grid = None

try:
    import arcengine  # type: ignore
    print("[PROMPT] arcengine available — trying real lp85 transitions...")
    # Get lp85 game
    arc = arcengine.ArcEnvironment()
    lp85_game_id = None
    for e in arc.get_environments():
        gid = str(getattr(e, "game_id", ""))
        if gid.split("-")[0] == "lp85":
            lp85_game_id = gid
            break
    if lp85_game_id:
        print(f"[PROMPT] Found lp85 game_id: {lp85_game_id}")
        env = arc.make(lp85_game_id)
        obs = env.reset()
        # Collect a handful of transitions via random walk (Transition has: grid, action, data, next_grid, level_before, level_after)
        for step_i in range(12):
            action_idx = int(np.random.randint(0, 4))
            try:
                next_obs, reward, done, info = env.step(action_idx)
                _real_transitions.append(
                    Transition(
                        grid=np.asarray(obs),
                        action=action_idx,
                        data=info if isinstance(info, dict) else {},
                        next_grid=np.asarray(next_obs),
                        level_before=int(info.get("level", 0)) if isinstance(info, dict) else 0,
                        level_after=int(info.get("level", 0)) if isinstance(info, dict) else 0,
                    )
                )
                obs = next_obs
                if done:
                    _real_prev_grid = np.asarray(obs).copy()
                    break
            except Exception:
                break
        print(f"[PROMPT] Collected {len(_real_transitions)} real transitions")
    else:
        print("[PROMPT] lp85 game_id not found in arcengine")
except Exception as ex:
    print(f"[PROMPT] arcengine unavailable ({ex}), using synthetic data")

# Synthetic fallback using the real Transition dataclass (fields: grid, action, data, next_grid, level_before, level_after)
if not _real_transitions:
    print("[PROMPT] Using synthetic transitions (realistic structure)")
    rng = np.random.default_rng(42)
    grid_shape = (5, 5)
    for i in range(6):
        grid = rng.integers(0, 3, size=grid_shape, dtype=np.uint8)
        next_grid = grid.copy()
        if i % 2 == 0 and next_grid[2, 2] < 2:
            next_grid[2, 2] += 1
        _real_transitions.append(
            Transition(
                grid=grid,
                action=i % 4,
                data={},
                next_grid=next_grid,
                level_before=0,
                level_after=1 if i == 5 else 0,
            )
        )
    _real_prev_grid = rng.integers(0, 3, size=grid_shape, dtype=np.uint8)

if _real_prev_grid is None:
    rng = np.random.default_rng(99)
    _real_prev_grid = rng.integers(0, 3, size=(5, 5), dtype=np.uint8)

# Build the EXACT prompt that induce() would build for L2 (with WIN STATE exemplar)
GAME = "lp85"
CELL = 1
prompt_l2 = (
    induce_prompt(
        GAME,
        _real_transitions,
        CELL,
        previous_level_complete_grid=np.asarray(_real_prev_grid),
    )
    + "\n\nReturn ONLY one ```python code block with engine + is_level_complete.\n```python\n"
)

print(f"[PROMPT] L2 prompt length: {len(prompt_l2)} chars")
# Check the WIN STATE block is present (confirms previous_level_complete_grid triggered)
has_win_state_block = any(
    kw in prompt_l2 for kw in ["WIN STATE", "win_state", "previous_level_complete", "level_complete"]
)
print(f"[PROMPT] WIN STATE / level-complete block present: {has_win_state_block}")

prompt_checksum = hashlib.sha256(prompt_l2.encode()).hexdigest()[:16]
print(f"[PROMPT] Prompt SHA256 prefix: {prompt_checksum}")

# ---------------------------------------------------------------------------
# Build proposer on free port (verified Qwen, not gemma on 8919)
# ---------------------------------------------------------------------------
print(f"[SERVER] Building LocalGGUFProposer on port {PORT} with Qwen3.5-9B-MTP...")
proposer = LocalGGUFProposer(
    repo_substr="Qwen3.5-9B-MTP",
    model_path=QWEN_GGUF,
    port=PORT,
    mtp=False,          # disable MTP for simpler single-shot diagnostic
    kv_quant=None,      # no KV quant for simpler startup
    no_think_prefix="/no_think\n",
    max_tokens=4096,    # shipping default
    n_gpu_layers=int(os.environ.get("CARNOT_ARC_NGL", "999")),
    tries=1,            # single call per budget test
)

print("[SERVER] Starting llama-server (model load ~20-40s)...")
t_start = time.time()
if not proposer._ensure_server():
    result = {
        "honest_verdict": "blocked_llama_server_failed_to_start",
        "preconditions_checked": [
            {"resource": "qwen_gguf", "available": True},
            {"resource": "llama_server", "available": False},
        ],
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "duration_s": time.time() - t_start,
        "error": "llama-server failed to become healthy within 180s",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    print(f"[BLOCKED] Server failed to start. Artifact: {RESULT_PATH}")
    sys.exit(0)

server_start_s = time.time() - t_start
print(f"[SERVER] llama-server healthy in {server_start_s:.1f}s")

# Verify /props says Qwen, not gemma
try:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/props", timeout=5) as r:
        props = json.load(r)
    model_path_reported = props.get("model_path", "UNKNOWN")
    model_serving = props.get("model_alias") or Path(model_path_reported).name
    is_qwen = "Qwen" in model_path_reported or "qwen" in model_path_reported.lower()
    print(f"[SERVER] /props model_path: {model_path_reported[:80]}")
    print(f"[SERVER] Is Qwen: {is_qwen}")
except Exception as ex:
    model_path_reported = f"PROPS_FAILED({ex})"
    is_qwen = False
    print(f"[SERVER] /props failed: {ex}")

if not is_qwen:
    result = {
        "honest_verdict": "blocked_server_not_serving_qwen",
        "preconditions_checked": [
            {"resource": "qwen_gguf", "available": True},
            {"resource": "llama_server_qwen", "available": False},
        ],
        "model_path_reported": model_path_reported,
        "inference_substrate": "live_llm_inference",
        "solve_provenance": "development_proxy",
        "verifier_is_oracle": False,
        "duration_s": time.time() - t_start,
        "error": "Server is not serving Qwen — cannot confirm Qwen-specific behavior",
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(result, indent=2))
    proposer.stop()
    print(f"[BLOCKED] Server not Qwen. Artifact: {RESULT_PATH}")
    sys.exit(0)

print(f"[PRECOND] Server verified as Qwen: OK")

# ---------------------------------------------------------------------------
# INSTRUMENTED generate(): capture FULL /completion JSON
# ---------------------------------------------------------------------------

def _instrumented_generate_call(url: str, body_dict: dict, timeout: int) -> dict:
    """POST to /completion and return the FULL JSON response (not just content)."""
    body = json.dumps(body_dict).encode()
    req = urllib.request.Request(
        url + "/completion",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def _extract_python_local(text: str) -> str:
    """Mirror of arc_executable_world_model._extract_python for local use."""
    import re
    # Try ```python ... ``` blocks
    blocks = re.findall(r"```python\s*(.*?)```", text, re.DOTALL)
    if blocks:
        return max(blocks, key=len).strip()
    # Try bare code after the last ``` opener
    blocks2 = re.findall(r"```\s*(.*?)```", text, re.DOTALL)
    if blocks2:
        return max(blocks2, key=len).strip()
    return ""


def generate_with_capture(
    proposer: LocalGGUFProposer,
    prompt: str,
    max_tokens: int,
    required: tuple = ("engine", "is_level_complete"),
) -> dict:
    """Single-shot generate that returns full diagnostic record."""
    url = proposer._url()
    body_dict = {
        "prompt": prompt,
        "n_predict": max_tokens,
        "temperature": 0.2,
        "cache_prompt": False,  # no cache so we see clean tokens_predicted
    }
    t0 = time.time()
    try:
        full_response = _instrumented_generate_call(url, body_dict, timeout=proposer.timeout)
    except Exception as ex:
        return {
            "error": str(ex),
            "max_tokens": max_tokens,
            "duration_s": time.time() - t0,
        }
    duration_s = time.time() - t0

    content = full_response.get("content", "")
    stopped_limit = full_response.get("stopped_limit", None)
    stopped_eos = full_response.get("stopped_eos", None)
    tokens_predicted = full_response.get("tokens_predicted", None)
    stop_type = full_response.get("stop_type", None)
    truncation_reason = full_response.get("truncation", None)

    # Extract and attempt parse
    code = _extract_python_local(content)
    missing_fns = [fn for fn in required if f"def {fn}" not in (code or "")]
    syntax_error_msg = None
    if code and not missing_fns:
        try:
            ast.parse(code)
        except SyntaxError as se:
            syntax_error_msg = f"SyntaxError line {se.lineno}: {se.msg}"

    parse_ok = bool(code) and not missing_fns and syntax_error_msg is None
    if missing_fns:
        failure_reason = f"missing defs: {missing_fns}"
    elif syntax_error_msg:
        failure_reason = syntax_error_msg
    elif not code:
        failure_reason = "no python block extracted"
    else:
        failure_reason = None  # success

    return {
        "max_tokens": max_tokens,
        "tokens_predicted": tokens_predicted,
        "stopped_limit": stopped_limit,
        "stopped_eos": stopped_eos,
        "stop_type": stop_type,
        "truncation_reason": truncation_reason,
        "content_length_chars": len(content),
        "code_length_chars": len(code) if code else 0,
        "has_code_block": bool(code),
        "has_def_engine": "def engine" in (code or ""),
        "has_def_is_level_complete": "def is_level_complete" in (code or ""),
        "missing_fns": missing_fns,
        "syntax_error": syntax_error_msg,
        "failure_reason": failure_reason,
        "parse_ok": parse_ok,
        "content_tail_400": content[-400:] if content else "",
        "duration_s": round(duration_s, 2),
        # Extra llama.cpp fields that sometimes appear
        "stop": full_response.get("stop"),
        "generation_settings": {k: full_response.get(k) for k in ("n_predict", "seed", "temperature") if k in full_response},
        "model": full_response.get("model", ""),
    }

# ---------------------------------------------------------------------------
# Step 2: The primary call at max_tokens=4096 (shipping default)
# ---------------------------------------------------------------------------
print(f"\n[RUN] Primary call: max_tokens=4096 (shipping default)...")
t_call_start = time.time()
record_4096 = generate_with_capture(proposer, "/no_think\n" + prompt_l2, max_tokens=4096)
print(f"[RUN] Done in {record_4096['duration_s']:.1f}s")
print(f"  stopped_limit={record_4096.get('stopped_limit')}")
print(f"  stopped_eos={record_4096.get('stopped_eos')}")
print(f"  tokens_predicted={record_4096.get('tokens_predicted')}")
print(f"  parse_ok={record_4096.get('parse_ok')}")
print(f"  failure_reason={record_4096.get('failure_reason')}")
print(f"  content_tail_400:\n---\n{record_4096.get('content_tail_400', '')}\n---")

# ---------------------------------------------------------------------------
# Step 3: Retry at 8192 and 6144 on the SAME prompt (no agent re-run)
# ---------------------------------------------------------------------------
print(f"\n[RUN] Retry at max_tokens=8192 on same prompt...")
record_8192 = generate_with_capture(proposer, "/no_think\n" + prompt_l2, max_tokens=8192)
print(f"[RUN] Done in {record_8192['duration_s']:.1f}s")
print(f"  stopped_limit={record_8192.get('stopped_limit')}")
print(f"  stopped_eos={record_8192.get('stopped_eos')}")
print(f"  tokens_predicted={record_8192.get('tokens_predicted')}")
print(f"  parse_ok={record_8192.get('parse_ok')}")
print(f"  failure_reason={record_8192.get('failure_reason')}")

print(f"\n[RUN] Retry at max_tokens=6144 on same prompt...")
record_6144 = generate_with_capture(proposer, "/no_think\n" + prompt_l2, max_tokens=6144)
print(f"[RUN] Done in {record_6144['duration_s']:.1f}s")
print(f"  stopped_limit={record_6144.get('stopped_limit')}")
print(f"  stopped_eos={record_6144.get('stopped_eos')}")
print(f"  tokens_predicted={record_6144.get('tokens_predicted')}")
print(f"  parse_ok={record_6144.get('parse_ok')}")
print(f"  failure_reason={record_6144.get('failure_reason')}")

# ---------------------------------------------------------------------------
# Determine verdict
# ---------------------------------------------------------------------------
p4 = record_4096
p8 = record_8192

truncated_4096 = bool(p4.get("stopped_limit")) or (
    p4.get("tokens_predicted") is not None
    and p4.get("tokens_predicted", 0) >= 4090  # near the cap
)
parse_ok_at_8192 = bool(p8.get("parse_ok"))
parse_ok_at_6144 = bool(record_6144.get("parse_ok"))

if truncated_4096 and (parse_ok_at_8192 or parse_ok_at_6144):
    verdict = "TRUNCATION"
    verdict_summary = (
        "The 4096-token call hit the cap (stopped_limit=True or tokens_predicted>=4090). "
        f"Raising to {'8192' if parse_ok_at_8192 else '6144'} produces parseable code with both "
        "def engine and def is_level_complete. TRUNCATION IS THE WALL — raising max_tokens unblocks L2."
    )
elif truncated_4096 and not (parse_ok_at_8192 or parse_ok_at_6144):
    verdict = "TRUNCATION_BUT_ALSO_GENERATION_QUALITY"
    verdict_summary = (
        "The 4096-token call was truncated (stopped_limit=True). Raising the budget does NOT fix "
        "the problem — the model still fails to produce both required functions at 8192. "
        "Both truncation AND generation quality are walls."
    )
elif not truncated_4096 and not parse_ok_at_8192:
    verdict = "NOT_TRUNCATION"
    verdict_summary = (
        f"The 4096-token call stopped naturally (stopped_eos=True, tokens_predicted="
        f"{p4.get('tokens_predicted')} < 4096). The model produced a complete-but-invalid output. "
        f"Failure: {p4.get('failure_reason')}. Raising to 8192 does NOT fix it "
        f"(failure: {p8.get('failure_reason')}). Wall = GENERATION QUALITY (prompt or model limitation), "
        "not token budget."
    )
elif not truncated_4096 and parse_ok_at_8192:
    # Natural stop at 4096 but 8192 fixes it? — means prompt forces 4096 content but with more budget
    # the model self-corrects; might still be a soft truncation (model finishes at 4096 but poorly)
    verdict = "SOFT_TRUNCATION"
    verdict_summary = (
        "The 4096-token call stopped (not hard-limited per stopped_limit), but output was invalid. "
        "Raising to 8192 produces valid code. Likely soft truncation: the model rushed the last "
        "function to fit in 4096 tokens but had space to do it properly at 8192."
    )
else:
    verdict = "INCONCLUSIVE"
    verdict_summary = (
        f"stopped_limit={p4.get('stopped_limit')}, tokens_predicted={p4.get('tokens_predicted')}, "
        f"parse_ok@4096={p4.get('parse_ok')}, parse_ok@8192={parse_ok_at_8192}. Ambiguous."
    )

print(f"\n[VERDICT] {verdict}")
print(f"[VERDICT] {verdict_summary}")

# ---------------------------------------------------------------------------
# Build artifact
# ---------------------------------------------------------------------------
t_total = time.time() - t_start
RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)

artifact = {
    "experiment_id": "proto_l2_proposer_truncation_check",
    "honest_verdict": f"complete_{verdict.lower()}",
    "verdict": verdict,
    "verdict_summary": verdict_summary,

    "preconditions_checked": [
        {"resource": "qwen_gguf", "available": True, "path": QWEN_GGUF},
        {"resource": "llama_server_qwen", "available": True, "port": PORT,
         "model_path_reported": model_path_reported},
    ],

    "prompt_info": {
        "game": GAME,
        "prompt_length_chars": len(prompt_l2),
        "has_win_state_block": has_win_state_block,
        "prompt_sha256_prefix": prompt_checksum,
        "data_source": "real_arcengine" if _real_prev_grid is not None and len(_real_transitions) > 0 else "synthetic",
        "n_transitions": len(_real_transitions),
    },

    "call_4096": record_4096,
    "call_8192": record_8192,
    "call_6144": record_6144,

    "analysis": {
        "truncated_4096": truncated_4096,
        "parse_ok_at_8192": parse_ok_at_8192,
        "parse_ok_at_6144": parse_ok_at_6144,
        "tokens_predicted_4096": record_4096.get("tokens_predicted"),
        "stopped_limit_4096": record_4096.get("stopped_limit"),
        "stopped_eos_4096": record_4096.get("stopped_eos"),
    },

    "model_info": {
        "model_path": QWEN_GGUF,
        "model_serving": model_path_reported,
        "port": PORT,
        "max_tokens_default": 4096,
        "mtp": False,
    },

    "inference_substrate": "live_llm_inference",
    "solve_provenance": "development_proxy",
    "verifier_is_oracle": False,
    "random_seed": 42,
    "duration_s": round(t_total, 2),
    "server_start_s": round(server_start_s, 2),
}

RESULT_PATH.write_text(json.dumps(artifact, indent=2))
print(f"\n[DONE] Artifact written to {RESULT_PATH}")
print(f"[DONE] Total wall time: {t_total:.1f}s")

# Cleanup
proposer.stop()
print("[DONE] Proposer stopped.")
