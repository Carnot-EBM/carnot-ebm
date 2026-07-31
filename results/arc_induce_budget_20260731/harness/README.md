# The harness, committed so the measurement can be re-run rather than re-derived

Order of operations, and what each piece is for:

1. `capture_prompts.py` — re-runs ft09's LLM-OFF arm with a proposer that RECORDS its `induce`
   arguments instead of answering them, and rebuilds the three induce prompts byte-exactly
   (code-only directive and opened fence included). ~6s, no GPU. Its validity was checked, not
   assumed: the capture's action trace is element-by-element identical to the banked live LLM-ON
   cell (60/60, sha `fba0baa5d5473eac`).
2. `tokcount.py` / `arithmetic.py` — token counts through the generator's own vocabulary
   (`vocab_only=True` on the `.gguf` path, never `AutoTokenizer` on a GGUF repo id), and the
   `n_ctx`/`max_tokens`/VRAM tables computed from the repo's own constants so they cannot go stale.
3. `budget_sweep.py` — the budget axis on the two induce calls. `refactor_sweep.py` — the same
   axis on the refactor call, rebuilding its prompt from the banked round-1 counterexample exactly
   as `_counterexample_result()` does. `sampler_sweep.py` — the sampler control (`off` /
   `repeat_penalty 1.1` / `dry 0.8`) at one fixed budget, all three arms inside ONE script on ONE
   server so the comparison never crosses a server process.
4. `chain.sh` — runs the follow-on lanes only AFTER the budget lanes finish. Concurrency is not an
   optimisation here: at the pinned `n_ctx=32768` a second in-flight request halves the per-slot
   allowance to below `prompt + 16384`, which would silently pool-truncate the 16384 tier and make
   it measure the harness instead of the lever.
5. `repetition.py` — `loop_frac` (consecutive identical lines) and `cycle_frac` (lines repeated
   >=5 times anywhere), because an ABAB cycle is the same failure as an AAAA run and the first
   metric alone reads 0.19 on a completion that is 1055 lines drawn from 67 distinct ones.
6. `score_engines.py` — runs every generated engine against the 25 real captured transitions.
   This is the load-bearing one: the AST "returns on every path" check is gameable and an identity
   engine gamed it. Cross-validated against the shipped gate to the cell (216/228 -> 0.9474,
   matching the live `onb` cell's `verify_correct_changed_cells` / `verify_cell_recall`).
7. `analyse.py` / `build_artifact.py` / `fill_note.py` — the tables, the artifact, and the
   research note's numbers. `fill_note.py` GENERATES the note's tables from the artifact
   rather than letting them be hand-typed, so the prose and the artifact cannot come to
   state different numbers — the same failure the freshness lint guards, one document out.

Every lane pins `n_ctx=32768`, `ffn_cpu_layers=0`, MTP off, `CUDA_VISIBLE_DEVICES=""` on the
parent, a non-default port, and PROVES the CUDA build from `/proc/<pid>/exe` plus a per-PID VRAM
row before generating a single token — at `n_ctx=81920` the 31B silently falls through to the iGPU
HIP build and then runs LLM-OFF while still reporting LLM-ON.

`ft09_transitions.pkl` (the 25 captured transitions) is NOT committed — it is regenerable in ~6s
from `capture_prompts.py`'s path, and pickled `Transition` objects would pin the artifact to a
class layout that is free to change.
