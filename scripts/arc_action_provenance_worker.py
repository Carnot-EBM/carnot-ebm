#!/usr/bin/env python3
"""ONE arm of the action-provenance measurement, in its own killable process.

**Researcher summary:** runs the SCORED live agent (`E3AgentPolicy`, reached through
`arc_actions_to_progress.run_bounded_progress` -- the same driver
`scripts/arc_holdout_generalization_probe.py` uses) against the OFFLINE arcade for one
game/seed, with the per-action provenance instrument either armed or not, and writes the
resulting action trace + provenance rows to a JSON file.

**Why a separate process.** Inducing a world model loads and EXECUTES LLM-authored engine
code (`results/arc_e3/<game>/world_model.py`). Repo rule: never run induced engine code in
the analysing interpreter. One arm per subprocess also means an arm that hangs or segfaults
costs that arm and nothing else, and it guarantees the two arms of the byte-identity
comparison share no in-process state -- module globals, RNG state, import-time caches --
which is precisely what a same-process comparison could not guarantee.

**This never plays a scored or online game.** `arc_solver_kit.offline_arcade()` runs
`OperationMode.OFFLINE` against the local `environment_files/` tree: no API key, no
network, no scorecard submission.

Spec: openspec/capabilities/arc-world-model-trust-energy/spec.md REQ-ARC-WMTE-6070
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(
    0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python")
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--max-inductions", type=int, default=3)
    ap.add_argument("--wall-s", type=float, default=900.0)
    ap.add_argument("--explore-budget", type=int, default=24)
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--generator",
        choices=("none", "live"),
        default="none",
        help=(
            "'none' = no LLM is constructed; run with CARNOT_ARC_DISABLE_INDUCTION=1 so the "
            "cascade is fully deterministic (the configuration the byte-identity proof needs). "
            "'live' = build the frozen live-submission LocalGGUFProposer, i.e. the real scored "
            "induce->verify->plan path, at the cost of sampling nondeterminism."
        ),
    )
    ap.add_argument("--cuda-port", type=int, default=8951)
    ap.add_argument(
        "--arm-label",
        default="",
        help="free-text label for this arm, echoed into the output for auditability",
    )
    args = ap.parse_args()

    from carnot.agentic import arc_actions_to_progress as atp
    from carnot.agentic.arc_action_provenance import provenance_enabled
    from carnot.agentic.arc_executable_world_model import E3_DIR, _TRACKED_E3_EVIDENCE_DIR

    # EVIDENCE GUARD. A live induction writes `<E3_DIR>/<game>/world_model.py`, and the
    # default `E3_DIR` is `results/arc_e3` -- TRACKED, READ-ONLY evidence. The module-level
    # guard that would catch this is scoped to pytest (correctly: the live agent writing
    # there is the store's purpose), so a measurement driver is exactly the case nothing
    # protects. Checked here, in the child, AFTER import, because `E3_DIR` is resolved at
    # import time and this therefore reads what the run will really use rather than what the
    # environment asked for.
    if E3_DIR.resolve() == _TRACKED_E3_EVIDENCE_DIR.resolve():
        print(
            "[worker] REFUSING: E3_DIR resolves to the tracked evidence store "
            f"({_TRACKED_E3_EVIDENCE_DIR}). Set CARNOT_ARC_E3_DIR to a scratch directory "
            "in the child's environment BEFORE the interpreter starts.",
            file=sys.stderr,
        )
        return 2

    armed = provenance_enabled()
    t0 = time.time()

    generator_witness: dict = {}
    proposer: object
    if args.generator == "live":
        from carnot.agentic.arc_executable_world_model import (
            LocalGGUFProposer,
            _generator_server_and_env,
        )

        # THE SUBSTRATE WITNESS, taken BEFORE the run. `_generator_server_and_env` silently
        # falls back to the AMD iGPU HIP build when the requested CUDA card has no headroom
        # (or is not visible, which is what happens if CUDA_VISIBLE_DEVICES has already
        # remapped it away). That fallback is not an error -- it produces a real,
        # slow, working generator -- so nothing downstream would have flagged it, and an
        # artifact claiming "the live scored generator on a 3090" would have been describing
        # a run that never touched one. Observed on the first attempt at this measurement,
        # which is why the witness exists rather than being assumed.
        server, launch_env = _generator_server_and_env()
        generator_witness = {
            "server_binary": str(server),
            "is_cuda_build": "build-hip" not in str(server),
            "launch_env_cuda_visible_devices": (launch_env or {}).get("CUDA_VISIBLE_DEVICES"),
            "requested_cuda_gpu": os.environ.get("CARNOT_ARC_GENERATOR_CUDA_GPU"),
            "ambient_cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        }
        if not generator_witness["is_cuda_build"]:
            out = {
                "arm_label": args.arm_label,
                "game": args.game,
                "seed": args.seed,
                "error": "generator_fell_back_to_igpu_hip_build_refusing_to_run",
                "generator_witness": generator_witness,
                "action_trace": [],
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump(out, fh, indent=1, default=str)
            print("[worker] BLOCKED:", out["error"], generator_witness)
            return 1

        # The FROZEN live-submission generator, taken from the module constants rather than
        # retyped, so this cannot drift away from what the submission actually ships (the
        # 2026-07-28 Qwen-9B -> gemma-4-31B switch is the reason those constants exist).
        from carnot.agentic.arc_competition_agent import (
            ARC_LIVE_GENERATOR_MTP_DEFAULT,
            ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
            ARC_LIVE_GENERATOR_REPO_SUBSTR,
        )

        proposer = LocalGGUFProposer(
            repo_substr=ARC_LIVE_GENERATOR_REPO_SUBSTR,
            mtp=(ARC_LIVE_GENERATOR_MTP_DEFAULT != "0"),
            kv_quant="q8_0",
            no_think_prefix=ARC_LIVE_GENERATOR_NO_THINK_PREFIX,
            max_tokens=4096,
            timeout=600,
            port=args.cuda_port,
        )
    else:
        # No LLM at all. `apply_arm` writes proposer config attributes unconditionally, so a
        # bare None crashes it; this stand-in absorbs those writes and constructs nothing.
        # Paired with CARNOT_ARC_DISABLE_INDUCTION=1 (set by the driver), the induce tier
        # short-circuits before any proposer method is called, so the stand-in is never
        # asked to do anything -- which is the point: a sampled token stream between two
        # arms would make an inertness claim unfalsifiable.
        class _NoGeneratorStandIn:
            include_playbook_exemplars = False
            no_think_prefix = ""
            max_tokens = 0
            tries = 0

        proposer = _NoGeneratorStandIn()

    result = atp.run_bounded_progress(
        args.game,
        "frozen",
        proposer=proposer,
        seed=args.seed,
        budget=args.budget,
        max_inductions=args.max_inductions,
        wall_s=args.wall_s,
        explore_budget=args.explore_budget,
    )

    out = {
        "arm_label": args.arm_label,
        "game": args.game,
        "seed": args.seed,
        "provenance_armed": armed,
        "generator": args.generator,
        "generator_witness": generator_witness,
        "induction_disabled_env": os.environ.get("CARNOT_ARC_DISABLE_INDUCTION"),
        "budget": args.budget,
        "max_inductions": args.max_inductions,
        "explore_budget": args.explore_budget,
        "wall_s_measured": round(time.time() - t0, 3),
        # THE byte-identity instrument: the ordered action trace in the canonical
        # `_action_label` encoding. Two arms are identical iff these lists are equal.
        "action_trace": list(result.action_trace),
        "result_row": result.to_row(include_events=True, include_trace=False),
        "provenance": None,
    }

    # The rows live on the policy, and `run_bounded_progress` constructs that policy
    # internally and returns only a metrics dataclass. Rather than edit that shared driver
    # to hand the policy back, the recorder registers itself at construction time and is
    # retrieved here. In a real scored run this is unnecessary -- `CarnotAgent.cleanup`
    # flushes the recorder off its own policy -- so the registry exists solely for
    # measurement drivers like this one, and is only ever populated when the flag is set.
    from carnot.agentic.arc_action_provenance import last_recorder

    rec = last_recorder()
    if rec is not None:
        out["provenance"] = rec.to_dict()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)
    print(
        f"[worker] arm={args.arm_label!r} armed={armed} actions={len(out['action_trace'])} "
        f"levels_gained={result.levels_gained} inductions={result.n_inductions} "
        f"plans={result.n_plans_found} err={result.error}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
