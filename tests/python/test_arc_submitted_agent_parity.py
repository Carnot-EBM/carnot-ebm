"""Parity guard for WHAT SHIPS — prevents the 2026-06-19 "0.08 incident" from recurring.

Incident: the offline eval measured STRONGER opt-in configs (explorer_bf unlocked cn04) while the
SUBMITTED default (make_carnot_agent(Agent)) shipped bare BFS, and nobody caught it because "better" was
opt-in-only and the headline metric was banked-replay levels, not the submitted path. These tests assert
the shipped agent matches the single-source-of-truth SUBMITTED_AGENT_CONFIG, and that the "wired" flags
reflect REALITY — so a silent divergence between what we measure and what we ship fails CI.
"""

import importlib.util
import inspect
import json
from pathlib import Path
import re

import pytest

from carnot import experiment_4551_offline_live_proposer_parity as exp4551
from carnot.agentic import arc_competition_agent as m
from carnot.agentic.arc_competition_agent import (
    E3AgentPolicy,
    SUBMITTED_AGENT_CONFIG,
    StepwiseExplorer,
    make_carnot_agent,
)
from carnot.agentic.arc_executable_world_model import (
    ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT,
    LocalGGUFProposer,
)
from carnot.agentic.arc_llm_strategy_proposer import SGECandidateRouter

REPO = Path(__file__).resolve().parents[2]
SPEC_PATH = REPO / "openspec" / "capabilities" / "arc-world-model-trust-energy" / "spec.md"
_GATE = REPO / "scripts" / "kaggle" / "arc_local_submission_gate.py"
_spec = importlib.util.spec_from_file_location("arc_local_submission_gate_parity", _GATE)
gate = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(gate)


def _imports(module_name: str, src: str) -> bool:
    """True only if `module_name` is actually IMPORTED (a from/import statement), not merely mentioned
    in a comment or docstring -- the TODO prose names these modules without importing them."""
    return bool(re.search(rf"^\s*(from|import)\s+[\w.]*{re.escape(module_name)}\b", src, re.M))


def test_submission_defaults_to_e3_cascade_not_banked_replay():
    # the submission is make_carnot_agent(Agent) with NO cascade arg -> must default to the generic
    # E3AgentPolicy cascade, NEVER the cascade=False banked-replay ("useless on the hidden eval").
    assert inspect.signature(make_carnot_agent).parameters["cascade"].default is True
    assert SUBMITTED_AGENT_CONFIG["cascade"] is True
    assert SUBMITTED_AGENT_CONFIG["policy"] == "E3AgentPolicy"


def test_shipped_explorer_config_matches_single_source_of_truth():
    """REQ-CAPSTONE-4605: shipped explorer config is the declared integrated default."""
    # the live explorer config must equal the declared SUBMITTED_AGENT_CONFIG; any silent change to the
    # E3AgentPolicy/StepwiseExplorer defaults fails here until SUBMITTED_AGENT_CONFIG is consciously updated.
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    exp = pol.explorer
    assert exp.value_weight == SUBMITTED_AGENT_CONFIG["value_weight"]
    assert exp.target_levels == SUBMITTED_AGENT_CONFIG["target_levels"]
    assert exp.search_mode == SUBMITTED_AGENT_CONFIG["search_mode"]
    assert exp.frontier_batch_size == SUBMITTED_AGENT_CONFIG["frontier_batch_size"]
    assert exp.navigation_cost_tiebreak == SUBMITTED_AGENT_CONFIG["navigation_cost_tiebreak"]
    # REQ-LEARN-4652: the component-labeling cost fix makes a bounded positive value route affordable.
    assert exp.value_weight == m.SUBMITTED_VALUE_WEIGHT
    assert 0.0 < exp.value_weight <= 1e-9
    assert SUBMITTED_AGENT_CONFIG["value_head_feature_subset"] == (
        "cross_game_features_v3:v2_plus_frame_delta"
    )
    assert exp.target_levels > 1
    assert exp.candidate_router is not None
    assert exp.frame_change_scorer is not None
    assert exp.action_effect_expansion_prior is not None
    assert SUBMITTED_AGENT_CONFIG["frame_change_predictor_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["frame_change_ranking_mode"] == "persistent_aem_plus_optional_cnn"
    assert SUBMITTED_AGENT_CONFIG["action_effect_expansion_prior_enabled"] is True
    assert (
        SUBMITTED_AGENT_CONFIG["action_effect_expansion_prior_mode"]
        == "persistent_aem_plus_optional_cnn_frontier_prior"
    )
    assert SUBMITTED_AGENT_CONFIG["discriminative_candidate_router_enabled"] is True
    assert exp.goal_bias is not None
    assert exp.goal_bias_label == "exp4020_graded_goal_satisfaction_energy"
    assert SUBMITTED_AGENT_CONFIG["goal_energy_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["goal_energy_source"] == "exp4020_graded_goal_satisfaction_energy"
    assert SUBMITTED_AGENT_CONFIG["goal_energy_alpha"] == 0.9
    assert SUBMITTED_AGENT_CONFIG["goal_energy_beta"] == 0.1
    assert SUBMITTED_AGENT_CONFIG["verifier_is_oracle"] is False
    assert pol.subgoal_search == SUBMITTED_AGENT_CONFIG["hierarchical_subgoal_search_enabled"]
    assert pol.subgoal_budget == SUBMITTED_AGENT_CONFIG["hierarchical_subgoal_budget"]
    assert pol.factored_planner == SUBMITTED_AGENT_CONFIG["factored_planner_enabled"]
    assert pol.factored_trust_threshold == SUBMITTED_AGENT_CONFIG["factored_trust_threshold"]
    assert (exp.controllable_novelty_policy is not None) == SUBMITTED_AGENT_CONFIG[
        "controllable_novelty_proposal_enabled"
    ]
    assert (
        exp.controllable_novelty_diagnostics()["enabled"]
        == SUBMITTED_AGENT_CONFIG["controllable_novelty_proposal_enabled"]
    )
    assert (
        pol.program_synthesis_filter_enabled
        == SUBMITTED_AGENT_CONFIG["program_synthesis_proposal_filter_enabled"]
    )
    assert (
        pol.program_synthesis_filter_trust_threshold
        == SUBMITTED_AGENT_CONFIG["program_synthesis_proposal_filter_trust_threshold"]
    )
    assert (
        exp.program_synthesis_filter_diagnostics()["enabled"]
        == SUBMITTED_AGENT_CONFIG["program_synthesis_proposal_filter_enabled"]
    )
    # REQ-ARC-WMTE-5836: the just-explore frontier-discipline mechanisms are wired but must ship
    # OFF, so the submitted agent's search order is unchanged until the A/B greenlights a flip.
    # FLIPPED ON 2026-07-25 (operator decision). These assertions previously pinned the flags OFF
    # and existed to force a conscious update before any flip -- which is what happened. Evidence:
    # results/experiment_5836_frontier_definitive.json, arm B2 mean 10.56 wins vs baseline 7.00 with
    # strict PER-SEED dominance in 8 of 9 condition x seed cells, stable under colour permutation and
    # reflection. The DISTANCE GRADIENT stays OFF (arm D was worse on both axes), so it is still
    # pinned False below. See the flag block in arc_competition_agent.py for the full rationale and
    # for the honest limit (public-game evidence; the positive control was confounded).
    assert SUBMITTED_AGENT_CONFIG["frontier_tier_exhaustion"] is True
    assert SUBMITTED_AGENT_CONFIG["frontier_tier_uniform_random"] is True
    assert SUBMITTED_AGENT_CONFIG["frontier_tier_click_vocab_only"] is True
    assert SUBMITTED_AGENT_CONFIG["frontier_distance_gradient"] is False
    assert exp.tier_exhaustion_enabled == SUBMITTED_AGENT_CONFIG["frontier_tier_exhaustion"]
    assert exp.tier_uniform_random_enabled == SUBMITTED_AGENT_CONFIG["frontier_tier_uniform_random"]
    assert exp.frontier_gradient_enabled == SUBMITTED_AGENT_CONFIG["frontier_distance_gradient"]
    assert exp.tier_count == SUBMITTED_AGENT_CONFIG["frontier_tier_count"]
    fd = exp.frontier_discipline_diagnostics()
    assert fd["tier_exhaustion_enabled"] is True  # flipped 2026-07-25, see above
    assert fd["frontier_gradient_enabled"] is False
    assert fd["active_tier"] == 0


def test_req_capstone_4605_live_stack_integrates_only_non_regression_levers():
    """REQ-CAPSTONE-4605/REQ-LEARN-4652: router/deepening ship with bounded value routing."""
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    exp = pol.explorer

    assert SUBMITTED_AGENT_CONFIG["target_levels"] > 1
    assert exp.target_levels > 1
    assert SUBMITTED_AGENT_CONFIG["value_weight"] == m.SUBMITTED_VALUE_WEIGHT
    assert exp.value_weight == m.SUBMITTED_VALUE_WEIGHT
    assert 0.0 < exp.value_weight <= 1e-9
    assert exp.frame_change_scorer is not None
    assert exp.frame_change_prune_threshold is None
    assert exp.action_effect_expansion_prior is not None
    assert exp.goal_bias is not None
    assert SUBMITTED_AGENT_CONFIG["frame_change_predictor_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["frame_change_prune_threshold"] is None
    assert SUBMITTED_AGENT_CONFIG["action_effect_expansion_prior_enabled"] is True
    # 2026-07-14 (submission-prep pre-flight incident, REQ-ARC-FCP-5591-3): disabled after a
    # real near-hang was found on the local submission gate (7/8 canonical games timed out) --
    # root cause (O(candidates x grid_cells) per-frame recomputation) is fixed, but the flag
    # stays off pending a fresh matched-budget A/B, since three follow-on live-path attempts
    # using it this same day all returned honest_null (zero measured benefit to offset the
    # residual per-step cost even post-fix).
    assert exp.action_prior is None
    assert exp.action_prior_prune_quantile is None
    assert SUBMITTED_AGENT_CONFIG["color_blob_salience_enabled"] is False
    assert (
        SUBMITTED_AGENT_CONFIG["color_blob_salience_mode"]
        == "single_color_connected_component_tiers"
    )
    assert SUBMITTED_AGENT_CONFIG["strategy_router_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["discriminative_candidate_router_enabled"] is True
    assert SUBMITTED_AGENT_CONFIG["explore_diversity_default"] is False


def test_req_capstone_4597_submitted_config_points_at_refreshed_package():
    """REQ-ARC-WMTE-4645: submitted metadata names the refreshed live package."""

    assert (
        SUBMITTED_AGENT_CONFIG["live_submit_package_path"]
        == "results/experiment_4643_submission_package_operator_resubmit.json"
    )
    assert (
        SUBMITTED_AGENT_CONFIG["live_submit_source"] == "experiment_4643_refresh_submission_package"
    )


def test_req_capstone_4744_submitted_config_declares_frozen_qwen_generator():
    """REQ-CAPSTONE-4744: submitted config declares the pinned generator package stack.

    RE-PINNED 2026-07-28 (operator directive: "We must use Gemma-4-31B and stop using Qwen-3.5-9B
    and Qwen-3.6-27B"). The 9B pin existed solely to fit an assumed 16GB Kaggle VRAM ceiling the
    operator has now declared void; a 13-game x 3-replicate head-to-head measured gemma-4-31B at
    0.3843 fail-as-zero vs 0.0627, matched tally 11-0-2, sign p=0.00098.

    THE FUNCTION NAME KEEPS "qwen" ON PURPOSE. Renaming it would silently orphan the
    REQ-CAPSTONE-4744 traceability that `scripts/check_spec_coverage.py` and the historical
    artifacts follow, and this repo's rule is to correct the record rather than rewrite it. What
    the test ASSERTS has moved; what it is CALLED is a historical identifier.

    mtp/spec_type flip to False/None because gemma-4-31B-it is dense and declares no MTP heads --
    leaving them on emits `--spec-type draft-mtp --model-draft <same 18.3GB file>`, i.e. the
    weights loaded twice. no_think_prefix empties because /no_think is a Qwen3 control token.

    ^^^ THE mtp SENTENCE ABOVE IS SUPERSEDED (2026-07-28, same day, measured); preserved per
    never-prune, and its warning about `--model-draft <the main weights>` is still exactly right.
    The premise is not: gemma-4-31B-it DOES have MTP, via a SEPARATE 491 MiB head
    (`mtp-gemma-4-31B-it-Q8_0.gguf`, arch `gemma4-assistant`) rather than heads embedded in the
    main GGUF -- which is why none were found there. Enabling it loads the head (+1290 MiB at the
    shipped n_ctx 81920), not a second copy of the weights, and buys a measured 1.398x decode on
    the binary this submission bundles.

    So this config -- which describes the SCORED launch -- now tracks
    `ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT`, NOT the local one. The two are correctly different: a
    24 GB dev card must offload ~14 FFN blocks to host the MTP-on server and that costs more decode
    than MTP returns, while the 96 GB scored card needs no offload at all.
    """

    frozen = SUBMITTED_AGENT_CONFIG["frozen_generator"]

    # RE-PINNED AGAIN 2026-07-31 (separate, later, quant-only repin -- the 2026-07-28 repin above
    # was a MODEL-FAMILY change, Qwen -> gemma-4-31B-it; this one is a QUANTIZATION-VARIANT change
    # on top of that, Q4_K_M -> qat). A 20-game head-to-head found qat and Q4_K_M statistically
    # indistinguishable on quality (sign test p=1.0); qat saves ~1GB VRAM and matches the required
    # qat MTP drafter. See tests/python/test_arc_live_generator_pin.py for the full pin contract.
    assert frozen["model_id"] == "unsloth/gemma-4-31B-it-qat-GGUF"
    assert frozen["repo_substr"] == "gemma-4-31B-it-qat"
    # ASSERT THE LITERAL SCORED CONTRACT, NEVER `frozen["mtp"] is <the same expression>`.
    #
    # An intermediate 2026-07-28 version of this test read
    # `scored_on = ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"` and then asserted
    # `frozen["mtp"] is scored_on`. But `frozen["mtp"]` in `arc_competition_agent.py` IS literally
    # `ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT != "0"` -- so that assertion was `X is X`, true for
    # every possible value of the constant. Verified by mutation: flipping the constant to "0" left
    # this test GREEN while genuinely changing what the submission launches. Comparing a config
    # field against the same expression that produced it tests the equals sign, not the contract.
    #
    # The scored contract is a FACT about the submission, decided by the operator directive of
    # 2026-07-28 ("when we submit we will want MTP enabled for speed when running on the Kaggle 96G
    # GPU hardware") plus the measured 1.398x decode win on the bundled binary. So it is written
    # here as a literal. If a future operator directive turns scored MTP off, this line is SUPPOSED
    # to go red -- that is the notification, and it is what the tautology suppressed.
    assert frozen["mtp"] is True
    assert frozen["spec_type"] == "draft-mtp"
    # Separately, pin that the constant the config is DERIVED from still says the same thing. This
    # is the "did somebody flip the knob" check; the two lines above are the "what does the
    # submission actually launch with" check. Keeping them apart is what makes either one able to
    # fail on its own.
    assert ARC_LIVE_GENERATOR_MTP_SCORED_DEFAULT == "1"
    # The draft is a DIFFERENT FILE from the weights. Asserting only "mtp is True" is what would
    # let `--model-draft` drift back onto the main GGUF -- accepted by llama.cpp, warned about,
    # then served with speculation silently disabled and no other observable difference.
    assert frozen["mtp_head_filename"] != frozen["model_filename"]
    assert frozen["kv_quant"] == "q8_0"
    assert frozen["max_tokens"] >= frozen["n_predict_min"] >= 2048
    assert frozen["no_think_prefix"] == ""
    assert frozen["llama_server_kind"] == "cuda-12.8-binary"
    assert frozen["binary_not_wheel"] is True
    assert frozen["wheel_fallback_allowed"] is False
    assert frozen["port_strategy"] == "free_non_8919"
    assert frozen["props_verify_endpoint"] == "/props"
    assert "libllama-common" in frozen["required_shared_libraries"]
    assert frozen["forbidden_models"] == ["gemma-8919"]
    assert frozen["forbidden_gpu_targets"] == ["3090"]


def test_wired_flags_reflect_actual_imports():
    """REQ-CAPSTONE-4605: shipped config declares real router/verifier/DSL imports."""
    # router_wired / world_model_dsl_wired must match whether the modules are ACTUALLY referenced in the
    # submission module -- catches BOTH "wired the module but left the flag stale" AND "flag claims wired
    # but the import is missing". This is the exact gap that shipped bare BFS at 0.08.
    src = inspect.getsource(m)
    assert SUBMITTED_AGENT_CONFIG["router_wired"] == _imports("arc_strategy_router", src), (
        "router_wired flag disagrees with whether arc_strategy_router is imported in the submission path"
    )
    assert SUBMITTED_AGENT_CONFIG["solve_learning_router_wired"] == _imports(
        "arc_solve_learning", src
    ), "solve_learning_router_wired flag disagrees with whether arc_solve_learning is imported"
    assert SUBMITTED_AGENT_CONFIG["discriminative_router_wired"] == _imports(
        "arc_discriminative_router", src
    ), (
        "discriminative_router_wired flag disagrees with whether arc_discriminative_router is imported"
    )
    assert SUBMITTED_AGENT_CONFIG["world_model_dsl_wired"] == _imports(
        "arc_world_model_dsl", src
    ), "world_model_dsl_wired flag disagrees with whether arc_world_model_dsl is imported"
    assert SUBMITTED_AGENT_CONFIG["goal_energy_wired"] == _imports("arc_goal_energy_live", src), (
        "goal_energy_wired flag disagrees with whether arc_goal_energy_live is imported"
    )
    # REQ-ARC-FCP-5699-11: SGE is genuinely live-path-reachable (a local import inside
    # _load_sge_candidate_router still matches _imports()'s ^\s*(from|import) regex, same
    # convention as arc_executable_world_model's own local imports throughout this file) --
    # confirmed independently via scripts/arc_orphan_solver_lint.py passing clean.
    assert SUBMITTED_AGENT_CONFIG["sge_candidate_router_wired"] == _imports(
        "arc_llm_strategy_proposer", src
    ), (
        "sge_candidate_router_wired flag disagrees with whether arc_llm_strategy_proposer is imported"
    )


def test_req_arc_fcp_5699_11_sge_candidate_router_disabled_by_default():
    """REQ-ARC-FCP-5699-11: SGE is wired but NOT the active default -- constructing a policy
    normally still yields the discriminative router (or None), never SGECandidateRouter,
    matching the SUBMITTED_COLOR_BLOB_SALIENCE_ENABLED precedent (built + reachable, gated
    off pending a real matched-budget win on the live path)."""
    assert m.SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED is False
    assert SUBMITTED_AGENT_CONFIG["sge_candidate_router_enabled"] is False
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    assert not isinstance(pol.explorer.candidate_router, SGECandidateRouter)


def test_req_arc_fcp_5699_12_env_var_enables_sge_without_touching_module_flag(monkeypatch):
    """REQ-ARC-FCP-5699-12: CARNOT_ARC_SGE_CANDIDATE_ROUTER=1 lets a subprocess-based
    measurement (the local submission gate) opt into SGE for a single run WITHOUT
    flipping the committed SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED default."""
    assert m._sge_candidate_router_requested() is False
    monkeypatch.setenv("CARNOT_ARC_SGE_CANDIDATE_ROUTER", "1")
    assert m._sge_candidate_router_requested() is True
    assert m.SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED is False  # module default untouched
    router = m._load_submitted_candidate_router(game_id="sp80")
    assert isinstance(router, SGECandidateRouter)
    assert router.game_id == "sp80"


def test_req_arc_fcp_5699_11_load_sge_candidate_router_reuses_frozen_generator_config(monkeypatch):
    """_load_sge_candidate_router() must build a LocalGGUFProposer configured IDENTICALLY
    to _proposer()'s own lazy default (same repo_substr/mtp/kv_quant/no_think_prefix) so it
    shares the SAME warm llama-server via port-based reuse -- never a second model load.

    The stakes ROSE on 2026-07-28: the generator moved from a 5.9GB Qwen3.5-9B Q4 to an 18.3GB
    gemma-4-31B-it Q4, so a config divergence here no longer merely wastes VRAM -- the second load
    does not fit any card this project runs on. (The original wording said "the Kaggle 16GB VRAM
    budget"; that ceiling is the very thing the operator declared void when re-pinning the model,
    so the reason is restated rather than left describing a constraint that no longer exists.)

    THE ENV IS EXPLICITLY CLEARED, for exactly the reason the sibling
    `test_req_arc_fcp_5699_35_*` below already documents -- that test was hardened against this and
    this one was left exposed, so it kept the bug. This test asserts the LAZY DEFAULT config, and
    since 2026-07-28 `timeout` is DERIVED (`_default_induce_timeout_s()`) and scales with the FFN
    offload, which itself auto-fits to a real card whenever `CARNOT_ARC_GENERATOR_CUDA_GPU` is set.
    Under the full suite an unrelated test's `monkeypatch.setenv` for that variable, plus a LIVE
    free-VRAM reading from a real 3090, produced `timeout == 866` here while the same test passed
    in isolation. Reproduced deterministically with `CARNOT_ARC_GENERATOR_CUDA_GPU=0`. A test whose
    result depends on how much VRAM another process happens to be holding is measuring the machine,
    not the code.
    """
    for var in (
        "CARNOT_ARC_FFN_CPU_LAYERS",
        "CARNOT_ARC_GENERATOR_CUDA_GPU",
        "CARNOT_ARC_INDUCE_MAX_TOKENS",
        "CARNOT_ARC_INDUCE_N_CTX",
        "CARNOT_ARC_INDUCE_TIMEOUT",
        "CARNOT_ARC_MTP",
    ):
        monkeypatch.delenv(var, raising=False)
    router = m._load_sge_candidate_router("g50t")
    assert isinstance(router, SGECandidateRouter)
    assert router.game_id == "g50t"
    assert router.k == 3
    completer = router.proposer.completer
    assert isinstance(completer, LocalGGUFProposer)
    # RE-PINNED 2026-07-31 (quant-only repin, see test_req_capstone_4744_... above for the full
    # rationale): ARC_LIVE_GENERATOR_REPO_SUBSTR moved Q4_K_M -> qat.
    assert completer.repo_substr == "gemma-4-31B-it-qat"
    # False because that is the LOCAL default (`ARC_LIVE_GENERATOR_MTP_DEFAULT`), NOT because the
    # model lacks MTP -- the old comment here said "gemma-4-31B has no MTP heads", which was
    # falsified the same day: the head is a SEPARATE 491 MiB GGUF (`mtp-gemma-4-31B-it-Q8_0.gguf`,
    # arch `gemma4-assistant`), which is why none was found inside the main file. Local stays off
    # because MTP-on needs ~14 offloaded FFN layers on a 24 GB card and that costs more decode than
    # MTP returns; the SCORED default is "1". Right value, wrong reason -- so the reason is fixed.
    assert completer.mtp is False
    assert completer.kv_quant == "q8_0"
    assert completer.no_think_prefix == ""  # /no_think is a Qwen3 token
    assert completer.max_tokens == 4096  # REQ-ARC-FCP-5699-35: graduated default, was 2560
    assert (
        completer.timeout == 600
    )  # REQ-ARC-FCP-5699-35: graduated default, was the 300 field default
    assert completer.port == 8919  # the DEFAULT port, same as _proposer()'s own default


def test_req_arc_fcp_5699_35_proposer_default_max_tokens_and_timeout_graduated(monkeypatch):
    """Neither env var set (production default) -- _proposer()'s lazy default now constructs
    with the REQ-ARC-FCP-5699-32-validated budget (max_tokens=4096, timeout=600), graduated
    from the REQ-ARC-FCP-5699-28 dev-only override (was 2560/300, the pre-32 default).

    THE ENV IS NOW EXPLICITLY CLEARED, and that is a real fix rather than tidying. This test
    always CLAIMED "neither env var set" and never ensured it, so it silently measured whatever
    the ambient environment happened to be. Since 2026-07-28 that matters: `timeout` is DERIVED
    (`_default_induce_timeout_s()`) and scales with the FFN offload, and the offload itself
    auto-fits to a card when `CARNOT_ARC_GENERATOR_CUDA_GPU` is set. Under the full suite an
    unrelated test's `monkeypatch.setenv("CARNOT_ARC_GENERATOR_CUDA_GPU", ...)` plus a live
    free-VRAM reading produced `timeout == 1016` here, while the same test passed in isolation --
    a failure that says nothing about the code under test.

    600 remains the correct assertion for the SCORED path: Kaggle sets `CARNOT_LLAMA_SERVER` and
    no CUDA opt-in, so the offload is 0, the slowdown factor is 1.0, and the derived timeout
    returns its floor -- the same 600 the literal used to hardcode.
    """
    monkeypatch.delenv("CARNOT_ARC_INDUCE_MAX_TOKENS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_INDUCE_TIMEOUT", raising=False)
    monkeypatch.delenv("CARNOT_ARC_FFN_CPU_LAYERS", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    proposer = pol._proposer()
    assert isinstance(proposer, LocalGGUFProposer)
    assert proposer.max_tokens == 4096
    assert proposer.timeout == 600


def test_the_derived_induce_timeout_grows_with_the_ffn_offload_that_slows_generation(
    monkeypatch,
) -> None:
    """The other half of the derived-timeout contract, and the reason it exists.

    The 600 s literal was calibrated for Qwen3.5-9B. The 2026-07-28 gemma-4-31B head-to-head's
    slowest of 39 real induce calls took 572.0 s -- 4.7% inside that limit, single-stream, at
    n_ctx 32768, with NO FFN offload. The live path runs n_ctx 81920 across 4 shared slots, and
    the local card now needs an FFN offload that costs 2.4x decode throughput at 12 layers. A
    fixed 600 s would turn the fix for one silent-LLM-off failure into the cause of another,
    because an induce timeout does not raise -- it returns (False, msg) and the agent proceeds
    LLM-OFF while still reporting itself LLM-on.
    """
    from carnot.agentic import arc_executable_world_model as wm

    monkeypatch.delenv("CARNOT_ARC_INDUCE_TIMEOUT", raising=False)
    monkeypatch.delenv("CARNOT_ARC_GENERATOR_CUDA_GPU", raising=False)

    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "0")
    assert wm._default_induce_timeout_s() == 600, "the scored/Kaggle path must be unchanged"

    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "12")
    at12 = wm._default_induce_timeout_s()
    monkeypatch.setenv("CARNOT_ARC_FFN_CPU_LAYERS", "24")
    at24 = wm._default_induce_timeout_s()
    assert 600 < at12 < at24, (at12, at24)
    # It must clear the observed worst case with room, not merely exceed the old floor.
    assert at12 > wm._INDUCE_OBSERVED_MAX_WALL_S

    # The explicit env override still wins outright -- this is a default, not a policy.
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TIMEOUT", "42")
    assert wm._default_induce_timeout_s() == 42


def test_req_arc_fcp_5699_28_proposer_max_tokens_and_timeout_env_override(monkeypatch):
    """CARNOT_ARC_INDUCE_MAX_TOKENS/CARNOT_ARC_INDUCE_TIMEOUT still let a diagnostic run
    override the (now-graduated) default to different values -- the override mechanism REQ-28
    introduced survives REQ-35's default-value graduation unchanged."""
    monkeypatch.setenv("CARNOT_ARC_INDUCE_MAX_TOKENS", "8192")
    monkeypatch.setenv("CARNOT_ARC_INDUCE_TIMEOUT", "900")
    pol = E3AgentPolicy("paritytest", proposer=None, value_head=lambda _frame: 0.0)
    proposer = pol._proposer()
    assert proposer.max_tokens == 8192
    assert proposer.timeout == 900


def test_req_arc_fcp_5699_11_load_submitted_candidate_router_uses_sge_when_enabled(monkeypatch):
    """When the flag is on, _load_submitted_candidate_router() returns the SGE router,
    correctly threaded with the CURRENT game's id (not a placeholder)."""
    monkeypatch.setattr(m, "SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED", True)
    router = m._load_submitted_candidate_router(game_id="sk48")
    assert isinstance(router, SGECandidateRouter)
    assert router.game_id == "sk48"


def test_req_arc_fcp_5699_11_load_submitted_candidate_router_falls_back_on_sge_failure(
    monkeypatch,
):
    """SGE construction failing for any reason must NEVER take down the live path --
    _load_submitted_candidate_router() falls through to the discriminative router, exactly
    like the pre-existing bare except Exception: return None pattern for that router."""
    monkeypatch.setattr(m, "SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED", True)

    def _boom(_game_id):
        raise RuntimeError("simulated SGE construction failure")

    monkeypatch.setattr(m, "_load_sge_candidate_router", _boom)
    router = m._load_submitted_candidate_router(game_id="cd82")
    assert not isinstance(router, SGECandidateRouter)


def test_e3_policy_builds_strategy_route_and_world_model_dsl():
    """SCENARIO-CAPSTONE-4605: E3 first contact has router + DSL state."""
    pol = E3AgentPolicy("tn36", proposer=None, value_head=lambda _frame: 0.0)
    assert pol.strategy_route["game"] == "tn36"
    assert pol.strategy_route["name"] == "program_editor"
    assert pol.strategy_route["uses_goal_distance_heuristic"] is False
    assert pol.approach_recommendation["strategy"] == pol.strategy_route
    assert pol.dsl_model.game_id == "tn36"
    assert pol.explore_budget < SUBMITTED_AGENT_CONFIG["graph_explore_budget"]


def test_stepwise_explorer_prefers_forward_shortest_path_over_reset():
    """SCENARIO-REPORT-4475-LIVE-STACK-FORWARD-NAV: forward edges beat RESET replay."""
    exp = StepwiseExplorer()
    exp.root = "A"
    exp.cur = "A"
    exp.start_level = 0
    exp.best_level = 0
    exp.graph = {
        "A": {"path": [], "untested": [], "value": 0.0},
        "B": {
            "path": [{"action": 7, "data": None}],
            "untested": [{"action": 2, "data": {"x": 1, "y": 2}}],
            "value": 0.0,
        },
    }
    exp.adj = {"A": [({"action": 7, "data": None}, "B")]}

    assert exp.next_move([], None) == (7, None)
    assert exp.next_move([], None) == (2, {"x": 1, "y": 2})
    assert exp.awaiting["origin"] == "B"
    assert exp.awaiting["action"] == 2
    assert exp.awaiting["data"] == {"x": 1, "y": 2}


def test_req_arc_wmte_4551_spec_declares_offline_live_proposer_parity_guard():
    """REQ-ARC-WMTE-4551: OpenSpec declares the offline/live proposer parity guard."""

    spec = SPEC_PATH.read_text(encoding="utf-8")

    assert "REQ-ARC-WMTE-4551" in spec
    assert "SCENARIO-ARC-WMTE-4551-PROPOSER-PARITY" in spec
    assert exp4551.RESULT_RELATIVE_PATH in spec
    for field, principle in exp4551.FIELD_PRINCIPLES.items():
        assert field in spec
        assert principle["principle"] in spec


def test_req_arc_wmte_4551_disabled_offline_induction_mismatch_fires():
    """REQ-ARC-WMTE-4551: disabled offline induction is flagged against submitted E3."""

    report = gate.proposer_config_parity_report(
        offline_config=gate.offline_gate_proposer_config(
            policy="e3",
            disable_induction=True,
        ),
        submitted_config=gate.submitted_agent_proposer_config(SUBMITTED_AGENT_CONFIG),
    )

    assert report["proposer_config_mismatch"] is True
    assert report["parity_guard"] == "offline_live_proposer_config_parity"
    assert any(
        item["field"] == "induction_enabled"
        and item["offline"] is False
        and item["submitted"] is True
        and "CARNOT_ARC_DISABLE_INDUCTION=1" in item["detail"]
        for item in report["proposer_config_divergence"]
    )

    annotated = gate.attach_proposer_config_parity(
        {"policy": "e3", "core_efficiency": 0.419},
        policy="e3",
        disable_induction=True,
        submitted_agent_config=SUBMITTED_AGENT_CONFIG,
    )
    assert annotated["proposer_config_mismatch"] is True
    assert annotated["core_efficiency"] == 0.419
    assert annotated["proposer_config_parity"]["offline_config"]["lower_bound_note"] == (
        "offline_core_efficiency_is_lower_bound_when_mismatch_true"
    )


def test_scenario_arc_wmte_4551_matched_proposer_config_passes_clean():
    """SCENARIO-ARC-WMTE-4551-PROPOSER-PARITY: matched live-proposer config is clean."""

    report = gate.proposer_config_parity_report(
        offline_config=gate.offline_gate_proposer_config(
            policy="e3",
            disable_induction=False,
        ),
        submitted_config=gate.submitted_agent_proposer_config(SUBMITTED_AGENT_CONFIG),
    )

    assert report["proposer_config_mismatch"] is False
    assert report["proposer_config_divergence"] == []
    assert report["offline_config"]["proposer_kind"] == "LocalGGUFProposer"
    assert report["submitted_config"]["proposer_kind"] == "LocalGGUFProposer"


def test_scenario_arc_wmte_4551_artifact_records_fixture_results(tmp_path: Path):
    """SCENARIO-ARC-WMTE-4551-PROPOSER-PARITY: artifact records both guard fixtures."""

    artifact = exp4551.run(
        root=tmp_path,
        preconditions_checked={
            "arc_competition_agent_import": True,
            "arc_local_submission_gate_present": True,
            "spec_has_req_4551": True,
            "research_conductor_modified": False,
            "ok": True,
        },
        write=True,
    )

    assert artifact["honest_verdict"] == ("shipped: offline_live_proposer_parity_guard_added")
    assert artifact["inference_substrate"] == exp4551.INFERENCE_SUBSTRATE
    assert artifact["proposer_config_mismatch_detected"] is True
    assert (
        artifact["fixture_results"]["disabled_induction_mismatch"]["proposer_config_mismatch"]
        is True
    )
    assert artifact["fixture_results"]["matched_config_clean"]["proposer_config_mismatch"] is False
    assert artifact["tests_added_pass"]["passed"] is True
    assert exp4551.validate_artifact(artifact) == []

    written = json.loads((tmp_path / exp4551.RESULT_RELATIVE_PATH).read_text(encoding="utf-8"))
    assert written["reproducibility_checksum"] == artifact["reproducibility_checksum"]


def test_req_arc_fcp_5699_11_spec_declares_sge_live_path_wiring() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-11") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-11",
        "SCENARIO-ARC-FCP-5699-11-SGE-REACHABLE-BUT-NOT-DEFAULT",
        "SUBMITTED_SGE_CANDIDATE_ROUTER_ENABLED",
        "_load_sge_candidate_router",
        "sge_candidate_router_wired",
    ):
        assert marker in section


def test_req_arc_fcp_5699_12_spec_declares_real_live_path_ab() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-12") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-12",
        "SCENARIO-ARC-FCP-5699-12-REAL-LIVE-PATH-AB-CONFIRMS-NO-WIN-AT-REAL-COST",
        "arc_sge_live_path_ab.py",
        "levels=0, reached=L0",
        "3.9x",
    ):
        assert marker in section


def test_req_arc_fcp_5699_13_spec_declares_generator_scope_corrigendum() -> None:
    spec_path = REPO / "openspec" / "capabilities" / "arc-human-replay-frame-change" / "spec.md"
    spec = spec_path.read_text(encoding="utf-8")
    section = spec[spec.index("### REQ-ARC-FCP-5699-13") : spec.index("### REQ-ARC-WMTE-5596")]

    for marker in (
        "REQ-ARC-FCP-5699-13",
        "SCENARIO-ARC-FCP-5699-13-DIAGNOSTIC-HARNESS-FINDINGS-DO-NOT-AUTOMATICALLY-TRANSFER-TO-PRODUCTION",
        "ActionDiverseLiveGenerator(max_candidates=8)",
        "SUBMITTED_QD_GENERATION_ENABLED",
        "rich_action_candidates",
    ):
        assert marker in section
