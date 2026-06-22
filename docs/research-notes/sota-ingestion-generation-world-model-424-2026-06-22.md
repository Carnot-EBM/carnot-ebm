# SOTA ingestion 2026-06-22: candidate generation and world-model induction for .425

```json
{
  "citations_verified": {
    "2502.00225": {
      "http_status": 200,
      "title": "Should You Use Your Large Language Model to Explore or Exploit?",
      "url": "https://arxiv.org/abs/2502.00225"
    },
    "2502.13200": {
      "http_status": 200,
      "title": "Learning To Explore With Predictive World Model Via Self-Supervised Learning",
      "url": "https://arxiv.org/abs/2502.13200"
    },
    "2505.19095": {
      "http_status": 200,
      "title": "ScreenExplorer: Training a Vision-Language Model for Diverse Exploration in Open GUI World",
      "url": "https://arxiv.org/abs/2505.19095"
    },
    "2507.12821": {
      "http_status": 200,
      "title": "Assessing Adaptive World Models in Machines with Novel Games",
      "url": "https://arxiv.org/abs/2507.12821"
    },
    "2510.04542": {
      "http_status": 200,
      "title": "Code World Models for General Game Playing",
      "url": "https://arxiv.org/abs/2510.04542"
    },
    "2510.12088": {
      "http_status": 200,
      "title": "One Life to Learn: Inferring Symbolic World Models for Stochastic Environments from Unguided Exploration",
      "url": "https://arxiv.org/abs/2510.12088"
    },
    "2603.17683": {
      "http_status": 200,
      "title": "Sensi: Learn One Thing at a Time -- Curriculum-Based Test-Time Learning for LLM Game Agents",
      "url": "https://arxiv.org/abs/2603.17683"
    },
    "2603.24621": {
      "http_status": 200,
      "title": "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence",
      "url": "https://arxiv.org/abs/2603.24621"
    },
    "2605.05138": {
      "http_status": 200,
      "title": "Executable World Models for ARC-AGI-3 in the Era of Coding Agents",
      "url": "https://arxiv.org/abs/2605.05138"
    },
    "2605.08083": {
      "http_status": 200,
      "title": "LLMs Improving LLMs: Agentic Discovery for Test-Time Scaling",
      "url": "https://arxiv.org/abs/2605.08083"
    },
    "2605.10999": {
      "http_status": 200,
      "title": "SkillGen: Verified Inference-Time Agent Skill Synthesis",
      "url": "https://arxiv.org/abs/2605.10999"
    },
    "2605.16986": {
      "http_status": 200,
      "title": "Skills on the Fly: Test-Time Adaptive Skill Synthesis for LLM Agents",
      "url": "https://arxiv.org/abs/2605.16986"
    }
  },
  "field_principles": {
    "citations_verified": "every method claim cites a verifiable arXiv ID/URL -- the no-fabrication bar (same as any results artifact).",
    "field_principles": "principle annotations for every top-level artifact field.",
    "flagged_for_next_roadmap": "the strongest method flagged as a .425 candidate -- closes the discover->ingest->plan loop.",
    "honest_verdict": "terminal prefix; complete: sota_ingestion_generation_mapped.",
    "inference_substrate": "aggregation_from_upstream_artifacts -- literature synthesis, no model load (100us floor).",
    "methods_mapped": "the 3-5 strongest GENERATION/world-model-induction methods with REAL arXiv IDs -- the shoulders-of-giants anti-rederivation check.",
    "preconditions_checked": "records resources verified; pre-empts missing-resource fabrication.",
    "random_seed": "bare integer seed for reproducible artifact generation.",
    "research_note_path": "repo-relative markdown path for deterministic parsing."
  },
  "flagged_for_next_roadmap": "flagged_for_v425: executable_world_model_energy_config_space_generation_prior (arXiv:2605.05138 + arXiv:2510.04542)",
  "honest_verdict": "complete: sota_ingestion_generation_mapped",
  "inference_substrate": "aggregation_from_upstream_artifacts",
  "methods_mapped": [
    {
      "fails_when": "the visible-state parser is wrong, the transition verifier accepts a near-identity or overfit model, the private-set harness exposes a leakage assumption, or the plan is scored only by the win oracle.",
      "generation_track": "executable_world_model_induction",
      "method": "Executable code world models plus verified planning",
      "source_ids": [
        "2605.05138",
        "2510.04542",
        "2603.24621"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4592 showed wiring can make one extra winner appear, but the toolkit still mostly emits no winning candidate. Executable World Models and Code World Models take over Exp 4592 by generating an explicit Python transition model, verifying it against observed transitions, and planning through it. They take over Exp 4594 by using objective energy as the trust/goal prior that selects and repairs model candidates before action generation.",
      "v425_candidate": "flagged_for_v425: executable_world_model_energy_config_space_generation_prior (arXiv:2605.05138 + arXiv:2510.04542)"
    },
    {
      "fails_when": "the LLM reads raw grid text incorrectly. Sensi v2 reports zero solved levels despite high sample efficiency because the bottleneck moved to perceptual grounding, which exactly matches Carnot's generation-not-ranking diagnosis.",
      "generation_track": "curriculum_perception_grounding",
      "method": "Sensi curriculum test-time learning with perception-gated generation",
      "source_ids": [
        "2603.17683",
        "2603.24621"
      ],
      "takes_over_current_a1_a3_mechanisms": "Sensi maps onto Exp 4592 as the warning and control for the LLM tail generator: split perception from action, advance through a small curriculum, and measure whether the agent can read the grid before asking it to generate a plan. For Exp 4594, the objective energy should gate curriculum advancement and reject perception-incoherent states rather than only rank final plans.",
      "v425_candidate": "flagged_for_v425: sensi_perception_gate_for_llm_tail_generator"
    },
    {
      "fails_when": "the synthesized skill is only prose, the skill is not executed against matched with/without controls, failed trajectories are omitted, or the controller is tuned on seen public games without hidden-style variant checks.",
      "generation_track": "skill_controller_synthesis",
      "method": "Verified inference-time skill and controller synthesis",
      "source_ids": [
        "2605.10999",
        "2605.16986",
        "2605.08083"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4592 currently wires known toolkit skills, but unseen mechanics need new candidate procedures. SkillGen, Skills on the Fly, and AutoTTS take over by synthesizing a temporary skill or controller from successful and failed trajectories, then verifying its net effect. Exp 4594's energy prior becomes the fitness signal for repairs, regressions, and controller pruning.",
      "v425_candidate": "flagged_for_v425: verified_skill_synthesis_over_arc_solver_kit_failures"
    },
    {
      "fails_when": "the action semantics are not grounded in visible objects, the oracle is treated as the exploiter instead of a candidate-set generator, or curiosity rewards no-op diversity rather than goal-relevant state changes.",
      "generation_track": "exploration_oracle_curiosity",
      "method": "LLM exploration oracle plus predictive-world-model curiosity",
      "source_ids": [
        "2502.00225",
        "2502.13200",
        "2505.19095"
      ],
      "takes_over_current_a1_a3_mechanisms": "Exp 4592 needs candidate action sets that are larger than a fixed router pool but smaller than blind search. The exploration-oracle pattern asks a model or heuristic to propose semantically plausible actions, then lets cheap search and environment feedback dispose of them. Exp 4594's energy prior becomes curiosity/novelty and predicted progress energy, not a terminal reranker.",
      "v425_candidate": "flagged_for_v425: semantic_action_set_generator_plus_energy_search"
    },
    {
      "fails_when": "the exploration trace is too short to identify hidden registers, the symbolic vocabulary cannot represent the mechanic, or the energy prior penalizes uncertainty so strongly that it avoids informative experiments.",
      "generation_track": "adaptive_symbolic_world_model",
      "method": "Adaptive symbolic world-model induction from novel-game exploration",
      "source_ids": [
        "2507.12821",
        "2510.12088"
      ],
      "takes_over_current_a1_a3_mechanisms": "The novel-games and One Life lines take over Exp 4592 by treating first contact as rapid symbolic world-model induction from unguided exploration rather than selection from an existing pool. They take over Exp 4594 by making objective energy an epistemic prior: prefer candidate models and action probes that explain more transitions and reduce uncertainty about goal-relevant dynamics.",
      "v425_candidate": "flagged_for_v425: symbolic_world_model_induction_with_epistemic_energy_prior"
    }
  ],
  "preconditions_checked": {
    "agents_md_read": true,
    "arxiv_api_reachable": true,
    "arxiv_http_200_verified_ids": [
      "2510.04542",
      "2507.12821",
      "2510.12088",
      "2502.13200",
      "2505.19095",
      "2603.17683",
      "2502.00225",
      "2605.05138",
      "2605.10999",
      "2603.24621",
      "2605.16986",
      "2605.08083"
    ],
    "codex_md_read": true,
    "deep_research_invoked": false,
    "exp4592_artifact_read": true,
    "exp4594_artifact_read": true,
    "leaderboard_submission": false,
    "live_llm_inference": false,
    "live_solve_claim": false,
    "ops_docs_modified": false,
    "planner_confirmation_addendum_filtered": true,
    "research_conductor_modified": false,
    "research_references_424_filtered": true,
    "research_studying_filtered": true,
    "research_studying_updated": true,
    "sweep_clusters_help_exit_0": true,
    "sweep_clusters_urls": [
      "http://export.arxiv.org/api/query?search_query=(abs:\"affordance\"+OR+abs:\"action+effect\"+OR+abs:\"clickability\"+OR+abs:\"frame+prediction\"+OR+abs:\"intrinsic+motivation\"+OR+abs:\"directed+exploration\"+OR+abs:\"novelty+search\")+AND+(abs:\"reinforcement+learning\"+OR+abs:\"agent\"+OR+abs:\"exploration\"+OR+abs:\"interactive+environment\"+OR+abs:\"ARC\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending",
      "http://export.arxiv.org/api/query?search_query=(abs:\"neural+guided+search\"+OR+abs:\"learned+heuristic\"+OR+abs:\"value+guided+search\"+OR+abs:\"program+induction\"+OR+abs:\"world+model\"+OR+abs:\"goal+induction\")+AND+(abs:\"planning\"+OR+abs:\"agent\"+OR+abs:\"reasoning\"+OR+abs:\"reinforcement+learning\")&start=0&max_results=8&sortBy=submittedDate&sortOrder=descending"
    ],
    "sweep_clusters_used": true,
    "sweep_semscholar_arxiv_ids": [],
    "sweep_semscholar_failed_queries": [
      "world model induction novel games exploration oracle predictive world model ScreenExplorer"
    ],
    "sweep_semscholar_queries": [
      "ARC-AGI-3 executable world models Code World Models Sensi SkillGen candidate generation",
      "world model induction novel games exploration oracle predictive world model ScreenExplorer"
    ],
    "sweep_semscholar_rate_limited_queries": [
      "ARC-AGI-3 executable world models Code World Models Sensi SkillGen candidate generation"
    ],
    "sweep_semscholar_used": true,
    "training_launched": false,
    "websearch_webfetch_top_sources": [
      "https://arxiv.org/abs/2605.05138",
      "https://arxiv.org/abs/2510.04542",
      "https://arxiv.org/abs/2603.17683",
      "https://arxiv.org/abs/2605.10999",
      "https://arxiv.org/abs/2502.00225",
      "https://arxiv.org/abs/2507.12821",
      "https://arxiv.org/abs/2510.12088",
      "https://arxiv.org/abs/2603.24621"
    ]
  },
  "random_seed": 4601,
  "research_note_path": "docs/research-notes/sota-ingestion-generation-world-model-424-2026-06-22.md"
}
```

## Fresh-pass provenance

Read `AGENTS.md`, `CODEX.md`, the `.424` sweep and planner-confirmation
addendum in `research-references.md`, `research-studying.md`, Exp 4592
(`results/experiment_4592_generation_completeness_wiring.json`), and Exp 4594
(`results/experiment_4594_goal_energy_generation_prior.json`). The filtered
track was candidate generation, first-contact world-model induction,
perceptual grounding, verified skill/controller synthesis, exploration oracles,
and objective energy as a generation prior.

Reliable-channel pass, not `/deep-research`:
- `.venv/bin/python scripts/sweep_clusters.py --help`
- `curl -sf -o /dev/null "https://export.arxiv.org/api/query?search_query=all:test"`
- `.venv/bin/python scripts/sweep_clusters.py 5 --max-results 8`
- `.venv/bin/python scripts/sweep_clusters.py 6 --max-results 8`
- `.venv/bin/python scripts/sweep_semscholar.py "ARC-AGI-3 executable world models Code World Models Sensi SkillGen candidate generation" --limit 8`
- `.venv/bin/python scripts/sweep_semscholar.py "world model induction novel games exploration oracle predictive world model ScreenExplorer" --limit 8`

Cluster helper URLs were emitted for exploration/affordance and
world-model/goal-induction tracks. Semantic Scholar returned HTTP 429 on the
focused ARC/CWM/Sensi/SkillGen query and HTTP 500 on the broader world-model
exploration query, so no S2-only paper is promoted. Low-concurrency
WebSearch/WebFetch plus direct arXiv HTTP-200 checks verified arXiv:2510.04542,
arXiv:2507.12821, arXiv:2510.12088, arXiv:2502.13200, arXiv:2505.19095,
arXiv:2603.17683, arXiv:2502.00225, arXiv:2605.05138, arXiv:2605.10999,
arXiv:2603.24621, arXiv:2605.16986, and arXiv:2605.08083.

No training, No live LLM inference, No leaderboard submission, and no live solve
claim were run or made. `ops/changelog.md`, `ops/status.md`,
`_bmad/traceability.md`, and `scripts/research_conductor.py` were not edited by
this workflow.

## Exp 4592 A1 and Exp 4594 A3 status

Exp 4592 is the current A1 wiring reference: `winner_generated=2/25`, up from
the 1/25 baseline. That is a real but small crack in the generation wall; most
held-out variants still do not get a winning candidate.

Exp 4594 is the current A3 objective-energy reference:
`complete: goal_energy_prior_no_value_honest_null_gap_sharpened`. The current
goal-energy prior did not lift winner generation, so the next use of objective
energy should be inside a stronger generator: trust a world model, select a
candidate skill/controller, or drive exploration toward informative states.

## SOTA -> experiment mapping

## Executable world models

**Sources:** Executable World Models, arXiv:2605.05138; Code World Models,
arXiv:2510.04542; ARC-AGI-3 report, arXiv:2603.24621.

**Mapping to Exp 4592/4594:** use the A1 wiring harness to run a generated
Python world model and planner, not just a pre-existing skill route. Use A3
objective energy as model-trust, goal-progress, and repair energy while the
candidate is being generated. This is the strongest .425 candidate because it
directly attacks "winner absent from the pool."

**Failure mode:** the visible-state parser or transition verifier can accept a
near-identity or overfit model; Sensi shows that bad perception can make a
sample-efficient system generate nothing useful.

## Sensi curriculum and perception gate

**Sources:** Sensi, arXiv:2603.17683; ARC-AGI-3 report, arXiv:2603.24621.

**Mapping to Exp 4592/4594:** Sensi is decision-grade negative evidence for
raw LLM-on-grid generation: its v2 curriculum reached high sample efficiency
but solved zero levels because the bottleneck moved to perceptual grounding.
Use it as a .425 diagnostic gate: before the LLM tail generator can plan, it
must pass an object-centric grid-reading check, and A3 energy should reject
perception-incoherent states. This is the perceptual-grounding wall in one line.

**Failure mode:** an LLM can become self-consistent about a wrong grid reading,
making the generated candidate set precise, cheap, and still wrong.

## Verified skill and controller synthesis

**Sources:** SkillGen, arXiv:2605.10999; Skills on the Fly, arXiv:2605.16986;
AutoTTS, arXiv:2605.08083.

**Mapping to Exp 4592/4594:** A1 can synthesize a temporary skill/controller
from successful and failed trajectories when the static toolkit has no winning
route. A3 energy should be the measured with/without fitness signal: repairs,
regressions, and controller-pruning decisions must be execution checked.

**Failure mode:** a prose skill that is not executed and ablated is just another
prompt. It can overfit public games or hide regressions on mechanics the static
toolkit already solved.

## Exploration oracle and predictive curiosity

**Sources:** Should You Use Your LLM to Explore or Exploit, arXiv:2502.00225;
Learning To Explore With Predictive World Model, arXiv:2502.13200;
ScreenExplorer, arXiv:2505.19095.

**Mapping to Exp 4592/4594:** let an LLM or structured heuristic generate a
small semantic action set, then let cheap search and environment feedback test
it. Objective energy becomes curiosity, novelty, and predicted progress during
candidate generation, not a final score on a fixed pool.

**Failure mode:** if the proposed action set is not grounded in visible objects,
the oracle narrows the search in the wrong direction. If curiosity rewards
frame churn or no-op diversity, it can worsen action efficiency.

## Adaptive symbolic world-model induction

**Sources:** Assessing Adaptive World Models in Machines with Novel Games,
arXiv:2507.12821; One Life to Learn, arXiv:2510.12088.

**Mapping to Exp 4592/4594:** treat a new ARC game as a rapid symbolic
world-model-induction task from unguided exploration. A3 energy becomes an
epistemic prior: choose probes and candidate models that explain observed
transitions and reduce uncertainty about goal-relevant dynamics.

**Failure mode:** the trace may be too short, hidden registers may be outside
the symbolic vocabulary, or the energy prior may punish uncertainty so hard that
the agent avoids informative probes.

## Flagged for .425

flagged_for_v425: executable_world_model_energy_config_space_generation_prior
(arXiv:2605.05138 + arXiv:2510.04542).

Bottom line: run executable world-model induction as the .425 candidate
generator. Keep Sensi as the perceptual-grounding guard, use SkillGen-style
synthesis for residual toolkit gaps, and use objective energy as the
trust/goal/repair prior inside generation.

