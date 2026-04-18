# research-roadmap vNEXT: DSPy-style typed signatures as the experiment prompt contract

Status: Draft change proposal. Author: research-conductor maintainer. Target milestone: 2026.05.x (pre-Exp-470 planning cycle).

## 1. Problem

- **Prompt drift across milestones.** Every planner-generated task in `research-roadmap.yaml` invents its own CONTEXT / EXISTING CODE TO READ FIRST / TASK layout. The wording is close but never identical, so subagents pattern-match inconsistently (some read CLAUDE.md, some skip it; some call `apply_env_autofix()` FIRST, some don't).
- **No enforceable schema contract.** The YAML `prompt:` is a free-form string. The *expected deliverable* (JSON fields, file path, honest_verdict vocabulary) is embedded in prose. The conductor has no way to validate "did this experiment actually emit the fields the task promised?" short of a human eye-balling the JSON.
- **Implicit coupling between prompt and deliverable.** Changing a required artifact field (e.g. adding `schema=` version tags) means hand-editing 12+ prompt strings and hoping none were missed. `scripts/experiment_template.py::REQUIRED_RESULT_FIELDS` is the closest we have to a schema, but it isn't referenced from the prompts.
- **Subagent output is ungraded.** When Claude Code finishes, there is no structured "did you satisfy the contract" check — only a pytest pass and a JSON file. Missing fields are caught (if at all) by downstream experiments that try to read them.
- **No reuse.** Every milestone re-derives boilerplate: workflow step list, CLAUDE.md reminder, do-not-push-do-not-modify-conductor footer. 40-80 lines of copy-paste per task.

## 2. Proposed design

Adopt DSPy `Signature` as the *data model* for an experiment task. The conductor renders the signature into a Claude Code prompt — DSPy is used as a typed-schema + prompt-serialiser library, not as a runtime for the subagent (the subagent is still Claude Code CLI; see §4).

### 2.1 Python-level sketch

```python
# python/carnot/autoresearch/experiment_signature.py
from typing import Literal
import dspy

class ExperimentSignature(dspy.Signature):
    """Execute one Carnot autoresearch experiment.

    Strategy (applies to every experiment unless overridden):
      1. Read CLAUDE.md; follow spec-first workflow.
      2. Read every path in `existing_code_to_read` before writing code.
      3. Add REQ-*/SCENARIO-* listed in `spec_requirements` to the named capability spec.
      4. Write failing tests first; target 100% coverage on new code.
      5. Write the deliverable script, producing an artifact at `deliverable_path`
         whose JSON conforms to `required_schema_fields` and `schema_version`.
      6. Run the test command; reconcile ops/*.md.
      7. Do NOT push. Do NOT modify scripts/research_conductor.py.
    """

    # ---- identity ----
    exp_id: str           = dspy.InputField(desc="Canonical id, e.g. exp451-live-precision-postfix")
    milestone: str        = dspy.InputField(desc="Milestone tag, e.g. 2026.04.34")
    title: str            = dspy.InputField()

    # ---- deliverable contract ----
    deliverable_path: str = dspy.InputField(desc="Artifact path relative to project root")
    schema_version: str   = dspy.InputField(desc="e.g. 'carnot.live_precision.v2'")
    required_schema_fields: list[str] = dspy.InputField(
        desc="Top-level JSON keys the artifact MUST contain")
    honest_verdict_vocab: list[str]   = dspy.InputField(
        desc="Allowed values for artifact.honest_verdict")

    # ---- spec anchors ----
    capability: str                    = dspy.InputField(desc="openspec capability name")
    spec_requirements: list[str]       = dspy.InputField(desc="REQ-* ids to add/touch")
    spec_scenarios:    list[str]       = dspy.InputField(desc="SCENARIO-* ids to add/touch")

    # ---- environment / preconditions ----
    requires_gpu: bool                 = dspy.InputField()
    gpu_layout: list[dict]             = dspy.InputField(desc="[{name,hf_id,gpu}]")
    env_prelude: list[str]             = dspy.InputField(
        desc="Mandatory calls, e.g. ['apply_env_autofix()']")
    blocked_by: list[str]              = dspy.InputField(desc="Prior exp_ids that must be success")

    # ---- reading list ----
    existing_code_to_read: list[str]   = dspy.InputField()
    baseline_artifact: str | None      = dspy.InputField(desc="Prior artifact this beats")

    # ---- free-form task narrative (the only prose field) ----
    task_narrative: str                = dspy.InputField(
        desc="Experiment-specific steps that don't fit the generic strategy")

    # ---- outputs (for trace/grading, not for the Claude prompt render) ----
    artifact_json: dict                = dspy.OutputField()
    test_report:   str                 = dspy.OutputField()
    honest_verdict: Literal["success","no_improvement","no_improvement_v2",
                            "first_positive","deferred_to_gpu","blocked"] = dspy.OutputField()
```

### 2.2 How `research_conductor.run_research_step()` changes

```python
def run_research_step(task: dict):
    sig = ExperimentSignature  # or dynamically if per-task overrides are declared
    demo = dspy.Example(**task["signature"])          # YAML now carries `signature:` dict
    prompt = dspy.ChatAdapter().format(sig, demo=demo, inputs=demo.toDict())
    # `prompt` is a deterministic string; pipe it to Claude Code as today.
    run_claude_code_subagent(stdin=prompt)
    artifact = json.load(open(task["signature"]["deliverable_path"]))
    validate_against_signature(artifact, sig, demo)   # NEW: fail-closed check
```

`validate_against_signature` checks: every name in `required_schema_fields` is present; `artifact["schema"] == schema_version`; `artifact["honest_verdict"] in honest_verdict_vocab`. Validation failure is a new conductor status `contract_violation`, distinct from `fail` / `success`.

### 2.3 Worked example — Exp 451 (Live Precision Re-Run)

```yaml
- id: exp451-live-precision-postfix
  milestone: 2026.04.34
  signature:
    exp_id: exp451-live-precision-postfix
    milestone: 2026.04.34
    title: "Live Precision Re-Run Post-Fix — first positive verify-repair number"
    deliverable_path: results/experiment_451_live_precision_postfix.json
    schema_version: carnot.live_precision.v2
    required_schema_fields:
      [schema, gemma4_loader, qwen_result, gemma4_result,
       first_positive_number, honest_verdict]
    honest_verdict_vocab: [first_positive, no_improvement_v2, deferred_to_gpu]
    capability: verifiable-reasoning
    spec_requirements: [REQ-BENCH-010, REQ-BENCH-011]
    spec_scenarios:    [SCENARIO-BENCH-020, SCENARIO-BENCH-021]
    requires_gpu: true
    gpu_layout:
      - {name: Gemma4-E4B-it, hf_id: google/gemma-4-E4B-it, gpu: 0}
      - {name: Qwen3.5-0.8B,  hf_id: Qwen/Qwen3.5-0.8B,     gpu: 1}
    env_prelude: ["apply_env_autofix()"]
    blocked_by: [exp450-gemma4-tokenizer-fix]
    existing_code_to_read:
      - python/carnot/pipeline/env_autofix.py
      - python/carnot/pipeline/gemma_loader.py
      - scripts/experiment_439_live_precision_micro.py
      - python/carnot/pipeline/verify_repair.py
      - python/carnot/extraction/crane.py
    baseline_artifact: results/experiment_439_live_precision_micro.json
    task_narrative: |
      Re-run the Exp 439 harness with the Exp 450 Gemma4 tokenizer fix.
      Use CRANE (arXiv 2504.15030) as the structured claim extractor.
      50 questions GSM8K per model. Report signed_improvement honestly.
      Emit LivePrecisionResult(model_id, pre_accuracy, post_accuracy) with
      signed_improvement and is_positive properties. first_positive_number = any is_positive.
```

### 2.4 What DSPy serialises this to (abridged)

```
Your task is to complete the ExperimentSignature task.

[[ ## exp_id ## ]]
exp451-live-precision-postfix
[[ ## title ## ]]
Live Precision Re-Run Post-Fix — first positive verify-repair number
...
[[ ## required_schema_fields ## ]]
["schema","gemma4_loader","qwen_result","gemma4_result",
 "first_positive_number","honest_verdict"]
[[ ## task_narrative ## ]]
Re-run the Exp 439 harness with the Exp 450 Gemma4 tokenizer fix...

Strategy:
  1. Read CLAUDE.md; follow spec-first workflow.
  ...
  7. Do NOT push. Do NOT modify scripts/research_conductor.py.

Respond with JSON matching the output fields artifact_json, test_report, honest_verdict.
```

Every task gets the same boilerplate steps rendered from the docstring — free-form drift goes away.

## 3. Migration path

1. **Introduce `signature:` block alongside existing `prompt:`** in `research-roadmap.yaml`. If both are present, `signature:` wins. If only `prompt:` is present, the conductor renders exactly as today (zero risk to in-flight experiments).
2. **Teach the planner agent** (the Claude subagent that writes new roadmap entries at milestone-planning time) to emit `signature:` blocks. The planner is already one file of instructions; one prompt update.
3. **Backfill lazily.** When any existing task fails validation under the new contract checker (or when an experiment is re-run), convert its `prompt:` to a `signature:` as part of that work. No big-bang migration, no throwaway translator script needed.
4. **Tests that must exist before shipping:**
   - `tests/python/test_experiment_signature.py` — round-trips a YAML `signature:` block through DSPy render, asserts stable prompt output.
   - `tests/python/test_conductor_contract_validation.py` — feeds a deliberately-malformed artifact, asserts `contract_violation` status.
   - `tests/python/test_roadmap_schema.py` — every entry in `research-roadmap.yaml` has either `prompt:` (legacy) or `signature:` (new); never both missing.
   - Golden-file test: render Exp 451 signature and diff against the committed expected string. Protects against DSPy adapter-version drift.
5. **Rollout order:** land contract validator behind a feature flag (`CARNOT_ENFORCE_SIGNATURE=1`) first — run it in audit-only mode for one milestone, then promote to fail-closed.

## 4. Risks / open questions

- **DSPy ↔ Claude Code impedance.** DSPy assumes it drives the LLM via its own adapter. We are using DSPy only to render the prompt; the subagent is Claude Code CLI, which knows nothing about DSPy's `[[ ## field ## ]]` delimiters. Risk: Claude Code ignores the structure and we gain nothing. Mitigation: do a one-day spike — render Exp 451 both ways, run 3 subagents on each, measure contract-violation rate. If Claude Code parses the typed blocks cleanly, ship. If not, fall back to a hand-written Jinja template populated from the same signature object (we still get the typed schema and validator; we lose only the free DSPy prompt-optimisation surface).
- **DSPy as a dependency for non-DSPy workflow.** Pulls in `dspy-ai` and a pile of transitive deps (litellm, pydantic-v2 constraints, openai). The conductor process is already Python-heavy so the weight is tolerable, but we should pin DSPy and not expose it in the `carnot-python` wheel.
- **No optimisation loop.** DSPy's real value is compiling signatures → prompts with metrics. We are not using that here because our "metric" is a human-adjudicated verify-repair number that takes hours to compute. This proposal is deliberately *pre-compilation*: adopt the schema, skip the optimiser. If we ever want GEPA-style signature evolution (per the predict-rlm README) we can add it later without changing the YAML shape.
- **Full RLM recursion is out of scope.** predict-rlm's core trick is letting the root LM write Python that calls `predict()` on a sub-LM inside a sandbox — i.e. the LM orchestrates its own recursion. Adopting that for Carnot experiments would mean the conductor *becomes* an RLM and each experiment is a `predict()` call. That is architecturally attractive (it maps cleanly onto our JAX sandbox → Rust transpile loop) but requires a Docker+gVisor sandbox we trust (per project policy), cost modelling, and trace capture we don't have. Mark as a 2026.Q3 follow-up; do not block this proposal on it.
- **Signature evolution.** What happens when we need to add a new required input field (say, `energy_budget_joules`)? Old YAML entries won't have it. Proposed: every new field is `Optional` at introduction; promote to required only after the roadmap is fully backfilled, gated by `test_roadmap_schema.py`.

## 5. References

Read during research for this proposal:

- `github.com/Trampoline-AI/predict-rlm/README.md` — RLM pitch, bitter-lesson framing, `AnalyzeImages` quick-start signature. https://github.com/Trampoline-AI/predict-rlm/blob/main/README.md
- `github.com/Trampoline-AI/predict-rlm/docs/how-it-works.md` — the signature-as-strategy pattern; `AnalyzeDocuments` with `File` input and structured `DocumentAnalysis` output; the `SUBMIT()` contract. https://github.com/Trampoline-AI/predict-rlm/blob/main/docs/how-it-works.md
- `github.com/Trampoline-AI/predict-rlm/src/predict_rlm/predict_rlm.py` — `PREDICT_RLM_INSTRUCTIONS` template showing how they splice `{inputs}` / `{output_fields}` / `{final_output_names}` into a boilerplate workflow — direct inspiration for our strategy-in-docstring pattern. https://github.com/Trampoline-AI/predict-rlm/blob/main/src/predict_rlm/predict_rlm.py
- `github.com/stanfordnlp/dspy/docs/docs/learn/programming/signatures.md` — canonical DSPy signature semantics: field names carry meaning, `instructions=` kwarg, inline `"context: list[str], question: str -> answer: str"` form. https://github.com/stanfordnlp/dspy/blob/main/docs/docs/learn/programming/signatures.md
- Local: `/home/ianblenke/github.com/ianblenke/carnot/research-roadmap.yaml` (Exp 451 entry) — current free-form prompt format this proposal replaces.
- Local: `/home/ianblenke/github.com/ianblenke/carnot/scripts/experiment_template.py` — existing `REQUIRED_RESULT_FIELDS` constant that becomes `required_schema_fields` in the signature.
