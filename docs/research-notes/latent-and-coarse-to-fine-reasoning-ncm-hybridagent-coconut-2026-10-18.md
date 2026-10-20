# Latent and coarse-to-fine reasoning: NCM, HybridAgent and Coconut

Date: 2026-10-18. Status: reading note. Nothing here was run. All numbers come from the
papers and are the authors' claims, not Carnot measurements.

## Scope of the reading

| Source | What was read | What was not read |
|---|---|---|
| Next-Concept Modeling (NCM), arXiv 2610.14162 | Pages 1-14 of the PDF: method, experiments, conclusion | Appendices (ablations, Sudoku trajectory, test-set size) |
| Hybrid Reasoning Agents (HybridAgent), arXiv 2610.18861 | Pages 1-12 of the PDF: method, main results, training dynamics, reward ablation | Table 5 (causal intervention), readout metric appendix, baseline setup appendix |
| Coconut, arXiv 2412.06769 (v4, Aug 2026) | Pages 1-9 of the PDF: method, ProsQA analysis, main results | Later pages (scaling, conclusion, limitations) |
| `HazyResearch/ncm`, `togethercomputer/HybridAgent` | The GitHub landing pages only | The code. Nothing was cloned or run. |

## What each paper does

**NCM.** A multiscale residual VQ-VAE turns text into K levels of discrete codes, short and
coarse first. A transformer predicts each level in one parallel pass, conditioned on the
reconstruction so far. Nested dropout makes every partial reconstruction decodable. The
tokenizer and generator are trained per task.

**HybridAgent.** The model switches between text mode and latent mode inside one generation.
In latent mode the last hidden state goes back in as the next input, through a small adapter.
Tool calls stay in text. Training is HybridSFT (54K trajectories, self-conditioned) then
HybridRL (policy updates on text positions only, with exact replay of the latents).

**Coconut.** The parent of HybridAgent. It feeds the last hidden state back as the next input
for a few steps, trained with a curriculum that removes language reasoning steps one stage at
a time. Base model is GPT-2.

## Reported results that matter for Carnot

NCM, Sudoku, fully solved boards (Table 3 of the paper):

| Model | Size | Function evaluations | Easy | Medium | Hard |
|---|---|---|---|---|---|
| Autoregressive sample | 25M | 128 | 13.9 | 5.1 | 0.6 |
| MDLM / DUO (diffusion) | 29M | 128 | 92.0 / 96.3 | 77.1 / 84.7 | 30.2 / 58.4 |
| FMLM+ (best baseline) | 29M | 16 | 97.7 | 92.8 | 81.3 |
| NCM | 25M | 12 | 99.5 | 96.5 | 86.1 |

The baseline numbers are copied from an earlier paper, not re-run. The margin over FMLM+ is
1.8, 3.7 and 4.8 points. The test-set size and seed count are in an appendix not read.

HybridAgent: large gains on agentic tasks (8B GAIA average 43.7 against 32.0 for GRPO and 38.8
for ARPO). Small gains on math: 14B AIME25 43.3 against 42.1, inside the stated noise of
plus or minus 0.031. HybridSFT alone only matches plain text SFT.

Coconut (Table 1): ProsQA 97.0 against CoT 77.5. GSM8k 34.1 against CoT 42.9. On ProsQA the
ablation with the curriculum but no latent vectors scores 95.5, and "pause as thought" scores
96.6. Without the curriculum, ProsQA falls to 76.1.

## What transfers, and what does not

**Does not reach the ARC live path.** Both latent methods need hidden-state feedback and
weight training. The pinned generator is a GGUF model behind `llama-server`. The HybridAgent
authors state that current serving engines lack this support. The Energy-Based
Calibrated-Decision Training Floor also excludes any weight update to the 27B generator.
NCM needs its own tokenizer and generator per task.

**What carries over as method:**

1. **The curriculum is the active ingredient in Coconut.** On the logic tasks the latent
   vectors add about 1 to 2 points over a curriculum alone. The breadth-first-search reading
   rests on probing, not on a necessity test.
2. **Latent reasoning is not free.** Coconut loses to CoT on GSM8k, and at 8B scale scores
   13.1 on AIME25 against 30.0 for plain text SFT in HybridAgent's own table.
3. **Value estimates blur far from the goal.** Coconut's Fig 7 shows correct and incorrect
   nodes separate near the leaves and overlap at greater height. This is a calibration
   statement about depth.
4. **Delayed commitment matches our explicit search.** `OfflineSolver` keeps a frontier and a
   verifier. Coconut does this inside the weights. We cannot change the weights, so the
   explicit form is the one available to us.
5. **Self-conditioned training closes a train and inference mismatch.** HybridAgent trains on
   its own predicted latents. The general lesson applies to any selector trained on our own
   noisy verifier output.
6. **Process-level error classes.** Coconut separates hallucinated edges, longer paths and
   wrong targets. Final-answer accuracy hides these.

## Two testable ideas (not run, not approved)

**Idea A. Verifier per level on NCM Sudoku.** The released Sudoku checkpoints decode at every
level. Score each level with `check_sudoku_validity` and plot constraint violations per level.
This tests the claim that later levels correct earlier partial solutions. Limits:

- Sudoku validity is the oracle itself. Any win is `verifier_is_oracle: true` and
  execution-grounded. It is not a verifier-moat result.
- It needs a GPU. GPU 1 is busy with the llm_on induction run until that run ends.
- Checkpoint and test-board details must be read first, in a sandbox and an empty directory.

**Idea B. Calibration by depth on our own solver traces.** Measure whether a verifier's
distance estimate is less reliable far from the goal, using offline solver traces. Report the
expected calibration error per depth bucket. This needs no new model and no GPU. It would add a
calibration metric to an existing verifier evaluation, which qualifies under the Calibrated
Decision floor. The hypothesis comes from Coconut's Fig 7 and has not been checked on our data.

## Cautions

- All three papers use small models or synthetic tasks. NCM generators are about 100M
  parameters, or 25M for Sudoku. Coconut uses GPT-2.
- NCM quality is measured by generative perplexity under GPT-2-large, a proxy that is easy to
  game. The authors also report entropy.
- HybridAgent speed numbers use serving code the authors wrote for their own method.
- Both new repositories are days old, with 5 commits each. Read the code before running it.
  HybridAgent needs a third-party search API key. Do not set one.
- Two sources are needed before any novelty claim, per the literature-priority rule. This note
  makes none.

## Open items

1. Read the NCM Sudoku appendix for the test-set size, the seed count and the trajectory.
2. Read HybridAgent Table 5 before citing the latent readout as faithful.
3. Decide whether Idea A or Idea B earns a task slot. Neither is scheduled.

## Addendum 2026-10-19 (video, appendix and repo read)

Sources read: the Code Bear video "NCM: The LLM Killer?" (description in full; transcript for the
first 236 of 393 seconds, from whisper `small`), the NCM appendix on Sudoku, HybridAgent Table 5
and the readability metric, and the two GitHub repositories (file listings, both READMEs, the
Sudoku config, the readout and reward code). Nothing was cloned or run. Most of both PDFs
(40 and 45 pages) was not read.

1. **The video adds no evidence.** It summarizes the two papers. Its Sudoku figures match
   Table 3: left-to-right baselines solve 0.6 to 1.0 percent of hard boards, NCM 86.1 percent,
   FMLM+ 81.3 percent. Its 99.8 percent rebuild figure for summaries was not checked. The title
   "LLM killer" is not supported: the largest NCM tested has 725M parameters, and NCM lost on
   translation to a diffusion model.
2. **Open item 1 is answered in part.** The appendix gives the trajectory (Figure 20): correct
   cells per pass are 16, 15, 16, 21, 19, 25, 28, 44, 55, 62, 64, 81. The drafts are not monotone.
   The paper and the repo config state no test-set size and no seed count. The config names only
   `puzzles.pickle`. The size would be in the HF dataset, which was not opened. The 4.8 point gap
   between NCM and FMLM+ on hard boards therefore cannot be tested for significance.
3. **Open item 2 is answered.** HybridAgent Table 5 (MATH500, 14B): no intervention 88.2;
   cross-problem latent replacement 48.6; random-token replacement 44.6; mean-latent
   replacement 58.4; nearest-token replacement 87.2; blocking latent mode 78.8. At 8B the
   figures are 84.6, 44.8, 41.2, 53.6, 82.6 and 80.4. Latent thoughts are used, and their
   content matters more than perturbation size. Latent mode itself is worth about 4 to 9 points
   over text-only reasoning. The readout is cosine nearest-token, so it shows alignment with a
   token embedding. It does not show that a latent means that token.
4. **Lesson for Carnot.** A replacement control (swap in the same feature from another example)
   is a cheap test that a learned feature carries content. The depth-calibration result for the
   cross-game value head is the same shape: the within-game signal was a step counter. Run a
   replacement or shuffle control before reading signal into a new learned feature.
5. **HybridAgent does not fit the pinned generator.** It needs hidden-state feedback and a
   trained adapter on Qwen3 8B and 14B bases. The pinned generator is a GGUF model behind
   `llama-server`, which gives no hidden-state feedback. Research reading only. Both repos
   are MIT, with released weights.
6. **Idea A is unchanged.** NCM releases tokenizer and generator checkpoints for easy, medium
   and hard Sudoku, with `evaluate.py`. It needs GPU 1, which the llm_on run holds.

Remaining open items: open the HF Sudoku dataset to get the test-set size; decide whether Idea A
earns a slot. Item 3 of the original list is unchanged.

## Addendum 2026-10-19 (MEMORILLA, NCM on ARC, training on public solutions)

1. **MEMORILLA (arXiv 2610.21375, snap-stanford/memorilla, Apache 2.0).** A frozen encoder and a
   learned memory module emit 16 query-dependent memory tokens into the input embedding space of
   a frozen Qwen3-8B. The abstract reports FactKG 84.33% against 80.43% with full context, at 108
   times fewer document tokens. It also reports TextWorld 18.6% to 24.2%. The seed count and the
   test size are not stated. Only the abstract page and the repo were read. The paper body, the
   ablations and any shuffle control were not read. The venue is unconfirmed.
2. **MEMORILLA does not fit the pinned generator.** The released checkpoint is Qwen3-8B only. It
   injects embeddings, so the serving stack must accept `inputs_embeds`. The pinned generator is a
   GGUF model behind `llama-server`. The repo has a vLLM integration. It does not state llama.cpp
   support. Whether `inputs_embeds` works on the pinned vLLM stack was not checked.
3. **NCM on ARC: no project found.** A search of the ARC leaderboard and the web found no project
   that uses an NCM model on ARC. The search collides with NCA (neural cellular automata), for
   example arXiv 2506.15746 and ARC-NCA arXiv 2505.08778. Leaderboard numbers found were
   fragmentary and conflicted, so none are quoted here. The common theme was that scaffolding and
   harness drive the gains. This is a search result, not proof that nobody tried.
4. **Recommendation: do not train an NCM model on the public-level solutions.** Three project
   rules apply. Trained weights are not the live mechanism. A solve made by outer-loop reverse
   engineering is `outer_loop_re`, and it does not count. The 2026-09-08 leave-one-game-out run
   disqualified itself, because withholding the module did not withhold the knowledge. Training
   on the public solutions repeats that leak. It also gives no clean held-out test.
5. **Related evidence: the frame-change CNN.** The held-out delta AUROC was 0.709. The
   cached-transition first-win rate went from 0.407 to 0.591, and the median action count went
   from 2 to 1. The model is tiny (8 channels, 497 examples). One follow-up run was a null. Read
   this as a small signal on cached data, not as live evidence.

Open: the MEMORILLA paper body and `inputs_embeds` feasibility were not checked. The NCM Sudoku
test-set size and seed count are still unknown.
