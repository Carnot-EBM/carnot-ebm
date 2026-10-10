# Dust and energy-based models: is there an experiment here? (2026-10-10)

Companion to `docs/research-notes/dust-backprop-free-pretraining-2026-10-10.md`, which
covers what Dust is and what its numbers show. This note answers one narrower question:
does Dust imply an energy-based-model (EBM) experiment for Carnot?

Labels used below:

- **[paper]** — what the Dust page or a cited paper says.
- **[retrieved]** — what I fetched this session (URL listed in Sources).
- **[code]** — what the Carnot repo says, with a file path.
- **[inference]** — my own reasoning. Treat it as a claim to check, not a fact.

## TL;DR

Dust is node perturbation plus one trick: it treats each token position as its own
perturbation trial, so one forward pass over a long sequence gives thousands of noisy
gradient samples at once. **That trick needs a model that emits a separate loss at every
position of its own forward pass.** No Carnot EBM does that. The Ising, Gibbs, Boltzmann
and KAN tiers emit one scalar energy per input. Carnot's EBT also emits one scalar per
sequence. Per-constraint energies in the verifier ensemble are dense over constraints,
but they come from separate functions with no shared layer to perturb.

The hardware angle also does not hold. Carnot's FPGA, D-Wave and THRML paths are
*samplers*. They produce model samples; the host computes the parameter gradient. For the
Ising tier that gradient is a closed-form correlation difference and never needed
backprop. For the deeper tiers, backprop runs on the GPU through a small JAX energy
network, with the sampler only supplying negative samples. A forward-only estimator would
replace an exact, cheap gradient with a noisy, expensive one.

The literature that does connect forward-only training to EBMs is equilibrium propagation
and its relatives, not Dust. Those methods matter only when the energy's parameters live
inside a physical device the host cannot differentiate. Carnot has no such device today.

**Decision: (a) no experiment.** The nearest candidate experiments either have a known
answer (exact gradients beat node perturbation on a 17-parameter selector) or collide with
retired scope (the THRML scaling sweep). The one conditional follow-up is already recorded
in `research-references.md` with a "defer until a compatible device exists" note.

## 1. Is there a real link between Dust's estimator and EBM training?

**What Dust estimates. [paper, via the companion note]** Dust adds Gaussian noise to each
linear layer's output, separately at every token. It scores each token's noise by the loss
drop at that token and, with decay, at later tokens. The reward-weighted noise estimates
the error signal at the layer output. That error times the layer input gives the weight
gradient, the same outer product backprop forms. Only the source of the output error
differs.

**What EBM training needs. [code]** Carnot's trainers are:

| Trainer | File | Needs |
|---|---|---|
| NCE | `python/carnot/training/nce.py` | `energy(x)` on data and on noise; gradient of a classification loss w.r.t. parameters |
| Denoising score matching | `python/carnot/training/score_matching.py` | `grad_energy(x)` w.r.t. the *input*, then a gradient of that w.r.t. parameters |
| Maximum likelihood / contrastive divergence (classic, not a Carnot file) | the gradient is `E_data[dE/dθ] − E_model[dE/dθ]`; model samples come from a sampler |

**Where the hardware sits. [code]** `python/carnot/samplers/backend.py:81` defines
`SamplerBackend` with `minimize_energy` and `sample` as its working methods. (Review
correction: the protocol also has `backend_name`, `set_constraints` and `dual_update_step`.
The last two are no-ops except for the CASAL primal-dual sampler, where `dual_update_step`
sets a Lagrange-multiplier step size. Neither updates energy parameters.) `fpga_backend.py`
pushes biases and couplings to the KV260 and reads spins back. The D-Wave and THRML
backends have the same shape. No backend exposes an update of the energy's parameters. The
hardware is a sampler, not a trainer.

**Why "hardware cannot backprop" is not a problem Carnot has. [inference, with one
retrieved anchor]**

- For the Ising tier, `E(x) = -0.5 xᵀJx − bᵀx` (`python/carnot/models/ising/__init__.py`).
  Its parameter gradient is `dE/dJ_ij = -x_i x_j` and `dE/db_i = -x_i`. The maximum
  likelihood gradient is a difference of data and model correlations. That is the
  Boltzmann-machine rule. It needs samples, not backprop. Amin et al. train a Boltzmann
  machine this way and discuss D-Wave as the sampler [retrieved, arXiv:1601.02036].
  Carnot's Ising tier could already train from FPGA samples with no backward pass.
- For the Gibbs and Boltzmann tiers, the energy is a small MLP in JAX. The backward pass
  runs on the GPU, costs about one forward pass, and is exact. The sampler only provides
  negative samples. Nothing here runs backprop *through* the hardware.
- Dust would replace the exact MLP gradient with a Monte Carlo estimate. The estimate
  needs many perturbations per step. Dust used 16k draws on 8 GPUs for its best cells
  [paper]. For a small MLP that is pure cost.

**Could a forward-only estimator ever help? [inference]** Yes, in exactly one setting: the
energy's parameters are physical quantities inside a device, the device relaxes to a
low-energy state on its own, and the host cannot differentiate through it. Examples are
an analog Hopfield chip, an oscillator Ising machine with on-chip couplings, or a future
thermodynamic part with in-device weights. Carnot's hardware portfolio (CLAUDE.md
"Hardware Acceleration Portfolio") lists the KV260 as proof-of-concept tier only, and the
Extropic Z1 as "awaiting public specs". So this setting is not reachable now.

Even in that setting, Dust is the wrong tool. The matching literature is equilibrium
propagation (EP), coupled learning, and multiplexed gradient descent (see section 3).
EP's update is itself a contrast of two equilibrium states, which is what an EBM already
computes. Dust's per-token population has no counterpart in a sampler: a chain produces
one configuration, not an ordered sequence of positions each with its own loss. A batch of
chains is ordinary population evolution strategies, which the perturbative-training
papers already cover.

**What breaks if someone tried anyway. [inference]**

1. No per-position loss (section 2), so the virtual population collapses to one trial per
   forward pass.
2. Variance scales with the number of perturbed activations. Oripov et al. report the time
   to estimate a gradient grows linearly with the parameter count [retrieved,
   arXiv:2501.15403].
3. For NCE, the noise term and the data term are two separate forward passes. A
   perturbation reward would have to be computed per pass and combined, doubling the
   population cost.
4. For score matching, the loss is on the *input* gradient. A forward-only estimator of a
   parameter gradient of an input gradient is a second-order object. Dust does not address
   it.

## 2. Which Carnot signals are dense, and does that give Dust its reward?

Dust's trick needs three things at once: (i) shared parameters across positions, (ii) a
loss at every position, and (iii) a causal order so later losses can be credited to
earlier noise with decay. [paper, restated]

**Per token. [code]** No Carnot EBM emits a per-token energy.

- `python/carnot/models/ebt.py` (Energy-Based Transformer): "mean-pool over sequence →
  linear → scalar energy". One number per sequence. This is the only Carnot model shaped
  like Dust's target, and it fails condition (ii).
- `python/carnot/training/sdpo_dense_reward.py` builds a token-level proxy from "an energy
  score + seed" (line 161). The dense signal is derived from one scalar; it is not a
  measured per-token loss.
- The FoVer corpus has 6548 labeled reasoning-step rows
  (`python/carnot/autoresearch/calibrated_decision_benchmark.py`, docstring). The labels
  are dense over LLM reasoning steps. They are inputs to the verifier, not losses at
  positions of the EBM's own forward pass. The selector still sees one feature vector and
  one label per row.

**Per constraint. [code]** `python/carnot/verify/constraint.py:264` `ComposedEnergy`
sums energies from separate constraint objects. The verifier ensemble (Z3, AST, SAT,
liveness, and the rest) has one score per constraint per candidate. This is dense over
constraints and satisfies (ii). It fails (i) and (iii): each constraint is its own
function, most are non-parametric, and there is no ordering. Rewarding a perturbation of
constraint k by constraint k's energy change is just independent training of independent
heads. Exact gradients already do that where a head has parameters.

**Per site. [inference]** The Ising energy decomposes over pairs `(i, j)`. Local energies
are dense over sites. But the parameter gradient for each pair is closed-form
(`-x_i x_j`), so there is nothing to estimate. Dust's estimator would add noise to a
quantity Carnot already has exactly.

**One score per answer. [code]** The GibbsModel selector in the `calibrated_decision`
benchmark emits one scalar energy per row. The verifier ensemble emits one composed score
per candidate. These are the project's main signals and they are the opposite of dense.

**Verdict on the strongest link.** It does not hold. A per-constraint or per-site
decomposition gives density over constraints, not over an ordered sequence of positions
that share parameters. Without shared parameters the population is empty; without order
the decayed credit has nothing to attach to.

## 3. Closest prior work

All items marked [retrieved] were fetched this session. The URL is in Sources. I report
abstracts only; I did not read full texts.

**Directly relevant — forward-only or backprop-free training of an EBM or Ising
machine:**

- **Laydevant, Markovic, Grollier, "Training an Ising Machine with Equilibrium
  Propagation", arXiv:2305.18321 (May 2023; Nature Communications 2024)** [retrieved].
  Trains a fully-connected network and a compact convolutional network on MNIST using a
  D-Wave annealer and EP. No backprop. Results "comparable" to software, no number in the
  abstract. This is the closest match to "train an EBM on hardware that cannot backprop".
- **Scellier, Ernoult, Kendall, Kumar, "Energy-based learning algorithms for analog
  computing: a comparative study", arXiv:2312.15103 (NeurIPS 2023)** [retrieved]. Seven
  algorithms: contrastive learning, EP and variants, coupled learning and variants, on
  deep convolutional Hopfield networks over MNIST, Fashion-MNIST, SVHN, CIFAR-10,
  CIFAR-100. Headline: "negative perturbations are better than positive ones"; centered
  EP (two opposite-sign nudges) is best, with the gap widening on harder tasks. If Carnot
  ever trains an in-device EBM, this is the algorithm shortlist.
- **Fan, Lu, Wu, Wang, Wang, "Hybridizing Equilibrium Propagation with Ising Machines
  for Efficient Energy-Based Learning", arXiv:2606.09112 (June 2026)** [retrieved].
  Keeps EP's local two-phase rule, replaces dissipative relaxation with extended
  phase-space dynamics. Simulation only; deep convolutional Hopfield networks on MNIST,
  FashionMNIST, CIFAR-10; "comparable to backpropagation". Already in
  `research-references.md` (several rows, all marked defer).
- **Oripov, Dienstfrey, McCaughan, Buckley, "Scaling of hardware-compatible perturbative
  training algorithms", arXiv:2501.15403 (Jan 2025)** [retrieved]. Multiplexed gradient
  descent with weight and node perturbation. "The time to estimate the gradient scales
  linearly with the number of network parameters", but time to a target accuracy often
  falls with network size. Gives a drop-in gradient for SGD with momentum. This is the
  honest cost model for any perturbative EBM trainer.
- **Amin, Andriyash, Rolfe, Kulchytskyy, Melko, "Quantum Boltzmann Machine",
  arXiv:1601.02036 (Jan 2016; PRX 2018)** [retrieved]. Trains a Boltzmann machine by
  sampling, with D-Wave discussed as the sampler. Confirms the classic point: a
  Boltzmann-machine gradient needs samples, not backprop.

**Nearby — the method lineage, not EBMs:**

- **Dalm, van Gerven, Ahmad, "Node Perturbation Can Effectively Train Multi-Layer Neural
  Networks", arXiv:2310.00965 (Oct 2023, rev. June 2026)** [retrieved]. The method Dust
  extends. Two forward passes, noise in activations, loss change as reward. Standard NP is
  "highly data inefficient and can be unstable"; adding input decorrelation and a
  directional-derivative view brings it near backprop. Not about EBMs.
- **Hinton, "The Forward-Forward Algorithm", arXiv:2212.13345 (Dec 2022)** [retrieved].
  Positive and negative forward passes; "goodness" is a layer's sum of squared activities.
  The abstract does not mention EBMs or Boltzmann machines. The positive/negative
  structure resembles contrastive EBM training, but the abstract makes no such claim and
  I do not add one.
- **Reifenstein, Leleu, "Neural Ising Machines via Unrolling and Zeroth-Order Training",
  arXiv:2602.00302 (Jan 2026)** [retrieved]. Trains the *update rule* of an Ising solver,
  not the couplings, with a zeroth-order optimizer because "backpropagation through long,
  recurrent Ising-machine dynamics leads to unstable and poorly informative gradients".
  Already in `research-references.md` with a simulation-only experiment hook.
  `python/carnot/samplers/thrml_npim_microprobe.py:456` already does a tiny zeroth-order
  grid over momentum and schedule. Nearby, not a Dust link.

**Seen only as search listings, not fetched (titles only, do not cite as read):**
arXiv:2510.12934 (EP on oscillator Ising machines; already summarized in
`research-references.md`), 2607.16271 (EPIC-CIM), 2606.13454 (optical EP on spatial
photonic Ising machines), 2602.03670 (EP for non-conservative systems), 2503.22810,
2406.16062, 2410.05966 (FLOPS), 2503.24322 (NoProp), 2511.01061, 2509.19063.

**What the repo already knows. [code]** `research-references.md` holds the EP-on-OIM
pair (2510.12934, 2505.02103) with a "pursue at the FPGA hardware milestone" note, and
2606.09112 in four separate ingestion rows, each concluding "defer" or "future training
route". `research-studying.md` notes Textual Equilibrium Propagation (2601.21064) as
adjacent. Nothing in `python/carnot` implements a clamped-versus-free-phase update; the
only `clamped` hits are Boolean clamping in reductions (`experiment_7392_v648_ising_reduction.py:512`).

## 4. Weak points in the claim chain, for an EBM use

The Dust paper's own limits are in the companion note (behind backprop at 10M and 20M
tokens, extrapolated power-law "limit", 8-GPU base run, no arXiv or review). The points
below are specific to an EBM use. All are [inference] unless marked.

1. **Variance versus population size.** Dust's win needs about 1k draws at 1M tokens and
   16k draws for its best cells [paper]. Those draws come free from token positions. An
   EBM has no positions, so every draw is a full forward pass. On two RTX 3090 cards the
   population would be hundreds, not thousands, per step. Oripov et al. put the gradient
   estimation time as linear in parameter count [retrieved]. For a Gibbs tier with
   default `hidden_dims=[512, 256]` that is a large multiplier over one exact backward pass.
2. **Transfer from transformers to small EBMs is untested.** Dust's models are 2M to 243M
   parameter transformers with a per-token cross-entropy [paper]. Carnot's selector in the
   calibrated-decision benchmark has 17 parameters (2 inputs, 4 hidden units, 1 output)
   [code]. The Ising tier at d=784 has about 615k closed-form parameters [code]. Neither
   resembles Dust's regime. No evidence in the paper covers it.
3. **The 1M-token win is a low-data regularization effect, not a general one.** Dust is
   ahead only at 100k and 1M tokens and falls behind as data grows [paper]. The lead is
   0.043 test loss over three seeds. For a selector whose metric is AUROC and Brier score
   on about 2000 training rows, noise-based regularization is available at zero cost by
   adding noise to an exact gradient. The forward-only part buys nothing.
4. **Circularity risk if misapplied.** The calibrated-decision benchmark's trust boundary
   recomputes metrics from submitted weights in a fresh subprocess [code]. A perturbative
   trainer would be a hypothesis inside that sandbox. It is allowed, but an "improvement"
   would mean the noise helped, not that forward-only training helped. The experiment
   could not separate the two without a noisy-backprop control, and with that control the
   question has a known answer.
5. **The hardware claim inverts.** The pitch "FPGAs cannot backprop, so use Dust" assumes
   the energy network lives on the FPGA. In Carnot it does not. The FPGA receives J and h
   and returns spins [code, `fpga_backend.py`]. Training already happens on the host.
6. **Dense-loss precondition fails on the one sequence model.** Carnot's EBT pools to one
   scalar [code]. Converting it to per-token energies is an architecture change with its
   own open questions, and the EBT lineage is research-class, not on the live path.

## 5. Decision

**(a) No experiment worth running.**

Reason, in one line: Dust's advantage is a per-position population trick, and no Carnot
EBM has per-position losses; its hardware motivation does not apply because Carnot's
hardware is a sampler whose parameter gradients are either closed-form or computed on
the host by exact autodiff.

**Alternatives I considered and ranked below (a):**

1. **(c) Reading-only.** Track centered-EP and coupled learning [retrieved,
   arXiv:2312.15103] as the training shortlist for a future in-device EBM. This is
   already done: `research-references.md` carries 2606.09112, 2510.12934 and 2505.02103
   with "defer until a compatible device" notes. No new reading is needed until a device
   with in-device trainable parameters arrives (the Extropic Z1 is "awaiting public
   specs" per CLAUDE.md). Re-open this note then.
2. **(b) Node perturbation versus exact gradient on the 17-parameter calibrated-decision
   selector.** Rejected. The answer is known: exact gradients are cheaper and at least as
   accurate on a model this size. A positive control (noise-regularized backprop) would
   match or beat it. The run would consume a conductor slot to confirm a textbook result.
3. **(b) Train the Ising tier from THRML or FPGA samples via the correlation-difference
   rule, no backprop.** Rejected. This is not a Dust idea; it is the Boltzmann-machine
   rule from 1985. It also scope-matches the retired THRML scaling sweep
   (`ops/exclusion_manifest.yaml`, `thrml_scaling_sweep_lineage_retired_after_vendoring`,
   `operator_reopen_required: true`) and would need an operator override.

**Retired-scope check.** I read `ops/exclusion_manifest.yaml`. No entry names Dust,
node perturbation, forward-forward, equilibrium propagation, or zeroth-order training.
The only collisions are with alternatives 2 and 3 above, which I do not propose.
`ops/known-issues.md` has no forward-only training entry (grep for forward-only,
evolution strategies, zeroth-order, EGGROLL, node perturbation, equilibrium propagation
returned only unrelated "forward-only correction" prose).

**What would change this decision.** Any one of: (1) a Carnot model with a measured
per-position loss on its own forward pass; (2) a physical device whose energy parameters
live in-device and cannot be differentiated by the host; (3) an independent reproduction
of Dust's 1M-token win on a non-transformer model. None exists today.

## Sources

Fetched this session (abstract pages):

- https://arxiv.org/abs/2305.18321 — Training an Ising Machine with Equilibrium Propagation
- https://arxiv.org/abs/2312.15103 — Energy-based learning algorithms for analog computing: a comparative study
- https://arxiv.org/abs/2501.15403 — Scaling of hardware-compatible perturbative training algorithms
- https://arxiv.org/abs/2606.09112 — Hybridizing Equilibrium Propagation with Ising Machines for Efficient Energy-Based Learning
- https://arxiv.org/abs/2602.00302 — Neural Ising Machines via Unrolling and Zeroth-Order Training
- https://arxiv.org/abs/2310.00965 — Node Perturbation Can Effectively Train Multi-Layer Neural Networks
- https://arxiv.org/abs/2212.13345 — The Forward-Forward Algorithm: Some Preliminary Investigations
- https://arxiv.org/abs/1601.02036 — Quantum Boltzmann Machine

Web searches run (three, low concurrency): "equilibrium propagation Ising machine training
without backpropagation arXiv"; "node perturbation OR weight perturbation training
energy-based model OR Hopfield network arXiv"; "zeroth-order OR evolution strategies OR
forward-forward training Boltzmann machine OR energy-based model without backpropagation".

Not fetched this session; taken from the companion note:

- https://qlabs.sh/research/dust — the Dust paper (no arXiv version)
- https://qlabs.sh/research/dust/appendix.html — its appendix
- https://github.com/qlabs-eng/dust — code (MIT)

Repo files read: `docs/research-notes/dust-backprop-free-pretraining-2026-10-10.md`,
`python/carnot/training/nce.py`, `python/carnot/training/score_matching.py`,
`python/carnot/models/ising/__init__.py`, `python/carnot/models/ebt.py` (header),
`python/carnot/samplers/backend.py`, `python/carnot/samplers/fpga_backend.py` (header),
`python/carnot/autoresearch/code_improvement.py`,
`python/carnot/autoresearch/calibrated_decision_benchmark.py`,
`python/carnot/training/sdpo_dense_reward.py` (grep), `python/carnot/verify/constraint.py`
(grep), `ops/verifier_gaps.md` (GAP-ORACLE-DISTINCT and GAP-DETECTOR-AUROC-4208 blocks),
`ops/exclusion_manifest.yaml`, `ops/known-issues.md` (grep), `research-references.md`
(EP and zeroth-order rows), `research-studying.md` (grep).
