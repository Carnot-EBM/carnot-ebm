# External Reproducer Outreach — Draft Messages

Phase 1 ship gate has six of seven items mechanically complete; the
seventh is "≥1 independent reproducer." Someone outside the project
running `pip install carnot-ebm` end-to-end is the last blocker
between Carnot and Phase 1 ship.

Below are three drafted messages adapted to three different recipient
contexts. Pick the one that fits your channel and recipient; edit the
opening to taste.

---

## Version 1 — Short (Slack / Discord DM, technical recipient)

> Hi [name] — I'm trying to ship the first public release of Carnot
> (open-source verifier-ensemble framework for LLM outputs, Apache 2.0,
> on PyPI as `carnot-ebm`). The last open item on the ship gate is
> getting one independent reproducer to confirm `pip install` + the
> quickstart works end-to-end on a machine that isn't mine. Would you
> have 30 minutes this week to try it? You wouldn't need any model
> weights or credentials — just Python 3.11+ and a fresh venv. Happy
> to walk through it live if easier.

---

## Version 2 — Medium (Email to a teammate / ML colleague)

> Hi [name],
>
> Quick favor: I've been building Carnot, an open-source framework
> for verifying LLM outputs using energy-based methods. The package is
> on PyPI as `carnot-ebm` and the model weights are mirrored at
> huggingface.co/Carnot-EBM. Apache 2.0, no vendor lock-in.
>
> Before I announce the v0.1 release, I need to confirm a third party
> can install and run the quickstart from a clean environment. Would
> you be willing to spend ~30 minutes doing exactly that?
>
> Concretely:
>
> 1. Create a fresh Python 3.11+ virtualenv
> 2. `pip install carnot-ebm`
> 3. Walk through `docs/getting-started.md` from
>    https://github.com/Carnot-EBM/carnot-ebm
> 4. Tell me what worked and what didn't
>
> I'm specifically NOT looking for a deep technical review — just
> confirmation that the install + the first one or two examples run
> as documented on a machine that isn't mine. If something is
> broken or unclear, that's the whole point: I'd rather find out
> from you than from a HN comment thread.
>
> Could also do it over a 30-min screenshare if that's easier.
>
> Thanks,
> Ian

---

## Version 3 — Public (HN / r/MachineLearning / Mastodon)

> I'm getting ready to publicly release Carnot — an open-source
> energy-based-model framework for verifying LLM outputs (Apache 2.0,
> PyPI: `carnot-ebm`, weights mirrored on HuggingFace). Before the v0.1
> announcement I want to confirm at least one independent reproducer
> can install and run the quickstart cold. Anyone willing to spend
> ~30 minutes on `pip install carnot-ebm` + the getting-started
> walk-through and tell me what breaks? No insider access required;
> the goal is precisely that you DON'T have any. Reply or DM if
> interested.

---

## What the reproducer artifact needs to contain

After the reproducer runs through the quickstart, the artifact this
work produces (for the Phase 1 ship gate) is a short markdown file
covering:

- Date, recipient identifier (initials or handle, if they consent
  to attribution)
- Hardware / OS / Python version they used
- Each quickstart step + whether it succeeded or failed
- Specific commands they ran (verbatim copy-paste)
- Any errors encountered + how (if at all) they resolved them
- Their honest read on whether the install path is something a
  stranger could actually use

The file lives at `ops/external-reproducer-<date>-<initials>.md`
and the artifact's existence is what closes the Phase 1 ship gate.

The reproducer does NOT need to verify scientific claims, evaluate
the verifier ensemble, or assess the paper — just the install +
quickstart path. The Phase 1 ship is a software-operational claim,
not a research claim (per CLAUDE.md "Project Vision (Three Phases +
Parallel Tracks)" Phase 1 ship gate definition).

---

## After this lands

Phase 1 ships. The blog series (5 posts), the cross-corpus matrix
(29 clean rows in v14, growing), the hardware portfolio (KV260 +
PolarFire + GateMate operational), the discipline machinery
(adversarial-verify + dual-condition + narrowing audit + substrate
declaration), the FoVer headline (0.9131 5-seed dual-condition,
defensible), and the v6 paper draft (narrowed per the 2026-05-23
Deep Think round) are all already in place. The reproducer artifact
is the final missing piece.
