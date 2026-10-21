# Post-training methods: OPD, OPSD and what they mean for Carnot (video note, 2026-10-21)

Status: secondary source. This note comes from one video transcript. The author did not read the papers it cites.

## Source

- Video: "Post-Train Your Local AI: Which Method Actually Works? Distillation?" from the Discover AI channel. It runs 2,243 seconds. Video id `oAtwkQg4ijI`.
- Method: YouTube captions were empty. The audio was downloaded through a local Invidious server and transcribed with faster-whisper (model `small`, CPU). The transcript is in the session scratchpad only. It is not in the repo.
- Papers the description names. None was opened for this note.
  - On-Policy Self-Distillation (OPSD), arXiv 2601.18734.
  - A Tsinghua paper on on-policy distillation (OPD): what it does, how it works, and a recipe.
  - OPDArena (Microsoft Research and Nankai University), dated 2026-10-19.
- Whisper-small garbled some names ("OPT" for OPD, "Quinn" for Qwen). Treat every number below as the narrator's figure, heard through that transcript.

## The method families in the video

| Family | Learning signal |
|---|---|
| Supervised fine-tuning (SFT) | A worked solution. The model raises the probability of each reference token. |
| Reinforcement learning with verifiable reward (RLVR) | A reward on the final answer, with a KL penalty toward a reference model. |
| Direct preference optimization (DPO) | A chosen response and a rejected response. No reward model. |
| Distillation | A teacher model's next-token probability distribution. |
| On-policy distillation (OPD) | The student writes the response. The teacher scores each token position. The loss is the reverse KL divergence. |
| On-policy self-distillation (OPSD) | The same model is student and teacher. Only the teacher sees a reference solution, in context. |

## Reported findings

1. SFT raises in-domain accuracy and can lower out-of-domain accuracy below the base model. The narrator quotes +14.9 points in domain for SFT and calls the out-of-domain drop catastrophic forgetting.
2. Self-OPD with an RLVR-trained teacher gave gains in domain and kept out-of-domain gains. The narrator calls it the best overall choice.
3. OPD needs teacher and student to have compatible token patterns. A teacher with a higher benchmark score can still give no new knowledge. The suggested fix is a teacher-generated SFT cold start, matched prompt format, and entropy and prompt-diversity monitoring.
4. The OPSD paper compared KL directions on Qwen3 1.7B (AIME 2025, 50 training steps). The narrator says forward KL beat reverse KL and Jensen-Shannon. I could not recover the exact figures from the transcript.
5. Two new papers distill decisions, not tokens: Action-OPD, and a tool-use OPD paper from Northeastern. The tool-use paper reports that credit concentrates on a few critical actions.

## What this means for Carnot

1. The SFT result supports the earlier caution about training on solved public-game levels. SFT on solved examples may hurt transfer to hidden games. The evidence is weak. The models are small and the tasks are math, not ARC.
2. Weight training is out of scope. The Energy-Based Calibrated-Decision Training Floor forbids any weight update to the 27B generator. It allows small selector models only.
3. OPSD cannot run on the live ARC agent. A hidden game has no reference solution to give the teacher. It could only serve offline development.
4. Verifiable reward as a training signal fits the verifier thesis. An executable oracle is circular for a "moat" claim under the Circularity / Oracle-Distinctness Discipline. As a training reward it is allowed.
5. Action-level credit is the one idea worth a later look. It resembles the trajectory-supervisor redirect ledger, which records which redirect arm led to a level-up. That ledger is empty until the supervisor fires.

## What this does not show

- It does not answer whether the LLM induction tier adds first wins over the same agent with the LLM off.
- It gives no method for the ARC agent or for any Carnot verifier.
- It does not show that any ranking holds for a 27B model or for ARC. A Carnot measurement is needed first.

## Open

- Read OPDArena (Microsoft and Nankai) directly and check the in-domain and out-of-domain figures.
- Decide whether the action-level credit idea needs a spec once the redirect ledger has data.
