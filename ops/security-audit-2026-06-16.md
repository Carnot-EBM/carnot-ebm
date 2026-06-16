# Security Audit — 2026-06-16 (outer-loop)

**Trigger:** operator request — "ensure no secrets have leaked into the codebase in the
clear and that there are no known security gaps." First dated audit; prior security touch
was `SECURITY.md` on 2026-04-06.

## Bottom line: NO secrets leaked in the clear. 3 issues found (1 fixed, 2 flagged).

## Secret scan — CLEAN

Scanned all TRACKED files (`git grep`) for high-signal secret patterns:

- API keys/tokens (`sk-ant-`, `sk-`, `hf_`, `ghp_`/`gho_`, `AKIA…`, `AIza…`, `xox[bapr]-`):
  **none found.**
- Private-key / PEM blocks: **none real.** The single match
  (`scripts/experiment_729_privacy_filter_kan_true_distillation.py`) is FAKE test PII
  (`"Private key: -----BEGIN RSA PRIVATE KEY----- (truncated for security)"` alongside fake
  driver's-license / patient-ID strings) — input for a privacy-filter experiment, not a key.
- Plaintext `password`/`api_key`/`access_token` assignments: **none** (all real secret access
  goes through `os.environ` / SOPS, not literals).
- The 37k committed `.tmp-pytest/` fixtures (see below) contain **no plaintext secrets**;
  the `hf_token.enc.yaml` copies in them are SOPS-encrypted ciphertext.

**Posture working as designed:** SOPS at rest (`.sops.yaml`, `secrets/*.enc.yaml`), gitleaks
pre-commit hook (`gitleaks protect --staged` + `.gitleaks.toml`), and `.gitignore` excluding
`.env*` / `*.key` / `*.pem` / `secrets.*.yaml` (only `*.enc.yaml` committed).

## Issue 1 (FIXED) — `.tmp-pytest/` test fixtures were tracked (hygiene/latent risk)

**37,561** pytest `tmp_path` basetemp files were committed and `.tmp-pytest/` was NOT
gitignored. No secret leak (verified above), but it's large bloat AND a latent risk: a future
test that copies a plaintext secret into a fixture would get auto-committed.

- **Fixed:** added `.tmp-pytest/` to `.gitignore` (stops future growth).
- **Recommended (operator go-ahead — 37k-file untrack):**
  `git rm -r --cached .tmp-pytest && git commit -m "untrack ephemeral pytest tmp fixtures"`
  (does NOT delete the working files; only removes them from version control). Held back from
  doing autonomously given the scale + concurrent conductor commits.

## Issue 2 (PARTIALLY FIXED) — hardcoded `trust_remote_code=True` (policy gap, PERVASIVE)

CLAUDE.md policy: HF model loading must gate remote-code execution behind
`CARNOT_TRUST_REMOTE_CODE=1` (default False). The CORE code (`inference/llm_solver.py`,
`inference/model_loader.py`, `pipeline/verify_repair.py`) already gates correctly. But the
pattern is hardcoded `True` in **~46 experiment/script sites across ~43 files** (a full sweep,
not the 5 the initial narrow grep showed).

- **FIXED (the in-loop / active path):** the 3 files the conductor actually runs —
  `agentic/arc_exp4077_verifier_reward_rft_corpus_build.py` (3 sites),
  `agentic/arc_exp4078_verifier_reward_rft_train_launch.py` (1),
  `scripts/collect_multi_dataset_activations.py` (2) — gated to
  `os.environ.get("CARNOT_TRUST_REMOTE_CODE", "") == "1"`. **NOTE: these now require
  `CARNOT_TRUST_REMOTE_CODE=1` to run** (secure default).
- **REMAINING (~43 legacy files, FLAGGED not fixed):** `scripts/experiment_2x/3x/5x/6x/...`,
  the activation collectors (`collect_qa/token/truthfulqa_activations*`), `train_ebm/ebt_*`.
  Lower autonomous-execution risk (NOT run by the conductor; one-off/manual; load known repos).
  A blind sed sweep is UNSAFE — several matches are in docstrings (`experiment_69`,
  `experiment_91`) and inside f-string-generated subprocess code
  (`experiment_117/58/67/68/93`) that would corrupt. Needs a careful per-file pass.
  **Operator decision:** full careful sweep of the 43, or accept as low-risk-legacy +
  add a lint that blocks NEW hardcoded `trust_remote_code=True` going forward.

## Issue 3 (FLAGGED, low) — `shell=True` subprocess calls

3 sites use `subprocess.run(..., shell=True)`:
- `python/carnot/eval/loopus_fr11_self_learning_v2.py` (L118)
- `python/carnot/experiment_3348_independent_reproducer_pack_evidence_matrix_v40.py` (L54, `" ".join(cmd)`)
- `scripts/experiment_1102_hf_spaces_gallery_update.py` (L47)

All build STATIC commands (git / hf ops) with no untrusted/user/model input on the audited
paths — so no live injection. Code smell; prefer list-form `subprocess.run([...])`. No action
required; noted for the next refactor pass.

## Not re-audited this pass (scoped out)
Git HISTORY (only the current tree was scanned — gitleaks runs on staged diffs going forward,
so historical leaks from before the hook would need a separate `gitleaks detect --no-git=false`
full-history scan); runtime sandbox config (gVisor) and the SOPS key custody.
