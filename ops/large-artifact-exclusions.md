# Large-Artifact Exclusions — files on disk but not in git

Durable record of artifacts deliberately removed from git history because they exceed a hosting
limit. **Append only. Never delete an entry** — an entry that disappears is indistinguishable
from a file nobody ever noticed was missing, which is the exact failure this file exists to
prevent.

Every entry must state: what was removed, why, where the substantive record survives, how to
verify the on-disk copy is intact, and how to regenerate it from scratch.

---

## `results/experiment_5852_three_family_paired_embeddings.rows.jsonl`

| | |
|---|---|
| Removed | 2026-07-31 |
| Size | 461,137,495 bytes (439.8 MiB) |
| Reason | Exceeds GitHub's hard 100 MB per-file limit |
| Method | `git filter-repo --path <path> --invert-paths --prune-empty never` |
| Introduced by | `300a3a373` (2026-07-23) "[conductor] Current-SOTA causal-pair embedding extraction across three model families" |
| Pre-rewrite backup | tag `pre-filter-repo-backup-20260731` → `ebce169de` |
| On disk? | **Yes** — untracked, ignored via `.gitignore` |

### Why this had to happen

The blob blocked **every** push to the canonical GitHub mirror from 2026-07-23 onward. GitHub's
pre-receive hook rejects the whole push if any commit in it carries a >100 MB file, so **411
commits sat unpushed for 8 days** while gitea stayed current — the two mirrors silently diverged,
which defeats the point of mirroring (CLAUDE.md "Decentralization-Respecting Design Constraints"
rule 3: the canonical URL is the one users mirror *from*).

Nothing about this was visible as a failure. `git push origin main` is a multi-URL push (gitea +
github); gitea accepted, github rejected, and the combined output looked like a single failure
against one remote. It was only found by checking `git ls-remote` per URL.

### This is NOT a never-prune violation

Only bulk float data left version control. The substantive record is intact and still tracked:

- `results/experiment_5852_three_family_paired_embeddings.json` (637 KB) — claims, methodology,
  `model_specs`, `preconditions_checked`, `honest_verdict`, `deterministic_embedding_config`
  (with `config_hash`), and a `row_file_receipt` carrying **per-row sha256 hashes for all 1,944
  rows** keyed by `candidate_id|model`, plus `row_hash_root` and `row_count`.
- `results/experiment_5853_paired_embedding_integrity_audit.json` — the independent integrity
  audit of this very corpus.

So every row remains individually attested, and the corpus is regenerable from a deterministic
config. What was lost from git is the float payload those hashes describe — not the evidence.

### Verify the on-disk copy is intact

Checks the data itself, not merely that a file exists. Recomputes each row's hash with the
experiment's own `source_row_hash` and compares against the recorded value:

```bash
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, "python")
from carnot.experiment_5852_three_family_paired_embeddings import source_row_hash
bad = n = 0
with open("results/experiment_5852_three_family_paired_embeddings.rows.jsonl") as fh:
    for line in fh:
        if not line.strip():
            continue
        row = json.loads(line); n += 1
        if row.get("row_hash") != source_row_hash(row):
            bad += 1
print(f"rows={n} mismatches={bad}", "INTACT" if bad == 0 else "CORRUPT")
PY
```

Verified 2026-07-31 immediately after the rewrite: **1,944 rows, 0 mismatches.** The restored
file is byte-identical to the original blob `b224cfa0733d517bdb510afc473bbae0d46e4be5`
(sha256 `a76ccc5f25e5aa626d601472e2843fa075810a513357984a745a22f5b9c21e23`).

Note the file's plain sha256 does **not** equal the receipt's `receipt_hash` or `row_hash_root`
— those are derived (a hash over the receipt structure, and a root over per-row hashes), not
file digests. Comparing against them directly will look like corruption and is not.

### Recover it

From the backup tag, if the on-disk copy is ever lost:

```bash
git show pre-filter-repo-backup-20260731:results/experiment_5852_three_family_paired_embeddings.rows.jsonl \
  > results/experiment_5852_three_family_paired_embeddings.rows.jsonl
```

That tag exists **only in the local repo and on gitea** — it is not pushed to GitHub (pushing it
would re-introduce the blob and re-block the mirror). If the local repo and gitea are both lost,
regenerate instead: `python/carnot/experiment_5852_three_family_paired_embeddings.py`, whose
`deterministic_embedding_config` (`config_hash`
`sha256:74bebba9e9291c4c0e34f965ed55c4d82661367f63bf4a211f8f11926695cd66`) pins the backend,
`n_ctx`, batch sizes, precision and padding vocabulary.

### Known consequence: a fresh clone does not have this file

Three tracked code paths reference it by name and will not find it in a clean checkout:

- `python/carnot/experiment_5852_three_family_paired_embeddings.py:47` (`ROW_FILE_RELATIVE_PATH`)
- `python/carnot/experiment_5862_v521_capstone_reconciliation.py:123`
- `tests/python/test_experiment_5853_paired_embedding_integrity_audit.py:387`

This matters most for the **G2 independent-reproducer gate**, which runs from a clean clone. Any
of these paths that a reproducer must exercise needs either a regeneration step or a skip-with-
reason. Recorded here rather than silently discovered later; not fixed as part of the rewrite,
because changing test behaviour and a history rewrite in one motion would make each harder to
review.

### If this recurs

Prefer **Git LFS** (`git lfs migrate import --above=100MB`) over another removal — it keeps large
artifacts under version control instead of trading them for a `.gitignore` line. It was not used
here because it needs LFS support and quota on *both* the GitHub and gitea remotes, and the
immediate problem was 411 stranded commits.

Blobs >100 MB exist elsewhere in this repo (~19 GB, mostly `models/openai_privacy_filter/**`) but
are **not reachable from `main`** — they live on feature branches and the
`pre-filter-repo-backup-20260424` tag. They do not block a `main` push and were left untouched.
Pushing any of those branches to GitHub **will** fail for the same reason.
