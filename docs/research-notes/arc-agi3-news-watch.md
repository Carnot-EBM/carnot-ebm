# ARC-AGI-3 Competition News Watch

Daily automated check (systemd timer `arc-news-watch.timer`, see `scripts/arc_news_watch.py`) for ARC Prize / ARC-AGI-3 competition announcements, ahead of the November 2026 Kaggle submission deadline. Entries below are appended, never rewritten, per the project's never-prune documentation discipline.

## 2026-07-12 03:19 UTC -- NEW

- ARC-AGI-3 scoring changed: the per-level human baseline moved from the second-best human run to the median human run, and the per-level score cap increased from 100% to 115%. [ARC Prize announcement](https://arcprize.org/blog/arc-agi-3-human-dataset)

- Kaggle switched evaluation hardware from H100s to RTX 6000 Pro after an H100 stockout. Maximum notebook runtime increased from six to nine hours; a separate ARC-AGI-3 runtime setting initially remained at six hours but was fixed May 19. [Accelerator update](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697720), [runtime update](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/697944), [runtime fix](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/699208)

- Kaggle accidentally allowed five daily submissions from May 27 through June 8. The intended one-submission-per-day cap was restored, and surplus submissions after each team’s first successful daily submission were slated for invalidation. [Official correction](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/705405)

- Milestone eligibility is based on the public leaderboard, and open-source publication is due by 23:59 UTC on the applicable milestone date. [Leaderboard clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/703056), [deadline clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713634)

- The Kaggle competition does not apply the technical report’s five-times-human-action cutoff; that cutoff applies only to the ARC Prize Verified leaderboard. Kaggle instead relies on its compute limit. [Official host clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/713921)

- The post-Milestone public leaderboard now has a three-way tie at 1.56: Mathurin Ache, anngle, and NoOneAhead. [Current leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-12 13:12 UTC -- NEW

- Tecnod8 took the public-leaderboard lead with **1.61**, breaking the previous three-way tie at 1.56. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-13 13:13 UTC -- NEW

- NoOneAhead reached **1.56**, joining Mathurin Ache and anngle in a new three-way tie for second behind Tecnod8.AI (**1.61**). [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-14 13:09 UTC -- NEW

- YUTO KOJIMA took the lead with **1.86**, ahead of Tecnod8.AI (**1.61**); the **1.56** trio now ties for third. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-15 13:13 UTC -- checked, nothing new

## 2026-07-16 13:14 UTC -- checked, nothing new

## 2026-07-17 13:15 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-19 13:13 UTC -- NEW

DhanaLakshmiMalla scored **1.60**, taking sole third place and pushing the **1.56** trio to joint fourth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-20 13:13 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-21 13:14 UTC -- NEW

- YUTO KOJIMA set a new Kaggle high score of **1.86**. Tecnod8.AI is second at **1.61**, and ippeiogawa entered fourth at **1.58**, moving the **1.56** trio to joint fifth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- ARC Prize reported **GPT-5.6 Sol Max at 7.8% verified ARC-AGI-3 performance**—13.33% Public and 7.78% Semi-Private—and the first model to win a public environment, scoring 87% on ft09. [Official results](https://arcprize.org/results/openai-gpt-5-6)

- Organizers analyzed 500 failed submissions: about one-third lacked a traceable visible error, while almost 20% required a GPU without enabling one. Kaggle additionally clarified its 10 MB log cap, 20 GB `/kaggle/working` quota, and 30 GB CPU-notebook memory limit. [Error analysis](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119), [runtime clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)

## 2026-07-22 13:14 UTC -- checked, nothing new

## 2026-07-23 13:12 UTC -- NEW

- DhanaLakshmiMalla entered third place at **1.60**, displacing ippeiogawa to fourth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-25 13:13 UTC -- NEW

- ARC Prize verified Claude Opus 5 (High) at **30.16%**, a new ARC-AGI-3 model high score; it completed five Public Demo environments no previous model had beaten. [Official result](https://arcprize.org/results/anthropic-claude-opus-5)
- The competition host published an analysis of 500 failed Kaggle submissions: roughly one-third had no traceable notebook error, nearly 20% required a GPU that was not enabled, and each remaining failure category accounted for under 5%. [Kaggle announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-07-26 13:13 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-27 13:11 UTC -- NEW

- The Kaggle Public Leaderboard is now led by **YUTO KOJIMA at 1.86%**, ahead of Tecnod8.AI at 1.61%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-07-28 13:13 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-29 13:09 UTC -- NEW

- Kaggle’s competition host published an analysis of 500 failed submissions: nearly 20% used GPU-dependent code without enabling a GPU; other recurring issues included missing datasets/dependencies, CUDA OOM, using the wrong API endpoint, and writing to read-only `/kaggle/input`. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-07-30 13:12 UTC -- check failed

CHECK_TIMED_OUT

## 2026-07-31 13:16 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-01 13:12 UTC -- NEW

- Kaggle’s public competition leaderboard has a new high score: YUTO KOJIMA leads with **1.86%**, from an August 1 submission. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
- ARC Prize’s separate verified-model leaderboard recorded **Claude Opus 5 (High) at 30.16%** on July 24, a new ARC-AGI-3 high score; it completed five Public Demo environments no previously tested model had beaten. [ARC Prize result](https://arcprize.org/results/anthropic-claude-opus-5)
- ARC Prize published a verified **Grok 4.5** evaluation on July 16: its best ARC-AGI-3 result was **0.32% at Medium reasoning effort**. [ARC Prize result](https://arcprize.org/results/xai-grok-4-5)

## 2026-08-02 13:14 UTC -- NEW

- Kaggle’s competition host published an analysis of 500 failed submissions: roughly one-third had no traceable notebook error, nearly 20% omitted required GPU configuration, and remaining failures included missing datasets/dependencies, CUDA OOM, incorrect API endpoints, and writes to read-only `/kaggle/input`. The host also clarified that organizers cannot access notebooks before participants open-source them. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-08-03 13:17 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-04 13:14 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-05 13:17 UTC -- NEW

- The public leaderboard is now led by YUTO KOJIMA (1.86), Andy liu (1.69), and Lord Han Solo (1.65). [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
- Competition host Greg Kamradt clarified that private scores are calculated during each submission’s original run and final selections are not rerun. Each run processes both hidden halves; only the public half is displayed during competition. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)
- The official agent repository replaced `three.arcprize.org` with `arcprize.org` and corrected the API-key link to `arcprize.org/platform` on August 3. [Merged PR](https://github.com/arcprize/ARC-AGI-3-Agents/pull/74)
- Kaggle clarified execution limits: Docker logs stop being captured after 10 MB; `/kaggle/working` has a 20 GB quota; CPU submissions receive 30 GB physical RAM enforced through cgroups; and Kaggle sets neither `RLIMIT_NPROC` nor `RLIMIT_AS`. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)

## 2026-08-07 13:11 UTC -- checked, nothing new

## 2026-08-08 13:11 UTC -- NEW

- Kaggle analyzed 500 failed submissions: about one-third had no traceable visible error, nearly 20% omitted required GPU configuration, and smaller categories included missing datasets/dependencies, CUDA OOM, obsolete endpoints, and writes to read-only `/kaggle/input`. Host Greg Kamradt added that organizers cannot inspect notebooks until teams open-source them. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-08-09 13:15 UTC -- NEW

- The public leaderboard is now led by **YUTO KOJIMA** with a **1.86** score from an August 9 submission, ahead of Andy liu at 1.69. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- Greg Kamradt clarified that final-selected notebooks are **not rerun**: private scores are retained from the original submission run. Each run evaluates both leaderboard halves, while only the public half is revealed. ARC-AGI-3’s runtime limit is **9 hours**. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- Kaggle execution limits were officially clarified: logs stop being captured after 10 MB without terminating the notebook; `/kaggle/working` has a 20 GB quota and exceeding it terminates the kernel; CPU notebooks have 30 GB RAM enforced through cgroups; native crash details remain hidden although exit code 139 identifies SIGSEGV. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)

- On August 3, ARC Prize merged a correction to the official Agents repository replacing the obsolete `three.arcprize.org` defaults and API-key documentation with `arcprize.org`. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/732419), [merged PR](https://github.com/arcprize/ARC-AGI-3-Agents/pull/74)

## 2026-08-10 13:14 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-11 13:12 UTC -- NEW

Lord Han Solo moved into third place with a 1.65 score from an August 10 submission, displacing GeniusYY (1.64) to fourth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-08-12 13:13 UTC -- NEW

- Leaderboard shakeup: **cstl** now leads with **2.52**, followed by **YUTO KOJIMA** at **1.86** and **Andy liu** at **1.69**. Lord Han Solo is fourth at 1.65. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
- Organizers analyzed 500 submission errors: roughly one-third stalled without a traceable visible error, and almost 20% involved GPU-dependent notebooks submitted without GPU enabled. They also clarified that organizers cannot access notebooks until teams open-source them. [Official Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-08-13 13:17 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-15 13:17 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-16 13:16 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-17 13:15 UTC -- NEW

- **Leaderboard shakeup:** Lord Han Solo now leads at **2.76**, followed by **cstl (2.70)**, **Daniel Franzen (2.58)**, and **rellik13 (2.53)**. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
- Organizers clarified that final-selected notebooks **will not be rerun**: private scores are fixed during the original submission run. Every run evaluates both leaderboard halves, but only the public half is displayed; ARC-AGI-3’s runtime limit is **9 hours**. [Official Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)
- Kaggle is investigating **RTX Pro 6000 capacity constraints** causing queued submissions and says it is working to free additional resources. [Official Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147)

## 2026-08-18 13:16 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-19 13:15 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-20 13:12 UTC -- NEW

- **Leaderboard shakeup:** **cstl** now leads at **3.57**; **Lord Han Solo** is second at **2.76**, **The AGI Boys** third at **2.66**, **Jonathan Wang2022** fourth at **2.59**, and **Daniel Franzen** fifth at **2.58**. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-08-21 13:11 UTC -- NEW

- Leaderboard shakeup: Tufa Labs is now #2 at 2.97; Akhil Tolani entered at #4 with 2.73; AbeLincoln1865 entered at #5 with 2.72. Lord Han Solo fell to #3, while The AGI Boys, Jonathan Wang2022, and Daniel Franzen fell to #6–#8. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- Kaggle added RTX 6000 (`g4-standard-48`) machines to the competition’s hardware pool. [Competition overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/overview/description)

- Scoring clarification: a valid non-RESET action counts toward both efficiency scoring and the action budget even when it leaves the frame unchanged. The host confirmed the scored server matches offline competition mode. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/718638)

- The host published an analysis of 500 failed submissions: roughly one-third had no traceable visible notebook error, while almost 20% required a GPU but had none enabled; all other identified causes were individually below 5%. [Kaggle host post](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-08-22 13:15 UTC -- check failed

CHECK_TIMED_OUT

