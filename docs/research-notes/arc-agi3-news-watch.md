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

## 2026-08-23 13:14 UTC -- NEW

- Leaderboard reshuffle: cstl leads at 3.57; Lord Han Solo rose to #2 at 3.36; Tufa Labs slipped to #3 despite improving to 3.04; Daniel Franzen climbed from #8 to #4 at 2.88; wking edewd entered #7 at 2.70. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- The host clarified that private scores are computed during the original submission run and are not rerun when final submissions are selected. Each run evaluates both public and private datasets, with 50% of tasks contributing to the visible public leaderboard; the ARC-AGI-3 notebook limit is nine hours. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- New runtime-limit clarification: Docker logs are capped at 10 MB; `/kaggle/working` has a 20 GB quota; CPU notebooks have 30 GB physical RAM enforced through cgroups; Kaggle sets neither `RLIMIT_AS` nor an explicit process/thread limit. Native crashes expose exit codes and stderr, but not core dumps. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)

- Kaggle confirmed it was investigating capacity constraints affecting the RTX Pro 6000 pool and working to free additional resources. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147)

## 2026-08-24 13:13 UTC -- NEW

- Tufa Labs jumped from #3 (3.04) to #1 with 4.58. cstl is now #2 (3.57), Lord Han Solo #3 (3.36), and Tony G entered #4 at 3.17, pushing Daniel Franzen to #5. wking edewd fell from #7 to #10. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-08-25 13:16 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-26 13:11 UTC -- NEW

- The leaderboard changed again: cstl is now #1 (5.99), Lord Han Solo #2 (4.99), and Tufa Labs #3 (4.67). Tong Hui Kang entered #4 (3.39), pushing Tony G to #5 (3.17), Daniel Franzen to #6 (2.88), and wking edewd to #11 (2.70). [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-08-27 13:14 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-28 13:14 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-29 13:15 UTC -- check failed

CHECK_TIMED_OUT

## 2026-08-30 13:14 UTC -- NEW

- Leaderboard shakeup: Tufa Labs improved to 4.71 (#3); Tong Hui Kang to 4.27 (#4); rfbr entered #5 at 3.37; Youssef Nader @ Aurelic is #6 at 3.19; Tony G fell to #7; Daniel Franzen is #8 at 3.15; Liao Zixu and sawada entered #9–10. wking edewd dropped to #18. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- On July 17, the host published an analysis of 500 failed submissions: roughly one-third had no traceable visible notebook error, nearly 20% required a GPU that had not been enabled, and remaining causes—each below 5%—included missing datasets/dependencies, logic errors, CUDA OOM, using the public API instead of competition settings, and writing to read-only `/kaggle/input`. [Official Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-08-31 13:13 UTC -- check failed

CHECK_TIMED_OUT

## 2026-09-01 13:15 UTC -- NEW

- Leaderboard shakeup: Tong Hui Kang improved to 4.45 (#4); Son Pham & Mark Barney entered #5 at 4.42; Daniel Franzen rose to #6 at 4.05; Nithin Kumar R C and Ebi entered #7–8 at 3.87/3.85; Youssef Nader improved to 3.67 but moved to #9; Jonathan Wang2022 entered #10 at 3.39. sawada, Tony G, and Liao Zixu are now #12, #15, and #17. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- On August 14, Kaggle staff acknowledged possible RTX 6000 pool capacity constraints and said they were working to free additional resources. [Official Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147)

- On August 3, the official Agents toolkit replaced `three.arcprize.org` references with `arcprize.org` and corrected the API-key URL to `arcprize.org/platform`. [Official repository change](https://github.com/arcprize/ARC-AGI-3-Agents/pull/74)

## 2026-09-02 13:17 UTC -- check failed

CHECK_TIMED_OUT

## 2026-09-03 12:55 UTC -- NEW

- Major leaderboard shakeup as of September 3: mostik.ai leads with 7.51, followed by seele (5.53), jinbo wang1 (5.49), Fususu (5.43), and Lord Han Solo (4.99). Tufa Labs is #6, Shuhan Yang #7, Son Pham & Mark Barney #8 (4.52), Tong Hui Kang #9, and MIKE DAN7 #10. Daniel Franzen fell to #14 and Nithin Kumar R C to #17; Ebi, Youssef Nader, and Jonathan Wang2022 are outside the top 20. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-09-03 13:15 UTC -- NEW

- Final scoring clarified: selected final submissions keep the private scores calculated during their original runs; notebooks are not rerun. Each run evaluates both hidden splits, while only the semi-private half appears on the public leaderboard. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- The official agent toolkit switched its API host from `three.arcprize.org` to `arcprize.org` on August 3; using the old endpoint is a documented submission-failure cause. [Merged toolkit change](https://github.com/arcprize/ARC-AGI-3-Agents/pull/74), [Kaggle announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

- Kaggle analyzed 500 failed submissions: roughly one-third stalled without a visible notebook error, and almost 20% required a GPU that was not enabled. All other identified causes were individually below 5%. ARC Prize also confirmed it cannot inspect private notebooks before teams open-source them. [Kaggle announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

- Kaggle disclosed additional runtime limits: 20 GB for `/kaggle/working`, 30 GB physical RAM for CPU notebooks, and 10 MB of captured container logs; exceeding the log cap truncates output rather than terminating the notebook. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/724841)

## 2026-09-04 13:17 UTC -- NEW

- On September 3, ARC Prize reported GPT-6 Astra scoring 62.7% on the semi-private set with its standard harness and 99.9% with a provider-adapter harness. Future ARC leaderboards will distinguish these harness types. These were separate benchmark evaluations, not Kaggle submissions. [ARC Prize announcement](https://arcprize.org/blog/astra)

- The Kaggle public leaderboard has changed substantially: mostik.ai leads at 7.51%, followed by Third Intelligence at 6.43% and jinbo wang1 at 5.49%; Milestone #1 winner Tufa Labs is eighth at 4.71%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- On August 14, Kaggle staff acknowledged potential RTX 6000 pool capacity constraints and said they were working to release additional resources. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/735147)

## 2026-09-05 13:09 UTC -- checked, nothing new

## 2026-09-06 13:10 UTC -- NEW

- Public leaderboard shakeup: Daniel Franzen now leads with 7.63%, ahead of mostik.ai (7.51%) and Third Intelligence (6.43%). Youssef Nader @ Aurelic rose to sixth (5.05%), Kyutai entered eighth (4.90%), and Tufa Labs dropped to eleventh. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)
## 2026-09-08 13:13 UTC -- checked, nothing new

## 2026-09-09 13:14 UTC -- checked, nothing new

## 2026-09-10 13:15 UTC -- checked, nothing new

## 2026-09-11 13:12 UTC -- NEW

- NVARC3 entered third place at 7.69%, pushing Daniel Franzen to fourth and mostik.ai to fifth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- Kaggle said recent queue-capacity backups have cleared. A finished/scored submission still displayed as “queued” is a UI-status glitch; its score remains valid. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739674)

## 2026-09-12 13:19 UTC -- NEW

- ARC Prize reported GPT-6 Astra scoring 62.7% on ARC-AGI-3 Semi-Private with its Standard harness and 99.9% with a Provider Adapter harness. ARC will now label both harness types separately on its benchmark leaderboard. This was a separate benchmark evaluation, not a Kaggle submission. [ARC Prize announcement](https://arcprize.org/blog/astra)

- Kaggle leaderboard update: Tufa Labs now leads with 11.04%; NVARC3 improved from 7.69%/third to 8.40%/second; Third Intelligence is third at 8.21%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- The competition host clarified that private scores are calculated during each submission’s original run and selected final submissions will not be rerun. Each run evaluates both hidden datasets; only the public half is displayed during the competition. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- Kaggle analyzed 500 failed submissions: roughly one-third had no notebook-visible traceable error, while almost 20% required a GPU that had not been enabled. [Official Kaggle post](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-09-13 13:13 UTC -- NEW

Ebi moved into second place at 8.68%, pushing NVARC3 to third and Third Intelligence to fourth. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-09-14 13:14 UTC -- NEW

- Tufa Labs has taken #1 at 18.81%; Lord Han Solo is now #3 at 8.44%, while NVARC3 and Third Intelligence have moved to #4 at 8.40% and #5 at 8.21%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- ARC Prize reported GPT-6 Astra scoring 62.7% on Semi-Private with the Standard harness and 99.9% with a Provider Adapter harness. Future verified leaderboard results will label these harness types separately. [ARC Prize announcement](https://arcprize.org/blog/astra)

- The host clarified that final-selected notebooks will not be rerun: private scores are fixed during each submission’s original run. Every run plays both hidden datasets, although only the public half affects the visible leaderboard. [Kaggle clarification](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- Kaggle acknowledged RTX 6000 capacity constraints in August; on September 10 it said the queue backups had cleared and confirmed that a submission shown as “queued” after finishing/scoring is experiencing a display glitch—the score remains valid. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/739674)

## 2026-09-15 13:12 UTC -- checked, nothing new

## 2026-09-16 13:12 UTC -- NEW

- Leaderboard shakeup: Lord Han Solo rose to #2 with 8.84%, while Ebi is now #3 with 8.68%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-09-17 13:15 UTC -- NEW

- Leaderboard shakeup as of September 17: Tufa Labs leads with 18.81%, followed by Daniel Franzen at 11.59% and NVARC3 at 11.04%. Lord Han Solo is now #4 at 9.81%; Ebi is #6 at 8.68%. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- Final-scoring clarification: private scores are calculated during each submission’s original run; selected final submissions will not be rerun. Each run evaluates both leaderboard subsets, while only the public half is shown during competition. The runtime limit is 9 hours. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- ARC Prize reported GPT-6 Astra scoring 62.7% on the Semi-Private set with its Standard harness and 99.9% with a Provider Adapter harness. Future ARC-AGI leaderboard reporting will label both harness types separately. [ARC Prize announcement](https://arcprize.org/blog/astra)

- The competition host analyzed 500 failed submissions: roughly one-third had no traceable visible notebook error, while almost 20% required a GPU but had not enabled one. [Kaggle host announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-09-19 13:17 UTC -- NEW

- Leaderboard shakeup: **Matija Ludvig & Zhongwei Wang** moved into #3 at **11.64%**; **Lord Han Solo** is #4 at **11.54%**, and **NVARC3** fell to #5 at **11.04%**. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

- ARC Prize reported GPT-6 Astra scoring **62.7%** on Semi-Private with the Standard harness and **99.9%** with the Provider Adapter harness. ARC Prize will now label and report both harness conditions separately. [Official announcement](https://arcprize.org/blog/astra)

- The host clarified final scoring: selected final notebooks are **not rerun**; their private scores are computed during the original submission run. Each run plays all 110 hidden environments, although only the public half determines the visible leaderboard score. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- Kaggle analyzed 500 failed submissions: roughly one-third had no traceable visible error, while nearly 20% involved GPU-dependent code submitted without enabling a GPU. [Host announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

## 2026-09-21 13:13 UTC -- NEW

- ARC Prize reported GPT-6 Astra scoring 62.7% on the Semi-Private set with its Standard harness and 99.9% with a Provider Adapter harness. ARC will now label both harness categories separately on its leaderboard. [ARC Prize announcement](https://arcprize.org/blog/astra)

- Lord Han Solo improved from 15.60% to 18.42%, moving from third to second place and overtaking NVARC3. [Kaggle leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/leaderboard)

## 2026-09-22 13:17 UTC -- NEW

- On July 27, organizers clarified that final-selected submissions are **not rerun**; their private scores are fixed during the original scoring run. Each submission evaluates against all 110 environments, while only the public-half score is displayed during the competition. [Kaggle discussion](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/729985)

- On July 17, ARC Prize published an analysis of 500 failed submissions: roughly one-third had no traceable notebook error, nearly 20% requested GPU workloads without enabling a GPU, and smaller categories included missing datasets/dependencies, CUDA OOM, incorrect API endpoints, and writes to read-only `/kaggle/input`. Organizers also confirmed that submission logs are intentionally restricted and private notebooks remain inaccessible to them until open-sourced. [Kaggle announcement](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/discussion/727119)

