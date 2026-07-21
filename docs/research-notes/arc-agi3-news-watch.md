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

