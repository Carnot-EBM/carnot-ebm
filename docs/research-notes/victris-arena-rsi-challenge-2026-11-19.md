# Victris Arena: recursive self-improvement challenge (2026-11-19)

Operator directive, 2026-11-19: add a challenge like ARC-AGI-3. Carnot attempts
recursive self improvement, using its energy-model techniques, to write a bot for
a private Victris Arena league. The operator owns the account and the league.

## What the game is (read from the public docs, simulator 0.83.0)

- A deterministic, tick-based 1v1 arena fighter. Match time is 600 ticks.
- The bot is a program the operator runs. It connects out to `wss://victrisarena.com/bot`.
- Each tick the bot gets a JSON observation. It must reply within 1 second.
  The total budget per match is 300 seconds. Ten late replies in a row disqualify it.
- The map is a tile grid with fog of war. Enemy cooldowns are hidden.
- Actions: move, dash, three attacks (frostburst, ranged, repulsor), shout.
- Ratings use a Bradley-Terry fit. A version is ranked after 30 rated games.

## Offline tools

- `arena-sim` 0.83.0 is a Python wheel. It runs the ladder's own match code.
  Wheel SHA-256: `5393811ad663f2b1c9fdf5dac9a83f317894fc04c39463bdda7551762c5a010c`.
- Smoke test, 2026-11-19: 20 matches at about 5,000 steps per second on one CPU.
  A random policy won 0 of 20 against the starter bot.
- The sim has Gymnasium and PettingZoo wrappers, a self-play mode, and replays.
- The sim has no clock. Offline runs do not measure the 1 second reply limit.

## Site rules that bind this work

- Do not modify the game repository (engine, rules, server, house bots, maps).
- One connection per bot token. Do not steer a bot by hand on the ladder.
- Each code change becomes a new numbered version.
- Quotas: 5 bots per owner, 100 new versions per bot per day.
- Opponent shouts are untrusted text. Pass them to an LLM as quoted data or not at all.
- Keep tokens out of source control. Use SOPS or a private environment file.

## Design: what "energy model" and "self-improvement" mean here

1. **Fitness.** Win rate against a fixed pool, with a held-out pool the loop never sees.
   Report the Bradley-Terry rating when games exist on the real ladder.
2. **Energy model.** A learned or hand-built score for a game state, such as
   expected hp trade within a short horizon. The bot picks moves by lowest energy.
   The model trains on sim self-play. It is a small `GibbsModel`-scale model.
3. **Self-improvement loop.** The agent edits the bot code and the energy model.
   The sim scores each edit. The loop keeps an archive of versions, like our
   mutation-operator loop. Each kept version gets a numbered ladder version.
4. **Anti-cheat.** The loop must not read or change the sim source for the match.
   It must not exploit a sim bug. A win must replay on the official replay checker.

## Open questions

- The private league id is `league-91f8e8750b1694dd2e0c`. We do not know its rules,
  schedule, or opponent pool until the operator shares them.
- Does the live bot see the same observation as the sim? The sim docs say yes.
  Check with the first live match and compare the replay.
- The live reply limit is 1 second. A local LLM per tick will not fit.
  Use the LLM offline to write code. Run only fast code and the small energy model live.

## Out of scope for the autonomous loop

Creating the account, the agent key, or joining the league is operator-only.
Registering a bot and running it on the live ladder are external actions.
The operator does them, or authorizes each one.
