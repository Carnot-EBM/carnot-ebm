# V652 ARC supervisor evidence

Experiment 7444 reads the two completed Experiment 7431 episodes. It does not
run a game or invoke a model. Both archived episodes used the scored policy with
per-game adapters and saved solutions disabled. Each recorded 62 actions, zero
banked progress, shadow supervisor mode, and zero supervisor firings.

The shipped firing threshold is 120 stagnant actions. The two 62-action rows
cannot reach that threshold. Experiment 7431 preserved a supervisor summary but
did not preserve detailed timestamped window rows in its terminal episode row.
The missing window contents therefore remain unknown. No window was invented.

Zero firings provide no evidence that an arm helped or failed. No arm is
promoted or retired. The fourth curated arm, `tool_loop_reinduction`, was not
enabled. A new arm is not justified. That conclusion would require all four
curated arms to fire on one level stretch and a later recorded exhausted window.

The next live comparison must keep adapters withheld, use the shipped 120-action
threshold, record an actual curated-arm firing, state whether it was applied,
and retain the later transient and banked outcome. The live policy remains
unchanged.
