"""LSEBMCL replay buffer for continual learning across constraint template sessions.

Why this exists:
    The ConstraintTemplateLibrary accumulates constraint templates across sessions.
    Without replay, training on session N templates overwrites session N-1 knowledge —
    catastrophic forgetting (arXiv 2501.05495, §3.1).

    LSEBMCL (Latent Space EBM for Continual Learning) fixes this by storing a compact
    snapshot of each session's constraint patterns and replaying them alongside new data.
    Because the replay set is drawn from the real EBM energy landscape of prior sessions,
    the model cannot drift away from previously learned constraints.

    This module is the replay-buffer half of LSEBMCL. It does not train an EBM itself;
    it stores pattern snapshots and exposes them for injection into the next training round.

Spec: REQ-SELF-021, SCENARIO-SELF-027, SCENARIO-SELF-028
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


@dataclass
class ReplaySession:
    """Snapshot of one session's constraint patterns plus their mean EBM energy.

    Why store mean_energy: it lets callers decide whether a session's patterns were
    high-energy (many violations) or low-energy (constraints mostly satisfied), without
    replaying all patterns in memory-constrained settings.
    """

    session_id: int
    n_templates: int
    template_patterns: list[str]
    ebm_energy_mean: float


class LSEBMCLReplayBuffer:
    """Replay buffer that prevents catastrophic forgetting of constraint templates.

    How it works (layman version):
        1. After each training session, call add_session() with the patterns seen.
           The buffer saves up to max_replay_per_session of them and records their
           average EBM energy score.
        2. Before training on session N, call generate_replay(N).  You get back all
           patterns from sessions 0…N-1, which you mix into the new training data.
        3. The model sees old patterns alongside new ones, so it cannot forget them.

    The forgetting_rate metric (compute_forgetting_rate) measures what fraction of
    prior-session patterns are NOT covered by replay.  A rate < 0.05 means the buffer
    is retaining ≥95% of knowledge — the LSEBMCL success criterion from arXiv 2501.05495.

    Spec: REQ-SELF-021
    """

    def __init__(
        self,
        energy_fn: Callable[[str], float],
        max_replay_per_session: int = 10,
    ) -> None:
        self.energy_fn = energy_fn
        self.max_replay = max_replay_per_session
        self.sessions: list[ReplaySession] = []

    def add_session(self, session_id: int, patterns: list[str]) -> ReplaySession:
        """Register a new session's constraint patterns in the replay buffer.

        Computes the mean EBM energy over all patterns (not just the truncated replay
        set) so the energy statistic reflects the full session, not a biased sample.
        """
        energies = [self.energy_fn(p) for p in patterns]
        mean_energy = sum(energies) / max(len(energies), 1)
        session = ReplaySession(
            session_id=session_id,
            n_templates=len(patterns),
            template_patterns=patterns[: self.max_replay],
            ebm_energy_mean=mean_energy,
        )
        self.sessions.append(session)
        return session

    def generate_replay(self, current_session_id: int) -> list[str]:
        """Return replay patterns from all sessions that predate current_session_id.

        Only sessions with session_id < current_session_id are included — we do not
        replay the current session because its patterns are already in the training batch.
        """
        replay: list[str] = []
        for session in self.sessions:
            if session.session_id < current_session_id:
                replay.extend(session.template_patterns[: self.max_replay])
        return replay

    def compute_forgetting_rate(self, session_patterns: list[list[str]]) -> float:
        """Measure what fraction of prior-session patterns are lost without replay.

        Algorithm:
            For each session i (1..N-1), compute which patterns from session i-1 appear
            in the replay set available at session i.  The forgotten fraction is the
            complement.  We aggregate across all consecutive pairs.

        A rate of 0.0 means perfect recall; a rate of 1.0 means total forgetting.
        The LSEBMCL success criterion is rate < 0.05 (arXiv 2501.05495, §4.3).
        """
        if len(session_patterns) < 2:
            return 0.0

        n_forgotten = 0
        n_total = 0
        for i in range(1, len(session_patterns)):
            prior_patterns = set(session_patterns[i - 1])
            replay = set(self.generate_replay(i))
            still_remembered = prior_patterns.intersection(replay)
            n_forgotten += len(prior_patterns) - len(still_remembered)
            n_total += len(prior_patterns)
        return n_forgotten / max(n_total, 1)
