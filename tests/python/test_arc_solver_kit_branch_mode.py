"""Unit test for OfflineSolver's deepcopy-per-node branch mode
(python/carnot/agentic/arc_solver_kit.py).

The default "replay" mode navigates between search nodes by replay-from-reset; the new "deepcopy" mode
snapshots/restores copy.deepcopy(env._game) per node (for games where replay-from-reset doesn't
faithfully reproduce the searched state). On a deterministic toy game the two modes MUST find the same
(reproducible) solution — this pins that the deepcopy branching is correct + the default is unchanged.

Spec: REQ-PHASE4-081, SCENARIO-PHASE4-081 (the ARC solve infrastructure).
"""
from carnot.agentic.arc_solver_kit import OfflineSolver


class _Frame:
    def __init__(self, pos):
        self.pos = pos
        self.levels_completed = 1 if pos >= 3 else 0   # win at pos 3


class _Game:
    __slots__ = ("pos",)

    def __init__(self, pos=0):
        self.pos = pos


class _LineEnv:
    """A 1-D toy 'game': step 'R'/'L' moves a position (clamped >=0); level-up at pos 3. All state is
    in env._game.pos, so it is both replay-reconstructable AND deepcopy-injectable."""

    def __init__(self):
        self._game = _Game(0)

    def reset(self):
        self._game = _Game(0)
        return _Frame(0)

    def step(self, label, **kw):
        self._game.pos = max(0, self._game.pos + (1 if label == "R" else -1))
        return _Frame(self._game.pos)


def _labels(env, frame=None, path=None):
    return ["R", "L"]


def _apply(env, label, frame):
    return env.step(label)


def _state_key(game, frame=None):
    return game.pos


def _verifier(game, frame=None):
    return float(3 - game.pos)     # distance to the goal (lower == closer)


def _solve(mode):
    env = _LineEnv()
    solver = OfflineSolver("toy", _labels, _apply, _state_key,
                           verifier=_verifier, branch_mode=mode)
    path, nodes = solver.solve_level(env, 0, [], depth_cap=10)
    # replay the found path on a fresh env -> it must actually win (reproduction check)
    f = env.reset()
    for lab in (path or []):
        f = _apply(env, lab, f)
    return path, f.levels_completed


def test_default_branch_mode_is_replay():
    s = OfflineSolver("toy", _labels, _apply, _state_key)
    assert s.branch_mode == "replay"


def test_deepcopy_mode_matches_replay_and_reproduces():
    rep_path, rep_win = _solve("replay")
    dc_path, dc_win = _solve("deepcopy")
    assert rep_win == 1 and dc_win == 1                 # both reach the win, reproducibly
    assert rep_path == ["R", "R", "R"]                  # verifier-routed shortest path
    assert dc_path == rep_path                          # deepcopy branching == replay branching


# ---- fresh_env mode: the fix for a NON-IDEMPOTENT reset (gotcha #7, tu93) ----
class _PFrame:
    def __init__(self, pos, win):
        self.pos = pos
        self.levels_completed = 1 if win else 0


class _PGame:
    __slots__ = ("pos",)

    def __init__(self, pos=0):
        self.pos = pos


class _PristineEnv:
    """Toy whose reset() is NON-IDEMPOTENT: each reset increments n_resets, and a win (pos>=2) only
    registers while the env is PRISTINE (n_resets==1). A reuse-one-env 'replay' search resets repeatedly
    (n_resets climbs past 1) so it can NEVER observe the win; a 'fresh_env' search evaluates each
    candidate on a brand-new env (n_resets==1) so it finds -- and reproduces -- the win. This is the
    unit-test analogue of tu93's parity-toggling reset (gotcha #7)."""

    def __init__(self):
        self._game = _PGame(0)
        self.n_resets = 0

    def reset(self):
        self._game = _PGame(0)
        self.n_resets += 1
        return _PFrame(0, False)

    def step(self, label, **kw):
        self._game.pos = max(0, self._game.pos + (1 if label == "R" else -1))
        return _PFrame(self._game.pos, self._game.pos >= 2 and self.n_resets == 1)


def _solve_pristine(mode):
    # env_factory mints pristine envs for fresh_env mode (the default factory is offline_arcade, which
    # only knows real games); replay/deepcopy ignore it and reuse the passed env.
    solver = OfflineSolver("pristine", _labels, _apply, _state_key, verifier=_verifier,
                           branch_mode=mode, env_factory=_PristineEnv)
    path, _ = solver.solve_level(_PristineEnv(), 0, [], depth_cap=6)
    g = _PristineEnv()                              # reproduction gate: replay on a brand-new env
    f = g.reset()
    for lab in (path or []):
        f = _apply(g, lab, f)
    return path, f.levels_completed


def test_fresh_env_mode_solves_nonidempotent_reset_game():
    # replay reuses one env -> reset count climbs past 1 -> the pristine-only win is never observable
    rep_path, rep_win = _solve_pristine("replay")
    assert rep_path is None and rep_win == 0
    # fresh_env evaluates each candidate on a brand-new env -> finds AND reproduces the win
    fe_path, fe_win = _solve_pristine("fresh_env")
    assert fe_win == 1 and fe_path == ["R", "R"]
