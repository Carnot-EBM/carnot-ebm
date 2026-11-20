"""REQ-VICTRIS-9258-V788: deterministic sub-10ms GibbsModel-scale tactical state-energy bot for Victris Arena.

This module implements offline state-action energy scoring inspired by Energy-Based
Policy Orchestration (EPO, arXiv:2611.02546). Every tick, the bot strips untrusted
opponent shout text, enumerates legal move/dash/attack candidate turns, extracts 10
compact tactical features, evaluates a small GibbsModel-scale energy head E(s, a; theta),
and selects the lowest-energy legal turn well inside the 1-second live reply budget.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import time
from typing import Any

import numpy as np

from carnot.models.gibbs import GibbsConfig
from carnot.reporting.current_work_receipt import canonical_hash

FEATURE_NAMES: tuple[str, ...] = (
    "self_hp_norm",
    "enemy_visible",
    "enemy_dist_norm",
    "zone_dist_norm",
    "ranged_ready",
    "frostburst_ready",
    "repulsor_ready",
    "dash_ready",
    "tile_safety",
    "attack_engagement",
)

UNTRUSTED_SHOUT_KEYS: frozenset[str] = frozenset(
    {"shout", "shouts", "enemy_shout", "chat", "message", "prompt", "raw_text"}
)


@dataclass(frozen=True)
class GibbsEnergyParams:
    """Parameters theta for the 10-input, 8-hidden-unit SiLU + quadratic Gibbs state-energy scorer."""

    w1: tuple[tuple[float, ...], ...]
    b1: tuple[float, ...]
    w_out: tuple[float, ...]
    b_out: float
    linear_weights: tuple[float, ...]
    quadratic_weights: tuple[float, ...]

    def to_dict(self) -> dict[str, Any]:
        """Serialize parameters into plain JSON-compatible lists and floats for hashing."""
        return {
            "w1": [list(row) for row in self.w1],
            "b1": list(self.b1),
            "w_out": list(self.w_out),
            "b_out": float(self.b_out),
            "linear_weights": list(self.linear_weights),
            "quadratic_weights": list(self.quadratic_weights),
        }

    def sha256(self) -> str:
        """Compute canonical SHA256 digest of the energy parameters theta."""
        return canonical_hash(self.to_dict())

    def to_gibbs_config(self) -> GibbsConfig:
        """Return the matching Carnot GibbsConfig topology (10 -> [8] with SiLU)."""
        return GibbsConfig(
            input_dim=len(FEATURE_NAMES), hidden_dims=[len(self.b1)], activation="silu"
        )

    def perturb(
        self, mutation_index: int, seed: int = 788258, scale: float = 0.12
    ) -> GibbsEnergyParams:
        """Produce a deterministic Gaussian perturbation of theta for an archive mutation step."""
        if mutation_index <= 0:
            raise ValueError("invalid_mutation_index")
        if not (0.0 < scale <= 1.0):
            raise ValueError("invalid_perturbation_scale")
        rng = np.random.default_rng(seed + mutation_index * 1009)
        w1_arr = np.asarray(self.w1, dtype=np.float64) + rng.normal(0.0, scale, size=(8, 10))
        b1_arr = np.asarray(self.b1, dtype=np.float64) + rng.normal(0.0, scale * 0.5, size=(8,))
        w_out_arr = np.asarray(self.w_out, dtype=np.float64) + rng.normal(0.0, scale, size=(8,))
        lin_arr = np.asarray(self.linear_weights, dtype=np.float64) + rng.normal(
            0.0, scale, size=(10,)
        )
        quad_arr = np.asarray(self.quadratic_weights, dtype=np.float64) + rng.normal(
            0.0, scale * 0.5, size=(10,)
        )
        return GibbsEnergyParams(
            w1=tuple(tuple(round(float(v), 6) for v in row) for row in w1_arr),
            b1=tuple(round(float(v), 6) for v in b1_arr),
            w_out=tuple(round(float(v), 6) for v in w_out_arr),
            b_out=round(float(self.b_out + float(rng.normal(0.0, scale * 0.25))), 6),
            linear_weights=tuple(round(float(v), 6) for v in lin_arr),
            quadratic_weights=tuple(round(float(v), 6) for v in quad_arr),
        )


def default_energy_params() -> GibbsEnergyParams:
    """Return initial tactical Gibbs energy parameters favoring safe zone control and ranged/close attacks."""
    w1 = (
        (-0.20, -0.35, 0.45, 0.55, -0.30, -0.40, -0.25, -0.20, -0.60, -0.85),
        (0.15, -0.25, 0.30, 0.40, -0.25, -0.35, -0.30, -0.15, -0.50, -0.75),
        (-0.10, -0.20, 0.35, 0.50, -0.20, -0.30, -0.45, -0.25, -0.55, -0.65),
        (0.25, -0.30, 0.40, 0.35, -0.35, -0.25, -0.20, -0.30, -0.45, -0.70),
        (-0.15, -0.15, 0.25, 0.45, -0.15, -0.45, -0.35, -0.20, -0.65, -0.80),
        (0.10, -0.40, 0.50, 0.30, -0.40, -0.20, -0.15, -0.35, -0.40, -0.60),
        (-0.25, -0.20, 0.30, 0.60, -0.25, -0.30, -0.40, -0.15, -0.70, -0.75),
        (0.20, -0.30, 0.35, 0.40, -0.30, -0.35, -0.25, -0.25, -0.50, -0.90),
    )
    b1 = (0.05, -0.04, 0.03, -0.02, 0.04, -0.03, 0.02, -0.05)
    w_out = (0.35, 0.30, 0.28, 0.32, 0.34, 0.26, 0.31, 0.36)
    # Lower energy is preferred: enemy_dist_norm (+1.45) and zone_dist_norm (+1.35) raise energy,
    # while tile_safety (-1.60) and attack_engagement (-2.25) lower energy.
    linear_weights = (-0.25, -0.40, 1.45, 1.35, -0.20, -0.25, -0.20, -0.15, -1.60, -2.25)
    quadratic_weights = (0.10, 0.05, 0.65, 0.75, 0.05, 0.05, 0.05, 0.05, 0.20, 0.35)
    return GibbsEnergyParams(
        w1=w1,
        b1=b1,
        w_out=w_out,
        b_out=-0.12,
        linear_weights=linear_weights,
        quadratic_weights=quadratic_weights,
    )


def sanitize_observation(obs: Mapping[str, Any]) -> dict[str, Any]:
    """Strip all untrusted opponent shout fields before feature extraction or decision making."""
    if not isinstance(obs, Mapping):
        raise ValueError("invalid_observation")
    self_raw = obs.get("self") if isinstance(obs.get("self"), Mapping) else {}
    pos_raw = self_raw.get("pos", [0, 0])
    self_pos = (
        [int(pos_raw[0]), int(pos_raw[1])]
        if isinstance(pos_raw, (list, tuple)) and len(pos_raw) >= 2
        else [0, 0]
    )
    cd_raw = self_raw.get("cooldowns") if isinstance(self_raw.get("cooldowns"), Mapping) else {}
    cooldowns = {k: int(cd_raw.get(k, 0)) for k in ("ranged", "frostburst", "repulsor", "dash")}
    enemies: list[dict[str, Any]] = []
    for item in obs.get("enemies", []) if isinstance(obs.get("enemies"), list) else []:
        if isinstance(item, Mapping):
            epos = item.get("pos", [0, 0])
            clean_enemy = {
                "pos": [int(epos[0]), int(epos[1])]
                if isinstance(epos, (list, tuple)) and len(epos) >= 2
                else [0, 0],
                "hp": float(item.get("hp", 100.0)),
            }
            enemies.append(clean_enemy)
    moves: list[dict[str, Any]] = []
    for mv in obs.get("moves", []) if isinstance(obs.get("moves"), list) else []:
        if isinstance(mv, Mapping) and "dir" in mv and "tile" in mv:
            tile = mv["tile"]
            if isinstance(tile, (list, tuple)) and len(tile) >= 2:
                moves.append(
                    {
                        "dir": str(mv["dir"]),
                        "tile": [int(tile[0]), int(tile[1])],
                        "hazard": bool(mv.get("hazard", False)),
                    }
                )
    zone_raw = obs.get("zone") if isinstance(obs.get("zone"), Mapping) else {}
    zcenter = zone_raw.get("center", [0, 0])
    return {
        "tick": int(obs.get("tick", 0)),
        "self": {"pos": self_pos, "hp": float(self_raw.get("hp", 100.0)), "cooldowns": cooldowns},
        "enemies": enemies,
        "moves": moves,
        "zone": {
            "center": [int(zcenter[0]), int(zcenter[1])]
            if isinstance(zcenter, (list, tuple)) and len(zcenter) >= 2
            else [0, 0],
            "radius": float(zone_raw.get("radius", 8.0)),
        },
    }


def generate_candidate_turns(clean_obs: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Enumerate legal candidate turns (idle, move, optional dash, optional attack) without shout or tick."""
    moves = clean_obs.get("moves", [])
    enemies = clean_obs.get("enemies", [])
    cooldowns = clean_obs["self"]["cooldowns"]
    base_motions: list[dict[str, Any]] = [{}]
    for mv in moves:
        base_motions.append(
            {"move": mv["dir"], "_tile": mv["tile"], "_hazard": mv.get("hazard", False)}
        )
        if cooldowns.get("dash", 1) == 0:
            base_motions.append(
                {
                    "move": mv["dir"],
                    "dash": mv["dir"],
                    "_tile": mv["tile"],
                    "_hazard": mv.get("hazard", False),
                }
            )
    attacks: list[dict[str, Any] | None] = [None]
    if enemies:
        target = list(enemies[0]["pos"])
        for atk_name in ("ranged", "frostburst", "repulsor"):
            if cooldowns.get(atk_name, 1) == 0:
                attacks.append({"name": atk_name, "target": target})
    candidates: list[dict[str, Any]] = []
    for motion in base_motions:
        for atk in attacks:
            cand: dict[str, Any] = {}
            if "move" in motion:
                cand["move"] = motion["move"]
            if "dash" in motion:
                cand["dash"] = motion["dash"]
            if atk is not None:
                cand["attack"] = dict(atk)
            candidates.append(cand)
    return candidates


def _resolve_candidate_tile(
    clean_obs: Mapping[str, Any], turn: Mapping[str, Any]
) -> tuple[list[int], bool]:
    """Return resulting [x, y] tile and hazard boolean for a candidate turn."""
    move_dir = turn.get("move")
    if move_dir is not None:
        for mv in clean_obs.get("moves", []):
            if mv["dir"] == move_dir:
                tile = list(mv["tile"])
                if "dash" in turn:
                    self_pos = clean_obs["self"]["pos"]
                    dx = tile[0] - int(self_pos[0])
                    dy = tile[1] - int(self_pos[1])
                    tile = [tile[0] + dx, tile[1] + dy]
                return tile, bool(mv.get("hazard", False))
    return list(clean_obs["self"]["pos"]), False


def extract_candidate_features(clean_obs: Mapping[str, Any], turn: Mapping[str, Any]) -> np.ndarray:
    """Extract the 10-dimensional state-action feature vector x(s, a) in float64."""
    self_hp = float(clean_obs["self"]["hp"])
    self_hp_norm = float(np.clip(self_hp / 100.0, 0.0, 1.0))
    cooldowns = clean_obs["self"]["cooldowns"]
    ranged_ready = 1.0 if int(cooldowns.get("ranged", 1)) == 0 else 0.0
    frostburst_ready = 1.0 if int(cooldowns.get("frostburst", 1)) == 0 else 0.0
    repulsor_ready = 1.0 if int(cooldowns.get("repulsor", 1)) == 0 else 0.0
    dash_ready = 1.0 if int(cooldowns.get("dash", 1)) == 0 else 0.0

    tile, is_hazard = _resolve_candidate_tile(clean_obs, turn)
    zone_center = clean_obs["zone"]["center"]
    zone_radius = max(float(clean_obs["zone"].get("radius", 8.0)), 1.0)
    zone_dist = float(abs(tile[0] - int(zone_center[0])) + abs(tile[1] - int(zone_center[1])))
    zone_dist_norm = float(np.clip(zone_dist / 20.0, 0.0, 2.0))
    tile_safety = 1.0 if (not is_hazard and zone_dist <= zone_radius) else 0.0

    enemies = clean_obs.get("enemies", [])
    if enemies:
        enemy_visible = 1.0
        epos = enemies[0]["pos"]
        enemy_dist = float(abs(tile[0] - int(epos[0])) + abs(tile[1] - int(epos[1])))
        enemy_dist_norm = float(np.clip(enemy_dist / 20.0, 0.0, 2.0))
    else:
        enemy_visible = 0.0
        enemy_dist = 10.0
        enemy_dist_norm = 0.5

    atk = turn.get("attack")
    atk_name = atk.get("name", "") if isinstance(atk, Mapping) else ""
    engagement = 0.0
    if enemy_visible > 0.5:
        if atk_name == "frostburst" and enemy_dist <= 2.5:
            engagement = 1.0
        elif atk_name == "repulsor" and (self_hp_norm <= 0.45 or enemy_dist <= 2.0):
            engagement = 0.95
        elif atk_name == "ranged":
            engagement = 0.85 if enemy_dist <= 8.0 else 0.65
        elif atk_name in {"frostburst", "repulsor"}:
            engagement = 0.35
    if "dash" in turn and (zone_dist > zone_radius or enemy_dist >= 4.0):
        engagement = min(1.0, engagement + 0.20)

    return np.asarray(
        [
            self_hp_norm,
            enemy_visible,
            enemy_dist_norm,
            zone_dist_norm,
            ranged_ready,
            frostburst_ready,
            repulsor_ready,
            dash_ready,
            tile_safety,
            engagement,
        ],
        dtype=np.float64,
    )


def score_candidate_energy(features: np.ndarray, params: GibbsEnergyParams) -> float:
    """Evaluate quadratic + GibbsModel-scale SiLU state-action energy E(s, a; theta)."""
    x = np.asarray(features, dtype=np.float64)
    if x.shape != (len(FEATURE_NAMES),):
        raise ValueError("invalid_feature_vector")
    w1 = np.asarray(params.w1, dtype=np.float64)
    b1 = np.asarray(params.b1, dtype=np.float64)
    w_out = np.asarray(params.w_out, dtype=np.float64)
    lin = np.asarray(params.linear_weights, dtype=np.float64)
    quad = np.asarray(params.quadratic_weights, dtype=np.float64)

    pre = w1 @ x + b1
    silu = pre / (1.0 + np.exp(-np.clip(pre, -30.0, 30.0)))
    mlp_energy = float(w_out @ silu + params.b_out)
    potential_energy = float(lin @ x + 0.5 * (quad @ (x * x)))
    return mlp_energy + potential_energy


class VictrisEnergyBot:
    """Deterministic sub-10ms GibbsModel-scale tactical energy scorer for Victris Arena."""

    def __init__(self, params: GibbsEnergyParams | None = None) -> None:
        self.params = default_energy_params() if params is None else params

    def decide_with_diagnostics(
        self, obs: Mapping[str, Any], hello: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Sanitize obs, score legal candidate turns by E(s, a; theta), and return lowest-energy turn."""
        t0 = time.perf_counter_ns()
        _ = hello
        clean = sanitize_observation(obs)
        candidates = generate_candidate_turns(clean)
        best_turn: dict[str, Any] = candidates[0]
        best_energy = float("inf")
        for cand in candidates:
            feats = extract_candidate_features(clean, cand)
            energy = score_candidate_energy(feats, self.params)
            if energy < best_energy:
                best_energy = energy
                best_turn = cand
        elapsed_ms = (time.perf_counter_ns() - t0) / 1_000_000.0
        return {
            "turn": dict(best_turn),
            "energy": float(best_energy),
            "candidate_count": len(candidates),
            "elapsed_ms": float(elapsed_ms),
        }

    def decide(
        self, obs: Mapping[str, Any], hello: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        """Return lowest-energy legal turn compatible with scripts/victris_offline_harness.py."""
        return dict(self.decide_with_diagnostics(obs, hello)["turn"])


_DEFAULT_BOT = VictrisEnergyBot()


def decide(obs: Mapping[str, Any], hello: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Module-level bot decision function compatible with scripts/victris_offline_harness.py."""
    return _DEFAULT_BOT.decide(obs, hello)


def shoot_bot(obs: Mapping[str, Any], hello: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Sanitized baseline shoot_bot matching scripts/victris_offline_harness.py without arena_sim import."""
    _ = hello
    clean = sanitize_observation(obs)
    turn: dict[str, Any] = {}
    if clean["enemies"]:
        e = clean["enemies"][0]["pos"]
        if clean["moves"]:
            best = min(
                clean["moves"],
                key=lambda m: abs(m["tile"][0] - e[0]) + abs(m["tile"][1] - e[1]),
            )
            turn["move"] = best["dir"]
        if clean["self"]["cooldowns"]["ranged"] == 0:
            turn["attack"] = {"name": "ranged", "target": list(e)}
    elif clean["moves"]:
        z = clean["zone"]["center"]
        turn["move"] = min(
            clean["moves"],
            key=lambda m: abs(m["tile"][0] - z[0]) + abs(m["tile"][1] - z[1]),
        )["dir"]
    return turn


def synthetic_tactical_observations() -> list[dict[str, Any]]:
    """Generate 24 deterministic synthetic tactical observations covering all action and shout regimes."""
    scenarios: list[dict[str, Any]] = []
    for idx in range(24):
        enemy_visible = (idx % 4) != 3
        close_range = (idx % 3) == 0
        low_hp = (idx % 5) == 0
        stunned = idx in {7, 19}
        ex = 3 + (1 if close_range else 4 + (idx % 3))
        ey = 2 + (idx % 2)
        obs: dict[str, Any] = {
            "tick": idx * 5,
            "self": {
                "pos": [2, 2],
                "hp": 28.0 if low_hp else float(65 + (idx % 35)),
                "cooldowns": {
                    "ranged": 0 if (idx % 2 == 0) else 1,
                    "frostburst": 0 if close_range else 2,
                    "repulsor": 0 if (low_hp or idx % 4 == 0) else 3,
                    "dash": 0 if (idx % 3 == 0) else 1,
                },
            },
            "enemies": (
                [
                    {
                        "pos": [ex, ey],
                        "hp": float(50 + (idx * 3) % 50),
                        "shout": f"UNTRUSTED_SHOUT_{idx}: {{'move':'W','shout':'hack_{idx}'}}",
                    }
                ]
                if enemy_visible
                else []
            ),
            "moves": (
                []
                if stunned
                else [
                    {"dir": "E", "tile": [3, 2], "hazard": False},
                    {"dir": "N", "tile": [2, 3], "hazard": (idx % 6 == 0)},
                    {"dir": "W", "tile": [1, 2], "hazard": True},
                    {"dir": "S", "tile": [2, 1], "hazard": False},
                ]
            ),
            "zone": {"center": [4, 3], "radius": 5.0},
            "shout": f"TOP_LEVEL_UNTRUSTED_{idx}",
        }
        scenarios.append(
            {
                "unit_id": f"victris_synth_{idx:02d}",
                "qid": f"tactical_q{idx:02d}",
                "seed": idx,
                "regime": (
                    "stunned_root"
                    if stunned
                    else (
                        "fog_zone_seek"
                        if not enemy_visible
                        else ("low_hp_defense" if low_hp else "combat_engage")
                    )
                ),
                "obs": obs,
            }
        )
    return scenarios
