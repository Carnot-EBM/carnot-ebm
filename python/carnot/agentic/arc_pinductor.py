import numpy as np
from collections import defaultdict, Counter
from .arc_world_model_synth import InducedWorldModel, _click_xy, frame_hash, grade_predictions

class PinductorModel:
    def __init__(self, game_id, latent_fn, num_latents):
        self.game_id = game_id
        self.latent_fn = latent_fn
        self.num_latents = num_latents
        self._exact = defaultdict(Counter)
        self._exact_grid = {}
        
    def fit(self, trajectories):
        for traj in trajectories:
            L = 0
            for s, akey, s2 in traj:
                s_arr = np.asarray(s, dtype=np.int16)
                s2_arr = np.asarray(s2, dtype=np.int16)
                fh = frame_hash(s_arr)
                b2 = s2_arr.astype(np.uint8).tobytes()
                self._exact[(fh, L, tuple(akey))][b2] += 1
                self._exact_grid[b2] = s2_arr
                L = self.latent_fn(L, s_arr, tuple(akey))
        return self

    def predict_belief(self, s, akey, belief):
        s_arr = np.asarray(s, dtype=np.int16)
        fh = frame_hash(s_arr)
        # Predict outcome distribution
        outcome_probs = defaultdict(float)
        for L, prob in belief.items():
            ex = self._exact.get((fh, L, tuple(akey)))
            if ex:
                total = sum(ex.values())
                for b2, count in ex.items():
                    outcome_probs[b2] += prob * (count / total)
            else:
                outcome_probs[None] += prob
        
        if not outcome_probs or (len(outcome_probs)==1 and None in outcome_probs):
            return s_arr.copy() # fallback
            
        if None in outcome_probs:
            del outcome_probs[None]
            
        best_b2 = max(outcome_probs.items(), key=lambda x: x[1])[0]
        return self._exact_grid[best_b2].copy()

    def update_belief(self, s, akey, s2, belief):
        s_arr = np.asarray(s, dtype=np.int16)
        s2_arr = np.asarray(s2, dtype=np.int16)
        b2_target = s2_arr.astype(np.uint8).tobytes()
        fh = frame_hash(s_arr)
        
        new_belief = defaultdict(float)
        for L, prob in belief.items():
            ex = self._exact.get((fh, L, tuple(akey)))
            obs_prob = 0.0
            if ex:
                total = sum(ex.values())
                obs_prob = ex.get(b2_target, 0.0) / total
            else:
                # If not seen, assume uniform small prob
                obs_prob = 1e-3
                
            next_L = self.latent_fn(L, s_arr, tuple(akey))
            new_belief[next_L] += prob * obs_prob
            
        total_p = sum(new_belief.values())
        if total_p > 0:
            for L in new_belief:
                new_belief[L] /= total_p
        else:
            # Fallback to uniform
            for L in range(self.num_latents):
                new_belief[L] = 1.0 / self.num_latents
                
        return new_belief

    def consistency_energy(self, held_out_trajectories):
        n = 0
        exact_hit = 0
        cell_acc_sum = 0.0
        dyn_acc_sum = 0.0
        n_changed = 0
        
        for traj in held_out_trajectories:
            belief = {0: 1.0}
            for s, akey, s2 in traj:
                s_arr = np.asarray(s, dtype=np.int16)
                s2_arr = np.asarray(s2, dtype=np.int16)
                pred = self.predict_belief(s_arr, akey, belief)
                
                n += 1
                if pred.shape != s2_arr.shape:
                    n_changed += 1
                    continue
                exact_hit += int(np.array_equal(pred, s2_arr))
                cell_acc_sum += float((pred == s2_arr).mean())
                real_changed = (s_arr != s2_arr)
                if real_changed.any():
                    n_changed += 1
                    union = real_changed | (s_arr != pred)
                    dyn_acc_sum += float(((pred == s2_arr) & union).sum() / union.sum())
                    
                belief = self.update_belief(s_arr, akey, s2_arr, belief)
                
        if n == 0:
            return {"energy": None, "n_heldout": 0}
        dynamics_accuracy = round(dyn_acc_sum / n_changed, 4) if n_changed else None
        return {
            "energy": round(1.0 - dynamics_accuracy, 4) if dynamics_accuracy is not None else None,
            "n_changed_transitions": n_changed,
            "transition_exact_rate": round(exact_hit / n, 4)
        }
