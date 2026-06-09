from .arc_world_model_synth import _click_xy

def get_candidates():
    candidates = []
    
    # 1. Step counters
    for k in [2, 3, 4, 5, 8]:
        def make_step(K):
            return lambda L, s, a: (L + 1) % K
        candidates.append((f"step_mod_{k}", make_step(k), k))
        
    # 2. Color click counters
    for k in [2, 3]:
        for color in range(10):
            def make_color(K, C):
                def fn(L, s, a):
                    xy = _click_xy(a)
                    if xy is not None:
                        x, y = xy
                        if 0 <= y < s.shape[0] and 0 <= x < s.shape[1]:
                            if s[y, x] == C:
                                return (L + 1) % K
                    return L
                return fn
            candidates.append((f"color_click_{color}_mod_{k}", make_color(k, color), k))
            
    # 3. Any click counter
    for k in [2, 3]:
        def make_any_click(K):
            def fn(L, s, a):
                if _click_xy(a) is not None:
                    return (L + 1) % K
                return L
            return fn
        candidates.append((f"any_click_mod_{k}", make_any_click(k), k))

    # 4. Action type counter
    for k in [2, 3]:
        for atype in [1, 2, 3, 4, 5]: # standard actions
            def make_action(K, A):
                def fn(L, s, a):
                    if a[0] == A:
                        return (L + 1) % K
                    return L
                return fn
            candidates.append((f"action_type_{atype}_mod_{k}", make_action(k, atype), k))
            
    return candidates
