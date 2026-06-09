from carnot.agentic.arc_pinductor import PinductorModel
import numpy as np

def step_fn(L, s, a):
    return (L + 1) % 2
    
def dummy_traj():
    # s0 -> s0 -> s1 -> s1
    # 0,0  0,0    1,1   1,1
    s0 = np.zeros((2,2), dtype=int)
    s1 = np.ones((2,2), dtype=int)
    a = (6, 0, 0)
    return [
        (s0, a, s0),
        (s0, a, s1),
        (s1, a, s1),
        (s1, a, s0)
    ]

m = PinductorModel("test", step_fn, 2)
m.fit([dummy_traj()])
print(m.consistency_energy([dummy_traj()]))
