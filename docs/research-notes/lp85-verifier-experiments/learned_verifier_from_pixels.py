"""(b) LEARNED verifier, step 1: prove the distance-to-win verifier is learnable from PIXELS ALONE
(no source/sprite access -- the real submission-time condition). Collect (grid, source_distance) over
L1+L2 play, train a small CPU CNN grid->distance, report held-out R². Source distance is the LABEL only;
at inference the model sees pixels. This is the deploy-without-source proof; step 2 (better value to
crack the local min) follows once L2 win-data exists.
"""
import time, json, hashlib
import numpy as np
import torch, torch.nn as nn
from arcengine import GameAction
from carnot.agentic import arc_solver_kit as kit
from carnot.agentic.arc_agi3_world_model import grid_of
from carnot.agentic.arc_agi3_live_adapter import _game_action,_levels_completed,_game_over
from carnot.agentic.arc_executable_world_model import to_logical,detect_cell

torch.manual_seed(0); np.random.seed(0)
arc=kit.offline_arcade(); env=arc.make("lp85",scorecard_id=arc.open_scorecard())
f0=env.reset(); cell=detect_cell(grid_of(f0))
LEFT={'x':4,'y':32}; RIGHT={'x':58,'y':32}
L2BTNS=[{'x':19,'y':16},{'x':38,'y':16},{'x':13,'y':25},{'x':47,'y':25},{'x':13,'y':34}]

def src_dist():
    L=env._game.current_level
    bp=[s for s in L._sprites if s.tags and 'bghvgbtwcb' in s.tags]
    op=[s for s in L._sprites if s.tags and 'fdgmtkfrxl' in s.tags]
    gs=[(s.x,s.y) for s in L._sprites if s.tags and 'goal' in s.tags]
    osr=[(s.x,s.y) for s in L._sprites if s.tags and 'goal-o' in s.tags]
    d=0
    for p in bp: d+=min(abs(p.x+1-gx)+abs(p.y+1-gy) for gx,gy in gs) if gs else 999
    for p in op: d+=min(abs(p.x+1-gx)+abs(p.y+1-gy) for gx,gy in osr) if osr else 999
    return d

# ---- collect (grid, dist, level) via random-button walks across L1+L2 ----
print("collecting (grid, distance) data...", flush=True)
data={}  # grid-hash -> (grid32, dist, level)
def record():
    g=to_logical(grid_of(env._game.frame if hasattr(env._game,'frame') else f0),cell) if False else None
def add_state(grid):
    h=hashlib.md5(grid.tobytes()).hexdigest()[:12]
    if h not in data:
        g32=grid[::2,::2]  # 32x32 downsample (camera is coarse)
        data[h]=(g32.astype(np.int8), src_dist(), _levels_completed_now())
def _levels_completed_now():
    return getattr(env._game,'level_index',0)
import random as _r; rng=_r.Random(0)
t0=time.time()
for episode in range(400):
    f=env.reset(); lvl=0; steps=0
    # randomly: sometimes start by solving L1 to reach L2
    seq_btns=[LEFT,RIGHT]
    while steps<40 and time.time()-t0<240:
        g=to_logical(grid_of(f),cell); add_state(g)
        b=rng.choice(seq_btns)
        f=env.step(_game_action(GameAction,6),data=b); steps+=1
        if f is None or _game_over(f): break
        nl=_levels_completed(f)
        if nl>lvl:  # advanced -> switch to L2 buttons
            lvl=nl; seq_btns=L2BTNS+[LEFT,RIGHT]
    if time.time()-t0>240: break
print(f"collected {len(data)} distinct states in {time.time()-t0:.0f}s; "
      f"dist range [{min(d for _,d,_ in data.values())},{max(d for _,d,_ in data.values())}]", flush=True)

# ---- build tensors: one-hot the 32x32 grid over colors ----
items=list(data.values())
colors=sorted(set(int(v) for g,_,_ in items for v in np.unique(g)))
cmap={c:i for i,c in enumerate(colors)}; C=len(colors)
def onehot(g):
    t=np.zeros((C,32,32),np.float32)
    for c,i in cmap.items(): t[i]=(g==c)
    return t
X=np.stack([onehot(g) for g,_,_ in items]); Y=np.array([d for _,d,_ in items],np.float32)
n=len(X); idx=np.random.permutation(n); tr=idx[:int(n*0.8)]; va=idx[int(n*0.8):]
Xt=torch.tensor(X); Yt=torch.tensor(Y)
print(f"dataset: {n} states, {C} colors, {len(tr)} train / {len(va)} val", flush=True)

# ---- small CPU CNN: grid -> distance ----
class V(nn.Module):
    def __init__(s):
        super().__init__()
        s.net=nn.Sequential(nn.Conv2d(C,16,3,padding=1),nn.ReLU(),nn.MaxPool2d(2),
                            nn.Conv2d(16,16,3,padding=1),nn.ReLU(),nn.AdaptiveAvgPool2d(4),
                            nn.Flatten(),nn.Linear(16*16,64),nn.ReLU(),nn.Linear(64,1))
    def forward(s,x): return s.net(x).squeeze(-1)
m=V(); opt=torch.optim.Adam(m.parameters(),1e-3); lossf=nn.MSELoss()
ymean=Yt[tr].mean()
for ep in range(60):
    m.train(); perm=tr[np.random.permutation(len(tr))]
    for i in range(0,len(perm),128):
        b=perm[i:i+128]; opt.zero_grad(); p=m(Xt[b]); l=lossf(p,Yt[b]); l.backward(); opt.step()
    if ep%15==0 or ep==59:
        m.eval()
        with torch.no_grad():
            pv=m(Xt[va]); mse=lossf(pv,Yt[va]).item()
            r2=1-((pv-Yt[va])**2).sum().item()/(((Yt[va]-ymean)**2).sum().item()+1e-9)
        print(f"  epoch {ep}: val MSE={mse:.2f} R2={r2:.3f}", flush=True)
torch.save(m.state_dict(),"/tmp/lp85_v_model.pt")
json.dump({"states":n,"colors":colors,"val_R2":round(r2,3),"val_MSE":round(mse,2)},
          open("/tmp/lp85_b_learn_result.json","w"),indent=1)
print(f"\n=== (b) step1: learned verifier from PIXELS ===")
print(f"  val R2={r2:.3f} (>0.8 = the source verifier is well-reproduced from pixels alone, no sprite/source access)")
print(f"  model saved /tmp/lp85_v_model.pt — this is a submission-deployable verifier (grid->distance)")
