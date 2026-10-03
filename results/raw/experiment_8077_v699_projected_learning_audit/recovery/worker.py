import json,os,signal,sys
from pathlib import Path
from carnot.verify import projected_online_8076 as m
d=json.loads(Path(sys.argv[1]).read_text())
original=m.old.Journal.emit
project=m.kernel.project
if sys.argv[5]=='force':
 def forced(*args,**kwargs): return project(*args,**dict(kwargs,budget=0))
 m.kernel.project=forced
def emit(self,kind,row):
 hit=kind==sys.argv[3] and (kind!='fallback' or row['reason']=='projection')
 if hit and sys.argv[4]=='before': os.kill(os.getpid(),signal.SIGKILL)
 original(self,kind,row)
 if hit and sys.argv[4]=='after': os.kill(os.getpid(),signal.SIGKILL)
m.old.Journal.emit=emit
m.measure(d,Path(sys.argv[2]),budget_s=60)
