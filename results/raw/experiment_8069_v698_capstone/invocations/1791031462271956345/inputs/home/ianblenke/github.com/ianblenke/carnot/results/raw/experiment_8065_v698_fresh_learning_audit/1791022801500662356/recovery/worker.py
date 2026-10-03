from pathlib import Path
import json,os,signal,sys
from carnot.verify import fresh_feedback_8064 as m
d=json.loads(Path(sys.argv[1]).read_text())
original=m.Journal.emit
def emit(self,kind,row):
 if kind==sys.argv[3] and sys.argv[4]=='before': os.kill(os.getpid(),signal.SIGKILL)
 original(self,kind,row)
 if kind==sys.argv[3] and sys.argv[4]=='after': os.kill(os.getpid(),signal.SIGKILL)
m.Journal.emit=emit
m.measure(d,Path(sys.argv[2]))
