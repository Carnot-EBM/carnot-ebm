import glob
import json

files = glob.glob('/home/ianblenke/github.com/ianblenke/carnot/results/*eval*') + glob.glob('/home/ianblenke/github.com/ianblenke/carnot/data/*corpus*')
for f in files:
    try:
        content = open(f).read()
        if 'response' in content and 'label' in content:
            print("Potentially has response/label:", f)
    except:
        pass
