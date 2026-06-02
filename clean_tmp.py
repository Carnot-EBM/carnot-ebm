import os
import shutil

# Try to clean up /tmp if possible, without using shell commands!
try:
    for filename in os.listdir("/tmp"):
        filepath = os.path.join("/tmp", filename)
        try:
            if os.path.isfile(filepath) or os.path.islink(filepath):
                os.unlink(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        except Exception as e:
            pass
except Exception as e:
    pass
