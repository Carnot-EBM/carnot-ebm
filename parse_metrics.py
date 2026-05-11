import re
with open("README.md") as f:
    print("README old:")
    for line in f:
        if "experiment records tracked through Exp" in line:
            print(line.strip())
        elif "Python Test Items" in line:
            print(line.strip())
        elif "archived records" in line:
            print(line.strip())
