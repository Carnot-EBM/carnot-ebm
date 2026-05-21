import re

with open("docs/index.html", "r") as f:
    content = f.read()

content = content.replace("138</div><div class=\"stat-label\">archived records through .125", "140</div><div class=\"stat-label\">archived records through .126")
content = content.replace("125/125</div><div class=\"stat-label\">experiments completed in .125", "151/151</div><div class=\"stat-label\">experiments completed in .126")
content = content.replace("utilization during .125 runs", "utilization during .126 runs")

with open("docs/index.html", "w") as f:
    f.write(content)
