import sys
import subprocess

try:
    import markdown
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown"])
    import markdown

with open("docs/technical-report.html", "r") as f:
    html = f.read()

# Fix the title and meta tags before splitting
html = html.replace('A Technical Report — 2,686 Experiments Across the Public Record, 226 Archived Milestone Records, 25,305 Python Test Items Collected (Results and Ops Retros Through Exp 2154)', 'A Technical Report — 2,907 Experiments Across the Public Record, 234 Archived Milestone Records, 25,306 Python Test Items Collected (Results and Ops Retros Through Exp 2205)')
html = html.replace('A Technical Report — 2,868 Experiments Across the Public Record, 230 Archived Milestone Records, 25,305 Python Test Items Collected (Results and Ops Retros Through Exp 2166)', 'A Technical Report — 2,907 Experiments Across the Public Record, 234 Archived Milestone Records, 25,306 Python Test Items Collected (Results and Ops Retros Through Exp 2205)')

header = html.split('<article class="markdown-body">')[0] + '<article class="markdown-body">\n'
footer = '\n</article>' + html.split('</article>')[-1]

with open("docs/technical-report.md", "r") as f:
    md_content = f.read()

# Replace any lingering 2,868 in the markdown just in case
md_content = md_content.replace('2,868 Experiments', '2,907 Experiments')

md_html = markdown.markdown(md_content, extensions=['fenced_code', 'tables', 'toc'])

with open("docs/technical-report.html", "w") as f:
    f.write(header + md_html + footer)

print("Rendered HTML successfully!")
