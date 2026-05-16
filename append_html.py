with open("docs/technical-report.html", "r") as f:
    html = f.read()

new_html = """
<h3 id="milestone-202605206-positive-updates">Milestone 2026.05.206 Positive Updates</h3>
<p>In milestone .206, the operational retrospective completed, analyzing 0 minutes of wall time and 0 experiments. No experiment commits were found since activation, leaving GPUs correctly idle. No new bottlenecks were identified.</p>
</article>
"""

if "Milestone 2026.05.206 Positive Updates" not in html:
    html = html.replace('</article>', new_html)

with open("docs/technical-report.html", "w") as f:
    f.write(html)
