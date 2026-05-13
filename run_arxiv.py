import urllib.request
import xml.etree.ElementTree as ET

queries = [
    "all:\"Energy-Guided Decoding\"+AND+all:hallucination",
    "all:\"Continual Learning\"+AND+all:\"Energy-Based\"",
    "all:\"Kolmogorov-Arnold Network\"+AND+all:\"Hardware\"",
    "all:\"Constraint Satisfaction\"+AND+all:\"Neural Solver\"",
    "all:\"Energy-Based Models\"+AND+all:reasoning"
]

results = []
for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={q}&start=0&max_results=3&sortBy=submittedDate&sortOrder=desc'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
            results.append(f"Title: {title}\nDate: {published}\nSummary: {summary[:200]}...\n")
    except Exception as e:
        results.append(f"Error for {q}: {e}")

with open("arxiv_results2.txt", "w") as f:
    f.write("\n".join(results))
