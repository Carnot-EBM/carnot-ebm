import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

queries = [
    'all:"energy based model"+AND+all:reasoning',
    'all:"Kolmogorov-Arnold"+AND+all:verification',
    'all:"constraint satisfaction"+AND+all:"neural network"',
    'all:"continual learning"+AND+all:"constraint"'
]

results = []
for q in queries:
    try:
        url = "http://export.arxiv.org/api/query?search_query=" + urllib.parse.quote(q, safe='+:') + "&start=0&max_results=3&sortBy=submittedDate&sortOrder=desc"
        response = urllib.request.urlopen(url)
        data = response.read()
        root = ET.fromstring(data)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.replace("\n", " ").strip()
            published = entry.find("{http://www.w3.org/2005/Atom}published").text
            summary = entry.find("{http://www.w3.org/2005/Atom}summary").text.replace("\n", " ").strip()
            results.append(f"Title: {title}\nDate: {published}\nSummary: {summary[:300]}...\n")
    except Exception as e:
        results.append(f"Error for {q}: {e}")

with open("arxiv_results.txt", "w") as f:
    f.write("\n".join(results))
