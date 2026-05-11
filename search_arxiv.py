import urllib.request
import xml.etree.ElementTree as ET

queries = [
    "all:\"energy based model\"+AND+all:reasoning",
    "all:\"Kolmogorov-Arnold\"+AND+all:verification",
    "all:\"constraint satisfaction\"+AND+all:\"neural network\"",
    "all:\"continual learning\"+AND+all:\"constraint\""
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

with open("arxiv_results.txt", "w") as f:
    f.write("\n".join(results))
