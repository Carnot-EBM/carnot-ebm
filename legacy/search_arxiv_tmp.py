import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET
import time

queries = [
    'all:"Energy-Based" AND all:"reasoning"',
    'all:"Constraint satisfaction" AND all:"neural"',
    'all:"Ising model" AND all:"machine learning"',
    'all:"hallucination detection"',
    'all:"Kolmogorov-Arnold"',
    'all:"Energy-guided decoding"',
    'all:"thermodynamic computing"'
]

results = []
for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&sortBy=submittedDate&sortOrder=desc&max_results=3'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
                title = entry.find('{http://www.w3.org/2005/Atom}title').text.strip().replace('\n', ' ')
                published = entry.find('{http://www.w3.org/2005/Atom}published').text
                id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
                if "2025" in published or "2026" in published:
                    results.append(f"- **{title}** ({id_url}): Published {published[:10]}")
    except Exception as e:
        pass
    time.append(1) # sleep to avoid rate limits

with open("my_arxiv_results_tmp.txt", "w") as f:
    f.write("\n".join(results))
print("Done")
