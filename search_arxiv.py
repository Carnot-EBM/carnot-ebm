import urllib.request
import xml.etree.ElementTree as ET
import json

queries = [
    'all:"Energy-Based Model" AND all:reasoning',
    'all:"Energy-Guided Decoding"',
    'all:"Kolmogorov-Arnold Network"',
    'all:"Energy-Based Fine-Tuning"',
    'all:"Constraint satisfaction" AND all:"neural network"',
]

results = []
for q in queries:
    url = f'http://export.arxiv.org/api/query?search_query={urllib.parse.quote(q)}&start=0&max_results=3&sortBy=submittedDate&sortOrder=desc'
    try:
        response = urllib.request.urlopen(url)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            id_url = entry.find('{http://www.w3.org/2005/Atom}id').text
            results.append({'title': title, 'published': published, 'id': id_url})
    except Exception as e:
        print(f"Error querying {q}: {e}")

print(json.dumps(results, indent=2))
