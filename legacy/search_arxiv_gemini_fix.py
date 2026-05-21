import urllib.request
import urllib.parse
import xml.etree.ElementTree as ET

queries = [
    'all:"energy-based model" AND (all:"verification" OR all:"reasoning")',
    'all:"Kolmogorov-Arnold Network"',
    'all:"constraint satisfaction" AND all:"neural network"',
    'all:"energy-guided decoding"'
]

print("Arxiv Search Results:")
for q in queries:
    # ArXiv API requires %22 for quotes and %20 for spaces
    encoded_q = urllib.parse.quote(q)
    url = f'http://export.arxiv.org/api/query?search_query={encoded_q}&sortBy=submittedDate&sortOrder=desc&max_results=4'
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        xml_data = response.read()
        root = ET.fromstring(xml_data)
        for entry in root.findall('{http://www.w3.org/2005/Atom}entry'):
            title = entry.find('{http://www.w3.org/2005/Atom}title').text.replace('\n', ' ')
            published = entry.find('{http://www.w3.org/2005/Atom}published').text
            summary = entry.find('{http://www.w3.org/2005/Atom}summary').text.replace('\n', ' ')[:300]
            print(f"- {published[:10]} | {title}\n  {summary}...")
    except Exception as e:
        print(f"Error querying {q}: {e}")
