import urllib.request
import xml.etree.ElementTree as ET
import urllib.parse

def search(query, max_results=5):
    query_url = urllib.parse.quote(query)
    url = f'http://export.arxiv.org/api/query?search_query=all:{query_url}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=desc'
    try:
        data = urllib.request.urlopen(url).read()
        root = ET.fromstring(data)
        namespace = {'atom': 'http://www.w3.org/2005/Atom'}
        entries = root.findall('atom:entry', namespace)
        if not entries:
            print("No results found.")
        for entry in entries:
            title = entry.find('atom:title', namespace).text.strip().replace('\n', ' ')
            published = entry.find('atom:published', namespace).text.strip()
            print(f"Title: {title}\nPublished: {published}\n" + "-"*40)
    except Exception as e:
        print(f"Error searching {query}: {e}")

print("SEARCHING: Energy-Based Models verification")
search("Energy-Based AND Models AND verification", 3)
print("SEARCHING: Constraint satisfaction neural networks")
search("constraint AND satisfaction AND neural AND networks", 3)
print("SEARCHING: Ising model machine learning")
search("Ising AND model AND machine AND learning", 3)
print("SEARCHING: Kolmogorov-Arnold Networks")
search("Kolmogorov-Arnold AND Networks", 3)
print("SEARCHING: Energy-guided decoding")
search("Energy-guided AND decoding", 3)
