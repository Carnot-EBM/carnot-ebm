import urllib.request
import xml.etree.ElementTree as ET

ids = ["2601.00003", "2601.05300", "2601.07767", "2601.17223", "2602.04248", "2602.14189", "2604.16753"]
url = "http://export.arxiv.org/api/query?id_list=" + ",".join(ids)
response = urllib.request.urlopen(url).read()
root = ET.fromstring(response)
ns = {'atom': 'http://www.w3.org/2005/Atom'}

for entry in root.findall('atom:entry', ns):
    id_val = entry.find('atom:id', ns).text.split('/')[-1]
    title = entry.find('atom:title', ns).text.replace('\n', ' ').strip()
    summary = entry.find('atom:summary', ns).text.replace('\n', ' ').strip()
    print(f"ID: {id_val}\nTITLE: {title}\nSUMMARY: {summary}\n")
