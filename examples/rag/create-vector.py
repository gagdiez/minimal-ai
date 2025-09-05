from glob import glob

import chromadb

client = chromadb.PersistentClient(path="./chroma/")
collection = client.create_collection(name='cities')

md_files = glob("./cities/*.md")
content = []
for f in md_files:
    with open(f, 'r') as file:
        content.append(file.read())

collection.add(
    documents=content,
    ids=md_files,
)
