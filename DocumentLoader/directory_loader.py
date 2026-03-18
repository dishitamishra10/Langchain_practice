from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path='Books',
    glob='*.pdf',
    loader_cls=PyPDFLoader
)

docs = loader.load()

print(docs[0].page_content)
print(docs[0].metadata)

# for doc in loader.lazy_load():
#     print(doc.page_content)
#     print(doc.metadata)
#     break   # remove break if you want all