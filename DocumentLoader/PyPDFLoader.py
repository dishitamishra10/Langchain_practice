# pypdf loader used for the textual data
from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader('virat_kohli.pdf')

docs = loader.load()

print(len(docs)) # no. of pages ... each page is considered as a docs

print(docs[0].page_content) # content of the 1st page
print(docs[0].metadata)