from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

splitter = CharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=0, # 10-20% of chunk_size is usually set
    separator=''
)

loader = PyPDFLoader('MachineLearning.pdf')
docs = loader.load()

result = splitter.split_documents(docs)

print(len(result))
print(result[10].page_content)
print(result[10].metadata)