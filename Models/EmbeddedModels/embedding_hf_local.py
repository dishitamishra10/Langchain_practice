from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

text = "What is the capital of India ?"

documnets = [
    "I love machine learning.",
    "Artificial intelligence is amazing.",
    "Pizza is my favorite food."
]

vector1 = embedding.embed_query(text)
vector2 = embedding.embed_documents(documnets)

print(str(vector1))
print(str(vector2))

