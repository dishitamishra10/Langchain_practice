import os
from dotenv import load_dotenv

from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# Load Environment Variables

load_dotenv()  


# Step 1: Get YouTube Transcript

video_id = "Gfr50f6ZBvo"

try:
    api = YouTubeTranscriptApi()
    transcript_list = api.fetch(video_id, languages=["en"])
    transcript = " ".join(chunk.text for chunk in transcript_list)
except TranscriptsDisabled:
    print("No captions available for this video.")
    exit()


# Step 2: Split Transcript
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = splitter.create_documents([transcript])



# Step 3: Create FAISS Vector Store

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    documents=documents,
    embedding=embedding
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)


# Step 4: Define Conversational LLM
llm = HuggingFaceEndpoint(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    task="conversational",
    max_new_tokens=150,
    temperature=0.3
)

chat_model = ChatHuggingFace(llm=llm)


# Step 5: Prompt Template

prompt = PromptTemplate(
    template="""
You are a helpful assistant.

Answer ONLY from the provided transcript context.
If the context is insufficient, say you don't know.

If the user asks for a summary:
Provide a concise summary in 4-6 sentences only.

Context:
{context}

Question: {question}
""",
    input_variables=["context", "question"]
)

# Step 6: Ask Question

question = "Is the topic of aliens discussed in this video? If yes, what was discussed?"

retrieved_docs = retriever.invoke(question)

context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)

final_prompt = prompt.format(
    context=context_text,
    question=question
)



# Step 7: Generate Answer (Chat Format)

response = chat_model.invoke(
    [HumanMessage(content=final_prompt)]
)

print("\nAnswer:\n")
print(response.content)

def format_docs(retrieved_docs):
  context_text = "\n\n".join(doc.page_content for doc in retrieved_docs)
  return context_text

parallel_chain = RunnableParallel({
    'context': retriever | RunnableLambda(format_docs),
    'question': RunnablePassthrough()
})

parallel_chain.invoke('who is Demis')

parser = StrOutputParser()

main_chain = parallel_chain | prompt | chat_model | parser

main_chain.invoke('Summarize the video in 5 sentences only.')
