import requests
import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_huggingface import HuggingFaceEmbeddings
# Model settings
model_id = "llama3"
max_tokens = 256
temperature = 0.5


# LLM 
def generate_response(prompt_txt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model_id,
            "prompt": prompt_txt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens
            }
        }
    )
    return response.json()["response"]


# Document loader
def document_loader(file):
    loader = PyPDFLoader(file)
    return loader.load()


# Text splitter
def text_splitter(data):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, #each chunk will have 1000 characters
        chunk_overlap=50 # Ovelap so the context is not lost between chunks
    )
    return splitter.split_documents(data)


# Vector DB 
def vector_database(chunks):
    embedding_model = HuggingFaceEmbeddings()
    vectordb = Chroma.from_documents(chunks, embedding_model)
    return vectordb


# Retriever
def retriever(file):
    docs = document_loader(file)
    chunks = text_splitter(docs)
    vectordb = vector_database(chunks)
    return vectordb.as_retriever()#Creates the search system


# QA Chain 
def retriever_qa(file, query):
    retriever_obj = retriever(file)

    # simple prompt injection into LLM
    context_docs = retriever_obj.get_relevant_documents(query)
    context = "\n".join([doc.page_content for doc in context_docs])

    final_prompt = f"""
    Answer the question using the context below.

    Context:
    {context}

    Question:
    {query}
    """

    return generate_response(final_prompt)


# Gradio UI
rag_application = gr.Interface(
    fn=retriever_qa,
    inputs=[
        gr.File(label="Upload PDF", file_types=['.pdf'], type="filepath"),
        gr.Textbox(label="Ask question")
    ],
    outputs=gr.Textbox(label="Answer"),
    title="Local RAG Chatbot",
    description="Ask questions from your PDF using local Llama3"
)

rag_application.launch()