import gradio as gr
import requests
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

#LLM (Ollama)
def generate_response(prompt):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


#mock profile data 
def load_profile():
    return """
    Name: Sarah Johnson
    Role: Software Engineer at Google
    Skills: Python, AI, Machine Learning, Robotics
    Interests: Space exploration, startups, music
    Experience: 4 years in backend systems and AI projects
    Education: Computer Science degree
    """


#split profile into chunks
def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=30
    )
    return splitter.split_text(text)


#get embedding model
def get_embedding_model():
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


#build vector database
def build_db(chunks):
    embeddings = get_embedding_model()
    db = Chroma.from_texts(chunks, embeddings)
    return db


def icebreaker_bot(user_query):

    #  load profile
    profile = load_profile()

    #  split into chunks
    chunks = split_text(profile)

    #  build vector DB
    db = build_db(chunks)

    #  retrieve relevant chunks
    docs = db.similarity_search(user_query, k=3)

    context = "\n".join([d.page_content for d in docs])

    # prompt 
    prompt = f"""
    You are an AI networking assistant.

    Based on this LinkedIn profile:
    {context}

    Task:
    Generate 3 smart and friendly icebreaker questions
    for a networking conversation.

    Make them natural and not robotic.
    """

    # call LLM (Ollama)
    return generate_response(prompt)



app = gr.Interface(
    fn=icebreaker_bot,
    inputs=gr.Textbox(label="Enter a topic (e.g. AI, career, robotics)"),
    outputs="text",
    title="AI Icebreaker Bot (Local RAG + Ollama)",
    description="Generate personalized networking icebreakers from LinkedIn-style profiles"
)

app.launch(server_name="127.0.0.1", server_port=7860)