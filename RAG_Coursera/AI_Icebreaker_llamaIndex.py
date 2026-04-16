import gradio as gr

from llama_index.core import VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama



#  MOCK PROFILE DATA

def load_profile():
    return """
    Name: Sarah Johnson
    Role: Software Engineer at Google
    Skills: Python, AI, Machine Learning, Robotics
    Interests: Space exploration, startups, music
    Experience: 4 years in backend systems and AI projects
    """



#  Index automatcally does chunkinh ,embeddings ,indexing and storage 

def build_index():
    text = load_profile()
    doc = Document(text=text)

    embed_model = HuggingFaceEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    index = VectorStoreIndex.from_documents(
        [doc],
        embed_model=embed_model
    )

    return index



# LLM (Ollama LOCAL)

llm = Ollama(
    model="llama3",
    request_timeout=120.0
)

index = build_index()
query_engine = index.as_query_engine(llm=llm)


#main function that will be called by gradio
def icebreaker_bot(user_input):

    prompt = f"""
    Based on the user's profile, generate 3 icebreaker questions
    for networking.

    Focus on: {user_input}
    """

    response = query_engine.query(prompt)
    return str(response)


app = gr.Interface(
    fn=icebreaker_bot,
    inputs=gr.Textbox(label="Enter topic (AI, robotics, career...)"),
    outputs="text",
    title="Icebreaker Bot (LlamaIndex + Ollama)",
)

app.launch(server_name="127.0.0.1", server_port=7860)