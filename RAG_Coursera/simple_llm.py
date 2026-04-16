import requests
import gradio as gr

# Model settings
model_id = "llama3" 
max_tokens = 256
temperature = 0.5

# Function to call Ollama
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


# Gradio UI
chat_application = gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Input", lines=2, placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Output"),
    title="Local Llama3 Chatbot",
    description="Ask any question (running locally with Ollama)"
)

# Launch
chat_application.launch(server_name="127.0.0.1", server_port=7860)