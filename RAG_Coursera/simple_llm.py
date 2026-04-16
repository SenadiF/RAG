import requests

response = requests.post(
    "http://localhost:11434/api/generate",
    json={
        "model": "llama3",
        "prompt": "Hello",
        "stream": False   # 🔥 THIS FIXES IT
    }
)

print(response.json()["response"])