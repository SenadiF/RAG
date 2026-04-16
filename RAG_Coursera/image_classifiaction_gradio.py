import torch 
import requests 
from torchvision import transforms
import gradio as gr

model=torch.hub.load("pytorch/vision:v0.10.0","resnet18",pretrained=True)

#Download human readable labels for the ImageNet dataset
response = requests.get("https://git.io/JJkYN")
labels = [l.strip() for l in response.text.split("\n") if l.strip()]

#Define image transformations to preprocess the input image for the model
transform=transforms.Compose([
    transforms.ToTensor(),
     transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# Takes the input image as a PIL image and converts to a pytorch tensor 
def predict(inp):
    # preprocess image
    inp = transform(inp).unsqueeze(0)
    # ensure model runs in inference mode
    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(inp)[0], dim=0)
    # map predictions to labels
    confidences = {
        labels[i]: float(prediction[i]) 
        for i in range(len(labels))
    }
    return confidences

interface_one=gr.Interface(fn=predict,
       inputs=gr.Image(type="pil"),
       outputs=gr.Label(num_top_classes=3),
       examples=["/content/lion.jpg", "/content/cheetah.jpg"])

interface_one.launch(server_name="127.0.0.1" ,server_port= 7860)


