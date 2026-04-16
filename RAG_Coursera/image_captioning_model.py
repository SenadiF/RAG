import gradio as gr 
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
#take the image and convert it to numbers(tensors ) so the model understands 
processor=BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
#Loads the model that takes the processed image and generates the caption 
model=BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

def generate_caption(image):

    inputs=processor(image,return_tensors="pt")
    out=model.generate(**inputs)
    # Converts the tokens into a human readable sentence 
    caption=processor.decode(out[0],skip_special_tokens=True)
    return caption

def caption_image(image):

    try:
        caption=generate_caption(image)
        return caption
    except Exception as e:
        return f"Error: {str(e)}"   
image_interface=gr.Interface(
    fn=caption_image,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Image Captioning with BLIP",
    description="Upload an image to generate a caption using the BLIP model."
)

image_interface.launch(server_name="127.0.0.1",server_port=7860)