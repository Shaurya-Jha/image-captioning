from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import gradio as gr

# create a processor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")

# create model
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# function to take image and process the image using the model
def generate_caption(img):
    # convert image to array
    img_input = Image.fromarray(img)
    print(img_input)
    
    inputs = processor(img_input, return_tensors="pt")
    # take inputs and run the model
    out = model.generate(**inputs)
    print(f"\n{out}")
    
    # decode
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

# gradio interface
demo = gr.Interface(
    fn = generate_caption, 
    inputs=[gr.Image(label="Image")], 
    outputs=[gr.Text(label="Caption")]
)

demo.launch(share=True)