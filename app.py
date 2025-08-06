import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("face_mask_detector_model (1).h5")

img_size = (128, 128)
# Define labels
labels = ['No Mask', 'Mask']

# Preprocessing function (adjust based on your training)
def predict_mask(img: Image.Image):
    img = img.resize(img_size)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)



    if prediction[0][0] < 0.5:
        return "Mask Detected"
    else:
        return "No Mask Detected"


# Create Gradio interface
gr.Interface(
    fn=predict_mask,
    inputs=gr.Image(type="pil"),
    outputs="text",
    title="Face Mask Detection",
    description="ارفع صورة، والموديل هيقول إذا كان فيها كمامة أو لا."
).launch()


