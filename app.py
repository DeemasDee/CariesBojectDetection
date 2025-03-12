from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import io
import base64
from PIL import Image

app = Flask(__name__)

# Load trained model
MODEL_PATH = "vgg16_unet_model.h5"  # Sesuaikan dengan path model Anda
model = tf.keras.models.load_model(MODEL_PATH)

# Function to preprocess image
def preprocess_image(image):
    image = image.resize((256, 256))  # Sesuaikan ukuran input model
    image = np.array(image) / 255.0  # Normalisasi
    image = np.expand_dims(image, axis=0)
    return image

# Function to process mask and determine caries level
def process_mask(mask):
    mask = mask.squeeze()  # Hilangkan dimensi ekstra
    mask = (mask > 0.5).astype(np.uint8) * 255  # Binarisasi mask

    # Hitung persentase pixel putih dalam mask segmentation
    white_pixel_ratio = np.sum(mask == 255) / mask.size  

    # Menentukan kategori caries berdasarkan ambang batas
    if white_pixel_ratio <= 0.05:
        diagnosis = "Tidak Caries"
    elif 0.05 < white_pixel_ratio <= 0.10:
        diagnosis = "Caries Ringan"
    elif 0.10 < white_pixel_ratio <= 0.15:
        diagnosis = "Caries Sedang"
    else:
        diagnosis = "Caries Parah"

    return Image.fromarray(mask, mode="L"), white_pixel_ratio, diagnosis

# Function to encode image to base64
def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    image = Image.open(file.stream).convert("RGB")
    preprocessed_image = preprocess_image(image)
    
    # Perform prediction
    prediction = model.predict(preprocessed_image)
    
    # Process mask and determine caries level
    mask, white_pixel_ratio, diagnosis = process_mask(prediction)

    # Convert images to base64
    input_image_base64 = encode_image(image)
    mask_image_base64 = encode_image(mask)

    return jsonify({
        "input_image": input_image_base64,
        "mask_image": mask_image_base64,
        "diagnosis": diagnosis,
        "white_pixel_percentage": round(white_pixel_ratio * 100, 2)  # Persentase pixel putih dalam %
    })

if __name__ == "__main__":
    app.run(debug=True)
