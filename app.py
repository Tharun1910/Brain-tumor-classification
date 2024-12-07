import os
import numpy as np
from flask import Flask, request, render_template
import cv2
from werkzeug.utils import secure_filename
import tensorflow as tf
from keras.models import load_model 
from PIL import Image
import io

# Initialize Flask app
app = Flask(__name__)

# Load the pre-trained model (DenseNet121)
model = tf.keras.models.load_model(r'C:\Projects\Computer Vision\densnet121.h5')

# Class labels for the model predictions
class_labels = {
    0: "GLIOMA",
    1: "MENINGIOMA",
    2: "NO TUMOR",
    3: "PITUITARY"
}

# Function to preprocess the image and extract features
def process_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes))
    img = cv2.cvtColor(np.array(img), cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_LANCZOS4)
    img = img.astype('float32')
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    return img

# Route for the homepage
@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template("index.html", prediction_text="No image uploaded.")

    image_file = request.files['image']
    if image_file:
        image_bytes = image_file.read()
        processed_img = process_image(image_bytes)

        # Make prediction using the model directly on the processed image
        prediction = model.predict(processed_img)
        
        # Get the predicted class
        predicted_class = np.argmax(prediction, axis=1)[0]
        output = class_labels[predicted_class]

        return render_template("index.html", prediction_text=f"Prediction: {output}")

    return render_template("index.html", prediction_text="Error in processing the image.")


if __name__ == "__main__":
    app.run(debug=True)
