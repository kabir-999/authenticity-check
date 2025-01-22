import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, render_template
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Flask app
app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load dataset and precomputed features
dataset_path = "datasets/cleaned_artifacts_with_descriptions.csv"
features_path = "datasets/precomputed_features.npy"
df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()
precomputed_features = np.load(features_path)

# Load TensorFlow Lite model
lite_model_path = "models/efficientnet_lite.tflite"
interpreter = tf.lite.Interpreter(model_path=lite_model_path)
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Function to preprocess images
def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not found or invalid file format.")
        img = cv2.resize(img, (224, 224))
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        return preprocess_input(img_array).astype(np.float32)
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to extract features using TensorFlow Lite model
def extract_features(image_path):
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None:
        return None

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()
    features = interpreter.get_tensor(output_details[0]['index'])
    return features.flatten()

# Authenticity checking function
def check_authenticity(image_path, similarity_threshold=0.7):
    image_features = extract_features(image_path)
    if image_features is None:
        return "<h1>Error: Unable to extract features from the image.</h1>"

    similarities = cosine_similarity([image_features], precomputed_features)[0]
    highest_similarity = max(similarities)
    closest_match_index = np.argmax(similarities)
    closest_match = df.iloc[closest_match_index]

    if highest_similarity >= similarity_threshold:
        return f"""
        <h1>Authentic Artifact</h1>
        <p><strong>Closest Match:</strong> {closest_match['Artifact Name']}</p>
        <p><strong>Date:</strong> {closest_match['Date']}</p>
        <p><strong>Culture/Region:</strong> {closest_match['Culture/Region']}</p>
        <p><strong>Material:</strong> {closest_match['Material']}</p>
        <p><strong>Dimensions:</strong> {closest_match['Dimensions']}</p>
        <p><strong>Category/Type:</strong> {closest_match['Category/Type']}</p>
        <p><strong>Description:</strong> {closest_match['Description']}</p>
        <p><strong>Similarity:</strong> {highest_similarity:.2f}</p>
        <img src="{closest_match['Artifact Image URL']}" alt="Artifact Image" style="max-width: 100%; height: auto;">
        """
    else:
        return f"<h1>Counterfeit Artifact</h1><p>Similarity: {highest_similarity:.2f}</p>"

# Routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return "<h1>Error: No file part in the request.</h1>", 400

    file = request.files["file"]
    if file.filename == "":
        return "<h1>Error: No file selected for upload.</h1>", 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Check authenticity
    result = check_authenticity(file_path)
    return result

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
