import os
import numpy as np
import pandas as pd
import cv2
from flask import Flask, request, render_template, jsonify
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics.pairwise import cosine_similarity

# Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config["UPLOAD_FOLDER"] = "./uploads"
os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)

# Load dataset and precomputed features
dataset_path = "datasets/cleaned_artifacts_with_descriptions.csv"
features_path = "datasets/precomputed_features.npy"

# Check if dataset and features exist
if not os.path.exists(dataset_path) or not os.path.exists(features_path):
    raise FileNotFoundError("Dataset or precomputed features not found.")

df = pd.read_csv(dataset_path)
df.columns = df.columns.str.strip()  # Clean column names
precomputed_features = np.load(features_path)

# Load TensorFlow Lite model
lite_model_path = os.path.join(os.getcwd(), "models", "efficientnet_lite.tflite")
if not os.path.exists(lite_model_path):
    raise FileNotFoundError("TensorFlow Lite model file not found.")

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
        
        # Get the expected input shape from the model
        input_shape = input_details[0]['shape']
        target_size = (input_shape[1], input_shape[2])  # Get width and height from model
        
        # Resize and preprocess image
        img = cv2.resize(img, target_size)
        img_array = img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0).astype(np.float32)
        
        # Normalize the image
        img_array = preprocess_input(img_array)
        
        return img_array
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

# Function to extract features using TensorFlow Lite model
def extract_features(image_path):
    preprocessed_img = preprocess_image(image_path)
    if preprocessed_img is None:
        return None
    
    # Check if dimensions match the model's input shape
    input_shape = input_details[0]['shape']
    if preprocessed_img.shape != tuple(input_shape):
        print(f"Error: Input shape mismatch. Expected {tuple(input_shape)}, got {preprocessed_img.shape}.")
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
        return {"status": "Error", "message": "Unable to extract features from the image."}
    
    similarities = cosine_similarity([image_features], precomputed_features)[0]
    highest_similarity = float(max(similarities))  # Convert to Python float for JSON compatibility
    closest_match_index = np.argmax(similarities)
    closest_match = df.iloc[closest_match_index]

    if highest_similarity >= similarity_threshold:
        return {
            "status": "Authentic",
            "similarity": highest_similarity,
            "artifact_details": closest_match.to_dict()
        }
    else:
        return {
            "status": "Counterfeit",
            "similarity": highest_similarity,
            "artifact_details": closest_match.to_dict()
        }

# Routes
@app.route("/")
def index():
    return render_template("index.html")  # HTML file for uploading images

@app.route("/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in the request."}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected for upload."}), 400

    file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(file_path)

    # Check authenticity
    result = check_authenticity(file_path)
    return jsonify(result)  # Return JSON response

# Run the app
if __name__ == "__main__":
    app.run(debug=True)
