from flask import Flask, request, jsonify
from flask_cors import CORS  # To handle CORS
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Create Flask app
app = Flask(__name__)

# Enable CORS
CORS(app)

# Load the trained model
MODEL_PATH = "waste_classification_model.h5"
model = load_model(MODEL_PATH)

# Class indices and disposal recommendations
CLASS_INDICES = {
    0: "E-waste",
    1: "Green-waste",
    2: "Renewable-waste",
    3: "Solid-waste",
    4: "Textile-waste",
    5: "Miscellaneous-waste"
}

DISPOSAL_RECOMMENDATIONS = {
    "E-waste": "Recycle through an authorized e-waste recycler.",
    "Green-waste": "Compost or use as organic fertilizer.",
    "Renewable-waste": "Check if it can be repurposed or sent to specialized facilities.",
    "Solid-waste": "Dispose in designated solid waste bins or landfill sites.",
    "Textile-waste": "Donate, recycle, or upcycle into new products.",
    "Miscellaneous-waste": "Follow local disposal guidelines for mixed waste."
}

# Function to predict the class of an image
def predict_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize image
    img_array = image.img_to_array(img)  # Convert image to array
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize
    predictions = model.predict(img_array)  # Get prediction
    pred_class = CLASS_INDICES[np.argmax(predictions)]  # Get the class with highest probability
    confidence = np.max(predictions) * 100  # Get confidence percentage
    recommendation = DISPOSAL_RECOMMENDATIONS[pred_class]  # Get disposal recommendation
    return pred_class, confidence, recommendation

# Endpoint to handle image prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    # Save the uploaded file temporarily
    save_path = os.path.join("uploads", file.filename)
    os.makedirs("uploads", exist_ok=True)
    file.save(save_path)

    # Perform prediction
    try:
        pred_class, confidence, recommendation = predict_image(save_path)

        # Clean up (optional: remove the uploaded file after processing)
        os.remove(save_path)

        return jsonify({
            "category": pred_class,
            "confidence": confidence,
            "recommendation": recommendation
        })
    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(debug=True)
