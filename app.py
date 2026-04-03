from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import base64

import os

app = Flask(__name__, static_folder='frontend', static_url_path='')
CORS(app)

# Load the trained model
model = tf.keras.models.load_model('mnist_cnn_model.h5')

@app.route('/')
def index():
    return app.send_static_file('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
    
    # Decode base64 image
    img_data = base64.b64decode(data['image'].split(',')[1])
    img = Image.open(io.BytesIO(img_data)).convert('L')
    
    # Resize to 28x28
    img = img.resize((28, 28))
    
    # Convert to array, INVERT colors (MNIST is white on black), and normalize
    img_array = np.array(img)
    img_array = (255 - img_array) / 255.0
    img_array = img_array.reshape((1, 28, 28, 1))
    
    # Make prediction
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))
    confidence = float(np.max(prediction))
    
    return jsonify({
        'digit': predicted_digit,
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
