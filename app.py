from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Flask App Setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load your trained model
model = load_model('model/fetal_ultrasound_model.h5')

# Define the class index mapping
index_to_class = {
    0: "Fetal brain",
    1: "Fetal thorax",
    2: "Fetal abdomen",
    3: "Fetal femur"
}

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(300, 300))  # Resize image
    img_array = img_to_array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    return img_array

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', error="No file uploaded")

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', error="No file selected")

    # Ensure the upload folder exists
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])

    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)

    try:
        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence = np.max(prediction) * 100

        predicted_label = index_to_class[predicted_class]

        return render_template(
            'index.html',
            prediction=predicted_label,
            confidence=f"{confidence:.2f}%",
            image_path=file_path
        )
    except Exception as e:
        return render_template('index.html', error=f"Prediction Error: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
