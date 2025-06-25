from flask import Flask, render_template, request
import os
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import json
from werkzeug.utils import secure_filename

# Initialize Flask
app = Flask(__name__, template_folder='app/templates', static_folder='app/static')

# Load model and class names
model = load_model("model/pattern_classifier_transfer.h5")
with open("model/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_names = list(class_indices.keys())

# Setup upload folder
UPLOAD_FOLDER = os.path.join('app', 'static', 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Route: Step 1 - Welcome Page
@app.route('/')
def welcome():
    return render_template('welcome.html')

# Route: Step 2 - Home Page
@app.route('/home')
def home():
    return render_template('home.html')

# Route: Step 3 - Upload Page
@app.route('/upload')
def upload():
    return render_template('index.html')

# Route: Step 4 - Prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part', 400
    file = request.files['file']
    if file.filename == '':
        return 'No selected file', 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Image preprocessing
    img = image.load_img(file_path, target_size=(128, 128))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    # Model prediction
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]

    image_url = f'/static/uploads/{filename}'
    return render_template('result.html', image_path=image_url, prediction=predicted_class)

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
