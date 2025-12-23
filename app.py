import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import tensorflow as tf
from flask import Flask, request, render_template, redirect, url_for, flash
from werkzeug.utils import secure_filename
from PIL import Image

# Initialize the Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads/'
app.config['SECRET_KEY'] = 'supersecretkey'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)


# Load the model
model_path = 'Team3model.h5'
model = tf.keras.models.load_model(model_path, compile=False)

# Manually configure the model's loss function
model.compile(optimizer='adam', loss=tf.keras.losses.CategoricalCrossentropy(reduction='sum_over_batch_size'))

img_width, img_height = 256, 256

# Class labels
class_labels = ['Bell Pepper-bacterial spot', 'Bell Pepper-healthy', 'Cassava-Bacterial Blight (CBB)',
                'Cassava-Brown Streak Disease (CBSD)', 'Cassava-Green Mottle (CGM)', 'Cassava-Healthy',
                'Cassava-Mosaic Disease (CMD)', 'Corn-cercospora leaf spot gray leaf spot', 'Corn-common rust',
                'Corn-healthy', 'Corn-northern leaf blight', 'Grape-black rot', 'Grape-esca (black measles)',
                'Grape-healthy', 'Grape-leaf blight (isariopsis leaf spot)', 'Mango-Anthracnose Fungal Leaf Disease',
                'Mango-Healthy Leaf', 'Mango-Rust Leaf Disease', 'Potato-early blight', 'Potato-healthy',
                'Potato-late blight', 'Rice-BrownSpot', 'Rice-Healthy', 'Rice-Hispa', 'Rice-LeafBlast',
                'Rose-Healthy Leaf', 'Rose-Rust', 'Rose-sawfly slug', 'Tomato-bacterial spot', 'Tomato-early blight',
                'Tomato-healthy', 'Tomato-late blight', 'Tomato-leaf mold', 'Tomato-mosaic virus',
                'Tomato-septoria leaf spot', 'Tomato-spider mites two-spotted spider mite', 'Tomato-target spot',
                'Tomato-yellow leaf curl virus']


# Precaution / treatment lookup
def get_precaution(label: str) -> str:
    """Return a recommended precaution or treatment string for a predicted class label."""
    # Specific exact-label mappings
    precaution_map = {
        'Bell Pepper-bacterial spot': 'Use pathogen-free seeds and transplants; avoid overhead irrigation; consider copper-based sprays if needed.',
        'Bell Pepper-healthy': 'Continue routine scouting and balanced fertilization; avoid excess nitrogen.',
        'Cassava-Bacterial Blight (CBB)': 'Use resistant varieties and clean planting materials; sterilize tools between cuttings and remove infected plants.',
        'Cassava-Brown Streak Disease (CBSD)': 'Use certified virus-free cuttings and control whitefly vectors.',
        'Cassava-Green Mottle (CGM)': 'Monitor and control mites; use resistant cultivars when available.',
        'Cassava-Healthy': 'Maintain good field hygiene and regular scouting.',
        'Cassava-Mosaic Disease (CMD)': 'Control whitefly populations and use virus-free cuttings.',
        'Corn-cercospora leaf spot gray leaf spot': 'Rotate crops, manage residue, and use resistant hybrids where available.',
        'Corn-common rust': 'Plant resistant hybrids and consider foliar fungicides under high disease pressure.',
        'Corn-healthy': 'Maintain current best practices: crop rotation and balanced nutrition.',
        'Corn-northern leaf blight': 'Use resistant hybrids and rotate with non-host crops.',
        'Grape-black rot': 'Prune for air flow, remove mummies, and apply fungicides during wet periods.',
        'Grape-esca (black measles)': 'Remove infected wood and protect pruning wounds; consult a viticulture specialist for severe cases.',
        'Grape-healthy': 'Keep canopy open and continue regular monitoring.',
        'Grape-leaf blight (isariopsis leaf spot)': 'Apply appropriate fungicides and improve canopy ventilation.',
        'Mango-Anthracnose Fungal Leaf Disease': 'Prune and remove infected tissue; apply copper fungicides during wet seasons.',
        'Mango-Healthy Leaf': 'Maintain orchard hygiene and monitor during rainy periods.',
        'Mango-Rust Leaf Disease': 'Use sulfur-based sprays and remove fallen infected leaves.',
        'Potato-early blight': 'Rotate crops, maintain plant vigor, and avoid late-day overhead irrigation.',
        'Potato-healthy': 'Continue routine scouting and use certified seed tubers.',
        'Potato-late blight': 'Use certified seed tubers, remove volunteer potatoes, and apply preventative fungicides.',
        'Rice-BrownSpot': 'Improve soil fertility (potassium) and use treated seed.',
        'Rice-Healthy': 'Keep standard water and nutrient management; continue scouting.',
        'Rice-Hispa': 'Monitor for larvae; use recommended insecticides or neem-based controls.',
        'Rice-LeafBlast': 'Use resistant varieties and avoid excessive nitrogen application.',
        'Rose-Healthy Leaf': 'Maintain good air circulation and remove debris.',
        'Rose-Rust': 'Remove and burn infected leaves; apply fungicides such as sulfur if needed.',
        'Rose-sawfly slug': 'Physically remove larvae or use insecticidal soap or neem oil for heavy infestations.',
        'Tomato-bacterial spot': 'Avoid overhead watering, use pathogen-free transplants, and consider copper-based sprays.',
        'Tomato-early blight': 'Rotate crops, remove infected lower leaves, and maintain plant spacing for airflow.',
        'Tomato-healthy': 'Continue routine scouting and balanced fertilization; monitor for pests.',
        'Tomato-late blight': 'Apply preventative fungicides and remove volunteer plants promptly.',
        'Tomato-leaf mold': 'Increase ventilation in greenhouses and keep humidity low.',
        'Tomato-mosaic virus': 'Remove and destroy infected plants; do not compost; sanitize hands and tools.',
        'Tomato-septoria leaf spot': 'Remove lower infected leaves and avoid working when plants are wet.',
        'Tomato-spider mites two-spotted spider mite': 'Increase humidity, use miticides or release predatory mites.',
        'Tomato-target spot': 'Maintain a preventative fungicide schedule and reduce canopy density.',
        'Tomato-yellow leaf curl virus': 'Control whiteflies and use reflective mulches or fine mesh in protected production.'
    }

    # Normalize label for simple pattern matches
    lower = label.lower()

    # General rules before default map lookup
    if 'bacterial blight' in lower or 'bacterial blight (cbb)' in lower:
        return 'Clean cuttings and tool sterilization; use resistant varieties if available.'
    if 'late blight' in lower:
        return 'Preventative fungicides and removing volunteer plants; use certified disease-free seed/tubers.'
    if 'bacterial spot' in lower and 'pepper' in lower:
        return 'Use pathogen-free seeds and avoid overhead irrigation; consider copper-based bactericides.'
    if 'healthy' in lower:
        return 'Continue routine scouting and balanced fertilization.'

    # Exact label lookup
    return precaution_map.get(label, 'No specific precaution found. Monitor and follow good cultural practices.')

# Function to predict the class of the plant disease
def model_prediction(test_image_path):
    # Ensure image is RGB (drop alpha channel if present) and resized
    image = Image.open(test_image_path).convert('RGB')
    image = image.resize((img_width, img_height))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.expand_dims(input_arr, axis=0)
    input_arr = input_arr / 255.0
    predictions = model.predict(input_arr)
    return np.argmax(predictions)

@app.route('/')
def index():
    # Serve the home page as the default root (login disabled)
    return render_template('disease-recognition.html')

# Login disabled: login route removed

# Removed separate /home route — root now serves the disease recognition page.

@app.route('/disease-recognition', methods=['GET', 'POST'])
def disease_recognition():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                file.save(filepath)
            except UnicodeEncodeError:
                flash('File name contains unsupported characters.')
                return redirect(request.url)
            try:
                result_index = model_prediction(filepath)
            except Exception as e:
                flash('Prediction error: {}'.format(str(e)))
                return redirect(request.url)
            prediction = class_labels[result_index]
            precaution = get_precaution(prediction)
            return render_template('prediction.html', predicted_disease=prediction, precaution=precaution, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('disease-recognition.html')

# Login/logout removed: no session management in this app

if __name__ == '__main__':
    app.run(debug=True)