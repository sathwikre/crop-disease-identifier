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
model_path = os.path.join(os.path.dirname(__file__), 'Team3model.h5')
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



    # Select the appropriate map based on language
    precaution_map = precaution_map_en if lang == 'en' else precaution_map_te

    # Normalize label for simple pattern matches
    lower = label.lower()

    # General rules before default map lookup
    if lang == 'en':
        if 'bacterial blight' in lower or 'bacterial blight (cbb)' in lower:
            return 'Clean cuttings and tool sterilization; use resistant varieties if available.'
        if 'late blight' in lower:
            return 'Preventative fungicides and removing volunteer plants; use certified disease-free seed/tubers.'
        if 'bacterial spot' in lower and 'pepper' in lower:
            return 'Use pathogen-free seeds and avoid overhead irrigation; consider copper-based bactericides.'
        if 'healthy' in lower:
            return 'Continue routine scouting and balanced fertilization.'
        default_msg = 'No specific precaution found. Monitor and follow good cultural practices.'
    else:  # Telugu
        if 'bacterial blight' in lower or 'bacterial blight (cbb)' in lower:
            return 'కటింగ్‌లను శుభ్రం చేయడం మరియు సాధనాల స్టెరిలైజేషన్; అందుబాటులో ఉంటే నిరోధక జాతులను ఉపయోగించండి.'
        if 'late blight' in lower:
            return 'నివారక ఫంగిసైడ్‌లు మరియు స్వచ్ఛంద మొక్కలను తీసివేయడం; ధృవీకరించబడిన వ్యాధి-ఉచిత విత్తనం/కందలను ఉపయోగించండి.'
        if 'bacterial spot' in lower and 'pepper' in lower:
            return 'రోగకారక-ఉచిత విత్తనాలను ఉపయోగించండి మరియు ఓవర్‌హెడ్ నీటిపోయేలా నివారించండి; కాపర్-ఆధారిత బాక్టీరిసైడ్‌లను పరిగణించండి.'
        if 'healthy' in lower:
            return 'క్రమం తప్పకుండా స్కౌటింగ్ మరియు సమతుల్య ఎరువును కొనసాగించండి.'
        default_msg = 'నిర్దిష్ట జాగ్రత్త కనుగొనబడలేదు. పర్యవేక్షించండి మరియు మంచి సాంస్కృతిక పద్ధతులను అనుసరించండి.'

    # Exact label lookup
    return precaution_map.get(label, default_msg)

# Function to predict the class of the plant disease
def model_prediction(test_image_path):
    # Ensure image is RGB (drop alpha channel if present) and resized
    try:
        image = Image.open(test_image_path).convert('RGB')
        image = image.resize((img_width, img_height))
        input_arr = tf.keras.preprocessing.image.img_to_array(image)
        input_arr = np.expand_dims(input_arr, axis=0)
        input_arr = input_arr / 255.0
        predictions = model.predict(input_arr, verbose=0)
        result_index = int(np.argmax(predictions))
        print(f"Model prediction shape: {predictions.shape}, max index: {result_index}, max value: {np.max(predictions)}")
        return result_index
    except Exception as e:
        print(f"Error in model_prediction: {str(e)}")
        raise

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
                print(f"Prediction result index: {result_index}")  # Debug output
            except Exception as e:
                print(f"Prediction error: {str(e)}")  # Debug output
                flash('Prediction error: {}'.format(str(e)))
                return redirect(request.url)
            
            # Validate result_index
            if result_index is None or result_index < 0 or result_index >= len(class_labels):
                error_msg = f'Invalid prediction index: {result_index}'
                print(error_msg)  # Debug output
                flash(error_msg)
                return redirect(request.url)
            
            prediction = class_labels[result_index]
            print(f"Predicted disease: {prediction}")  # Debug output
            
            # Get language from request (cookie or default to 'en')
            lang = request.cookies.get('language', 'en')
            if lang not in ['en', 'te']:
                lang = 'en'
            precaution = get_precaution(prediction, lang)
            print(f"Precaution retrieved: {precaution[:50]}...")  # Debug output
            
            return render_template('prediction.html', predicted_disease=prediction, precaution=precaution, image_url=url_for('static', filename='uploads/' + filename))
    return render_template('disease-recognition.html')

# Login/logout removed: no session management in this app

if __name__ == '__main__':
    app.run(debug=True)
