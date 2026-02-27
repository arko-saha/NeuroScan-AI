from flask import Flask, render_template, request
import os
import sys

# Add src to path to import inference
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inference import BrainTumorClassifier

app = Flask(__name__)

# Configure upload folder
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'images')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model - using a relative path that can be easily updated
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'model_vgg.h5')
classifier = BrainTumorClassifier(MODEL_PATH)

@app.route('/', methods=['GET'])
def index():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    if 'imagefile' not in request.files:
        return render_template('index.html', prediction="No file uploaded")
    
    imagefile = request.files['imagefile']
    if imagefile.filename == '':
        return render_template('index.html', prediction="No file selected")

    if imagefile:
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], imagefile.filename)
        imagefile.save(image_path)
        
        # Use the modular classifier
        prediction = classifier.predict(image_path)
        
        return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(port=3000, debug=True)
