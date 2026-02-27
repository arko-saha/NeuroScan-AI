from flask import Flask, render_template, request
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2

from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the model
dnn1 = load_model('D:/Projects/UoR/Term 2/DLA/Coursework/Submission/model_vgg.h5')

def names(number):
    if number == 0:
        return 'Tumor'
    elif number == 1:
        return 'Tumor'
    elif number == 2:
        return 'No Tumor'
    elif number == 3:
        return 'Tumor'
    else:
        return 'unknown'
    
@app.route('/', methods=['GET'])

def hello_world():
    return render_template("index.html")

@app.route('/', methods=['POST'])
def predict():
    # Get the uploaded file
    imagefile = request.files['imagefile']
    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Load and resize the image
    img = image.load_img(image_path, target_size=(128, 128))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    # Make a prediction on the uploaded image
    res = dnn1.predict_on_batch(x)

    # Get the classification with highest probability
    classification = np.where(res == np.amax(res))[1][0]

    return render_template('index.html', prediction=names(classification))

if __name__ == '__main__':
    app.run(port=3000, debug=True)
