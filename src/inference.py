import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

class BrainTumorClassifier:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model = None
        if os.path.exists(model_path):
            self.model = load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file not found at {model_path}")

    def get_class_name(self, class_idx):
        classes = {
            0: 'Tumor (Glioma)',
            1: 'Tumor (Meningioma)',
            2: 'No Tumor',
            3: 'Tumor (Pituitary)'
        }
        return classes.get(class_idx, 'Unknown')

    def predict(self, image_path):
        if self.model is None:
            return "Model not loaded"
        
        # Load and preprocess image
        img = image.load_img(image_path, target_size=(128, 128))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        
        # Predict
        res = self.model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        
        return self.get_class_name(classification)
