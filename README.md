# NeuroScan AI: Brain Tumor Detection from MRI

<img width="3780" height="1890" alt="your-image-here" src="https://github.com/user-attachments/assets/2779095b-c8ca-4eaa-aebc-c526e6939e38" />


## 🚀 Overview
**NeuroScan AI** is a professional-grade medical imaging application that leverages Deep Learning to detect and classify brain tumors from MRI scans. Developed using **TensorFlow/Keras**, the system classifies MRI images into four distinct categories with high precision, providing a robust interface for researchers and clinicians to interact with neural network predictions.

### Key Features
-   **Multi-Class Classification**: Identifies Glioma, Meningioma, Pituitary tumors, and No Tumor cases.
-   **VGG-Based Architecture**: Utilizes a customized VGG-16 backbone optimized for subtle medical feature extraction.
-   **Modular Design**: Clean separation of inference logic, web interface, and research artifacts.
-   **Premium UI**: A sleek, modern dark-mode dashboard for seamless image analysis.

---

## 🏗️ Project Structure

```text
├── app/                  # Web Application
│   ├── app.py            # Flask Server
│   ├── templates/        # UI Components (HTML/CSS)
│   └── images/           # Temporary Upload Storage (Git Ignored)
├── models/               # Trained Model Storage (.h5)
├── notebooks/            # Research & Training Artifacts
│   ├── Visualization.ipynb
│   ├── Modeling.ipynb
│   └── Evaluation.ipynb
├── src/                  # Production Source Code
│   └── inference.py      # Modular ML Inference Pipeline
├── requirements.txt      # Dependency Specification
├── README.md             # Project Documentation
└── .gitignore            # Git exclusion rules
```

---

## 🛠️ Installation & Setup

### 1. Prerequisites
-   Python 3.8+
-   Virtual Environment (Recommended)

### 2. Setup
```bash
# Clone the repository
git clone https://github.com/arko-saha/Brain-Tumor-Detection-from-MRIs-using-Deep-Learning-techniques
cd Brain-Tumor-Detection-from-MRIs-using-Deep-Learning-techniques

# Install dependencies
pip install -r requirements.txt
```

### 3. Model Deployment
Ensure the pre-trained model (`model_vgg.h5`) is placed in the `models/` directory.

### 4. Running the Application
```bash
python app/app.py
```
Open your browser and navigate to `http://localhost:3000`.

---

## 🔬 Methodology

### Data Preprocessing
Images are resized to **128x128 pixels** and normalized to ensure consistent input distribution. Data augmentation techniques (rotations, flips, zooms) were employed during training to improve generalization.

### Model Architecture
The primary model is built on a **VGG-16** architecture, fine-tuned on the Brain Tumor MRI Dataset.
-   **Optimizer**: Adam
-   **Loss Function**: Categorical Cross-Entropy
-   **Performance**: Achieved >95% accuracy on the test set.

---

## 📊 Dataset
The project uses the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from Kaggle, containing thousands of scans across four classes.

---

## 👨‍💻 Author
**Arko Saha**  
[LinkedIn](https://www.linkedin.com/in/arko-saha/) | [GitHub](https://github.com/arko-saha)

---

## 📝 License
This project is licensed under the MIT License - see the LICENSE file for details.
