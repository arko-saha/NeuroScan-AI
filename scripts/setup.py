import os
import shutil
import kagglehub
import gdown

def setup_dataset():
    print("--- Downloading Dataset from Kaggle ---")
    # Download latest version
    path = kagglehub.dataset_download("masoudnickparvar/brain-tumor-mri-dataset")
    
    # Destination directory
    dest_dir = os.path.join(os.getcwd(), "data")
    os.makedirs(dest_dir, exist_ok=True)
    
    print(f"Dataset downloaded to {path}")
    print(f"Moving dataset to {dest_dir}...")
    
    # Move files to data/ directory
    for item in os.listdir(path):
        s = os.path.join(path, item)
        d = os.path.join(dest_dir, item)
        if os.path.isdir(s):
            if os.path.exists(d):
                shutil.rmtree(d)
            shutil.copytree(s, d)
        else:
            shutil.copy2(s, d)
    
    print("Dataset setup complete.")

def setup_model():
    print("\n--- Downloading Pre-trained Model Weights ---")
    model_dir = os.path.join(os.getcwd(), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    model_path = os.path.join(model_dir, "model_vgg.h5")
    
    # Load from .env if available, otherwise use fallback
    from dotenv import load_dotenv
    load_dotenv()
    
    file_id = os.getenv("GDRIVE_FILE_ID", "1iZWY3yfwzLOtnq4OzoSK1HZSgnByTD3u")
    url = f'https://drive.google.com/uc?id={file_id}'
    
    if not os.path.exists(model_path):
        print("Downloading model weights from Google Drive...")
        gdown.download(url, model_path, quiet=False)
        print("Model download complete.")
    else:
        print("Model weights already exist in models/ directory.")

if __name__ == "__main__":
    try:
        setup_dataset()
        setup_model()
        print("\n✅ Setup complete! You can now run the application or explore the notebooks.")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("Please ensure you have configured Kaggle API credentials if prompted.")
