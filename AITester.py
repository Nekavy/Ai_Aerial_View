import os
import cv2
import numpy as np
import joblib  # Para carregar o modelo

def load_model(model_path="models/water_earth_model.pkl"):
    """Carrega o modelo treinado."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modelo não encontrado em {model_path}. Por favor, treine o modelo primeiro.")
    model = joblib.load(model_path)
    print(f"Modelo carregado de {model_path}")
    return model

def process_image(image_path):
    """Processa a imagem para o modelo."""
    image = cv2.imread(image_path)
    image_resized = cv2.resize(image, (256, 256))
    image_pixels = image_resized.reshape(-1, 3)
    return image_resized, image_pixels

def predict_image(image_pixels, model):
    """Realiza a predição com o modelo."""
    predicted_labels = model.predict(image_pixels)
    predicted_mask = predicted_labels.reshape(256, 256).astype(np.uint8) * 255
    return predicted_mask
