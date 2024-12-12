import os
import cv2
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib  # Para salvar o modelo

def train_ai_model():
    """Treina o modelo de IA para distinguir água de terra."""
    image_folder = "ImgWater"
    mask_folder = "ImgWaterMask"
    model_path = "models/water_earth_model.pkl"

    images, labels = [], []

    # Carregar imagens e máscaras para treinamento
    for filename in os.listdir(image_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(image_folder, filename)
            mask_path = os.path.join(mask_folder, filename)

            # Carregar imagem e máscara
            image = cv2.imread(image_path)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is not None and mask is not None:
                image = cv2.resize(image, (256, 256))
                mask = cv2.resize(mask, (256, 256))

                image_pixels = image.reshape(-1, 3)
                mask_labels = (mask.reshape(-1) > 128).astype(int)

                images.append(image_pixels)
                labels.append(mask_labels)

    # Formatar dados
    X = np.vstack(images)
    y = np.hstack(labels)

    # Dividir em treinamento e validação
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Treinamento do modelo
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Avaliar o modelo
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Precisão do modelo: {accuracy * 100:.2f}%")

    # Salvar o modelo
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Modelo salvo em {model_path}")
