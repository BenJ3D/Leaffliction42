import os
import sys
import argparse
import json
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from Transformation import gaussian_blur, create_masked_image, _create_binary_mask

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Erreur: {message}\n")
        sys.exit(2)

def load_model():
    """Charger le modèle entraîné et le mappage des classes."""
    model_path = 'model/leaffliction_model.keras'
    class_mapping_path = 'model/class_mapping.json'
    
    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle '{model_path}' n'existe pas.")
        return None, None
        
    if not os.path.exists(class_mapping_path):
        print(f"Erreur: Le fichier de mappage des classes '{class_mapping_path}' n'existe pas.")
        return None, None
    
    try:
        model = keras.models.load_model(model_path)
        
        with open(class_mapping_path, 'r') as f:
            class_mapping = json.load(f)
            
        return model, class_mapping
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None

def preprocess_image(img_path, img_size=(224, 224)):
    """Prétraiter l'image pour la prédiction."""
    try:
        # Charger l'image
        img = cv2.imread(img_path)
        if img is None:
            print(f"Erreur: Impossible de charger l'image '{img_path}'.")
            return None, None
        
        # Convertir en RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Redimensionner
        img_resized = cv2.resize(img_rgb, img_size)
        
        # Normaliser
        img_normalized = img_resized / 255.0
        
        # Transformation pour la visualisation
        masked_img = create_masked_image(img)
        
        return img_normalized, masked_img
    except Exception as e:
        print(f"Erreur lors du prétraitement de l'image: {e}")
        return None, None

def predict_disease(model, class_mapping, img_preprocessed):
    """Prédire la maladie d'une feuille."""
    try:
        # Ajouter la dimension du batch
        img_batch = np.expand_dims(img_preprocessed, axis=0)
        
        # Faire la prédiction
        predictions = model.predict(img_batch)
        
        # Trouver la classe avec la plus grande probabilité
        pred_idx = np.argmax(predictions[0])
        confidence = predictions[0][pred_idx] * 100
        
        # Obtenir le nom de la classe
        pred_class = class_mapping.get(str(pred_idx), "Classe inconnue")
        
        return pred_class, confidence
    except Exception as e:
        print(f"Erreur lors de la prédiction: {e}")
        return "Erreur de prédiction", 0.0

def display_results(original_img, transformed_img, pred_class, confidence):
    """Afficher les résultats de la prédiction."""
    plt.figure(figsize=(12, 5))
    
    # Image originale
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB))
    plt.title("Image originale")
    plt.axis('off')
    
    # Image transformée
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    plt.title("Image traitée")
    plt.axis('off')
    
    # Ajouter le texte de prédiction
    plt.suptitle(f"Prédiction: {pred_class} (Confiance: {confidence:.2f}%)", fontsize=14)
    
    plt.tight_layout()
    plt.show()

def main():
    parser = CustomArgumentParser(
        description="Programme pour prédire la maladie d'une feuille à partir d'une image."
    )
    parser.add_argument(
        "image",
        help="Chemin de l'image à analyser"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        parser.error(f"L'image '{args.image}' n'existe pas.")

    # Charger le modèle et le mappage des classes
    model, class_mapping = load_model()
    if model is None or class_mapping is None:
        return

    # Prétraiter l'image
    img_preprocessed, transformed_img = preprocess_image(args.image)
    if img_preprocessed is None:
        return

    # Faire la prédiction
    pred_class, confidence = predict_disease(model, class_mapping, img_preprocessed)

    # Charger l'image originale pour l'affichage
    original_img = cv2.imread(args.image)

    # Afficher les résultats
    display_results(original_img, transformed_img, pred_class, confidence)
    print(f"Prédiction: {pred_class}")
    print(f"Confiance: {confidence:.2f}%")

if __name__ == "__main__":
    main()
