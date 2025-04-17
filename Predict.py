import argparse
import json
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras

from Transformation import (
    create_masked_image,
)

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # 0=DEBUG, 1=INFO, 2=WARNING,
# 3=ERROR
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'  # Désactive l'utilisation de CUDA
# os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'  # Empêche TF de
# préallouer toute la mémoire
# Rediriger temporairement stderr pour supprimer les messages initiaux
# d'erreur
# original_stderr = sys.stderr
# sys.stderr = open(os.devnull, 'w')


# Taille des images attendue par le modèle (doit correspondre à l'entraînement)
IMG_SIZE = 256


class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Erreur: {message}\n")
        sys.exit(2)


def load_model(model_path=None):
    """Charger le modèle entraîné et le mappage des classes."""
    if model_path is None:
        model_path = "model/leaffliction_model.keras"
        class_mapping_path = "model/class_mapping.json"
    else:
        # Trouver le répertoire du modèle pour y chercher le fichier de mappage
        model_dir = os.path.dirname(model_path)
        class_mapping_path = os.path.join(model_dir, "class_mapping.json")

    if not os.path.exists(model_path):
        print(f"Erreur: Le modèle '{model_path}' n'existe pas.")
        return None, None

    if not os.path.exists(class_mapping_path):
        print(
            f"Erreur: Le fichier de mappage des classes '{class_mapping_path}'"
            f" n'existe pas."
        )
        return None, None

    try:
        model = keras.models.load_model(model_path)

        with open(class_mapping_path, "r") as f:
            class_mapping = json.load(f)

        return model, class_mapping
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None, None


def preprocess_image(img_path, img_size=(IMG_SIZE, IMG_SIZE)):
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
        predictions = model.predict(img_batch, verbose=0)

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
    plt.axis("off")

    # Image transformée
    plt.subplot(1, 2, 2)
    plt.imshow(cv2.cvtColor(transformed_img, cv2.COLOR_BGR2RGB))
    plt.title("Image traitée")
    plt.axis("off")

    # Ajouter le texte de prédiction
    plt.suptitle(f"Prédiction: {pred_class}", fontsize=14)

    plt.tight_layout()
    plt.show()


def evaluate_directory(model, class_mapping, test_dir, percentage=100):
    """Évaluer le modèle sur toutes les images d'un répertoire et enregistrer
    les résultats dans un fichier.

    Args:
        model: Le modèle à évaluer
        class_mapping: Le mappage des classes
        test_dir: Le répertoire contenant les images de test
        percentage: Le pourcentage d'images à évaluer (1-100)
    """
    total_images = 0
    correct_predictions = 0
    per_class_stats = {}
    log_filepath = "predict_results.txt"

    with open(log_filepath, "w") as log_file:
        log_file.write(f"Évaluation du répertoire : {test_dir}\n")
        log_file.write(f"Pourcentage d'images évaluées : {percentage}%\n")
        log_file.write("-" * 60 + "\n")
        print(f"Évaluation du répertoire : {test_dir}")
        print(f"Pourcentage d'images évaluées : {percentage}%")
        print("-" * 60)

        # Parcourir les sous-dossiers (chaque sous-dossier doit porter le nom
        # d'une classe)
        for real_class in sorted(os.listdir(test_dir)):
            class_dir = os.path.join(test_dir, real_class)
            if not os.path.isdir(class_dir):
                continue

            image_files = [
                f
                for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if not image_files:
                continue

            # Sélectionner un sous-ensemble des images en fonction du
            # pourcentage
            if percentage < 100:
                np.random.seed(42)  # Pour la reproductibilité
                selected_count = max(
                    1, int(len(image_files) * percentage / 100)
                )
                image_files = np.random.choice(
                    image_files, selected_count, replace=False
                ).tolist()

            log_line = f"Traitement de la classe : {real_class}\n"
            print(log_line, end="")
            log_file.write(log_line)
            for img_file in image_files:
                img_path = os.path.join(class_dir, img_file)
                img_preprocessed, _ = preprocess_image(img_path)
                if img_preprocessed is None:
                    log_line = (f"{img_path} | {real_class:<16} | "
                                f"PREPROC ERROR       | False\n")
                    print(log_line, end="")
                    log_file.write(log_line)
                    continue

                total_images += 1
                pred_class, _ = predict_disease(
                    model, class_mapping, img_preprocessed
                )
                is_correct = pred_class == real_class
                log_line = (f"{img_path} | {real_class:<16} | {pred_class:<16}"
                            f" | {is_correct}\n")
                print(log_line, end="")
                log_file.write(log_line)

                if real_class not in per_class_stats:
                    per_class_stats[real_class] = {"total": 0, "correct": 0}
                per_class_stats[real_class]["total"] += 1
                if is_correct:
                    correct_predictions += 1
                    per_class_stats[real_class]["correct"] += 1

        log_file.write("-" * 60 + "\n")
        print("-" * 60)
        delta = total_images - correct_predictions
        if total_images > 0:
            overall_accuracy = (correct_predictions / total_images) * 100
            log_file.write("\nRésultat final:\n")
            log_file.write(f"  Images testées       : {total_images}\n")
            log_file.write(f"  Prédictions correctes: {correct_predictions}\n")
            # if delta > 0 :
            log_file.write(f"  Delta : {delta}\n")
            log_file.write(
                f"  Précision (Accuracy) : {overall_accuracy:.2f}%\n"
            )
            print("\nRésultat final:")
            print(f"  Images testées       : {total_images}")
            print(f"  Prédictions correctes: {correct_predictions}")
            print(f"  Précision (Accuracy) : {overall_accuracy:.2f}%")
            # Afficher le détail par classe
            for cls, stats in per_class_stats.items():
                class_accuracy = (
                    (stats["correct"] / stats["total"]) * 100
                    if stats["total"] > 0
                    else 0.0
                )
                detail_line = (f"  {cls}: {stats['total']} images, précision:"
                               f" {class_accuracy:.2f}%\n")
                log_file.write(detail_line)
                print(detail_line, end="")
        else:
            log_file.write("Aucune image n'a pu être traitée.\n")
            print("Aucune image n'a pu être traitée.")


def main():
    parser = CustomArgumentParser(
        description="Programme pour prédire la maladie d'une feuille"
                    " (mode image ou répertoire)."
    )
    parser.add_argument(
        "input", help="Chemin de l'image ou du "
                      "répertoire à analyser"
    )
    parser.add_argument(
        "--percentage",
        type=int,
        default=100,
        help="Pourcentage d'images à évaluer (1-100)",
    )
    parser.add_argument(
        "--model",
        help="Chemin vers le modèle à utiliser (le fichier"
             " class_mapping.json doit être dans le même répertoire)",
    )
    args = parser.parse_args()

    # Vérification de la validité du pourcentage
    if args.percentage < 1 or args.percentage > 100:
        parser.error("Le pourcentage doit être compris entre 1 et 100.")

    if not os.path.exists(args.input):
        parser.error(f"Le chemin '{args.input}' n'existe pas.")

    # Charger le modèle et le mappage des classes
    model, class_mapping = load_model(args.model)
    if model is None or class_mapping is None:
        return

    # Si c'est un répertoire, on lance l'évaluation globale
    if os.path.isdir(args.input):
        evaluate_directory(model, class_mapping, args.input, args.percentage)
    # Sinon, on effectue la prédiction sur une seule image avec affichage
    elif os.path.isfile(args.input):
        # Traiter une seule image
        img_preprocessed, transformed_img = preprocess_image(args.input)
        if img_preprocessed is None:
            print(f"Erreur: Impossible de prétraiter l'image '{args.input}'.")
            return

        pred_class, confidence = predict_disease(
            model, class_mapping, img_preprocessed
        )
        original_img = cv2.imread(args.input)
        display_results(original_img, transformed_img, pred_class, confidence)
        print(f"Prédiction: {pred_class}")
        print(f"Confiance: {confidence:.2f}%")
    else:
        parser.error(
            "Le chemin spécifié n'est ni un fichier ni un répertoire."
        )


if __name__ == "__main__":
    main()
