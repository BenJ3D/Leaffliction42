import os
import argparse
import sys
import json
import numpy as np
import shutil # copie fichier
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from utils.augmentation_utils import augment_dataset_balanced

# Variables globales de configuration pour l'entraînement
# Modifiez ces valeurs pour ajuster l'entraînement
IMG_SIZE = 256                 # Taille des images (hauteur et largeur)
EPOCHS = 50                    # Nombre d'époques d'entraînement
BATCH_SIZE = 32                # Taille du lot pour l'entraînement
LEARNING_RATE = 0.001          # Taux d'apprentissage initial
DROPOUT_RATE = 0.5             # Taux de dropout pour éviter le surapprentissage
EARLY_STOP_PATIENCE = 10       # Patience pour l'arrêt anticipé (époques)
REDUCE_LR_PATIENCE = 10        # Patience pour la réduction du taux d'apprentissage

# Mode rapide pour les tests (mettre à True pour un entraînement plus rapide mais moins accuracy)
QUICK_MODE = False
if QUICK_MODE:
    IMG_SIZE = 64
    EPOCHS = 15
    BATCH_SIZE = 32
    LEARNING_RATE = 0.0018 

VALIDATION_SPLIT=0.2
class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Erreur: {message}\n")
        sys.exit(2)

def extract_classes(directory):
    """Extrait les noms de classes à partir des sous-dossiers d'un répertoire donné."""
    classes = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and not item == "output":
            classes.append(item)
    return sorted(classes)

def save_classes_to_json(classes):
    """Sauvegarde les noms de classes dans un fichier JSON"""
    classes_json = {"classes": classes}
    os.makedirs("train", exist_ok=True)
    json_path = os.path.join(os.getcwd(), "train/classes.json")
    with open(json_path, 'w') as f:
        json.dump(classes_json, f, indent=4)
    return json_path

def create_cnn_model(input_shape, num_classes):
    """Créé un modèle CNN pour la classification d'images."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def split_dataset_and_save(src_dir, train_dir, val_dir, validation_split=VALIDATION_SPLIT):
    """
    Divise le dataset en ensembles d'entraînement et de validation
    en sauvegardant les fichiers dans les dossiers correspondants.
    Renvoie le nombre de fichiers pour chaque ensemble.
    """
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    train_count = 0
    val_count = 0
    
    # Parcourir les classes (sous-dossiers)
    for class_name in os.listdir(src_dir):
        class_dir = os.path.join(src_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Créer les dossiers de sortie pour cette classe
        train_class_dir = os.path.join(train_dir, class_name)
        val_class_dir = os.path.join(val_dir, class_name)
        os.makedirs(train_class_dir, exist_ok=True)
        os.makedirs(val_class_dir, exist_ok=True)
        
        # Lister les images
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Mélanger aléatoirement
        np.random.seed(42)
        np.random.shuffle(image_files)
        
        # Calculer l'index de séparation
        split_idx = int(len(image_files) * (1 - validation_split))
        
        # Séparer en ensembles d'entraînement et de validation
        train_files = image_files[:split_idx]
        val_files = image_files[split_idx:]
        
        train_count += len(train_files)
        val_count += len(val_files)
        
        print(f"Classe '{class_name}': {len(train_files)} images d'entraînement, {len(val_files)} images de validation")
        
        # Copier les fichiers d'entraînement
        for img_file in tqdm(train_files, desc=f"Copie train {class_name}"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(train_class_dir, img_file)
            shutil.copy2(src_path, dst_path)
        
        # Copier les fichiers de validation
        for img_file in tqdm(val_files, desc=f"Copie validation {class_name}"):
            src_path = os.path.join(class_dir, img_file)
            dst_path = os.path.join(val_class_dir, img_file)
            shutil.copy2(src_path, dst_path)
    
    return train_count, val_count

def train_with_generator(train_dir, val_dir, num_classes, img_size=(IMG_SIZE, IMG_SIZE)):
    """
    Entraîne le modèle en utilisant des générateurs pour économiser la mémoire.
    """
    # Générateur pour l'ensemble d'entraînement avec data augmentation
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    # Générateur pour l'ensemble de validation (seulement normalisation)
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    # Charger les images à partir des dossiers
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    validation_generator = val_datagen.flow_from_directory(
        val_dir,
        target_size=img_size,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Créer et compiler le modèle
    model = create_cnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    print(model.summary())
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=EARLY_STOP_PATIENCE, 
            monitor='val_accuracy', 
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            factor=0.2, 
            patience=REDUCE_LR_PATIENCE, 
            min_lr=0.00001
        )
    ]
    
    # Entraîner le modèle
    print("\nDébut de l'entraînement du modèle...")
    print(f"Configuration: epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    
    # Calculer steps_per_epoch et validation_steps
    steps_per_epoch = train_generator.samples // BATCH_SIZE
    if steps_per_epoch == 0:
        steps_per_epoch = 1
    
    validation_steps = validation_generator.samples // BATCH_SIZE
    if validation_steps == 0:
        validation_steps = 1
    
    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        callbacks=callbacks
    )
    
    # Évaluation du modèle
    print("\nÉvaluation du modèle...")
    val_loss, val_acc = model.evaluate(validation_generator, steps=validation_steps)
    print(f"Précision sur l'ensemble de validation: {val_acc:.4f}")
    
    # Tracer les courbes d'apprentissage
    plot_training_history(history)
    
    return model, history, train_generator.class_indices

def plot_training_history(history):
    """Plot training & validation accuracy and loss."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Précision du modèle')
    ax1.set_ylabel('Précision')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'], loc='upper left')
    
    # Loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Perte du modèle')
    ax2.set_ylabel('Perte')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'], loc='upper left')
    
    plt.tight_layout()
    plt.savefig('train/training_history.png')
    print("Graphique d'apprentissage sauvegardé dans 'training_history.png'")
    plt.close()

def save_model(model, class_indices):
    """Save the trained model and class mappings."""
    # Init dossier 'model'
    os.makedirs('model', exist_ok=True)
    
    # Save modèle au format Keras
    model_path = 'model/leaffliction_model.keras'
    model.save(model_path)
    
    # Inverser le dictionnaire class_indices pour obtenir index -> nom_classe
    class_mapping = {str(idx): class_name for class_name, idx in class_indices.items()}
    
    # Save le mappage des classes
    with open('model/class_mapping.json', 'w') as f:
        json.dump(class_mapping, f, indent=4)
    
    print("Modèle sauvegardé dans le dossier 'model/'")
    return model_path

def main():
    parser = CustomArgumentParser(
        description="Programme pour lancer l'entraînement sur un modèle de détection de maladies des feuilles."
    )
    parser.add_argument(
        "src",
        help="Chemin du dossier récursif contenant toutes les images d'entrainement"
    )

    args = parser.parse_args()

    if not os.path.exists(args.src):
        parser.error(f"Le dossier source '{args.src}' n'existe pas.")
    
    # Création des dossiers nécessaires
    os.makedirs("train", exist_ok=True)
    
    # Équilibrage automatique des classes
    print("Équilibrage des classes avec augmentation d'images...")
    balanced_dir = os.path.join("train", "augmented_directory")
    if augment_dataset_balanced(args.src, balanced_dir):
        train_directory = balanced_dir
        print(f"Équilibrage terminé. Utilisation du dossier '{train_directory}' pour l'entraînement.")
    else:
        print("L'équilibrage des classes a échoué. Utilisation du dossier d'origine pour l'entraînement.")
        train_directory = args.src
    
    # Extraction des classes
    classes = extract_classes(train_directory)
    if not classes:
        print("Aucune classe (sous-dossier) trouvée dans le répertoire source.")
        return
    
    json_path = save_classes_to_json(classes)
    print(f"Classes disponibles ({len(classes)}):")
    for cls in classes:
        print(f"- {cls}")
    print(f"Liste des classes sauvegardée dans: {json_path}")
    
    # Image preprocessing settings
    img_size = (IMG_SIZE, IMG_SIZE)
    
    # Dossiers pour l'entraînement et la validation
    train_dir = "train/train_dir"
    validation_dir = "train/validation_dir"
    
    # Diviser le dataset et save les images dans les dossiers correspondants
    print("\nDivision et sauvegarde du dataset en ensembles d'entraînement et de validation...")
    train_count, val_count = split_dataset_and_save(train_directory, train_dir, validation_dir, VALIDATION_SPLIT)
    
    total_count = train_count + val_count
    print(f"\nEnsemble de données divisé: {total_count} images au total")
    print(f"Ensemble d'entraînement: {train_count} images ({train_count/total_count*100:.1f}%)")
    print(f"Ensemble de validation: {val_count} images ({val_count/total_count*100:.1f}%)")
    
    # Entraîner le modèle avec les générateurs d'images
    model, history, class_indices = train_with_generator(train_dir, validation_dir, len(classes), img_size)
    
    # Save le modèle
    model_path = save_model(model, class_indices)
    
    print(f"\nEntraînement terminé! Modèle sauvegardé dans: {model_path}")
    print("Vous pouvez maintenant utiliser ce modèle pour faire des prédictions avec Predict.py")

if __name__ == "__main__":
    main()