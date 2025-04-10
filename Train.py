import os
import argparse
import sys
import json
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

import tensorflow as tf

# Active la précision mixte pour accélérer l'entraînement sur GPU
mixed_precision = tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Variables globales de configuration pour l'entraînement
# Modifiez ces valeurs pour ajuster l'entraînement sans changer le code
IMG_SIZE = 224                 # Taille des images (hauteur et largeur)
EPOCHS = 50                    # Nombre d'époques d'entraînement
BATCH_SIZE = 32                # Taille du lot pour l'entraînement
LEARNING_RATE = 0.001          # Taux d'apprentissage initial
DROPOUT_RATE = 0.5             # Taux de dropout pour éviter le surapprentissage
VALIDATION_SPLIT = 0.2         # Proportion de données pour la validation
EARLY_STOP_PATIENCE = 5       # Patience pour l'arrêt anticipé (époques)
REDUCE_LR_PATIENCE = 5         # Patience pour la réduction du taux d'apprentissage

# Mode rapide pour les tests (mettre à True pour un entraînement plus rapide)
QUICK_MODE = False
if QUICK_MODE:
    IMG_SIZE = 64
    EPOCHS = 3
    BATCH_SIZE = 32

class CustomArgumentParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"Erreur: {message}\n")
        sys.exit(2)


# Vérifier la disponibilité du GPU
print("Nombre de GPUs disponibles:", len(tf.config.list_physical_devices('GPU')))
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    print(f"GPUs détectés: {gpus}")
    print("TensorFlow utilisera le GPU")
    # Configuration mémoire optimisée pour le GPU
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(f"Erreur de configuration du GPU: {e}")
else:
    print("Aucun GPU détecté. TensorFlow utilisera le CPU.")



def extract_classes(directory):
    """Extract class names from subdirectories."""
    classes = []
    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)
        if os.path.isdir(item_path) and not item == "output":
            classes.append(item)
    return sorted(classes)

def save_classes_to_json(classes):
    """Save the class names to a JSON file in the current working directory."""
    classes_json = {"classes": classes}
    json_path = os.path.join(os.getcwd(), "classes.json")
    with open(json_path, 'w') as f:
        json.dump(classes_json, f, indent=4)
    return json_path

def create_cnn_model(input_shape, num_classes):
    """Create a CNN model for image classification avec une architecture optimisée."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.GlobalAveragePooling2D(),  # Remplace Flatten pour réduire drastiquement les paramètres
        layers.Dense(256, activation='relu'),
        layers.Dropout(DROPOUT_RATE),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def load_and_preprocess_data(directory, classes, img_size=(IMG_SIZE, IMG_SIZE)):
    """Load images from directory and preprocess them."""
    X = []
    y = []
    
    print("Chargement et prétraitement des images...")
    for class_idx, class_name in enumerate(classes):
        class_dir = os.path.join(directory, class_name)
        if not os.path.isdir(class_dir):
            continue
            
        print(f"Traitement de la classe '{class_name}'...")
        class_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for img_file in tqdm(class_files, desc=class_name):
            img_path = os.path.join(class_dir, img_file)
            try:
                # Charger et redimensionner l'image
                img = cv2.imread(img_path)
                if img is None:
                    print(f"Impossible de charger l'image: {img_path}")
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_resized = cv2.resize(img, img_size)
                
                # Normaliser l'image
                img_normalized = img_resized / 255.0
                
                X.append(img_normalized)
                y.append(class_idx)
            except Exception as e:
                print(f"Erreur lors du traitement de l'image {img_path}: {e}")
    
    return np.array(X), np.array(y)

def train_model(X_train, y_train, X_val, y_val, num_classes, img_size=(IMG_SIZE, IMG_SIZE)):
    """Train the CNN model."""
    # Convertir les étiquettes en format catégorique
    y_train_cat = keras.utils.to_categorical(y_train, num_classes)
    y_val_cat = keras.utils.to_categorical(y_val, num_classes)
    
    # Créer le modèle
    model = create_cnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    print(model.summary())
    
    # Data augmentation pour l'entraînement
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    
    # Callbacks pour améliorer l'entraînement - utiliser les variables globales
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
    
    # Entraînement du modèle - utiliser les variables globales
    print("\nDébut de l'entraînement du modèle...")
    print(f"Configuration: epochs={EPOCHS}, batch_size={BATCH_SIZE}, learning_rate={LEARNING_RATE}")
    history = model.fit(
        datagen.flow(X_train, y_train_cat, batch_size=BATCH_SIZE),
        steps_per_epoch=len(X_train) // BATCH_SIZE,
        epochs=EPOCHS,
        validation_data=(X_val, y_val_cat),
        callbacks=callbacks
    )
    
    # Évaluation du modèle
    print("\nÉvaluation du modèle...")
    val_loss, val_acc = model.evaluate(X_val, y_val_cat)
    print(f"Précision sur l'ensemble de validation: {val_acc:.4f}")
    
    # Tracer les courbes d'apprentissage
    plot_training_history(history)
    
    return model, history

def train_with_generators(directory, classes, img_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE, validation_split=VALIDATION_SPLIT):
    """Version optimisée qui utilise le multithreading pour le chargement des données."""
    
    # Même générateur qu'avant mais avec multithreading
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=validation_split
    )
    
    # Ajouter workers pour paralléliser le chargement des images
    train_generator = train_datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    validation_generator = train_datagen.flow_from_directory(
        directory,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )
    
    # Créer le modèle
    model = create_cnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=len(classes))
    print(model.summary())
    
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
    
    print("\nDébut de l'entraînement du modèle...")
    print(f"Configuration: epochs={EPOCHS}, batch_size={batch_size}, learning_rate={LEARNING_RATE}")
    
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size,
        epochs=EPOCHS,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size,
        callbacks=callbacks
    )
    
    # Évaluation du modèle avec le générateur de validation
    print("\nÉvaluation du modèle...")
    val_loss, val_acc = model.evaluate(validation_generator)
    print(f"Précision sur l'ensemble de validation: {val_acc:.4f}")
    
    # Tracer les courbes d'apprentissage
    plot_training_history(history)
    
    return model, history

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
    plt.savefig('training_history.png')
    print("Graphique d'apprentissage sauvegardé dans 'training_history.png'")
    plt.close()

def save_model(model, classes):
    """Save the trained model and class mappings."""
    # Créer un dossier 'model' s'il n'existe pas
    os.makedirs('model', exist_ok=True)
    
    # Sauvegarder le modèle au format Keras natif
    model_path = 'model/leaffliction_model.keras'
    model.save(model_path)
    
    # Sauvegarder le mappage des classes
    class_mapping = {i: class_name for i, class_name in enumerate(classes)}
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
    
    # Permettre de remplacer les variables globales par des arguments
    parser.add_argument(
        "--img_size", 
        type=int, 
        default=IMG_SIZE, 
        help=f"Taille des images (défaut: {IMG_SIZE})"
    )
    parser.add_argument(
        "--val_split", 
        type=float, 
        default=VALIDATION_SPLIT, 
        help=f"Proportion de données pour la validation (défaut: {VALIDATION_SPLIT})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.src):
        parser.error(f"Le dossier source '{args.src}' n'existe pas.")
    
    # Extract classes from subdirectories
    classes = extract_classes(args.src)
    if not classes:
        print("Aucune classe (sous-dossier) trouvée dans le répertoire source.")
        return
    
    json_path = save_classes_to_json(classes)
    print(f"Classes disponibles ({len(classes)}):")
    for cls in classes:
        print(f"- {cls}")
    print(f"Liste des classes sauvegardée dans: {json_path}")

    # Image preprocessing settings - Utiliser l'argument qui peut remplacer la variable globale
    img_size = (args.img_size, args.img_size)
    
    # Entraîner le modèle avec des générateurs
    model, history = train_with_generators(
        args.src, 
        classes,
        img_size=img_size,
        batch_size=BATCH_SIZE,
        validation_split=args.val_split
    )
    
    # Sauvegarder le modèle
    model_path = save_model(model, classes)
    
    print(f"\nEntraînement terminé! Modèle sauvegardé dans: {model_path}")
    print("Vous pouvez maintenant utiliser ce modèle pour faire des prédictions avec Predict.py")

if __name__ == "__main__":
    main()