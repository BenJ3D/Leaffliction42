import os
import argparse
from PIL import Image
import shutil

def augment_image(image, augmentations):
    # Applique l'ensemble des transformations définies et retourne un dictionnaire
    # regroupant le nom de la transformation et l'image augmentée.
    augmented_images = {}
    for name, func in augmentations:
        try:
            augmented_images[name] = func(image)
        except Exception as e:
            print(f"Erreur lors de l'augmentation {name}: {e}")
    return augmented_images

def flip(img):
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(img):
    return img.rotate(90, expand=True)

def skew(img):
    skew_factor = 0.3
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, skew_factor, -skew_factor * h / 2, 0, 1, 0),
        resample=Image.BICUBIC
    )

def shear(img):
    shear_factor = 0.3
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, shear_factor, 1, -shear_factor * w / 2),
        resample=Image.BICUBIC
    )

def crop(img):
    w, h = img.size
    margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    return img.crop((margin_w, margin_h, w - margin_w, h - margin_h))

def distortion(img):
    w, h = img.size
    offset_w, offset_h = int(w * 0.1), int(h * 0.1)
    quad = (
        offset_w, offset_h,
        w - int(w * 0.05), offset_h,
        w - offset_w, h - offset_h,
        int(w * 0.05), h - offset_h
    )
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)

def get_augmentations():
    return [
        ("Flip", flip),
        ("Rotate", rotate),
        ("Skew", skew),
        ("Shear", shear),
        ("Crop", crop),
        ("Distortion", distortion)
    ]

def augment_image_file(input_file, output_directory=None):
    if not os.path.isfile(input_file):
        print(f"Le fichier d'entrée {input_file} n'existe pas.")
        return

    if output_directory is None:
        output_directory = os.path.dirname(input_file)
    else:
        os.makedirs(output_directory, exist_ok=True)

    try:
        with Image.open(input_file) as img:
            augmentations = get_augmentations()
            base_name, ext = os.path.splitext(os.path.basename(input_file))
            total_augmented = 0
            for aug_name, func in augmentations:
                try:
                    aug_img = func(img)
                    output_file = os.path.join(output_directory, f"{base_name}_{aug_name}{ext}")
                    aug_img.save(output_file)
                    total_augmented += 1
                except Exception as e:
                    print(f"Erreur lors de l'augmentation {aug_name} : {e}")
            print(f"Augmentation terminée. Total d'images générées : {total_augmented}")
    except Exception as e:
        print(f"Erreur lors de l'ouverture de l'image {input_file} : {e}")

def augment_dataset(input_directory, output_directory):
    if not os.path.isdir(input_directory):
        print(f"Le répertoire d'entrée {input_directory} n'existe pas.")
        return

    os.makedirs(output_directory, exist_ok=True)
    augmentations = get_augmentations()
    total_augmented = 0

    # Pour chaque catégorie (sous-dossier) du dataset
    for category in os.listdir(input_directory):
        category_input_path = os.path.join(input_directory, category)
        category_output_path = os.path.join(output_directory, category)
        if os.path.isdir(category_input_path):
            os.makedirs(category_output_path, exist_ok=True)
            images = [f for f in os.listdir(category_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in images:
                img_path = os.path.join(category_input_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        augment_results = augment_image(img, augmentations)
                        base_name, ext = os.path.splitext(img_name)
                        for aug_name, aug_img in augment_results.items():
                            output_file = os.path.join(category_output_path, f"{base_name}_{aug_name}{ext}")
                            aug_img.save(output_file)
                            total_augmented += 1
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {img_path}: {e}")
    print(f"Augmentation terminée. Total d'images générées : {total_augmented}")

def augment_dataset_balanced(input_directory, output_directory):
    # Traite l'ensemble du dataset par catégories et augmente les images de chaque catégorie de sorte
    # que chaque catégorie atteigne un nombre d'images égal au maximum trouvé parmi les catégories.
    # Les images sont générées en appliquant les augmentations de manière cyclique.

    if not os.path.isdir(input_directory):
        print(f"Le répertoire d'entrée {input_directory} n'existe pas.")
        return

    os.makedirs(output_directory, exist_ok=True)
    categories = {}

    # Listing des catégories et de leurs images
    for category in os.listdir(input_directory):
        category_path = os.path.join(input_directory, category)
        if os.path.isdir(category_path):
            images = [f for f in os.listdir(category_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                categories[category] = images

    if not categories:
        print("Aucune catégorie trouvée dans le dataset.")
        return

    # Déterminer la cible : le nombre maximum d'images originales parmi les catégories
    target = max(len(imgs) for imgs in categories.values())
    augmentations = get_augmentations()
    print(f"Nombre d'images cible par catégorie : {target}")

    # Pour chaque catégorie, copier les images originales puis compléter jusqu'à atteindre le nombre cible
    for category, images in categories.items():
        category_input_path = os.path.join(input_directory, category)
        category_output_path = os.path.join(output_directory, category)
        os.makedirs(category_output_path, exist_ok=True)
        
        # Copier toutes les images originales dans le dossier final
        for img_name in images:
            src = os.path.join(category_input_path, img_name)
            dst = os.path.join(category_output_path, img_name)
            try:
                shutil.copy2(src, dst)
            except Exception as e:
                print(f"Erreur lors de la copie de {src}: {e}")
                
        current_count = len(images)
        needed = target - current_count
        print(f"Catégorie '{category}': {current_count} images originales copiées, besoin de {needed} images augmentées.")

        img_index = 0
        aug_index = 0
        while needed > 0:
            img_name = images[img_index % len(images)]
            img_path = os.path.join(category_input_path, img_name)
            try:
                with Image.open(img_path) as img:
                    aug_name, func = augmentations[aug_index % len(augmentations)]
                    aug_img = func(img)
                    base_name, ext = os.path.splitext(img_name)
                    output_file = os.path.join(category_output_path, f"{base_name}_{aug_name}_{aug_index // len(augmentations)}{ext}")
                    aug_img.save(output_file)
                    needed -= 1
                    aug_index += 1
            except Exception as e:
                print(f"Erreur lors du traitement de {img_path} : {e}")
            img_index += 1
        print(f"Catégorie '{category}' équilibrée à {target} images.")

    print("Augmentation équilibrée terminée.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applique plusieurs transformations sur une image unique ou sur l'ensemble d'un dataset."
    )
    parser.add_argument(
        "input_path",
        type=str,
        help="Chemin vers l'image à augmenter OU vers le répertoire contenant les catégories d'images si '--all' est spécifié."
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help="(Optionnel) Dossier où enregistrer les images augmentées"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Si précisé, traite l'ensemble du dataset (input_path doit alors être un répertoire) et équilibre les catégories."
    )
    args = parser.parse_args()

    if args.all:
        # En mode dataset, input_path doit être un répertoire
        if not os.path.isdir(args.input_path):
            print("Erreur : en mode '--all', le chemin d'entrée doit être un répertoire.")
        else:
            # Si aucun dossier de sortie n'est défini, on crée un dossier 'augmented' à côté du répertoire source
            output_directory = args.out_dir or os.path.join(os.path.dirname(args.input_path), "augmented")
            augment_dataset_balanced(args.input_path, output_directory)
    else:
        # Sinon, traiter le fichier unique
        if os.path.isdir(args.input_path):
            print("Erreur : pour traiter un seul fichier, veuillez spécifier directement le chemin de l'image ou utiliser '--all' pour un dataset.")
        else:
            output_directory = args.out_dir
            augment_image_file(args.input_path, output_directory)