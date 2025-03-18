import os
import argparse
from PIL import Image, ImageEnhance

def augment_image(image, augmentations):
    #Applique l'ensemble des transformations définies et retourne un dictionnaire
    #regroupant le nom de la transformation et l'image augmentée.
    augmented_images = {}
    for name, func in augmentations:
        try:
            augmented_images[name] = func(image)
        except Exception as e:
            print(f"Erreur lors de l'augmentation {name}: {e}")
    return augmented_images

def flip(img):
    """Retourne l'image retournée horizontalement."""
    return img.transpose(Image.FLIP_LEFT_RIGHT)

def rotate(img):
    """Retourne l'image pivotée de 90°."""
    return img.rotate(90, expand=True)

def skew(img):
    """Applique une transformation affine pour skewer l'image sur l'axe X."""
    skew_factor = 0.3
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, skew_factor, -skew_factor * h / 2, 0, 1, 0),
        resample=Image.BICUBIC
    )

def shear(img):
    """Applique une transformation affine pour shearrer l'image sur l'axe Y."""
    shear_factor = 0.3
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, shear_factor, 1, -shear_factor * w / 2),
        resample=Image.BICUBIC
    )

def crop(img):
    """Retourne l'image recadrée en retirant 10 % des bords."""
    w, h = img.size
    margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    return img.crop((margin_w, margin_h, w - margin_w, h - margin_h))

def distortion(img):
    """Applique une légère distorsion de l'image à l'aide d'une transformation QUAD."""
    w, h = img.size
    # Définir des décalages pour chaque coin (10% de la taille de l'image)
    offset_w, offset_h = int(w * 0.1), int(h * 0.1)
    # Coordonnées cibles pour chaque coin
    quad = (
        offset_w, offset_h,
        w - int(w * 0.05), offset_h,
        w - offset_w, h - offset_h,
        int(w * 0.05), h - offset_h
    )
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)

def get_augmentations():
    """
    Retourne la liste des tuples (nom, fonction) définissant les augmentations à appliquer.
    Noms conformes à l'exemple du sujet.
    """
    return [
        ("Flip", flip),
        ("Rotate", rotate),
        ("Skew", skew),
        ("Shear", shear),
        ("Crop", crop),
        ("Distortion", distortion)
    ]

def augment_dataset(input_directory, output_directory):
    if not os.path.isdir(input_directory):
        print(f"Le répertoire d'entrée {input_directory} n'existe pas.")
        return

    # Création du répertoire de sortie s'il n'existe pas déjà
    os.makedirs(output_directory, exist_ok=True)

    augmentations = get_augmentations()
    total_augmented = 0

    # Pour chaque catégorie (sous-dossier) du dataset
    for category in os.listdir(input_directory):
        category_input_path = os.path.join(input_directory, category)
        category_output_path = os.path.join(output_directory, category)
        if os.path.isdir(category_input_path):
            os.makedirs(category_output_path, exist_ok=True)
            # On récupère les images (extensions jpg/jpeg/png)
            images = [f for f in os.listdir(category_input_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            for img_name in images:
                img_path = os.path.join(category_input_path, img_name)
                try:
                    with Image.open(img_path) as img:
                        # Appliquer toutes les augmentations définies
                        augment_results = augment_image(img, augmentations)
                        base_name, ext = os.path.splitext(img_name)
                        # Optionnel : sauvegarder éventuellement l'image originale
                        # img.save(os.path.join(category_output_path, f"{base_name}_original{ext}"))
                        for aug_name, aug_img in augment_results.items():
                            output_file = os.path.join(category_output_path, f"{base_name}_{aug_name}{ext}")
                            aug_img.save(output_file)
                            total_augmented += 1
                except Exception as e:
                    print(f"Erreur lors du traitement de l'image {img_path}: {e}")
    print(f"Augmentation terminée. Total d'images générées : {total_augmented}")

def augment_image_file(input_file, output_directory=None):
    if not os.path.isfile(input_file):
        print(f"Le fichier d'entrée {input_file} n'existe pas.")
        return

    # Si aucun répertoire de sortie n'est défini, on utilise le dossier de l'image source
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Applique plusieurs transformations sur une image unique."
    )
    parser.add_argument(
        "input_file",
        type=str,
        help="Chemin vers l'image à augmenter"
    )
    parser.add_argument(
        "--output_directory",
        type=str,
        default=None,
        help="(Optionnel) Dossier où enregistrer les images augmentées"
    )
    args = parser.parse_args()
    augment_image_file(args.input_file, args.output_directory)