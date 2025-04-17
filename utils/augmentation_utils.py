import os
import shutil
from PIL import Image


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
        resample=Image.BICUBIC,
    )


def shear(img):
    shear_factor = 0.3
    w, h = img.size
    return img.transform(
        (w, h),
        Image.AFFINE,
        (1, 0, 0, shear_factor, 1, -shear_factor * w / 2),
        resample=Image.BICUBIC,
    )


def crop(img):
    w, h = img.size
    margin_w, margin_h = int(w * 0.1), int(h * 0.1)
    return img.crop((margin_w, margin_h, w - margin_w, h - margin_h))


def distortion(img):
    w, h = img.size
    offset_w, offset_h = int(w * 0.1), int(h * 0.1)
    quad = (
        offset_w,
        offset_h,
        w - int(w * 0.05),
        offset_h,
        w - offset_w,
        h - offset_h,
        int(w * 0.05),
        h - offset_h,
    )
    return img.transform((w, h), Image.QUAD, quad, resample=Image.BICUBIC)


def get_augmentations():
    return [
        ("Flip", flip),
        ("Rotate", rotate),
        ("Skew", skew),
        ("Shear", shear),
        ("Crop", crop),
        ("Distortion", distortion),
    ]


def augment_dataset_balanced(input_directory, output_directory):
    """
    Équilibre les classes en ajoutant des images augmentées jusqu'à ce
     que chaque classe
    ait le même nombre d'images (basé sur la classe la plus nombreuse).
    """
    if not os.path.isdir(input_directory):
        print(f"Le répertoire d'entrée {input_directory} n'existe pas.")
        return False

    os.makedirs(output_directory, exist_ok=True)
    categories = {}

    # Listing des catégories et de leurs images
    for category in os.listdir(input_directory):
        category_path = os.path.join(input_directory, category)
        if os.path.isdir(category_path):
            images = [
                f
                for f in os.listdir(category_path)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ]
            if images:
                categories[category] = images

    if not categories:
        print("Aucune catégorie trouvée dans le dataset.")
        return False

    # Déterminer la cible : le nombre maximum d'images originales parmi
    # les catégories
    target = max(len(imgs) for imgs in categories.values())
    augmentations = get_augmentations()
    print(f"Nombre d'images cible par catégorie : {target}")

    # Pour chaque catégorie, copier les images originales puis compléter
    # jusqu'à atteindre le nombre cible
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
        print(
            f"Catégorie '{category}': {current_count} images originales "
            f"copiées, besoin de {needed} images augmentées."
        )

        img_index = 0
        aug_index = 0
        while needed > 0:
            img_name = images[img_index % len(images)]
            img_path = os.path.join(category_input_path, img_name)
            try:
                with Image.open(img_path) as img:
                    aug_name, func = augmentations[
                        aug_index % len(augmentations)
                    ]
                    aug_img = func(img)
                    base_name, ext = os.path.splitext(img_name)
                    output_file = os.path.join(
                        category_output_path,
                        f"{base_name}_{aug_name}_{aug_index
                                                  // len(augmentations)}{ext}",
                    )
                    aug_img.save(output_file)
                    needed -= 1
                    aug_index += 1
            except Exception as e:
                print(f"Erreur lors du traitement de {img_path} : {e}")
            img_index += 1
        print(f"Catégorie '{category}' équilibrée à {target} images.")

    print("Augmentation équilibrée terminée.")
    return True
