import os
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def show_or_save(img, title, filename=None):
    """
    Affiche une image ou la sauvegarde selon les paramètres.
    """
    if filename:
        pcv.outputs.save_image(img, filename)
    else:
        plt.figure()
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()


def original_image(img, output_path=None):
    """Affiche l'image originale."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if output_path:
        pcv.outputs.save_image(rgb_img, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(rgb_img)
        plt.title("Figure IV.1: Original")
        plt.axis('off')
        plt.show()
    return rgb_img


def gaussian_blur(img, output_path=None):
    img_blur = pcv.gaussian_blur(img=img, ksize=(5, 5), sigma_x=0, sigma_y=None)
    
    if output_path:
        pcv.outputs.save_image(img_blur, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(img_blur, cv2.COLOR_BGR2RGB))
        plt.title("Figure IV.2: Gaussian blur")
        plt.axis('off')
        plt.show()
    return img_blur


def create_mask(img, output_path=None):
    """Crée un masque à partir de l'image en utilisant PlantCV."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    s = pcv.rgb2gray_lab(rgb_img=rgb_img, channel='a')
    s_thresh = pcv.threshold.binary(gray_img=s, threshold=100, object_type='light')
    if s_thresh.max() <= 1:
        s_thresh = (s_thresh * 255).astype(np.uint8)
    else:
        s_thresh = s_thresh.astype(np.uint8)
    s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
    s_fill = pcv.fill(bin_img=s_mblur, size=200)
    mask = pcv.fill_holes(bin_img=s_fill)
    
    if np.sum(mask) < mask.size * 0.01:
        s = pcv.rgb2gray_lab(rgb_img=rgb_img, channel='b')
        s_thresh = pcv.threshold.binary(gray_img=s, threshold=100, object_type='dark')
        if s_thresh.max() <= 1:
            s_thresh = (s_thresh * 255).astype(np.uint8)
        else:
            s_thresh = s_thresh.astype(np.uint8)
        s_mblur = pcv.median_blur(gray_img=s_thresh, ksize=5)
        s_fill = pcv.fill(bin_img=s_mblur, size=200)
        mask = pcv.fill_holes(bin_img=s_fill)

    # S'assurer que le masque final est 0 ou 255
    if mask.max() <= 1:
        mask = (mask * 255).astype(np.uint8)
    else:
        mask = mask.astype(np.uint8)

    if output_path:
        pcv.outputs.save_image(mask, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.title("Figure IV.3: Mask")
        plt.axis('off')
        plt.show()
    return mask


def roi_objects(img, mask, output_path=None):
    """Identifie les objets dans les régions d'intérêt en utilisant OpenCV."""
    # Remplacer pcv.threshold.find_objects par cv2.findContours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_img = img.copy()
    if contours:
        for i, cnt in enumerate(contours):
            cv2.drawContours(roi_img, contours, i, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
    if output_path:
        pcv.outputs.save_image(roi_img, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
        plt.title("Figure IV.4: ROI objects")
        plt.axis('off')
        plt.show()
    return roi_img


def analyze_object(img, mask, output_path=None):
    """Analyse les caractéristiques des objets avec OpenCV."""
    # Remplacer pcv.threshold.find_objects par cv2.findContours
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analyze_img = img.copy()
    if contours:
        obj = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze_img, [obj], -1, (0, 255, 0), 3)
        m = cv2.moments(obj)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(analyze_img, (cx, cy), 10, (255, 0, 0), -1)
    if output_path:
        pcv.outputs.save_image(analyze_img, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(analyze_img, cv2.COLOR_BGR2RGB))
        plt.title("Figure IV.5: Analyze object")
        plt.axis('off')
        plt.show()
    return analyze_img


def pseudolandmarks(img, mask, output_path=None):
    """Extraire des points caractéristiques (pseudolandmarks) avec OpenCV."""
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmark_img = img.copy()
    if contours:
        obj = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(obj)
        # Trace un rectangle à la place de landmark_reference_pt_dist
        cv2.rectangle(landmark_img, (x, y), (x+w, y+h), (255, 255, 0), 2)
    if output_path:
        pcv.outputs.save_image(landmark_img, output_path)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(landmark_img, cv2.COLOR_BGR2RGB))
        plt.title("Figure IV.6: Pseudolandmarks")
        plt.axis('off')
        plt.show()
    return landmark_img


def color_histogram(img, output_path=None):
    """Affiche l'histogramme de couleurs de l'image en utilisant PlantCV."""
    # Créer un histogramme de l'image
    hist_fig = pcv.visualize.colorspaces(rgb_img=cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    
    # Convertir la figure matplotlib en une image
    if output_path:
        hist_fig.savefig(output_path)
        plt.close(hist_fig)
    else:
        plt.figure(figsize=(10, 6))
        plt.title("Figure IV.7: Color histogram")
        plt.show()
    
    return hist_fig


def process_image(input_file, output_dir=None, transformations=None):
    """Traite une image avec les transformations sélectionnées."""
    try:
        # Charger l'image
        img = cv2.imread(input_file)
        if img is None:
            print(f"Erreur: Impossible de charger l'image {input_file}")
            return False
        
        # Liste de toutes les transformations disponibles
        all_transformations = {
            'original': original_image,
            'blur': gaussian_blur,
            'mask': create_mask,
            'roi': roi_objects,
            'analyze': analyze_object,
            'landmarks': pseudolandmarks,
            'histogram': color_histogram
        }
        
        # Si aucune transformation n'est spécifiée, les appliquer toutes
        if not transformations:
            transformations = list(all_transformations.keys())
        
        # Vérifier les transformations demandées
        for t in transformations:
            if t not in all_transformations:
                print(f"Transformation inconnue: {t}")
                return False
        
        # Créer le répertoire de sortie si nécessaire
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Appliquer les transformations
        mask = None  # Pour stocker le masque qui sera utilisé par d'autres transformations
        
        for t in transformations:
            if t == 'mask':
                mask = create_mask(img, output_dir and os.path.join(output_dir, f"{base_filename}_mask.png"))
            elif t == 'roi' or t == 'analyze' or t == 'landmarks':
                # Ces transformations nécessitent un masque
                if mask is None:
                    mask = create_mask(img, None)
                
                if t == 'roi':
                    roi_objects(img, mask, output_dir and os.path.join(output_dir, f"{base_filename}_roi.png"))
                elif t == 'analyze':
                    analyze_object(img, mask, output_dir and os.path.join(output_dir, f"{base_filename}_analyze.png"))
                elif t == 'landmarks':
                    pseudolandmarks(img, mask, output_dir and os.path.join(output_dir, f"{base_filename}_landmarks.png"))
            else:
                # Autres transformations qui ne nécessitent pas de masque
                all_transformations[t](img, output_dir and os.path.join(output_dir, f"{base_filename}_{t}.png"))
        
        return True
    
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {input_file}: {e}")
        return False


def process_directory(input_dir, output_dir, transformations=None):
    """Traite toutes les images d'un répertoire."""
    if not os.path.isdir(input_dir):
        print(f"Le répertoire source {input_dir} n'existe pas.")
        return False
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Liste des extensions d'image courants
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
    # Parcourir tous les fichiers du répertoire
    success_count = 0
    total_count = 0
    
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path) and any(file.lower().endswith(ext) for ext in image_extensions):
            total_count += 1
            if process_image(file_path, output_dir, transformations):
                success_count += 1
    
    print(f"Traitement terminé: {success_count}/{total_count} images traitées avec succès.")
    return True


def main():
    parser = argparse.ArgumentParser(description="Programme de transformation d'images pour l'analyse de feuilles.")
    
    # Ajouter un groupe d'arguments mutuellement exclusifs pour les entrées
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('image', nargs='?', help="Chemin vers l'image à traiter", default=None)
    input_group.add_argument('-src', help="Répertoire source contenant les images à traiter")
    
    # Arguments pour spécifier les transformations et le répertoire de destination
    parser.add_argument('-dst', help="Répertoire de destination pour sauvegarder les transformations")
    parser.add_argument('-original', action='store_true', help="Afficher/sauvegarder l'image originale")
    parser.add_argument('-blur', action='store_true', help="Appliquer un flou gaussien")
    parser.add_argument('-mask', action='store_true', help="Créer un masque de l'image")
    parser.add_argument('-roi', action='store_true', help="Identifier les objets dans les régions d'intérêt")
    parser.add_argument('-analyze', action='store_true', help="Analyser les caractéristiques des objets")
    parser.add_argument('-landmarks', action='store_true', help="Extraire des points caractéristiques")
    parser.add_argument('-histogram', action='store_true', help="Afficher l'histogramme de couleurs")
    
    args = parser.parse_args()
    
    # Déterminer les transformations demandées
    transformations = []
    if args.original:
        transformations.append('original')
    if args.blur:
        transformations.append('blur')
    if args.mask:
        transformations.append('mask')
    if args.roi:
        transformations.append('roi')
    if args.analyze:
        transformations.append('analyze')
    if args.landmarks:
        transformations.append('landmarks')
    if args.histogram:
        transformations.append('histogram')
    
    # Si aucune transformation n'est spécifiée, les appliquer toutes
    if not transformations:
        transformations = ['original', 'blur', 'mask', 'roi', 'analyze', 'landmarks', 'histogram']
    
    # Traiter une image unique ou un répertoire
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Erreur: Le fichier {args.image} n'existe pas.")
            return
        process_image(args.image, args.dst, transformations)
    else:
        if not args.dst:
            print("Erreur: Le répertoire de destination (-dst) est requis lors du traitement d'un répertoire.")
            return
        process_directory(args.src, args.dst, transformations)


if __name__ == "__main__":
    main()