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
    """Renvoie l'image originale convertie en RGB."""
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if output_path:
        pcv.outputs.save_image(rgb_img, output_path)
    return rgb_img


def gaussian_blur(img, output_path=None):
    """Applique un flou gaussien à l'image et renvoie le résultat."""
    img_blur = pcv.gaussian_blur(img=img, ksize=(7, 7), sigma_x=0, sigma_y=None)
    if output_path:
        pcv.outputs.save_image(img_blur, output_path)
    return img_blur


def create_mask(img, output_path=None):
    """Crée un masque binaire et renvoie le résultat."""
    s = pcv.rgb2gray_hsv(img, 's')
    s_thresh = pcv.threshold.binary(s, threshold=90, object_type='light')
    s_mblur = pcv.median_blur(s_thresh, 5)
    struct_elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opened = cv2.morphologyEx(s_mblur, cv2.MORPH_OPEN, struct_elem)
    b = pcv.rgb2gray_lab(img, 'b')
    b_thresh = pcv.threshold.binary(b, threshold=160, object_type='light')
    combined = pcv.logical_or(opened, b_thresh)
    masked = pcv.apply_mask(img, combined, 'white')
    filled = pcv.fill(combined, 200)
    final_mask = pcv.apply_mask(masked, filled, 'white')
    if output_path:
        cv2.imwrite(output_path, final_mask)
    return final_mask


def roi_objects(img, mask, output_path=None):
    """Identifie les objets dans les régions d'intérêt et renvoie l'image annotée."""
    if len(mask.shape) != 2:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_img = img.copy()
    if contours:
        for idx, cnt in enumerate(contours):
            cv2.drawContours(roi_img, [cnt], -1, (0, 255, 0), 2)
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(roi_img, (x, y), (x + w, y + h), (255, 0, 0), 2)
            cv2.putText(roi_img, f"ROI {idx+1}", (x, max(y-10,0)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    if output_path:
        pcv.outputs.save_image(roi_img, output_path)
    return roi_img


def analyze_object(img, mask, output_path=None):
    """Analyse l'objet principal et renvoie l'image annotée."""
    if len(mask.shape) != 2:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analyze_img = img.copy()
    if contours:
        obj = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze_img, [obj], -1, (0, 255, 0), 2)
        m = cv2.moments(obj)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(analyze_img, (cx, cy), 10, (0, 0, 255), -1)
            cv2.putText(analyze_img, "Centre", (cx - 20, cy - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        topmost = tuple(obj[obj[:, :, 1].argmin()][0])
        bottommost = tuple(obj[obj[:, :, 1].argmax()][0])
        leftmost = tuple(obj[obj[:, :, 0].argmin()][0])
        rightmost = tuple(obj[obj[:, :, 0].argmax()][0])
        cv2.line(analyze_img, topmost, bottommost, (255, 0, 255), 2)
        cv2.line(analyze_img, leftmost, rightmost, (255, 0, 255), 2)
    if output_path:
        pcv.outputs.save_image(analyze_img, output_path)
    return analyze_img


def pseudolandmarks(img, mask, output_path=None):
    """Extrait des points caractéristiques et renvoie l'image annotée."""
    if len(mask.shape) != 2:
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    else:
        mask_gray = mask.copy()
    _, mask_gray = cv2.threshold(mask_gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmark_img = img.copy()
    if contours:
        obj = max(contours, key=cv2.contourArea)
        points = obj.reshape(-1, 2)
        nb_points = 20
        step = max(1, len(points) // nb_points)
        for i in range(0, len(points), step):
            x, y = points[i]
            cv2.circle(landmark_img, (x, y), 4, (0, 140, 255), -1)
    if output_path:
        pcv.outputs.save_image(landmark_img, output_path)
    return landmark_img


def color_histogram(img, output_path=None):
    """
    Affiche l'histogramme de plusieurs canaux (BGR, HSV, Lab) avec amélioration visuelle.
    """
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b2 = cv2.split(lab)
    channels = {
        'blue': b,
        'green': g,
        'red': r,
        'hue': h,
        'saturation': s,
        'value': v,
        'lightness': l,
        'green-magenta': a,
        'blue-yellow': b2
    }
    plt.figure(figsize=(8, 6))
    for name, channel in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0,256])
        hist_norm = hist.ravel() / hist.sum()
        plt.plot(hist_norm, label=name)
    plt.xlim([0,256])
    plt.ylim([0, 0.12])
    plt.xlabel("Intensité des pixels")
    plt.ylabel("Proportion normalisée")
    plt.title("Figure IV.7: Color Histogram (amélioré)")
    plt.legend()
    plt.grid(True)
    if output_path:
        plt.savefig(output_path)
        plt.close()
    else:
        plt.show()
    return None


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
        
        # Si aucune transformation n'est spécifiée, on les applique toutes
        if not transformations:
            transformations = list(all_transformations.keys())
        
        # Vérifier la validité des transformations demandées
        for t in transformations:
            if t not in all_transformations:
                print(f"Transformation inconnue: {t}")
                return False
        
        # Créer le répertoire de sortie si besoin
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]
        
        # Variable pour stocker le masque (utilisé dans certaines transformations)
        mask = None
        
        # Appliquer les transformations dans l'ordre
        for t in transformations:
            if t == 'mask':
                mask = create_mask(
                    img,
                    output_dir and os.path.join(output_dir, f"{base_filename}_mask.png")
                )
            elif t in ['roi', 'analyze', 'landmarks']:
                # Ces transformations nécessitent un masque
                if mask is None:
                    mask = create_mask(img, None)
                
                if t == 'roi':
                    roi_objects(
                        img, mask,
                        output_dir and os.path.join(output_dir, f"{base_filename}_roi.png")
                    )
                elif t == 'analyze':
                    analyze_object(
                        img, mask,
                        output_dir and os.path.join(output_dir, f"{base_filename}_analyze.png")
                    )
                elif t == 'landmarks':
                    pseudolandmarks(
                        img, mask,
                        output_dir and os.path.join(output_dir, f"{base_filename}_landmarks.png")
                    )
            else:
                # Transformations qui ne nécessitent pas de masque
                all_transformations[t](
                    img,
                    output_dir and os.path.join(output_dir, f"{base_filename}_{t}.png")
                )
        
        return True
    
    except Exception as e:
        print(f"Erreur lors du traitement de l'image {input_file}: {e}")
        return False


def process_directory(input_dir, output_dir, transformations=None):
    """Traite toutes les images d'un répertoire en parcourant récursivement."""
    if not os.path.isdir(input_dir):
        print(f"Le répertoire source {input_dir} n'existe pas.")
        return False
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    success_count = 0
    total_count = 0
    # Parcours récursif des sous-dossiers
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_count += 1
                file_path = os.path.join(root, file)
                if process_image(file_path, output_dir, transformations):
                    success_count += 1
    print(f"Traitement terminé: {success_count}/{total_count} images traitées avec succès.")
    return True


def display_combined_results(input_file):
    """
    Applique toutes les transformations à une image et les affiche dans une seule fenêtre.
    """
    img = cv2.imread(input_file)
    if img is None:
        print(f"Erreur: Impossible de charger l'image {input_file}")
        return
    # Appliquer les transformations
    orig = original_image(img, None)
    blur = gaussian_blur(img, None)
    mask = create_mask(img, None)
    roi = roi_objects(img, mask, None)
    analyze = analyze_object(img, mask, None)
    landmarks = pseudolandmarks(img, mask, None)
    
    # Calcul de l'histogramme
    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b2 = cv2.split(lab)
    
    # Création d'une figure combinée avec une grille 3x2 pour les images et l'histogramme en dessous
    fig = plt.figure(figsize=(14, 12))  # Ajustement du format pour plus de hauteur
    grid = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 0.8, 0.8])  # 3 lignes pour les images, 2 pour l'histogramme
    
    # Ajouter les images dans une grille 3x2
    ax1 = fig.add_subplot(grid[0, 0])
    ax1.imshow(orig)
    ax1.set_title("Original")
    ax1.axis('off')
    
    ax2 = fig.add_subplot(grid[0, 1])
    ax2.imshow(cv2.cvtColor(blur, cv2.COLOR_BGR2RGB))
    ax2.set_title("Gaussian Blur")
    ax2.axis('off')
    
    ax3 = fig.add_subplot(grid[0, 2])
    ax3.imshow(mask, cmap='gray')
    ax3.set_title("Mask")
    ax3.axis('off')
    
    ax4 = fig.add_subplot(grid[1, 0])
    ax4.imshow(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
    ax4.set_title("ROI")
    ax4.axis('off')
    
    ax5 = fig.add_subplot(grid[1, 1])
    ax5.imshow(cv2.cvtColor(analyze, cv2.COLOR_BGR2RGB))
    ax5.set_title("Analyze Object")
    ax5.axis('off')
    
    ax6 = fig.add_subplot(grid[1, 2])
    ax6.imshow(cv2.cvtColor(landmarks, cv2.COLOR_BGR2RGB))
    ax6.set_title("Pseudolandmarks")
    ax6.axis('off')
    
    # Ajouter l'histogramme en dessous, sur deux lignes
    ax7 = fig.add_subplot(grid[3:, :])
    for name, channel in {
        'blue': b,
        'green': g,
        'red': r,
        'hue': h,
        'saturation': s,
        'value': v,
        'lightness': l,
        'green-magenta': a,
        'blue-yellow': b2
    }.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0,256])
        hist_norm = hist.ravel() / hist.sum()
        ax7.plot(hist_norm, label=name)
    ax7.set_xlim([0, 256])
    ax7.set_ylim([0, 0.12])
    ax7.set_title("Color Histogram")
    ax7.legend(fontsize='small')
    ax7.grid(True)
    
    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Programme de transformation d'images pour l'analyse de feuilles.")
    
    # Groupe mutuellement exclusif pour l'entrée (image ou répertoire)
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
    
    # Récupérer les transformations demandées
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
    
    # Si aucune transformation n'est spécifiée, on les applique toutes
    if not transformations:
        transformations = ['original', 'blur', 'mask', 'roi', 'analyze', 'landmarks', 'histogram']
    
    # Traiter une image unique ou un répertoire
    if args.image:
        # Cas d'une seule image
        if not os.path.isfile(args.image):
            print(f"Erreur: Le fichier {args.image} n'existe pas.")
            return
        # Si aucune destination n'est précisée, afficher tous les graphes dans une seule fenêtre
        if args.dst is None:
            display_combined_results(args.image)
        else:
            process_image(args.image, args.dst, transformations)
    else:
        # Cas d'un répertoire
        if not args.dst:
            print("Erreur: Le répertoire de destination (-dst) est requis lors du traitement d'un répertoire.")
            return
        process_directory(args.src, args.dst, transformations)


if __name__ == "__main__":
    main()
