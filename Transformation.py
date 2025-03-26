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
    """Applique un flou gaussien à l'image."""
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
    """
    Crée un masque binaire à partir de l'image, en exploitant la luminosité (Value) en HSV.
    Puis applique des opérations morphologiques (érosion/dilatation) pour améliorer le masque.
    """
    # 1) Convertir l'image BGR -> HSV
    # img_blur = pcv.gaussian_blur(img=img, ksize=(12, 12), sigma_x=0, sigma_y=None)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    

    _, mask = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 3) Opérations morphologiques
    #    - On utilise un élément structurant en forme d'ellipse de taille 11x11 puis 3x3
    kernel_ellipse_11 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel_ellipse_11, iterations=1)
    mask = cv2.dilate(mask, kernel_ellipse_11, iterations=1)
    
    kernel_ellipse_3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, kernel_ellipse_3, iterations=1)
    mask = cv2.dilate(mask, kernel_ellipse_3, iterations=1)
    
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    
    # 4) Sauvegarde ou affichage
    if output_path:
        cv2.imwrite(output_path, mask)
    else:
        plt.figure(figsize=(8, 6))
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.title("Figure IV.3: Mask")
        plt.show()
    
    return mask




def roi_objects(img, mask, output_path=None):
    """
    Identifie les objets dans les régions d'intérêt.
    - Contours en vert
    - Rectangle englobant (bounding box) en bleu
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    roi_img = img.copy()

    if contours:
        for i, cnt in enumerate(contours):
            # Dessiner le contour en vert (BGR = (0,255,0))
            cv2.drawContours(roi_img, [cnt], -1, (0, 255, 0), 2)
            # Dessiner la bounding box en bleu (BGR = (255,0,0))
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
    """
    Analyse l'objet principal.
    - Dessine le contour en vert
    - Calcule et dessine le centroid en bleu
    - Trace deux lignes diagonales (haut-bas et gauche-droite) en rose (BGR=(255,0,255))
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    analyze_img = img.copy()

    if contours:
        # On prend le plus grand contour (la feuille)
        obj = max(contours, key=cv2.contourArea)
        # Contour en vert
        cv2.drawContours(analyze_img, [obj], -1, (0, 255, 0), 2)

        # Centroid
        m = cv2.moments(obj)
        if m["m00"] != 0:
            cx = int(m["m10"] / m["m00"])
            cy = int(m["m01"] / m["m00"])
            cv2.circle(analyze_img, (cx, cy), 10, (255, 0, 0), -1)  # bleu

        # Points extrêmes
        topmost = tuple(obj[obj[:, :, 1].argmin()][0])    # y minimal
        bottommost = tuple(obj[obj[:, :, 1].argmax()][0]) # y maximal
        leftmost = tuple(obj[obj[:, :, 0].argmin()][0])   # x minimal
        rightmost = tuple(obj[obj[:, :, 0].argmax()][0])  # x maximal

        # Lignes diagonales en rose
        cv2.line(analyze_img, topmost, bottommost, (255, 0, 255), 2)
        cv2.line(analyze_img, leftmost, rightmost, (255, 0, 255), 2)

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
    """
    Extrait des points caractéristiques (pseudolandmarks) sur le contour.
    - On échantillonne le contour de la feuille et on dessine des points (cercle orange).
    """
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    landmark_img = img.copy()

    if contours:
        obj = max(contours, key=cv2.contourArea)
        # On transforme le contour en liste de (x, y)
        points = obj.reshape(-1, 2)

        # Nombre de points à afficher
        nb_points = 20
        step = max(1, len(points) // nb_points)

        for i in range(0, len(points), step):
            x, y = points[i]
            # Cercle orange (BGR=(0,165,255))
            cv2.circle(landmark_img, (x, y), 3, (0, 165, 255), -1)

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
    """
    Affiche l'histogramme de plusieurs canaux (BGR, HSV, Lab).
    Les histogrammes sont normalisés et superposés sur une seule figure.
    """
    # Séparation BGR
    b, g, r = cv2.split(img)

    # Convertir en HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # Convertir en Lab
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b2 = cv2.split(lab)

    # Canaux à tracer
    channels = {
        'blue': b,
        'green': g,
        'red': r,
        'hue': h,
        'saturation': s,
        'value': v,
        'lightness': l,
        'green-magenta': a,   # canal a de Lab
        'blue-yellow': b2     # canal b de Lab
    }

    plt.figure(figsize=(8, 6))
    for name, channel in channels.items():
        hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
        hist_norm = hist.ravel() / hist.sum()
        plt.plot(hist_norm, label=name)

    plt.xlim([0, 256])
    plt.ylim([0, 0.1])  # Ajuster si besoin
    plt.xlabel("Pixel intensity")
    plt.ylabel("Proportion of Pixels (%)")
    plt.title("Figure IV.7: Color histogram")
    plt.legend()

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
    """Traite toutes les images d'un répertoire."""
    if not os.path.isdir(input_dir):
        print(f"Le répertoire source {input_dir} n'existe pas.")
        return False
    
    # Créer le répertoire de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Extensions d'image courantes
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
    
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
        process_image(args.image, args.dst, transformations)
    else:
        # Cas d'un répertoire
        if not args.dst:
            print("Erreur: Le répertoire de destination (-dst) est requis lors du traitement d'un répertoire.")
            return
        process_directory(args.src, args.dst, transformations)


if __name__ == "__main__":
    main()
