# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
from plantcv import plantcv as pcv


# ### FONCTION UTILITAIRE ###


def _create_binary_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask_green = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255))
    mask_brown = cv2.inRange(hsv, (8, 50, 20), (30, 255, 200))
    mask_yellow = cv2.inRange(hsv, (15, 50, 50), (40, 255, 255))
    combined_mask = cv2.bitwise_or(mask_green, mask_brown)
    combined_mask = cv2.bitwise_or(combined_mask, mask_yellow)
    kernel = np.ones((5, 5), np.uint8)
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_CLOSE, kernel, iterations=2
    )
    combined_mask = cv2.morphologyEx(
        combined_mask, cv2.MORPH_OPEN, kernel, iterations=1
    )
    _, bin_mask_for_fill = cv2.threshold(
        combined_mask, 1, 255, cv2.THRESH_BINARY
    )
    filled_mask = pcv.fill_holes(bin_mask_for_fill)
    return filled_mask


# ### TRANSFORMATIONS D'IMAGES ###


def gaussian_blur(img, output_path=None):
    """Applique masque, seuillage et flou gaussien."""
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = _create_binary_mask(img)
    if mask is None:
        blurred_gray = cv2.GaussianBlur(gray_img, (7, 7), 0)
        if output_path:
            cv2.imwrite(output_path, blurred_gray)
        return blurred_gray

    masked_gray = cv2.bitwise_and(gray_img, gray_img, mask=mask)
    threshold_value = 55
    max_value = 220
    ret, high_contrast_masked = cv2.threshold(
        masked_gray, threshold_value, max_value, cv2.THRESH_BINARY
    )
    img_blur_final = cv2.GaussianBlur(high_contrast_masked, (7, 7), 0)

    if output_path:
        cv2.imwrite(output_path, img_blur_final)
    return img_blur_final


def original_image(img, output_path=None):
    """Renvoie l'image originale."""
    if output_path:
        cv2.imwrite(output_path, img)
    return img


def create_masked_image(img, output_path=None):
    """Crée une image avec fond blanc."""
    mask = _create_binary_mask(img)
    masked_img_white_bg = pcv.apply_mask(
        img=img.copy(), mask=mask, mask_color="white"
    )
    if output_path:
        cv2.imwrite(output_path, masked_img_white_bg)
    return masked_img_white_bg


def roi_objects(img, mask, output_path=None):
    """Visualise masque superposé et cadre englobant."""
    roi_visualization_img = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        if output_path:
            cv2.imwrite(output_path, roi_visualization_img)
        return roi_visualization_img
    try:
        colored_masks = pcv.visualize.colorize_masks(
            masks=[mask], colors=["green"]
        )
        roi_visualization_img = pcv.visualize.overlay_two_imgs(
            img1=roi_visualization_img, img2=colored_masks, alpha=0.5
        )
    except Exception:
        green_color = np.array([0, 255, 0], dtype=np.uint8)
        green_mask_viz = np.zeros_like(roi_visualization_img)
        green_mask_viz[mask > 0] = green_color
        roi_visualization_img = cv2.addWeighted(
            roi_visualization_img, 0.7, green_mask_viz, 0.3, 0
        )

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        main_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(main_contour) > 10:
            x, y, w, h = cv2.boundingRect(main_contour)
            cv2.rectangle(
                roi_visualization_img, (x, y), (x + w, y + h), (255, 0, 0), 2
            )
    if output_path:
        cv2.imwrite(output_path, roi_visualization_img)
    return roi_visualization_img


def analyze_object(img, mask, output_path=None):
    """Analyse objet: contour et lignes extrêmes."""
    analyze_img = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        if output_path:
            cv2.imwrite(output_path, analyze_img)
        return analyze_img

    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        obj = max(contours, key=cv2.contourArea)
        cv2.drawContours(analyze_img, [obj], -1, (255, 0, 0), 2)
        if len(obj) > 0:
            try:
                topmost = tuple(obj[obj[:, :, 1].argmin()][0])
                bottommost = tuple(obj[obj[:, :, 1].argmax()][0])
                leftmost = tuple(obj[obj[:, :, 0].argmin()][0])
                rightmost = tuple(obj[obj[:, :, 0].argmax()][0])
                line_thickness = 2
                magenta_color = (255, 0, 255)
                cv2.line(
                    analyze_img,
                    topmost,
                    bottommost,
                    magenta_color,
                    line_thickness,
                )
                cv2.line(
                    analyze_img,
                    leftmost,
                    rightmost,
                    magenta_color,
                    line_thickness,
                )
            except IndexError:
                pass
    if output_path:
        cv2.imwrite(output_path, analyze_img)
    return analyze_img


def pseudolandmarks(img, mask, output_path=None):
    """Génère pseudolandmarks avec PlantCV."""
    landmark_img = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        if output_path:
            cv2.imwrite(output_path, landmark_img)
        return landmark_img

    if len(mask.shape) > 2 or mask.dtype != np.uint8:
        mask = np.clip(mask, 0, 255).astype(np.uint8)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
    _, mask_bin = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
    if cv2.countNonZero(mask_bin) == 0:
        if output_path:
            cv2.imwrite(output_path, landmark_img)
        return landmark_img

    original_debug = pcv.params.debug
    pcv.params.debug = None
    pcv.outputs.clear()

    try:
        landmark_groups = pcv.homology.x_axis_pseudolandmarks(
            img=img, mask=mask_bin
        )
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        point_radius = 3
        point_thickness = -1

        for i, group in enumerate(landmark_groups):
            if i >= len(colors):
                break
            color = colors[i]
            for point_contour in group:
                if point_contour.shape == (1, 2):
                    x, y = map(int, point_contour[0])
                    cv2.circle(
                        landmark_img,
                        (x, y),
                        point_radius,
                        color,
                        point_thickness,
                    )
    except Exception as e:
        print(f"Erreur pcv.homology.x_axis_pseudolandmarks: {e}")
    finally:
        pcv.params.debug = original_debug

    if output_path:
        cv2.imwrite(output_path, landmark_img)
    return landmark_img


def color_histogram(img, output_path=None, display_mode=False):
    """Génère l'histogramme. Retourne True si
     un plot est créé en display_mode."""
    mask_hist = _create_binary_mask(img)
    if mask_hist is None or cv2.countNonZero(mask_hist) == 0:
        return None if not display_mode else False

    if len(img.shape) < 3 or img.shape[2] != 3:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        else:
            return None if not display_mode else False

    b, g, r = cv2.split(img)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
    l, a, b2 = cv2.split(lab)

    channels_to_plot = {
        "blue": (b, "blue"),
        "green": (g, "darkgreen"),
        "red": (r, "red"),
        "hue": (h, "darkviolet"),
        "saturation": (s, "cyan"),
        "value": (v, "orange"),
        "lightness": (l, "dimgray"),
        "green-magenta": (a, "magenta"),
        "blue-yellow": (b2, "gold"),
    }

    total_pixels = cv2.countNonZero(mask_hist)
    if total_pixels == 0:
        return None if not display_mode else False

    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    fig_hist.canvas.manager.set_window_title("Figure IV.7: Histogramme")

    plot_successful = False
    for name, (channel_data, color) in channels_to_plot.items():
        hist = cv2.calcHist([channel_data], [0], mask_hist, [256], [0, 256])
        if hist is not None and total_pixels > 0:
            hist_percent = (hist / total_pixels) * 100
            ax_hist.plot(hist_percent, color=color, label=name, linewidth=1.2)
            plot_successful = True

    if not plot_successful:
        plt.close(fig_hist)
        return None if not display_mode else False

    ax_hist.set_xlim([0, 255])
    ax_hist.set_xlabel("Pixel intensity", fontsize=12)
    ax_hist.set_ylabel("Proportion of pixels (%)", fontsize=12)
    ax_hist.grid(True, color="white", linestyle="-", linewidth=0.75)
    ax_hist.set_facecolor("#EAEAEA")
    ax_hist.legend(
        title="color Channel",
        loc="center left",
        bbox_to_anchor=(1, 0.5),
        fontsize="medium",
        title_fontsize="large",
    )
    try:
        fig_hist.tight_layout(rect=[0, 0, 0.82, 1])
    except Exception:
        pass

    if display_mode:
        return True
    elif output_path:
        try:
            fig_hist.savefig(output_path, facecolor=fig_hist.get_facecolor())
            plt.close(fig_hist)
            hist_img_saved = cv2.imread(output_path)
            if hist_img_saved is None:
                return np.zeros((100, 300, 3), dtype=np.uint8)
            return hist_img_saved
        except Exception:
            plt.close(fig_hist)
            return np.zeros((100, 300, 3), dtype=np.uint8)
    else:
        plt.close(fig_hist)
        return None


# ### TRAITEMENT PRINCIPAL ###


def process_image(input_file, output_dir=None, transformations=None):
    """Traite une image. Retourne (results_dict, hist_plot_created)
    en mode affichage."""
    try:
        img = cv2.imread(input_file)
        if img is None:
            return (None, False) if output_dir is None else None

        img_orig = img.copy()

        all_transformations = {
            "original": original_image,
            "blur": gaussian_blur,
            "mask": create_masked_image,
            "roi": roi_objects,
            "analyze": analyze_object,
            "landmarks": pseudolandmarks,
            "histogram": color_histogram,
        }

        if not transformations:
            transformations = list(all_transformations.keys())
        valid_transformations = [
            t for t in transformations if t in all_transformations
        ]
        if not valid_transformations:
            return (None, False) if output_dir is None else None

        base_filename = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            base_filename = os.path.splitext(os.path.basename(input_file))[0]

        results = {}
        mask_binary = None
        histogram_created = False
        needs_binary_mask = any(
            t in ["roi", "analyze", "landmarks"] for t in valid_transformations
        )

        if needs_binary_mask:
            mask_binary = _create_binary_mask(img_orig)
            if mask_binary is None or np.count_nonzero(mask_binary) == 0:
                pass

        for t in valid_transformations:
            func = all_transformations[t]
            output_path = None
            if output_dir and t != "histogram":
                output_path = os.path.join(
                    output_dir, f"{base_filename}_{t}.png"
                )

            try:
                result_img = None
                if t == "histogram":
                    hist_output_path = None
                    if output_dir:
                        hist_output_path = os.path.join(
                            output_dir, f"{base_filename}_histogram.png"
                        )
                    hist_result = func(
                        img_orig,
                        output_path=hist_output_path,
                        display_mode=(output_dir is None),
                    )
                    if output_dir is None:
                        histogram_created = hist_result
                    else:
                        pass

                elif t in ["roi", "analyze", "landmarks"]:
                    result_img = func(img_orig, mask_binary, output_path)
                elif t in ["original", "blur", "mask"]:
                    result_img = func(img_orig, output_path)

                if (
                    result_img is not None
                    and output_dir is None
                    and t != "histogram"
                ):
                    results[t] = result_img

            except Exception as e:
                print(
                    f"Erreur application transfo '{t}' sur {input_file}: {e}"
                )

        if output_dir is None:
            return results, histogram_created
        else:
            return True

    except Exception as e:
        print(f"Erreur grave traitement image {input_file}: {e}")
        return (None, False) if output_dir is None else False


# ### TRAITEMENT PAR LOT ###


def process_directory(input_dir, output_dir, transformations=None):
    """Traite toutes les images d'un répertoire."""
    if not os.path.isdir(input_dir):
        print(f"Erreur: Répertoire source {input_dir} inexistant.")
        return False
    os.makedirs(output_dir, exist_ok=True)
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    success_count = 0
    total_count = 0
    error_files = []
    print(f"Début traitement répertoire : {input_dir} -> {output_dir}")
    print(f"Transformations : {transformations or 'Toutes'}")
    for root, dirs, files in os.walk(input_dir):
        relative_path = os.path.relpath(root, input_dir)
        current_output_dir = os.path.join(output_dir, relative_path)
        if relative_path == ".":
            current_output_dir = output_dir
        if not os.path.exists(current_output_dir):
            os.makedirs(current_output_dir, exist_ok=True)
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                total_count += 1
                file_path = os.path.join(root, file)
                print(f"-- Traitement ({total_count}) : {file_path}", end="")
                if process_image(
                    file_path, current_output_dir, transformations
                ):
                    success_count += 1
                    print(" [OK]")
                else:
                    error_files.append(file_path)
                    print(" [ERREUR]")
    print("-" * 30)
    print(
        f"Traitement terminé. {success_count}/{total_count} images traitées."
    )
    if error_files:
        print("Erreurs signalées pour:", *[f"\n  - {f}" for f in error_files])
        print("-" * 30)
    return len(error_files) == 0


# ### AFFICHAGE DES RÉSULTATS ###


def display_combined_results(results_dict):
    """Affiche les images de la grille dans une fenêtre. Retourne True si
    la grille est créée."""
    if not results_dict:
        return False

    titles_map = {
        "original": "Fig IV.1: Original",
        "blur": "Fig IV.2: Flou Gaussien",
        "mask": "Fig IV.3: Masque",
        "roi": "Fig IV.4: Objets Roi",
        "analyze": "Fig IV.5: Analyse Objet",
        "landmarks": "Fig IV.6: Pseudo-repères",
    }

    images_to_display = {}
    titles_to_display = {}
    cmaps = {}
    grid_keys = ["original", "blur", "mask", "roi", "analyze", "landmarks"]

    for key in grid_keys:
        if key in results_dict:
            img = results_dict[key]
            title = titles_map.get(key, key.capitalize())
            if img is None:
                continue

            display_img = None
            cmap = None
            if len(img.shape) == 2:
                display_img = img
                cmap = "gray"
            elif len(img.shape) == 3:
                if img.shape[2] == 3:
                    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                elif img.shape[2] == 4:
                    display_img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
                else:
                    display_img = img
            else:
                continue

            images_to_display[key] = display_img
            titles_to_display[key] = title
            cmaps[key] = cmap

    if not images_to_display:
        return False

    num_images = len(images_to_display)
    ncols = 3
    nrows = (num_images + ncols - 1) // ncols
    fig_grid, axes = plt.subplots(nrows, ncols, figsize=(15, 5 * nrows))
    fig_grid.canvas.manager.set_window_title("Transformations Images")
    axes = axes.ravel()

    img_idx = 0
    display_order = grid_keys
    for key in display_order:
        if key in images_to_display:
            ax = axes[img_idx]
            ax.imshow(images_to_display[key], cmap=cmaps[key])
            ax.set_title(titles_to_display[key])
            ax.axis("off")
            img_idx += 1

    for i in range(img_idx, len(axes)):
        axes[i].axis("off")

    try:
        fig_grid.tight_layout()
    except Exception:
        pass

    return True


# ### POINT D'ENTRÉE PRINCIPAL ###


def main():
    parser = argparse.ArgumentParser(
        description="Programme léger de transformation d'images pour feuilles."
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "image", nargs="?", help="Chemin image unique.", default=None
    )
    input_group.add_argument("-src", help="Répertoire source (récursif).")
    parser.add_argument(
        "-dst",
        help="Répertoire destination. Si omis avec image unique, affiche.",
    )
    parser.add_argument(
        "--original", action="store_true", help="Inclure Fig IV.1"
    )
    parser.add_argument("--blur",
                        action="store_true",
                        help="Inclure Fig IV.2")
    parser.add_argument("--mask",
                        action="store_true",
                        help="Inclure Fig IV.3")
    parser.add_argument("--roi",
                        action="store_true",
                        help="Inclure Fig IV.4")
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Inclure Fig IV.5"
    )
    parser.add_argument(
        "--landmarks",
        action="store_true",
        help="Inclure Fig IV.6"
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Inclure Fig IV.7"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Toutes transformations (défaut si rien spécifié).",
    )
    args = parser.parse_args()

    transformations = []
    any_transformation_flag = (
        args.original
        or args.blur
        or args.mask
        or args.roi
        or args.analyze
        or args.landmarks
        or args.histogram
    )
    default_order = [
        "original",
        "blur",
        "mask",
        "roi",
        "analyze",
        "landmarks",
        "histogram",
    ]

    if args.all or not any_transformation_flag:
        transformations = default_order
    else:
        if args.histogram:
            transformations.append("histogram")
        for key in default_order[:-1]:
            if getattr(args, key, False):
                transformations.append(key)
        if "histogram" in transformations:
            transformations.remove("histogram")
            transformations.append("histogram")

    if args.image:
        if not os.path.isfile(args.image):
            print(f"Erreur: Fichier '{args.image}' inexistant.")
            return
        if args.dst is None:
            print("Mode affichage.")
            results_dict, histogram_created = process_image(
                args.image, output_dir=None, transformations=transformations
            )

            grid_created = False
            if results_dict:
                grid_created = display_combined_results(results_dict)

            if grid_created or histogram_created:
                print(
                    "Affichage des fenêtres. Fermez toutes les fenêtres"
                    " pour quitter."
                )
                plt.show()
            else:
                print(
                    "Aucun résultat (ni grille, ni histogramme) n'a été "
                    "généré pour l'affichage."
                )

        else:
            print(f"Mode sauvegarde -> '{args.dst}'.")
            process_image(args.image, args.dst, transformations)

    elif args.src:
        if not args.dst:
            print("Erreur: -dst requis pour -src.")
            return
        print(f"Mode traitement répertoire : '{args.src}' -> '{args.dst}'.")
        if "histogram" not in transformations and (args.histogram or args.all):
            transformations.append("histogram")
        process_directory(args.src, args.dst, transformations)
    else:
        print("Erreur: Spécifier une image ou -src.")
        parser.print_help()


if __name__ == "__main__":
    main()
