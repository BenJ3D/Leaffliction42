# -*- coding: utf-8 -*-
import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt
from plantcv import plantcv as pcv


def get_leaf_mask(img):
    """Create a binary mask of leaf regions using HSV thresholds."""
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # HSV ranges for green, brown, and yellow
    ranges = [
        ((30, 40, 40), (90, 255, 255)),
        ((8, 50, 20), (30, 255, 200)),
        ((15, 50, 50), (40, 255, 255)),
    ]
    mask = None
    for low, high in ranges:
        part = cv2.inRange(hsv, np.array(low), np.array(high))
        mask = part if mask is None else cv2.bitwise_or(mask, part)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = (mask > 0).astype(np.uint8) * 255
    filled = pcv.fill_holes(mask)
    return filled


def blur_image(img, mask=None):
    """Convert to grayscale, apply mask and threshold, then blur."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if mask is not None and cv2.countNonZero(mask) > 0:
        gray = cv2.bitwise_and(gray, gray, mask=mask)
        _, gray = cv2.threshold(gray, 55, 220, cv2.THRESH_BINARY)
    return cv2.GaussianBlur(gray, (7, 7), 0)


def original_image(img):
    """Return the original image."""
    return img


def masked_image(img, mask):
    """Apply a white background outside the mask."""
    return pcv.apply_mask(img, mask, mask_color="white")


def draw_roi(img, mask):
    """Overlay mask and draw a bounding box on the largest object."""
    out = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    try:
        colored = pcv.visualize.colorize_masks([mask], ["green"])
        out = pcv.visualize.overlay_two_imgs(out, colored, alpha=0.5)
    except Exception:
        overlay = np.zeros_like(out)
        overlay[mask > 0] = (0, 255, 0)
        out = cv2.addWeighted(out, 0.7, overlay, 0.3, 0)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if contours:
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) > 10:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(out, (x, y), (x + w, y + h), (255, 0, 0), 2)
    return out


def analyze_object(img, mask):
    """Draw object contour and extreme-axis lines."""
    out = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    if not contours:
        return out
    c = max(contours, key=cv2.contourArea)
    cv2.drawContours(out, [c], -1, (255, 0, 0), 2)
    try:
        pts = c.reshape(-1, 2)
        extremes = {
            "top": tuple(pts[pts[:, 1].argmin()]),
            "bottom": tuple(pts[pts[:, 1].argmax()]),
            "left": tuple(pts[pts[:, 0].argmin()]),
            "right": tuple(pts[pts[:, 0].argmax()]),
        }
        cv2.line(out, extremes["top"], extremes["bottom"], (255, 0, 255), 2)
        cv2.line(out, extremes["left"], extremes["right"], (255, 0, 255), 2)
    except Exception:
        pass
    return out


def draw_pseudolandmarks(img, mask):
    """Generate pseudolandmarks on the x-axis using PlantCV."""
    out = img.copy()
    if mask is None or cv2.countNonZero(mask) == 0:
        return out
    bin_mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
    prev = pcv.params.debug
    pcv.params.debug = None
    pcv.outputs.clear()
    try:
        groups = pcv.homology.x_axis_pseudolandmarks(img, bin_mask)
        colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        for i, grp in enumerate(groups[: len(colors)]):
            for pt in grp:
                if pt.shape == (1, 2):
                    x, y = map(int, pt[0])
                    cv2.circle(out, (x, y), 3, colors[i], -1)
    except Exception as e:
        print(f"Pseudolandmarks error: {e}")
    finally:
        pcv.params.debug = prev
    return out


def plot_color_histogram(img, mask, output_path=None, display_mode=False):
    """Plot multi-channel color histograms; save or display."""
    if mask is None or cv2.countNonZero(mask) == 0:
        return False
    b, g, r = cv2.split(img)
    h, s, v = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
    l, a, b2 = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2Lab))
    channels = {
        "red": r,
        "green": g,
        "blue": b,
        "hue": h,
        "saturation": s,
        "value": v,
        "lightness": l,
        "green-magenta": a,
        "blue-yellow": b2,
    }
    total = cv2.countNonZero(mask)
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, data in channels.items():
        hist = cv2.calcHist([data], [0], mask, [256], [0, 256])
        ax.plot((hist / total) * 100, label=name, linewidth=1.2)
    ax.set_xlim(0, 255)
    ax.set_xlabel("Pixel intensity")
    ax.set_ylabel("Percentage of pixels (%)")
    ax.grid(True)
    ax.legend(title="Channel", loc="center left",
              bbox_to_anchor=(1, 0.5))
    fig.tight_layout(rect=[0, 0, 0.8, 1])
    if display_mode:
        return True
    if output_path:
        fig.savefig(output_path,
                    facecolor=fig.get_facecolor())
    plt.close(fig)
    return True


def process_image(file, output_dir=None, transformations=None):
    """Apply selected transformations to a single image."""
    img = cv2.imread(file)
    if img is None:
        return (None, False) if output_dir is None else False
    funcs = {
        "original": (original_image, False),
        "blur": (blur_image, True),
        "mask": (masked_image, True),
        "roi": (draw_roi, True),
        "analyze": (analyze_object, True),
        "landmarks": (draw_pseudolandmarks, True),
        "histogram": (plot_color_histogram, False),
    }
    order = list(funcs.keys())
    if not transformations:
        transformations = order[:]
    valid = [t for t in transformations if t in funcs]
    if not valid:
        return (None, False) if output_dir is None else False
    base = os.path.splitext(os.path.basename(file))[0]
    results = {}
    mask = None
    if any(t in ["blur", "mask", "roi", "analyze", "landmarks"]
           for t in valid):
        mask = get_leaf_mask(img)
    hist = False
    for t in valid:
        fn, use_mask = funcs[t]
        out = None
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            out = os.path.join(output_dir, f"{base}_{t}.png")
        if t == "histogram":
            created = fn(img, mask, output_path=out,
                         display_mode=(output_dir is None))
            if output_dir is None:
                hist = created
        else:
            res = fn(img, mask) if use_mask else fn(img)
            if output_dir:
                cv2.imwrite(out, res)
            elif res is not None:
                results[t] = res
    if output_dir:
        return True
    return results, hist


def process_directory(src, dst, transformations=None):
    """Recursively apply transformations to all images in a directory."""
    if not os.path.isdir(src):
        print(f"Error: Source directory '{src}' not found.")
        return False
    os.makedirs(dst, exist_ok=True)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    print(f"Starting processing: {src} -> {dst}")
    for root, _, files in os.walk(src):
        rel = os.path.relpath(root, src)
        out = dst if rel == "." else os.path.join(dst, rel)
        os.makedirs(out, exist_ok=True)
        for f in files:
            if os.path.splitext(f.lower())[1] in exts:
                path = os.path.join(root, f)
                print(f"Processing: {path}", end="")
                ok = process_image(path, out, transformations)
                print(" [OK]" if ok else " [ERROR]")
    return True


def display_combined_results(results):
    """Display a grid of result images with titles."""
    if not results:
        return False
    titles = {
        "original": "Fig IV.1: Original",
        "blur": "Fig IV.2: Gaussian blur",
        "mask": "Fig IV.3: Mask",
        "roi": "Fig IV.4: ROI objects",
        "analyze": "Fig IV.5: Object analysis",
        "landmarks": "Fig IV.6: Pseudo-landmarks",
    }
    grid = [k for k in titles if k in results]
    n = len(grid)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    axes = axes.ravel()
    for i, key in enumerate(grid):
        img = results[key]
        ax = axes[i]
        if img.ndim == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(titles[key])
        ax.axis("off")
    for j in range(i + 1, rows * cols):
        axes[j].axis("off")
    fig.tight_layout()
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Lightweight image transformation "
                    "program for leaf images."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "image", nargs="?", help="Path to a single image.",
        default=None
    )
    group.add_argument(
        "-src", help="Source directory (recursive)."
    )
    parser.add_argument(
        "-dst", help="Destination directory. "
        "If omitted for single image, display results."
    )
    for idx, flag in enumerate(
        ["original", "blur", "mask", "roi",
         "analyze", "landmarks", "histogram"], start=1
    ):
        parser.add_argument(
            f"--{flag}", action="store_true",
            help=f"Include Fig IV.{idx}."
        )
    parser.add_argument(
        "--all", action="store_true",
        help="All transformations (default if none specified)."
    )
    args = parser.parse_args()

    flags = any(getattr(args, f) for f in [
        "original", "blur", "mask", "roi",
        "analyze", "landmarks", "histogram"
    ])
    default = [
        "original", "blur", "mask", "roi",
        "analyze", "landmarks", "histogram"
    ]
    transforms = default if args.all or not flags else [
        f for f in default if getattr(args, f)
    ]

    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: File '{args.image}' not found.")
            return
        if not args.dst:
            res, hist = process_image(args.image, None, transforms)
            shown = display_combined_results(res) if res else False
            if shown or hist:
                print("Displaying windows. Close all windows to exit.")
                plt.show()
            else:
                print("No results generated for display.")
        else:
            print(f"Saving mode -> '{args.dst}'.")
            process_image(args.image, args.dst, transforms)
    else:
        if not args.dst:
            print("Error: -dst required for -src.")
            return
        print(f"Directory processing mode: '{args.src}' -> '{args.dst}'")
        process_directory(args.src, args.dst, transforms)


if __name__ == "__main__":
    main()
