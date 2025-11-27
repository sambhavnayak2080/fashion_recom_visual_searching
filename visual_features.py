"""
scripts/visual_features.py — Phase 5 of FitFind
----------------------------------------
Extracts dominant colors, brightness level, and pattern clues from fashion images
using OpenCV + KMeans, then saves results to outputs/visual_features.json
"""

import cv2
import numpy as np
import os
import json
from tqdm import tqdm
from sklearn.cluster import KMeans


def extract_dominant_colors(image_path, k=3):
    """Return top-k dominant RGB colors in the image using KMeans."""
    img = cv2.imread(image_path)
    if img is None:
        return []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.reshape((-1, 3))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
    kmeans.fit(img)
    colors = kmeans.cluster_centers_.astype(int)
    return [tuple(c) for c in colors]


def color_name(rgb):
    """Approximate color name for RGB tuple."""
    r, g, b = rgb
    if r > 180 and g < 80 and b < 80:
        return "red"
    if g > 180 and r < 80 and b < 80:
        return "green"
    if b > 180 and r < 80 and g < 80:
        return "blue"
    if r > 200 and g > 200 and b < 100:
        return "yellow"
    if all(v > 200 for v in (r, g, b)):
        return "white"
    if all(v < 50 for v in (r, g, b)):
        return "black"
    if r > 160 and b > 160:
        return "pink/purple"
    return "mixed"


def extract_brightness_and_pattern(image_path):
    """Estimate brightness + simple pattern type."""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return "unknown", "unknown"

    # brightness
    brightness = np.mean(img)
    if brightness < 80:
        bright_label = "dark"
    elif brightness > 180:
        bright_label = "bright"
    else:
        bright_label = "medium"

    # texture/pattern clue via edges
    edges = cv2.Canny(img, 50, 150)
    density = np.sum(edges > 0) / edges.size
    if density < 0.02:
        pattern = "plain"
    elif density < 0.06:
        pattern = "textured"
    else:
        pattern = "complex"

    return bright_label, pattern


def extract_visual_features(input_json="./outputs/combined_preprocessed.json",
                            output_json="./outputs/visual_features.json"):
    """Main pipeline: load data, compute visual features, save results."""
    if not os.path.exists(input_json):
        raise FileNotFoundError(f"❌ Input file not found: {input_json}")

    with open(input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    results = []
    for item in tqdm(data, desc="🔍 Extracting visual features"):
        img_path = item.get("processed_path") or item.get("image_path")
        if not img_path or not os.path.exists(img_path):
            continue

        colors = extract_dominant_colors(img_path, k=3)
        color_names = [color_name(c) for c in colors]
        brightness, pattern = extract_brightness_and_pattern(img_path)

        item["dominant_colors"] = color_names
        item["brightness"] = brightness
        item["pattern"] = pattern

        results.append(item)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("✅ Visual feature extraction complete.")
    print(f"📦 Saved enriched data to: {output_json}")
    print(f"🖼️ Total processed images: {len(results)}")


if __name__ == "__main__":
    extract_visual_features()
