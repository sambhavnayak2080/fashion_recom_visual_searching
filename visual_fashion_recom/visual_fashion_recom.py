"""
visual_fashion_recommender.py

Live webcam -> CLIP embeddings -> nearest neighbor search over local catalog -> recommendations + tips.

Requirements:
pip install transformers torch torchvision pillow opencv-python scikit-learn pandas numpy tqdm
"""

import os
import time
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import torch

# ------------------------------
# Config
CATALOG_DIR = "catalog_images"
METADATA_CSV = "catalog_metadata.csv"
EMBEDDINGS_NPY = "catalog_embs.npy"      # will be created
IDS_NPY = "catalog_ids.npy"
MODEL_NAME = "openai/clip-vit-base-patch32"
TOP_K = 5
THUMBNAIL_H = 120
THUMBNAIL_W = 80
# ------------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Loading CLIP model:", MODEL_NAME, "on", device)
model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
processor = CLIPProcessor.from_pretrained(MODEL_NAME)

# ------------------------------
# Utility functions
# ------------------------------
def load_metadata(path):
    df = pd.read_csv(path)
    # ensure filename paths
    df["filepath"] = df["filename"].apply(lambda x: os.path.join(CATALOG_DIR, x))
    return df

def compute_image_embedding(pil_img):
    """Return normalized numpy embedding (1D) for a PIL.Image"""
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        img_feats = model.get_image_features(**inputs)
    emb = img_feats.cpu().numpy()[0]
    # normalize
    emb = emb / np.linalg.norm(emb)
    return emb

def build_catalog_index(metadata_df, force_rebuild=False):
    """Compute & save embeddings for catalog images"""
    if os.path.exists(EMBEDDINGS_NPY) and os.path.exists(IDS_NPY) and not force_rebuild:
        print("Loading precomputed embeddings...")
        embs = np.load(EMBEDDINGS_NPY)
        ids = np.load(IDS_NPY)
        return embs, ids
    print("Computing embeddings for catalog images...")
    embeddings = []
    ids = []
    for _, row in tqdm(metadata_df.iterrows(), total=len(metadata_df)):
        fp = row["filepath"]
        if not os.path.exists(fp):
            print("Missing:", fp)
            embeddings.append(np.zeros(model.config.projection_dim))
            ids.append(row["id"])
            continue
        pil = Image.open(fp).convert("RGB")
        emb = compute_image_embedding(pil)
        embeddings.append(emb)
        ids.append(row["id"])
    embs = np.vstack(embeddings)
    np.save(EMBEDDINGS_NPY, embs)
    np.save(IDS_NPY, np.array(ids))
    return embs, np.array(ids)

def extract_dominant_colors(pil_img, n_colors=3):
    """Return n dominant colors in hex from a PIL image using k-means on pixels"""
    img = pil_img.resize((150, 150))
    arr = np.array(img).reshape(-1, 3).astype(float)
    # sample for speed
    if arr.shape[0] > 5000:
        idx = np.random.choice(arr.shape[0], 5000, replace=False)
        arr_sample = arr[idx]
    else:
        arr_sample = arr
    kmeans = KMeans(n_clusters=n_colors, random_state=0).fit(arr_sample)
    centers = kmeans.cluster_centers_.astype(int)
    hexes = [f"#{c[0]:02x}{c[1]:02x}{c[2]:02x}" for c in centers]
    return hexes

def colorfulness_metric(img_bgr):
    """Compute a simple colorfulness measure from Hasler & Suesstrunk (approx)"""
    img = img_bgr.astype("float")
    (B, G, R) = cv2.split(img)
    rg = np.abs(R - G)
    yb = np.abs(0.5 * (R + G) - B)
    std_root = np.sqrt(np.var(rg) ** 2 + np.var(yb) ** 2)
    mean_root = np.sqrt(np.mean(rg) ** 2 + np.mean(yb) ** 2)
    return std_root + 0.3 * mean_root

def edge_density(img_gray):
    edges = cv2.Canny(img_gray, 100, 200)
    return edges.mean() / 255.0

def skin_ratio(img_bgr):
    """Approx fraction of image area detected as skin using HSV thresholds (rough)"""
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(img_hsv)
    # simple skin HSV thresholds
    lower = np.array([0, 20, 70], dtype=np.uint8)
    upper = np.array([25, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_hsv, lower, upper)
    return (mask > 0).mean()

def get_top_k_similar(q_emb, catalog_embs, k=5):
    sims = cosine_similarity(q_emb.reshape(1, -1), catalog_embs)[0]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def generate_tips(user_features, neighbor_rows):
    """
    Simple heuristic tips:
      - color-based: if user is dark, suggest lighter for day events
      - if neighbor occasions indicate 'formal' but user appears casual -> suggest adding blazer
      - pattern-based: if user is very textured, recommend plain accessories
      - skin visible: suggest adding layers for formal events
    """
    tips = []
    brightness = user_features["brightness"]
    colorfulness = user_features["colorfulness"]
    edge = user_features["edge_density"]
    skin = user_features["skin_ratio"]

    # Occasion majority
    all_occasions = []
    for r in neighbor_rows:
        occs = str(r.get("occasions", ""))
        all_occasions += [o.strip().lower() for o in occs.split(",") if o.strip()]
    occasion_counts = pd.Series(all_occasions).value_counts() if len(all_occasions)>0 else pd.Series([])
    top_occ = occasion_counts.index[0] if len(occasion_counts)>0 else None

    if brightness < 90:
        tips.append("Your outfit looks dark. For daytime events consider lighter tones or a brighter accessory.")
    if brightness >= 180:
        tips.append("Your outfit is very light/bright — for evening/formal events consider adding darker layers for contrast.")
    if edge > 0.06:
        tips.append("You have a busy/patterned outfit — prefer plain accessories (solid shoes/bag) to balance it.")
    if colorfulness > 30:
        tips.append("Your outfit is colorful — keep accessories neutral to avoid clashing.")
    if skin > 0.12 and top_occ in ("formal","office"):
        tips.append("You're showing skin; for formal/office occasions consider adding a blazer or scarf.")
    if top_occ is not None:
        tips.append(f"Top matched occasion(s) among recommendations: {', '.join(occasion_counts.index[:3].tolist())}.")
    # Suggest swapping items if neighbors show consistent additional layers
    # Heuristic: if many neighbors have tag 'blazer' or 'jacket' and user seems casual => recommend layer
    neighbor_tags = ",".join([str(r.get("tags","")) for r in neighbor_rows]).lower()
    if ("blazer" in neighbor_tags or "jacket" in neighbor_tags) and not ("blazer" in neighbor_tags and user_features.get("has_layer", False)):
        tips.append("Consider adding a blazer/jacket to match the recommended outfits' formality.")
    if len(tips)==0:
        tips.append("Looks good! Try toggling accessories (belt, watch) or changing shoes to adapt to different occasions.")
    return tips

# ------------------------------
# Main runtime
# ------------------------------
def main():
    if not os.path.exists(METADATA_CSV):
        raise FileNotFoundError(f"Metadata CSV not found: {METADATA_CSV}")
    metadata_df = load_metadata(METADATA_CSV)
    catalog_embs, catalog_ids = build_catalog_index(metadata_df)

    # convert metadata to id->row dict for quick lookups
    meta_by_id = {int(row["id"]): row for _, row in metadata_df.iterrows()}

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam (index 0).")

    print("Press SPACE to capture and get recommendations. Press ESC to exit.")
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Frame failed")
            break
        display = frame.copy()
        cv2.putText(display, "Press SPACE to analyze, ESC to quit", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.imshow("Live - Visual Fashion Recommender", display)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        if key == 32:  # SPACE: capture
            capture = frame.copy()
            # convert BGR->PIL
            pil = Image.fromarray(cv2.cvtColor(capture, cv2.COLOR_BGR2RGB))
            q_emb = compute_image_embedding(pil)
            # compute basic features
            gray = cv2.cvtColor(capture, cv2.COLOR_BGR2GRAY)
            brightness = gray.mean()
            edge = edge_density(gray)
            skin = skin_ratio(capture)
            colorfulness = colorfulness_metric(capture)
            user_features = {
                "brightness": brightness,
                "edge_density": edge,
                "skin_ratio": skin,
                "colorfulness": colorfulness
            }
            # search
            idxs, sims = get_top_k_similar(q_emb, catalog_embs, k=TOP_K)
            neighbor_rows = []
            for i, sim in zip(idxs, sims):
                cid = int(catalog_ids[i])
                row = meta_by_id.get(cid, {})
                row = dict(row) if hasattr(row, "to_dict") else row
                row["sim"] = float(sim)
                neighbor_rows.append(row)

            # Build tips
            tips = generate_tips(user_features, neighbor_rows)

            # Build result canvas: original + thumbnails + text
            h, w, _ = capture.shape
            canvas_h = max(h, THUMBNAIL_H + 200)
            canvas_w = w + THUMBNAIL_W * TOP_K + 40
            canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 240

            # place original capture on left
            scale_w = min(w, 640)
            scale_h = int(h * (scale_w / w))
            small = cv2.resize(capture, (scale_w, scale_h))
            canvas[10:10+scale_h, 10:10+scale_w] = small

            # place thumbnails
            tx = 20 + scale_w
            ty = 10
            for i, nr in enumerate(neighbor_rows):
                fp = nr.get("filepath")
                if fp and os.path.exists(fp):
                    timg = cv2.imread(fp)
                    timg = cv2.resize(timg, (THUMBNAIL_W, THUMBNAIL_H))
                else:
                    timg = np.ones((THUMBNAIL_H, THUMBNAIL_W,3), dtype=np.uint8)*200
                canvas[ty:ty+THUMBNAIL_H, tx + i*(THUMBNAIL_W+10): tx + i*(THUMBNAIL_W+10)+THUMBNAIL_W] = timg
                # write sim score under thumbnail
                cv2.putText(canvas, f"{nr.get('sim',0):.2f}", (tx + i*(THUMBNAIL_W+10), ty+THUMBNAIL_H+18),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30,30,30), 1)

            # write textual metadata and tips on the right
            text_x = 20 + scale_w
            ty2 = THUMBNAIL_H + 40
            cv2.putText(canvas, "Top recommendations (filename - tags - occasions - score):", (text_x, ty2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            line_y = ty2 + 20
            for nr in neighbor_rows:
                txt = f"{nr.get('filename','?')} | {nr.get('tags','')} | {nr.get('occasions','')} | {nr.get('sim',0):.2f}"
                cv2.putText(canvas, txt, (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10,10,10), 1)
                line_y += 18

            # user feature values
            line_y += 8
            cv2.putText(canvas, f"User features: brightness={brightness:.0f}, edge={edge:.3f}, colorfulness={colorfulness:.2f}, skin_ratio={skin:.3f}",
                        (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (10,10,10), 1)
            line_y += 20
            cv2.putText(canvas, "Tips:", (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
            line_y += 18
            for t in tips:
                # wrap text approx
                cv2.putText(canvas, "- " + t, (text_x, line_y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,0,0), 1)
                line_y += 16
                if line_y > canvas_h - 20:
                    break

            # show dominant colors
            dom = extract_dominant_colors(pil, n_colors=3)
            cx = text_x
            cy = canvas_h - 40
            cv2.putText(canvas, "Dominant colors:", (cx, cy-18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,0,0), 1)
            for i, hx in enumerate(dom):
                # convert hex to BGR
                r = int(hx[1:3], 16); g = int(hx[3:5], 16); b = int(hx[5:7], 16)
                cv2.rectangle(canvas, (cx + i*40, cy), (cx + i*40 + 30, cy+20), (b,g,r), -1)
                cv2.putText(canvas, hx, (cx + i*40, cy+35), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)

            cv2.imshow("Results - Visual Fashion Recommender", canvas)
            # wait until a key to close results
            print("Top recommendations (printed):")
            for nr in neighbor_rows:
                print(f"  {nr.get('filename')}  | tags={nr.get('tags')} | occasions={nr.get('occasions')} | score={nr.get('sim'):.3f}")
            print("Tips:")
            for t in tips:
                print(" -", t)
            cv2.waitKey(0)
            cv2.destroyWindow("Results - Visual Fashion Recommender")
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
