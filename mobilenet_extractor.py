import torch #type: ignore
from torchvision import models, transforms #type: ignore
from PIL import Image, UnidentifiedImageError
import numpy as np
import json
from tqdm import tqdm
import os


class MobileNetFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # ✅ Use pretrained MobileNetV3 (small)
        self.model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        self.model.classifier = torch.nn.Identity()  # remove final classifier
        self.model.eval().to(self.device)

        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def extract(self, img_path):
        img = Image.open(img_path).convert("RGB")
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor).squeeze().cpu().numpy()
        return features.tolist()


def extract_image_embeddings():
    print("🖼️ Extracting image embeddings using MobileNetV3-Small...")
    extractor = MobileNetFeatureExtractor()
    
    with open("./outputs/combined_preprocessed.json", "r") as f:
        data = json.load(f)

    image_embeddings = {}
    total_items = len(data)
    skipped = 0

    for item in tqdm(data, desc="Extracting image features"):
        img_path = item.get("processed_path") or item.get("image_path")
        if not img_path or not os.path.exists(img_path):
            skipped += 1
            continue

        try:
            emb = extractor.extract(img_path)
            image_embeddings[img_path] = emb
        except (UnidentifiedImageError, OSError) as e:
            print(f"❌ Corrupted image: {img_path}")
            skipped += 1
        except Exception as e:
            print(f"⚠️ Error with {img_path}: {e}")
            skipped += 1

    np.save("./outputs/image_embeddings.npy", image_embeddings)
    print(f"✅ Saved {len(image_embeddings)} embeddings (skipped {skipped}/{total_items})")


if __name__ == "__main__":
    extract_image_embeddings()
