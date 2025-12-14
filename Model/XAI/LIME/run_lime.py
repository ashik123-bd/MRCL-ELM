import os
import torch
from PIL import Image
import numpy as np

from config import CATEGORIES, DATASET_PATH, MODEL_PATH
from model import EfficientNetLSTMModel
from data_utils import test_transform, denormalize
from lime_explain import generate_lime
from visualize import save_lime


def main():
   
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    # ------------------- Load Model -------------------
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    model = EfficientNetLSTMModel().to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print("[INFO] Model loaded successfully")

    # ------------------- Loop Over Classes -------------------
    for class_name in CATEGORIES:
        class_idx = CATEGORIES.index(class_name)
        class_dir = os.path.join(DATASET_PATH, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARN] Folder not found for class '{class_name}', skipping")
            continue

        print(f"\n[INFO] Searching for correctly predicted sample: {class_name}")
        found = False

        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tiff")):
                continue

            img_path = os.path.join(class_dir, img_name)
            image = Image.open(img_path).convert("RGB")

            input_tensor = test_transform(image).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                pred_class = torch.argmax(output, dim=1).item()
                confidence = torch.softmax(output, dim=1)[0, pred_class].item()

            if pred_class == class_idx:
                print(f"[FOUND] {img_name} | Confidence: {confidence:.3f}")

                # ------------------- Prepare Image -------------------
                img_denorm = denormalize(input_tensor[0])
                img_uint8 = (img_denorm * 255).astype(np.uint8)

                # ------------------- Generate LIME -------------------
                print("[INFO] Generating LIME explanation (may take ~1–2 minutes)")
                lime_vis = generate_lime(
                    image_uint8=img_uint8,
                    model=model,
                    device=device
                )

                # ------------------- Save Visualization -------------------
                save_lime(
                    original=img_denorm,
                    lime_vis=lime_vis,
                    class_name=class_name
                )

                print(f"[SAVED] lime_explanation_{class_name}.png")
                found = True
                break  # one image per class

        if not found:
            print(f"[WARN] No correctly predicted sample found for '{class_name}'")

    print("\n[INFO] LIME explanations completed for all classes")


if __name__ == "__main__":
    main()
