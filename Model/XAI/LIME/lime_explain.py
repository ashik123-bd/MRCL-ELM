# lime_explain.py
import torch
import numpy as np
from PIL import Image
from lime import lime_image
from skimage.morphology import binary_dilation, disk
from skimage.segmentation import mark_boundaries

from data_utils import test_transform
from config import NUM_SAMPLES, NUM_FEATURES

def predict_fn(images, model, device):
    batch = []
    for img in images:
        pil_img = Image.fromarray(img)
        batch.append(test_transform(pil_img))
    batch = torch.stack(batch).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(batch), dim=1)
    return probs.cpu().numpy()


def generate_lime(image_uint8, model, device):
    explainer = lime_image.LimeImageExplainer()
    explanation = explainer.explain_instance(
        image_uint8,
        lambda x: predict_fn(x, model, device),
        top_labels=1,
        hide_color=0,
        num_samples=NUM_SAMPLES,
        num_features=NUM_FEATURES
    )

    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=True,
        num_features=NUM_FEATURES,
        hide_rest=True
    )

    thick_mask = binary_dilation(mask, footprint=disk(3))
    return mark_boundaries(temp / 255.0, thick_mask, color=(1, 1, 0))
