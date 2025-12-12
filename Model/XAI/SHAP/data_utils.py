from PIL import Image
import numpy as np
from torchvision import transforms
from pathlib import Path
from config import DATA_DIR, CLASS_NAMES, SAMPLES_PER_CLASS, IMAGE_SIZE, DEVICE


val_transform = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])


def preprocess_image(path):
    img = Image.open(path).convert('RGB')
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    img_np = np.array(img) / 255.0
    return tensor, img_np


def collect_image_paths(data_dir=DATA_DIR, class_names=CLASS_NAMES, samples_per_class=SAMPLES_PER_CLASS):
    data_dir = Path(data_dir)
    image_paths = []
    labels = []
    class_to_idx = {c: i for i, c in enumerate(class_names)}

    for cls in class_names:
        cls_dir = data_dir / cls
        imgs = sorted([
            p.name for p in cls_dir.iterdir()
            if p.suffix.lower() in ('.jpg', '.jpeg', '.png')
        ])[:samples_per_class]
        
        image_paths.extend([str(cls_dir / f) for f in imgs])
        labels.extend([class_to_idx[cls]] * len(imgs))

    return image_paths, labels


def make_background_tensors(image_paths, count=10):
    from config import DEVICE
    background_tensors = []
    
    for p in image_paths[:count]:
        t, _ = preprocess_image(p)
        background_tensors.append(t.squeeze(0))
    
    import torch
    background = torch.stack(background_tensors).to(DEVICE)
    return background