import os
from pathlib import Path


# Paths (edit to your environment)
ROOT = Path.cwd()
MODEL_PATH = Path('/kaggle/input/mrcl/pytorch/default/1/best_model.pth')
DATA_DIR = Path('/kaggle/input/sat-image-rgb/EuroSAT_RGB')
OUTPUT_DIR = ROOT / 'shap_explanations_svgs'
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Other config
CLASS_NAMES = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial',
'Pasture', 'PermanentCrop', 'Residential', 'River', 'SeaLake']
SAMPLES_PER_CLASS = 6
BACKGROUND_COUNT = 10
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
IMAGE_SIZE = (128, 128)


# Visualization
SVG_DPI = 600
VMAX_PERCENTILE = 99.5