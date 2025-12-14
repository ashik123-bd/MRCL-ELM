import os
from pathlib import Path


ROOT = Path.cwd()
MODEL_PATH = Path('model path')
DATA_DIR = Path('dataset path')
OUTPUT_DIR = ROOT / 
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Other config
CLASS_NAMES = ['dataset classes']
SAMPLES_PER_CLASS = 6
BACKGROUND_COUNT = 10
DEVICE = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
IMAGE_SIZE = (128, 128)


# Visualization
SVG_DPI = 600
VMAX_PERCENTILE = 99.5