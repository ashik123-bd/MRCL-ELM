# XAI - EuroSAT SHAP


Repository containing a modularized pipeline that computes SHAP explanations for a trained PyTorch model (EfficientNet backbone + custom heads).


## Files
- `config.py` - configuration constants and paths
- `model.py` - model classes (ResidualBlock, ProjectionPath, ELMLayer, EfficientNetLSTMModel)
- `data_utils.py` - preprocessing, image collection, background creation
- `shap_explain.py` - SHAP explainer creation and computing SHAP maps
- `visualize.py` - saving visualizations (SVGs)
- `run_shap.py` - orchestrator script to run the complete pipeline


## Usage
1. Install requirements: `pip install -r requirements.txt`
2. Edit `config.py` to match paths (MODEL_PATH, DATA_DIR, OUTPUT_DIR)
3. Run: `python run_shap.py`