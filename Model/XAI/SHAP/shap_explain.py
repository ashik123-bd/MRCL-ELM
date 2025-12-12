import torch
import shap
import numpy as np
from model import EfficientNetLSTMModel
from data_utils import preprocess_image, collect_image_paths, make_background_tensors
from config import MODEL_PATH, DEVICE


class WrappedModel(torch.nn.Module):
    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, x):
      
        return self.m(x)


def load_model(model_path=MODEL_PATH):
    model = EfficientNetLSTMModel().to(DEVICE)
    state_dict = torch.load(model_path, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def compute_shap_for_paths(model, image_paths, labels, background, class_names):
  
    wrapped = WrappedModel(model)


    print("Creating SHAP GradientExplainer...")
    explainer = shap.GradientExplainer(wrapped, background)

    all_shap_maps = []
    all_original_imgs = []
    all_labels_str = []

    print(f"Computing SHAP values for {len(image_paths)} images...\n")

    for idx, (path, true_idx) in enumerate(zip(image_paths, labels)):
        print(f"[{idx+1}/{len(image_paths)}] Explaining: {path.name} | True: {class_names[true_idx]}")

        # Preprocess image
        x_tensor, img_np = preprocess_image(path) 
        try:
       
            shap_values = explainer.shap_values(x_tensor) 

        
            sv = shap_values[true_idx]         
            sv = np.abs(sv[0]).mean(axis=0)     
            sv = (sv - sv.min()) / (sv.ptp() + 1e-8) 

            all_shap_maps.append(sv)
            all_original_imgs.append(img_np)
            all_labels_str.append(class_names[true_idx])

        except Exception as e:
            print(f"Warning: SHAP failed for {path}: {e}")
           
            H, W = img_np.shape[:2]
            all_shap_maps.append(np.zeros((H, W), dtype=np.float32))
            all_original_imgs.append(img_np)
            all_labels_str.append(class_names[true_idx])


        del x_tensor, shap_values
        torch.cuda.empty_cache()

    return all_shap_maps, all_original_imgs, all_labels_str