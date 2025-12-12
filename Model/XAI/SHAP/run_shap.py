import gc
import torch
from config import CLASS_NAMES, SAMPLES_PER_CLASS, BACKGROUND_COUNT
from data_utils import collect_image_paths, make_background_tensors
from shap_explain import load_model, compute_shap_for_paths
from visualize import save_shap_svgs


def main():
    image_paths, labels = collect_image_paths()
    print(f"Collected {len(image_paths)} images")


    background = make_background_tensors(image_paths, count=BACKGROUND_COUNT)
    print(f"Background tensor created with {background.shape[0]} samples")

    # Load the trained model
    model = load_model()
    model.eval()  


    print("Computing SHAP explanations...")
    all_shap_maps, all_original_imgs, all_labels_str = compute_shap_for_paths(
        model, image_paths, labels, background, CLASS_NAMES
    )

 
    print("Saving SHAP visualizations as SVG...")
    save_shap_svgs(all_shap_maps, all_original_imgs, all_labels_str)

    # Clean up memory
    del background, all_shap_maps, all_original_imgs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("Done! All SHAP visualizations saved successfully.")


if __name__ == '__main__':
    main()