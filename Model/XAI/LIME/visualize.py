# visualize.py
import matplotlib.pyplot as plt

def save_lime(original, lime_vis, class_name):
    fig, axs = plt.subplots(1, 2, figsize=(14, 7))

    axs[0].imshow(original)
    axs[0].set_title(f"True/Pred: {class_name}")
    axs[0].axis('off')

    axs[1].imshow(lime_vis)
    axs[1].set_title("LIME Explanation")
    axs[1].axis('off')

    plt.tight_layout()
    plt.savefig(f"lime_explanation_{class_name}.png", dpi=300)
    plt.close()
