import os
import matplotlib.pyplot as plt
import numpy as np
from config import OUTPUT_DIR, SVG_DPI, VMAX_PERCENTILE


def save_shap_svgs(
    all_shap_maps,
    all_original_imgs,
    all_labels_str,
    output_dir=OUTPUT_DIR,
    dpi=SVG_DPI,
    vmax_percentile=VMAX_PERCENTILE
):
  
    os.makedirs(output_dir, exist_ok=True)

   
    if vmax_percentile < 100:
        vmax = np.percentile(np.stack(all_shap_maps), vmax_percentile)
        print(f"Using {vmax_percentile}th percentile for color scaling: vmax = {vmax:.4f}")
    else:
        vmax = None  

    print(f"Saving {len(all_shap_maps)} SHAP visualizations to:\n    {os.path.abspath(output_dir)}\n")

    for i, (shap_map, img, label) in enumerate(zip(all_shap_maps, all_original_imgs, all_labels_str)):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7), dpi=100)

        # Original image
        ax1.imshow(img)
        ax1.set_title(f"Original Image\nTrue: {label}", fontsize=22, fontweight='bold', pad=20)
        ax1.axis('off')

        # SHAP heatmap
        im = ax2.imshow(
            shap_map,
            cmap='jet',
            vmin=0,
            vmax=vmax,
            alpha=0.9
        )
        ax2.set_title("SHAP Feature Importance Heatmap", fontsize=22, fontweight='bold', pad=20)
        ax2.axis('off')

        # Colorbar
        cbar = fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, aspect=20)
        cbar.set_label('Normalized SHAP Importance', fontsize=16, fontweight='bold')
        cbar.ax.tick_params(labelsize=14)

        # Overall title
        fig.suptitle(f"Sample {i+1:02d}", fontsize=18, y=0.95, fontweight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        # Safe filename
        safe_label = "".join(c for c in label if c.isalnum() or c in " _-").rstrip()
        safe_label = safe_label.replace(" ", "_")
        svg_path = os.path.join(output_dir, f"shap_{i+1:02d}_{safe_label}.svg")

        plt.savefig(
            svg_path,
            format="svg",
            dpi=dpi,
            bbox_inches="tight",
            transparent=False,
            facecolor="white"
        )
        plt.close(fig)

        print(f"Saved: {os.path.basename(svg_path)}")

    print(f"\nAll {len(all_shap_maps)} visualizations saved successfully!")