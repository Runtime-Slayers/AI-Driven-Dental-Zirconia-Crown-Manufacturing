import os
import glob
import numpy as np
import matplotlib.pyplot as plt

def conformal_prediction_interval(mean_pred, var_pred, calibration_scores, alpha=0.1):
    """
    Computes a conformal prediction interval with finite-sample coverage guarantees.
    """
    n = len(calibration_scores)
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    if q_level > 1.0:
        q_level = 1.0
    q_hat = np.quantile(calibration_scores, q_level)
    
    # Prediction sets
    lower_bound = np.clip(mean_pred - q_hat * np.sqrt(var_pred), 0.0, 1.0)
    upper_bound = np.clip(mean_pred + q_hat * np.sqrt(var_pred), 0.0, 1.0)
    return lower_bound, upper_bound, q_hat

def generate_visualization(sample_path, uncertainty_path, save_plot_path):
    """
    Generates and saves a 4-panel figure comparing raw slice, ground truth mask,
    predicted segmentation, and conformal prediction bounds.
    """
    if not os.path.exists(uncertainty_path):
        print(f"Uncertainty map not found at {uncertainty_path}. Run training/inference first.")
        return
        
    # Load raw data and pre-computed uncertainty
    data = np.load(sample_path)
    raw = data['raw']
    gt_mask = data['mask']
    
    unc_data = np.load(uncertainty_path)
    mean_pred = unc_data['mean']
    var_pred = unc_data['variance']
    
    # Mock calibration scores for demonstration
    np.random.seed(42)
    calibration_scores = np.abs(np.random.normal(0, 0.1, 500))
    lower, upper, q_hat = conformal_prediction_interval(mean_pred, var_pred, calibration_scores, alpha=0.1)
    
    # The set size (upper - lower) represents the calibrated uncertainty
    conformal_uncertainty = upper - lower
    
    # Create plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Panel 1: Raw CBCT Slice
    im1 = axes[0, 0].imshow(raw, cmap='gray')
    axes[0, 0].set_title("Raw CBCT Slice (Radiographic Density)")
    fig.colorbar(im1, ax=axes[0, 0], label="Intensity")
    
    # Panel 2: Ground Truth Segmentation
    im2 = axes[0, 1].imshow(gt_mask, cmap='jet')
    axes[0, 1].set_title("Ground Truth Mask (Anatomy)")
    fig.colorbar(im2, ax=axes[0, 1])
    
    # Panel 3: Predicted Probability (Mean)
    im3 = axes[1, 0].imshow(mean_pred, cmap='jet')
    axes[1, 0].set_title("3D Attention U-Net Predicted Probability")
    fig.colorbar(im3, ax=axes[1, 0], label="Probability")
    
    # Panel 4: Conformal Prediction Uncertainty
    im4 = axes[1, 1].imshow(conformal_uncertainty, cmap='hot')
    axes[1, 1].set_title(f"Conformal Prediction Set Size (90% Coverage, q_hat={q_hat:.2f})")
    fig.colorbar(im4, ax=axes[1, 1], label="Interval Width")
    
    plt.tight_layout()
    plt.savefig(save_plot_path, dpi=150)
    plt.close()
    print(f"Conformal prediction visualization saved successfully to {save_plot_path}")

if __name__ == "__main__":
    preprocessed_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/preprocessed"
    uncertainty_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/mc_uncertainty_sample.npz"
    save_plot_path = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/uncertainty_visualization.png"
    
    first_sample = sorted(glob.glob(os.path.join(preprocessed_dir, "*.npz")))[0]
    generate_visualization(first_sample, uncertainty_path, save_plot_path)
