import os
import numpy as np
from PIL import Image

def grayscale_to_hu(pixel_val):
    """
    Linearly maps grayscale pixel intensity [0, 255] to Hounsfield Units (HU) [-1000, 2825].
    """
    return 15.0 * pixel_val - 1000.0

def hu_to_youngs_modulus(hu_val):
    """
    Maps HU value to Young's Modulus (GPa) based on Misch D1-D4 classification.
    """
    if hu_val < 150:
        return 0.001  # Very soft tissue / air
    elif 150 <= hu_val < 350:
        return 1.5   # D4: Fine trabecular bone
    elif 350 <= hu_val < 850:
        return 5.0   # D3: Thin porous cortical & fine trabecular bone
    elif 850 <= hu_val < 1250:
        return 10.0  # D2: Thick porous cortical & coarse trabecular bone
    else:
        return 15.0  # D1: Dense cortical bone / tooth structure

def preprocess_and_save(image_dir, mask_dir, output_numpy_dir):
    """
    Reads images, performs HU calibration, maps to Young's Modulus, and saves as numpy arrays.
    """
    os.makedirs(output_numpy_dir, exist_ok=True)
    images = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    
    print(f"Preprocessing {len(images)} images...")
    for img_name in images:
        img_path = os.path.join(image_dir, img_name)
        
        # Load grayscale image
        img = np.array(Image.open(img_path).convert('L'))
        
        # Apply calibration
        hu = grayscale_to_hu(img)
        E = np.vectorize(hu_to_youngs_modulus)(hu)
        
        # Determine if mask exists (for labeled dataset)
        mask_name = img_name.replace(".png", "_mask.png")
        mask_path = os.path.join(mask_dir, mask_name)
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('1')).astype(np.uint8)
        else:
            mask = np.zeros_like(img)
            
        # Save as npz file
        sample_id = os.path.splitext(img_name)[0]
        np.savez_compressed(
            os.path.join(output_numpy_dir, f"{sample_id}.npz"),
            raw=img,
            hu=hu,
            E=E,
            mask=mask
        )
    print("Preprocessing completed!")

if __name__ == "__main__":
    image_dir = "/Users/saranboddu/Downloads/Maths_Dataset_Sem_5/Dental_Robotics_Datasets/Volumetric_Anatomy/extracted_tooth_data/labeled/images"
    mask_dir = "/Users/saranboddu/Downloads/Maths_Dataset_Sem_5/Dental_Robotics_Datasets/Volumetric_Anatomy/extracted_tooth_data/labeled/masks"
    output_numpy_dir = "/Users/saranboddu/Desktop/Amrita/Year_3/maths/Project/Stage1_Segmentation/preprocessed"
    
    preprocess_and_save(image_dir, mask_dir, output_numpy_dir)
