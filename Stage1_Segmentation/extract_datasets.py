import os
import io
import pandas as pd
from PIL import Image
from tqdm import tqdm

def extract_parquet(parquet_path, output_images_dir, output_masks_dir=None):
    """
    Extracts images (and optionally masks) from a parquet file and saves them.
    """
    print(f"Reading {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    # Ensure output directories exist
    os.makedirs(output_images_dir, exist_ok=True)
    if output_masks_dir:
        os.makedirs(output_masks_dir, exist_ok=True)
        
    num_rows = len(df)
    print(f"Processing {num_rows} records...")
    
    for _, row in tqdm(df.iterrows(), total=num_rows, desc=os.path.basename(parquet_path)):
        sample_id = row['sample_id']
        
        # 1. Extract and save image
        image_data = row['image']
        if image_data is not None and 'bytes' in image_data:
            img = Image.open(io.BytesIO(image_data['bytes']))
            img.save(os.path.join(output_images_dir, f"{sample_id}.png"))
            
        # 2. Extract and save mask (if applicable and present)
        if output_masks_dir:
            mask_data = row['mask']
            if mask_data is not None and 'bytes' in mask_data:
                msk = Image.open(io.BytesIO(mask_data['bytes']))
                msk.save(os.path.join(output_masks_dir, f"{sample_id}_mask.png"))

def main():
    base_dir = "."
    output_base = os.path.join(base_dir, "extracted_tooth_data")
    
    # Define files and output directories
    datasets = [
        # Labeled files (have both images and masks)
        {
            "path": os.path.join(base_dir, "labeled", "c_pxi_labeled-00000-of-00001.parquet"),
            "images_dir": os.path.join(output_base, "labeled", "images"),
            "masks_dir": os.path.join(output_base, "labeled", "masks")
        },
        {
            "path": os.path.join(base_dir, "labeled", "a_pxi_labeled-00000-of-00001.parquet"),
            "images_dir": os.path.join(output_base, "labeled", "images"),
            "masks_dir": os.path.join(output_base, "labeled", "masks")
        },
        # Unlabeled files (only have images)
        {
            "path": os.path.join(base_dir, "unlabeled", "c_pxi_unlabeled-00000-of-00001.parquet"),
            "images_dir": os.path.join(output_base, "unlabeled", "images"),
            "masks_dir": None
        },
        {
            "path": os.path.join(base_dir, "unlabeled", "a_pxi_unlabeled-00000-of-00001.parquet"),
            "images_dir": os.path.join(output_base, "unlabeled", "images"),
            "masks_dir": None
        }
    ]
    
    for dataset in datasets:
        if os.path.exists(dataset["path"]):
            extract_parquet(
                parquet_path=dataset["path"],
                output_images_dir=dataset["images_dir"],
                output_masks_dir=dataset["masks_dir"]
            )
        else:
            print(f"File not found: {dataset['path']}")
            
    print("\nExtraction complete! All files saved to 'extracted_tooth_data/'.")

if __name__ == "__main__":
    main()
