import os
import numpy as np
import cv2
from glob import glob

def ensemble_model_outputs(
    base_dir, 
    model_names, 
    threshold=None
):
    if(not model_names):
        return
    # Determine threshold for majority voting
    if threshold is None:
        threshold = 0.5
    
    image_folders = os.listdir(base_dir)
    image_folders.remove("log")
    image_folders.remove("Training.h5")
    image_folders.remove("model")

    # Iterate through all directories in the base path
    for folder in image_folders:
        folder_path = os.path.join(base_dir, folder, "auto segmentation")
        if not os.path.isdir(folder_path):
            continue
        
        print(f"Processing folder: {folder_path}")
        
        # Load all model outputs for this case
        model_outputs = []
        for model_name in model_names:
            model_path = os.path.join(folder_path, model_name)
            if not os.path.isdir(model_path):
                print(f"Model path not found: {model_path}, skipping...")
                continue
            
            # Read all slices for the model
            model_slices = []
            for tiff_file in sorted(glob(os.path.join(model_path, "*.tiff"))):
                slice_data = cv2.imread(tiff_file, cv2.IMREAD_GRAYSCALE)
                if slice_data is None:
                    print(f"Failed to read: {tiff_file}")
                    continue
                model_slices.append(slice_data)
            
            if model_slices:
                model_outputs.append(np.array(model_slices))
        
        # Ensure there are enough model outputs for voting
        if len(model_outputs) <= 1:
            print(f"Not enough model outputs for {folder}, skipping...")
            continue
        
        # Stack and compute majority vote
        stacked_outputs = np.stack(model_outputs, axis=0)
        majority_vote = np.mean(stacked_outputs, axis=0) > (threshold * 255)
        
        # Save majority voted slices as TIFF files
        save_path = os.path.join(base_dir, folder, "auto segmentation", "ensemble")
        os.makedirs(save_path, exist_ok=True)
        
        for i, slice_data in enumerate(majority_vote):
            output_filename = os.path.join(save_path, f"slice{i + 1:03}.tiff")
            cv2.imwrite(output_filename, np.uint8(255 * slice_data))
        
        print(f"Results saved for {folder} in {save_path}")
