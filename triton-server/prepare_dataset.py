#!/usr/bin/env python3
"""
Dataset Preparation Script for Product Matching System
Downloads STL-10 dataset subset for product matching demo
"""

import os
import json
from PIL import Image
import shutil
import torchvision

def prepare_stl10_dataset(max_images=200, output_dir="data"):
    """
    Download and prepare STL-10 dataset subset
    
    Args:
        max_images: Maximum number of images to download (default: 200)
        output_dir: Directory to store the dataset (default: "data")
    """
    print(f"Preparing STL-10 dataset (max {max_images} images)...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    try:
        # STL-10 class names
        class_names = [
            "airplane", "bird", "car", "cat", "deer", 
            "dog", "horse", "monkey", "ship", "truck"
        ]
        
        # Load STL-10 dataset (train split)
        print("Loading STL-10 dataset...")
        dataset = torchvision.datasets.STL10(
            root='./stl10_cache', 
            split='train', 
            download=True,
        )
        
        metadata = []
        
        print(f"Processing first {max_images} images...")
        
        for idx in range(min(len(dataset), max_images)):
            try:
                image, label = dataset[idx]
                category = class_names[label]
                
                # Generate filename
                filename = f"stl10_{idx:04d}.jpg"
                image_path = os.path.join(images_dir, filename)
                
                # Save image without resizing (keep original 96x96 resolution)
                if isinstance(image, Image.Image):
                    # Convert to RGB if needed
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    image.save(image_path, "JPEG", quality=95)
                else:
                    print(f"Skipping invalid image at index {idx}")
                    continue
                
                # Extract metadata
                item_metadata = {
                    "id": idx,
                    "filename": filename,
                    "image_path": image_path,
                    "category": category,
                }
                
                metadata.append(item_metadata)
                
                if (idx + 1) % 50 == 0:
                    print(f"Processed {idx + 1} images...")
                    
            except Exception as e:
                print(f"Error processing image {idx}: {e}")
                continue
        
        # Save metadata
        metadata_path = os.path.join(output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Dataset preparation complete!")
        print(f"- Downloaded {len(metadata)} images")
        print(f"- Saved to: {output_dir}/")
        print(f"- Metadata: {metadata_path}")
        print(f"- Categories: {set(item['category'] for item in metadata)}")
        
        return metadata
        
    except Exception as e:
        print(f"Error loading STL-10 dataset: {e}")
        print("Dataset preparation failed. Please check your internet connection and try again.")
        return []

def main():
    """Main function to prepare dataset"""
    print("Starting dataset preparation...")
    
    # Prepare dataset with STL-10
    metadata = prepare_stl10_dataset(max_images=200)
    
    print("\nDataset preparation summary:")
    print(f"- Total items: {len(metadata)}")
    if metadata:
        print(f"- Categories: {set(item['category'] for item in metadata)}")
    print(f"- Storage location: data/")
    print("\nReady for next step: MongoDB Mock Setup")

if __name__ == "__main__":
    main() 