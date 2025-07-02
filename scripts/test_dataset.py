#!/usr/bin/env python3
"""
Test script for the PlayingCardDataset and related functionality.
This demonstrates how to use the integrated dataset components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data import (
    PlayingCardDataset, 
    get_basic_transforms, 
    create_dataloaders, 
    get_dataset_info
)
import torch


def test_basic_dataset():
    """Test basic dataset functionality."""
    print("=== Testing Basic Dataset ===")
    
    # Assuming you have train/valid/test directories in your project root
    train_dir = "train"
    
    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} directory not found. Skipping dataset test.")
        return
    
    # Create dataset without transforms first
    dataset = PlayingCardDataset(data_dir=train_dir)
    print(f"Dataset length: {len(dataset)}")
    print(f"Number of classes: {len(dataset.classes)}")
    
    # Test getting a sample
    image, label = dataset[0]
    print(f"Sample image shape: {image.shape}")
    print(f"Sample label: {label}")
    
    # Get class mapping
    target_to_class = dataset.get_target_to_class()
    print(f"Target 0 corresponds to: {target_to_class[0]}")
    print(f"Target {label} corresponds to: {target_to_class[label]}")


def test_transforms():
    """Test transform functionality."""
    print("\n=== Testing Transforms ===")
    
    # Create basic transforms
    transform = get_basic_transforms(image_size=(128, 128))
    print(f"Transform created: {transform}")
    
    # Test with dataset
    train_dir = "train"
    if os.path.exists(train_dir):
        dataset = PlayingCardDataset(train_dir, transform=transform)
        image, label = dataset[0]
        print(f"Transformed image shape: {image.shape}")
        print(f"Image tensor type: {type(image)}")


def test_dataloader():
    """Test dataloader functionality."""
    print("\n=== Testing DataLoader ===")
    
    train_dir = "train"
    valid_dir = "valid"
    
    if not (os.path.exists(train_dir) and os.path.exists(valid_dir)):
        print(f"Warning: Required directories not found. Skipping dataloader test.")
        return
    
    # Create dataloaders
    train_dataloader, val_dataloader = create_dataloaders(
        train_dir=train_dir,
        val_dir=valid_dir,
        batch_size=16,
        num_workers=0,  # Use 0 for testing to avoid multiprocessing issues
        image_size=(128, 128)
    )
    
    print(f"Number of training batches: {len(train_dataloader)}")
    print(f"Number of validation batches: {len(val_dataloader)}")
    
    # Test iterating through dataloader
    for batch_idx, (images, labels) in enumerate(train_dataloader):
        print(f"Batch {batch_idx}: images shape {images.shape}, labels shape {labels.shape}")
        print(f"Labels: {labels}")
        break  # Only test first batch


def test_dataset_info():
    """Test dataset information functionality."""
    print("\n=== Testing Dataset Info ===")
    
    train_dir = "train"
    if not os.path.exists(train_dir):
        print(f"Warning: {train_dir} directory not found. Skipping info test.")
        return
    
    info = get_dataset_info(train_dir)
    print("Dataset Information:")
    for key, value in info.items():
        if key != 'classes':  # Don't print all class names to keep output clean
            print(f"  {key}: {value}")
    print(f"  classes: {len(info['classes'])} classes")


if __name__ == "__main__":
    print("Testing PlayingCardDataset Integration")
    print("=" * 50)
    
    test_basic_dataset()
    test_transforms()
    test_dataloader()
    test_dataset_info()
    
    print("\n" + "=" * 50)
    print("Testing complete!") 