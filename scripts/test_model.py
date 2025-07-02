#!/usr/bin/env python3
"""
Test script for the card classifier models and related functionality.
This demonstrates how to use the integrated model components.
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
from src.models import (
    SimpleCardClassifier,
    AdvancedCardClassifier,
    create_model,
    count_parameters,
    get_model_size_mb,
    get_model_summary
)


def test_simple_model():
    """Test the SimpleCardClassifier model."""
    print("=== Testing SimpleCardClassifier ===")
    
    # Create the model
    model = SimpleCardClassifier(num_classes=53)
    print(f"Model created successfully")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 53)")
    
    # Test model info
    print(f"Number of classes: {model.num_classes}")
    print(f"Model name: {model.model_name}")
    print(f"Feature dimension: {model.feature_dim}")


def test_advanced_model():
    """Test the AdvancedCardClassifier model."""
    print("\n=== Testing AdvancedCardClassifier ===")
    
    # Create the model
    model = AdvancedCardClassifier(
        num_classes=53,
        hidden_size=512,
        dropout_rate=0.3
    )
    print(f"Advanced model created successfully")
    
    # Test forward pass
    batch_size = 4
    input_tensor = torch.randn(batch_size, 3, 224, 224)
    
    with torch.no_grad():
        output = model(input_tensor)
        print(f"Input shape: {input_tensor.shape}")
        print(f"Output shape: {output.shape}")
        print(f"Expected output shape: ({batch_size}, 53)")


def test_model_factory():
    """Test the model factory function."""
    print("\n=== Testing Model Factory ===")
    
    # Create simple model using factory
    simple_model = create_model(
        model_type='simple',
        num_classes=53,
        model_name='efficientnet_b0',
        pretrained=True
    )
    print(f"Simple model created via factory")
    
    # Create advanced model using factory
    advanced_model = create_model(
        model_type='advanced',
        num_classes=53,
        model_name='efficientnet_b0',
        pretrained=True,
        hidden_size=512
    )
    print(f"Advanced model created via factory")
    
    # Test that they work
    input_tensor = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        simple_output = simple_model(input_tensor)
        advanced_output = advanced_model(input_tensor)
        
        print(f"Simple model output shape: {simple_output.shape}")
        print(f"Advanced model output shape: {advanced_output.shape}")


def test_model_utilities():
    """Test model utility functions."""
    print("\n=== Testing Model Utilities ===")
    
    # Create a model
    model = SimpleCardClassifier(num_classes=53)
    
    # Count parameters
    param_counts = count_parameters(model)
    print("Parameter counts:")
    for key, value in param_counts.items():
        print(f"  {key}: {value:,}")
    
    # Get model size
    model_size = get_model_size_mb(model)
    print(f"Model size: {model_size:.2f} MB")
    
    # Get comprehensive summary
    summary = get_model_summary(model)
    print("Model summary:")
    for key, value in summary.items():
        if key == 'parameters':
            print(f"  {key}:")
            for param_key, param_value in value.items():
                print(f"    {param_key}: {param_value:,}")
        else:
            print(f"  {key}: {value}")


def test_model_methods():
    """Test model-specific methods."""
    print("\n=== Testing Model Methods ===")
    
    model = SimpleCardClassifier(num_classes=53)
    
    # Test feature extractor
    feature_extractor = model.get_feature_extractor()
    print(f"Feature extractor type: {type(feature_extractor)}")
    
    # Test parameter freezing/unfreezing
    print("Testing parameter freezing...")
    model.freeze_backbone()
    
    trainable_params = model.get_trainable_parameters()
    print(f"Trainable parameters after freezing: {len(trainable_params)}")
    
    model.unfreeze_backbone()
    trainable_params = model.get_trainable_parameters()
    print(f"Trainable parameters after unfreezing: {len(trainable_params)}")


def test_different_models():
    """Test different base models."""
    print("\n=== Testing Different Base Models ===")
    
    # Test with different EfficientNet variants
    model_names = ['efficientnet_b0', 'efficientnet_b1', 'efficientnet_b2']
    
    for model_name in model_names:
        try:
            model = SimpleCardClassifier(
                num_classes=53,
                model_name=model_name,
                pretrained=True
            )
            print(f"Successfully created model with {model_name}")
            
            # Test forward pass
            input_tensor = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                output = model(input_tensor)
                print(f"  Output shape: {output.shape}")
                
        except Exception as e:
            print(f"Failed to create model with {model_name}: {e}")


if __name__ == "__main__":
    print("Testing Card Classifier Models")
    print("=" * 50)
    
    test_simple_model()
    test_advanced_model()
    test_model_factory()
    test_model_utilities()
    test_model_methods()
    test_different_models()
    
    print("\n" + "=" * 50)
    print("Model testing complete!") 