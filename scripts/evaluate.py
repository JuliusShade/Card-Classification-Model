#!/usr/bin/env python3
"""
Evaluate the trained card classifier and visualize predictions.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from glob import glob

from src.data import PlayingCardDataset, get_basic_transforms
from src.models import create_model, load_model_checkpoint

# Config
TEST_DIR = 'test'
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
MODEL_TYPE = 'simple'
MODEL_NAME = 'efficientnet_b0'
NUM_CLASSES = 53
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def preprocess_image(image_path, transform):
    image = Image.open(image_path).convert("RGB")
    return image, transform(image).unsqueeze(0)


def predict(model, image_tensor, device):
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return probabilities.cpu().numpy().flatten()


def visualize_predictions(original_image, probabilities, class_names):
    fig, axarr = plt.subplots(1, 2, figsize=(14, 7))
    axarr[0].imshow(original_image)
    axarr[0].axis("off")
    axarr[1].barh(class_names, probabilities)
    axarr[1].set_xlabel("Probability")
    axarr[1].set_title("Class Predictions")
    axarr[1].set_xlim(0, 1)
    plt.tight_layout()
    plt.show()


def main():
    # Load dataset for class names
    test_dataset = PlayingCardDataset(TEST_DIR, transform=get_basic_transforms((128, 128)))
    class_names = test_dataset.classes

    # Load model
    model = create_model(
        model_type=MODEL_TYPE,
        num_classes=NUM_CLASSES,
        model_name=MODEL_NAME,
        pretrained=False
    )
    optimizer = torch.optim.Adam(model.parameters())  # Dummy optimizer for loading
    checkpoint = load_model_checkpoint(model, optimizer, CHECKPOINT_PATH, device=DEVICE)
    model.to(DEVICE)

    # Get test images
    test_images = glob(os.path.join(TEST_DIR, '*', '*'))
    np.random.seed(42)
    test_examples = np.random.choice(test_images, 10, replace=False)

    # Transform
    transform = get_basic_transforms((128, 128))

    for example in test_examples:
        original_image, image_tensor = preprocess_image(example, transform)
        probabilities = predict(model, image_tensor, DEVICE)
        visualize_predictions(original_image, probabilities, class_names)

if __name__ == "__main__":
    main()
