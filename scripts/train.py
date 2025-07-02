#!/usr/bin/env python3
"""
Train a card classifier model using PyTorch.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

from src.data import create_dataloaders
from src.models import create_model, save_model_checkpoint

# Config
TRAIN_DIR = 'train'
VAL_DIR = 'valid'
BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_TYPE = 'simple'  # or 'advanced'
MODEL_NAME = 'efficientnet_b0'
NUM_CLASSES = 53
CHECKPOINT_PATH = 'checkpoints/best_model.pth'
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def main():
    # Data
    train_loader, val_loader = create_dataloaders(
        train_dir=TRAIN_DIR,
        val_dir=VAL_DIR,
        batch_size=BATCH_SIZE,
        image_size=(128, 128),
        shuffle=True,
        augmentation=True,
        num_workers=0  # Set >0 for faster loading if not on Windows
    )

    # Model
    model = create_model(
        model_type=MODEL_TYPE,
        num_classes=NUM_CLASSES,
        model_name=MODEL_NAME,
        pretrained=True
    )
    model.to(DEVICE)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    train_losses, val_losses = [], []
    best_val_loss = float('inf')

    for epoch in range(NUM_EPOCHS):
        # Training phase
        model.train()
        running_loss = 0.0
        for images, labels in tqdm(train_loader, desc=f'Epoch {epoch+1} Training'):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
        train_loss = running_loss / len(train_loader.dataset)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} Validation'):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                running_loss += loss.item() * labels.size(0)
        val_loss = running_loss / len(val_loader.dataset)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{NUM_EPOCHS} - Train loss: {train_loss:.4f}, Validation loss: {val_loss:.4f}")

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
            save_model_checkpoint(
                model, optimizer, epoch+1, val_loss, 0.0, CHECKPOINT_PATH
            )
            print(f"Saved new best model to {CHECKPOINT_PATH}")

    # Plot losses
    plt.figure()
    plt.plot(train_losses, label='Training loss')
    plt.plot(val_losses, label='Validation loss')
    plt.legend()
    plt.title("Loss over epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.savefig('loss_curve.png')
    plt.show()
    print("Training complete! Loss curve saved as loss_curve.png")

if __name__ == "__main__":
    main()
