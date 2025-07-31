import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from models.bilinear_cnn import BilinearCNN
from datasets.car_dataset import CarDataset
from transforms.data_transforms import train_transform
from utils.train_eval import train_model, evaluate_model
from config import *

import torch
import torch.nn as nn
import torch.optim as optim

def load_data():
    x_train = np.load(DATA_DIR + 'x_train.npy').astype('float32') / 255.0
    y_train = np.load(DATA_DIR + 'y_train.npy').astype('int64') - 1
    x_test = np.load(DATA_DIR + 'x_test.npy').astype('float32') / 255.0
    y_test = np.load(DATA_DIR + 'y_test.npy').astype('int64') - 1

    x_train, x_val, y_train, y_val = train_test_split(
        x_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def main():
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_data(); input()

    train_ds = CarDataset(x_train, y_train, transform=train_transform)
    val_ds = CarDataset(x_val, y_val)
    test_ds = CarDataset(x_test, y_test)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    model = BilinearCNN(num_classes=NUM_CLASSES).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, DEVICE, NUM_EPOCHS)

    acc = evaluate_model(model, test_loader, DEVICE)
    print(f'Test Accuracy: {acc:.2f}%')

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
