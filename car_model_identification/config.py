import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 20
BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
DATA_DIR = "data/"
MODEL_PATH = "car_classification_bilinear.pth"
