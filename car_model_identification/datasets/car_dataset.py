import torch
from torch.utils.data import Dataset

class CarDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = torch.from_numpy(self.images[idx]).permute(2, 0, 1)
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
