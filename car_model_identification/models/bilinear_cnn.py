import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class BilinearCNN(nn.Module):
    def __init__(self, num_classes=20):
        super().__init__()
        resnet = models.resnet50(weights='IMAGENET1K_V1')
        vgg = models.vgg16(weights='IMAGENET1K_V1')
        self.resnet_features = nn.Sequential(*list(resnet.children())[:-2])
        self.vgg_features = nn.Sequential(*list(vgg.features.children()))
        self.classifier = nn.Linear(2048 * 512, num_classes)

    def outer_product(self, x1, x2):
        b, _, h, w = x1.shape
        x2 = F.interpolate(x2, size=(h, w), mode='bilinear', align_corners=False)
        phi_I = torch.einsum('bchw,bdhw->bcd', x1, x2).reshape(b, -1) / (h * w)
        y_ssqrt = torch.sign(phi_I) * torch.sqrt(torch.abs(phi_I) + 1e-12)
        return F.normalize(y_ssqrt, p=2, dim=1)

    def forward(self, x):
        x1 = self.resnet_features(x)
        x2 = self.vgg_features(x)
        bilinear = self.outer_product(x1, x2)
        return self.classifier(bilinear)
