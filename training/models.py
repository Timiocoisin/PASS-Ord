"""CNN architecture copied unchanged from the retained nested implementation."""
import torch.nn as nn
from torchvision import models


class MultiTaskCNN(nn.Module):
    def __init__(self):
        super().__init__()
        bb = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        in_f = bb.fc.in_features
        bb.fc = nn.Identity()
        self.backbone = bb
        self.head_cls3 = nn.Linear(in_f, 3)
        self.head_cls5 = nn.Linear(in_f, 5)
        self.head_prog = nn.Linear(in_f, 1)

    def forward(self, x):
        f = self.backbone(x)
        return self.head_cls3(f), self.head_cls5(f), self.head_prog(f).squeeze(-1)
