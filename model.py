# model.py

import torch
import torch.nn as nn
import torchvision.models as models

class CombinedModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, hidden_dim=512, dropout=0.5):
        super(CombinedModel, self).__init__()
        # ---- MobileNetV2 ----
        self.mobilenet = models.mobilenet_v2(pretrained=pretrained)
        self.mobilenet_features = self.mobilenet.features
        self.mobilenet_pool = nn.AdaptiveAvgPool2d((1,1))
        for p in self.mobilenet_features.parameters():
            p.requires_grad = False

        # ---- ResNet50 ----
        resnet = models.resnet50(pretrained=pretrained)
        modules = list(resnet.children())[:-2]  # drop avgpool & fc
        self.resnet_features = nn.Sequential(*modules)
        self.resnet_pool = nn.AdaptiveAvgPool2d((1,1))
        for p in self.resnet_features.parameters():
            p.requires_grad = False

        # feature dims
        mobilenet_dim = 1280   # mobilenet_v2 last channel size
        resnet_dim    = 2048

        # ---- Classification head ----
        self.classifier = nn.Sequential(
            nn.Linear(mobilenet_dim + resnet_dim, hidden_dim),
            nn.ReLU(inplace=True),
            # nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        # MobileNet path
        m = self.mobilenet_features(x)
        m = self.mobilenet_pool(m)
        m = torch.flatten(m, 1)
        # ResNet path
        r = self.resnet_features(x)
        r = self.resnet_pool(r)
        r = torch.flatten(r, 1)
        # concat & classify
        out = self.classifier(torch.cat((m, r), dim=1))
        return out
