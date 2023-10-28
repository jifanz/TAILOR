import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights, ResNet50_Weights


def get_model_class(name):
    if name == "resnet18":
        return Resnet18


class Resnet18(nn.Module):
    def __init__(self, num_output, model=None, ret_emb=False, pretrained=True):
        super(Resnet18, self).__init__()
        if model is None:
            if pretrained:
                self.model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            else:
                self.model = models.resnet18()
        else:
            self.model = model
        self.model.fc = nn.Identity()
        self.linear = nn.Linear(512, num_output)
        self.ret_emb = ret_emb
        self.num_output = num_output

    def forward(self, imgs, ret_features=False, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                features = self.model(imgs)
        else:
            features = self.model(imgs)
        if ret_features:
            return self.linear(features), features.data
        elif self.ret_emb or last:
            return self.linear(features), features
        else:
            return self.linear(features)

    @staticmethod
    def get_embedding_dim():
        return 512
