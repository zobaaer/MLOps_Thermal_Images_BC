import torch.nn as nn
from torchvision import models

# Define the ClassificationNet model
class ClassificationNet(nn.Module):
    def __init__(self, num_classes=2):  # Default to binary classification
        super(ClassificationNet, self).__init__()
        # Load pretrained ResNet18
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        # Replace the final fully connected layer
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)