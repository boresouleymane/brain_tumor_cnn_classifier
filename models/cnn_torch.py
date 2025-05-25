import torch.nn as nn
from torchvision import models

def get_pretrained_model():
    # Charger un ResNet18 pré-entraîné
    resnet = models.resnet18(weights='DEFAULT')

    # Geler tous les paramètres
    for param in resnet.parameters():
        param.requires_grad = False

    # Remplacer la dernière couche pour la classification binaire (2 classes)
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 4)  # Cancer du sein : bénin ou malin

    return resnet
