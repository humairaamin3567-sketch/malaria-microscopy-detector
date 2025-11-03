import torch.nn as nn
import torchvision.models as models

def get_model(name='resnet50', num_classes=2, pretrained=True, dropout_p=0.3):
    if name.startswith('resnet'):
        model = models.resnet50(pretrained=pretrained)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
    elif name.startswith('efficientnet'):
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(dropout_p),
            nn.Linear(in_features, num_classes)
        )
    else:
        raise ValueError('Unknown model')
    return model
