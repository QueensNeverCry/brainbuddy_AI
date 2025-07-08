import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

mobilenet = models.mobilenet_v2(pretrained=True).features.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_cnn_features(sequence):
    tensors = torch.stack([transform(f) for f in sequence])
    features = mobilenet(tensors)
    pooled = nn.AdaptiveAvgPool2d(1)(features)
    return pooled.view(len(tensors), -1)
