import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenet = models.mobilenet_v2(pretrained=True).features.eval().to(device)#평가모드적용

transform = transforms.Compose([
    transforms.ToPILImage(), # openCV의 이미지는 numpy 배열 -> torchvision은 PIL : 변환필요
    transforms.ToTensor(),# PIL 이미지를 [0,1] 범위의 PyTorch 텐서로 변환해야 모델이 처리가능
    transforms.Normalize([0.485, 0.456, 0.406],#정규화
                         [0.229, 0.224, 0.225])
])

@torch.no_grad()
def extract_cnn_features(sequence, device):
    tensors = torch.stack([transform(f) for f in sequence]).to(device)
    features = mobilenet(tensors) #(batch_size,1280,h,w)
    pooled = nn.AdaptiveAvgPool2d(1)(features)#(batch_size,1280,1,1) : h,w를 1*1크기로 (7*7을 평균내어 하나의 값으로 압축)
    return pooled.view(len(tensors), -1) # view로 (배치크기, 1280)-> h,w 없앰. 이미지별로 1280차원의 특징벡터 1개가 나옴
