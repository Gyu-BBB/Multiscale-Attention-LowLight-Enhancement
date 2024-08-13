# Multiscale-Attention-LowLight-Enhancement

### [실험 1] 모델 학습 및 성능 평가
이 코드는 제안된 멀티스케일 어텐션 네트워크를 학습시키고, 그 성능을 평가하기 위한 것이다.

```{python}
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pytorch_msssim import ssim, SSIM
from torchvision import transforms, datasets

# 데이터 증강 및 전처리
class RandomNoise:
    def __call__(self, img):
        noise = torch.randn_like(img) * 0.1
        return img + noise

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    RandomNoise(),
    transforms.Normalize((0.5,), (0.5,))
])

# 데이터셋 불러오기
train_dataset = datasets.ImageFolder(root='path/to/LOL_dataset/train', transform=transform)
test_dataset = datasets.ImageFolder(root='path/to/LOL_dataset/test', transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)

# 멀티스케일 어텐션 네트워크 정의
class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, padding=3)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=9, padding=4)
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        
    def forward(self, x):
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        return x1, x2, x3, x4

class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        self.bn = nn.BatchNorm2d(in_channels)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.max(x, dim=(2, 3), keepdim=True)[0]
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        out = x * torch.sigmoid(avg_out + max_out)
        return self.bn(out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.conv1(x)
        return self.bn(x * torch.sigmoid(x))

class MultiScaleAttentionNetwork(nn.Module):
    def __init__(self):
        super(MultiScaleAttentionNetwork, self).__init__()
        self.feature_extraction = MultiScaleFeatureExtraction()
        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(128)
        self.channel_attention3 = ChannelAttention(256)
        self.channel_attention4 = ChannelAttention(512)
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()
        self.spatial_attention4 = SpatialAttention()
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.upconv4 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1, x2, x3, x4 = self.feature_extraction(x)
        x1 = self.channel_attention1(x1) * self.spatial_attention1(x1)
        x2 = self.channel_attention2(x2) * self.spatial_attention2(x2)
        x3 = self.channel_attention3(x3) * self.spatial_attention3(x3)
        x4 = self.channel_attention4(x4) * self.spatial_attention4(x4)
        x = F.relu(self.upconv1(x4 + x3))
        x = F.relu(self.upconv2(x + x2))
        x = F.relu(self.upconv3(x + x1))
        x = self.upconv4(x)
        return torch.sigmoid(x)  # [0,1]로 스케일된 출력

# 모델, 손실 함수 및 옵티마이저 초기화
model = MultiScaleAttentionNetwork().to('cuda')
criterion = nn.L1Loss()
ssim_loss = SSIM(data_range=1.0, size_average=True, channel=3)
optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.999))

# 학습 루프
epochs = 100
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        
        optimizer.zero_grad()
        outputs = model(inputs)
        
        l1_loss = criterion(outputs, targets)
        ssim_val = ssim_loss(outputs, targets)
        loss = l1_loss + (1 - ssim_val)  # L1 손실과 SSIM 손실의 결합
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")
```

### [실험 2] PSNR 및 SSIM 평가
이 코드는 제안된 모델의 성능을 PSNR 및 SSIM 지표를 사용하여 평가한다.
```{python}
import torchvision.utils as vutils
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim

model.eval()
with torch.no_grad():
    psnr_values = []
    ssim_values = []
    
    for inputs, targets in test_loader:
        inputs, targets = inputs.to('cuda'), targets.to('cuda')
        outputs = model(inputs)
        
        for i in range(outputs.size(0)):
            output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
            target_img = targets[i].cpu().numpy().transpose(1, 2, 0)
            
            psnr_val = psnr(output_img, target_img)
            ssim_val = ssim(output_img, target_img, multichannel=True)
            
            psnr_values.append(psnr_val)
            ssim_values.append(ssim_val)

    print(f"Average PSNR: {sum(psnr_values) / len(psnr_values)}")
    print(f"Average SSIM: {sum(ssim_values) / len(ssim_values)}")
```

### [실험 3] 실시간 처리 및 객체 인식 통합
이 코드는 실시간으로 처리된 이미지를 통해 객체 인식 및 글자 인식(OCR) 성능을 평가한다.
```{python}
import cv2
import pytesseract

model.eval()
with torch.no_grad():
    for inputs in test_loader:
        inputs = inputs.to('cuda')
        outputs = model(inputs)
        
        for i in range(outputs.size(0)):
            output_img = outputs[i].cpu().numpy().transpose(1, 2, 0)
            enhanced_image = (output_img * 255).astype('uint8')
            gray_image = cv2.cvtColor(enhanced_image, cv2.COLOR_BGR2GRAY)
            text = pytesseract.image_to_string(gray_image)
            print(f"Detected Text: {text}")
```
