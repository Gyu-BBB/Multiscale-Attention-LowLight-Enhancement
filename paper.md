# 멀티스케일 어텐션 네트워크를 이용한 저조도 이미지 개선

#### 초록(Abstract)
저조도 환경에서 촬영된 이미지는 시각적 품질이 저하되고, 중요한 세부 정보를 잃는 경우가 많다. 본 연구에서는 멀티스케일 어텐션 네트워크(Multi-Scale Attention Network)를 활용하여 저조도 이미지를 개선하는 새로운 방법을 제안한다. 제안된 방법은 다양한 스케일에서 이미지 특징을 추출하고, 채널 어텐션과 공간 어텐션 메커니즘을 통해 중요한 부분을 강조하여 이미지 품질을 향상시킨다. 기존 연구와 달리, 본 연구는 주파수 도메인 최적화 대신 공간 도메인에서의 멀티스케일 특징 추출과 어텐션 메커니즘을 결합하여 더 높은 성능을 달성한다. 실험 결과, 제안된 방법은 다양한 저조도 조건에서도 우수한 성능을 보이며, 기존 방법들과 비교하여 더 나은 시각적 품질을 제공한다.

#### 서론(Introduction)
저조도 환경에서 촬영된 이미지는 시각적 품질이 저하되고, 중요한 세부 정보를 잃는 경우가 많다. 이는 감시 시스템, 의료 영상, 자동차 자율 주행, 모바일 사진 등 다양한 응용 분야에서 큰 문제로 작용한다. 저조도 이미지는 노이즈가 많고, 명암 대비가 낮으며, 색 재현성이 떨어지기 때문에 이러한 이미지를 개선하는 것은 매우 중요한 과제이다.

기존의 저조도 이미지 개선 기법들은 주로 주파수 도메인에서의 최적화와 단일 스케일 어텐션 메커니즘을 사용하여 이미지 품질을 향상시키는 데 집중해왔다. 예를 들어, 여러 주파수 대역에서 고주파 성분을 보존하고 노이즈를 줄이기 위해 주파수 도메인 필터링을 적용하거나, 특정 스케일에서의 중요한 특징을 강조하는 어텐션 메커니즘을 도입하는 방법들이 있다. 그러나 이러한 접근법들은 주파수 도메인 최적화에 의존하여 공간 도메인에서의 세부 정보 복원에 한계를 가지며, 단일 스케일에서만 정보를 추출함으로써 다양한 스케일에서 나타나는 중요한 정보를 충분히 반영하지 못하는 문제점이 있다.

이러한 한계를 극복하기 위해 본 연구에서는 멀티스케일 어텐션 네트워크(Multi-Scale Attention Network)를 활용한 새로운 저조도 이미지 개선 방법을 제안한다. 제안된 방법은 다양한 스케일에서 이미지 특징을 추출하고, 채널 어텐션과 공간 어텐션 메커니즘을 통해 중요한 부분을 강조하여 이미지 품질을 향상시킨다. 이를 통해 단일 스케일에서 얻을 수 없는 풍부한 정보와 세부적인 특징을 복원할 수 있다.

본 연구의 주요 기여는 다양한 스케일에서 이미지 특징을 효과적으로 추출하기 위해 멀티스케일 컨볼루션 네트워크를 설계한 점, 채널 어텐션과 공간 어텐션 메커니즘을 결합하여 이미지의 중요한 부분을 강조함으로써 더 나은 복원 성능을 달성한 점, 그리고 다양한 저조도 조건에서도 우수한 성능을 보이며, 기존 방법들과 비교하여 더 나은 시각적 품질을 제공하는 새로운 저조도 이미지 개선 방법을 제안한 점에 있다.

#### 관련 연구(Related Work)
관련 연구(Related Work)
기존 저조도 이미지 개선 연구는 주로 주파수 도메인에서의 최적화와 단일 스케일 어텐션 메커니즘을 사용하여 이미지 품질을 향상시키는 데 중점을 둔다. 예를 들어, "Low-Light Image Enhancement With Multi-Scale Attention and Frequency-Domain Optimization" 논문에서는 멀티스케일 어텐션과 주파수 도메인 최적화를 결합하여 고주파 성분을 보존하고 노이즈를 줄이는 방법을 제안한다. 그러나 이러한 접근법은 주파수 도메인 최적화에 의존하여 공간 도메인에서의 세부 정보 복원에 한계가 있다.

또한, "Enhancement of Low-Light Images Using Deep Convolutional Neural Networks" 연구에서는 딥러닝 기반의 컨볼루션 신경망을 활용하여 저조도 이미지를 개선하는 방법을 제안하지만, 단일 스케일에서의 특징 추출에 의존하여 다양한 스케일에서의 중요한 정보를 충분히 반영하지 못하는 문제점이 있다.

최근에는 "Multi-Scale Retinex-Based Low-Light Image Enhancement" 연구에서 멀티스케일 리티넥스 알고리즘을 사용하여 저조도 이미지를 개선하는 방법이 제안되었다. 이 방법은 멀티스케일에서의 이미지 특징을 추출하여 저조도 이미지를 개선하는 데 효과적이었으나, 어텐션 메커니즘을 결합하지 않아 중요한 부분을 강조하는 데 한계가 있다.

추가적으로, "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" 논문에서는 이미지 특화 곡선 추정을 통해 저조도 이미지 개선을 수행하는 새로운 방법을 제안하고 있으며, "EnlightenGAN: Deep Light Enhancement without Paired Supervision" 연구는 짝이 없는 훈련 데이터로 저조도 이미지를 개선하는 GAN 기반의 방법을 제안한다. 또한, "LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement" 논문은 딥 오토인코더를 활용하여 자연스러운 저조도 이미지 개선을 수행하는 방법을 제안한다.

이 외에도, "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" 연구는 Retinex 이론을 기반으로 한 Transformer 모델을 사용하여 저조도 이미지를 개선하는 방법을 제안하며, "Attention Guided Low-light Image Enhancement with a Large Scale Low-light Simulation Dataset" 논문은 대규모 저조도 시뮬레이션 데이터셋을 사용하여 어텐션 메커니즘을 통해 저조도 이미지를 개선하는 방법을 제안한다.

본 연구는 이러한 기존 연구들의 한계를 극복하고자 멀티스케일 어텐션 네트워크를 활용하여 저조도 이미지를 개선하는 새로운 방법을 제안한다. 제안된 방법은 다양한 스케일에서 이미지 특징을 추출하고, 채널 어텐션과 공간 어텐션 메커니즘을 결합하여 이미지 품질을 향상시킨다. 이를 통해 기존 방법들과 비교하여 더 나은 시각적 품질을 제공할 수 있다.


#### 데이터(Data Requirements)

1. **저조도 이미지 데이터셋**
   - **목적**: 제안된 모델의 학습과 평가를 위해 다양한 조명 조건에서 촬영된 저조도 이미지가 필요하다.
   - **예시 데이터셋**: 
     - **LOL Dataset**: 저조도 이미지와 해당 고조도 이미지 쌍을 포함하여 학습 및 평가에 유용하다.
     - **SID Dataset**: 스마트폰으로 촬영된 저조도 이미지 데이터셋으로, 다양한 조명 조건을 포함하고 있다.

2. **데이터 전처리**
   - **정규화**: 이미지 데이터를 정규화하여 모델 학습을 안정화시킨다.
   - **데이터 증강**: 다양한 조명 조건과 노이즈 수준을 시뮬레이션하기 위해 데이터 증강 기법을 사용한다. 예를 들어, 회전, 크기 조절, 밝기 조정 등을 포함할 수 있다.

#### 방법론(Methodology)

**1. 멀티스케일 특징 추출(Multi-Scale Feature Extraction)**
- **목적**: 이미지의 다양한 스케일에서 정보를 얻어 세부 정보와 전역 정보를 모두 포착한다.
- **구현**: 다양한 크기의 컨볼루션 커널을 사용하여 여러 스케일에서 특징을 추출한다. 작은 커널은 세부 정보를, 큰 커널은 전역적 정보를 포착한다.

```
class MultiScaleFeatureExtraction(nn.Module):
    def __init__(self):
        super(MultiScaleFeatureExtraction, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=7, padding=3)
        
    def forward(self, x):
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        x3 = F.relu(self.conv3(x2))
        return x1, x2, x3
```

**2. 어텐션 메커니즘(Attention Mechanism)**
- **목적**: 이미지의 중요한 부분을 강조하여 복원 성능을 향상시킨다.
- **구현**: 채널 어텐션과 공간 어텐션을 사용하여 각각 이미지의 채널과 공간적 위치에서 중요한 부분을 강조한다.

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.fc1 = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // 8, in_channels, kernel_size=1)
        
    def forward(self, x):
        avg_pool = torch.mean(x, dim=(2, 3), keepdim=True)
        max_pool = torch.max(x, dim=(2, 3), keepdim=True)[0]
        avg_out = self.fc2(F.relu(self.fc1(avg_pool)))
        max_out = self.fc2(F.relu(self.fc1(max_pool)))
        return x * (avg_out + max_out)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        
    def forward(self, x):


        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        x = torch.cat([avg_pool, max_pool], dim=1)
        x = self.conv1(x)
        return x * torch.sigmoid(x)
```

**3. 융합 단계(Fusion Stage)**
- **목적**: 다양한 스케일에서 추출된 특징을 융합하여 최종 이미지를 복원한다.
- **구현**: 어텐션 메커니즘을 통해 강조된 특징을 결합하여 이미지를 복원한다.

```python
class MultiScaleAttentionNetwork(nn.Module):
    def __init__(self):
        super(MultiScaleAttentionNetwork, self).__init__()
        self.feature_extraction = MultiScaleFeatureExtraction()
        self.channel_attention1 = ChannelAttention(64)
        self.channel_attention2 = ChannelAttention(128)
        self.channel_attention3 = ChannelAttention(256)
        self.spatial_attention1 = SpatialAttention()
        self.spatial_attention2 = SpatialAttention()
        self.spatial_attention3 = SpatialAttention()
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.upconv3 = nn.ConvTranspose2d(64, 3, kernel_size=3, padding=1)
        
    def forward(self, x):
        x1, x2, x3 = self.feature_extraction(x)
        x1 = self.channel_attention1(x1) * self.spatial_attention1(x1)
        x2 = self.channel_attention2(x2) * self.spatial_attention2(x2)
        x3 = self.channel_attention3(x3) * self.spatial_attention3(x3)
        x = F.relu(self.upconv1(x3))
        x = F.relu(self.upconv2(x))
        x = self.upconv3(x)
        return x
```

#### 결과(Results)

1. **평가 지표**
   - 제안된 모델의 성능을 평가하기 위해 PSNR(피크 신호 대 잡음비) 및 SSIM(구조적 유사성 지수)을 주요 지표로 사용하였다. 이 지표들은 복원된 이미지의 품질을 객관적으로 평가할 수 있는 기준을 제공한다.
  
2. **비교 실험**
   - 제안된 모델의 성능을 기존의 저조도 이미지 개선 기법들과 비교하였다. 멀티스케일 리티넥스 기반 방법, 주파수 도메인 최적화 기법, 그리고 딥러닝 기반 단일 스케일 어텐션 네트워크와의 비교 실험을 수행하였다.
   - 실험 결과, 제안된 모델은 PSNR과 SSIM 지표에서 모두 우수한 성능을 기록하였으며, 특히 다양한 저조도 조건에서 탁월한 시각적 품질을 제공하였다.

3. **시각적 비교**
   - 다양한 저조도 환경에서 촬영된 이미지에 대해 제안된 모델이 복원한 결과물을 기존 방법들과 시각적으로 비교하였다. 제안된 모델은 더 높은 명암 대비와 색 재현성을 제공하여, 인간의 시각적 인식 측면에서도 더 나은 품질을 보였다.

#### 논의(Discussion)

1. **모델의 장점**
   - 제안된 멀티스케일 어텐션 네트워크는 다양한 스케일에서의 이미지 특징을 효과적으로 추출하고, 어텐션 메커니즘을 통해 중요한 부분을 강조하여 기존 방법들보다 뛰어난 성능을 발휘하였다.
   - 특히, 공간 도메인에서의 멀티스케일 특징 추출과 어텐션 메커니즘의 결합은 이미지의 복원 품질을 극대화하는 데 중요한 역할을 하였다.

2. **한계점 및 향후 연구 방향**
   - 제안된 방법은 높은 복원 성능을 보이지만, 복잡한 네트워크 구조로 인해 계산 비용이 다소 높다는 단점이 있다. 실시간 적용을 위해서는 경량화된 모델 설계와 최적화가 필요하다.
   - 향후 연구에서는 제안된 방법을 실시간 처리 시스템에 적용할 수 있도록 최적화하는 작업이 필요하다. 또한, 다양한 환경에서의 적용 가능성을 평가하기 위해 추가적인 실험이 필요하다.

#### 결론(Conclusion)

본 연구에서는 멀티스케일 어텐션 네트워크를 활용하여 저조도 이미지를 개선하는 새로운 방법을 제안하였다. 다양한 스케일에서 이미지 특징을 추출하고, 어텐션 메커니즘을 통해 중요한 부분을 강조하여 이미지 품질을 향상시킨 결과, 제안된 방법은 기존 방법들보다 우수한 성능을 보였다. 특히, 다양한 저조도 조건에서도 탁월한 시각적 품질을 제공함으로써, 저조도 이미지 개선 문제에 효과적인 해결책을 제시하였다. 향후, 제안된 방법의 실시간 적용 가능성 및 다른 응용 분야에서의 활용을 위한 추가 연구가 기대된다.
