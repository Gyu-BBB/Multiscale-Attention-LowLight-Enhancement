# 멀티스케일 어텐션 네트워크를 이용한 저조도 이미지 개선

## 1. 서론

저조도 환경에서 촬영된 이미지는 시각적 품질이 저하되고 중요한 세부 정보를 잃는 경우가 많다. 이는 감시 시스템, 의료 영상, 자동차 자율 주행, 모바일 사진 등 다양한 응용 분야에서 큰 문제로 작용한다. 특히, 감시 시스템에서 저조도 이미지의 품질 저하는 CCTV 영상에서 자동차 번호판과 차종, 사람의 나이, 성별, 장애 여부 등을 정확히 인식하는 데 큰 어려움을 초래할 수 있다. 이러한 문제를 해결하기 위해 저조도 이미지 개선 기술이 필수적이다. 본 연구에서는 멀티스케일 어텐션 네트워크(Multi-Scale Attention Network)를 활용하여 저조도 이미지를 개선하는 새로운 방법을 제안한다. 제안된 방법은 다양한 스케일에서 이미지 특징을 추출하고, 채널 어텐션과 공간 어텐션 메커니즘을 통해 중요한 부분을 강조하여 이미지 품질을 향상시킨다.

## 2. 관련 연구

기존 저조도 이미지 개선 연구는 주로 주파수 도메인에서의 최적화와 단일 스케일 어텐션 메커니즘을 사용하여 이미지 품질을 향상시키는 데 중점을 둔다. 예를 들어, "Low-Light Image Enhancement With Multi-Scale Attention and Frequency-Domain Optimization" 논문에서는 멀티스케일 어텐션과 주파수 도메인 최적화를 결합하여 고주파 성분을 보존하고 노이즈를 줄이는 방법을 제안한다. 그러나 이러한 접근법은 주파수 도메인 최적화에 의존하여 공간 도메인에서의 세부 정보 복원에 한계가 있다.

또한, "Enhancement of Low-Light Images Using Deep Convolutional Neural Networks" 연구에서는 딥러닝 기반의 컨볼루션 신경망을 활용하여 저조도 이미지를 개선하는 방법을 제안하지만, 단일 스케일에서의 특징 추출에 의존하여 다양한 스케일에서의 중요한 정보를 충분히 반영하지 못하는 문제점이 있다.

최근에는 "Multi-Scale Retinex-Based Low-Light Image Enhancement" 연구에서 멀티스케일 리티넥스 알고리즘을 사용하여 저조도 이미지를 개선하는 방법이 제안되었다. 이 방법은 멀티스케일에서의 이미지 특징을 추출하여 저조도 이미지를 개선하는 데 효과적이었으나, 어텐션 메커니즘을 결합하지 않아 중요한 부분을 강조하는 데 한계가 있다.

추가적으로, "Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement" 논문에서는 이미지 특화 곡선 추정을 통해 저조도 이미지 개선을 수행하는 새로운 방법을 제안하고 있으며, "EnlightenGAN: Deep Light Enhancement without Paired Supervision" 연구는 짝이 없는 훈련 데이터로 저조도 이미지를 개선하는 GAN 기반의 방법을 제안한다. 또한, "LLNet: A Deep Autoencoder Approach to Natural Low-light Image Enhancement" 논문은 딥 오토인코더를 활용하여 자연스러운 저조도 이미지 개선을 수행하는 방법을 제안한다.

이 외에도, "Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement" 연구는 Retinex 이론을 기반으로 한 Transformer 모델을 사용하여 저조도 이미지를 개선하는 방법을 제안하며, "Attention Guided Low-light Image Enhancement with a Large Scale Low-light Simulation Dataset" 논문은 대규모 저조도 시뮬레이션 데이터셋을 사용하여 어텐션 메커니즘을 통해 저조도 이미지를 개선하는 방법을 제안한다.

본 연구는 이러한 기존 연구들의 한계를 극복하고자 멀티스케일 어텐션 네트워크를 활용하여 저조도 이미지를 개선하는 새로운 방법을 제안한다. 제안된 방법은 다양한 스케일에서 이미지 특징을 추출하고, 채널 어텐션과 공간 어텐션 메커니즘을 결합하여 이미지 품질을 향상시킨다. 이를 통해 기존 방법들과 비교하여 더 나은 시각적 품질을 제공할 수 있다. 또한, 감시 시스템의 CCTV 영상에서 자동차 번호판과 사람의 특징을 더 정확하게 인식할 수 있는 가능성을 열어준다.

## 3. 제안 모델

본 연구에서 제안한 멀티스케일 어텐션 네트워크는 저조도 이미지의 복원 및 개선을 위해 설계된 심층 학습 모델로, 다음의 세 가지 주요 모듈로 구성된다: **멀티스케일 특징 추출 모듈**, **채널 어텐션 메커니즘**, 그리고 **공간 어텐션 메커니즘**.

### 3.1 멀티스케일 특징 추출

멀티스케일 특징 추출 모듈은 다양한 크기의 컨볼루션 커널을 사용하여 이미지의 다양한 스케일에서 정보를 추출한다. 이 모듈의 주요 목표는 이미지의 세부 정보와 전역 정보를 모두 포착하는 것이다. 

- **컨볼루션 레이어**: 여러 크기의 컨볼루션 커널을 사용하여 다양한 스케일에서의 특징을 추출한다. 작은 커널은 이미지의 미세한 세부 정보를 포착하는 데 사용되며, 큰 커널은 이미지의 전역적 특성을 포착한다. 이로 인해, 각 스케일에서 얻어진 특징이 통합되어 이미지의 복원 과정에서 중요한 역할을 한다.

- **배치 정규화**: 각 컨볼루션 레이어 후에 배치 정규화를 적용하여 학습을 안정화시키고, 더 나은 일반화 성능을 얻을 수 있다.

- **활성화 함수**: 비선형성을 도입하기 위해 ReLU 활성화 함수를 사용한다. 이 함수는 네트워크의 표현력을 높이고, 복잡한 패턴을 학습하는 데 기여한다.

### 3.2 채널 어텐션 메커니즘

채널 어텐션 메커니즘은 입력 이미지의 각 채널이 가지는 중요도를 평가하고, 중요한 채널에 더 높은 가중치를 부여하여 이미지의 특정 채널에서 더 많은 정보를 추출할 수 있도록 한다.

- **전역 평균 풀링**: 입력 이미지의 각 채널에 대해 전역 평균 풀링을 수행하여 채널별로 평균값을 구한다. 이를 통해 각 채널의 전체적인 중요도를 파악할 수 있다.

- **전역 최대 풀링**: 전역 평균 풀링과 함께 전역 최대 풀링도 수행하여 각 채널의 최대값을 구한다. 이 두 값을 결합하여 채널의 중요도를 더욱 명확하게 결정할 수 있다.

- **채널 어텐션 계산**: 풀링 결과를 바탕으로 작은 FC (Fully Connected) 레이어를 통과시켜 각 채널의 중요도를 계산하고, Sigmoid 함수를 적용하여 0과 1 사이의 값으로 정규화한다. 이를 통해 각 채널의 중요도를 반영한 채널 어텐션 맵을 생성한다.

### 3.3 공간 어텐션 메커니즘

공간 어텐션 메커니즘은 이미지의 공간적 위치에서 중요한 부분을 강조하는 역할을 한다. 이 메커니즘은 이미지 내의 특정 위치나 영역이 다른 위치에 비해 더 중요한 정보를 포함하고 있을 때 그 위치를 강조하는 데 사용된다.

- **채널 축을 따라 평균 풀링 및 최대 풀링**: 각 채널에 대해 공간적으로 평균 풀링과 최대 풀링을 수행하여 두 개의 2D 맵을 생성한다. 이 맵들은 이미지의 공간적 중요도를 나타낸다.

- **공간 어텐션 계산**: 평균 풀링 맵과 최대 풀링 맵을 결합하여 2D 공간 어텐션 맵을 생성한다. 이 맵은 이미지의 각 위치에 대해 중요도를 할당하며, Sigmoid 함수를 통해 정규화된 값을 얻는다.

- **최종 출력**: 생성된 공간 어텐션 맵은 원래의 이미지와 곱해져 중요한 공간적 정보를 강조한다. 이를 통해 복원된 이미지에서 중요한 위치의 정보가 더 명확하게 나타나도록 한다.

### 3.4 융합 단계

최종적으로, 멀티스케일 특징 추출 모듈에서 얻어진 특징 맵들과 채널 어텐션, 공간 어텐션 메커니즘에서 강조된 특징들이 융합된다. 이 단계에서는 저해상도에서 고해상도로 변환하기 위해 업샘플링 레이어를 사용하며, 각 단계에서 얻어진 특징 맵들이 결합되어 최종 복원된 이미지를 생성한다.

- **업샘플링 레이어**: 복원된 이미지의 해상도를 높이기 위해 여러 단계의 업샘플링을 수행한다. 각 업샘플링 레이어는 이전 단계의 출력과 멀

티스케일 특징 맵을 결합하여 보다 풍부한 정보를 포함한 고해상도 이미지를 생성한다.

- **결합 연산**: 채널 어텐션 및 공간 어텐션 메커니즘을 통해 강조된 특징 맵들이 업샘플링된 이미지와 결합된다. 이 과정을 통해 최종 출력 이미지는 중요한 정보가 더욱 명확하게 복원된 형태로 생성된다.

이러한 구조를 통해 제안된 멀티스케일 어텐션 네트워크는 저조도 이미지에서 중요한 세부 정보를 효과적으로 복원하고, 다양한 응용 분야에서 유용한 고품질 이미지를 제공할 수 있다.

## 4. 실험결과

### 4.1 데이터 수집

실험에는 저조도 이미지 개선을 위해 LOL 데이터셋과 SID 데이터셋을 사용하였다. 이 데이터셋들은 다양한 조명 조건에서 촬영된 저조도 이미지와 해당 고조도 이미지를 포함하고 있어, 제안된 모델의 학습과 평가에 적합하다. 데이터 증강 기법을 통해 다양한 조명 조건과 노이즈 수준을 시뮬레이션하여 모델의 일반화 성능을 향상시켰다.

### 4.2 실험 환경

실험은 NVIDIA GPU가 장착된 환경에서 PyTorch 프레임워크를 사용하여 수행되었다. 학습 과정에서는 L1 손실 함수와 SSIM 손실 함수를 결합하여 사용하였으며, Adam 옵티마이저를 통해 모델의 파라미터를 학습시켰다. [실험1]에서는 모델 학습과 관련된 실험을 진행하였다.

### 4.3 실험결과

**[실험2]**에서는 제안된 멀티스케일 어텐션 네트워크의 성능을 PSNR 및 SSIM 지표로 평가하였다. 실험 결과, 제안된 모델은 기존 방법들에 비해 더 높은 PSNR과 SSIM 값을 기록하였으며, 특히 다양한 저조도 조건에서 탁월한 시각적 품질을 제공하였다. 이로 인해 감시 시스템의 CCTV 영상에서 자동차 번호판과 사람의 특징을 더욱 정확하게 인식할 수 있는 가능성이 확인되었다.

또한, **[실험3]**에서는 제안된 모델이 실시간으로 처리 가능한지, 그리고 개선된 이미지를 사용하여 객체 인식 및 글자 인식(OCR)에서 성능이 향상되는지를 평가하였다. 이 실험을 통해 실시간 응용 가능성도 확인할 수 있었다.

## 5. Discussion & Future Work

본 연구에서 제안된 멀티스케일 어텐션 네트워크는 저조도 이미지 개선에 있어 매우 유망한 성능을 보였다. 특히, 감시 시스템에서의 응용 가능성이 높아 범죄 예방과 수사에서 중요한 역할을 할 수 있다. 그러나 제안된 모델은 높은 복원 성능을 보이는 반면, 복잡한 네트워크 구조로 인해 계산 비용이 다소 높다는 단점이 있다. 향후 연구에서는 이러한 네트워크를 경량화하여 실시간 적용 가능성을 높이는 것이 필요하다. 또한, 다양한 감시 환경에서의 적용 가능성을 평가하기 위해 추가적인 실험이 필요하다.
