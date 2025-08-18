# 프로젝트 명
BrainBuddyAI : Deep Learning Based Engagement Measuring Model (CNN → LSTM)

본 프로젝트는 **CNN → LSTM 구조**를 활용하여  
영상 데이터를 기반으로 **집중도**를 측정하는 모델을 구현하고,  
다양한 하이퍼파라미터 및 모델 구조 변경 실험을 통해 최적의 성능을 탐색하였습니다.
<br>

## 기술 스택
- 언어 & 환경
![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
	- Python 3.10.0

- 딥러닝 / 모델링
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?logo=pytorch&logoColor=white)
![Torchvision](https://img.shields.io/badge/Torchvision-%23EE4C2C.svg?logo=pytorch&logoColor=white)
	- PyTorch – 모델 구현 및 학습
	- Torchvision – CNN 백본 및 이미지 변환

- 컴퓨터 비전 / 전처리
![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?logo=opencv&logoColor=black)
![Mediapipe](https://img.shields.io/badge/Mediapipe-4285F4?logo=google&logoColor=white)
	- OpenCV – 영상 프레임 처리
	- Mediapipe FaceDetection – 얼굴 검출 및 크롭

- 평가 & 시각화
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E.svg?logo=scikit-learn&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243.svg?logo=plotly&logoColor=white)
![UMAP](https://img.shields.io/badge/UMAP--learn-5D3FD3.svg?logo=python&logoColor=white)
	- scikit-learn – 평가 지표 (F1, Recall, Confusion Matrix)
	- Matplotlib – 학습 곡선, 혼동 행렬, 시각화
	- UMAP-learn – 임베딩 차원 축소 및 시각화
<br>

## 📂 폴더 구조(예시)
project/ <br>
┣ data/ # 원본 및 가공 데이터 <br>
┣ notebooks/ # EDA, 실험용 Jupyter Notebook <br>
┣ src/ # 주요 Python 소스코드 <br>
┃ ┣ preprocessing.py # 데이터 전처리 코드 <br>
┃ ┣ modeling.py # 모델 정의/학습 코드 <br>
┃ ┣ train.py # 학습 스크립트 <br>
┃ ┗ evaluate.py # 평가 스크립트 <br>
┣ results/ # 결과 (모델 성능, 그래프, 로그)  <br>
┣ requirements.txt # 의존성 패키지  <br>
┗ README.md # 리드미 <br>
<br>

## 0. 모델 구조
<img src="https://private-user-images.githubusercontent.com/127655800/478826222-cb4224d7-ce55-48b6-bade-26b545e3e211.jpg?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3NTU0ODEzNzAsIm5iZiI6MTc1NTQ4MTA3MCwicGF0aCI6Ii8xMjc2NTU4MDAvNDc4ODI2MjIyLWNiNDIyNGQ3LWNlNTUtNDhiNi1iYWRlLTI2YjU0NWUzZTIxMS5qcGc_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjUwODE4JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI1MDgxOFQwMTM3NTBaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT1jYTlhYmE0ZjA4MDljMTEzNDNlM2UyNGFjMWM4ZDlhN2QyZWI1N2U2ZTVjODFjNWI1ZmM5ZTA2MDllOWU4NDU1JlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.SzVjXjGBm1bULG11dd999y5GfN_42wZ_5MkKIDm0unk" width="500" >
30프레임 시퀀스 -> CNN(MobileNetV3-Large) -> LSTM -> 집중여부(0/1)
<br>

## 1. 데이터 
### 사용 데이터셋
AIHub dataset : [학습태도 및 성향 관찰 데이터](https://www.aihub.or.kr/aihubdata/data/view.do?currMenu=115&topMenu=100&aihubDataSe=realm&dataSetSn=71715)
 <br>
### 전처리 및 라벨링
`python -m preprocess2.ext` mediapipe로 facecrop 후 10초에 30frame씩 추출

`python -m preprocess2.labeling` (폴더 경로, 라벨) 값을 .pkl 에 저장
<br>

## 2. 모델 학습
`python train.py`
- Epoch: 15
- Early Stopping patience: 4
- Batch size = 8
- Optimizer: AdamW
- Loss Function: BCEWithLogitsLoss + lr scheduler
- Gradient Accumulation = 32 step
<br>
 
## 3. 모델 테스트 및 성능
AIhub test set 성능 : 
자체 개발 test set 성능 : 
 <br>


## 4. 직접 실행해보기
1. "best_model.pt"를 다운로드
2. `real_time.py`의 CKPT_PATH에 해당 .pt 경로 지정
3. `python real_time.py` 
를 통해 로컬에서 실시간으로 테스트 해볼 수 있습니다.
 <br>


