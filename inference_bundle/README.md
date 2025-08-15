# Minimal Inference Bundle (CNN+LSTM Focus Classifier)

이 폴더는 백엔드/로컬에서 모델 추론을 돌리기 위한 최소 구성입니다.

## 구성 파일
- `requirements.txt` — 추론에 필요한 라이브러리
- `model.py` — CNN_LSTM 모델 구조 정의
- `preprocess.py` — 전처리 및 얼굴 오토줌
- `infer.py` — 폴더 단위 예측 실행
- `the_best.pth` — 학습된 가중치 (별도 제공)

## 실행 방법
```bash
pip install -r requirements.txt
python infer.py --ckpt the_best.pth --folder "C:/path/to/seq_folder"
