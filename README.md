<img width="239" height="45" alt="Image" src="https://github.com/user-attachments/assets/a75f96d7-fe6d-4071-86b5-778640ea3938" />

# SOD-DETR
### Transformer Based Small Object Detection with Selective Feature Fusion

군 경계 작전 환경에서의 소형 객체 탐지를 위한 RF-DETR 기반 프레임워크. **SCA** (Selective Cross-Attention) 모듈과 **NWD** (Normalized Wasserstein Distance) matching cost를 적용하여 소형 객체(특히 Person, Bird)의 탐지 정밀도를 개선한다.

---

## Highlights

- **SCA (Selective Cross-Attention)** — DINOv2 Block 4의 self-attention map을 사전 지식으로 활용하여, 경량 CNN 분기(stride-8)의 상위 25% 토큰만 ViT 특징에 교차 어텐션으로 융합 (논문 Section 3.1).
- **NWD (Normalized Wasserstein Distance)** — 헝가리안 정합 비용에서 GIoU를 NWD로 대체하여 소형 객체의 1~2px 오차로 IoU가 급변하는 문제를 완화 (논문 Section 3.2). 학습 시에만 동작 → 추론 비용 0%, 추가 파라미터 0개.
- **AGX Orin TensorRT FP16 배포** — Baseline 대비 latency overhead +7.0% (6.97 → 7.46 ms).

| Model | mAP@50:95 | Person AP | Total FP | FP 감소율 |
|-------|-----------|-----------|----------|-----------|
| RF-DETR-M (Baseline) | 0.902 | 0.673 | 121 | — |
| SOD-SCA | 0.904 | 0.704 | 83 | −31.4% |
| **SOD-SCA+NWD** | **0.906** | **0.713** | **74** | **−38.8%** |

전체 벤치마크 결과는 [benchmark/BENCHMARK_result.md](benchmark/BENCHMARK_result.md) 참조.

---

## Repository Structure

```
Defense_Quality_Research_Council/
├── Source_codes/
│   ├── modules/                  # SOD-DETR 핵심 모듈 (rfdetr/models/에 배치)
│   │   ├── sca.py                # SCA: Selective Cross-Attention (신규)
│   │   ├── nwd.py                # NWD: pairwise Wasserstein 거리 (신규)
│   │   ├── backbone.py           # SCA 통합 (원본 교체)
│   │   ├── lwdetr.py             # SCA/NWD 파라미터 전달 (원본 교체)
│   │   ├── matcher.py            # NWD matching cost (원본 교체)
│   │   └── SOD-DETR_소스파일_배치_가이드.md
│   └── train/                    # 학습 스크립트
│       ├── train_rfdetr_baseline.py
│       ├── train_sod_sca_only.py
│       ├── train_sod_sca_nwd.py
│       └── train_yolov8.py / train_yolov11.py / train_yolov12.py
├── RF-DETR/
│   ├── SOD/                      # SOD-DETR 학습 체크포인트 (Git LFS)
│   ├── rfdetr_m/                 # RF-DETR-M baseline 학습 로그
│   └── rfdetr_l/                 # RF-DETR-L baseline 학습 로그
├── YOLO/
│   ├── V8/  V11/  V12/           # YOLO 비교 학습 결과
│   ├── data.yaml
│   └── merge_yolo_format.py      # AI-HUB → YOLO 포맷 변환 스크립트
├── benchmark/
│   ├── BENCHMARK_result.md       # RTX 5090 / AGX Orin 벤치마크
│   └── figure_result_sod/
├── Documents/
│   ├── paper/                    # SOD-DETR.pdf, RF-DETR.pdf
│   └── data/                     # Figure.pptx, 논문유사도검사결과_확인서.pdf
└── requirements.txt
```

---

## Datasets

AI-HUB의 군 경계 작전 환경 데이터셋 2종을 사용한다. 10개 클래스 (Fishing_Boat, Merchant_Ship, Warship, Person, Bird, Fixed_Wing, Rotary_Wing, UAV, Leaflet, Trash_Bomb), 테스트 영상 20,172건, GT 인스턴스 41,472건.

| 데이터셋 | 다운로드 |
|---------|---------|
| 군 경계 작전 환경 내 인식 데이터 (실데이터) | [AI-HUB ID 71858](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EA%B5%B0%EA%B2%BD%EA%B3%84&aihubDataSe=data&dataSetSn=71858) |
| 군 경계 작전 환경 합성 데이터 | [AI-HUB ID 71856](https://aihub.or.kr/aihubdata/data/view.do?srchOptnCnd=OPTNCND001&currMenu=115&topMenu=100&searchKeyword=%EA%B5%B0%EA%B2%BD%EA%B3%84&aihubDataSe=data&dataSetSn=71856) |

---

## Installation

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) PyTorch (CUDA 12.8 환경 기준 — 본인 환경에 맞춰 변경)
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
```

### SOD-DETR 모듈 배치

`Source_codes/modules/` 내 파일들을 설치된 `rfdetr` 패키지의 `rfdetr/models/` 경로에 배치한다.

```bash
# rfdetr/models/ 경로 확인
python -c "import rfdetr, os; print(os.path.join(os.path.dirname(rfdetr.__file__), 'models'))"
```

| 파일 | 동작 |
|------|------|
| `sca.py`, `nwd.py` | **신규 추가** |
| `backbone.py`, `lwdetr.py`, `matcher.py` | **원본 덮어쓰기** |

원본 복원이 필요한 경우 `pip install --force-reinstall rfdetr==1.5.2`.

상세한 설명은 [Source_codes/modules/SOD-DETR_소스파일_배치_가이드.md](Source_codes/modules/SOD-DETR_%EC%86%8C%EC%8A%A4%ED%8C%8C%EC%9D%BC_%EB%B0%B0%EC%B9%98_%EA%B0%80%EC%9D%B4%EB%93%9C.md) 참조.

---

## Data Preparation

AI-HUB 원본 데이터를 YOLO 포맷으로 변환한다. (RF-DETR은 추가로 COCO 포맷 변환 필요)

```bash
# 1) merge_yolo_format.py 내부의 BASE_PATH / OUTPUT_DIR 경로 수정
# 2) 변환 실행
python YOLO/merge_yolo_format.py
```

---

## Training

학습 스크립트 내의 `dataset_dir`, `output_dir`, `pretrain_weights` 등 경로를 본인 환경에 맞춰 수정한 후 실행한다.

### RF-DETR Baseline

```bash
python Source_codes/train/train_rfdetr_baseline.py --size medium   # 또는 large
```

### SOD-DETR (SCA only)

```bash
python Source_codes/train/train_sod_sca_only.py
```

### SOD-DETR (SCA + NWD)

```bash
python Source_codes/train/train_sod_sca_nwd.py
```

### YOLO 비교군

```bash
python Source_codes/train/train_yolov8.py
python Source_codes/train/train_yolov11.py
python Source_codes/train/train_yolov12.py
```

---

## Benchmark Environment

| 항목 | 사양 |
|------|------|
| GPU | NVIDIA GeForce RTX 5090 32GB |
| 온디바이스 | NVIDIA Jetson AGX Orin 64GB (TensorRT FP16) |
| PyTorch | 2.10.0+cu128 |
| CUDA | 12.8 |
| 속도 측정 | warmup 50, 200 runs × 10 rounds, IQR 1.5× trimming, median of means |

---

## Documents

- 논문: [Documents/paper/SOD-DETR.pdf](Documents/paper/SOD-DETR.pdf)
- 참고 논문: [Documents/paper/RF-DETR.pdf](Documents/paper/RF-DETR.pdf)
- 벤치마크 상세: [benchmark/BENCHMARK_result.md](benchmark/BENCHMARK_result.md)

---

## Acknowledgements

본 코드는 다음 오픈소스 프로젝트를 기반으로 한다.

- [RF-DETR](https://github.com/roboflow/rf-detr) (Roboflow, Apache 2.0)
- [LW-DETR](https://github.com/Atten4Vis/LW-DETR) (Baidu)
- [DETR](https://github.com/facebookresearch/detr) (Facebook AI Research)
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- NWD: J. Wang et al., "A Normalized Gaussian Wasserstein Distance for Tiny Object Detection," CVPR 2022.
