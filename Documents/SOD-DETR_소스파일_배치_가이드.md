# SOD-DETR 소스파일 배치 가이드

SOD-DETR은 RF-DETR(rfdetr 1.5.2) 기반의 군 경계 감시 소형 객체 탐지 프레임워크이다.
SCA(Selective Cross-Attention)와 NWD(Normalized Wasserstein Distance) 두 개 모듈을 적용하여 소형 객체의 bbox 정밀도를 개선한다.

## 사전 요구사항

```bash
pip install rfdetr==1.5.2
```

설치 후 rfdetr 패키지 경로를 확인한다.

```bash
pip show rfdetr | grep Location
```

이하 출력된 경로를 `{SITE}`로 표기한다.

## 파일 배치

모든 파일은 `{SITE}/rfdetr/models/` 디렉토리에 배치한다.

| 파일 | 상태 | 설명 |
|------|------|------|
| `__init__.py` | 원본과 동일 | |
| `backbone.py` | 수정 | SCA 모듈 통합 (encoder→projector 간극) |
| `lwdetr.py` | 수정 | SCA 파라미터를 backbone에 전달 |
| `transformer.py` | 원본과 동일 | |
| `matcher.py` | 수정 | 헝가리안 정합 비용에 NWD 항 추가 |
| `nwd_matching.py` | 신규 | NWD pairwise 거리 계산 |
| `sca.py` | 신규 | SCA 전체 (CNN Branch + 히트맵 + Top-K + Cross-Attention) |

### 적용 명령어

```bash
SITE=$(python -c "import rfdetr; import os; print(os.path.dirname(rfdetr.__file__))")
TARGET="${SITE}/models"

# 원본 백업
cp "${TARGET}/backbone.py" "${TARGET}/backbone.py.orig"
cp "${TARGET}/lwdetr.py" "${TARGET}/lwdetr.py.orig"
cp "${TARGET}/matcher.py" "${TARGET}/matcher.py.orig"

# 파일 복사
cp __init__.py backbone.py lwdetr.py transformer.py matcher.py nwd_matching.py sca.py "${TARGET}/"
```

## 디렉토리 구조

```
rfdetr/models/
├── __init__.py          # 원본과 동일
├── backbone.py          # 수정: SCA 모듈 통합
├── lwdetr.py            # 수정: SCA 파라미터 전달
├── transformer.py       # 원본과 동일
├── matcher.py           # 수정: NWD 정합 비용 추가
├── nwd_matching.py      # 신규: NWD pairwise 계산
├── sca.py               # 신규: Selective Cross-Attention
├── position_encoding.py # 원본 (수정 없음)
├── segmentation_head.py # 원본 (수정 없음)
├── backbone/            # 원본 (수정 없음)
└── ops/                 # 원본 (수정 없음)
```

## 모듈 설명

### SCA (Selective Cross-Attention)

DINOv2 블록 4의 self-attention map을 사전 지식으로 활용하여, 경량 CNN 분기(stride-8)의 상위 25% 토큰만 선택적으로 ViT 특징에 교차 어텐션으로 융합한다. 구성 요소는 다음과 같다.

- CNN Branch: Conv3x3(stride 2) x3 + Conv3x3(stride 1) x1, GroupNorm, 출력 [B, 256, 72, 72]
- 소형 객체 히트맵: CNN 특징과 블록 4 어텐션 맵을 결합(Conv1x1 + Sigmoid)
- Top-K 선택: 히트맵 상위 25% 위치의 CNN 토큰만 선택
- 교차 어텐션: 선택된 CNN 토큰을 K/V, ViT 전체 토큰을 Q로 2D sinusoidal PE 적용
- 게이팅 잔차 연결: P3' = P3 + α_gate · CA(Q, K, V), α = 0.01 초기화

관련 파일: `sca.py`, `backbone.py`

### NWD (Normalized Wasserstein Distance)

헝가리안 정합의 비용 함수에서 GIoU를 NWD로 대체한다. bbox를 2D 가우시안 분포 N(cx, cy, w/2, h/2)로 모델링하여 Wasserstein-2 거리를 계산하고, 지수 함수로 0~1 범위의 유사도로 정규화한다. 소형 객체에서 1~2px 오차에도 IoU가 급변하는 문제를 완화하여 안정적인 GT 할당을 제공한다. 학습 시에만 동작하므로 추론 비용 0%, 추가 파라미터 0개이다.

관련 파일: `nwd_matching.py`, `matcher.py`

## 원본 복원

```bash
SITE=$(python -c "import rfdetr; import os; print(os.path.dirname(rfdetr.__file__))")
TARGET="${SITE}/models"

mv "${TARGET}/backbone.py.orig" "${TARGET}/backbone.py"
mv "${TARGET}/lwdetr.py.orig" "${TARGET}/lwdetr.py"
mv "${TARGET}/matcher.py.orig" "${TARGET}/matcher.py"
rm "${TARGET}/nwd_matching.py" "${TARGET}/sca.py"
```

또는 rfdetr을 재설치한다.

```bash
pip install --force-reinstall rfdetr==1.5.2
```
