# VIVID-Det 소스 파일 배치 가이드
 
RF-DETR git clone 후 VIVID-Det 모듈을 삽입하기 위한 파일 교체/생성 위치 안내
 
---
 
## 전제 조건
 
- conda 환경 `vividdet` 활성화 상태
- RF-DETR 소스가 개발 모드로 설치되어 있어야 한다
 
```bash
conda activate vividdet
pip uninstall rfdetr -y
git clone https://github.com/roboflow/rf-detr.git C:\gitnconda\Swin-Transformer\rf-detr
cd C:\gitnconda\Swin-Transformer\rf-detr
pip install -e .
```
 
`pip show rfdetr`의 `Editable project location`이 `C:\gitnconda\Swin-Transformer\rf-detr`을 가리키는지 확인한다.
 
---
 
## RF-DETR 원본 소스 구조
 
```
C:\gitnconda\Swin-Transformer\rf-detr\
└── src\rfdetr\
    ├── __init__.py
    ├── main.py
    ├── config.py                        ← 수정 대상
    ├── detr.py
    ├── engine.py
    ├── models\
    │   ├── lwdetr.py                    ← 수정 대상
    │   ├── matcher.py                   ← 수정 대상
    │   ├── transformer.py               ← 수정 대상
    │   ├── position_encoding.py
    │   ├── segmentation_head.py
    │   ├── backbone\
    │   │   ├── __init__.py              ← 수정 대상
    │   │   ├── backbone.py              ← 수정 대상
    │   │   ├── dinov2.py
    │   │   ├── dinov2_with_windowed_attn.py
    │   │   ├── projector.py
    │   │   └── base.py
    │   └── ops\
    ├── lit\
    │   ├── module.py
    │   ├── datamodule.py
    │   └── callbacks\
    ├── datasets\
    └── util\
        └── box_ops.py
```
 
---
 
## 파일 작업 총괄
 
### 수정 파일 (6개) -- 원본을 VIVID-Det 버전으로 교체
 
| 파일 경로 (rf-detr 루트 기준) | 버전 | 변경 내용 |
|---|---|---|
| `src/rfdetr/config.py` | - | `ModelConfig`에 `ssfa_enabled`, `tfcm_enabled` 등 VIVID-Det 플래그 추가 |
| `src/rfdetr/models/backbone/backbone.py` | v2.1.0 | `Backbone.__init__`에 SSFA/TFCM 모듈 등록, `forward`에 삽입 지점 A 구현 |
| `src/rfdetr/models/backbone/__init__.py` | v2.1.0 | SSFA/TFCM import 경로 등록 |
| `src/rfdetr/models/lwdetr.py` | - | `LWDETR.forward`에 SAQG 연결 (삽입 지점 B), `SetCriterion.loss_boxes`에 SQALB 연결 (삽입 지점 D), `build_criterion_and_postprocessors`에 `weight_dict` 확장 |
| `src/rfdetr/models/matcher.py` | - | `HungarianMatcher`에 `cost_nwd` 파라미터 추가, `forward`에 NWD cost 합산 (삽입 지점 C) |
| `src/rfdetr/models/transformer.py` | - | SOPM 전달을 위한 인터페이스 확장 |
 
### 신규 생성 파일 (5개) -- models 디렉토리에 추가
 
| 파일 경로 (rf-detr 루트 기준) | 모듈 | 역할 |
|---|---|---|
| `src/rfdetr/models/sawm.py` | SAWM | NWD 함수, TPAM topology regularization |
| `src/rfdetr/models/ssfa.py` | SSFA | CNN Branch, Cross-Attention, SOPM 생성 |
| `src/rfdetr/models/saqg.py` | SAQG | SOPM 기반 적응적 쿼리 밀도 배치 |
| `src/rfdetr/models/sqalb.py` | SQALB | 5D MLP, area-bin EMA 적응적 loss 밸런싱 |
| `src/rfdetr/models/tfcm.py` | TFCM | Feature Buffer, cosine similarity 시계열 융합 |
 
---
 
## 적용 후 소스 구조
 
```
C:\gitnconda\Swin-Transformer\rf-detr\
└── src\rfdetr\
    ├── __init__.py
    ├── main.py
    ├── config.py                        [수정됨] VIVID-Det 플래그 추가
    ├── detr.py
    ├── engine.py
    ├── models\
    │   ├── lwdetr.py                    [수정됨] SAQG/SQALB/weight_dict 연결
    │   ├── matcher.py                   [수정됨] SAWM NWD cost 추가
    │   ├── transformer.py               [수정됨] SOPM 전달 인터페이스
    │   ├── sawm.py                      [신규] SAWM 모듈
    │   ├── ssfa.py                      [신규] SSFA 모듈
    │   ├── saqg.py                      [신규] SAQG 모듈
    │   ├── sqalb.py                     [신규] SQALB 모듈
    │   ├── tfcm.py                      [신규] TFCM 모듈
    │   ├── position_encoding.py
    │   ├── segmentation_head.py
    │   ├── backbone\
    │   │   ├── __init__.py              [수정됨] import 경로 등록
    │   │   ├── backbone.py              [수정됨] SSFA/TFCM 삽입
    │   │   ├── dinov2.py
    │   │   ├── dinov2_with_windowed_attn.py
    │   │   ├── projector.py
    │   │   └── base.py
    │   └── ops\
    ├── lit\
    │   ├── module.py
    │   ├── datamodule.py
    │   └── callbacks\
    ├── datasets\
    └── util\
        └── box_ops.py
```
 
---
 
## 파일별 상세 배치 안내
 
### 1. config.py -- 수정 (교체)
 
경로: `src/rfdetr/config.py`
 
`ModelConfig` 데이터클래스에 VIVID-Det enable/disable 플래그를 추가한다. `RFDETRMedium(ssfa_enabled=True)` 형태로 모듈 활성화를 제어하는 진입점이다.
 
추가되는 주요 필드:
 
- `ssfa_enabled: bool = True`
- `saqg_enabled: bool = True`
- `sawm_enabled: bool = True`
- `sqalb_enabled: bool = True`
- `tfcm_enabled: bool = False` (Stage 1에서는 비활성)
- `set_cost_nwd: float = 0.0` (기본값 0으로 원본 동작 보존)
 
기존 config.py 전체를 VIVID-Det 버전으로 교체한다. `getattr(args, "attr_name", default)` 패턴으로 하위 호환성을 유지한다.
 
---
 
### 2. backbone.py -- 수정 (교체)
 
경로: `src/rfdetr/models/backbone/backbone.py` (v2.1.0)
 
원본 `Backbone.forward`의 encoder-projector 직결 구간에 SSFA와 TFCM을 삽입한다.
 
원본 흐름:
```
encoder → projector
```
 
변경 후 흐름:
```
encoder → [TFCM] → [SSFA] → projector
```
 
구체적으로 `forward` 내부에서:
 
```python
feats = self.encoder(tensor_list.tensors)       # DINOv2 출력
# --- VIVID-Det 삽입 지점 A ---
feats = self.tfcm(feats, ...)                   # 시계열 융합 (tfcm_enabled일 때)
feats, sopm = self.ssfa(feats, tensor_list.tensors)  # SSFA (ssfa_enabled일 때)
# ---
feats = self.projector(feats)                   # MultiScaleProjector
```
 
`__init__`에서 SSFA, TFCM 모듈 인스턴스를 등록한다. 비활성 시 identity로 동작하도록 분기 처리한다.
 
기존 backbone.py 전체를 VIVID-Det 버전으로 교체한다.
 
---
 
### 3. backbone/\_\_init\_\_.py -- 수정 (교체)
 
경로: `src/rfdetr/models/backbone/__init__.py` (v2.1.0)
 
SSFA, TFCM 모듈의 import 경로를 등록한다. backbone 패키지에서 모듈을 접근할 수 있도록 한다.
 
기존 \_\_init\_\_.py 전체를 VIVID-Det 버전으로 교체한다.
 
---
 
### 4. lwdetr.py -- 수정 (교체)
 
경로: `src/rfdetr/models/lwdetr.py`
 
3곳을 수정한다:
 
**(a) `LWDETR.forward` -- 삽입 지점 B (SAQG)**
 
```python
refpoint_embed_weight = self.refpoint_embed.weight
query_feat_weight = self.query_feat.weight
# --- VIVID-Det 삽입 ---
refpoint_embed_weight = self.saqg(refpoint_embed_weight, sopm)  # SOPM 기반 쿼리 재배치
# ---
hs, ref_unsigmoid, hs_enc, ref_enc = self.transformer(...)
```
 
**(b) `SetCriterion.loss_boxes` -- 삽입 지점 D (SQALB)**
 
원본의 고정 L1+GIoU loss를 SQALB의 적응적 NWD+WIoU 밸런싱으로 교체한다.
 
```python
alpha = self.sqalb(log_area, conf, iou_quality, layer_idx, matched_cost_ema)
loss_nwd = nwd_loss(src_boxes, target_boxes)
loss_wiou = wiou_loss(src_boxes, target_boxes)
losses["loss_bbox"] = (alpha * loss_nwd + (1 - alpha) * loss_wiou).sum() / num_boxes
```
 
**(c) `build_criterion_and_postprocessors` -- weight_dict 확장**
 
```python
weight_dict = {
    "loss_ce": args.cls_loss_coef,
    "loss_bbox": args.bbox_loss_coef,
    "loss_giou": args.giou_loss_coef,
    "loss_sopm": 0.5,                  # SOPM FocalLoss 감독 (신규)
}
```
 
기존 lwdetr.py 전체를 VIVID-Det 버전으로 교체한다.
 
---
 
### 5. matcher.py -- 수정 (교체)
 
경로: `src/rfdetr/models/matcher.py`
 
`HungarianMatcher`의 cost matrix에 NWD cost를 추가한다 (삽입 지점 C).
 
수정 항목:
- `__init__`: `cost_nwd` 파라미터 추가
- `forward`: NWD 계산 + cost matrix 합산
- `build_matcher`: args에서 `cost_nwd` 전달
 
```python
cost_nwd = 1 - nwd(out_bbox, tgt_bbox)
C = (self.cost_bbox * cost_bbox + self.cost_class * cost_class
     + self.cost_giou * cost_giou + self.cost_nwd * cost_nwd)
```
 
`set_cost_nwd=0` 기본값으로 원본 동작을 보존한다.
 
기존 matcher.py 전체를 VIVID-Det 버전으로 교체한다.
 
---
 
### 6. transformer.py -- 수정 (교체)
 
경로: `src/rfdetr/models/transformer.py`
 
SOPM 텐서를 backbone에서 decoder까지 전달하기 위한 인터페이스를 확장한다. Deformable Decoder의 forward 시그니처에 sopm 인자를 추가하고 내부에서 SAQG로 전달한다.
 
기존 transformer.py 전체를 VIVID-Det 버전으로 교체한다.
 
---
 
### 7. sawm.py -- 신규 생성
 
경로: `src/rfdetr/models/sawm.py`
 
NWD (Normalized Wasserstein Distance) 함수와 TPAM topology regularization을 구현한다.
 
- 학습 전용 모듈, 추론 비용 0%, 추가 파라미터 0
- `nwd()` 함수: matcher.py에서 호출
- `set_cost_nwd=0`이면 cost에 기여하지 않음 (원본 동작)
 
---
 
### 8. ssfa.py -- 신규 생성
 
경로: `src/rfdetr/models/ssfa.py`
 
SSFA (Small-object Selective Fusion Attention) 모듈 전체를 구현한다. 약 1.5M 파라미터.
 
내부 구성:
- CNN Branch: stride-8, GroupNorm(32), 고해상도 특징 추출
- Attention Prior: DINOv2 Block 4 Self-Attn Map에서 추출 (`encoder.encoder.encoder.layer[3].attention.attention` hook)
- Cross-Attention: alpha=0.01 초기화, 2D sinusoidal PE
- SOPM 생성: FocalLoss 감독, SAQG/TFCM에 전달
 
backbone.py의 `Backbone.__init__`에서 인스턴스화되고, `Backbone.forward`에서 호출된다.
 
---
 
### 9. saqg.py -- 신규 생성
 
경로: `src/rfdetr/models/saqg.py`
 
SAQG (Scale-Adaptive Query Generation) 모듈을 구현한다.
 
- SOPM 기반 적응적 쿼리 밀도 배치
- `ratio_mlp`의 gradient는 0 (detach + multinomial 설계상 정상)
- SSFA의 SOPM 출력에 의존
 
lwdetr.py의 `LWDETR.__init__`에서 인스턴스화되고, `LWDETR.forward`에서 호출된다.
 
---
 
### 10. sqalb.py -- 신규 생성
 
경로: `src/rfdetr/models/sqalb.py`
 
SQALB (Scale-Quality Aware Loss Balancer) 모듈을 구현한다.
 
- 학습 전용 모듈, 추론 비용 0%, 약 400 파라미터
- 5D MLP 입력: `[log(area), conf.detach(), iou_quality.detach(), layer_idx, matched_cost_ema]`
- area-bin EMA: 737 area-bin, resolution-aware, 3구간(small/medium/large) COCO 표준 경계
- SAWM의 NWD에 의존
 
lwdetr.py의 `SetCriterion.__init__`에서 인스턴스화되고, `SetCriterion.loss_boxes`에서 호출된다.
 
---
 
### 11. tfcm.py -- 신규 생성
 
경로: `src/rfdetr/models/tfcm.py`
 
TFCM (Temporal Feature Correspondence Module) 모듈을 구현한다.
 
- 학습 가능 파라미터 2개: `log_tau`, `alpha_raw`
- `alpha = sigmoid(-10) ≈ 0`으로 identity start
- `tau = 0.03` learnable temperature
- Feature Memory Buffer: FIFO, N=4
- DINOv2 cosine similarity → softmax(tau) → alpha 융합
- SOPM(t-1) Scale-Aware Mask 적용
- 이미지 모드: bypass (cost 0%)
- `tfcm_enabled=False` (Stage 1), `tfcm_enabled=True` (Stage 2)
 
backbone.py의 `Backbone.__init__`에서 인스턴스화되고, `Backbone.forward`에서 SSFA 이전에 호출된다.
 
---
 
## Forward Pass 삽입 지점 요약
 
```
LWDETR.forward(samples)
  |
  +-- Backbone.forward(tensor_list)
  |     |
  |     +-- encoder(tensor_list.tensors)          # DINOv2 ViT-S/14
  |     |     -> feats [B, 384, H, W]
  |     |
  |     |   [삽입 지점 A] backbone.py
  |     |   +-- TFCM(feats, buffer, sopm_cache)   # tfcm.py (Stage 2)
  |     |   +-- SSFA(feats, raw_images)            # ssfa.py -> sopm 생성
  |     |
  |     +-- projector(feats)                       # P3/P4/P5
  |           -> features, poss
  |
  |   [삽입 지점 B] lwdetr.py
  |   +-- SAQG(refpoint_embed, sopm)               # saqg.py
  |
  +-- transformer(srcs, masks, poss, refpoint, query)
  |     -> hs, ref_unsigmoid
  |
  +-- class_embed(hs), bbox_embed(hs)
  |     -> out = {pred_logits, pred_boxes}
  |
  +-- (학습 시) SetCriterion.forward(out, targets)
        |
        +-- matcher(out, targets)
        |     [삽입 지점 C] matcher.py
        |     +-- SAWM: NWD cost 추가               # sawm.py
        |     -> indices
        |
        +-- loss_boxes(out, targets, indices)
              [삽입 지점 D] lwdetr.py
              +-- SQALB: 적응적 NWD/WIoU 밸런싱      # sqalb.py
              -> losses
```
 
---
 
## 빠른 적용 절차
 
RF-DETR을 새로 클론한 후 VIVID-Det을 적용하는 전체 순서:
 
```bash
# 1. 소스 클론 및 개발 모드 설치
git clone https://github.com/roboflow/rf-detr.git C:\gitnconda\Swin-Transformer\rf-detr
cd C:\gitnconda\Swin-Transformer\rf-detr
pip install -e .
 
# 2. 수정 파일 교체 (6개)
#    원본 파일을 VIVID-Det 버전으로 덮어쓴다
copy /Y VIVID_src\config.py           src\rfdetr\config.py
copy /Y VIVID_src\backbone.py         src\rfdetr\models\backbone\backbone.py
copy /Y VIVID_src\backbone_init.py    src\rfdetr\models\backbone\__init__.py
copy /Y VIVID_src\lwdetr.py           src\rfdetr\models\lwdetr.py
copy /Y VIVID_src\matcher.py          src\rfdetr\models\matcher.py
copy /Y VIVID_src\transformer.py      src\rfdetr\models\transformer.py
 
# 3. 신규 파일 추가 (5개)
#    models 디렉토리에 복사한다
copy /Y VIVID_src\sawm.py             src\rfdetr\models\sawm.py
copy /Y VIVID_src\ssfa.py             src\rfdetr\models\ssfa.py
copy /Y VIVID_src\saqg.py             src\rfdetr\models\saqg.py
copy /Y VIVID_src\sqalb.py            src\rfdetr\models\sqalb.py
copy /Y VIVID_src\tfcm.py             src\rfdetr\models\tfcm.py
 
# 4. 개발 모드이므로 재설치 불필요 (소스 수정 즉시 반영)
 
# 5. import 검증
python -c "from rfdetr import RFDETRMedium; print('OK')"
```
 
위 예시에서 `VIVID_src\`는 VIVID-Det 수정/생성 파일이 보관된 로컬 디렉토리를 의미한다. 실제 경로에 맞게 조정한다.
 
---
 
## 모듈별 의존 관계
 
```
SAWM (sawm.py)           독립 -- matcher.py에서 호출
  |
SSFA (ssfa.py)            독립 -- backbone.py에서 호출, SOPM 생성
  |
  +-- SAQG (saqg.py)      SSFA의 SOPM에 의존 -- lwdetr.py에서 호출
  |
  +-- TFCM (tfcm.py)      SSFA의 SOPM(t-1)에 의존 -- backbone.py에서 호출
  |
SQALB (sqalb.py)          SAWM의 NWD에 의존 -- lwdetr.py에서 호출
```
 
---
 
## 학습 단계별 활성화 설정
 
| 플래그 | Stage 1 (이미지) | Stage 2 (비디오) |
|---|---|---|
| `ssfa_enabled` | True | True |
| `saqg_enabled` | True | True |
| `sawm_enabled` | True | True |
| `sqalb_enabled` | True | True |
| `tfcm_enabled` | **False** | **True** |
 
Stage 1에서는 `tfcm_enabled=False`로 4개 공간 모듈만 활성화한다. Stage 2에서 `tfcm_enabled=True`로 전환하여 시계열 모듈을 추가한다.
 
---
 
## 검증
 
파일 배치 후 7-test 통합 테스트로 모듈 동작을 확인한다:
 
```bash
python -c "
from rfdetr import RFDETRMedium
model = RFDETRMedium(ssfa_enabled=True)
print('모듈 로드 성공')
"
```
 
모든 enable/disable 플래그 조합에서 forward pass가 정상 동작하는지 확인한다. 특히 `ssfa_enabled=False` 시 원본 RF-DETR과 출력이 동일해야 한다 (bypass 동치 검증).
