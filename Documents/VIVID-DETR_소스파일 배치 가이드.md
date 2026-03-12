# VIVID-Det 소스 파일 배치 가이드
 
RF-DETR git clone 후 VIVID-Det 모듈을 삽입하기 위한 파일 교체/생성/패치 위치 안내
 
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
    ├── main.py                        ← 패치 대상 (patch_main_sawm.py)
    ├── config.py                      ← 수동 수정 대상
    ├── detr.py
    ├── engine.py
    ├── models\
    │   ├── lwdetr.py                  ← 패치 대상 (patch_saqg.py, patch_sqalb.py)
    │   ├── matcher.py                 ← 교체 대상
    │   ├── transformer.py             ← 패치 대상 (patch_saqg.py)
    │   ├── position_encoding.py
    │   ├── segmentation_head.py
    │   ├── backbone\
    │   │   ├── __init__.py            ← 교체 대상
    │   │   ├── backbone.py            ← 교체 대상
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
 
VIVID-Det 적용은 교체, 신규 생성, 패치 스크립트, 수동 수정의 4가지 방식으로 이루어진다.
 
### 교체 파일 (3개) -- 원본을 VIVID-Det 버전으로 덮어쓴다
 
| 파일 | 대상 경로 (rf-detr 루트 기준) | 버전 |
|---|---|---|
| `backbone.py` | `src/rfdetr/models/backbone/backbone.py` | v2.1.0 |
| `__init__.py` | `src/rfdetr/models/backbone/__init__.py` | v2.1.0 |
| `matcher.py` | `src/rfdetr/models/matcher.py` | - |
 
### 신규 생성 파일 (5개) -- models 디렉토리에 추가
 
| 파일 | 대상 경로 (rf-detr 루트 기준) | 상태 |
|---|---|---|
| `sawm.py` | `src/rfdetr/models/sawm.py` | 확보 완료 |
| `ssfa.py` | `src/rfdetr/models/ssfa.py` | 확보 완료 |
| `saqg.py` | `src/rfdetr/models/saqg.py` | 확보 완료 |
| `sqalb.py` | `src/rfdetr/models/sqalb.py` | **미확보 -- 작성 필요** |
| `tfcm.py` | `src/rfdetr/models/tfcm.py` | 확보 완료 |
 
### 패치 스크립트 (3개) -- 원본 파일을 부분 수정한다
 
| 스크립트 | 패치 대상 | 동작 |
|---|---|---|
| `patch_main_sawm.py` | `main.py` | `set_cost_nwd`, `nwd_C` 파라미터 4곳 삽입 |
| `patch_saqg.py` | `lwdetr.py`, `transformer.py` | SAQG import/init/forward + transformer dim 체크 |
| `patch_sqalb.py` | `lwdetr.py` | SQALB import/init/loss_boxes 분기/build 연결 |
 
### 수동 수정 (1개) -- 패치 스크립트 미존재
 
| 파일 | 대상 경로 | 수정 내용 |
|---|---|---|
| `config.py` | `src/rfdetr/config.py` | `ModelConfig`에 enable 플래그 추가 |
 
---
 
## 적용 후 소스 구조
 
```
C:\gitnconda\Swin-Transformer\rf-detr\
└── src\rfdetr\
    ├── __init__.py
    ├── main.py                        [패치됨] SAWM 파라미터 추가
    ├── config.py                      [수동 수정] enable 플래그 추가
    ├── detr.py
    ├── engine.py
    ├── models\
    │   ├── lwdetr.py                  [패치됨] SAQG/SQALB 연결
    │   ├── matcher.py                 [교체됨] SAWM NWD cost 추가
    │   ├── transformer.py             [패치됨] SAQG dim 체크
    │   ├── sawm.py                    [신규] SAWM 모듈
    │   ├── ssfa.py                    [신규] SSFA 모듈
    │   ├── saqg.py                    [신규] SAQG 모듈
    │   ├── sqalb.py                   [신규] SQALB 모듈 ← 미확보
    │   ├── tfcm.py                    [신규] TFCM 모듈
    │   ├── position_encoding.py
    │   ├── segmentation_head.py
    │   ├── backbone\
    │   │   ├── __init__.py            [교체됨] SSFA/TFCM 파라미터 전달
    │   │   ├── backbone.py            [교체됨] SSFA/TFCM 삽입
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
 
## 적용 절차
 
RF-DETR을 새로 클론한 후 VIVID-Det을 적용하는 전체 순서. 모든 명령은 프로젝트 루트 `C:\gitnconda\Swin-Transformer`에서 실행한다. `VIVID_src\`는 VIVID-Det 파일이 보관된 로컬 디렉토리를 의미하며, 실제 경로에 맞게 조정한다.
 
### 1단계: 교체 파일 복사 (3개)
 
```bash
copy /Y VIVID_src\backbone.py     rf-detr\src\rfdetr\models\backbone\backbone.py
copy /Y VIVID_src\__init__.py     rf-detr\src\rfdetr\models\backbone\__init__.py
copy /Y VIVID_src\matcher.py      rf-detr\src\rfdetr\models\matcher.py
```
 
### 2단계: 신규 모듈 파일 복사 (5개)
 
```bash
copy /Y VIVID_src\sawm.py         rf-detr\src\rfdetr\models\sawm.py
copy /Y VIVID_src\ssfa.py         rf-detr\src\rfdetr\models\ssfa.py
copy /Y VIVID_src\saqg.py         rf-detr\src\rfdetr\models\saqg.py
copy /Y VIVID_src\sqalb.py        rf-detr\src\rfdetr\models\sqalb.py
copy /Y VIVID_src\tfcm.py         rf-detr\src\rfdetr\models\tfcm.py
```
 
`sqalb.py`가 미확보 상태이므로, 작성 후 이 단계에서 함께 복사한다.
 
### 3단계: 패치 스크립트 실행 (3개, 순서 준수)
 
패치 스크립트는 프로젝트 루트(`C:\gitnconda\Swin-Transformer`)에서 실행한다. 각 스크립트가 `rf-detr\src\rfdetr\` 기준 상대 경로를 참조한다.
 
```bash
cd C:\gitnconda\Swin-Transformer
 
# 3-1. main.py 패치: SAWM 파라미터 (set_cost_nwd, nwd_C) 4곳 삽입
python patch_main_sawm.py
 
# 3-2. lwdetr.py + transformer.py 패치: SAQG 연결
python patch_saqg.py
 
# 3-3. lwdetr.py 패치: SQALB 연결 (반드시 3-2 이후 실행)
python patch_sqalb.py
```
 
`patch_saqg.py`와 `patch_sqalb.py`는 모두 `lwdetr.py`를 수정하므로, 반드시 `patch_saqg.py`를 먼저 실행한다. `patch_sqalb.py`는 SAQG import가 이미 존재하는 상태를 앵커로 사용하기 때문이다.
 
각 스크립트가 참조하는 파일 경로:
 
| 스크립트 | 내부 경로 | 확인 사항 |
|---|---|---|
| `patch_main_sawm.py` | 절대 경로 `C:\gitnconda\...\main.py` | 경로 하드코딩, 환경에 맞는지 확인 |
| `patch_saqg.py` | 상대 경로 `rf-detr\src\rfdetr\models\` | CWD가 프로젝트 루트여야 함 |
| `patch_sqalb.py` | 상대 경로 `rf-detr\src\rfdetr\models\lwdetr.py` | CWD가 프로젝트 루트여야 함 |
 
### 4단계: config.py 수동 수정
 
`src/rfdetr/config.py`의 `ModelConfig` 데이터클래스에 VIVID-Det enable/disable 플래그를 추가한다. 패치 스크립트가 없으므로 직접 편집한다.
 
추가할 필드:
 
```python
ssfa_enabled: bool = False
saqg_enabled: bool = True    # SSFA 활성 시 자동 연동
sawm_enabled: bool = True    # matcher에서 cost_nwd > 0으로 제어
sqalb_enabled: bool = True   # SSFA 활성 시 자동 연동
tfcm_enabled: bool = False   # Stage 2에서만 True
set_cost_nwd: float = 0.0    # 0이면 원본 동작 보존
nwd_C: float = 0.5
```
 
`RFDETRMedium(ssfa_enabled=True)` 형태로 모듈 활성화를 제어한다. `getattr(args, "attr_name", default)` 패턴으로 하위 호환성을 유지한다.
 
### 5단계: 검증
 
```bash
python -c "from rfdetr import RFDETRMedium; print('import OK')"
python -c "from rfdetr import RFDETRMedium; m = RFDETRMedium(ssfa_enabled=True); print('SSFA OK')"
```
 
---
 
## 파일별 상세 설명
 
### 교체 파일
 
**backbone.py** (v2.1.0, 423줄)
 
경로: `src/rfdetr/models/backbone/backbone.py`
 
원본 `Backbone.forward`의 encoder-projector 직결 구간에 TFCM과 SSFA를 삽입한다.
 
변경 후 forward 흐름:
```
encoder → [TFCM] → [SSFA] → projector
```
 
주요 수정:
- `__init__`: SSFA, TFCM 모듈 인스턴스 등록. `ssfa_enabled`, `tfcm_enabled` 플래그로 분기.
- `forward`: 삽입 지점 A 구현. `temporal_mode` 인자 추가.
- `forward_export`: 비디오 추론 시 `temporal_mode=True` 고정.
- `sopm` property: SAQG/TFCM이 접근하는 SOPM 캐시.
- `ssfa_losses` property: SOPM FocalLoss 캐시.
- `get_named_param_lr_pairs`: SSFA (CNN Branch lr=1e-3, 나머지 lr=1e-4), TFCM (lr=2e-4, wd=0) 파라미터 그룹 추가.
 
**backbone/\_\_init\_\_.py** (v2.1.0)
 
경로: `src/rfdetr/models/backbone/__init__.py`
 
주요 수정:
- `Joiner.forward`: `temporal_mode` 인자를 Backbone으로 전달.
- `build_backbone`: SSFA 8개 파라미터 + TFCM 6개 파라미터를 받아 Backbone 생성자로 전달.
 
**matcher.py**
 
경로: `src/rfdetr/models/matcher.py`
 
주요 수정:
- `HungarianMatcher.__init__`: `cost_nwd`, `nwd_C` 파라미터 추가.
- `HungarianMatcher.forward`: `cost_nwd > 0`일 때 `nwd_pairwise` 계산 후 cost matrix에 합산.
- `build_matcher`: `args.set_cost_nwd`에서 값 전달.
- `from rfdetr.models.sawm import nwd_pairwise` import 추가.
 
`cost_nwd=0` 기본값으로 원본 동작을 보존한다.
 
---
 
### 신규 모듈 파일
 
**sawm.py**
 
경로: `src/rfdetr/models/sawm.py`
 
NWD (Normalized Wasserstein Distance) 함수를 구현한다. bbox를 2D Gaussian으로 모델링하여 Wasserstein-2 거리 계산 후 `exp(-W2/C)`로 정규화한다.
 
- `nwd_pairwise(pred_boxes, tgt_boxes, C=0.5)` → matcher.py에서 호출
- 학습 전용, 추론 비용 0%, 추가 파라미터 0
 
**ssfa.py** (683줄)
 
경로: `src/rfdetr/models/ssfa.py`
 
SSFA 모듈 전체를 구현한다. 약 1.5M 파라미터.
 
내부 구성:
- `CNNBranch`: stride-8, GroupNorm(32), 256ch 고해상도 특징 추출
- `AttentionPriorExtractor`: DINOv2 Block 4 Self-Attn hook
- `SOPMHead`: concat(D3, attn_prior) → Sigmoid
- `SelectiveCrossAttention`: Top-K 25%, 2D sinusoidal PE
- `SOPMFocalLoss`: GT small heatmap 직접 감독
- `SSFA`: 통합 모듈, alpha=0.01 초기화
 
backbone.py의 `Backbone.__init__`에서 인스턴스화되고, `Backbone.forward`에서 호출된다.
 
**saqg.py**
 
경로: `src/rfdetr/models/saqg.py`
 
SAQG 모듈을 구현한다. 약 0.03M 파라미터.
 
- SOPM 통계 → MLP → 적응적 비율 (0.5~0.9)
- N_small 쿼리: SOPM 고밀도 영역에서 multinomial/top-k 샘플링
- N_general 쿼리: 원본 refpoint_embed 유지
- `ratio_mlp`의 gradient는 0 (detach + multinomial 설계상 정상)
 
`patch_saqg.py`가 lwdetr.py에 import/init/forward 호출을 삽입한다.
 
**sqalb.py -- 미확보**
 
경로: `src/rfdetr/models/sqalb.py`
 
SQALB 모듈 본체. `patch_sqalb.py`가 `from rfdetr.models.sqalb import SQALB`을 참조하므로 반드시 필요하다.
 
구현 요구사항:
- 학습 전용, 추론 비용 0%, 약 400 파라미터
- 5D MLP 입력: `[log(area), conf.detach(), iou_quality.detach(), layer_idx, matched_cost_ema]`
- area-bin EMA: 737 area-bin, resolution-aware, 3구간(small/medium/large) COCO 표준 경계
- SAWM의 NWD에 의존
- `forward(src_boxes, target_boxes, conf, iou, layer_idx)` → `(loss, alpha_mean)` 반환
 
**tfcm.py**
 
경로: `src/rfdetr/models/tfcm.py`
 
TFCM 모듈을 구현한다.
 
- 학습 가능 파라미터 2개: `_log_tau`, `_alpha_raw`
- `alpha = sigmoid(-10) ≈ 0`으로 identity start
- `tau = 0.03` learnable temperature
- `FeatureMemoryBuffer`: FIFO, N=4, detach 저장
- DINOv2 cosine similarity → softmax(tau) → alpha 융합
- SOPM(t-1) Scale-Aware Mask 적용 (tau_small=0.3)
- Cold start gradual activation: `alpha_eff = alpha * (N_avail / N_max)`
- `tfcm_enabled=False` (Stage 1), `tfcm_enabled=True` (Stage 2)
 
backbone.py의 `Backbone.__init__`에서 인스턴스화되고, `Backbone.forward`에서 SSFA 이전에 호출된다.
 
---
 
### 패치 스크립트
 
**patch_main_sawm.py**
 
패치 대상: `C:\gitnconda\Swin-Transformer\rf-detr\src\rfdetr\main.py`
 
4곳에 `set_cost_nwd`, `nwd_C` 파라미터를 삽입한다:
1. 허용 파라미터 목록 (`set_cost_giou` 아래)
2. argparse 인자 정의
3. 함수 기본값
4. args 전달
 
원본은 `main.py.bak`으로 자동 백업된다. 경로가 절대 경로로 하드코딩되어 있으므로 환경이 다르면 스크립트 내 `MAIN_PY` 변수를 수정한다.
 
**patch_saqg.py**
 
패치 대상: `rf-detr/src/rfdetr/models/transformer.py`, `rf-detr/src/rfdetr/models/lwdetr.py`
 
transformer.py 수정:
- `refpoint_embed.unsqueeze(0).repeat(bs, 1, 1)` 앞에 `dim() == 2` 체크 추가. SAQG가 이미 `[B, N, 4]`를 반환하면 expand를 건너뛴다.
 
lwdetr.py 수정 (3곳):
1. `from rfdetr.models.saqg import SAQG` import 추가
2. `LWDETR.__init__`에 `self.saqg = SAQG()` 등록 (`backbone[0].ssfa_enabled` 조건)
3. `LWDETR.forward`에서 transformer 호출 직전에 `self.saqg(refpoint_embed_weight, sopm, batch_size)` 호출
 
**patch_sqalb.py**
 
패치 대상: `rf-detr/src/rfdetr/models/lwdetr.py`
 
lwdetr.py 수정 (4곳):
1. `from rfdetr.models.sqalb import SQALB` import 추가
2. `SetCriterion.__init__`에 `self.sqalb = None` 등록
3. `loss_boxes`에 SQALB 분기 추가: `self.sqalb is not None and self.training`일 때 적응적 NWD/WIoU 밸런싱, 아닐 때 원본 L1 loss 유지
4. `build_criterion_and_postprocessors`에서 `criterion.sqalb = SQALB(...)` 연결
 
반드시 `patch_saqg.py` 이후에 실행한다. SAQG import 존재를 앵커로 사용하기 때문이다.
 
---
 
## Forward Pass 삽입 지점 요약
 
```
LWDETR.forward(samples)
  |
  +-- Backbone.forward(tensor_list, temporal_mode)      [backbone.py]
  |     |
  |     +-- encoder(tensor_list.tensors)                 # DINOv2 ViT-S/14
  |     |     -> feats [리스트, feats[0] = Block4 출력]
  |     |
  |     |   [삽입 지점 A-1] backbone.py:231
  |     |   +-- TFCM(feats[0], temporal_mode)             # tfcm.py (Stage 2)
  |     |
  |     |   [삽입 지점 A-2] backbone.py:247
  |     |   +-- SSFA(feats, images)                       # ssfa.py -> sopm 생성
  |     |   +-- tfcm.update_sopm_cache(sopm)              # 다음 프레임용 캐시
  |     |
  |     +-- projector(feats)                              # P3/P4/P5
  |           -> features, poss
  |
  |   [삽입 지점 B] lwdetr.py (patch_saqg.py가 삽입)
  |   +-- SAQG(refpoint_embed, sopm, batch_size)          # saqg.py
  |
  +-- transformer(srcs, masks, poss, refpoint, query)     [transformer.py]
  |     dim==2 체크 (patch_saqg.py가 삽입)
  |     -> hs, ref_unsigmoid
  |
  +-- class_embed(hs), bbox_embed(hs)
  |     -> out = {pred_logits, pred_boxes}
  |
  +-- (학습 시) SetCriterion.forward(out, targets)
        |
        +-- matcher(out, targets)                         [matcher.py]
        |     [삽입 지점 C] cost_nwd > 0이면 NWD cost 합산
        |     +-- nwd_pairwise(out_bbox, tgt_bbox)        # sawm.py
        |     -> indices
        |
        +-- loss_boxes(out, targets, indices)              [lwdetr.py]
              [삽입 지점 D] (patch_sqalb.py가 삽입)
              +-- SQALB: 적응적 NWD/WIoU 밸런싱            # sqalb.py
              -> losses
```
 
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
 
| 플래그 / 파라미터 | Stage 1 (이미지) | Stage 2 (비디오) |
|---|---|---|
| `ssfa_enabled` | True | True |
| `tfcm_enabled` | **False** | **True** |
| `set_cost_nwd` | > 0 (예: 2.0) | > 0 |
 
SAQG와 SQALB는 `ssfa_enabled=True`일 때 자동으로 연결된다 (`patch_saqg.py`, `patch_sqalb.py`가 `backbone[0].ssfa_enabled` 조건으로 분기).
 
---
 
## 미완료 항목
 
| 항목 | 상태 | 우선순위 | 비고 |
|---|---|---|---|
| `sqalb.py` 모듈 본체 | **미확보** | 높음 | patch_sqalb.py가 import하므로 없으면 에러 |
| `config.py` 수정 | 미적용 | 중간 | enable 플래그 추가 필요. 패치 스크립트 없음 |
| 7-test 통합 테스트 | 통과 기록 있음 | - | sqalb.py 확보 후 재검증 권장 |
 
`sqalb.py`가 없는 상태에서 `patch_sqalb.py`를 실행하면 패치 자체는 성공하지만, 이후 `from rfdetr.models.sqalb import SQALB`에서 `ModuleNotFoundError`가 발생한다. `sqalb.py`를 먼저 작성한 후 전체 적용 절차를 진행해야 한다.
