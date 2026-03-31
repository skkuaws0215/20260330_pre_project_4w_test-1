# 모델 선별·확장 전략 (PPTX 반영)

**근거 자료:** `drug_repurposing_model_selection (1).pptx`  
**적용 방침 (팀4):** PPT에 나온 모델을 **전부** 쓰지 않음. 확장 시 대략 **ML 4~5종 · DL 3~4종 · Graph 3~4종**, **우선순위 순**으로 추가.  
**제외:** PPT **마지막 슬라이드(최종 1종 요약)** 는 블루프린트 참고용으로 두고, 본 문서의 “다중 모델 벤치마크” 계획과 **혼동하지 않음** (실제로는 Track별 여러 모델 후 랭킹).

---

## Track A — Traditional ML (분류/예측)

PPT 후보(슬라이드 3). ★ = PPT 상 우선 후보.

| 순위(참고) | 모델 | 메모 (PPT 요지) |
|------------|------|-----------------|
| 1 ★ | **LightGBM** | 고차원·속도·메모리, 범주형·ADMET 피처, SageMaker built-in |
| 2 | XGBoost | 희소·LINCS 등, SHAP |
| 3 | Random Forest | 중요도·baseline, TCGA/METABRIC 결측 robust |
| 4 | Elastic Net | Morgan + 오믹스 선형, 계수 해석·LLM 근거 |
| 5 | CatBoost | 고카디널 범주형, 소규모 코호트 |

**확장 목표:** 위에서 **4~5종**까지 선택·실험 (보통 **LightGBM → XGBoost → RF → …** 순으로 파일럿).

---

## Track B — Deep Learning (표현 학습)

PPT 후보(슬라이드 4).

| 순위(참고) | 모델 | 메모 (PPT 요지) |
|------------|------|-----------------|
| 1 ★ | **VAE** | LINCS perturbation latent → Track A/C 재사용, late fusion |
| 2 | Multi-task MLP | 다 암종·다 약물, ADMET task head 가능 |
| 3 | TabNet | 테이블 어텐션·해석 |
| 4 | 1D-CNN (SMILES) | Morgan 대체·end-to-end, VAE latent와 concat |
| 5 | Contrastive (SimCLR) | 플랫폼 불변 표현 |

**확장 목표:** **3~4종** (예: **VAE → TabNet → MLP → …**).

---

## Track C — Graph & Ranking

PPT 후보(슬라이드 5).

| 순위(참고) | 모델 | 메모 (PPT 요지) |
|------------|------|-----------------|
| 1 ★ | **Network Proximity (z-score)** | STRING·OpenTargets, Nextflow 구현 부담 낮음, ADMET 후단과 연결 |
| 2 | Signature Reversal (CMap) | LINCS·질환 서명, LINCS 전처리 선행 |
| 3 | GNN / GraphSAGE | PPI 임베딩, PyG |
| 4 | Target/Pathway Overlap | Jaccard·hypergeometric, 해석·Bedrock 근거 |
| 5 | Knowledge Graph Embedding | 통합 비용 큼, 후순위 |

**확장 목표:** **3~4종** (예: **Proximity → Signature → Overlap → GNN**).

---

## Nextflow / 실험 배치 제안

1. **파일럿:** Track A에서 **1종(권장 LightGBM)** + 피처 테이블 소규모로 E2E.
2. **확장:** 동일 split·metric으로 Track별 우선순위대로 모델 추가 → **랭킹표** → 최종 후보 축소.
3. **METABRIC / ADMET / Bedrock** 은 `pipeline_overview.html` · README 워크플로 순서 유지.

---

## 슬라이드 6 (METABRIC & ADMET 호환성)

PPT 요지: METABRIC은 도메인 갭(GDSC vs 환자)을 **실패가 아닌 일반화 지표**로 해석. ADMET는 LightGBM/XGB/ElasticNet/MLP는 피처·head로 정합, VAE는 latent+ADMET concat, Track C는 Week4 필터와 구조적으로 맞춤 — **별도 `admet_integration_notes.md` 로 풀어도 됨.**

---

**ADMET (후처리 필터):** 모델 수정이 아니라 순위 리스트에 기준 적용. 도구·컷오프·실행 레이어는 `admet_postprocessing_strategy.md` 참고.

---

## 결측 임계값 A/B/C 실험 프로토콜 (팀4 FE v2)

목적: 느슨한 결측 기준(70%)과 엄격한 결측 기준(30%)의 차이, 그리고 `SMILES` 정보 반영 효과를 분리해서 본다.

### 데이터 버전

- **70% run**: `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_batch_runs/20260330_batch_miss70_v2/`
- **30% run**: `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_batch_runs/20260330_batch_miss30_v2/`
- 현재 차이 컬럼(결측 필터 영향):
  - 70%에는 유지: `drug__smiles`, `drug__canonical_smiles_raw`
  - 30%에서는 제거

### 실험군 정의

| 그룹 | 입력 데이터 | SMILES 처리 | 목적 |
|------|-------------|------------|------|
| A (엄격) | miss30_v2 | 없음 | 엄격 결측 기준 baseline |
| B (느슨) | miss70_v2 | 문자열 SMILES는 학습 제외 | 결측 기준 차이만 확인 |
| C (SMILES+) | miss70_v2 | SMILES를 수치 피처(예: fingerprint/descriptor)로 변환 후 추가 | SMILES 정보 기여도 확인 |

### 해석 포인트

- **A vs B**: 결측 임계값(30 vs 70) 영향
- **B vs C**: 동일 임계값(70)에서 SMILES 정보 자체의 순증분

### 공정 비교 규칙 (고정)

1. split 방법/seed/검증 folds 고정
2. metric 고정 (회귀: RMSE, MAE, Spearman 권장)
3. 동일 모델 후보군으로 A/B/C 모두 반복
4. 반복 실험(최소 3 seeds) 평균과 표준편차 함께 기록

### 권장 우선순위

1. **B 먼저**: 파이프라인 안정 baseline 확보
2. **A 추가**: 엄격 필터 민감도 확인
3. **C 확장**: RDKit 기반 SMILES 수치 피처 붙여 최종 성능 개선 검증
