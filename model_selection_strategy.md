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
