# ADMET 전략 — 후처리 필터 (모델 수정 아님)

**역할:** 순위화된 후보 약물 리스트에 **ADMET 기준을 적용하는 후처리 단계**. 모델 구조를 바꾸는 단계가 아님.

**입력 전제 (팀4):** Nextflow는 **`results/`** 를 입력으로 쓰므로, ADMET 관련 컬럼이 이미 있으면 예: **`results/admet/25_admet_results.parquet`** 를 읽어 피처·후단 필터에 연결할 수 있음. 없거나 불완전하면 **예측 도구로 보완·검증**.

---

## 범주 1 — ADMET 예측 도구 (데이터 있으면 보완용)

| 도구 | 용도 | 형태 | 비고 |
|------|------|------|------|
| **ADMETlab 3.0** | Toxicity, Solubility, BBB, hERG 등 30+ 프로퍼티 | 웹/API | 포괄적. SMILES → JSON |
| **SwissADME** | 약물유사성(Lipinski), 용해도, 투과성 | 웹 | 빠른 검증. 배치 제한 |
| **pkCSM** | ADME + Toxicity 통합 | 웹/API | Ames, hERG 등 독성 특화 |
| **DeepPurpose** | DL 기반 ADMET | Python 패키지 | 파이프라인 자동화 |
| **RDKit** | Lipinski Ro5, LogP, TPSA 등 기초 | Python | 로컬·무료 |

**운영:** 기존 ADMET 테이블이 있으면 **ADMETlab / pkCSM 등으로 누락 약물만 보완**하면 됨.

---

## 범주 2 — 후처리 필터링 실행 도구

Network Proximity / CMap 등으로 만든 **순위 CSV(또는 테이블)** 에 컷오프를 적용하는 단계.

| 도구 | 용도 | 사용 시점 |
|------|------|-----------|
| **Pandas + Python** | 컷오프 필터 (예: `tox < 0.5`, `solubility > -4`) | 기본·재현성 높음 |
| **AWS Athena** | S3 ADMET 피처 테이블 SQL 필터 `WHERE ...` | S3 상 직접 쿼리 |
| **AWS Glue** | 대량 후보 Spark 배치 필터 | 후보 수만 건 이상 |
| **SageMaker Processing** | 필터 스크립트를 MLOps 파이프에 통합 | Week4 자동화 시 |

팀4 Nextflow FE는 **`results/`→피처** 이므로, ADMET(범주 1)은 **`results/admet/` 조인** 또는 그래프 내 예측 단계로 두고, 후단(범주 2)에서 **순위 + ADMET 조인 + 필터**를 구현하면 됨.

---

## 범주 3 — 컷오프 기준 (합의 필요)

필터 구현보다 **어떤 임계값을 쓸지**가 중요. 아래는 참고 초안 — 임상·팀 리뷰 후 확정.

| ADMET 항목 | 일반적인 컷오프 | 근거 |
|------------|-----------------|------|
| Toxicity (Ames) | 음성 (non-mutagenic) | 변이원성 없는 약물만 |
| hERG 독성 | IC50 > 10 µM | 심장독성 회피 |
| Solubility (LogS) | > −4 | 경구 흡수 가능 범위 |
| LogP | −0.4 ~ 5.6 (Lipinski) | 세포막 투과 |
| Lipinski Rule of 5 | 위반 ≤ 1 | 경구 약물 유사성 |
| Bioavailability | > 0.55 | SwissADME 기준 (참고) |

**주의:** 파이프라인의 **피처 단계 ADMET 컬럼**(모델 입력)과 이 **후처리 Gate**는 역할이 다름. 동일 수치를 재사용할 수는 있으나, **정책(통과/탈락)** 은 이 절에서 정의.

---

## 팀 액션

1. `25_admet_results.parquet` **컬럼명**과 위 Gate 표를 **1:1 매핑** (없는 항목은 N/A 또는 보완 도구 지정).
2. 후처리는 **Pandas 스크립트부터** 고정하고, 규모 커지면 Athena/Glue로 이전.
3. 컷오프 수치는 **문헌·내부 SOP**에 맞게 조정 후 이 문서 갱신.
