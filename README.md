# 4주차 사전 프로젝트 (팀4)

이 저장소는 **팀 AWS 계정**으로 S3 등 리소스에 접근할 때, 개인 PC의 기본 AWS 설정(`%USERPROFILE%\.aws`)과 섞이지 않도록 **프로젝트 로컬에만** 자격 증명을 두는 구성을 사용합니다.

## 폴더 구성

| 경로 | 설명 |
|------|------|
| `.aws/config` | 팀 프로필 `4team-project`의 리전·출력 형식 (`output`은 반드시 소문자 `json`) |
| `.aws/credentials` | 팀 IAM 액세스 키 (로컬 전용, **Git에 올리지 않음**) |
| `.aws/credentials.example` | `credentials` 작성용 예시. 복사 후 값만 채우면 됨 |
| `use-team-aws.ps1` | 현재 PowerShell 세션에서만 AWS CLI가 위 `.aws`를 쓰도록 환경 변수 설정 |
| `.gitignore` | `.aws/credentials` 커밋 방지 |
| `pipeline_overview.html` | 파이프라인 개요 보고서 — §2에 Nextflow·Batch·METABRIC·ADMET·Bedrock 플로우, §18~§19에 4주 산출물·Final 로드맵 반영 |
| `index.html` | 루트 HTML **목차** — SageMaker·METABRIC/ADMET/Survival·DL·Graph 등 대시보드 링크 (`serve_dashboards.py` / GitHub Pages) |
| `sagemaker_experiment_dashboard_20260403.html` | **SageMaker 통합 실험** 대시보드 — 스냅샷 `20260403_v9`, **운영 정책: FDA 승인 의약품만** 최종 후보 허용(§4.3 청색 박스); 현재 `final_shortlist.csv`는 **승인 필터 적용 전** 스냅샷. §4.1–4.6, §4.4 METABRIC-like **proxy** 등 |
| `metabric_admet_survival_dashboard_20260402_v2.html` | **METABRIC · ADMET · Survival-linked** — **운영 정책: FDA 승인 약만** (§2 청색 박스); §2 표는 **승인 필터 전** 스냅샷 + PubChem명. **§3** 생존, **§4–§6** proxy, **§7** 파이프라인 |
| `streamlit_app.py` | 진행용 **Streamlit 대시보드** (용어·체크리스트·HTML 보고서·README 뷰어) |
| `requirements.txt` | `streamlit` 등 Python 의존성 |
| `nextflow/` | Nextflow 코드 예정 (`main.nf` 등) · 업로드 방법 `nextflow/README.md` |
| `results/features_nextflow_team4/README.txt` | S3 동일 키 — **본인 전용** FE·ML 산출 prefix 안내 (공유 `results/<소스>/` 와 구분, 로컬=S3 구조) |
| `model_selection_strategy.md` | PPTX 기반 ML/DL/Graph 후보·우선순위·확장 규모(4~5 / 3~4 / 3~4) |
| `admet_postprocessing_strategy.md` | ADMET **후처리 필터** (예측 도구 / 실행 도구 / 컷오프 참고) |

## Streamlit 대시보드

문서와 체크리스트를 보며 진행할 때:

```powershell
cd C:\Users\biso8\dev\20260330_pre_project_4w_test-1
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

`streamlit` 명령을 찾지 못한다면(`CommandNotFoundException`), 위처럼 **`python -m streamlit`** 을 사용하세요. Windows에서 pip가 사용자 폴더에만 설치하면 `Scripts` 경로가 PATH에 없는 경우가 많습니다.

브라우저에서 사이드바로 **전체 요약·현황**, **핵심 용어**, **§2 플로우**, **Pre-Project 체크리스트**, **`pipeline_overview.html` 전체** 등을 전환해 확인합니다.

## Git / GitHub

로컬에 커밋된 상태입니다. **원격 저장소를 만든 뒤** (GitHub에서 New repository, README 없이 생성 권장):

원격 저장소: **https://github.com/skkuaws0215/20260330_pre_project_4w_test-1**

```powershell
cd C:\Users\biso8\dev\20260330_pre_project_4w_test-1
git remote add origin https://github.com/skkuaws0215/20260330_pre_project_4w_test-1.git
git push -u origin main
```

(이미 `origin`이 있으면 `git remote set-url origin https://github.com/skkuaws0215/20260330_pre_project_4w_test-1.git`)

`.aws/credentials` 는 `.gitignore`에 있어 **커밋되지 않습니다.**

## 개인 계정 vs 팀 계정

- **개인:** Windows 사용자 폴더의 `C:\Users\<이름>\.aws\` — 평소 쓰는 프로필/SSO 등
- **팀:** 이 프로젝트의 `.aws\` — 팀 버킷·역할용 키만 보관

둘을 한 파일에 섞지 말고, 팀 작업 시에는 아래 스크립트로 **이 폴더의 설정만** 쓰면 됩니다.

## 사용 방법 (PowerShell)

1. 팀 키가 아직 없다면 예시를 복사해 편집합니다.

   ```powershell
   Copy-Item .aws\credentials.example .aws\credentials
   notepad .aws\credentials
   ```

2. 프로젝트 루트에서 팀 환경을 켭니다.

   ```powershell
   .\use-team-aws.ps1
   ```

3. 동작 확인:

   ```powershell
   aws sts get-caller-identity
   ```

4. S3 데이터 경로·ARN·CLI 예시는 아래 **팀 데이터 버킷 (S3)** 절을 참고합니다.

**참고:** 새 PowerShell 창을 열 때마다 `use-team-aws.ps1`을 다시 실행해야 팀 `.aws`가 적용됩니다.

## 팀 데이터 버킷 (S3)

| 구분 | 위치 |
|------|------|
| 버킷 ARN | `arn:aws:s3:::drug-discovery-joe-raw-data-team4` |
| Raw(원본) | `s3://drug-discovery-joe-raw-data-team4/` — **`results/` 바깥** 상위 접두사에 원본 데이터 |
| 전처리 완료 (팀 **공유**) | `s3://drug-discovery-joe-raw-data-team4/results/<소스>/` 등 — **팀원 4인이 함께 쓰는 구역**. 여기에 본인 FE 산출을 올리지 말 것(경로·권한 혼선 방지). |
| **본인 전용 FE + ML 입력** | `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/` — **앞으로 본인이 만든 FE·실험 산출은 전부 이 prefix 아래**에 업로드. 이 폴더에서 나온 데이터셋으로 ML 모델 테스트·학습. |

**폴더 헷갈리지 않기:** `results/` = 공유 전처리만 참고(읽기 입력). **쓰기(본인 작업물)** 는 **`results/features_nextflow_team4/`** 만 사용.

**팀4 Nextflow 입력·산출 정책**

- **입력:** **raw에서 파이프라인으로 이미 만들어 둔 `s3://…/results/<소스>/…`** 전처리 산출(parquet 등)을 Nextflow가 읽어 **피처 엔지니어링·조인**을 수행한다. (README § Parquet 표 경로·용량 참고.)
- **사용하지 않음(입력):** **`ml_ready/`** — 이미 통합된 ML-ready 테이블이 아니라, **`results/` 소스별 산출**에서 Nextflow가 조합한다.
- **산출:** 모델 학습용 피처 테이블·메타·중간 산출은 **`results/features_nextflow_team4/<run_id>/`** 등에만 둔다. **S3에 FE 관련 신규 데이터를 올릴 때도 이 prefix 하위로** 올려 팀 공유 `results/<소스>/` 와 섞이지 않게 한다. 학습(SageMaker 등) 입력은 이 산출을 사용.

### FE contract (2026-03-30 합의)

- **Target unit:** `sample-drug pair`
- **Main label:** 회귀(`IC50/AUC/response score`)
- **Aux label:** 이진(`sensitive/resistant`)
- **Feature rules:** high-missing 제거, median/UNK imputation, variance filtering, leakage 컬럼 제거
- **Branch:** tree용 기본 피처 + DL용 정규화 브랜치 분리
- **Run outputs:** `features.parquet`, `labels.parquet`, `feature_manifest.json` (옵션: `features_dl.parquet`)

### 결측 임계값 비교 실험 (70% vs 30%)

팀4 FE 산출 기준으로 결측 임계값 비교용 run을 분리 저장 (**배치 산출 경로**: `results/features_nextflow_team4/fe_re_batch_runs/<run_id>/`, Nextflow `main.nf`의 `publishDir`와 동일). 폴더명은 **신규 FE 재생성·재배치 배치**를 한곳에 모으려는 의도(`re_batch`); `fe_` 접두는 `input/`, `abc_inputs/` 등 팀4 prefix와 같이 쓰기 쉽게 맞춤.

- **느슨한 결측(70%)**: `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss70_v2/`
- **엄격한 결측(30%)**: `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss30_v2/`

비교 결과 요약:

- 두 run 모두 행 수 동일 (`14,497`)
- 70%: feature `17,922` cols
- 30%: feature `17,920` cols
- 차이 컬럼(30%에서 제거): `drug__smiles`, `drug__canonical_smiles_raw`

권장 실험 프레임(A/B/C):

1. **A (엄격 baseline)**: miss30_v2 그대로 사용
2. **B (느슨 baseline)**: miss70_v2 사용, 문자열 SMILES는 학습 입력에서 제외
3. **C (SMILES+)**: miss70_v2 사용, SMILES를 descriptor/fingerprint로 수치화 후 추가

상세 프로토콜은 `model_selection_strategy.md`의 “결측 임계값 A/B/C 실험 프로토콜” 절을 따른다.

### A/B/C 입력셋 (팀4 FE 폴더에 고정)

아래 입력셋은 모두 **팀4 FE 생성 폴더**에 생성/정리:
`s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/`

| 순서 | 실험군 | 설명 | features URI | labels URI |
|------|--------|------|--------------|------------|
| 1 | A | 엄격 결측 baseline (**결측치 30%**, miss30) | `.../A/features.parquet` | `.../A/labels.parquet` |
| 2 | B | 느슨 기준 (**결측치 70%**) + 문자열 SMILES 제외 | `.../B/features_b.parquet` | `.../B/labels.parquet` |
| 3 | C | 느슨 기준 (**결측치 70%**) + SMILES 확장용 원본 | `.../C/features.parquet` | `.../C/labels.parquet` |

전체 인덱스(혼선 방지):  
`s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/abc_index.json`

### A/B/C 보조 스크립트 (템플릿)

- B 입력 생성(문자열 SMILES 제외):
  - `nextflow/scripts/prepare_b_input.py`
- A/B/C 공통 학습 템플릿(회귀 baseline):
  - `nextflow/scripts/train_abc_template.py`

예시:

```bash
python3 nextflow/scripts/prepare_b_input.py \
  --features-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss70_v2/features.parquet" \
  --labels-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss70_v2/labels.parquet" \
  --output-prefix "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/ab_tests/b_input_v1"

python3 nextflow/scripts/train_abc_template.py \
  --features-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss30_v2/features.parquet" \
  --labels-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260330_batch_miss30_v2/labels.parquet" \
  --output-prefix "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/ab_tests/train_v1" \
  --experiment-tag "A_miss30_seed42"
```

### FE 데이터셋 산출물 구조 (무엇이 몇 개 생기나?)

기본 FE run(`nextflow/main.nf`) 1회당 핵심 산출은 보통 **3~4개**:

1. `features.parquet`  
   - 학습 입력 피처 테이블 (키 포함: `sample_id`, `canonical_drug_id`)
2. `labels.parquet`  
   - 정답값 테이블 (`label_regression`, `label_binary` 등)
3. `feature_manifest.json`  
   - 전처리/필터링/행열 수/입출력 경로 메타데이터
4. `features_dl.parquet` *(옵션)*  
   - DL용 정규화 브랜치(`normalization_branch`가 `dl`/`both`일 때)

즉 FE 단계는 **학습값(피처)**·**정답값(라벨)**·**실험 메타**를 만든다.  
**예측값은 FE 단계에서 생성되지 않고**, 모델 학습/추론 단계에서 별도 생성한다.

A/B/C 학습 템플릿 기준 예측 산출은 run당 보통 **2개**:

- `<experiment_tag>_predictions.parquet` (예측값 + 정답 + split)
- `<experiment_tag>_metrics.json` (RMSE/MAE/Spearman 등)

### 구현 파일 (nextflow/)

- `nextflow/main.nf`: FE workflow
- `nextflow/nextflow.config`: `local`/`awsbatch` 프로필
- `nextflow/scripts/build_features.py`: join + FE 규칙 구현
- `nextflow/Dockerfile`, `nextflow/requirements-fe.txt`: Batch 컨테이너 빌드용

CLI 예시(`use-team-aws.ps1` 실행 후):

```powershell
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/" --recursive | findstr /i ".parquet"
```

### `results/` 스냅샷 (조회: 2026-03-29)

**1단계 prefix** (`aws s3 ls …/results/`): `admet/`, `chembl/`, `cross_platform/`, `depmap/`, `drugbank/`, `gdsc/`, `id_mapping/`, `lincs/`, `metabric/`, `msigdb/`, `opentargets/`, `pubmed/`, `string/`, `tcga/` 및 루트 `final_qc.csv`.

**Parquet 22개** (**팀4 Nextflow FE 입력** 후보 — `results/<소스>/`; raw 기반 전처리 산출. 용량 큰 파일 주의):

| 경로 (버킷 내 `results/` 기준) | 비고 |
|--------------------------------|------|
| `tcga/01_raw_counts.parquet` ~ `07_qc_passed.parquet` | 대용량 (~0.9–1.0GB 단위) |
| `metabric/08_raw.parquet`, `54_filtered.parquet` | ~238MB |
| `lincs/16_mcf7_processed.parquet` | ~7.6GB |
| `chembl/15_activities.parquet`, `drugbank/14_drug_targets.parquet`, `gdsc/21_ic50.parquet`, `22_binary_labels.parquet` | |
| `string/19_ppi_filtered.parquet`, `msigdb/20_gene_sets.parquet`, `depmap/57_crispr.parquet`, `58_drug_labels.parquet` | |
| `cross_platform/11_intersection.parquet`, `id_mapping/55_harmonized.parquet`, `admet/25_admet_results.parquet` | |
| `opentargets/26_associations.parquet`, `pubmed/23_corpus.parquet` | |

팀에 prefix를 알리려면 S3 **`results/features_nextflow_team4/README.txt`** 에 올립니다. 저장소에는 동일 상대 경로 **`results/features_nextflow_team4/README.txt`** 로 두어 Git·S3 키를 맞춥니다. 업로드: `nextflow/README.md` 의 `aws s3 cp` 예시.

## Pre-Project 목표 워크플로 (계획 확인용)

아래는 `pipeline_overview.html`에 **명시적으로 빠져 있거나 한 줄로만 언급된 부분**을 보강한 **의도한 단계 순서**입니다. 팀 검토·확인용이며, 구현 후 이 절을 실제 산출물 경로와 맞춰 갱신하면 됩니다.

### 1) HTML 대비 보강: 빠진 단계

| 단계 | 역할 | 비고 |
|------|------|------|
| **METABRIC (외부 코호트 검증)** | TCGA 등으로 학습·튜닝한 모델·랭킹이 **독립 코호트에서도 일반화**되는지 검증 | 슬라이드 상 “TCGA (Train) + METABRIC (Validation)” 퍼널과 동일 목적. HTML §10에 외부 AUROC 등은 있으나, **워크플로 다이어그램 단계로는 분리 표기가 부족**함. |
| **ADMET (임상 진입 전 생물학적 필터)** | 후보 약물의 **용해도, CYP 대사, hERG, 간독성** 등 **4 Gate**로 안전·약동학 측면 선별 | HTML §6·후처리에 ADMET 요소는 있으나, **Week 4 Part A처럼 “임상 진입 필터” 단계로 명시**되진 않음. |
| **Bedrock (LLM) 추가 검증·XAI** | 원시/모델 점수에 **문헌·지식 그래프 RAG**를 붙여 근거 수집, 기전 설명, 리스크, **모순 탐지** → 임상 문서형 산출 | ADMET·METABRIC 결과를 **설명·교차검증**하는 층으로 두는 것이 목표. |

**권장 순서(요약):**  
데이터(raw → **`results/`** 전처리) → **Nextflow가 `results/`를 읽어 피처 엔지니어링** (`ml_ready/`는 입력으로 쓰지 않음) → (파일럿) ML / DL / Graph 학습·비교 → 확장·**모델 랭킹**·최적 모델 선정 → **METABRIC 일반화 검증** → **ADMET Gate** → **Bedrock 기반 검증·설명 보고서**.

### 2) 데이터 소스 (S3)와 Nextflow

- 버킷에는 **raw**, 전처리 **`results/`**, 통합 **`ml_ready/`** 가 함께 있을 수 있음(상표 참고).
- **Nextflow(팀4)** 는 **`results/<소스>/`** 전처리 산출을 **입력**으로 쓰고, **피처·조인·학습용 테이블**은 **`results/features_nextflow_team4/`** 에 새로 쓴다. **`ml_ready/`는 FE 입력으로 사용하지 않는다.** (통합은 Nextflow 그래프에서 수행.)

### 3) 모델 전략: 소수 파일럿 → 전체 확장 → 랭킹

1. 피처 엔지니어링 결과 중 **소수 테이블·부분 샘플**만으로 ML / DL / Graph **스모크 테스트**.
2. 파이프라인·지표가 안정적이면 **전체 데이터로 확장** 학습.
3. **모델별 성능·안정성**을 비교해 **랭킹**하고, **최고 모델(또는 앙상블)**로 후속(METABRIC·ADMET·Bedrock)에 사용.

### 4) Final-Project 방향 (로드맵 요약)

- Pre-Project 코어에서 출발해 **멀티오믹스·고급 GNN·멀티모달(병리/영상)·구조 기반(도킹/MD)·고급 ADMET·Bedrock 자동 리포트/Q&A** 등으로 확장하는 그림은 별도 로드맵 슬라이드와 정합.

---

## AWS Batch로 돌릴지에 대한 의견

**결론:** 파일·잡 수가 많고 **병렬로 쪼개 실행**할 피처 생성·하이퍼파라미터 스윕·여러 모델 재현 실험이라면 **AWS Batch는 합리적인 선택**입니다.

**잘 맞는 경우**

- Nextflow **`awsbatch` 실행자**로 프로세스당 1 Batch 작업 매핑 → **I/O는 S3**, 컴퓨트는 **온디맨드 스케일**.
- 동질적인 작업이 대량(수백~수천 컨테이너)으로 **서로 독립**일 때 비용·운영 단순.

**설계 시 확인할 점**

- **GPU**가 필요한 DL/Graph 작업은 **별도 GPU 컴퓨트 환경·큐**와 **인스턴스 제한**을 분리하는 것이 좋음(CPU 피처 vs GPU 학습).
- **대용량 parquet**는 작업당 **S3 직접 읽기**보다, 반복 실험이면 **EFS/FSx for Lustre** 등으로 스테이징해 **동일 데이터 재읽기 비용·지연**을 줄이는 방안을 검토.
- **컨테이너 이미지**(ECR), **IAM(S3 Get/Put)**, **작업 재시도·타임아웃**, **Spot 사용 여부**(비용 vs 중단)를 Nextflow 프로필로 나누면 파일럿→전체 확장에 유리.

**대안(참고)**

- **SageMaker Processing / Training**: ML/DL에 맞춘 관리형이지만, 범용 배치·Nextflow와의 궁합은 Batch가 더 직관적인 경우가 많음.
- **단일 대형 EC2/SageMaker 노트북**: 파일럿·디버깅에는 좋고, **전체 확장·재현성**은 Batch+Nextflow 쪽이 유리.

이 저장소에는 Nextflow FE 뼈대(`main.nf`, `nextflow.config`, `build_features.py`)가 추가되어 있음. Batch 적용 시 `nextflow.config`의 `awsbatch` 프로필에 **실제 큐 이름·ECR 이미지**를 반영해 실행.

## 아키텍처 합의 (Batch vs SageMaker)

| 구간 | 담당 |
|------|------|
| **피처 엔지니어링** | **Nextflow + AWS Batch** — 병렬 처리·대량 S3 I/O |
| **학습·하이퍼파라미터 튜닝** | 주로 **Amazon SageMaker**(노트북/Studio, Training Job, built-in 또는 커스텀, GPU 예: `ml.g4dn.*`) |
| **선택** | 동일 Docker 이미지를 **Batch GPU** 큐에 올려 학습만 실행 |

PPTX의 LightGBM/XGB **SageMaker built-in** 등은 위 “학습·튜닝” 행에 해당합니다.

## 모델 학습·튜닝 (SageMaker)

위 표와 같이, 학습·튜닝의 기본 무대는 **SageMaker**로 두고, Batch는 FE 쪽과 (선택 시) GPU 학습 재사용에 활용합니다.

### 군별 대표 3종 최종 실행 (탐색 아님, 재현·정리)

로컬 CV로 **선정 근거가 고정**된 뒤, SageMaker에서는 **XGBoost · ResidualMLP · GCN만** 최종 학습·아티팩트·로그를 정리합니다. **추가 튜닝·모델 탐색은 이 단계에서 하지 않습니다.** 동일 데이터·join·지표 정의를 유지하고, Graph는 **drug-group CV 기준 GCN**과 **`gcn_tuning_summary.json`의 baseline A 하이퍼파라미터**를 따릅니다.

- **산출 루트:** `results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three/`
  - `final_model_comparison.csv` — 행별 로컬 CV mean + SageMaker job·평가·S3 URI(잡 후 채움)
  - `final_model_summary.json` — 실행 원칙·경로·로컬 근거 JSON 포인터
  - `artifacts/xgb/`, `artifacts/residualmlp/`, `artifacts/gcn/` — 모델 번들·`metrics.json` 등
  - `logs/` — CloudWatch/로그 포인터
  - `README.txt` — 레이아웃·`metrics.json` sidecar 스키마
- **비교표 재생성 (로컬 근거 반영):** `python3 ml/pilot_sagemaker/aggregate_final_representative_outputs.py`
- **잡 후 sidecar 병합:** 각 `artifacts/*/metrics.json`을 채운 뒤 `python3 ml/pilot_sagemaker/aggregate_final_representative_outputs.py --collect`
- **XGBoost 제출:** `ml/pilot_sagemaker/submit_final_xgb_sagemaker.py` → `train_tabular.py` (`full_train off`, `test_size 0.1`, tuned `xgb_*`; pathway_addon 피처·`labels_B_graph` 기본 스테이징).
- **ResidualMLP 제출:** `ml/pilot_sagemaker/submit_final_residual_mlp_sagemaker.py` → `train_residual_mlp_final.py`
- **GCN 제출:** `ml/pilot_sagemaker/submit_final_gcn_sagemaker.py` → `train_gcn_final.py` (baseline A, disease genes 번들 포함)
- **잡 산출물 반영:** `ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py` 로 `model.tar.gz`를 `artifacts/{xgb|residualmlp|gcn}/`에 풀기 → `aggregate_final_representative_outputs.py --collect`
- **비교 CSV 열:** `local_validation_type`(선정용 CV 요약), `sagemaker_validation_type`·`sagemaker_evaluation_note`(최종 잡의 검증 정의·주의문; GCN은 row holdout이므로 drug-group CV mean과 직접 비교하지 않음).

#### SageMaker 최종 실행 완료 (2026-04-02)

- Job 완료:
  - XGBoost: `team4-final-xgb-20260331-1775110781`
  - ResidualMLP: `team4-final-resmlp-1775110990`
  - GCN: `team4-final-gcn-1775111671`
- 최종 비교 산출:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three/final_model_comparison.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three/final_model_summary.json`
- GCN `evaluation_note`는 유지: SageMaker 지표는 row holdout, 대표 선정은 drug-group CV 기준.

#### SageMaker 20260402 Dual Validation (holdout + CV)

- 산출 경로: `results/features_nextflow_team4/fe_re_batch_runs/20260402/sagemaker_dual_validation/`
- 생성 파일:
  - `final_model_comparison.csv`
  - `final_model_summary.json`
  - `holdout/artifacts/{xgb,residualmlp,gcn}/metrics.json`
  - `cv/artifacts/{xgb,residualmlp,gcn}/*_fold_metrics.csv`, `*_metrics.json`

**Holdout (single split)**

| family | model | job | RMSE | MAE | Spearman | NDCG@20 | Hit@20 |
|------|------|------|------|-----|----------|---------|--------|
| ML | XGBoost | `team4-final-xgb-20260331-1775114416` | 2.1036 | 1.5768 | 0.4701 | - | - |
| DL | ResidualMLP | `team4-final-resmlp-1775114623` | 2.1069 | 1.5806 | 0.4621 | 0.8234 | 1.0000 |
| Graph | GCN | `team4-final-gcn-1775114858` | 1.4987 | 1.0942 | 0.8422 | 0.9543 | 1.0000 |

**CV (5-fold)**

| family | model | validation_type | job | RMSE(mean) | MAE(mean) | Spearman(mean) | NDCG@20(mean) | Hit@20(mean) |
|------|------|------------------|------|------------|-----------|----------------|---------------|--------------|
| ML | XGBoost | row KFold | `team4-cv-xgb-20260402-1775115653` | 2.0703 | 1.5414 | 0.4731 | 0.8401 | 1.0000 |
| DL | ResidualMLP | row KFold | `team4-cv-residualmlp-20260402-1775115890` | 2.0728 | 1.5371 | 0.4688 | 0.8401 | 1.0000 |
| Graph | GCN | GroupKFold(by `canonical_drug_id`) | `team4-cv-gcn-group-20260402-1775116156` | 2.5793 | 2.0959 | 0.2301 | 0.7452 | 0.9931 |

- validation_type은 비교표에서 `holdout` vs `cv`로 명시해 구분.
- GCN은 holdout과 CV의 검증 정의가 다르므로(CV=drug-group), 행 간 직접 비교 시 주의.

#### 다음 단계 (로컬 분석): 3모델 ensemble + ranking

SageMaker 추가 실행 없이, 저장된 대표 모델 아티팩트의 예측값을 같은 pair 키(`sample_id`, `canonical_drug_id`)로 결합해 후속 랭킹을 만든다.

- 스크립트: `ml/pilot_sagemaker/build_final_ensemble_ranking.py`
- 가중치: `XGBoost 0.5`, `ResidualMLP 0.3`, `GCN 0.2`
- 실행:

```bash
python3 ml/pilot_sagemaker/build_final_ensemble_ranking.py
```

- 출력:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three/final_ensemble_ranking.csv`
  - 컬럼: `pred_xgb`, `pred_residualmlp`, `pred_gcn`, `ensemble_score`, `rank`, `is_top100`

#### 20260402 Post-Analysis 완료 (ensemble v2 → METABRIC → ADMET → shortlist)

- 산출 경로: `results/features_nextflow_team4/fe_re_batch_runs/20260402/sagemaker_dual_validation/`
- 생성 파일:
  - `final_ensemble_ranking_v2.csv`
  - `metabric_validation_summary.csv`
  - `admet_filter_log.json`
  - `final_shortlist.csv`

**Ensemble v2**

- 입력: 기존 예측 테이블(`sample_id`, `canonical_drug_id`)
- 가중치: `XGBoost 0.5`, `ResidualMLP 0.4`, `GCN 0.1`
- 추가 컬럼: `ensemble_score_v2`, `rank_v2`, `is_top100_v2`

**METABRIC external validation** (SageMaker 대시보드 **§4.4**·METABRIC HTML **§3**과 동일 전제: **sample pair 투영·METABRIC-like proxy**, 코호트 전체 임상 라벨 그대로의 “진짜 외부 정답” 검증이 아님)

- 기준: Top100(unique drug) 후보 + METABRIC-like sample set(`sample_expression_crispr_full`)
- 출력: `metabric_validation_summary.csv` (model별 `RMSE`, `MAE`, `Spearman`)
- 정렬 방식: `canonical_drug_id` 기준 drug-level score를 외부 sample pair에 투영
- 상태: **METABRIC external validation 완료**
- 요약 수치: **METABRIC evaluation pairs = 2648**
- 해석: 외부 검증에서는 `GCN`이 Spearman/RMSE 모두 1위를 기록해 일반화 신호가 가장 강했음

**ADMET filtering**

- 출력: `admet_filter_log.json`
- 이번 run 메모: candidate drug ID와 직접 조인 가능한 분자 ADMET 테이블 부재로, **direct molecular lookup이 아닌 proxy rule-based screening** 적용
- 요약 수치: **ADMET passed = 53 / 100**
- 의미: Top100 후보 중 53개가 1차 게이트를 통과했으며, 현재 단계는 **proxy screening**이므로 direct molecular ADMET 확증 이전의 pre-filter 단계임

**Final shortlist**

- 조건: ADMET 통과 + ranking 유지
- 출력: `final_shortlist.csv` (Top30)
- 상태: **Top 30 행 생성 완료** (파이프라인 스냅샷)
- **운영 정책 (팀 합의):** 대외·최종 후보로 쓰는 약물은 **FDA 승인 의약품만** 허용. 본 CSV는 **승인 필터를 아직 적용하지 않음** — OpenFDA·Orange Book·RxNorm 등과 조인한 **승인약 전용 shortlist**로 교체하기 전까지는 “최종 후보”가 아님.
- 파일 컬럼 개선: `final_shortlist.csv`에 `rank_v2`, `admet_pass`, `bucket` alias 컬럼 추가 (기존 원본 컬럼 유지)

핵심 컬럼(요약): `canonical_drug_id`, `drug_rank_v2`(=`rank_v2`), `ensemble_score_v2`, `shortlist_bucket`(=`bucket`)

**METABRIC 단계 모델 비교 (외부 + 내부 기준 함께 보기)**

| model | METABRIC Spearman | METABRIC RMSE | 내부 TCGA/CV NDCG@20(mean) | 해석 |
|------|-------------------:|--------------:|----------------------------:|------|
| GCN | **0.4045 (1위)** | **1.2732 (1위)** | 0.7452 | 외부 일반화 성능 우수 |
| Ensemble_v2 | 0.0742 (2위) | 2.7014 (2위) | N/A | 단일 GCN 대비 낮지만 절충 성능 |
| ResidualMLP | -0.0556 (3위) | 3.3210 (3위) | **0.8401 (공동 1위)** | 내부 대비 외부 성능 하락 |
| XGBoost | -0.0711 (4위) | 7.1289 (4위) | **0.8401 (공동 1위)** | 내부 대비 외부 성능 하락폭 큼 |

> 참고: 이번 외부 요약 파일(`metabric_validation_summary.csv`)에는 NDCG가 포함되지 않아, NDCG 비교는 내부 TCGA/CV 결과를 함께 병기함.

**"진짜 METABRIC 검증" 체크리스트 (운영 기준)**

- 필수 컬럼: `canonical_drug_id`, external sample key, target label, prediction score
- 라벨 정합: 내부/외부의 label 정의·스케일·방향성 동일성 확인
- 정렬 키 품질: join cardinality(one-to-one/one-to-many)와 drop/mismatch 로그 저장
- 누수 점검: train에 쓰인 sample/drug가 external 평가에 중복되지 않음을 증빙
- 지표 완결: `Spearman`, `Kendall tau`, `TopK overlap@K`, (가능 시) `NDCG@K`

**내부↔외부 rank consistency 진단 운영**

- 현재 상태: 외부 요약 파일이 aggregate(`metabric_validation_summary.csv`) 중심이라 `Kendall/TopK overlap` 계산용 raw external rank table이 없음
- 다음 실행 권장 산출: `model, canonical_drug_id, internal_rank, external_rank, internal_score, external_score`
- 신뢰도 레벨:
  - `A`: 진짜 METABRIC split + 누수검증 통과 + Spearman/Kendall/TopK 완결
  - `B`: 외부 라벨 정합 확인 + Spearman 중심 + 일부 일관성 지표
  - `C`: proxy external validation (현재 단계)

**METABRIC true validation — 샘플 브릿지 및 파이프라인 (저장 위치 고정)**

- 외부 원천(`s3://.../8/metabric/`)과 혼동되지 않도록, **모든 산출은** `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/` 하위에만 둡니다.
- 후보 테이블: `build_manual_bridge_candidates.py` → `bridge_candidates/manual_bridge_candidates.csv`
- 확정 브릿지: `finalize_metabric_sample_bridge.py` (`manual_selected=True` 행만) → `bridge_finalized/finalized_metabric_sample_bridge.csv` + `finalized_metabric_sample_bridge_summary.json` (중복 MB/internal, 커버리지, `status`)
- 검증: `validate_metabric_sample_bridge.py --format finalized --bridge-csv .../finalized_metabric_sample_bridge.csv`
- 오케스트레이션: `run_metabric_true_validation_pipeline.py` (저장소 루트에서 실행) → `pipeline/metabric_true_validation_pipeline_manifest.json` (다음 단계: 외부 pair FE + 추론)

**METABRIC-native cohort FE · external inference 정책 (20260402 확정)**

- **공식 external inference ranking (운영 기준):**  
  `fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features_run/metabric_native_ensemble_ranking_no_lincs.csv`  
  (입력 피처: `metabric_native_pair_features_no_lincs.parquet`, `build_metabric_native_pair_features.py --omit-lincs-features`)
- **with-LINCS native ranking:** `metabric_native_ensemble_ranking.csv` 등 LINCS가 FE parquet에 포함된 경로는 **레거시·비권장**, 동일 조건 **비교·감사용**으로만 유지.
- **with-LINCS vs no-LINCS 랭킹 비교에서 차이가 0에 가까웠던 이유:**  
  - **XGBoost:** 학습 번들의 `feature_columns`에 LINCS 열이 없어, 외부 추론에서도 LINCS를 쓰지 않음.  
  - **ResidualMLP / GCN:** 체크포인트 `feat_cols`에는 LINCS 슬롯이 있으나, `build_final_ensemble_ranking.py`는 피처 테이블에 없는 열을 **0으로 채워** 텐서 너비를 맞춤. with-LINCS native parquet에서도 학습 시와 동일하게 **LINCS 구간이 0**이면 no-LINCS parquet과 **동일 입력**이 되어 랭킹이 일치할 수 있음.  
  - 정리: **리포트·TopK 운영은 no-LINCS 산출물을 단일 기준**으로 쓰고, 위 이유는 “수치가 같아 보여도 정책적으로 no-LINCS를 채택한다”는 근거로 문서화함.
- **비교 산출:** `compare_metabric_native_rankings_no_lincs.py` → `metabric_native_ranking_comparison_no_lincs.csv`, `metabric_native_ranking_comparison_summary.json` (`policy_final_external_inference_ranking` 키 포함).
- **FDA drug universe 랭킹 (재추론):** `ml/pilot_sagemaker/run_fda_only_metabric_ranking.py` — Drugs@FDA(+선택 DrugBank) CID 유니버스 × METABRIC 샘플 → 동일 native FE·동결 앙상블 → `fda_only_universe/fda_only_ranking.csv`, `fda_top30_shortlist.csv`. S3·행렬이 없을 때는 `backfill_fda_only_ranking_from_native.py`로 관측 쌍은 native 점수 유지·미관측 쌍은 pred 평균 impute(데모용).
- **Direct ADMET (Top30 → hard/soft → Top10):** `ml/pilot_sagemaker/evaluate_fda_shortlist_direct_admet.py` — PubChem SMILES, RDKit `swissadme_proxy_*`, TCGA `final_shortlist.csv`의 `admet_pass` 조인(가능 시), hERG·간독성 **프록시** hard gate, `soft_composite_score`로 순위 → `fda_only_universe/direct_admet_results.csv`, `admet_combined_summary.csv`, `final_top10_fda_drugs.csv`. SwissADME·ADMETlab 웹/API 전체 배치는 포털로 보완(스크립트 내 메모 컬럼 참고).
- **SwissADME 웹 배치 → 병합 → 1차 ADME 랭킹 (FDA Top15 스냅샷):** `build_swissadme_input_from_fda_shortlist.py`로 `swissadme_input_top30.txt`·`swissadme_input_top30_with_meta.csv` 생성 → SwissADME 사이트에서 결과 저장 후 `merge_swissadme_web_results.py --swissadme-export <다운로드.csv>`로 `swissadme_web_merged_n*.csv` 생성 → `rank_swissadme_adme_two_stage.py`로 **Hard filter** (Lipinski≥2, GI Low, PAINS, Brenk≥2 제거) 및 **Soft** (`adme_final_score = ensemble − 0.2×penalty + 0.1×F_bonus`) → `swissadme_adme_hard_pass.csv`, `swissadme_adme_soft_top10.csv`, `swissadme_adme_report.json`. **2차:** `admetlab_stage2_queue.csv`에 담긴 통과 후보만 ADMETlab(독성·대사) 배치 후 재필터; **최종 후보**는 SwissADME+ADMETlab 모두 통과분을 모델 점수와 함께 정렬.
- **다음 단계 (no-LINCS 기준):** 유니버스 cap 해제 후 전체 FDA 후보 FE·SwissADME/ADMETlab 배치·컷오프 합의.
- **별도 실험 트랙 (메모, 미착수):** LINCS를 **모델 입력에 포함**하도록 XGB / ResidualMLP / GCN **전부 재학습**(피처 스키마·스케일러·그래프 헤드 차원 정합 포함)하는 **with-LINCS full retraining** — 현재 운영 정책과 분리된 후속 연구 옵션으로만 기록.

### A/B/C 파일럿: ML 4종 병렬 Training Job

- **전제:** 아래 기본 버킷·실행 역할·SageMaker 기본 버킷(`sagemaker-ap-northeast-2-666803869796/…`)은 **팀 프로젝트(팀4) 공용 인프라**를 가정한다. 개인 AWS 계정 기준이 아니다. 다른 팀·계정이면 스크립트의 `--role`, `--code-bucket`, `--sagemaker-account-id`, 데이터 URI 오버라이드 등을 맞출 것.
- 입력(프리셋): `abc_inputs/20260330_abc_v1/` — **A** `A/features.parquet`, **B** `B/features_b.parquet`, **C** `C/features.parquet` (각각 `labels.parquet` 동일 폴더).
- 코드: `ml/pilot_sagemaker/train_tabular.py` (관리형 **PyTorch** 이미지를 Python 런타임으로만 사용)
- C 실행 시(`--dataset c`) 학습 스크립트가 `drug__smiles`를 문자 n-gram 해시 피처(기본 2048차원)로 변환해 숫자 피처와 함께 학습한다. A/B는 기본적으로 SMILES 해시를 끈다.
- 제출 (동일 스크립트, 데이터셋만 변경):

  - B(기본): `python3 ml/pilot_sagemaker/submit_b_parallel.py`
  - A: `python3 ml/pilot_sagemaker/submit_b_parallel.py --dataset a`
  - C: `python3 ml/pilot_sagemaker/submit_b_parallel.py --dataset c`

- 산출 S3 prefix (기본): `s3://sagemaker-ap-northeast-2-666803869796/{team-tag}-pilot-train/output/{a|b|c}/` — 기본 `--team-tag team4`. 실행 역할이 팀 버킷에 `PutObject` 없어도 동작. 팀 버킷에 쓰려면 `--use-team-output-prefix` (역할 권한·KMS 필요). 팀 경로: A/C는 `.../sagemaker/{a|c}_pilot_4models`, **B는 `.../sagemaker/team4_b_pilot_runs`** (B 산출만 한 폴더로 묶음).
- **스테이징(재발 방지)**  
  - 원본 parquet는 팀 버킷에 두고, 제출 시 로컬·CLI 자격으로 **`aws s3 cp … --sse AES256`** 로 SageMaker 기본 버킷 `…/{team-tag}-pilot-train/data/{a|b|c}/<timestamp>/features.parquet` 등으로 복사한 뒤 그 URI로 학습한다. (실행 역할이 팀 버킷 `GetObject` 없을 때 Training Job이 데이터에서 죽는 문제 방지.)  
  - 학습 코드 tarball도 동일하게 **AES256 업로드** 후 `source_dir`로 넘긴다. (팀 객체가 SSE-KMS일 때 실행 역할 **kms:Decrypt** 없어 sourcedir 다운로드가 실패하는 문제 방지.)
- 각 잡 이름 예: `{team-tag}-a-pilot-lightgbm-<ts>` (기본 `team4-…`). 팀 구분만 바꿀 때 `--team-tag`(또는 환경 변수 `SAGEMAKER_PILOT_TEAM_TAG`). 잡 종료 후 `model.tar.gz` 안에 `metrics.json`, `artifact.joblib` (CloudWatch에도 `[METRICS]`).

**실패 시 점검:** 팀 버킷 원본 경로가 없거나 로컬 `aws s3 cp` 자격이 없으면 스테이징 단계에서 먼저 실패한다. Training Job은 **스테이징된** S3 객체만 읽는다.

### ABC + SMILES 실험 결과 대시보드

- **친구·팀 공유 — GitHub Pages (브랜치 배포, 방법 1)**  
  GitHub 계정 없이 브라우저만 있으면 볼 수 있다. 저장소를 **Public**으로 두는 것을 권장한다(비공개면 무료 플랜에서 Pages·접근이 막힐 수 있음).

  **공유용 링크 (Pages 켠 뒤 아래 그대로 전달):**

  - 대시보드 직링크: [https://skkuaws0215.github.io/20260330_pre_project_4w_test-1/abc_smiles_experiment_dashboard.html](https://skkuaws0215.github.io/20260330_pre_project_4w_test-1/abc_smiles_experiment_dashboard.html)  
  - 목차(루트): [https://skkuaws0215.github.io/20260330_pre_project_4w_test-1/](https://skkuaws0215.github.io/20260330_pre_project_4w_test-1/)

  **저장소에서 한 번만 설정:** **Settings** → **Pages** → **Build and deployment** → **Source:** **Deploy from a branch** → Branch **main**, Folder **/ (root)** → **Save**.  
  1~2분 후 위 링크가 열리면 된다. **404**면 이 단계가 아직 안 된 것이다. (루트 `.nojekyll`로 Jekyll이 정적 HTML을 건너뛴다.)

  **참고:** 같은 Pages에 **GitHub Actions** 배포도 쓰려면 Source를 **한 가지**만 선택한다. 방법 1만 쓸 때는 **Deploy from a branch**만 켜 두면 된다.
- 파일: `abc_smiles_experiment_dashboard.html`
- 범위: A/B/C baseline + C 재실험(SMILES feature enabled) 진행 현황/성능 비교
- 핵심: baseline(A/B/C old) 대비 new-C(3개 모델 완료 기준)에서 valid RMSE 하락, Spearman 상승 확인
- 주의: 현재 new-C는 `drug__smiles`를 **char n-gram hash(2048)** 로 변환한 실험이며, RDKit descriptor/fingerprint 실험은 별도 후속 단계

### New FE(20260331) 품질 점검 대시보드

- 파일: `newfe_quality_dashboard_20260331.html`
- 범위: `fe_re_batch_runs/20260331/final` 산출물 품질 점검
  - row/col, key uniqueness, null ratio, zero-variance, numeric 분포 요약
  - `pair_features_newfe` vs `pair_features_newfe_v2` 컬럼 diff
  - `pair_target_features` 0 비율, `pair_lincs_features` constant-like 비율
- 현재 해석:
  - 비교 가치 높음: `drug_chem_features.parquet`, `pair_features_newfe.parquet`
  - 신호 약함: `pair_lincs_features.parquet`, `pair_target_features.parquet`
  - 원인: LINCS proxy 입력 / target symbol 체계 mismatch
- 다음 단계 필수:
  1. `sig_id(BRD) -> canonical_drug_id` 정규화
  2. `UniProt -> gene symbol(or entrez)` 정규화

#### 다음 액션(1단계): 4모델 결과 정밀분석

- 대시보드 `3-1) 1단계 정밀 분석 프레임` 기준으로 모델별 추가 지표를 채운다.
- 최소 채움 항목: `RMSE`, `MAE`, `R2`, `Spearman`, 학습 시간, residual 요약(p90 abs error).
- 데이터셋 영향 표를 반드시 채운다: `A vs B`, `C old vs C new`, `miss30 vs miss70`.
- FE 상세 설명은 모델별로 구분한다: LGBM/XGB(importance), ElasticNet(계수), RF(중단 또는 경량 재시도 근거).
- 1단계 완료 후에만 “기존 results 고정 vs 전처리 재실행” 의사결정을 진행한다.
- 입력 템플릿: `ml/pilot_sagemaker/analysis_templates/` (`model_metrics_template.csv`, `dataset_impact_template.csv`, `ANALYSIS_TEMPLATE.md`)

#### 20260331 Target-only 비교 실행 결과 (LINCS 제외, 동일 split)

- 실행 스크립트: `ml/pilot_sagemaker/run_target_only_comparison.py`
- 고정 조건:
  - 실험셋: `baseline`, `newfe`, `newfe_v2(target-only)`
  - 모델: `LightGBM`, `XGBoost`, `ElasticNet`
  - split: `seed=42`, `test_size=0.2`
  - 공통 key 교집합만 사용: `14497` rows
  - `normalize_lincs_mapping.py` 미실행 + LINCS 컬럼 제외
- 결과 파일:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/model_dataset_metrics.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/model_delta_summary.csv`

| model | dataset | RMSE | MAE | Spearman | NDCG@20 | Hit@20 |
|------|---------|------|-----|----------|---------|--------|
| LightGBM | baseline | 2.483 | 1.927 | 0.264 | 0.736 | 1.000 |
| LightGBM | newfe | 2.119 | 1.596 | 0.441 | 0.832 | 1.000 |
| LightGBM | newfe_v2 | 2.111 | 1.593 | 0.441 | 0.836 | 1.000 |
| XGBoost | baseline | 2.485 | 1.930 | 0.263 | 0.736 | 1.000 |
| XGBoost | newfe | 2.119 | 1.595 | 0.437 | 0.831 | 1.000 |
| XGBoost | newfe_v2 | 2.107 | 1.585 | 0.445 | 0.836 | 1.000 |
| ElasticNet | baseline | 2.491 | 1.938 | 0.262 | 0.728 | 1.000 |
| ElasticNet | newfe | 2.126 | 1.611 | 0.435 | 0.830 | 1.000 |
| ElasticNet | newfe_v2 | 2.123 | 1.607 | 0.437 | 0.832 | 1.000 |

Delta 요약:

| model | delta_newfe_spear | delta_target_spear | delta_target_ndcg |
|------|--------------------|--------------------|-------------------|
| LightGBM | 0.177 | 0.001 | 0.004 |
| XGBoost | 0.174 | 0.009 | 0.005 |
| ElasticNet | 0.173 | 0.002 | 0.002 |

해석:

- `baseline -> newfe`: 3모델 모두 큰 개선.
- `newfe -> newfe_v2`: target feature 추가로 소폭 추가 개선.
- target feature 추가 효과(특히 Spearman/NDCG)는 `XGBoost`가 가장 큼.
- 현재 best model 후보: `XGBoost + newfe_v2(target-only)`.

#### 20260401 DL baseline + MLP Variants 비교 결과 (로컬, 동일 split)

- DL 대시보드: `dl_experiment_dashboard_20260331.html` (`Snapshot · 20260401_v1`)
- 공통 조건:
  - 데이터셋: `pair_features_newfe_v2.parquet` (LINCS 제외)
  - key inner join: `sample_id`, `canonical_drug_id`
  - split: `seed=42`, `test_size=0.2`
  - 전처리: 결측 `0`, continuous `standard scaling`, binary passthrough
- 비교 모델:
  - `XGBoost` (best ML baseline)
  - `FlatMLP` (기존 baseline)
  - `BlockWiseMLP` (pathway/chem/target block encoder)
  - `ResidualMLP` (flat MLP + residual blocks)
- 결과 파일:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/final_comparison/xgb_mlp_vae_tabnet_final_comparison.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/final_comparison/xgb_mlp_vae_tabnet_final_comparison.md`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_variants/mlp_variants_metrics.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_variants/xgb_mlp_variants_comparison.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_variants/mlp_variants_learning_curve.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/mlp_variants/run_summary.json`

| model | RMSE | MAE | Spearman | NDCG@20 | Hit@20 | comment |
|------|------|-----|----------|---------|--------|---------|
| XGBoost | 2.107257 | 1.584793 | 0.445384 | 0.835994 | 1.000000 | best ML baseline |
| FlatMLP | 2.112787 | 1.598048 | 0.440021 | 0.836289 | 1.000000 | DL baseline |
| BlockWiseMLP | 2.106591 | 1.583276 | 0.445905 | 0.837439 | 1.000000 | flat 대비 소폭 개선 |
| ResidualMLP | 2.103150 | 1.582552 | 0.452853 | 0.839462 | 1.000000 | 현재 MLP 계열 best |

해석:

- 구조 변경만으로도(`Flat -> BlockWise/Residual`) 성능 개선이 확인됨.
- `ResidualMLP`는 주지표 Spearman 기준으로 `XGBoost`를 소폭 상회.
- 과도한 튜닝 없이도 DL 구조 개선 효과가 확인되어, 다음 단계(예: graph 전환 전 최종 DL 확정)에 유의미한 근거를 제공.

#### 20260401 ResidualMLP 5-fold CV 검증 (XGBoost tuned CV 동일 조건 비교)

- 실행 스크립트: `ml/pilot_sagemaker/run_residual_mlp_cv_local.py`
- 조건:
  - 데이터: `pair_features_newfe_v2.parquet` (LINCS 제외)
  - key join: `sample_id`, `canonical_drug_id` inner join
  - split: `5-fold KFold(shuffle=True, random_state=42)`
  - 전처리: 결측 0, continuous standard scaling, binary passthrough
  - 비교 기준: tuned XGBoost CV (`max_depth=4`, `learning_rate=0.05`, `n_estimators=400`, `subsample=0.8`, `colsample_bytree=0.8`)
- 출력 파일:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/residual_mlp_cv/residual_mlp_cv_fold_metrics.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/residual_mlp_cv/residual_mlp_cv_summary.json`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/residual_mlp_cv/xgb_vs_residual_mlp_cv_comparison.csv`

| model | RMSE_mean | RMSE_std | MAE_mean | MAE_std | Spearman_mean | Spearman_std | NDCG@20_mean | NDCG@20_std | Hit@20_mean | Hit@20_std |
|------|-----------|----------|----------|---------|---------------|--------------|--------------|-------------|-------------|------------|
| XGBoost_tuned_cv | 2.133085 | 0.040457 | 1.594438 | 0.021274 | 0.430751 | 0.008948 | 0.839619 | 0.004367 | 1.000000 | 0.000000 |
| ResidualMLP_cv | 2.128796 | 0.041159 | 1.588846 | 0.019151 | 0.436680 | 0.010579 | 0.840609 | 0.003035 | 1.000000 | 0.000000 |

해석:

- holdout 단일 split에서 보인 우세가 5-fold CV에서도 재확인됨.
- `ResidualMLP`는 Spearman/RMSE/MAE/NDCG@20에서 tuned XGBoost CV 대비 소폭 우세.
- `BlockWiseMLP`는 block grouping 완성도 이슈로 현재 후순위, Residual 계열을 우선 확장 대상으로 유지.

#### DL 군 대표 선정용 통합 비교 (동일 holdout, 5모델)

- 목적: 최종 단일 모델이 아니라 **ML / DL / Graph 군별 대표 1–2종**을 고르기 위한 DL 군 정리.
- 실행 스크립트: `ml/pilot_sagemaker/run_dl_family_comparison_local.py`
- 비교 모델: `FlatMLP`, `VAE` (latent 32·64 중 Spearman 최고를 대표 행으로 사용), `TabNet`, `BlockWiseMLP`, `ResidualMLP`
- 출력:
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/dl_family/dl_family_comparison.csv`
  - `results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/dl_family/dl_family_summary.json`
- 대시보드: `dl_experiment_dashboard_20260331.html` → **6) DL family final comparison**

#### 5-fold CV — XGBoost(tuned) vs MLP 3종 (동일 폴드)

- 스크립트: `ml/pilot_sagemaker/run_xgb_mlp3_cv_local.py`
- 비교: `XGBoost_tuned`, `FlatMLP`, `BlockWiseMLP`, `ResidualMLP`
- 출력 (동일 `dl_family/` 폴더):
  - `xgb_mlp3_cv_fold_metrics.csv`
  - `xgb_mlp3_cv_comparison.csv`
  - `xgb_mlp3_cv_summary.json`
- 대시보드: `dl_experiment_dashboard_20260331.html` → **7)**
- 대시보드: `dl_experiment_dashboard_20260331.html` → **8)** 의사결정 참고 Agent 요약 의견 (DL 채택·제외·ML/Graph 관계 주석)

#### 로컬 검증·문서화 완료 — 군별 대표 & SageMaker 최종 3종

- **확정 대표:** ML **XGBoost** (tuned) · DL **ResidualMLP** · Graph **GCN**.
- **Graph 정책:** Spearman mean 기준 **drug-group CV**에서 GCN이 앞섬. Round1 **행 단위 CV**에서는 GraphSAGE가 잠정 1위였으나 optimistic bias 가능 → **GraphSAGE는 temporary candidate로만** 기록하고, **최종 Graph 대표는 GCN**.
- **GCN 경량 튜닝 (동일 drug-group CV, A–D):** 튜닝 **완료**. **Graph representative는 GCN 유지.** 그리드 내 Spearman mean 최고는 **구성 D**(weight_decay 1e-4)이나, baseline **A** 대비 개선폭은 약 **+0.0054**로 팀 기준(**≥ +0.01**)에 **미달** → **의미 있는 개선으로 보지 않음** → **최종 Graph 군 대표로 쓰는 GCN은 baseline 하이퍼파라미터(A: hidden 64, lr 1e-3, weight_decay 1e-5) 유지.** **구성 E**(96 / 5e-4 / 5e-5)는 **이번 단계에서 보류.** 근거: `graph_baseline_round1/gcn_tuning_summary.json`, `gcn_tuning_comparison.csv` · 스크립트 `ml/pilot_sagemaker/run_gcn_groupcv_tuning.py`.
- **한 장 비교표·지표·검증 타입·SageMaker 입력/설정/산출물 체크리스트:** `dl_experiment_dashboard_20260331.html` **0)**.
- **SageMaker 통합 실험 대시보드:** `sagemaker_experiment_dashboard_20260403.html` — 스냅샷 **20260403_v9**. **운영 정책:** 최종 후보는 **FDA 승인 의약품만**(§4.3). 현재 `final_shortlist.csv` Top30은 **승인 필터 미적용** 스냅샷(감사용). 로컬 기준선·`sagemaker_final_three`·§4.1–4.6. **§4.4** METABRIC proxy 콜아웃, METABRIC HTML 링크.
- **METABRIC · ADMET · Survival-linked 대시보드:** `metabric_admet_survival_dashboard_20260402_v2.html` — **§2**에 **FDA 승인약만** 운영 정책 명시; 동 절 표는 **승인 필터 전** Top30 + PubChem명. **§3** 생존·KM. **§4–§6** proxy. **§7** 파이프라인. 승인 조인·재산출 전에는 §2·§4.3 목록을 최종 후보로 쓰지 않음.
- **수치 원본:** `analysis_target_only/residual_mlp_cv/residual_mlp_cv_summary.json` (ML/DL 행 5-fold mean); `graph_baseline_round1/graph_family_groupcv_summary.json` (GCN group 5-fold mean, 3종 대표 선정). GCN 하이퍼 그리드 결론은 **`graph_baseline_round1/gcn_tuning_summary.json`** (영문 policy 필드·`notes` 포함). 행 간 숫자는 검증 정의가 다르므로 직접 승패 비교 시 주의.

#### Graph 군 Round1 — Network Proximity · GraphSAGE · GCN (동일 스키마·동일 `cv_fold_indices.json`)

- 대시보드: `graph_experiment_dashboard_20260401.html`
- **로컬 기본 경로(저장소 루트 기준):**
  - 라벨: `results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet` (B `labels`와 동일 스키마, `final_pathway_addon` 피처와 inner join 시 **n=14497** = `cv_fold_indices.json`과 정합)
  - 피처: `.../final_pathway_addon/pair_features_newfe_v2.parquet`
  - drug–target: `.../input_refs/drug_target_map_20260331.parquet`
- 공통 병합: `graph_baseline_data.load_merged_pair_frame`, 회귀 라벨 `label_regression`, CV는 `model_selection_stage1/cv_fold_indices.json`.
- **Network Proximity:** `run_network_proximity_baseline.py` — **rule-based, non-sample-specific** drug-level z-score. 검증 폴드에서 **RMSE/MAE**는 train에서 적합한 `y ~ a·z+b`로 스케일 맞춘 예측으로 계산하고, **Spearman/NDCG/Hit**는 원시 z로 계산(`graph_schema.json`의 `proximity_calibration` 참고).
- **GraphSAGE / GCN:** `run_graph_gnn_cv.py --model sage|gcn` → `graph_gnn_*_partial.csv`.
- **병합:** `merge_graph_family_outputs.py` → `graph_family_comparison.csv`, `graph_family_summary.json`, 그리고 ML/DL 대표와 한 줄 비교용 **`ml_dl_graph_family_mean.csv`** (소스: `analysis_target_only/residual_mlp_cv/residual_mlp_cv_summary.json`의 XGBoost·ResidualMLP + Graph 대표).
- **Round1 vs 최종:** `graph_family_summary.json`에 Round1 기준 `temporary_graph_representative_candidate: "GraphSAGE"`(행 단위 CV 이력). **최종 Graph 군 대표는 `graph_family_groupcv_summary.json` 기준 GCN**으로 확정(SageMaker Graph 슬롯).
- `s3://` URI는 `s3fs` 설치·자격 증명이 있으면 그대로 사용 가능.
- Python: `ml/pilot_sagemaker/requirements.txt` (`torch`, `s3fs`).

#### Graph 군 drug-group CV (stricter, `canonical_drug_id`)

- **목적:** 행 단위 `KFold`에서는 동일 약물이 train·valid에 동시에 올 수 있어 GNN drug node 표현이 낙관적으로 보일 수 있음 → **`GroupKFold(n_splits=5, shuffle=True, random_state=42)`** 로 재분할 후 동일 3종(Proximity, GraphSAGE, GCN)·동일 지표로 재평가.
- **폴드 정의:** `ml/pilot_sagemaker/build_cv_fold_indices_drug_group.py` → `graph_baseline_round1/cv_fold_indices_drug_group.json`.
- **일괄 실행:** `ml/pilot_sagemaker/run_graph_groupcv_pipeline.py` (Proximity → sage/gcn → `merge_graph_family_outputs.py --preset groupcv`).
- **산출물 (기본 디렉터리 `.../graph_baseline_round1/`):**
  - `graph_family_groupcv_comparison.csv`, `graph_family_groupcv_summary.json`
  - `graph_gnn_sage_groupcv_partial.csv`, `graph_gnn_gcn_groupcv_partial.csv`
  - `ml_dl_graph_family_mean_groupcv.csv`, `graph_schema_groupcv.json`
- **Proximity 보정:** drug-group 분할에서 train-only `y ~ a·z+b`가 불안정할 수 있어, 기울기 상한·비정상 RMSE 시 train 평균 예측 폴백 (`run_network_proximity_baseline.py`).
- **요약 JSON:** `recommended_graph_representative`(Spearman mean 최대), `temporary_graph_representative_candidate`, `representative_finalization_policy`, `gnn_transductive_caveat` 등.
- **대시보드:** `graph_experiment_dashboard_20260401.html` (섹션 8 drug-group CV + GCN 미니튜닝 표·결론, 결과 표 아래 <strong>펼치기(details)</strong>에 지표 정의 표·요약), DL 대시보드 `dl_experiment_dashboard_20260331.html`에서 Graph 쪽 교차 링크.

**대시보드 HTML이 “안 열릴” 때:** GitHub **`blob` URL**은 HTML을 **렌더하지 않습니다**. 다른 폴더에서 `python3 -m http.server`만 실행하면 **루트 HTML이 없어** 빈 화면·404가 납니다.

**1) 로컬·동일 Wi‑Fi 공유 (가장 확실):** 저장소 **루트**에서:

```bash
python3 serve_dashboards.py
```

터미널에 `http://127.0.0.1:8765/` 및 LAN 주소가 출력되고, 브라우저가 **`index.html` 목차**를 연 뒤 DL/Graph 링크를 누르면 됩니다.  
Streamlit을 쓰는 경우: `streamlit run streamlit_app.py` → 사이드바 **「실험 대시보드 (DL/Graph HTML)」** 에서 동일 안내·링크.

**2) 팀에 고정 URL (GitHub Pages):** 저장소 **Settings → Pages → Build and deployment → Source: GitHub Actions** 로 두면, 푸시 시 `.github/workflows/deploy-dashboards.yml` 이 루트의 `*.html` 을 배포합니다.  
배포 후 주소 형식: `https://skkuaws0215.github.io/20260330_pre_project_4w_test-1/index.html` (첫 설정·첫 배포가 끝나야 열립니다.)

**3) `htmlpreview.github.io`:** 망·서비스 이슈로 실패하는 경우가 많아 의존하지 않는 것을 권장합니다.

**소스만 볼 때:** `https://github.com/skkuaws0215/20260330_pre_project_4w_test-1/blob/main/graph_experiment_dashboard_20260401.html` 등.

#### Pathway (Hallmark GMT) — `pathway__*` 보강 및 BlockWise 정식 블록 CV

- **GMT 파일:** `nextflow/refs/h.all.v7.5.symbols.gmt` (MSigDB Hallmark v7.5 symbols). `build_pair_features_newfe_v2.py`에 `--pathway-gmt`를 주지 않으면 `sample_pathway_features.parquet`에 `pathway__*` 점수 열이 **0개**로 남는다. 표현 행렬 열이 `crispr__TP53`처럼 접두어가 있어도 **열 이름 끝 심볼**과 GMT를 맞추도록 빌드 스크립트가 처리한다.
- **SMILES:** 정식 pair 피처는 `--drug-uri`에 **SMILES 포함** drug 테이블이 필요하다. pathway만 덧붙일 때는 기존 SMILES 기반 `final/pair_features_newfe_v2.parquet`에 `sample_pathway_features`의 `pathway__*`만 `sample_id`로 merge한 **`fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet`** 를 사용한다 (`merge_pathway_into_pair_features.py`, 감사 `final_pathway_addon/pathway_merge_audit.json`).
- **5-fold CV (pathway 50개, BlockWise pathway 블록 > 0):** `ml/pilot_sagemaker/run_xgb_mlp3_cv_local.py` — 산출 `analysis_target_only/xgb_mlp3_cv_pathway_addon/` (요약 표: 동 폴더 `comparison_vs_baseline.md`). 이전 `dl_family/xgb_mlp3_cv_summary.json`(pathway 0)과 동일 폴드 설정으로 baseline 대비 비교 가능.
- **운영·재현 상세:** `nextflow/README.md` → **「Pathway GMT·SMILES·학습용 addon (정리)」**.

## `credentials` 작성 시 주의

- 파일 상단에 **`#` 주석을 넣지 않는 것**을 권장합니다. (일부 환경에서 AWS CLI 파싱 오류가 날 수 있음)
- 형식은 다음 두 줄과 `[4team-project]` 섹션만 유지하면 됩니다.

  - `aws_access_key_id`
  - `aws_secret_access_key`

## 보안

- `.aws/credentials`는 **절대 저장소에 커밋하지 마세요.** (`.gitignore`에 포함됨)
- 키가 유출되었을 가능성이 있으면 IAM에서 해당 액세스 키를 비활성화하고 새로 발급하세요.
