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

#### 다음 액션(1단계): 4모델 결과 정밀분석

- 대시보드 `3-1) 1단계 정밀 분석 프레임` 기준으로 모델별 추가 지표를 채운다.
- 최소 채움 항목: `RMSE`, `MAE`, `R2`, `Spearman`, 학습 시간, residual 요약(p90 abs error).
- 데이터셋 영향 표를 반드시 채운다: `A vs B`, `C old vs C new`, `miss30 vs miss70`.
- FE 상세 설명은 모델별로 구분한다: LGBM/XGB(importance), ElasticNet(계수), RF(중단 또는 경량 재시도 근거).
- 1단계 완료 후에만 “기존 results 고정 vs 전처리 재실행” 의사결정을 진행한다.
- 입력 템플릿: `ml/pilot_sagemaker/analysis_templates/` (`model_metrics_template.csv`, `dataset_impact_template.csv`, `ANALYSIS_TEMPLATE.md`)

## `credentials` 작성 시 주의

- 파일 상단에 **`#` 주석을 넣지 않는 것**을 권장합니다. (일부 환경에서 AWS CLI 파싱 오류가 날 수 있음)
- 형식은 다음 두 줄과 `[4team-project]` 섹션만 유지하면 됩니다.

  - `aws_access_key_id`
  - `aws_secret_access_key`

## 보안

- `.aws/credentials`는 **절대 저장소에 커밋하지 마세요.** (`.gitignore`에 포함됨)
- 키가 유출되었을 가능성이 있으면 IAM에서 해당 액세스 키를 비활성화하고 새로 발급하세요.
