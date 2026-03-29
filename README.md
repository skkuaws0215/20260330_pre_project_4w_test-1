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
| `nextflow/` | Nextflow용 폴더 — S3 prefix 안내 `s3_features_nextflow_team4_README.txt`, 업로드 방법 `README.md` |
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
| 전처리 완료 | `s3://drug-discovery-joe-raw-data-team4/results/` |
| **Nextflow 피처 산출 (팀4 합의)** | `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/` — 전처리 `results/<소스>/` 와 구분, 다른 팀원 FE 경로와 충돌 방지 |

CLI 예시(`use-team-aws.ps1` 실행 후):

```powershell
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/"
aws s3 ls "s3://drug-discovery-joe-raw-data-team4/results/" --recursive | findstr /i ".parquet"
```

### `results/` 스냅샷 (조회: 2026-03-29)

**1단계 prefix** (`aws s3 ls …/results/`): `admet/`, `chembl/`, `cross_platform/`, `depmap/`, `drugbank/`, `gdsc/`, `id_mapping/`, `lincs/`, `metabric/`, `msigdb/`, `opentargets/`, `pubmed/`, `string/`, `tcga/` 및 루트 `final_qc.csv`.

**Parquet 22개** (Nextflow 입력 설계 시 경로 참고 — 용량 큰 파일 주의):

| 경로 (버킷 내 `results/` 기준) | 비고 |
|--------------------------------|------|
| `tcga/01_raw_counts.parquet` ~ `07_qc_passed.parquet` | 대용량 (~0.9–1.0GB 단위) |
| `metabric/08_raw.parquet`, `54_filtered.parquet` | ~238MB |
| `lincs/16_mcf7_processed.parquet` | ~7.6GB |
| `chembl/15_activities.parquet`, `drugbank/14_drug_targets.parquet`, `gdsc/21_ic50.parquet`, `22_binary_labels.parquet` | |
| `string/19_ppi_filtered.parquet`, `msigdb/20_gene_sets.parquet`, `depmap/57_crispr.parquet`, `58_drug_labels.parquet` | |
| `cross_platform/11_intersection.parquet`, `id_mapping/55_harmonized.parquet`, `admet/25_admet_results.parquet` | |
| `opentargets/26_associations.parquet`, `pubmed/23_corpus.parquet` | |

첫 Nextflow 업로드 시 **`results/features_nextflow_team4/`** 아래에 `README.txt` 를 두어 팀에 prefix를 알리는 것을 권장. 로컬 초안: `nextflow/s3_features_nextflow_team4_README.txt` → `nextflow/README.md` 의 `aws s3 cp` 예시 참고.

## Pre-Project 목표 워크플로 (계획 확인용)

아래는 `pipeline_overview.html`에 **명시적으로 빠져 있거나 한 줄로만 언급된 부분**을 보강한 **의도한 단계 순서**입니다. 팀 검토·확인용이며, 구현 후 이 절을 실제 산출물 경로와 맞춰 갱신하면 됩니다.

### 1) HTML 대비 보강: 빠진 단계

| 단계 | 역할 | 비고 |
|------|------|------|
| **METABRIC (외부 코호트 검증)** | TCGA 등으로 학습·튜닝한 모델·랭킹이 **독립 코호트에서도 일반화**되는지 검증 | 슬라이드 상 “TCGA (Train) + METABRIC (Validation)” 퍼널과 동일 목적. HTML §10에 외부 AUROC 등은 있으나, **워크플로 다이어그램 단계로는 분리 표기가 부족**함. |
| **ADMET (임상 진입 전 생물학적 필터)** | 후보 약물의 **용해도, CYP 대사, hERG, 간독성** 등 **4 Gate**로 안전·약동학 측면 선별 | HTML §6·후처리에 ADMET 요소는 있으나, **Week 4 Part A처럼 “임상 진입 필터” 단계로 명시**되진 않음. |
| **Bedrock (LLM) 추가 검증·XAI** | 원시/모델 점수에 **문헌·지식 그래프 RAG**를 붙여 근거 수집, 기전 설명, 리스크, **모순 탐지** → 임상 문서형 산출 | ADMET·METABRIC 결과를 **설명·교차검증**하는 층으로 두는 것이 목표. |

**권장 순서(요약):**  
데이터(S3 raw + `results/`) → **Nextflow 피처 엔지니어링** → (파일럿) ML / DL / Graph 학습·비교 → 확장·**모델 랭킹**·최적 모델 선정 → **METABRIC 일반화 검증** → **ADMET Gate** → **Bedrock 기반 검증·설명 보고서**.

### 2) 데이터 소스 (S3)와 Nextflow

- 버킷에는 **raw**와 전처리 완료 **`results/`** 가 함께 있음(상표 참고).
- **Nextflow**로 `results/`(및 필요 시 raw)를 입력으로 **피처 엔지니어링**을 하고, 산출은 **`results/features_nextflow_team4/`** 에 쓰는 구성(팀4 합의).

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

이 저장소에는 아직 Nextflow·Batch 템플릿이 없음. 확정되면 `nextflow.config`(awsbatch 프로필)와 **IAM·큐 이름**만 README에 링크 형태로 추가하는 것을 권장.

## 모델 학습·튜닝 (SageMaker)

계획상 **학습·하이퍼파라미터 튜닝**은 주로 **Amazon SageMaker**(노트북/Studio, Training Job, 내장 알고리즘 또는 커스텀 스크립트·컨테이너, GPU 예: `ml.g4dn.*`)에서 진행하는 흐름을 전제로 합니다. PPTX에서도 LightGBM/XGB 등 **SageMaker built-in** 언급이 이에 해당합니다.

**AWS Batch**는 Nextflow **피처 엔지니어링** 병렬화에 두는 그림이 자연스럽고, 학습 잡을 Batch GPU 큐에 올리는 것은 **팀 선택**(동일 컨테이너를 ECR에 두고 Batch에서 실행)으로 가능합니다.

## `credentials` 작성 시 주의

- 파일 상단에 **`#` 주석을 넣지 않는 것**을 권장합니다. (일부 환경에서 AWS CLI 파싱 오류가 날 수 있음)
- 형식은 다음 두 줄과 `[4team-project]` 섹션만 유지하면 됩니다.

  - `aws_access_key_id`
  - `aws_secret_access_key`

## 보안

- `.aws/credentials`는 **절대 저장소에 커밋하지 마세요.** (`.gitignore`에 포함됨)
- 키가 유출되었을 가능성이 있으면 IAM에서 해당 액세스 키를 비활성화하고 새로 발급하세요.
