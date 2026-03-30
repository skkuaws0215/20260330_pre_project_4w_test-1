# Nextflow (팀4)

## S3 폴더 역할 (헷갈리지 않기)

| 경로 | 역할 |
|------|------|
| `results/<소스>/` | 팀 **공유** 전처리(raw→parquet 등). **읽기 입력**으로만 사용. 본인 산출 업로드 금지. |
| **`results/features_nextflow_team4/`** | **본인 전용**. 생성하는 FE 결과·실험 산출·ML용 데이터셋은 **전부 여기**에 업로드. 모델 학습·테스트도 이 폴더 산출을 입력으로 사용. |

## 입력 정책 (팀4 합의)

- **사용(입력):** **raw에서 만든** `s3://…/results/<소스>/…` 전처리 parquet·산출물. (저장소 `README.md` Parquet 표 참고.)
- **사용하지 않음(입력):** **`ml_ready/`** — 통합 테이블은 Nextflow가 `results/` 소스들을 조인·FE 하여 새로 만든다.
- **산출:** `results/features_nextflow_team4/` (권장: `run_id` 하위폴더).

## S3 산출 위치

- **Prefix:** `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/`
- **팀 안내용 마커:** 같은 경로의 **`README.txt`** 는 저장소의 **`results/features_nextflow_team4/README.txt`** 와 동일 내용을 두는 것을 권장 (로컬 = S3 키 구조와 맞춤).

### S3에 올리기

```powershell
.\use-team-aws.ps1
aws s3 cp .\results\features_nextflow_team4\README.txt "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/README.txt"
```

## 파이프라인 코드

`main.nf`, `nextflow.config`, `scripts/build_features.py`를 **`nextflow/`** 에 두고, FE 산출만 `results/features_nextflow_team4/` 로 업로드한다.

## FE Contract (확정)

- **Target unit:** `sample-drug pair`
- **Main task:** regression (`IC50/AUC/response score`)
- **Aux task:** binary (`sensitive/resistant`)
- **Input source:** `results/<소스>/...` (공유 전처리 결과, 읽기 전용)
- **Do not use:** `ml_ready/` as FE input
- **Outputs (run_id 하위):**
  - `features.parquet`
  - `labels.parquet`
  - `feature_manifest.json`
  - `features_dl.parquet` (정규화 브랜치 사용 시)

## 실행 파일

| 파일 | 역할 |
|------|------|
| `nextflow/main.nf` | FE workflow (회귀 메인 + 이진 보조 라벨 생성) |
| `nextflow/nextflow.config` | `local` / `awsbatch` 프로필 |
| `nextflow/scripts/build_features.py` | join, imputation, variance filtering, leakage 제거 |
| `nextflow/Dockerfile` | Batch 실행용 컨테이너 이미지 빌드 |

## 실행 예시

### 1) 로컬 검증

```powershell
nextflow run nextflow/main.nf -profile local `
  --sample_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/cross_platform/11_intersection.parquet" `
  --drug_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/chembl/15_activities.parquet" `
  --label_uri "s3://drug-discovery-joe-raw-data-team4/results/gdsc/21_ic50.parquet" `
  --run_id "20260330_local_reg_main"
```

### 2) AWS Batch 실행

1. `nextflow/nextflow.config` 의 `awsbatch` 프로필에서 `process.queue`, `process.container` 값을 실제 값으로 바꾼다.
2. ECR 이미지 빌드/푸시 후 실행:

```powershell
nextflow run nextflow/main.nf -profile awsbatch `
  --sample_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/cross_platform/11_intersection.parquet" `
  --drug_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/chembl/15_activities.parquet" `
  --label_uri "s3://drug-discovery-joe-raw-data-team4/results/gdsc/21_ic50.parquet" `
  --run_id "20260330_batch_reg_main"
```

## 지금 바로 할 일 (Batch 본 실행)

현재 구성 기준(예시):

- Compute environment: `team4-fe-ce-cpu`
- Job queue: `team4-fe-queue-cpu`
- Job definition: `team4-fe-jobdef:1`
- ECR image: `666803869796.dkr.ecr.ap-northeast-2.amazonaws.com/skku-project/pre-4team:fe-latest`

아래 순서로 실행:

1. **Batch 리소스 생성**
   - Compute environment
   - Job queue
2. **ECR 이미지 빌드/푸시**
   - `./nextflow/build_and_push_ecr.ps1 -ImageTag fe-latest`
3. **Nextflow config 반영**
   - `nextflow/nextflow.config`의 `awsbatch` 프로필에서 `process.queue`, `process.container` 실제 값으로 교체
4. **파라미터 파일 실행**
   - `nextflow/params/team4.awsbatch.example.json` 값을 실제 입력 경로/컬럼으로 수정
   - `./nextflow/run_fe_batch.ps1 -ParamsFile nextflow/params/team4.awsbatch.example.json`

## 아키텍처 (합의)

| 구간 | 권장 |
|------|------|
| **피처 엔지니어링** | **Nextflow + AWS Batch** — 병렬·대량 I/O |
| **학습·튜닝** | 주로 **Amazon SageMaker** |
| **선택** | 동일 Docker를 **Batch GPU**에 올려 학습만 실행 |
