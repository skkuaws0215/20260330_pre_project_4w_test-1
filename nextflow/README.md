# Nextflow (팀4)

## S3 폴더 역할 (헷갈리지 않기)

| 경로 | 역할 |
|------|------|
| `results/<소스>/` | 팀 **공유** 전처리(raw→parquet 등). **읽기 입력**으로만 사용. 본인 산출 업로드 금지. |
| **`results/features_nextflow_team4/`** | **본인 전용**. 생성하는 FE 결과·실험 산출·ML용 데이터셋은 **전부 여기**에 업로드. 모델 학습·테스트도 이 폴더 산출을 입력으로 사용. |

## 입력 정책 (팀4 합의)

- **사용(입력):** **raw에서 만든** `s3://…/results/<소스>/…` 전처리 parquet·산출물. (저장소 `README.md` Parquet 표 참고.)
- **사용하지 않음(입력):** **`ml_ready/`** — 통합 테이블은 Nextflow가 `results/` 소스들을 조인·FE 하여 새로 만든다.
- **산출:** `results/features_nextflow_team4/` — `main.nf`는 FE parquet를 **`fe_re_batch_runs/<run_id>/`** 에 두도록 기본 설정(`publishDir`). 그 외 `input/`, `work/` 등은 기존처럼 prefix 루트에 둔다.

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
| `nextflow/scripts/prepare_fe_inputs.py` | 브리지 전처리: `sample_features` / `drug_features` / `labels` + mapping/QC/manifest 생성 |
| `nextflow/Dockerfile` | Batch 실행용 컨테이너 이미지 빌드 |

## 실행 예시

### 0) 브리지 전처리 (권장)

`results/<소스>/` 전처리 산출을 Nextflow FE 입력 계약 스키마로 맞춘다.

```powershell
python nextflow/scripts/prepare_fe_inputs.py `
  --run-id "20260330_bridge_v1" `
  --output-prefix "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1" `
  --label-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/source_snapshot/gdsc/21_ic50.parquet" `
  --sample-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/source_snapshot/depmap/57_crispr.parquet" `
  --drug-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/source_snapshot/chembl/15_activities.parquet"
```

산출:

- `sample_features.parquet` (`sample_id` key)
- `drug_features.parquet` (`canonical_drug_id` key, `smiles`/`has_smiles` 포함)
- `labels.parquet` (`sample_id`, `canonical_drug_id`, `ic50`, `binary_label`)
- `mapping_table.parquet`
- `join_qc_report.json`
- `bridge_manifest.json`

### 1) 로컬 검증

```powershell
nextflow run nextflow/main.nf -profile local `
  --sample_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/sample_features.parquet" `
  --drug_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/drug_features.parquet" `
  --label_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/labels.parquet" `
  --run_id "20260330_local_reg_main"
```

### 2) AWS Batch 실행

1. `nextflow/nextflow.config` 의 `awsbatch` 프로필에서 `process.queue`, `process.container` 값을 실제 값으로 바꾼다.
2. ECR 이미지 빌드/푸시 후 실행:

```powershell
nextflow run nextflow/main.nf -profile awsbatch `
  --sample_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/sample_features.parquet" `
  --drug_feature_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/drug_features.parquet" `
  --label_uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/labels.parquet" `
  --run_id "20260330_batch_reg_main"
```

## 신규 FE v2 (20260331) - 3개 + target

스크립트: `nextflow/scripts/build_pair_features_newfe_v2.py`

생성 FE:

1. sample-level pathway feature
2. drug-level chemistry feature (Morgan + RDKit descriptor)
3. pair-level LINCS interaction feature
4. pair-level target interaction feature (v2 추가)

실행 예시:

```powershell
python nextflow/scripts/build_pair_features_newfe_v2.py `
  --run-id "20260331" `
  --pairs-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/labels.parquet" `
  --sample-expression-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/sample_features.parquet" `
  --drug-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/drug_features.parquet" `
  --lincs-drug-signature-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260330_bridge_v1/drug_features.parquet" `
  --drug-target-uri "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/input/20260331/drug_target_map.parquet" `
  --pathway-gmt "nextflow/refs/h.all.v7.5.symbols.gmt" `
  --high-z-threshold 1.0 `
  --low-z-threshold -1.0 `
  --out-dir "results/features_nextflow_team4/fe_re_batch_runs/20260331"
```

출력 파일:

- `sample_pathway_features.parquet`
- `drug_chem_features.parquet`
- `pair_lincs_features.parquet`
- `pair_target_features.parquet`
- `pair_features_newfe.parquet` (신규 3개)
- `pair_features_newfe_v2.parquet` (신규 3개 + target)
- `feature_manifest.json`

중요:

- `feature_manifest.json`에 target high/low 기준을 기록한다.
  - 예: `high_gene_rule = zscore >= 1.0`, `low_gene_rule = zscore <= -1.0`
- overlap은 count + ratio를 모두 생성한다.
- pathway relevance는 mean + hit_count를 모두 생성한다.

### Pathway GMT·SMILES·학습용 addon (정리)

- **`--pathway-gmt` 비어 있으면** `sample_pathway_features.parquet`는 `sample_id`만 있고 `pathway__*` 점수 열이 **0개**다. Hallmark 등 GMT는 저장소 예시로 `nextflow/refs/h.all.v7.5.symbols.gmt`를 둔다.
- **유전자 심볼 매칭:** 표현 행렬 열이 `crispr__TP53`처럼 접두어가 있으면, 빌드 스크립트는 **열 이름 끝 토큰(TP53)** 과 GMT 유전자를 맞춘다. GMT 유전자는 파싱 시 대문자로 통일한다.
- **`--drug-uri`는 반드시 SMILES 포함:** `canonical_drug_id` + `canonical_smiles`(또는 스크립트가 기대하는 smiles 컬럼)가 있어야 Morgan·RDKit 서술자가 유효하다. SMILES 없는 stub만 넣으면 화학 블록이 무의미해지므로 **정식 pair 산출에 쓰지 않는다.**
- **pathway만 보강할 때:** 이미 SMILES 기반으로 만든 `final/pair_features_newfe_v2.parquet`가 있으면, 새로 만든 `sample_pathway_features.parquet`의 `pathway__*`만 `sample_id`로 left merge한 **`final_pathway_addon/pair_features_newfe_v2.parquet`** 를 학습 입력으로 쓸 수 있다. 재현 스크립트: `ml/pilot_sagemaker/merge_pathway_into_pair_features.py`. 감사·경로 요약: `final_pathway_addon/pathway_merge_audit.json`.
- **5-fold CV (pathway 50개 포함):** `analysis_target_only/xgb_mlp3_cv_pathway_addon/` 및 `comparison_vs_baseline.md` 참고.

### LINCS BRD 매핑 입력 스키마 (고정)

`normalize_lincs_mapping.py` 입력 `brd_map`은 CSV 또는 Parquet를 받을 수 있고 스키마는 아래와 같다.

- **필수 컬럼**
  - `brd_id`
  - `canonical_drug_id`
- **선택 컬럼**
  - `sig_id`
  - `pert_iname`
  - `source`
  - `mapping_confidence`
  - `note`

샘플 템플릿(20260331):

- `results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/brd_map_20260331_template.csv`
- `results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/brd_map_20260331_template.parquet`

실행 예시:

```powershell
python nextflow/scripts/normalize_lincs_mapping.py `
  --lincs-uri "s3://drug-discovery-joe-raw-data-team4/results/lincs/16_mcf7_processed.parquet" `
  --brd-map-uri "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/brd_map_20260331_template.csv" `
  --out-parquet "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/lincs_drug_signature_mapped_20260331.parquet" `
  --out-report "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/lincs_mapping_report_20260331.json"
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
