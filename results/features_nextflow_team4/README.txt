Nextflow feature engineering — dedicated prefix (this member only)
Bucket: drug-discovery-joe-raw-data-team4
Prefix: results/features_nextflow_team4/

KO: results/ 아래 소스 폴더(tcga/, gdsc/, …)는 팀원 4인 공유 전처리 구역이다.
앞으로 본인이 생성하는 FE 결과·실험 산출·ML용 데이터셋은 전부 이 prefix 아래에만 올릴 것.
ML 모델 테스트·학습 입력도 이 폴더에서 나온 데이터셋을 사용한다. 공유 results/ 와 헷갈리지 말 것.

EN: results/<source>/ is shared team preprocessing (4 members). Do not upload
your FE outputs there. All your generated FE artifacts and ML-ready exports
go under this prefix only. Train/test models from datasets produced here.

Input policy: Read team preprocessed data from s3://.../results/<source>/...
(parquet from raw). Do NOT use ml_ready/ as pipeline input;
integration + FE happen inside Nextflow.

- Keep outputs only under this prefix (never mix into results/<source>/).
- FE batch outputs (miss* runs from main.nf): results/features_nextflow_team4/fe_batch_runs/<run_id>/
- Other folders (input/, work/, abc_inputs/, sagemaker/, …) stay as needed.

Agreed: 2026-03-29 (update if changed)

Git path = S3 key: results/features_nextflow_team4/README.txt
