# Nextflow (팀4)

- **S3 산출 prefix:** `results/features_nextflow_team4/` (버킷 `drug-discovery-joe-raw-data-team4`)
- 팀에 prefix를 알리려면 `s3_features_nextflow_team4_README.txt` 를 S3에 `README.txt` 로 업로드:

```powershell
.\use-team-aws.ps1
aws s3 cp .\nextflow\s3_features_nextflow_team4_README.txt "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/README.txt"
```

이후 `main.nf`, `nextflow.config` 등은 여기에 추가 예정.
