# Nextflow (팀4)

## S3 산출 위치

- **Prefix:** `s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/`
- **팀 안내용 마커:** 같은 경로의 **`README.txt`** 는 저장소의 **`results/features_nextflow_team4/README.txt`** 와 동일 내용을 두는 것을 권장 (로컬 = S3 키 구조와 맞춤).

### S3에 올리기

```powershell
.\use-team-aws.ps1
aws s3 cp .\results\features_nextflow_team4\README.txt "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4/README.txt"
```

## 파이프라인 코드

이후 `main.nf`, `nextflow.config` 등은 **`nextflow/`** 에 추가 (S3 prefix 설명 파일은 **`results/features_nextflow_team4/`** 에만 둠).

## 아키텍처 (합의)

| 구간 | 권장 |
|------|------|
| **피처 엔지니어링** | **Nextflow + AWS Batch** — 병렬·대량 I/O |
| **학습·튜닝** | 주로 **Amazon SageMaker** |
| **선택** | 동일 Docker를 **Batch GPU**에 올려 학습만 실행 |
