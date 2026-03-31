"""
Submit four SageMaker Training Jobs in parallel (B dataset): lightgbm, xgboost, rf, elasticnet.

Requires: pip install sagemaker boto3
Run from repo root:
  python3 ml/pilot_sagemaker/submit_b_parallel.py

Default role is the latest service-linked execution role seen in the team account; override with --role if needed.
"""
from __future__ import annotations

import argparse
import time

import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

DEFAULT_ROLE = "arn:aws:iam::666803869796:role/service-role/AmazonSageMaker-ExecutionRole-20260310T094243"
BUCKET = "drug-discovery-joe-raw-data-team4"
FEATURES_S3 = (
    f"s3://{BUCKET}/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/B/features_b.parquet"
)
LABELS_S3 = f"s3://{BUCKET}/results/features_nextflow_team4/abc_inputs/20260330_abc_v1/B/labels.parquet"
OUTPUT_PREFIX = f"s3://{BUCKET}/results/features_nextflow_team4/sagemaker/b_pilot_4models"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default="ap-northeast-2")
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument("--output-prefix", default=OUTPUT_PREFIX)
    p.add_argument("--features-s3", default=FEATURES_S3)
    p.add_argument("--labels-s3", default=LABELS_S3)
    p.add_argument(
        "--framework-version",
        default="2.1.0",
        help="PyTorch version for the managed training image (Python runtime only).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    boto_sess = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_sess, default_bucket=BUCKET)

    from pathlib import Path

    source_dir = str(Path(__file__).resolve().parent)
    ts = int(time.time())
    models = ["lightgbm", "xgboost", "rf", "elasticnet"]
    jobs: list[tuple[str, str]] = []

    for model in models:
        job_name = f"team4-b-pilot-{model}-{ts}".replace("_", "-")[:63]
        est = PyTorch(
            entry_point="train_tabular.py",
            source_dir=source_dir,
            role=args.role,
            framework_version=args.framework_version,
            py_version="py310",
            instance_type=args.instance_type,
            instance_count=1,
            sagemaker_session=sess,
            output_path=args.output_prefix,
            hyperparameters={
                "model": model,
                "features_s3": args.features_s3,
                "labels_s3": args.labels_s3,
                "seed": "42",
                "test_size": "0.2",
            },
        )
        est.fit(wait=False, job_name=job_name)
        name = est.latest_training_job.name
        jobs.append((model, name))
        print(f"Submitted {model}: {name}")

    print("All four jobs submitted (parallel on service quota).")
    for model, name in jobs:
        print(f"  - {model}: {name}")


if __name__ == "__main__":
    main()
