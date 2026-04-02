from __future__ import annotations

import argparse
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

DEFAULT_ROLE = "arn:aws:iam::666803869796:role/service-role/AmazonSageMaker-ExecutionRole-20260310T094243"
DEFAULT_REGION = "ap-northeast-2"
DEFAULT_SAGEMAKER_ACCOUNT = "666803869796"
TEAM_BUCKET = "drug-discovery-joe-raw-data-team4"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit SageMaker CV job for a single model.")
    p.add_argument("--model", choices=["xgb", "residualmlp"], required=True)
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument("--framework-version", default="2.1.0")
    p.add_argument("--sagemaker-account-id", default=DEFAULT_SAGEMAKER_ACCOUNT)
    p.add_argument("--code-bucket", default=None)
    p.add_argument(
        "--features-uri",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
    )
    p.add_argument(
        "--labels-uri",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet",
    )
    p.add_argument("--output-prefix", default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/sagemaker/20260402/cv_single")
    p.add_argument("--wait", action="store_true")
    return p.parse_args()


def _upload_source(pilot_dir: Path, code_bucket: str, region: str, ts: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in ("train_single_model_cv_sagemaker.py", "requirements.txt"):
                tar.add(pilot_dir / name, arcname=name)
        s3_uri = f"s3://{code_bucket}/team4-final-cv/source/cv-single-{ts}.tar.gz"
        subprocess.run(["aws", "s3", "cp", tar_path, s3_uri, "--region", region, "--sse", "AES256"], check=True)
        return s3_uri
    finally:
        Path(tar_path).unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    ts = int(time.time())
    code_bucket = args.code_bucket or f"sagemaker-{args.region}-{args.sagemaker_account_id}"
    pilot_dir = Path(__file__).resolve().parent
    source_s3 = _upload_source(pilot_dir, code_bucket, args.region, ts)
    feat_dst = f"s3://{code_bucket}/team4-final-cv/data/{args.model}/{ts}/features.parquet"
    lab_dst = f"s3://{code_bucket}/team4-final-cv/data/{args.model}/{ts}/labels.parquet"
    subprocess.run(["aws", "s3", "cp", args.features_uri, feat_dst, "--region", args.region, "--sse", "AES256"], check=True)
    subprocess.run(["aws", "s3", "cp", args.labels_uri, lab_dst, "--region", args.region, "--sse", "AES256"], check=True)
    boto_sess = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_sess, default_bucket=code_bucket)
    est = PyTorch(
        entry_point="train_single_model_cv_sagemaker.py",
        source_dir=source_s3,
        role=args.role,
        framework_version=args.framework_version,
        py_version="py310",
        instance_type=args.instance_type,
        instance_count=1,
        sagemaker_session=sess,
        output_path=args.output_prefix,
        hyperparameters={"model": args.model, "features_s3": feat_dst, "labels_s3": lab_dst, "seed": "42", "n_splits": "5"},
    )
    est.fit(wait=args.wait, job_name=f"team4-cv-{args.model}-20260402-{ts}"[:63])
    print("submitted_job_name:", est.latest_training_job.name)
    print("model_data:", est.model_data)


if __name__ == "__main__":
    main()
