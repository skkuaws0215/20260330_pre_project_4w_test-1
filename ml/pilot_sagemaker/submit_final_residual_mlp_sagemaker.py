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
FE_REL = "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
LAB_REL = "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit final ResidualMLP Training Job (same data prep as local CV script).")
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument("--framework-version", default="2.1.0")
    p.add_argument("--sagemaker-account-id", default=DEFAULT_SAGEMAKER_ACCOUNT)
    p.add_argument("--code-bucket", default=None)
    p.add_argument(
        "--features-uri",
        default=f"s3://{TEAM_BUCKET}/{FE_REL}",
        help="S3 or local path to pair_features (pathway_addon LINCS excluded).",
    )
    p.add_argument(
        "--labels-uri",
        default=f"s3://{TEAM_BUCKET}/{LAB_REL}",
        help="Labels parquet aligned with features (n=14497 when merged).",
    )
    p.add_argument(
        "--output-prefix",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/sagemaker/final_residualmlp_20260331",
    )
    return p.parse_args()


def _upload_source_tarball(pilot_dir: Path, code_bucket: str, region: str, staging_root: str, ts: int) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in ("train_residual_mlp_final.py", "requirements.txt"):
                p = pilot_dir / name
                if not p.is_file():
                    raise FileNotFoundError(p)
                tar.add(p, arcname=name)
        key = f"{staging_root}/source/final-residualmlp-{ts}.tar.gz"
        s3_uri = f"s3://{code_bucket}/{key}"
        subprocess.run(["aws", "s3", "cp", tar_path, s3_uri, "--region", region, "--sse", "AES256"], check=True)
        return s3_uri
    finally:
        Path(tar_path).unlink(missing_ok=True)


def _stage_pair(code_bucket: str, region: str, staging_root: str, ts: int, features_src: str, labels_src: str) -> tuple[str, str]:
    base = f"s3://{code_bucket}/{staging_root}/data/final_residualmlp/{ts}/"
    feat_dst = f"{base}features.parquet"
    lab_dst = f"{base}labels.parquet"
    subprocess.run(["aws", "s3", "cp", features_src, feat_dst, "--region", region, "--sse", "AES256"], check=True)
    subprocess.run(["aws", "s3", "cp", labels_src, lab_dst, "--region", region, "--sse", "AES256"], check=True)
    return feat_dst, lab_dst


def main() -> None:
    args = parse_args()
    ts = int(time.time())
    staging_root = "team4-final-residualmlp-train"
    code_bucket = args.code_bucket or f"sagemaker-{args.region}-{args.sagemaker_account_id}"
    pilot_dir = Path(__file__).resolve().parent

    source_s3 = _upload_source_tarball(pilot_dir, code_bucket, args.region, staging_root, ts)
    features_s3, labels_s3 = _stage_pair(
        code_bucket, args.region, staging_root, ts, args.features_uri, args.labels_uri
    )

    boto_sess = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_sess, default_bucket=code_bucket)
    job_name = f"team4-final-resmlp-{ts}"[:63]
    est = PyTorch(
        entry_point="train_residual_mlp_final.py",
        source_dir=source_s3,
        role=args.role,
        framework_version=args.framework_version,
        py_version="py310",
        instance_type=args.instance_type,
        instance_count=1,
        sagemaker_session=sess,
        output_path=args.output_prefix,
        hyperparameters={
            "features_s3": features_s3,
            "labels_s3": labels_s3,
            "seed": "42",
            "test_size": "0.1",
            "epochs": "45",
            "batch_size": "256",
            "lr": "0.001",
            "patience": "8",
        },
    )
    est.fit(wait=False, job_name=job_name)
    print("submitted_job_name:", est.latest_training_job.name)
    print("source_s3:", source_s3)
    print("staged_features_s3:", features_s3)
    print("staged_labels_s3:", labels_s3)
    print("output_prefix:", args.output_prefix)
    print(
        "After job: aws s3 cp <output>/model.tar.gz /tmp/m.tar.gz && "
        "python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py "
        "--family residualmlp --model-tar /tmp/m.tar.gz"
    )


if __name__ == "__main__":
    main()
