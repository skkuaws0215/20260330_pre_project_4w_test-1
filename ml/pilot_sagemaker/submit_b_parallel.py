"""
Submit four SageMaker Training Jobs in parallel (A/B/C ABC datasets): lightgbm, xgboost, rf, elasticnet.

Defaults target the **team-shared** AWS layout (team4 data bucket, shared execution role, account-scoped
SageMaker default bucket) — not a personal AWS account. Override paths/role if your cohort differs.

Requires: pip install sagemaker boto3
AWS CLI configured (for `aws s3 cp --sse AES256`).

Run from repo root:
  python3 ml/pilot_sagemaker/submit_b_parallel.py              # B (default)
  python3 ml/pilot_sagemaker/submit_b_parallel.py --dataset a
  python3 ml/pilot_sagemaker/submit_b_parallel.py --dataset c

Guards against past failures:
- Code tarball uploaded with **SSE-S3 (AES256)** so the execution role does not need kms:Decrypt on team KMS objects.
- Training data is **mirrored** from the team bucket into the SageMaker default bucket (AES256) so the role does not need GetObject on the team bucket during training.
- Default **output_path** is under the SageMaker bucket unless --use-team-output-prefix (role needs team PutObject).

Why the filename `submit_b_parallel.py`: historical; use --dataset for A/C.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import tarfile
import tempfile
import time
from pathlib import Path

import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.session import Session

# Team project defaults (팀4 공용 버킷·실행 역할·계정 — 개인 계정 아님).
DEFAULT_ROLE = "arn:aws:iam::666803869796:role/service-role/AmazonSageMaker-ExecutionRole-20260310T094243"
DEFAULT_SAGEMAKER_ACCOUNT = "666803869796"
BUCKET = "drug-discovery-joe-raw-data-team4"
ABC_BASE = f"s3://{BUCKET}/results/features_nextflow_team4/abc_inputs/20260330_abc_v1"

# Preset URIs (team bucket; mirrored before training — do not rely on role reading these at runtime).
DATASET_PRESETS: dict[str, tuple[str, str]] = {
    "a": (f"{ABC_BASE}/A/features.parquet", f"{ABC_BASE}/A/labels.parquet"),
    "b": (f"{ABC_BASE}/B/features_b.parquet", f"{ABC_BASE}/B/labels.parquet"),
    "c": (f"{ABC_BASE}/C/features.parquet", f"{ABC_BASE}/C/labels.parquet"),
}

OUTPUT_PREFIX_TEAM_BY_DATASET: dict[str, str] = {
    "a": f"s3://{BUCKET}/results/features_nextflow_team4/sagemaker/a_pilot_4models",
    # B-only artifacts are grouped under one folder to keep sagemaker/ tidy.
    "b": f"s3://{BUCKET}/results/features_nextflow_team4/sagemaker/team4_b_pilot_runs",
    "c": f"s3://{BUCKET}/results/features_nextflow_team4/sagemaker/c_pilot_4models",
}

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SageMaker 4-model pilot for ABC tabular regression.")
    p.add_argument(
        "--dataset",
        choices=["a", "b", "c"],
        default="b",
        help="Which ABC preset to use for features/labels (unless --features-s3/--labels-s3 set).",
    )
    p.add_argument(
        "--team-tag",
        default=os.environ.get("SAGEMAKER_PILOT_TEAM_TAG", "team4"),
        metavar="TAG",
        help="Team id for SageMaker staging keys and job names (default: team4). Env: SAGEMAKER_PILOT_TEAM_TAG.",
    )
    p.add_argument(
        "--sagemaker-account-id",
        default=os.environ.get("SAGEMAKER_PILOT_ACCOUNT_ID", DEFAULT_SAGEMAKER_ACCOUNT),
        metavar="ID",
        help="Account id for default sagemaker-{region}-{account} bucket (team project default).",
    )
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default="ap-northeast-2")
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument(
        "--output-prefix",
        default=None,
        help="S3 prefix for Training Job model artifacts. Default: sagemaker bucket under pilot output/{dataset}/.",
    )
    p.add_argument(
        "--use-team-output-prefix",
        action="store_true",
        help="Write artifacts under the team bucket (needs execution role s3:PutObject + any KMS).",
    )
    p.add_argument(
        "--features-s3",
        default=None,
        help="Override features parquet S3 URI (still mirrored to the SageMaker bucket before training).",
    )
    p.add_argument(
        "--labels-s3",
        default=None,
        help="Override labels parquet S3 URI (still mirrored to the SageMaker bucket before training).",
    )
    p.add_argument(
        "--framework-version",
        default="2.1.0",
        help="PyTorch version for the managed training image (Python runtime only).",
    )
    p.add_argument(
        "--code-bucket",
        default=None,
        help="Bucket for code tarball and staged data (default: sagemaker-{region}-{account}).",
    )
    args = p.parse_args()
    feat_d, lab_d = DATASET_PRESETS[args.dataset]
    args.features_s3 = args.features_s3 or feat_d
    args.labels_s3 = args.labels_s3 or lab_d
    args.staging_root = f"{args.team_tag}-pilot-train"
    return args


def _upload_source_tarball(
    code_bucket: str,
    region: str,
    pilot_dir: Path,
    staging_root: str,
    dataset: str,
    ts: int,
) -> str:
    """Build a small tar.gz (train script + requirements) and upload with SSE-S3 (AES256)."""
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in ("train_tabular.py", "requirements.txt"):
                p = pilot_dir / name
                if not p.is_file():
                    raise FileNotFoundError(p)
                tar.add(p, arcname=name)
        key = f"{staging_root}/source/pilot-{dataset}-{ts}.tar.gz"
        s3_uri = f"s3://{code_bucket}/{key}"
        subprocess.run(
            ["aws", "s3", "cp", tar_path, s3_uri, "--region", region, "--sse", "AES256"],
            check=True,
        )
        return s3_uri
    finally:
        Path(tar_path).unlink(missing_ok=True)


def _mirror_data_to_sagemaker_bucket(
    code_bucket: str,
    region: str,
    staging_root: str,
    dataset: str,
    ts: int,
    features_src: str,
    labels_src: str,
) -> tuple[str, str]:
    """Copy parquets into the SageMaker bucket (SSE-S3) so the training job role can read them."""
    base = f"s3://{code_bucket}/{staging_root}/data/{dataset}/{ts}/"
    feat_dst = f"{base}features.parquet"
    lab_dst = f"{base}labels.parquet"
    for src, dst in ((features_src, feat_dst), (labels_src, lab_dst)):
        subprocess.run(
            ["aws", "s3", "cp", src, dst, "--region", region, "--sse", "AES256"],
            check=True,
        )
    return feat_dst, lab_dst


def main() -> None:
    args = parse_args()
    boto_sess = boto3.Session(region_name=args.region)
    code_bucket = args.code_bucket or f"sagemaker-{args.region}-{args.sagemaker_account_id}"
    sess = Session(boto_session=boto_sess, default_bucket=code_bucket)

    pilot_dir = Path(__file__).resolve().parent
    ds = args.dataset
    ts = int(time.time())
    sr = args.staging_root
    source_s3 = _upload_source_tarball(code_bucket, args.region, pilot_dir, sr, ds, ts)
    print(f"Uploaded training code (SSE-S3): {source_s3}")

    features_s3, labels_s3 = _mirror_data_to_sagemaker_bucket(
        code_bucket, args.region, sr, ds, ts, args.features_s3, args.labels_s3
    )
    print(
        "Staged training data (execution-role readable):\n"
        f"  {features_s3}\n"
        f"  {labels_s3}"
    )

    if args.use_team_output_prefix:
        output_prefix = OUTPUT_PREFIX_TEAM_BY_DATASET[ds]
    elif args.output_prefix:
        output_prefix = args.output_prefix
    else:
        output_prefix = f"s3://{code_bucket}/{sr}/output/{ds}"

    models = ["lightgbm", "xgboost", "rf", "elasticnet"]
    jobs: list[tuple[str, str]] = []
    team_job = args.team_tag.replace("_", "-")

    for model in models:
        job_name = f"{team_job}-{ds}-pilot-{model}-{ts}".replace("_", "-")[:63]
        est = PyTorch(
            entry_point="train_tabular.py",
            source_dir=source_s3,
            role=args.role,
            framework_version=args.framework_version,
            py_version="py310",
            instance_type=args.instance_type,
            instance_count=1,
            sagemaker_session=sess,
            output_path=output_prefix,
            hyperparameters={
                "model": model,
                "features_s3": features_s3,
                "labels_s3": labels_s3,
                "seed": "42",
                "test_size": "0.2",
                "use_smiles": "on" if ds == "c" else "off",
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
