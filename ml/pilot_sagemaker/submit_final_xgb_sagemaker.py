from __future__ import annotations

import argparse
import subprocess
import sys
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
    p = argparse.ArgumentParser(description="Submit final full-train XGBoost job to SageMaker.")
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default="ml.m5.2xlarge")
    p.add_argument("--framework-version", default="2.1.0")
    p.add_argument("--sagemaker-account-id", default=DEFAULT_SAGEMAKER_ACCOUNT)
    p.add_argument("--code-bucket", default=None)
    p.add_argument(
        "--features-uri",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet",
        help="S3 or local path; default matches ML/DL local CV (pathway_addon, LINCS excluded in train).",
    )
    p.add_argument(
        "--labels-uri",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet",
        help="Aligned with features (inner join n=14497).",
    )
    p.add_argument(
        "--output-prefix",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/sagemaker/final_xgb_20260331",
    )
    p.add_argument("--wait", action="store_true", help="Block until the training job completes.")
    p.add_argument(
        "--sync-to-final-three",
        action="store_true",
        help="After a successful job (requires --wait), download model.tar.gz and sync into sagemaker_final_three/artifacts/xgb/.",
    )
    return p.parse_args()


def _upload_source_tarball(
    pilot_dir: Path,
    code_bucket: str,
    region: str,
    staging_root: str,
    ts: int,
) -> str:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in ("train_tabular.py", "requirements.txt"):
                p = pilot_dir / name
                if not p.is_file():
                    raise FileNotFoundError(p)
                tar.add(p, arcname=name)
        key = f"{staging_root}/source/final-xgb-{ts}.tar.gz"
        s3_uri = f"s3://{code_bucket}/{key}"
        subprocess.run(["aws", "s3", "cp", tar_path, s3_uri, "--region", region, "--sse", "AES256"], check=True)
        return s3_uri
    finally:
        Path(tar_path).unlink(missing_ok=True)


def _stage_data(
    code_bucket: str,
    region: str,
    staging_root: str,
    ts: int,
    features_src: str,
    labels_src: str,
) -> tuple[str, str]:
    base = f"s3://{code_bucket}/{staging_root}/data/final_xgb/{ts}/"
    feat_dst = f"{base}features.parquet"
    lab_dst = f"{base}labels.parquet"
    subprocess.run(["aws", "s3", "cp", features_src, feat_dst, "--region", region, "--sse", "AES256"], check=True)
    subprocess.run(["aws", "s3", "cp", labels_src, lab_dst, "--region", region, "--sse", "AES256"], check=True)
    return feat_dst, lab_dst


def main() -> None:
    args = parse_args()
    if args.sync_to_final_three and not args.wait:
        raise SystemExit("--sync-to-final-three requires --wait")
    ts = int(time.time())
    staging_root = "team4-final-xgb-train"
    code_bucket = args.code_bucket or f"sagemaker-{args.region}-{args.sagemaker_account_id}"
    pilot_dir = Path(__file__).resolve().parent

    source_s3 = _upload_source_tarball(pilot_dir, code_bucket, args.region, staging_root, ts)
    features_s3, labels_s3 = _stage_data(
        code_bucket=code_bucket,
        region=args.region,
        staging_root=staging_root,
        ts=ts,
        features_src=args.features_uri,
        labels_src=args.labels_uri,
    )

    boto_sess = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_sess, default_bucket=code_bucket)

    job_name = f"team4-final-xgb-20260331-{ts}"[:63]
    est = PyTorch(
        entry_point="train_tabular.py",
        source_dir=source_s3,
        role=args.role,
        framework_version=args.framework_version,
        py_version="py310",
        instance_type=args.instance_type,
        instance_count=1,
        sagemaker_session=sess,
        output_path=args.output_prefix,
        hyperparameters={
            "model": "xgboost",
            "features_s3": features_s3,
            "labels_s3": labels_s3,
            "seed": "42",
            "full_train": "off",
            "test_size": "0.1",
            "exclude_lincs": "on",
            "use_smiles": "off",
            "xgb_max_depth": "4",
            "xgb_learning_rate": "0.05",
            "xgb_n_estimators": "400",
            "xgb_subsample": "0.8",
            "xgb_colsample_bytree": "0.8",
        },
    )
    est.fit(wait=args.wait, job_name=job_name)
    print("submitted_job_name:", est.latest_training_job.name)
    print("source_s3:", source_s3)
    print("staged_features_s3:", features_s3)
    print("staged_labels_s3:", labels_s3)
    print("output_prefix:", args.output_prefix)
    if args.wait and args.sync_to_final_three:
        md = est.model_data
        if not md or not str(md).startswith("s3://"):
            raise RuntimeError(f"Unexpected model_data after training: {md!r}")
        subprocess.run(
            [
                sys.executable,
                str(pilot_dir / "sagemaker_final_sync.py"),
                "--family",
                "xgb",
                "--model-tar-s3",
                str(md),
                "--region",
                args.region,
            ],
            check=True,
        )
    elif not args.wait:
        print(
            "After the job completes: download model.tar.gz from the job output, then run\n"
            "  python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py "
            "--family xgb --model-tar /path/to/model.tar.gz"
        )


if __name__ == "__main__":
    main()
