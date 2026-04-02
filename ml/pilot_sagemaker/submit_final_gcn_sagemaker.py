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
FE_REL = "results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet"
LAB_REL = "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_derived/labels_B_graph.parquet"
DT_REL = "results/features_nextflow_team4/fe_re_batch_runs/20260331/input_refs/drug_target_map_20260331.parquet"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Submit final GCN Training Job (baseline A, same graph inputs as run_graph_gnn_cv).")
    p.add_argument("--role", default=DEFAULT_ROLE)
    p.add_argument("--region", default=DEFAULT_REGION)
    p.add_argument("--instance-type", default="ml.g4dn.xlarge")
    p.add_argument("--framework-version", default="2.1.0")
    p.add_argument("--sagemaker-account-id", default=DEFAULT_SAGEMAKER_ACCOUNT)
    p.add_argument("--code-bucket", default=None)
    p.add_argument("--features-uri", default=f"s3://{TEAM_BUCKET}/{FE_REL}")
    p.add_argument("--labels-uri", default=f"s3://{TEAM_BUCKET}/{LAB_REL}")
    p.add_argument("--drug-target-uri", default=f"s3://{TEAM_BUCKET}/{DT_REL}")
    p.add_argument(
        "--output-prefix",
        default=f"s3://{TEAM_BUCKET}/results/features_nextflow_team4/sagemaker/final_gcn_20260331",
    )
    p.add_argument("--wait", action="store_true", help="Block until the training job completes.")
    p.add_argument(
        "--sync-to-final-three",
        action="store_true",
        help="After a successful job (requires --wait), download model.tar.gz and sync into sagemaker_final_three/artifacts/gcn/.",
    )
    return p.parse_args()


def _upload_source_tarball(pilot_dir: Path, repo_root: Path, code_bucket: str, region: str, staging_root: str, ts: int) -> str:
    disease = repo_root / "data/graph_baseline/disease_genes_common_v1.txt"
    if not disease.is_file():
        raise FileNotFoundError(disease)
    bundle = [
        "train_gcn_final.py",
        "graph_baseline_data.py",
        "run_network_proximity_baseline.py",
        "run_graph_gnn_cv.py",
        "requirements.txt",
    ]
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        tar_path = tf.name
    try:
        with tarfile.open(tar_path, "w:gz") as tar:
            for name in bundle:
                p = pilot_dir / name
                if not p.is_file():
                    raise FileNotFoundError(p)
                tar.add(p, arcname=name)
            tar.add(disease, arcname="data/graph_baseline/disease_genes_common_v1.txt")
        key = f"{staging_root}/source/final-gcn-{ts}.tar.gz"
        s3_uri = f"s3://{code_bucket}/{key}"
        subprocess.run(["aws", "s3", "cp", tar_path, s3_uri, "--region", region, "--sse", "AES256"], check=True)
        return s3_uri
    finally:
        Path(tar_path).unlink(missing_ok=True)


def _stage_triple(
    code_bucket: str,
    region: str,
    staging_root: str,
    ts: int,
    features_src: str,
    labels_src: str,
    drug_target_src: str,
) -> tuple[str, str, str]:
    base = f"s3://{code_bucket}/{staging_root}/data/final_gcn/{ts}/"
    f_dst = f"{base}features.parquet"
    l_dst = f"{base}labels.parquet"
    d_dst = f"{base}drug_target.parquet"
    subprocess.run(["aws", "s3", "cp", features_src, f_dst, "--region", region, "--sse", "AES256"], check=True)
    subprocess.run(["aws", "s3", "cp", labels_src, l_dst, "--region", region, "--sse", "AES256"], check=True)
    subprocess.run(["aws", "s3", "cp", drug_target_src, d_dst, "--region", region, "--sse", "AES256"], check=True)
    return f_dst, l_dst, d_dst


def main() -> None:
    args = parse_args()
    if args.sync_to_final_three and not args.wait:
        raise SystemExit("--sync-to-final-three requires --wait")
    ts = int(time.time())
    staging_root = "team4-final-gcn-train"
    code_bucket = args.code_bucket or f"sagemaker-{args.region}-{args.sagemaker_account_id}"
    pilot_dir = Path(__file__).resolve().parent
    # pilot_dir = <repo>/ml/pilot_sagemaker → repo root is parents[1], not parents[2]
    repo_root = pilot_dir.parents[1]

    source_s3 = _upload_source_tarball(pilot_dir, repo_root, code_bucket, args.region, staging_root, ts)
    fs, ls, ds = _stage_triple(
        code_bucket, args.region, staging_root, ts, args.features_uri, args.labels_uri, args.drug_target_uri
    )

    boto_sess = boto3.Session(region_name=args.region)
    sess = Session(boto_session=boto_sess, default_bucket=code_bucket)
    job_name = f"team4-final-gcn-{ts}"[:63]
    est = PyTorch(
        entry_point="train_gcn_final.py",
        source_dir=source_s3,
        role=args.role,
        framework_version=args.framework_version,
        py_version="py310",
        instance_type=args.instance_type,
        instance_count=1,
        sagemaker_session=sess,
        output_path=args.output_prefix,
        hyperparameters={
            "labels_s3": ls,
            "features_s3": fs,
            "drug_target_s3": ds,
            "disease_genes_path": "data/graph_baseline/disease_genes_common_v1.txt",
            # Omit ppi_edges_s3 when unused: empty str becomes `--ppi_edges_s3` with no value → argparse error.
            "seed": "42",
            "test_size": "0.1",
            "hidden_dim": "64",
            "lr": "0.001",
            "weight_decay": "0.00001",
            "epochs": "80",
            "patience": "12",
        },
    )
    est.fit(wait=args.wait, job_name=job_name)
    print("submitted_job_name:", est.latest_training_job.name)
    print("source_s3:", source_s3)
    print("staged_features_s3:", fs)
    print("staged_labels_s3:", ls)
    print("staged_drug_target_s3:", ds)
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
                "gcn",
                "--model-tar-s3",
                str(md),
                "--region",
                args.region,
            ],
            check=True,
        )
    elif not args.wait:
        print(
            "After job: aws s3 cp <output>/model.tar.gz /tmp/m.tar.gz && "
            "python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py --family gcn --model-tar /tmp/m.tar.gz"
        )


if __name__ == "__main__":
    main()
