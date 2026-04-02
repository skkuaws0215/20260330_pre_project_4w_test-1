"""Shared helpers: download model.tar.gz from S3 and sync into sagemaker_final_three/artifacts/."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    _REPO
    / "results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three"
)


def sync_extracted_files(tar_path: Path, dest: Path) -> int:
    dest.mkdir(parents=True, exist_ok=True)
    copied = 0
    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tdir)
        for p in tdir.rglob("*"):
            if p.is_file():
                out = dest / p.name
                shutil.copy2(p, out)
                copied += 1
    if copied == 0:
        raise RuntimeError(f"No files found inside {tar_path}")
    return copied


def download_s3_to_temp(uri: str, region: str) -> Path:
    with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as tf:
        local = Path(tf.name)
    subprocess.run(["aws", "s3", "cp", uri, str(local), "--region", region], check=True)
    return local


def sync_from_s3(uri: str, family: str, region: str, out_root: Path | None = None) -> None:
    out = out_root or _DEFAULT_OUT
    local = download_s3_to_temp(uri, region)
    try:
        n = sync_extracted_files(local, out / "artifacts" / family)
        print(f"sync_from_s3: {n} file(s) -> {out / 'artifacts' / family}")
    finally:
        local.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sync SageMaker model.tar.gz into sagemaker_final_three.")
    p.add_argument("--family", choices=["xgb", "residualmlp", "gcn"], required=True)
    p.add_argument("--model-tar", type=Path, default=None, help="Local model.tar.gz")
    p.add_argument("--model-tar-s3", type=str, default=None, help="s3://.../model.tar.gz to download first")
    p.add_argument("--region", default="ap-northeast-2")
    p.add_argument("--out-root", type=Path, default=_DEFAULT_OUT)
    args = p.parse_args()
    if args.model_tar is None and args.model_tar_s3 is None:
        p.error("Provide --model-tar or --model-tar-s3")
    return args


def main() -> None:
    args = parse_args()
    if args.model_tar_s3:
        sync_from_s3(args.model_tar_s3, args.family, args.region, args.out_root)
        return
    tar_path = args.model_tar.resolve()
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)
    n = sync_extracted_files(tar_path, args.out_root / "artifacts" / args.family)
    print(f"Copied {n} file(s) from {tar_path} -> {args.out_root / 'artifacts' / args.family}")


if __name__ == "__main__":
    main()
