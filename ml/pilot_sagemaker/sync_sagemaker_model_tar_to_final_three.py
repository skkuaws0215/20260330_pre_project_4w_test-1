"""
Copy SageMaker Training Job model.tar.gz contents into sagemaker_final_three/artifacts/{family}/.

Usage:
  aws s3 cp s3://bucket/prefix/team4-final-xgb-.../output/model.tar.gz /tmp/model.tar.gz
  python3 ml/pilot_sagemaker/sync_sagemaker_model_tar_to_final_three.py --family xgb --model-tar /tmp/model.tar.gz
"""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_DEFAULT_OUT = (
    _REPO
    / "results/features_nextflow_team4/fe_re_batch_runs/20260331/sagemaker_final_three"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--family", choices=["xgb", "residualmlp", "gcn"], required=True)
    p.add_argument("--model-tar", type=Path, required=True, help="Local path to model.tar.gz from the Training Job.")
    p.add_argument("--out-root", type=Path, default=_DEFAULT_OUT)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    tar_path = args.model_tar.resolve()
    if not tar_path.is_file():
        raise FileNotFoundError(tar_path)
    dest = args.out_root / "artifacts" / args.family
    dest.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tdir = Path(tmp)
        with tarfile.open(tar_path, "r:gz") as tf:
            tf.extractall(tdir)
        copied = 0
        for p in tdir.rglob("*"):
            if p.is_file():
                rel = p.relative_to(tdir)
                out = dest / rel.name
                shutil.copy2(p, out)
                copied += 1
        if copied == 0:
            raise RuntimeError(f"No files found inside {tar_path}")

    print(f"Copied {copied} file(s) from {tar_path} -> {dest}")


if __name__ == "__main__":
    main()
