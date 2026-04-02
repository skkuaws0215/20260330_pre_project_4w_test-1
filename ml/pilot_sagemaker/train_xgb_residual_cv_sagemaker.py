from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--labels_s3", required=True)
    p.add_argument("--features_s3", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_splits", type=int, default=5)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    out_dir = Path(os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    out_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            str(root / "run_residual_mlp_cv_local.py"),
            "--labels-uri",
            args.labels_s3,
            "--features-uri",
            args.features_s3,
            "--out-dir",
            str(out_dir),
            "--seed",
            str(args.seed),
            "--n-splits",
            str(args.n_splits),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
