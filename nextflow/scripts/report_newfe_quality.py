from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Quality report for new FE outputs (newfe/newfe_v2 + component files)."
    )
    p.add_argument("--run-dir", required=True, help="Run directory path (contains final/)")
    p.add_argument("--out-json", default="quality_report.json")
    p.add_argument("--out-md", default="quality_report.md")
    p.add_argument("--sample-limit", type=int, default=20, help="Top columns to show in summaries")
    return p.parse_args()


def _load(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path)


def _numeric_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]


def _describe_file(df: pd.DataFrame, key_cols: list[str], top_n: int) -> dict[str, Any]:
    num_cols = _numeric_cols(df)
    null_ratio = df.isna().mean().sort_values(ascending=False)
    top_null = {
        k: float(v)
        for k, v in null_ratio.head(top_n).items()
        if float(v) > 0.0
    }

    near_zero_cols = []
    for c in num_cols:
        s = df[c].fillna(0.0)
        nz_ratio = float((s != 0).mean())
        if nz_ratio <= 0.001:
            near_zero_cols.append((c, nz_ratio))
    near_zero_cols = sorted(near_zero_cols, key=lambda x: x[1])[:top_n]

    key_dup = int(df.duplicated(subset=key_cols).sum()) if all(k in df.columns for k in key_cols) else None

    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "numeric_cols": int(len(num_cols)),
        "duplicate_key_rows": key_dup,
        "top_null_ratio": top_null,
        "top_near_zero_numeric_cols": [{k: float(v)} for k, v in near_zero_cols],
    }


def _target_signal_report(df_target: pd.DataFrame) -> dict[str, Any]:
    target_cols = [c for c in df_target.columns if c.startswith("target_")]
    out: dict[str, Any] = {"target_cols": target_cols}
    if not target_cols:
        return out
    stats = {}
    for c in target_cols:
        s = pd.to_numeric(df_target[c], errors="coerce").fillna(0.0)
        stats[c] = {
            "mean": float(s.mean()),
            "std": float(s.std()),
            "nonzero_ratio": float((s != 0).mean()),
            "max": float(s.max()),
        }
    out["summary"] = stats
    return out


def _newfe_diff_report(df_newfe: pd.DataFrame, df_newfe_v2: pd.DataFrame) -> dict[str, Any]:
    key_cols = ["sample_id", "canonical_drug_id"]
    report: dict[str, Any] = {}
    if not all(k in df_newfe.columns for k in key_cols) or not all(k in df_newfe_v2.columns for k in key_cols):
        report["error"] = "missing key columns in pair_features_newfe/newfe_v2"
        return report

    merged = df_newfe[key_cols].merge(
        df_newfe_v2[key_cols],
        on=key_cols,
        how="outer",
        indicator=True,
    )
    report["key_match"] = {
        "left_only": int((merged["_merge"] == "left_only").sum()),
        "right_only": int((merged["_merge"] == "right_only").sum()),
        "both": int((merged["_merge"] == "both").sum()),
    }

    target_cols = sorted([c for c in df_newfe_v2.columns if c.startswith("target_")])
    report["target_cols_in_v2"] = target_cols

    if target_cols:
        v2_num = df_newfe_v2[target_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        report["target_nonzero_ratio"] = {
            c: float((v2_num[c] != 0).mean()) for c in target_cols
        }
    return report


def _to_markdown(obj: dict[str, Any]) -> str:
    lines = []
    lines.append("# New FE Quality Report")
    lines.append("")
    lines.append("## Overview")
    lines.append("")
    files = obj.get("files", {})
    for name, info in files.items():
        lines.append(f"- `{name}`: rows={info['rows']}, cols={info['cols']}, numeric={info['numeric_cols']}, dup_keys={info['duplicate_key_rows']}")
    lines.append("")
    lines.append("## newfe vs newfe_v2")
    lines.append("")
    km = obj.get("newfe_diff", {}).get("key_match", {})
    if km:
        lines.append(f"- key match: both={km.get('both', 0)}, left_only={km.get('left_only', 0)}, right_only={km.get('right_only', 0)}")
    tnr = obj.get("newfe_diff", {}).get("target_nonzero_ratio", {})
    if tnr:
        lines.append("- target non-zero ratio (top):")
        for k, v in list(tnr.items())[:10]:
            lines.append(f"  - `{k}`: {v:.4f}")
    lines.append("")
    lines.append("## Target Signal")
    lines.append("")
    ts = obj.get("target_signal", {}).get("summary", {})
    if ts:
        for c, s in list(ts.items())[:10]:
            lines.append(f"- `{c}`: mean={s['mean']:.6f}, std={s['std']:.6f}, nonzero_ratio={s['nonzero_ratio']:.4f}, max={s['max']:.6f}")
    else:
        lines.append("- no target_* columns found")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    run_dir = Path(args.run_dir)
    final_dir = run_dir / "final"
    if not final_dir.exists():
        raise FileNotFoundError(f"final directory not found: {final_dir}")

    paths = {
        "sample_pathway_features": final_dir / "sample_pathway_features.parquet",
        "drug_chem_features": final_dir / "drug_chem_features.parquet",
        "pair_lincs_features": final_dir / "pair_lincs_features.parquet",
        "pair_target_features": final_dir / "pair_target_features.parquet",
        "pair_features_newfe": final_dir / "pair_features_newfe.parquet",
        "pair_features_newfe_v2": final_dir / "pair_features_newfe_v2.parquet",
    }

    dfs: dict[str, pd.DataFrame] = {k: _load(v) for k, v in paths.items()}
    key_map = {
        "sample_pathway_features": ["sample_id"],
        "drug_chem_features": ["canonical_drug_id"],
        "pair_lincs_features": ["sample_id", "canonical_drug_id"],
        "pair_target_features": ["sample_id", "canonical_drug_id"],
        "pair_features_newfe": ["sample_id", "canonical_drug_id"],
        "pair_features_newfe_v2": ["sample_id", "canonical_drug_id"],
    }

    report = {
        "run_dir": str(run_dir),
        "files": {
            name: _describe_file(df, key_map[name], args.sample_limit)
            for name, df in dfs.items()
        },
        "newfe_diff": _newfe_diff_report(
            dfs["pair_features_newfe"],
            dfs["pair_features_newfe_v2"],
        ),
        "target_signal": _target_signal_report(dfs["pair_target_features"]),
    }

    out_json = run_dir / args.out_json
    out_md = run_dir / args.out_md
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    out_md.write_text(_to_markdown(report), encoding="utf-8")
    print(str(out_json))
    print(str(out_md))


if __name__ == "__main__":
    main()
