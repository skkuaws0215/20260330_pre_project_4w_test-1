from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "IQR-based outlier summary for newfe_v2 continuous features "
            "(pathway scores, RDKit descriptors, target_* continuous). "
            "Excludes binary indicators (e.g. Morgan bits) and LINCS columns."
        )
    )
    p.add_argument(
        "--parquet-uri",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/final/pair_features_newfe_v2.parquet",
        help="Path to pair_features_newfe_v2.parquet",
    )
    p.add_argument(
        "--out-dir",
        default="results/features_nextflow_team4/fe_re_batch_runs/20260331/analysis_target_only/outlier_report",
        help="Directory for feature_outlier_summary.csv and outlier_report.json",
    )
    p.add_argument("--sample-id-col", default="sample_id")
    p.add_argument("--drug-id-col", default="canonical_drug_id")
    return p.parse_args()


def _feature_group(col: str) -> str | None:
    if col.startswith("pathway_"):
        return "pathway"
    if col.startswith("drug_desc_"):
        return "rdkit_descriptor"
    if col.startswith("target_"):
        return "target_continuous"
    return None


def _in_scope_columns(columns: list[str], sample_id_col: str, drug_id_col: str) -> list[str]:
    keys = {sample_id_col, drug_id_col}
    out: list[str] = []
    for c in columns:
        if c in keys:
            continue
        if "lincs" in c.lower():
            continue
        if c.startswith("drug_morgan_"):
            continue
        if _feature_group(c) is not None:
            out.append(c)
    return out


def _is_binary_or_indicator(s: pd.Series) -> bool:
    """True if every non-null value is (approximately) 0 or 1."""
    v = pd.to_numeric(s, errors="coerce").dropna()
    if v.empty:
        return True
    u = np.unique(v.to_numpy(dtype=float))
    if not np.all(np.isfinite(u)):
        return True
    return bool(np.all((np.abs(u) < 1e-9) | (np.abs(u - 1.0) < 1e-9)))


def _is_effectively_continuous(s: pd.Series) -> bool:
    v = pd.to_numeric(s, errors="coerce")
    if v.notna().sum() == 0:
        return False
    if _is_binary_or_indicator(v):
        return False
    return True


def _column_stats(s: pd.Series) -> dict[str, Any]:
    v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(v)
    n = int(mask.sum())
    if n == 0:
        return {
            "n_non_null": 0,
            "outlier_count": 0,
            "outlier_ratio": float("nan"),
            "q1": float("nan"),
            "q3": float("nan"),
            "iqr": float("nan"),
            "lower_fence": float("nan"),
            "upper_fence": float("nan"),
            "min": float("nan"),
            "p1": float("nan"),
            "p50": float("nan"),
            "p99": float("nan"),
            "max": float("nan"),
            "iqr_degenerate": True,
        }

    x = v[mask]
    q1, q3 = np.percentile(x, [25.0, 75.0])
    iqr = float(q3 - q1)
    lower = float(q1 - 1.5 * iqr)
    upper = float(q3 + 1.5 * iqr)
    scale = max(1.0, abs(q3))
    degenerate = iqr < 1e-12 * scale

    if degenerate:
        outlier_count = 0
    else:
        outlier_count = int(np.sum((x < lower) | (x > upper)))

    return {
        "n_non_null": n,
        "outlier_count": outlier_count,
        "outlier_ratio": float(outlier_count / n) if n else float("nan"),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "lower_fence": lower,
        "upper_fence": upper,
        "min": float(np.min(x)),
        "p1": float(np.percentile(x, 1.0)),
        "p50": float(np.percentile(x, 50.0)),
        "p99": float(np.percentile(x, 99.0)),
        "max": float(np.max(x)),
        "iqr_degenerate": degenerate,
    }


def main() -> None:
    args = parse_args()
    path = Path(args.parquet_uri)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(path)
    n_rows = int(len(df))
    scope_cols = _in_scope_columns(list(df.columns), args.sample_id_col, args.drug_id_col)

    rows: list[dict[str, Any]] = []
    skipped_binary: list[str] = []
    for c in scope_cols:
        if not _is_effectively_continuous(df[c]):
            skipped_binary.append(c)
            continue
        st = _column_stats(df[c])
        rows.append(
            {
                "column": c,
                "feature_group": _feature_group(c),
                "n_rows": n_rows,
                **st,
            }
        )

    summary_df = pd.DataFrame(rows)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(["outlier_ratio", "outlier_count"], ascending=[False, False])
    csv_path = out_dir / "feature_outlier_summary.csv"
    json_path = out_dir / "outlier_report.json"
    summary_df.to_csv(csv_path, index=False)

    by_group: dict[str, Any] = {}
    for g, sub in summary_df.groupby("feature_group"):
        by_group[str(g)] = {
            "n_columns": int(len(sub)),
            "mean_outlier_ratio": float(sub["outlier_ratio"].mean()) if len(sub) else float("nan"),
            "median_outlier_ratio": float(sub["outlier_ratio"].median()) if len(sub) else float("nan"),
            "max_outlier_ratio": float(sub["outlier_ratio"].max()) if len(sub) else float("nan"),
        }

    top_ratio = []
    if not summary_df.empty:
        head = summary_df.head(15)
        for _, r in head.iterrows():
            top_ratio.append(
                {
                    "column": r["column"],
                    "feature_group": r["feature_group"],
                    "outlier_ratio": float(r["outlier_ratio"]),
                    "outlier_count": int(r["outlier_count"]),
                    "iqr_degenerate": bool(r["iqr_degenerate"]),
                }
            )

    report: dict[str, Any] = {
        "source_parquet": path.as_posix(),
        "n_rows": n_rows,
        "method": {
            "outlier_rule": "Tukey IQR: value < Q1 - 1.5*IQR or value > Q3 + 1.5*IQR",
            "percentiles": "Q1/Q3 via numpy percentile 25/75; p1/p50/p99 at 1/50/99",
            "iqr_degenerate": "If IQR is ~0 relative to |Q3|, outlier_count forced to 0 (no fence spread).",
        },
        "scope": {
            "included_patterns": [
                "pathway_* (pathway block; e.g. pathway__<name> from GMT)",
                "drug_desc_* (RDKit descriptors, continuous)",
                "target_* (target block; binary-only columns excluded)",
            ],
            "excluded": [
                "sample_id, canonical_drug_id",
                "columns with 'lincs' in name (aligned with target-only modeling)",
                "drug_morgan_* (binary fingerprint bits)",
                "binary/indicator columns: non-null values only in {0, 1}",
            ],
        },
        "columns_in_scope_before_continuous_filter": len(scope_cols),
        "continuous_columns_analyzed": int(len(summary_df)),
        "skipped_binary_or_indicator_in_scope": skipped_binary,
        "by_feature_group": by_group,
        "top_outlier_ratio_columns": top_ratio,
    }

    # Short interpretation for dashboards / readers
    high = summary_df[summary_df["outlier_ratio"] >= 0.05] if not summary_df.empty else pd.DataFrame()
    report["interpretation_note"] = (
        "Outlier ratios are row-fractions per feature under Tukey IQR. "
        "High ratios may warrant winsorization, log1p, or robust scaling for neural models; "
        "tree models are often less sensitive. Compare with training preprocessing (e.g. missing→0, standard scaling)."
    )
    report["columns_with_outlier_ratio_ge_0.05"] = int(len(high)) if not high.empty else 0
    report["outputs"] = {
        "csv": csv_path.as_posix(),
        "json": json_path.as_posix(),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Wrote {csv_path}")
    print(f"Wrote {json_path}")


if __name__ == "__main__":
    main()
