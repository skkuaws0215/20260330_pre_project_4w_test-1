#!/usr/bin/env python3
"""
METABRIC survival-linked clinical relevance validation (NOT drug-response label validation).

Uses model-specific top drug sets from native ensemble ranking, derives per-sample pathway
activation summaries from pair features, joins cBioPortal-style clinical OS, then runs
Kaplan–Meier, log-rank, and Cox (optional) on high vs low activation (median split).
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import logrank_test


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _s3_field_reference() -> dict[str, Any]:
    """Documented S3 layout under s3://drug-discovery-joe-raw-data-team4/8/ (verified via aws s3 ls)."""
    return {
        "bucket": "drug-discovery-joe-raw-data-team4",
        "prefix_8": "s3://drug-discovery-joe-raw-data-team4/8/",
        "metabric_expression": {
            "08_raw.parquet": "s3://drug-discovery-joe-raw-data-team4/8/metabric/08_raw.parquet",
            "54_filtered.parquet": "s3://drug-discovery-joe-raw-data-team4/8/metabric/54_filtered.parquet",
            "shape_note": "Rows ~19k genes/features; columns ~2k METABRIC sample IDs (MB-xxxx). No OS/clinical columns in-matrix.",
        },
        "team_preprocessed_mirror": {
            "note": "README lists results/metabric/08_raw.parquet and 54_filtered.parquet under shared results/ (same scientific content as 8/metabric).",
        },
        "survival_and_clinical_fields": {
            "source": "Not present in 8/metabric parquets. Use cBioPortal METABRIC (study brca_metabric) clinical export, e.g. data_clinical_sample / merged TSV.",
            "overall_survival_time_months": {
                "example_column": "Overall Survival (Months)",
                "description": "Follow-up time in months for overall survival endpoint.",
            },
            "overall_survival_event": {
                "example_column": "Overall Survival Status",
                "encoding_note": "Often 0:LIVING (censored) vs 1:DECEASED (event); script parses DECEASED/LIVING tokens.",
            },
            "sample_key": {
                "example_column": "Sample ID",
                "join_to_native_ranking": "Must match sample_id in metabric_native_ensemble_ranking_no_lincs.csv (MB-xxxx).",
            },
            "auxiliary_clinical": [
                "Pam50 + Claudin-low subtype",
                "ER Status",
                "PR Status",
                "HER2 Status",
                "3-Gene classifier subtype",
                "Tumor Stage",
                "Neoplasm Histologic Grade",
            ],
        },
    }


def _load_clinical(tsv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(tsv_path, sep="\t", low_memory=False)
    df["sample_id"] = df["Sample ID"].astype(str).str.strip()
    df["os_months"] = pd.to_numeric(df["Overall Survival (Months)"], errors="coerce")
    stat = df["Overall Survival Status"].astype(str).str.upper()
    df["os_event"] = np.where(stat.str.contains("DECEASED"), 1, 0)
    df.loc[stat.str.contains("LIVING", na=False) & ~stat.str.contains("DECEASED", na=False), "os_event"] = 0
    df["pam50_subtype"] = df.get("Pam50 + Claudin-low subtype", pd.Series(index=df.index)).astype(str)
    df["er_status"] = df.get("ER Status", pd.Series(index=df.index)).astype(str)
    df["her2_status"] = df.get("HER2 Status", pd.Series(index=df.index)).astype(str)
    df["tumor_stage"] = df.get("Tumor Stage", pd.Series(index=df.index)).astype(str)
    return df


def _top_drugs_by_model(rank: pd.DataFrame, col: str, top_k: int) -> list[str]:
    """Drugs with best average within-sample rank (1 = top in that sample). Stable when batch preds tie within sample."""
    r = rank.copy()
    r["_within_rank"] = r.groupby("sample_id", group_keys=False)[col].rank(ascending=False, method="first")
    agg = r.groupby("canonical_drug_id")["_within_rank"].mean().sort_values()
    return [str(x) for x in agg.head(top_k).index.tolist()]


def _add_pathway_activation_column(pairs: pd.DataFrame) -> pd.DataFrame:
    out = pairs.copy()
    pw_cols = [c for c in out.columns if c.startswith("pathway__")]
    if pw_cols:
        out["pathway_activation_mean"] = out[pw_cols].mean(axis=1)
    elif "target_pathway_score_mean" in out.columns:
        out["pathway_activation_mean"] = out["target_pathway_score_mean"]
    else:
        out["pathway_activation_mean"] = 0.0
    return out


def _per_sample_pathway_activation(pairs: pd.DataFrame, drug_ids: list[Any], activation_col: str) -> pd.DataFrame:
    sub = pairs[pairs["canonical_drug_id"].isin(drug_ids)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["sample_id", "activation_score", "n_drugs_in_set"])
    return (
        sub.groupby("sample_id", as_index=False)
        .agg(
            activation_score=(activation_col, "mean"),
            n_drugs_in_set=("canonical_drug_id", "nunique"),
        )
    )


def _per_sample_model_score_mean(rank: pd.DataFrame, drug_ids: list[str], score_col: str) -> pd.DataFrame:
    """Mean model prediction over the candidate top-K drugs (varies by model column)."""
    sub = rank[rank["canonical_drug_id"].isin(drug_ids)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["sample_id", "activation_score", "n_drugs_in_set"])
    return (
        sub.groupby("sample_id", as_index=False)
        .agg(
            activation_score=(score_col, "mean"),
            n_drugs_in_set=("canonical_drug_id", "nunique"),
        )
    )


def _median_split(scores: pd.Series) -> pd.Series:
    med = scores.median()
    if scores.nunique(dropna=True) <= 1:
        return pd.Series(0, index=scores.index, dtype=int)
    return (scores > med).astype(int)


def _safe_float(x: Any) -> float | None:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return None
    try:
        v = float(x)
        if math.isnan(v) or math.isinf(v):
            return None
        return v
    except (TypeError, ValueError):
        return None


def _run_one_analysis(
    label: str,
    ranking_col: str,
    drug_ids: list[str],
    rank: pd.DataFrame,
    pairs: pd.DataFrame,
    clin_sub: pd.DataFrame,
    plot_path: Path | None,
    pathway_activation_col: str,
) -> dict[str, Any]:
    act = _per_sample_model_score_mean(rank, drug_ids, ranking_col)
    pw_act = _per_sample_pathway_activation(pairs, drug_ids, pathway_activation_col)
    m = clin_sub.merge(act, on="sample_id", how="inner")
    m = m.dropna(subset=["os_months", "activation_score"])
    m = m[m["os_months"] > 0]
    if len(m) < 10:
        return {
            "model_ranking_source": label,
            "ranking_score_column": ranking_col,
            "n_samples": int(len(m)),
            "error": "insufficient_samples_after_merge",
        }

    if m["activation_score"].nunique(dropna=True) <= 1:
        return {
            "model_ranking_source": label,
            "ranking_score_column": ranking_col,
            "n_samples": int(len(m)),
            "error": "activation_score_constant_cannot_split",
            "activation_definition": f"mean {ranking_col} over top-K pairs",
        }

    m["high_activation"] = _median_split(m["activation_score"])
    low = m[m["high_activation"] == 0]
    high = m[m["high_activation"] == 1]
    lr = logrank_test(
        low["os_months"],
        high["os_months"],
        low["os_event"],
        high["os_event"],
    )
    lr_p = _safe_float(lr.p_value)

    cox_hr = cox_lo = cox_hi = cox_p = None
    cox_note = None
    try:
        cdf = m[["os_months", "os_event", "high_activation"]].copy()
        cdf = cdf.astype(float)
        cph = CoxPHFitter(penalizer=0.05)
        cph.fit(cdf, duration_col="os_months", event_col="os_event", robust=True)
        summ = cph.summary
        if "high_activation" in summ.index:
            cox_hr = _safe_float(summ.loc["high_activation", "exp(coef)"])
            cox_lo = _safe_float(summ.loc["high_activation", "exp(coef) lower 95%"])
            cox_hi = _safe_float(summ.loc["high_activation", "exp(coef) upper 95%"])
            cox_p = _safe_float(summ.loc["high_activation", "p"])
    except Exception as exc:  # noqa: BLE001
        cox_note = str(exc)

    if (
        plot_path is not None
        and len(low) > 0
        and len(high) > 0
        and lr_p is not None
    ):
        plot_path.parent.mkdir(parents=True, exist_ok=True)
        kmf_l = KaplanMeierFitter()
        kmf_h = KaplanMeierFitter()
        fig, ax = plt.subplots(figsize=(6, 4))
        kmf_l.fit(low["os_months"], low["os_event"], label="low score (mean over top-K)")
        kmf_l.plot_survival_function(ax=ax, color="#6ea8fe")
        kmf_h.fit(high["os_months"], high["os_event"], label="high score (mean over top-K)")
        kmf_h.plot_survival_function(ax=ax, color="#ff6b6b")
        ax.set_xlabel("Months")
        ax.set_ylabel("Overall survival probability")
        ax.set_title(f"METABRIC OS · {label}\nlog-rank p={lr_p:.4g}" if lr_p is not None else f"METABRIC OS · {label}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)

    pathway_median_high = pathway_median_low = None
    if len(pw_act) > 0:
        pw2 = pw_act.rename(columns={"activation_score": "pathway_activation_score"})
        supp = m[["sample_id", "high_activation"]].merge(
            pw2[["sample_id", "pathway_activation_score"]], on="sample_id", how="left"
        )
        pathway_median_high = _safe_float(
            supp.loc[supp["high_activation"] == 1, "pathway_activation_score"].median()
        )
        pathway_median_low = _safe_float(
            supp.loc[supp["high_activation"] == 0, "pathway_activation_score"].median()
        )

    return {
        "model_ranking_source": label,
        "ranking_score_column": ranking_col,
        "top_drug_ids": [int(x) if str(x).isdigit() else x for x in drug_ids],
        "n_samples": int(len(m)),
        "activation_definition": f"mean {ranking_col} over top-K drug pairs (sample-specific)",
        "pathway_activation_median_high_group": pathway_median_high,
        "pathway_activation_median_low_group": pathway_median_low,
        "median_model_score_high_group": _safe_float(m.loc[m["high_activation"] == 1, "activation_score"].median()),
        "median_model_score_low_group": _safe_float(m.loc[m["high_activation"] == 0, "activation_score"].median()),
        "logrank_p_value": lr_p,
        "cox_hr_high_vs_low": cox_hr,
        "cox_hr_ci_lower": cox_lo,
        "cox_hr_ci_upper": cox_hi,
        "cox_p_value": cox_p,
        "cox_fit_note": cox_note,
        "km_plot_path": str(plot_path) if plot_path else None,
    }


def _subtype_breakdown(
    label: str,
    ranking_col: str,
    drug_ids: list[str],
    rank: pd.DataFrame,
    clin_sub: pd.DataFrame,
    min_n: int,
) -> list[dict[str, Any]]:
    act = _per_sample_model_score_mean(rank, drug_ids, ranking_col)
    m = clin_sub.merge(act, on="sample_id", how="inner")
    m = m.dropna(subset=["os_months", "activation_score"])
    m = m[m["os_months"] > 0]
    if m["activation_score"].nunique(dropna=True) <= 1:
        return []
    m["high_activation"] = _median_split(m["activation_score"])
    rows: list[dict[str, Any]] = []
    for sub, grp in m.groupby("pam50_subtype"):
        if len(grp) < min_n:
            continue
        low = grp[grp["high_activation"] == 0]
        high = grp[grp["high_activation"] == 1]
        if len(low) < 3 or len(high) < 3:
            continue
        lr = logrank_test(
            low["os_months"],
            high["os_months"],
            low["os_event"],
            high["os_event"],
        )
        rows.append(
            {
                "model_ranking_source": label,
                "pam50_subtype": str(sub),
                "n_samples": int(len(grp)),
                "logrank_p_value": _safe_float(lr.p_value),
            }
        )
    return rows


def main() -> None:
    repo = _repo_root()
    ap = argparse.ArgumentParser(description="METABRIC survival-linked validation for top drug sets.")
    ap.add_argument(
        "--ensemble-ranking-csv",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features_run/metabric_native_ensemble_ranking_no_lincs.csv"
        ),
    )
    ap.add_argument(
        "--pair-features-parquet",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/native_pair_features_run/metabric_native_pair_features_no_lincs.parquet"
        ),
    )
    ap.add_argument(
        "--clinical-tsv",
        default=os.environ.get(
            "METABRIC_CLINICAL_TSV",
            str(Path.home() / "Downloads" / "brca_metabric_clinical_data.tsv"),
        ),
        help="cBioPortal-style METABRIC clinical TSV (tab). Override with METABRIC_CLINICAL_TSV.",
    )
    ap.add_argument("--top-k-drugs", type=int, default=10)
    ap.add_argument(
        "--out-dir",
        default=str(
            repo
            / "results/features_nextflow_team4/fe_re_batch_runs/20260402/metabric_true_validation_prep/metabric_survival_validation"
        ),
    )
    ap.add_argument("--subtype-min-n", type=int, default=12)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    km_dir = out_dir / "km_plots"
    km_dir.mkdir(parents=True, exist_ok=True)

    rank = pd.read_csv(args.ensemble_ranking_csv)
    rank["sample_id"] = rank["sample_id"].astype(str).str.strip()
    rank["canonical_drug_id"] = rank["canonical_drug_id"].astype(str).str.strip()

    pairs = pd.read_parquet(args.pair_features_parquet)
    pairs["sample_id"] = pairs["sample_id"].astype(str).str.strip()
    pairs["canonical_drug_id"] = pairs["canonical_drug_id"].astype(str).str.strip()
    pairs = _add_pathway_activation_column(pairs)
    activation_col = "pathway_activation_mean"

    clin_path = Path(args.clinical_tsv)
    if not clin_path.is_file():
        raise SystemExit(
            f"Clinical TSV not found: {clin_path}. "
            "Set --clinical-tsv or METABRIC_CLINICAL_TSV to cBioPortal METABRIC export."
        )
    clin = _load_clinical(clin_path)
    samples = rank["sample_id"].unique().tolist()
    clin_sub = clin[clin["sample_id"].isin(samples)].drop_duplicates(subset=["sample_id"])

    models = [
        ("xgb_top_set", "pred_xgb"),
        ("residualmlp_top_set", "pred_residualmlp"),
        ("gcn_top_set", "pred_gcn"),
        ("ensemble_top_set", "ensemble_score"),
    ]

    summary_rows: list[dict[str, Any]] = []
    analyses: list[dict[str, Any]] = []
    subtype_rows: list[dict[str, Any]] = []

    for label, col in models:
        drugs = _top_drugs_by_model(rank, col, args.top_k_drugs)
        plot_p = km_dir / f"km_{label}.png"
        res = _run_one_analysis(label, col, drugs, rank, pairs, clin_sub, plot_p, activation_col)
        analyses.append(res)
        summary_rows.append(
            {
                "validation_purpose": "survival_linked_clinical_relevance",
                "not_drug_response_validation": True,
                "model_ranking_source": res.get("model_ranking_source"),
                "ranking_score_column": res.get("ranking_score_column"),
                "top_k": args.top_k_drugs,
                "top_drug_ids_json": json.dumps(res.get("top_drug_ids", [])),
                "n_samples": res.get("n_samples"),
                "logrank_p_value": res.get("logrank_p_value"),
                "cox_hr_high_vs_low": res.get("cox_hr_high_vs_low"),
                "cox_hr_ci_lower": res.get("cox_hr_ci_lower"),
                "cox_hr_ci_upper": res.get("cox_hr_ci_upper"),
                "cox_p_value": res.get("cox_p_value"),
                "cox_fit_note": res.get("cox_fit_note"),
                "km_plot_relative": str(plot_p.relative_to(out_dir)) if res.get("km_plot_path") else None,
                "error": res.get("error"),
            }
        )
        subtype_rows.extend(_subtype_breakdown(label, col, drugs, rank, clin_sub, args.subtype_min_n))

    # Rank models by smallest log-rank p (exploratory)
    valid = [r for r in summary_rows if r.get("logrank_p_value") is not None]
    ranked = sorted(valid, key=lambda r: (r["logrank_p_value"] is None, r["logrank_p_value"] or 1.0))

    report: dict[str, Any] = {
        "validation_type": "METABRIC survival-linked clinical relevance",
        "explicitly_not": "drug response / label external validation",
        "s3_and_clinical_field_reference": _s3_field_reference(),
        "inputs": {
            "ensemble_ranking_csv": str(Path(args.ensemble_ranking_csv).resolve()),
            "pair_features_parquet": str(Path(args.pair_features_parquet).resolve()),
            "clinical_tsv": str(clin_path.resolve()),
        },
        "parameters": {
            "top_k_drugs": args.top_k_drugs,
            "top_drug_selection": "lowest mean within-sample rank of model score (per sample, rank 1 = highest score)",
            "survival_group_score": "per model: mean of that model's prediction column over the top-K drugs (sample-specific)",
            "pathway_supplement": "mean of pathway__* columns over top-K pairs reported as secondary medians (may be drug-invariant in some FE builds)",
        },
        "per_model_analyses": analyses,
        "model_comparison_by_logrank_p": [
            {"model_ranking_source": r["model_ranking_source"], "logrank_p_value": r["logrank_p_value"]} for r in ranked
        ],
        "interpretation_note": (
            "High vs low groups use a median split on the sample-level mean model score over each model's top-K drugs. "
            "Cox HR is for high vs low of that split (exploratory; not causal treatment effect). "
            "Pathway medians in per_model_analyses describe the same samples under a mean HALLMARK pathway vector over the top-K pairs. "
            "If a model's prediction is identical across drugs within a sample (common for XGB in this run), that mean equals the per-sample pred and the top-K choice does not change the grouping."
        ),
        "subtype_logrank_supplement": subtype_rows,
    }
    if ranked:
        report["best_logrank_model"] = ranked[0]["model_ranking_source"]
        report["best_logrank_p"] = ranked[0]["logrank_p_value"]

    out_csv = out_dir / "metabric_survival_validation_summary.csv"
    out_json = out_dir / "metabric_survival_validation_report.json"
    pd.DataFrame(summary_rows).to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Wrote {out_csv}")
    print(f"Wrote {out_json}")
    print(f"KM plots under {km_dir}")


if __name__ == "__main__":
    main()
