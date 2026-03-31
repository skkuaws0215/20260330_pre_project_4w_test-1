# New FE Quality Report

## Overview

- `sample_pathway_features`: rows=53, cols=1, numeric=0, dup_keys=0
- `drug_chem_features`: rows=601, cols=2059, numeric=2058, dup_keys=0
- `pair_lincs_features`: rows=25575, cols=7, numeric=5, dup_keys=0
- `pair_target_features`: rows=25575, cols=12, numeric=10, dup_keys=0
- `pair_features_newfe`: rows=25575, cols=2065, numeric=2063, dup_keys=0
- `pair_features_newfe_v2`: rows=25575, cols=2075, numeric=2073, dup_keys=0

## PASS / FAIL Gate

- overall: FAIL
- target gate: PASS (alive_cols=6)
- lincs gate: FAIL (fail_cols=5)
- target alive cols:
  - `target_overlap_down_count`
  - `target_overlap_down_ratio`
  - `target_expr_mean`
  - `target_expr_std`
  - `target_gene_coverage_ratio`
  - `target_gene_count`
- lincs fail cols:
  - `lincs_cosine`
  - `lincs_pearson`
  - `lincs_spearman`
  - `lincs_reverse_score_top50`
  - `lincs_reverse_score_top100`

## newfe vs newfe_v2

- key match: both=25575, left_only=0, right_only=0
- target non-zero ratio (top):
  - `target_expr_mean`: 0.1791
  - `target_expr_std`: 0.1148
  - `target_gene_count`: 0.3200
  - `target_gene_coverage_ratio`: 0.3200
  - `target_overlap_count`: 0.0000
  - `target_overlap_down_count`: 0.0536
  - `target_overlap_down_ratio`: 0.0536
  - `target_overlap_ratio`: 0.0000
  - `target_pathway_hit_count`: 0.0000
  - `target_pathway_score_mean`: 0.0000

## Target Signal

- `target_overlap_count`: mean=0.000000, std=0.000000, nonzero_ratio=0.0000, max=0.000000
- `target_overlap_ratio`: mean=0.000000, std=0.000000, nonzero_ratio=0.0000, max=0.000000
- `target_overlap_down_count`: mean=0.069560, std=0.320151, nonzero_ratio=0.0536, max=4.000000
- `target_overlap_down_ratio`: mean=0.029846, std=0.144658, nonzero_ratio=0.0536, max=1.000000
- `target_expr_mean`: mean=-0.075104, std=0.305606, nonzero_ratio=0.1791, max=0.464553
- `target_expr_std`: mean=0.042609, std=0.173297, nonzero_ratio=0.1148, max=1.868331
- `target_pathway_score_mean`: mean=0.000000, std=0.000000, nonzero_ratio=0.0000, max=0.000000
- `target_pathway_hit_count`: mean=0.000000, std=0.000000, nonzero_ratio=0.0000, max=0.000000
- `target_gene_coverage_ratio`: mean=0.318469, std=0.464505, nonzero_ratio=0.3200, max=1.000000
- `target_gene_count`: mean=1.092043, std=2.914512, nonzero_ratio=0.3200, max=40.000000

## LINCS Signal

- `lincs_cosine`: mean=0.000000, std=0.000000, nunique=1, nonzero_ratio=0.0000, max=0.000000
- `lincs_pearson`: mean=0.000000, std=0.000000, nunique=1, nonzero_ratio=0.0000, max=0.000000
- `lincs_spearman`: mean=0.000000, std=0.000000, nunique=1, nonzero_ratio=0.0000, max=0.000000
- `lincs_reverse_score_top50`: mean=0.000000, std=0.000000, nunique=1, nonzero_ratio=0.0000, max=0.000000
- `lincs_reverse_score_top100`: mean=0.000000, std=0.000000, nunique=1, nonzero_ratio=0.0000, max=0.000000
