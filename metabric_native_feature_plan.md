# METABRIC cohort-native feature generation plan

This document prepares the path from **bridge-based train-row reuse** (`build_metabric_external_eval_tables.py`) to **METABRIC-native** pair features: recomputing the same FE *definitions* from METABRIC expression (and shared drug assets), then applying **train-frozen** transforms for inference.

---

## 1. Goals and constraints

| Goal | Detail |
|------|--------|
| Schema parity | Final pair table must expose **the same feature column names and order** as TCGA train `pair_features_newfe_v2.parquet` (excluding key columns, any join helpers are separate). |
| Transform-only (models) | **No** refitting feature selection, **no** refitting scalers/encoders used at inference. Use artifacts from the trained bundle (e.g. XGB `feature_columns`; ResidualMLP `feat_cols` + `cont_idx` + `scaler_mean` / `scaler_scale` in `checkpoint.pt`; GCN checkpoint fields as in `build_final_ensemble_ranking.py`). |
| Cohort-native biology | Sample-side signal (pathway, LINCS overlap, target overlap / expr stats) must be computed from **METABRIC** expression prepared to match the FE script’s expectations—not copied from a TCGA `sample_id` row. |

---

## 2. Reference implementation (TCGA / train)

The authoritative FE recipe is implemented in:

- `nextflow/scripts/build_pair_features_newfe_v2.py`

It builds, in order of assembly:

1. **Sample pathway** — GMT (`--pathway-gmt`, e.g. Hallmark) + per-sample gene matrix; columns `pathway__*`.
2. **Drug chemistry** — RDKit Morgan (`drug_morgan_0000` …) + descriptors (`drug_desc_*`, `drug_has_valid_smiles`) from `canonical_smiles`.
3. **Pair LINCS** — intersection of numeric columns between **sample expression** and **LINCS drug signature** matrices; cosine, Pearson, Spearman, reverse-score top-k.
4. **Pair target** — drug targets vs sample high/low gene sets, pathway relevance, coverage; uses **the same numeric thresholds** on sample gene values as in training (`high_z`, `low_z`).

Frozen hyperparameters and semantics for a concrete train run are recorded in:

- `results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_hallmark/feature_manifest.json`  
  (example: `high_gene_rule = zscore >= 1.0`, `low_gene_rule = zscore <= -1.0`, Morgan radius `2`, `2048` bits, `lincs_reverse_score_top50` / `top100`.)

**Gold column order** for parity checks: read the header from the train artifact actually used for SageMaker final training, e.g.:

- `results/features_nextflow_team4/fe_re_batch_runs/20260331/final_pathway_addon/pair_features_newfe_v2.parquet`

Store a checked-in or run-scoped `feature_column_order_train.json` (list of non-key feature names in order) if you want CI-style drift detection.

---

## 3. Bridge-based vs METABRIC-native (documentation of difference)

| Aspect | Bridge-based external table (`build_metabric_external_eval_tables.py`) | METABRIC-native (this plan) |
|--------|------------------------------------------------------------------------|-----------------------------|
| **Primary key** | `(internal TCGA sample_id, canonical_drug_id)` after bridge; METABRIC ID carried as metadata. | Natural key `(metabric_sample_id, canonical_drug_id)`; `sample_id` in FE inputs should be **MB-*** (or a stable METABRIC id) so all sample-derived blocks are cohort-consistent. |
| **Pathway / sample block** | TCGA pipeline values for the **mapped** internal `sample_id`. | Pathway scores from **METABRIC** gene matrix + **same GMT + same aggregation** (mean of matched genes per pathway). |
| **LINCS pair block** | TCGA sample vector × LINCS drug vector (intersection of columns). | METABRIC sample vector × **same** LINCS drug table; requires **harmonized column names** with the METABRIC-wide sample matrix. |
| **Target pair block** | TCGA z-scored (or train-normalized) gene values for mapped sample. | METABRIC-based gene values with **explicitly defined** z-score (or rank) reference so thresholds `high_z` / `low_z` remain interpretable. |
| **Drug chemistry block** | Identical to train row for that drug if copied from train FE. | **Should be identical** if `canonical_drug_id` + SMILES + RDKit version match; recomputing from the same drug table is preferred for reproducibility. |
| **Labels** | Internal `labels_B_graph` subset (not METABRIC clinical response). | Must attach a **separate** external label source (see §6). |
| **Validation meaning** | “Same features as train for a proxy sample id” — useful for pipeline debugging. | “True external cohort” for sample-side generalization; still not independent drugs if drug table is shared. |

---

## 4. Proposed native generation path (high level)

1. **Define pair universe**  
   Cartesian or curated list: `(metabric_sample_id, canonical_drug_id)` for samples present in METABRIC matrix and drugs in scope (e.g. overlap with train drug set).

2. **Build `sample_expression` parquet (METABRIC)**  
   - Input: e.g. `8/metabric/54_filtered.parquet` (wide matrix; **samples-as-columns** in current bucket layout).  
   - Output: one row per `metabric_sample_id`, columns aligned to the **same naming convention** as TCGA `sample_features` (e.g. `crispr__SYMBOL` or project-standard prefix + HGNC uppercase token) so `build_pair_features_newfe_v2.py` logic applies unchanged.  
   - **Transpose** and **rename** rows/columns; resolve gene IDs via a frozen mapping table (see checklist).

3. **Apply gene-level normalization consistent with target rules**  
   The target extension assumes **thresholding on numeric gene values** (documented as z-scores in `feature_manifest.json`). You must either:  
   - match TCGA’s per-gene scaling recipe on METABRIC (e.g. z within METABRIC cohort), and accept domain shift; or  
   - freeze TCGA reference mean/std per gene and apply to METABRIC (stronger “train-defined space”, still not refitting model scalers).  
   Record the chosen rule in a new `metabric_native_feature_manifest.json`.

4. **Reuse frozen non-sample inputs**  
   - Same `--pathway-gmt` as train (or explicitly versioned GMT file).  
   - Same `--drug-uri` / SMILES source as train (or versioned snapshot).  
   - Same `--lincs-drug-signature-uri` and `--drug-target-uri` as train.  
   - Same CLI knobs: `--morgan-radius`, `--morgan-nbits`, `--high-z-threshold`, `--low-z-threshold`, `--reverse-topk-small`, `--reverse-topk-large`.

5. **Run FE merge**  
   Either call `build_pair_features_newfe_v2.py` with `--pairs-uri` listing METABRIC pairs and `--sample-expression-uri` pointing to the METABRIC-wide table, or factor a small library wrapper that imports its functions (avoid duplicating logic).

6. **Schema alignment pass**  
   - Reorder columns to match train `pair_features_newfe_v2`.  
   - For any column missing after native build (e.g. gene absent in METABRIC), apply the **same fill policy** as train (`fillna(0.0)` for numerics in the builder).  
   - Assert no extra columns or fail with a diff report.

7. **Model inference (transform-only)**  
   Load raw feature matrix in train column order; apply **only** stored `cont_idx` + scaler vectors for MLP; XGB column subset from joblib; GCN as implemented. No `StandardScaler().fit` on METABRIC.

---

## 5. Transform-only: what is frozen where

| Component | Frozen source | Notes |
|-----------|---------------|--------|
| Feature **names and order** | Train `pair_features_newfe_v2.parquet` header (or exported JSON). | Native builder output must match after alignment. |
| Pathway gene sets | Versioned GMT file path recorded in manifest. | Changing GMT changes semantics. |
| Morgan / descriptors | `morgan_radius`, `morgan_nbits`, RDKit version. | Pin RDKit in env. |
| LINCS top-k | `reverse_topk_small`, `reverse_topk_large`. | Must match train manifest. |
| Target high/low cutoffs | `high_z_threshold`, `low_z_threshold`. | Must match train manifest. |
| XGB feature set | `artifact.joblib` → `feature_columns`. | Subset of full parquet; LINCS columns may be excluded in training—follow bundle. |
| ResidualMLP scaling | `checkpoint.pt` → `cont_idx`, `scaler_mean`, `scaler_scale`, `feat_cols`. | Apply only to continuous indices; binary columns untouched (see `build_final_ensemble_ranking.py`). |
| GCN | `gcn_checkpoint.pt` nodes, `feat_cols`, `h0`, etc. | Same pattern: no refit. |

**Important:** Cohort-native recomputation of pathway/LINCS/target **does not** violate transform-only for sklearn/torch **as long as** those steps are pure functions of METABRIC inputs and train-**fixed** hyperparameters. The subtle part is **gene normalization**: refitting global z-scores on METABRIC-only data would be a new transform; prefer a **documented, frozen** normalization recipe (§4 step 3).

---

## 6. External label source connection points

Native evaluation needs labels keyed to **pairs** or joinable keys:

| Potential source | Key shape | Connection point |
|------------------|-----------|-------------------|
| Current internal graph labels | `(TCGA sample_id, canonical_drug_id)` | Only valid for bridge table; **not** native METABRIC truth. |
| METABRIC clinical / treatment tables | `metabric_sample_id` (+ time/event) | Join on sample; drug mapping to `canonical_drug_id` via a curated map (brand → InChIKey/SMILES → internal id). |
| Drug sensitivity / screening (e.g. GDSC-style) | `(cell line id, drug id)` | Requires mapping METABRIC sample to cell line or using a different validation cohort; document lineage. |
| User-supplied response CSV | `(metabric_sample_id, canonical_drug_id)` → `label_regression` | Easiest for pipeline glue: same columns as `labels_B_graph` where possible (`label_regression`, optional `label_binary`). |

Recommended artifact:

- `metabric_native_labels.parquet` with at least `metabric_sample_id`, `canonical_drug_id`, `label_regression`, plus provenance columns (`label_source`, `label_version`).

Join to native features on `(metabric_sample_id, canonical_drug_id)` (or normalize `sample_id` = MB-* in both tables).

---

## 7. Inputs / mappings / labels checklist

### 7.1 Required input files

- [ ] METABRIC expression matrix (e.g. `54_filtered.parquet`) — confirm orientation (genes × samples vs samples × genes) and dtype.
- [ ] Gene identifier manifest for matrix rows (Ensembl / probe / symbol) if not already symbol-level.
- [ ] Frozen mapping **METABRIC row id → HGNC (or project) symbol** compatible with TCGA `sample_features` column tokenization in `build_pair_features_newfe_v2.py` (`split("__")[-1].upper()`).
- [ ] Train **`pair_features_newfe_v2.parquet`** (or header export) for column order and parity tests.
- [ ] Train **`feature_manifest.json`** (or equivalent) for `high_z`, `low_z`, Morgan, top-k, GMT path.
- [ ] **`drug_target_map`** parquet (same as train).
- [ ] **Drug table** with `canonical_drug_id`, `canonical_smiles` (train snapshot).
- [ ] **LINCS drug signature** parquet keyed by `canonical_drug_id` (train snapshot).
- [ ] **Hallmark (or same) GMT** file path pinned to train.

### 7.2 Mapping and QC gates

- [ ] **Transpose + reshape** METABRIC → wide `sample_id` × gene columns; spot-check overlap fraction with TCGA gene set used in train `sample_features`.
- [ ] **LINCS column intersection** non-empty after harmonization (script raises if `common` is empty).
- [ ] **Pathway coverage**: count of pathways with ≥1 matched gene vs train.
- [ ] **Pair table row count** vs expected pairs; duplicate key check on `(sample_id, canonical_drug_id)`.
- [ ] **Schema diff report**: missing / extra / reorder vs train feature columns.
- [ ] **RDKit version** logged next to Morgan output for audit trail.

### 7.3 Label source readiness

- [ ] Chosen label table identified and licensed for use.
- [ ] Stable join keys to `metabric_sample_id` and `canonical_drug_id`.
- [ ] Definition of `label_regression` aligned with model training target (e.g. same transform of IC50 / AUC / z-score).
- [ ] Handling of censored / missing labels documented (drop pair vs mask in metrics).

### 7.4 Inference bundle readiness

- [ ] XGB `artifact.joblib` path fixed.
- [ ] ResidualMLP `checkpoint.pt` with `feat_cols`, `cont_idx`, scaler arrays.
- [ ] GCN `gcn_checkpoint.pt` and PPI / disease gene inputs if used in ensemble.
- [ ] End-to-end script: read native parquet → predict → write `metabric_native_predictions.parquet` + metric summary.

---

## 8. Suggested next implementation artifacts (after this plan)

1. `metabric_to_sample_expression.py` (or Nextflow process) — matrix → `sample_expression`-shaped parquet + QC JSON.  
2. `build_metabric_native_pair_features.py` — thin orchestrator: pairs → call shared FE → column realign → manifest.  
3. `metabric_native_feature_manifest.json` — pins all paths, versions, and normalization choice.  
4. Optional: extend `run_metabric_true_validation_pipeline.py` manifest with a `cohort_native` branch once (1)–(3) exist.

---

## 9. Related repo files

| File | Role |
|------|------|
| `nextflow/scripts/build_pair_features_newfe_v2.py` | FE definition for native replay. |
| `ml/pilot_sagemaker/build_metabric_external_eval_tables.py` | Bridge-based reuse (contrast). |
| `ml/pilot_sagemaker/build_final_ensemble_ranking.py` | Reference for transform-only inference on a feature parquet. |
| `ml/pilot_sagemaker/train_residual_mlp_final.py` | Saves `feat_cols`, `cont_idx`, scaler stats in `checkpoint.pt`. |
| `ml/pilot_sagemaker/prepare_metabric_true_validation_inputs.py` | Schema audit for raw METABRIC parquets. |

This plan stops at **design and checklists**; implementation of §8 can follow in a separate change set.
