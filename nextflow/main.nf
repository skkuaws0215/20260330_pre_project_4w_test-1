nextflow.enable.dsl = 2

params.run_id = params.run_id ?: new Date().format("yyyyMMdd_HHmm")
params.out_prefix = params.out_prefix ?: "s3://drug-discovery-joe-raw-data-team4/results/features_nextflow_team4"

/*
 Required input parquet URIs (shared team results/ as read-only input):
   -params.sample_feature_uri
   -params.drug_feature_uri
   -params.label_uri
*/
params.sample_feature_uri = params.sample_feature_uri ?: null
params.drug_feature_uri = params.drug_feature_uri ?: null
params.label_uri = params.label_uri ?: null

params.sample_id_col = params.sample_id_col ?: "sample_id"
params.drug_id_col = params.drug_id_col ?: "canonical_drug_id"
params.drug_fallback_col = params.drug_fallback_col ?: "inchikey"
params.regression_label_col = params.regression_label_col ?: "ic50"
params.binary_label_col = params.binary_label_col ?: "binary_label"

params.missing_threshold = params.missing_threshold ?: 0.7
params.variance_threshold = params.variance_threshold ?: 0.0
params.leakage_cols = params.leakage_cols ?: ""
params.normalization_branch = params.normalization_branch ?: "both" // tree|dl|both
params.binary_from_quantile = params.binary_from_quantile ?: 0.3

if (!params.sample_feature_uri || !params.drug_feature_uri || !params.label_uri) {
    exit 1, "Missing required params: sample_feature_uri, drug_feature_uri, label_uri"
}

workflow {
    Channel
        .of(tuple(params.sample_feature_uri, params.drug_feature_uri, params.label_uri, file("${projectDir}/scripts/build_features.py")))
        | BUILD_FEATURE_TABLE
}

process BUILD_FEATURE_TABLE {
    tag "${params.run_id}"

    publishDir "${params.out_prefix}/${params.run_id}", mode: "copy", overwrite: true

    input:
    tuple val(sample_feature_uri), val(drug_feature_uri), val(label_uri), path(build_script)

    output:
    path("features.parquet"), emit: features
    path("labels.parquet"), emit: labels
    path("feature_manifest.json"), emit: manifest
    path("features_dl.parquet"), optional: true, emit: features_dl

    script:
    """
    python3 "${build_script}" \
      --sample-feature-uri "${sample_feature_uri}" \
      --drug-feature-uri "${drug_feature_uri}" \
      --label-uri "${label_uri}" \
      --sample-id-col "${params.sample_id_col}" \
      --drug-id-col "${params.drug_id_col}" \
      --drug-fallback-col "${params.drug_fallback_col}" \
      --regression-label-col "${params.regression_label_col}" \
      --binary-label-col "${params.binary_label_col}" \
      --missing-threshold "${params.missing_threshold}" \
      --variance-threshold "${params.variance_threshold}" \
      --leakage-cols "${params.leakage_cols}" \
      --normalization-branch "${params.normalization_branch}" \
      --binary-from-quantile "${params.binary_from_quantile}" \
      --run-id "${params.run_id}" \
      --out-features "features.parquet" \
      --out-labels "labels.parquet" \
      --out-manifest "feature_manifest.json"
    """
}
