# Code Bundle

This folder groups the scripts used for the teacher-model and global-surrogate experiments.

## Structure

- `global_surrogate/`
  - Scripts used to train and evaluate EBM and GAMI-Net surrogates.
  - Preserves the original `artifact_2`, `artifact_3`, and `artifact_4` package layout so relative imports still match the experiment code.
- `teacher_training/`
  - Script used to build the processed feature matrix, split the dataset, and train the teacher models.
- `sampling/`
  - Script used to generate local-permutation and VAE synthetic datasets.

## Teacher Training

- `teacher_training/train_model_pipeline.py`
  - Loads `business_phishing_dataset.csv`
  - Engineers the processed feature matrix
  - Splits data into `train/val/test`
  - Saves `artifacts/processed_dataset_with_split.csv`
  - Trains and saves `random_forest.pkl` and `deep_neural_net.pkl`

## Sampling

- `sampling/generate_synthetic_data.py`
  - Generates `synthetic_local_permutation_*.csv`
  - Trains the VAE and generates `synthetic_vae_*.csv`
  - Saves VAE reconstruction CSVs and summary JSONs
- `sampling/score_teacher_plausibility.py`
  - Optional teacher-plausibility filtering helper used by the VAE generation script

## Global Surrogate Scripts By Experiment

### 92k raw train split only

- `global_surrogate/artifact_4/prepare_real_train_teacher_datasets_artifact4.py`
  - Pseudo-labels the real `train` split with each teacher
- `global_surrogate/artifact_4/train_ebm_surrogates_artifact4.py`
  - Trains EBM on the 92,602-row real-train pseudo-label set
- `global_surrogate/artifact_4/train_gaminet_surrogates_artifact4.py`
  - Trains GAMI-Net on the 92,602-row real-train pseudo-label set
- `global_surrogate/artifact_4/collect_artifact4_report.py`
  - Collects fidelity/error-fidelity summaries

### Local permutation only

- `global_surrogate/artifact_2/train_ebm_surrogates_artifact2_variants.py`
  - Used for EBM local-only variants
- `global_surrogate/artifact_3/prepare_local_only_teacher_datasets_artifact3.py`
  - Prepares local-only pseudo-label datasets for GAMI-Net
- `global_surrogate/artifact_3/train_gaminet_surrogates_artifact3.py`
  - Trains GAMI-Net on local-only pseudo-label datasets
- `global_surrogate/artifact_3/collect_artifact3_report.py`
  - Collects local-only GAMI-Net summaries

### Local permutation + VAE

- `global_surrogate/artifact_2/train_ebm_surrogates_artifact2.py`
  - Main 300k local permutation + 100k VAE EBM run
- `global_surrogate/artifact_2/train_gaminet_surrogates_artifact2.py`
  - Main 300k local permutation + 100k VAE GAMI-Net run
- `global_surrogate/artifact_2/train_ebm_surrogates_artifact2_variants.py`
  - Variant-capable EBM script for local+VAE mixes
- `global_surrogate/artifact_2/train_gaminet_surrogates_artifact2_variant.py`
  - Variant-capable GAMI-Net script for local+VAE mixes
- `global_surrogate/artifact_2/collect_artifact2_report.py`
- `global_surrogate/artifact_2/collect_artifact2_variant_report.py`

## Shared Global-Surrogate Helpers

- `global_surrogate/plot_ebm_rf.py`
- `global_surrogate/plot_gaminet.py`
- `global_surrogate/train_gaminet_deep_neural_net.py`

