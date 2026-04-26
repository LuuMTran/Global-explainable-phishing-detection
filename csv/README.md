# CSV Bundle

This folder contains the main CSV files used by the teacher-training and global-surrogate experiments.

## Included files

- `raw/business_phishing_dataset.csv`
  - Raw phishing-email dataset used for teacher-model training and processed feature generation
- `processed/processed_dataset_with_split.csv`
  - Engineered feature matrix with `train/val/test` split labels
- `synthetic/synthetic_local_permutation_300k.csv`
  - Local-permutation synthetic dataset used in surrogate training
- `synthetic/synthetic_vae_ld20_warm10_temp0p85_filtered_300k.csv`
  - Main filtered VAE synthetic dataset used in the mixed local+VAE surrogate experiments

