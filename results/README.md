# Results Bundle

This folder contains readable copies of the result artifacts used for the paper tables and result-observation notebook.

## Teacher Models

- `teacher_models/teacher_test_metrics.json`: test-set performance for the Random Forest and MLP teachers.
- `teacher_models/teacher_peer_fidelity_summary.csv`: peer fidelity and peer error fidelity between the two teachers.

## Main 300k Local Perturbation + VAE Result

- `main_300k_plus_vae/surrogate_fidelity_error_summary.csv`: EBM and GAMI-Net overall fidelity, F1 fidelity, and error fidelity.
- `main_300k_plus_vae/surrogate_fidelity_error_report.json`: detailed nested fidelity report.
- `main_300k_plus_vae/logistic_regression_baseline_comparison.csv`: logistic-regression baseline comparison.
- `main_300k_plus_vae/decision_tree_depth5_baseline_comparison.csv`: depth-5 decision-tree baseline comparison.
- `main_300k_plus_vae/decision_tree_depth_leaf_sweep_summary.csv`: decision-tree depth/leaf sweep.
- `main_300k_plus_vae/decision_tree_depth_leaf_extended_sweep_summary.csv`: deeper decision-tree sweep.
- `main_300k_plus_vae/main_effect_comparison_summary.json`: EBM vs GAMI-Net main-effect comparison metadata.
- `main_300k_plus_vae/selected_common_top5_features.csv`: selected common features used in main-effect comparisons.
- `main_300k_plus_vae/surrogate_training_summary.json`: training summary for the main synthetic surrogate run.

## Ablation Results

- `ablation/ablation_error_fidelity_summary.csv`: corrected paper-ready ablation table.
- `ablation/raw_training_data_error_fidelity_summary.csv`: raw-training-data surrogate error-fidelity summary.
- `ablation/raw_training_data_ebm_summary.json`: raw-training-data EBM detailed summary.
- `ablation/raw_training_data_gaminet_summary.json`: raw-training-data GAMI-Net detailed summary.
- `ablation/local_permutation_only_ebm_summary.json`: local-perturbation-only EBM summary.
- `ablation/local_permutation_only_gaminet_summary.json`: local-perturbation-only GAMI-Net summary.

## Plots

- `plots/`: committed copies of the paper-facing feature-importance and main-effect comparison plots.
