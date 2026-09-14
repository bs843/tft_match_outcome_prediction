| model | cv_f1_mean | cv_f1_std | cv_folds | validation_accuracy | validation_precision | validation_recall | validation_f1 | validation_roc_auc | validation_pr_auc | test_accuracy | test_precision | test_recall | test_f1 | test_roc_auc | test_pr_auc | cv_to_validation_f1_gap | validation_to_test_f1_gap | cv_to_test_f1_gap | abs_cv_to_validation_f1_gap | abs_validation_to_test_f1_gap | abs_cv_to_test_f1_gap | cv_to_test_gap_in_sd | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.000 | 0.000 | 5 | 0.500 | 0.000 | 0.000 | 0.000 | 0.500 | 0.500 | — | — | — | — | — | — | 0.000 | — | — | 0.000 | — | — | — | False |
| logistic_regression | 0.849 | 0.003 | 5 | 0.851 | 0.853 | 0.850 | 0.851 | 0.929 | 0.931 | — | — | — | — | — | — | 0.002 | — | — | 0.002 | — | — | — | False |
| random_forest | 0.835 | 0.003 | 5 | 0.836 | 0.839 | 0.831 | 0.835 | 0.918 | 0.917 | — | — | — | — | — | — | -0.000 | — | — | 0.000 | — | — | — | False |
| xgboost | 0.849 | 0.004 | 5 | 0.855 | 0.857 | 0.852 | 0.854 | 0.931 | 0.933 | 0.853 | 0.852 | 0.856 | 0.854 | 0.930 | 0.933 | 0.005 | -0.001 | 0.004 | 0.005 | 0.001 | 0.004 | 0.972 | True |
| lightgbm | 0.850 | 0.006 | 5 | 0.853 | 0.854 | 0.851 | 0.853 | 0.931 | 0.932 | — | — | — | — | — | — | 0.003 | — | — | 0.003 | — | — | — | False |
| catboost | 0.848 | 0.005 | 5 | 0.852 | 0.852 | 0.853 | 0.852 | 0.930 | 0.931 | — | — | — | — | — | — | 0.005 | — | — | 0.005 | — | — | — | False |
