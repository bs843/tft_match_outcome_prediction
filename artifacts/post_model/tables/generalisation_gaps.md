| model | cv_f1_mean | cv_f1_std | validation_f1 | test_f1 | cv_to_validation_f1_gap | validation_to_test_f1_gap | cv_to_test_f1_gap | abs_cv_to_validation_f1_gap | abs_validation_to_test_f1_gap | abs_cv_to_test_f1_gap | cv_to_test_gap_in_sd | selected |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| dummy | 0.000 | 0.000 | 0.000 | — | 0.000 | — | — | 0.000 | — | — | — | False |
| logistic_regression | 0.849 | 0.003 | 0.851 | — | 0.002 | — | — | 0.002 | — | — | — | False |
| random_forest | 0.835 | 0.003 | 0.835 | — | -0.000 | — | — | 0.000 | — | — | — | False |
| xgboost | 0.849 | 0.004 | 0.854 | 0.854 | 0.005 | -0.001 | 0.004 | 0.005 | 0.001 | 0.004 | 0.972 | True |
| lightgbm | 0.850 | 0.006 | 0.853 | — | 0.003 | — | — | 0.003 | — | — | — | False |
| catboost | 0.848 | 0.005 | 0.852 | — | 0.005 | — | — | 0.005 | — | — | — | False |
