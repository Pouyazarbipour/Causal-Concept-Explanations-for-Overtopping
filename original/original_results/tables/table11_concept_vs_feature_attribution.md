| quantity                                                                                          |   value |
|:--------------------------------------------------------------------------------------------------|--------:|
| rank agreement of concept importance: CCW slot ablation vs grouped |SHAP| (Spearman, 10 concepts) | 0.06667 |
| attribution share per absent (null-state) concept: grouped TreeSHAP, XGBoost                      | 0.06448 |
| attribution share per absent (null-state) concept: CCW-Net                                        | 0.03406 |
| attribution share per absent (null-state) concept: unaligned wrapper                              | 0.04538 |
| attribution share per absent (null-state) concept: concept bottleneck                             | 0.09607 |
| attribution share per active concept: grouped TreeSHAP, XGBoost                                   | 0.1146  |
| attribution share per active concept: CCW-Net                                                     | 0.1419  |
| matched estimator, share of the changed concept: CCW-Net (0.10 = indiscriminate)                  | 0.1706  |
| matched estimator, share of the changed concept: unaligned wrapper                                | 0.1687  |
| matched estimator, share per held absent concept: CCW-Net (0.10 = indiscriminate)                 | 0.03753 |
| matched estimator, share per held absent concept: unaligned wrapper                               | 0.05237 |
| out-of-geometry RMSE: XGBoost (explained by SHAP)                                                 | 0.5015  |
| out-of-geometry RMSE: CCW-Net (explained by concept slots)                                        | 0.6008  |
