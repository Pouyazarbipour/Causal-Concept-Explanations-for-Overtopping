| quantity                                                                                                  |   value |
|:----------------------------------------------------------------------------------------------------------|--------:|
| rank agreement of concept importance: CCW slot ablation vs grouped |SHAP| (Spearman, 10 concepts)         | 0.0303  |
| attribution share per absent (null-state) concept: grouped TreeSHAP, XGBoost                              | 0.06448 |
| attribution share per absent (null-state) concept: CCW-Net                                                | 0.02468 |
| attribution share per absent (null-state) concept: unaligned wrapper                                      | 0.03804 |
| attribution share per absent (null-state) concept: concept bottleneck                                     | 0.09363 |
| attribution share per active concept: grouped TreeSHAP, XGBoost                                           | 0.1146  |
| attribution share per active concept: CCW-Net                                                             | 0.1488  |
| share routed through the concept whose label changed (matched estimator): CCW-Net (0.10 = indiscriminate) | 0.1345  |
| out-of-geometry RMSE: XGBoost (explained by SHAP)                                                         | 0.5015  |
| out-of-geometry RMSE: CCW-Net (explained by concept slots)                                                | 0.5903  |
