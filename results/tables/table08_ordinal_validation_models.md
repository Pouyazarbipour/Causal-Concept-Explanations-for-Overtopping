| outcome   | covariate                 |   odds_ratio_per_sd |   or_lo |   or_hi |   boot_p |    n |   n_clusters | model                                          |
|:----------|:--------------------------|--------------------:|--------:|--------:|---------:|-----:|-------------:|:-----------------------------------------------|
| RF        | abs_resid                 |              1.172  |  0.9828 |  1.383  |     0.2  | 9333 |          162 | RF ~ |r|                                       |
| RF        | abs_within_geometry_resid |              1.061  |  0.9733 |  1.277  |     0.4  | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_geometry_mean_resid   |              1.18   |  0.9384 |  1.31   |     0.2  | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_resid                 |              1.142  |  0.9572 |  1.395  |     0.1  | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | tm10_default              |              2.058  |  1.363  |  3.126  |     0.05 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | deep_missing              |              1.175  |  0.9388 |  1.735  |     0.4  | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | ref_missing               |              1.018  |  0.8385 |  1.274  |     0.8  | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | log_scale                 |              0.8317 |  0.6465 |  0.9667 |     0.05 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| CF        | abs_resid                 |              1.081  |  0.8146 |  1.246  |     0.7  | 9333 |          162 | CF ~ |r| (record level, descriptive)           |
| CF        | abs_mean_resid            |              1.066  |  0.8399 |  1.296  |     0.4  | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
| CF        | log_n                     |              0.9468 |  0.572  |  1.394  |     0.4  | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
