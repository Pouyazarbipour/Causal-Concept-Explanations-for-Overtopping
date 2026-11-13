| outcome   | covariate                 |   odds_ratio_per_sd |   or_lo |   or_hi |   boot_p |    n |   n_clusters | model                                          |
|:----------|:--------------------------|--------------------:|--------:|--------:|---------:|-----:|-------------:|:-----------------------------------------------|
| RF        | abs_resid                 |              1.172  |  0.9899 |   1.367 |     0.08 | 9333 |          162 | RF ~ |r|                                       |
| RF        | abs_within_geometry_resid |              1.061  |  0.9298 |   1.226 |     0.41 | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_geometry_mean_resid   |              1.18   |  0.9229 |   1.394 |     0.16 | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_resid                 |              1.142  |  0.9734 |   1.394 |     0.1  | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | tm10_default              |              2.058  |  1.14   |   3.534 |     0.01 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | deep_missing              |              1.175  |  0.9139 |   1.631 |     0.26 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | ref_missing               |              1.018  |  0.7641 |   1.368 |     0.94 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | log_scale                 |              0.8317 |  0.6141 |   1.043 |     0.08 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| CF        | abs_resid                 |              1.081  |  0.8242 |   1.331 |     0.62 | 9333 |          162 | CF ~ |r| (record level, descriptive)           |
| CF        | abs_mean_resid            |              1.066  |  0.8736 |   1.255 |     0.61 | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
| CF        | log_n                     |              0.9468 |  0.6174 |   1.387 |     0.63 | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
