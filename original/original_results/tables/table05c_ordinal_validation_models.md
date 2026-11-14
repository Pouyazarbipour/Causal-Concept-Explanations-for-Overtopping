| outcome   | covariate                 |   odds_ratio_per_sd |   or_lo |   or_hi |   boot_p |    n |   n_clusters | model                                          |
|:----------|:--------------------------|--------------------:|--------:|--------:|---------:|-----:|-------------:|:-----------------------------------------------|
| RF        | abs_resid                 |              1.171  |  0.9832 |   1.345 |     0.1  | 9333 |          162 | RF ~ |r|                                       |
| RF        | abs_within_geometry_resid |              1.048  |  0.8997 |   1.21  |     0.51 | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_geometry_mean_resid   |              1.186  |  0.9431 |   1.399 |     0.15 | 9333 |          162 | RF ~ within + between geometry                 |
| RF        | abs_resid                 |              1.143  |  0.9676 |   1.389 |     0.12 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | tm10_default              |              2.058  |  1.141  |   3.534 |     0.01 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | deep_missing              |              1.176  |  0.9132 |   1.63  |     0.24 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | ref_missing               |              1.016  |  0.7614 |   1.365 |     0.96 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| RF        | log_scale                 |              0.8317 |  0.6141 |   1.044 |     0.08 | 9333 |          162 | RF ~ |r| + completeness indicators + scale     |
| CF        | abs_resid                 |              1.077  |  0.8171 |   1.34  |     0.68 | 9333 |          162 | CF ~ |r| (record level, descriptive)           |
| CF        | abs_mean_resid            |              1.07   |  0.8855 |   1.24  |     0.55 | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
| CF        | log_n                     |              0.9481 |  0.6178 |   1.393 |     0.62 | 2046 |          162 | CF ~ |geometry mean residual| (geometry level) |
