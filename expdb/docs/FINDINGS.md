# Findings — TS-SSL pilot, Avacha killer-whale events (NVIDIA GeForce RTX 5090 Laptop GPU, torch 2.11.0+cu128, 3.14.4, WSL2)

Protocol: 16 kHz, 80-mel/10 ms, 2-frame tokens (20 ms unless stated), 8-layer d=384 encoder (14.8M vanilla / 17.8M Gemma-style), 20k steps x 32 x 5 s synthesized mixtures from 45,429 lab clips; frozen weighted-layer linear probe, 3 recording-held-out folds (6/6/5), thresholds on 2 dev recordings, event F1 with onset collar tau, offset tolerance max(tau, 0.5 dur). **One seed per row** unless stated; fold spread is ~±0.05 F1.

## 1. Probe results (pooled over the 3 test folds)

Rows are the latest record per model name (re-probes replace earlier ones); P(A>B) in §3b is the share of positive bootstrap differences, not a p-value.

| id | model | F1 cls τ=.05 | τ=.2 | τ=1 (plateau D) | F1 any τ=.2 | τ=1 | onset p50 / p90 ms | screening R / P | tok ms | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 115 | ft_kmaves_modern | 0.491 | 0.678 | 0.765 | 0.726 | 0.847 | 31 / 168 | - / - | 20 | ✓ |
| 137 | ft_kmaves_modern_seed1 | 0.524 | 0.669 | 0.766 | 0.720 | 0.847 | 27 / 184 | - / - | 20 | ✓ |
| 174 | fte_R3noov_seed0 | 0.479 | 0.668 | 0.757 | 0.735 | 0.849 | 32 / 146 | - / - | 20 | ✓ |
| 145 | fto_d2v_modern_seed1 | 0.504 | 0.665 | 0.763 | 0.721 | 0.845 | 29 / 197 | - / - | 20 | ✓ |
| 176 | fte_R3noov_seed1 | 0.479 | 0.659 | 0.766 | 0.733 | 0.865 | 35 / 158 | - / - | 20 | ✓ |
| 139 | ft_d2v_modern_seed2 | 0.485 | 0.659 | 0.757 | 0.728 | 0.847 | 31 / 197 | - / - | 20 | ✓ |
| 113 | ft_d2v_modern | 0.484 | 0.657 | 0.756 | 0.719 | 0.840 | 30 / 161 | - / - | 20 | ✓ |
| 162 | fte_R2_seed1 | 0.540 | 0.656 | 0.744 | 0.709 | 0.839 | 25 / 142 | - / - | 20 | ✓ |
| 165 | fte_R2_seed2 | 0.495 | 0.655 | 0.758 | 0.706 | 0.827 | 28 / 165 | - / - | 20 | ✓ |
| 144 | fto_d2v_modern_seed0 | 0.520 | 0.653 | 0.738 | 0.716 | 0.836 | 29 / 162 | - / - | 20 | ✓ |
| 160 | fte_R3_seed0 | 0.470 | 0.653 | 0.749 | 0.719 | 0.855 | 31 / 158 | - / - | 20 | ✓ |
| 117 | ft_kmiter2_seed0 | 0.423 | 0.648 | 0.743 | 0.689 | 0.814 | 37 / 208 | - / - | 20 | ✓ |
| 149 | fto_aves_seed2 | 0.525 | 0.648 | 0.716 | 0.767 | 0.868 | 22 / 125 | - / - | 20 | ✓ |
| 146 | fto_d2v_modern_seed2 | 0.500 | 0.647 | 0.743 | 0.707 | 0.831 | 30 / 195 | - / - | 20 | ✓ |
| 166 | fte_R3_seed2 | 0.458 | 0.647 | 0.759 | 0.707 | 0.842 | 34 / 154 | - / - | 20 | ✓ |
| 136 | ft_kmaves_seed1 | 0.433 | 0.644 | 0.755 | 0.691 | 0.835 | 33 / 183 | - / - | 20 | ✓ |
| 159 | fte_R2_seed0 | 0.507 | 0.643 | 0.749 | 0.696 | 0.827 | 25 / 139 | - / - | 20 | ✓ |
| 164 | fte_R1_seed2 | 0.474 | 0.641 | 0.732 | 0.723 | 0.842 | 31 / 154 | - / - | 20 | ✓ |
| 143 | ft_aves_seed1 | 0.542 | 0.639 | 0.729 | 0.738 | 0.862 | 23 / 161 | - / - | 20 | ✓ |
| 148 | fto_aves_seed1 | 0.548 | 0.637 | 0.723 | 0.741 | 0.856 | 21 / 125 | - / - | 20 | ✓ |
| 147 | fto_aves_seed0 | 0.549 | 0.636 | 0.729 | 0.747 | 0.867 | 23 / 131 | - / - | 20 | ✓ |
| 158 | fte_R1_seed0 | 0.473 | 0.634 | 0.749 | 0.666 | 0.808 | 33 / 168 | - / - | 20 | ✓ |
| 135 | ft_d2v_modern_seed1 | 0.461 | 0.633 | 0.742 | 0.698 | 0.843 | 34 / 224 | - / - | 20 | ✓ |
| 141 | ft_kmaves_modern_seed2 | 0.477 | 0.632 | 0.732 | 0.697 | 0.818 | 28 / 161 | - / - | 20 | ✓ |
| 124 | ft_aves | 0.529 | 0.632 | 0.712 | 0.740 | 0.866 | 24 / 161 | - / - | 20 | ✓ |
| 121 | ft_kmiter2_seed2 | 0.430 | 0.625 | 0.722 | 0.673 | 0.811 | 37 / 195 | - / - | 20 | ✓ |
| 112 | ft_kmiter2_seed1 | 0.381 | 0.624 | 0.731 | 0.670 | 0.811 | 41 / 195 | - / - | 20 | ✓ |
| 163 | fte_R3_seed1 | 0.472 | 0.623 | 0.734 | 0.697 | 0.844 | 31 / 171 | - / - | 20 | ✓ |
| 114 | ft_kmaves | 0.433 | 0.619 | 0.717 | 0.699 | 0.827 | 33 / 187 | - / - | 20 | ✓ |
| 140 | ft_kmaves_seed2 | 0.444 | 0.619 | 0.714 | 0.679 | 0.814 | 36 / 222 | - / - | 20 | ✓ |
| 178 | fte_R3noov_seed2 | 0.446 | 0.617 | 0.714 | 0.707 | 0.829 | 33 / 155 | - / - | 20 | ✓ |
| 73 | ft_d2v_modern_s1 | 0.469 | 0.616 | 0.701 | 0.699 | 0.834 | 30 / 174 | - / - | 10 | ✓ |
| 161 | fte_R1_seed1 | 0.463 | 0.616 | 0.702 | 0.691 | 0.809 | 33 / 195 | - / - | 20 | ✓ |
| 125 | ft_jepa_modern_short_seed0 | 0.406 | 0.612 | 0.711 | 0.686 | 0.833 | 37 / 228 | - / - | 20 | ✓ |
| 134 | ft_rq_modern_seed2 | 0.429 | 0.597 | 0.692 | 0.670 | 0.811 | 31 / 224 | - / - | 20 | ✓ |
| 131 | ft_rq_modern_seed1 | 0.421 | 0.592 | 0.672 | 0.675 | 0.812 | 33 / 204 | - / - | 20 | ✓ |
| 128 | ft_rq_modern_seed0 | 0.427 | 0.580 | 0.671 | 0.674 | 0.810 | 34 / 250 | - / - | 20 | ✓ |
| 168 | fte_R3van_seed0 | 0.398 | 0.566 | 0.661 | 0.695 | 0.842 | 39 / 168 | - / - | 20 | ✓ |
| 92 | ft_kmoracle_seed0 | 0.323 | 0.566 | 0.664 | 0.706 | 0.833 | 44 / 209 | - / - | 20 | ✓ |
| 172 | fte_R3van_seed2 | 0.380 | 0.555 | 0.660 | 0.703 | 0.845 | 37 / 181 | - / - | 20 | ✓ |
| 170 | fte_R3van_seed1 | 0.362 | 0.533 | 0.638 | 0.670 | 0.821 | 36 / 181 | - / - | 20 | ✓ |
| 95 | ft_kmoracle_seed1 | 0.324 | 0.518 | 0.610 | 0.638 | 0.776 | 43 / 223 | - / - | 20 | ✓ |
| 47 | hybrid_modern_seed0 | 0.353 | 0.503 | 0.631 | 0.580 | 0.754 | 40 / 321 | 0.695 / 0.447 | 20 | ✓ |
| 63 | d2v_modern_seed2 | 0.334 | 0.502 | 0.605 | 0.586 | 0.763 | 40 / 305 | 0.674 / 0.507 | 20 | ✓ |
| 120 | aves | 0.348 | 0.497 | 0.598 | 0.619 | 0.778 | 34 / 274 | 0.734 / 0.587 | 20 | ✓ |
| 142 | ft_scratch_modern_seed2 | 0.305 | 0.475 | 0.567 | 0.614 | 0.775 | 43 / 282 | - / - | 20 | ✓ |
| 55 | kmaves_seed1 | 0.299 | 0.468 | 0.581 | 0.581 | 0.746 | 42 / 255 | 0.706 / 0.488 | 20 | ✓ |
| 101 | ft_kmrandseg_seed1 | 0.260 | 0.463 | 0.573 | 0.579 | 0.749 | 55 / 285 | - / - | 20 | ✓ |
| 116 | ft_scratch_modern | 0.301 | 0.459 | 0.573 | 0.592 | 0.766 | 40 / 289 | - / - | 20 | ✓ |
| 138 | ft_scratch_modern_seed1 | 0.281 | 0.457 | 0.574 | 0.604 | 0.773 | 42 / 287 | - / - | 20 | ✓ |
| 57 | kmaves_modern | 0.270 | 0.451 | 0.559 | 0.552 | 0.714 | 51 / 373 | 0.679 / 0.407 | 20 | ✓ |
| 61 | d2v_modern | 0.271 | 0.449 | 0.582 | 0.547 | 0.744 | 48 / 230 | 0.667 / 0.501 | 20 | ✓ |
| 58 | kmaves_modern_seed1 | 0.277 | 0.449 | 0.574 | 0.526 | 0.717 | 49 / 356 | 0.638 / 0.483 | 20 | ✓ |
| 62 | d2v_modern_seed1 | 0.272 | 0.448 | 0.556 | 0.585 | 0.761 | 46 / 227 | 0.677 / 0.557 | 20 | ✓ |
| 51 | hybrid_modern_seed2 | 0.269 | 0.445 | 0.568 | 0.546 | 0.734 | 47 / 388 | 0.690 / 0.431 | 20 | ✓ |
| 59 | kmaves_modern_seed2 | 0.258 | 0.444 | 0.560 | 0.540 | 0.721 | 56 / 292 | 0.646 / 0.528 | 20 | ✓ |
| 98 | ft_kmrandseg_seed0 | 0.243 | 0.430 | 0.549 | 0.576 | 0.743 | 56 / 308 | - / - | 20 | ✓ |
| 119 | kmiter2_seed2 | 0.246 | 0.429 | 0.560 | 0.522 | 0.701 | 54 / 321 | 0.685 / 0.498 | 20 | ✓ |
| 49 | hybrid_modern_seed1 | 0.272 | 0.429 | 0.546 | 0.516 | 0.692 | 46 / 371 | 0.623 / 0.424 | 20 | ✓ |
| 54 | kmaves | 0.257 | 0.426 | 0.579 | 0.533 | 0.752 | 56 / 338 | 0.682 / 0.544 | 20 | ✓ |
| 105 | kmiter2_seed0 | 0.243 | 0.421 | 0.561 | 0.528 | 0.736 | 53 / 324 | 0.698 / 0.425 | 20 | ✓ |
| 56 | kmaves_seed2 | 0.256 | 0.417 | 0.552 | 0.509 | 0.699 | 53 / 349 | 0.680 / 0.381 | 20 | ✓ |
| 111 | kmiter2_seed1 | 0.247 | 0.416 | 0.543 | 0.505 | 0.693 | 53 / 322 | 0.667 / 0.426 | 20 | ✓ |
| 123 | jepa_modern_short_seed0 | 0.215 | 0.415 | 0.537 | 0.530 | 0.696 | 60 / 270 | 0.644 / 0.386 | 20 | ✓ |
| 18 | kmseg | 0.224 | 0.412 | 0.556 | 0.535 | 0.741 | 64 / 341 | 0.700 / 0.514 | 20 | ✓ |
| 24 | kmboth | 0.220 | 0.410 | 0.536 | 0.542 | 0.727 | 58 / 318 | 0.698 / 0.372 | 20 | ✓ |
| 72 | d2v_modern_s1 | 0.270 | 0.406 | 0.535 | 0.532 | 0.720 | 43 / 318 | 0.679 / 0.479 | 10 | ✓ |
| 26 | kmseg_modern | 0.248 | 0.403 | 0.527 | 0.532 | 0.715 | 46 / 379 | 0.623 / 0.396 | 20 | ✓ |
| 109 | ft_jepa_modern_seed0 | 0.230 | 0.392 | 0.506 | 0.581 | 0.741 | 54 / 291 | - / - | 20 | ✓ |
| 130 | rq_modern_seed1 | 0.228 | 0.377 | 0.501 | 0.496 | 0.686 | 53 / 309 | 0.607 / 0.479 | 20 | ✓ |
| 91 | kmoracle_seed0 | 0.191 | 0.367 | 0.477 | 0.515 | 0.681 | 64 / 309 | 0.669 / 0.391 | 20 | ✓ |
| 133 | rq_modern_seed2 | 0.225 | 0.363 | 0.474 | 0.481 | 0.675 | 56 / 360 | 0.586 / 0.447 | 20 | ✓ |
| 94 | kmoracle_seed1 | 0.190 | 0.354 | 0.460 | 0.495 | 0.676 | 64 / 344 | 0.682 / 0.459 | 20 | ✓ |
| 21 | rq_s1 | 0.195 | 0.347 | 0.469 | 0.501 | 0.714 | 58 / 315 | 0.701 / 0.379 | 10 | ✓ |
| 97 | kmrandseg_seed0 | 0.170 | 0.326 | 0.450 | 0.468 | 0.681 | 70 / 360 | 0.630 / 0.427 | 20 | ✓ |
| 64 | ft_scratch | 0.159 | 0.311 | 0.460 | 0.458 | 0.689 | 77 / 336 | - / - | 20 | ✓ |
| 20 | rq | 0.179 | 0.310 | 0.418 | 0.473 | 0.679 | 67 / 360 | 0.662 / 0.368 | 20 | ✓ |
| 127 | rq_modern_seed0 | 0.170 | 0.305 | 0.397 | 0.436 | 0.611 | 58 / 344 | 0.579 / 0.428 | 20 | ✓ |
| 33 | rq_s1_half | 0.170 | 0.303 | 0.409 | 0.439 | 0.632 | 60 / 295 | 0.664 / 0.297 | 10 | ✓ |
| 60 | d2v | 0.139 | 0.282 | 0.372 | 0.415 | 0.570 | 67 / 316 | 0.612 / 0.253 | 20 | ✓ |
| 100 | kmrandseg_seed1 | 0.143 | 0.280 | 0.397 | 0.400 | 0.619 | 79 / 426 | 0.582 / 0.412 | 20 | ✓ |
| 29 | d2v_seed1 | 0.139 | 0.277 | 0.390 | 0.409 | 0.632 | 63 / 405 | 0.602 / 0.393 | 20 | ✓ |
| 108 | jepa_modern_seed0 | 0.125 | 0.276 | 0.389 | 0.417 | 0.618 | 76 / 333 | 0.571 / 0.412 | 20 | ✓ |
| 53 | mel | 0.156 | 0.268 | 0.391 | 0.351 | 0.564 | 56 / 318 | 0.545 / 0.383 | 20 | ✓ |
| 45 | d2v_seed2 | 0.138 | 0.265 | 0.375 | 0.396 | 0.604 | 74 / 417 | 0.649 / 0.353 | 20 | ✓ |
| 17 | kmmel | 0.134 | 0.261 | 0.361 | 0.460 | 0.661 | 62 / 325 | 0.615 / 0.401 | 20 | ✓ |
| 151 | ft_evsmoke | 0.162 | 0.234 | 0.320 | 0.453 | 0.629 | 32 / 401 | - / - | 20 | ✓ |
| 22 | rq_s4 | 0.100 | 0.225 | 0.346 | 0.403 | 0.622 | 86 / 396 | 0.573 / 0.335 | 40 | ✓ |
| 23 | scratch | 0.107 | 0.187 | 0.314 | 0.348 | 0.577 | 78 / 387 | - / - | 20 | ✓ |
| 12 | aves_8k | 0.051 | 0.178 | 0.435 | 0.245 | 0.665 | - / - | - / - | 20 | ✓ |
| 76 | beamng_d2v_probe | 0.022 | 0.128 | 0.305 | 0.212 | 0.525 | - / - | - / - | - | ✓ |
| 75 | beamng_scratch_probe | 0.026 | 0.116 | 0.303 | 0.210 | 0.550 | - / - | - / - | - | ✓ |
| 89 | beamng_d2v_probe_inductive | 0.019 | 0.116 | 0.330 | 0.211 | 0.584 | - / - | - / - | - | ✓ |

### 1b. Seed statistics (mean ± sd over seeds, families with ≥2 seeds)

| family | n | F1 cls τ=.2 | plateau τ=1 | F1 any τ=.2 | onset p50 ms | onset p90 ms |
|---|---|---|---|---|---|---|
| ft_kmaves_modern | 3 | 0.660 ± 0.024 | 0.754 ± 0.020 | 0.715 ± 0.015 | 29 ± 2 | 171 ± 12 |
| fto_d2v_modern | 3 | 0.655 ± 0.009 | 0.748 ± 0.013 | 0.715 ± 0.007 | 29 ± 1 | 185 ± 20 |
| fte_R2 | 3 | 0.651 ± 0.007 | 0.750 ± 0.007 | 0.704 ± 0.007 | 26 ± 2 | 149 ± 14 |
| ft_d2v_modern | 3 | 0.650 ± 0.014 | 0.752 ± 0.008 | 0.715 ± 0.015 | 32 ± 2 | 194 ± 32 |
| fte_R3noov | 3 | 0.648 ± 0.027 | 0.746 ± 0.028 | 0.725 ± 0.015 | 33 ± 2 | 153 ± 6 |
| fte_R3 | 3 | 0.641 ± 0.016 | 0.747 ± 0.013 | 0.708 ± 0.011 | 32 ± 2 | 161 ± 9 |
| fto_aves | 3 | 0.640 ± 0.007 | 0.723 ± 0.007 | 0.752 ± 0.014 | 22 ± 1 | 127 ± 3 |
| ft_aves | 2 | 0.635 ± 0.005 | 0.721 ± 0.012 | 0.739 ± 0.001 | 24 ± 1 | 161 ± 0 |
| ft_kmiter2 | 3 | 0.633 ± 0.014 | 0.732 ± 0.010 | 0.677 ± 0.010 | 38 ± 2 | 199 ± 8 |
| fte_R1 | 3 | 0.630 ± 0.013 | 0.728 ± 0.024 | 0.694 ± 0.029 | 32 ± 1 | 172 ± 21 |
| ft_kmaves | 3 | 0.627 ± 0.015 | 0.729 ± 0.023 | 0.690 ± 0.010 | 34 ± 2 | 197 ± 21 |
| ft_rq_modern | 3 | 0.590 ± 0.009 | 0.678 ± 0.012 | 0.673 ± 0.003 | 33 ± 2 | 226 ± 23 |
| fte_R3van | 3 | 0.551 ± 0.017 | 0.653 ± 0.013 | 0.689 ± 0.017 | 37 ± 2 | 177 ± 8 |
| ft_kmoracle | 2 | 0.542 ± 0.033 | 0.637 ± 0.039 | 0.672 ± 0.048 | 44 ± 1 | 216 ± 10 |
| d2v_modern | 3 | 0.466 ± 0.031 | 0.581 ± 0.024 | 0.573 ± 0.022 | 45 ± 4 | 254 ± 44 |
| ft_scratch_modern | 3 | 0.464 ± 0.010 | 0.571 ± 0.004 | 0.603 ± 0.011 | 42 ± 2 | 286 ± 4 |
| hybrid_modern | 3 | 0.459 ± 0.039 | 0.582 ± 0.044 | 0.547 ± 0.032 | 44 ± 4 | 360 ± 35 |
| kmaves_modern | 3 | 0.448 ± 0.003 | 0.564 ± 0.009 | 0.539 ± 0.013 | 52 ± 4 | 340 ± 43 |
| ft_kmrandseg | 2 | 0.447 ± 0.023 | 0.561 ± 0.017 | 0.578 ± 0.002 | 56 ± 1 | 296 ± 16 |
| kmaves | 3 | 0.437 ± 0.027 | 0.571 ± 0.016 | 0.541 ± 0.037 | 50 ± 7 | 314 ± 51 |
| kmiter2 | 3 | 0.422 ± 0.007 | 0.555 ± 0.010 | 0.518 ± 0.012 | 53 ± 1 | 322 ± 2 |
| kmoracle | 2 | 0.361 ± 0.009 | 0.468 ± 0.012 | 0.505 ± 0.014 | 64 ± 0 | 326 ± 25 |
| rq_modern | 3 | 0.348 ± 0.038 | 0.458 ± 0.054 | 0.471 ± 0.031 | 56 ± 3 | 338 ± 26 |
| kmrandseg | 2 | 0.303 ± 0.033 | 0.423 ± 0.037 | 0.434 ± 0.048 | 74 ± 6 | 393 ± 47 |
| d2v | 3 | 0.275 ± 0.009 | 0.379 ± 0.010 | 0.407 ± 0.010 | 68 ± 6 | 379 ± 55 |

## 2. Pretraining runs

| id | name | block | tok ms | seed | steps | final loss | unit acc | min | M params | valid |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | d2v | vanilla | 20 | 0 | 20000 | 0.364 | 0.000 | 36.6 | 14.84 | ✓ |
| 2 | d2v_modern | modern | 20 | 0 | 20000 | 0.411 | 0.000 | 35.4 | 17.81 | ✓ |
| 3 | kmaves | vanilla | 20 | 0 | 20000 | 1.969 | 0.456 | 31.6 | 14.84 | ✓ |
| 4 | kmaves_modern | modern | 20 | 0 | 20000 | 1.727 | 0.524 | 36.6 | 17.81 | ✓ |
| 6 | kmmel | vanilla | 20 | 0 | 20000 | 1.804 | 0.483 | 30.5 | 14.84 | ✓ |
| 7 | kmseg | vanilla | 20 | 0 | 20000 | 2.669 | 0.529 | 31.3 | 14.84 | ✓ |
| 8 | rq | vanilla | 20 | 0 | 20000 | 2.249 | 0.372 | 29.7 | 14.84 | ✓ |
| 9 | rq_s1 | vanilla | 10 | 0 | 20000 | 1.956 | 0.440 | 32.6 | 14.84 | ✓ |
| 10 | rq_s4 | vanilla | 40 | 0 | 20000 | 2.631 | 0.353 | 30.8 | 14.84 | ✓ |
| 25 | kmseg_modern | modern | 20 | 0 | 20000 | 2.589 | 0.543 | 42.12789070606232 | 17.852901 | ✓ |
| 27 | kmboth | vanilla | 20 | 0 | 20000 | 4.105 | 0.410 | 32.7 | 14.88 | ✓ |
| 28 | d2v_seed1 | vanilla | 20 | 1 | 20000 | 0.348 | 0.000 | 40.17094278335571 | 14.988288 | ✓ |
| 30 | d2v_modern_seed1 | modern | 20 | 1 | 20000 | 0.419 | 0.000 | 44.62190005381902 | 17.961856 | ✓ |
| 32 | rq_s1_half | vanilla | 10 | 0 | 10000 | 1.903 | 0.460 | 21.107576755682626 | 14.848228 | ✓ |
| 34 | kmaves_seed1 | vanilla | 20 | 1 | 20000 | 1.995 | 0.453 | 39.392771271864575 | 14.878948 | ✓ |
| 36 | kmaves_seed2 | vanilla | 20 | 2 | 20000 | 1.982 | 0.442 | 38.5199489792188 | 14.878948 | ✓ |
| 38 | d2v_modern_seed2 | modern | 20 | 2 | 20000 | 0.394 | 0.000 | 44.99428270260493 | 17.961856 | ✓ |
| 40 | kmaves_modern_seed1 | modern | 20 | 1 | 20000 | 1.802 | 0.482 | 43.1533900419871 | 17.852516 | ✓ |
| 42 | kmaves_modern_seed2 | modern | 20 | 2 | 20000 | 1.817 | 0.459 | 42.218267703056334 | 17.852516 | ✓ |
| 44 | d2v_seed2 | vanilla | 20 | 2 | 20000 | 0.329 | 0.000 | 41.14820258617401 | 14.988288 | ✓ |
| 46 | hybrid_modern_seed0 | modern | 20 | 0 | 20000 | 2.116 | 0.494 | 44.77866384585698 | 18.000356 | ✓ |
| 48 | hybrid_modern_seed1 | modern | 20 | 1 | 20000 | 2.161 | 0.480 | 45.3550387263298 | 18.000356 | ✓ |
| 50 | hybrid_modern_seed2 | modern | 20 | 2 | 20000 | 2.133 | 0.490 | 45.535595671335855 | 18.000356 | ✓ |
| 71 | d2v_modern_s1 | modern | 10 | 0 | 20000 | 0.305 | 0.000 | 71.76065961122512 | 17.931136 | ✓ |
| 90 | kmoracle_seed0 | vanilla | 20 | 0 | 20000 | 0.903 | 0.747 | 30.28215477069219 | 14.878948 | ✓ |
| 93 | kmoracle_seed1 | vanilla | 20 | 1 | 20000 | 1.094 | 0.678 | 36.77250121037165 | 14.878948 | ✓ |
| 96 | kmrandseg_seed0 | vanilla | 20 | 0 | 20000 | 1.410 | 0.580 | 31.092891399065653 | 14.878948 | ✓ |
| 99 | kmrandseg_seed1 | vanilla | 20 | 1 | 20000 | 1.540 | 0.531 | 31.648737092812855 | 14.878948 | ✓ |
| 104 | kmiter2_seed0 | vanilla | 20 | 0 | 20000 | 1.083 | 0.679 | 40.92459645668666 | 14.878948 | ✓ |
| 107 | jepa_modern_seed0 | modern | 20 | 0 | 20000 | 0.941 | 0.000 | 58.4211979786555 | 21.511168 | ✓ |
| 110 | kmiter2_seed1 | vanilla | 20 | 1 | 20000 | 1.069 | 0.702 | 40.953366760412855 | 14.878948 | ✓ |
| 118 | kmiter2_seed2 | vanilla | 20 | 2 | 20000 | 1.031 | 0.698 | 46.13772981564204 | 14.878948 | ✓ |
| 122 | jepa_modern_short_seed0 | modern | 20 | 0 | 20000 | 0.980 | 0.000 | 110.20033917029699 | 21.511168 | ✓ |
| 126 | rq_modern_seed0 | modern | 20 | 0 | 20000 | 2.079 | 0.387 | 51.32519759337107 | 17.852516 | ✓ |
| 129 | rq_modern_seed1 | modern | 20 | 1 | 20000 | 2.161 | 0.414 | 33.543648266792296 | 17.852516 | ✓ |
| 132 | rq_modern_seed2 | modern | 20 | 2 | 20000 | 2.102 | 0.381 | 33.49864672422409 | 17.852516 | ✓ |
| 150 | kmaves_evsmoke | modern | 20 | 0 | 120 | 5.784 | 0.034 | 0.23462810118993124 | 17.853671 | ✗ stopped at 100/120 |
| 152 | kmaves_modern_ev_seed0 | modern | 20 | 0 | 20000 | 2.127 | 0.496 | 32.214293746153515 | 17.853671 | ✓ |
| 153 | kmaves_modern_evonly_seed0 | modern | 20 | 0 | 20000 | 0.355 | 0.005 | 32.96881460348765 | 17.853671 | ✓ |
| 154 | kmaves_modern_ev_seed1 | modern | 20 | 1 | 20000 | 2.145 | 0.468 | 33.21840885480245 | 17.853671 | ✓ |
| 155 | kmaves_modern_evonly_seed1 | modern | 20 | 1 | 20000 | 0.348 | 0.005 | 32.84498516718546 | 17.853671 | ✓ |
| 156 | kmaves_modern_ev_seed2 | modern | 20 | 2 | 20000 | 2.088 | 0.491 | 32.92479198376338 | 17.853671 | ✓ |
| 157 | kmaves_modern_evonly_seed2 | modern | 20 | 2 | 20000 | 0.364 | 0.013 | 33.510382970174156 | 17.853671 | ✓ |
| 167 | kmaves_vanilla_evonly_seed0 | vanilla | 20 | 0 | 20000 | 0.451 | 0.001 | 30.60934319893519 | 14.880103 | ✓ |
| 169 | kmaves_vanilla_evonly_seed1 | vanilla | 20 | 1 | 20000 | 0.452 | 0.015 | 34.99278421401978 | 14.880103 | ✓ |
| 171 | kmaves_vanilla_evonly_seed2 | vanilla | 20 | 2 | 20000 | 0.457 | 0.014 | 35.63572838306427 | 14.880103 | ✓ |
| 173 | kmaves_modern_evonly_noov_seed0 | modern | 20 | 0 | 20000 | 0.283 | 0.008 | 33.9071551879247 | 17.853671 | ✓ |
| 175 | kmaves_modern_evonly_noov_seed1 | modern | 20 | 1 | 20000 | 0.332 | 0.007 | 36.603076155980425 | 17.853671 | ✓ |
| 177 | kmaves_modern_evonly_noov_seed2 | modern | 20 | 2 | 20000 | 0.278 | 0.003 | 43.535018976529436 | 17.853671 | ✓ |

Minutes were measured with other jobs sharing the GPU (manual probes, target prep); they are upper bounds, not benchmarks.

## 3. Unit quality (analysis_units.py)

- `kmmel_synth`: {'purity': 0.639, 'pnmi': 0.149, 'switch_at_boundary': 0.575, 'switch_inside': 0.423, 'switch_overall': 0.43, 'n_tokens': 500000}
- `kmaves_synth`: {'purity': 0.685, 'pnmi': 0.272, 'switch_at_boundary': 0.577, 'switch_inside': 0.454, 'switch_overall': 0.46, 'n_tokens': 498000}
- `kmaves_field`: {'purity': 0.905, 'pnmi': 0.44, 'switch_at_boundary': 0.574, 'switch_inside': 0.515, 'switch_overall': 0.516, 'n_tokens': 219822, 'call_token_frac': 0.117}
- `majority_share_field`: 0.883

## 3b. Paired bootstrap over recordings (2000 resamples of the 17 test recordings, seed-0 models)

| comparison | τ | F1 A [95% CI] | F1 B [95% CI] | A − B [95% CI] | P(A>B) |
|---|---|---|---|---|---|
| aves vs kmaves | 0.05 | 0.349 [0.281, 0.410] | 0.257 [0.223, 0.296] | +0.092 [+0.026, +0.161] | 0.998 |
| aves vs kmaves | 0.2 | 0.498 [0.416, 0.577] | 0.428 [0.373, 0.497] | +0.071 [-0.014, +0.148] | 0.954 |
| aves vs kmaves | 1.0 | 0.600 [0.535, 0.663] | 0.582 [0.524, 0.634] | +0.018 [-0.044, +0.075] | 0.714 |
| d2v_modern vs d2v | 0.05 | 0.269 [0.219, 0.321] | 0.141 [0.101, 0.187] | +0.128 [+0.062, +0.192] | 1.000 |
| d2v_modern vs d2v | 0.2 | 0.448 [0.391, 0.507] | 0.286 [0.209, 0.370] | +0.161 [+0.072, +0.259] | 1.000 |
| d2v_modern vs d2v | 1.0 | 0.581 [0.531, 0.627] | 0.376 [0.295, 0.467] | +0.205 [+0.112, +0.298] | 1.000 |
| d2v_modern vs kmaves | 0.05 | 0.269 [0.219, 0.321] | 0.257 [0.223, 0.296] | +0.012 [-0.042, +0.062] | 0.695 |
| d2v_modern vs kmaves | 0.2 | 0.448 [0.391, 0.507] | 0.428 [0.373, 0.497] | +0.020 [-0.045, +0.078] | 0.740 |
| d2v_modern vs kmaves | 1.0 | 0.581 [0.531, 0.627] | 0.582 [0.524, 0.634] | -0.001 [-0.081, +0.073] | 0.497 |
| ft_aves vs ft_d2v_modern | 0.05 | 0.533 [0.472, 0.594] | 0.484 [0.428, 0.534] | +0.050 [-0.005, +0.113] | 0.958 |
| ft_aves vs ft_d2v_modern | 0.2 | 0.639 [0.566, 0.710] | 0.649 [0.583, 0.717] | -0.011 [-0.052, +0.040] | 0.318 |
| ft_aves vs ft_d2v_modern | 1.0 | 0.715 [0.659, 0.775] | 0.751 [0.704, 0.800] | -0.036 [-0.073, +0.010] | 0.058 |
| ft_d2v_modern vs ft_kmaves | 0.05 | 0.484 [0.428, 0.534] | 0.433 [0.367, 0.495] | +0.050 [-0.003, +0.098] | 0.972 |
| ft_d2v_modern vs ft_kmaves | 0.2 | 0.649 [0.583, 0.717] | 0.621 [0.542, 0.702] | +0.028 [-0.023, +0.076] | 0.856 |
| ft_d2v_modern vs ft_kmaves | 1.0 | 0.751 [0.704, 0.800] | 0.716 [0.660, 0.774] | +0.035 [-0.003, +0.070] | 0.963 |
| ft_d2v_modern vs ft_scratch_modern | 0.05 | 0.484 [0.428, 0.534] | 0.302 [0.253, 0.352] | +0.182 [+0.130, +0.232] | 1.000 |
| ft_d2v_modern vs ft_scratch_modern | 0.2 | 0.649 [0.583, 0.717] | 0.462 [0.388, 0.538] | +0.187 [+0.126, +0.254] | 1.000 |
| ft_d2v_modern vs ft_scratch_modern | 1.0 | 0.751 [0.704, 0.800] | 0.570 [0.494, 0.644] | +0.181 [+0.127, +0.233] | 1.000 |
| ft_kmaves_modern vs ft_aves | 0.05 | 0.491 [0.428, 0.553] | 0.531 [0.471, 0.590] | -0.040 [-0.092, +0.009] | 0.064 |
| ft_kmaves_modern vs ft_aves | 0.2 | 0.680 [0.593, 0.763] | 0.633 [0.564, 0.702] | +0.046 [-0.005, +0.095] | 0.965 |
| ft_kmaves_modern vs ft_aves | 1.0 | 0.765 [0.703, 0.824] | 0.714 [0.657, 0.776] | +0.052 [+0.002, +0.098] | 0.977 |
| ft_kmaves_modern vs ft_kmaves | 0.05 | 0.491 [0.428, 0.553] | 0.433 [0.367, 0.495] | +0.058 [+0.033, +0.077] | 1.000 |
| ft_kmaves_modern vs ft_kmaves | 0.2 | 0.680 [0.593, 0.763] | 0.621 [0.542, 0.701] | +0.059 [+0.018, +0.094] | 0.997 |
| ft_kmaves_modern vs ft_kmaves | 1.0 | 0.765 [0.703, 0.824] | 0.717 [0.660, 0.774] | +0.048 [+0.017, +0.079] | 0.998 |
| ft_kmaves vs ft_kmoracle_seed0 | 0.05 | 0.433 [0.367, 0.495] | 0.323 [0.273, 0.374] | +0.110 [+0.049, +0.167] | 1.000 |
| ft_kmaves vs ft_kmoracle_seed0 | 0.2 | 0.621 [0.542, 0.702] | 0.567 [0.496, 0.645] | +0.055 [-0.004, +0.105] | 0.968 |
| ft_kmaves vs ft_kmoracle_seed0 | 1.0 | 0.716 [0.660, 0.774] | 0.665 [0.602, 0.734] | +0.052 [-0.002, +0.097] | 0.971 |
| ft_kmaves vs kmaves | 0.05 | 0.433 [0.367, 0.495] | 0.257 [0.223, 0.296] | +0.176 [+0.117, +0.229] | 1.000 |
| ft_kmaves vs kmaves | 0.2 | 0.621 [0.542, 0.702] | 0.428 [0.373, 0.497] | +0.194 [+0.138, +0.251] | 1.000 |
| ft_kmaves vs kmaves | 1.0 | 0.716 [0.660, 0.774] | 0.582 [0.524, 0.634] | +0.134 [+0.072, +0.191] | 1.000 |
| ft_kmoracle_seed0 vs ft_kmrandseg_seed0 | 0.05 | 0.323 [0.273, 0.374] | 0.242 [0.198, 0.288] | +0.081 [+0.017, +0.154] | 0.996 |
| ft_kmoracle_seed0 vs ft_kmrandseg_seed0 | 0.2 | 0.567 [0.496, 0.645] | 0.430 [0.358, 0.516] | +0.137 [+0.079, +0.206] | 1.000 |
| ft_kmoracle_seed0 vs ft_kmrandseg_seed0 | 1.0 | 0.665 [0.602, 0.734] | 0.550 [0.470, 0.640] | +0.115 [+0.058, +0.166] | 1.000 |
| kmaves vs aves | 0.05 | 0.257 [0.223, 0.296] | 0.047 [0.025, 0.069] | +0.210 [+0.162, +0.262] | 1.000 |
| kmaves vs aves | 0.2 | 0.428 [0.373, 0.497] | 0.172 [0.121, 0.223] | +0.256 [+0.178, +0.343] | 1.000 |
| kmaves vs aves | 1.0 | 0.582 [0.524, 0.634] | 0.445 [0.369, 0.517] | +0.137 [+0.058, +0.225] | 1.000 |

## 3c. Main criterion: class-agnostic recall at fixed false-alarm budgets (thresholds from dev per fold; test pooled; τ=0.5)

| model | R @≤50/h (test FAR) | R @≤100/h | R @≤300/h | K12 @300 | K21 @300 |
|---|---|---|---|---|---|
| ft_d2v_modern | 0.715 (67) | 0.706 (79) | 0.723 (107) | 0.605 | 0.857 |
| ft_d2v_modern_seed1 | 0.695 (66) | 0.700 (93) | 0.728 (156) | 0.605 | 0.857 |
| ft_d2v_modern_seed2 | 0.739 (56) | 0.721 (70) | 0.732 (79) | 0.5 | 1.0 |
| ft_kmaves | 0.695 (61) | 0.695 (68) | 0.715 (85) | 0.5 | 0.857 |
| ft_kmaves_seed1 | 0.697 (65) | 0.697 (79) | 0.710 (114) | 0.5 | 0.857 |
| ft_kmaves_seed2 | 0.698 (68) | 0.713 (92) | 0.731 (172) | 0.447 | 0.857 |
| ft_kmaves_modern | 0.698 (62) | 0.698 (63) | 0.708 (78) | 0.421 | 0.857 |
| ft_kmaves_modern_seed1 | 0.728 (58) | 0.728 (58) | 0.739 (158) | 0.447 | 0.857 |
| ft_kmaves_modern_seed2 | 0.669 (64) | 0.679 (70) | 0.697 (89) | 0.421 | 0.857 |
| ft_scratch_modern | 0.644 (116) | 0.644 (116) | 0.653 (216) | 0.421 | 1.0 |
| ft_scratch_modern_seed1 | 0.649 (108) | 0.646 (117) | 0.666 (226) | 0.421 | 1.0 |
| ft_scratch_modern_seed2 | 0.651 (124) | 0.651 (124) | 0.633 (170) | 0.5 | 0.857 |
| ft_aves | 0.755 (71) | 0.752 (99) | 0.759 (123) | 0.579 | 0.857 |
| ft_aves_seed1 | 0.732 (67) | 0.732 (70) | 0.747 (83) | 0.553 | 0.857 |
| ft_kmiter2_seed0 | 0.664 (79) | 0.659 (82) | 0.675 (100) | 0.421 | 0.857 |
| aves | 0.729 (142) | 0.721 (157) | 0.729 (203) | 0.553 | 1.0 |

## 4. Hypotheses → verdicts

**H1** [rejected-with-modern-block (2 seeds)] Discrete HuBERT-style targets beat continuous (data2vec/JEPA) latent targets for boundary-precise event detection at equal backbone and compute  
  contract: Avacha probe protocol, 15M vanilla encoder, 20 ms tokens, 20k steps, 1 seed; expected: discrete > continuous at tau<=0.2
  - 2026-10-18 **inconclusive** (runs [13, 17, 20]): vanilla block: d2v 0.282 vs kmmel 0.261 vs rq 0.310 F1cls@0.2 - inside fold spread, 1 seed; any@0.2 d2v 0.415 below discrete 0.46-0.47
  - 2026-10-18 **rejected-with-modern-block** (runs [14, 16]): with the Gemma-style block data2vec (no teacher) 0.449 equals kmaves 0.451; second seed queued (d2v_seed1, d2v_modern_seed1)
  - 2026-10-18 **rejected-with-modern-block** (runs [14, 13]): seed 1 reproduces: d2v 0.277 (seed0 0.282), d2v+Gemma 0.448 (seed0 0.449), best any@0.2 0.585 and onset p50/p90 46/227 ms of all runs. Continuous targets with the stabilised block match teacher-distilled discrete units without any teacher
  - 2026-10-18 **rejected (3 seeds)** (runs []): 3 seeds each: d2v+Gemma 0.466±0.031 vs kmaves 0.437±0.027 F1cls@0.2; plateau 0.581±0.024 vs 0.571±0.016; onset p50 45±4 vs 50±7 ms; p90 254±44 vs 314±51 ms. Continuous targets with the stabilised block are at least equal to teacher-distilled discrete units, without a teacher

**H2** [confirmed] The teacher (unit source) sets the F1 plateau D (semantics) but not the boundary error tau50  
  contract: same encoder, targets kmmel / rq / d2v / kmaves; expected: D ordered by PNMI of units; onset p50 equal
  - 2026-10-18 **confirmed** (runs [13, 15, 17, 20]): onset p50 56-67 ms for all four; plateau F1cls@1.0 0.36 (kmmel, PNMI .149) / 0.37 (d2v) / 0.42 (rq) / 0.58 (kmaves, PNMI .272)
  - 2026-10-18 **confirmed-after-audit** (runs []): External review flagged that timing errors were computed on different matched sets per model. Re-audit (audit_metric.py): on the COMMON set of 125 events matched by all 7 models (class-aware, tau=1) onset p50 is mel 49 / kmaves 49 / kmaves_modern 42 / d2v 62 / d2v_modern 48 ms; only FT moves it (25-28 ms). Class-aware vs class-agnostic timing differ by <5 ms. The plateau is NOT reached at tau=1 s: F1cls rises by 0.03-0.12 up to tau=10 s (kmaves 0.579->0.627, d2v_modern 0.582->0.651); D must be estimated at tau>=5 s. Rankings are unchanged. The decomposition is kept as an empirical analysis scheme, not as proven independent axes
  - 2026-10-18 **corrected** (runs []): Field unit statistics with fixed timing: km(AVES) PNMI 0.44 (was 0.286), P(switch|boundary) 0.574 vs 0.515 inside (was 0.483/0.517): units DO switch somewhat more often at event boundaries. The flicker rate (0.52) stands

**H3** [split (2 seeds for continuous)] Modernising the block (RMSNorm, GeGLU, RoPE, QK-norm, local/global 3:1 attention, gated conv) gives only small gains vs the target design  
  contract: kmaves and d2v targets, vanilla vs modern, 1 seed; expected: <= +0.03 F1cls@0.2
  - 2026-10-18 **confirmed-for-discrete** (runs [15, 16]): kmaves 0.426 -> 0.451 (+0.025); unit prediction acc 0.456 -> 0.524
  - 2026-10-18 **rejected-for-continuous** (runs [13, 14]): d2v 0.282 -> 0.449 (+0.17); onset p90 316 -> 230 ms; recall-mode precision 0.25 -> 0.50. Continuous targets need the stabilised block
  - 2026-10-18 **rejected-for-continuous** (runs []): seed 1: +0.171 again (0.277 -> 0.448); seed-to-seed spread 0.005
  - 2026-10-18 **confirmed-for-discrete (3 seeds)** (runs []): kmaves modern 0.448±0.004 vs vanilla 0.437±0.027: +0.011, inside noise; d2v 3 seeds 0.275±0.009 vs modern 0.466±0.031: +0.19

**H4** [confirmed when compute-matched] Token stride: 10 ms ~ 20 ms (model-limited, onset p50 ~60 ms >> g/2) and 40 ms clearly worse  
  contract: BEST-RQ targets, vanilla block, 20k steps (not compute-matched); expected: F1(10)=F1(20)>F1(40)
  - 2026-10-18 **partially-confirmed** (runs [20, 21, 22]): F1cls@0.2 0.347 / 0.310 / 0.225 for 10/20/40 ms; onset p50 58/67/86 ms. 40 ms worse as predicted; 10 ms better than 20 ms on both plateau and tail, contrary to prediction - confounded by 2x tokens per sequence. Compute-matched run rq_s1_half queued
  - 2026-10-18 **confirmed** (runs [20, 21]): compute-matched 10 ms (10k steps) 0.303 vs 20 ms (20k) 0.310, plateau 0.409 vs 0.418, onset p50 60 vs 67 ms: at equal compute 10 ms ~ 20 ms; the earlier +0.037 was extra token-updates, not resolution. Model-limited regime confirmed

**H5** [rejected] Variable-length segment units (cosine change-points of teacher features) plus a boundary head shift tau50 left  
  contract: kmseg targets from AVES L9, mean segment 3.9 tokens, vanilla block; expected: onset p50 < 56 ms, F1cls@0.05 up
  - 2026-10-18 **rejected** (runs [15, 18]): onset p50 64 vs 56 ms, F1cls@0.2 0.412 vs 0.426, plateau 0.556 vs 0.579; recall-mode recall 0.70 vs 0.68. Cause (analysis_units): segment starts hit event boundaries with P=0.39 vs 0.25 inside - teacher change-points are not event boundaries. Next: boundary signal from spectral flux / student self-iteration, gate P(start|boundary)>=0.6 before training
  - 2026-10-18 **rejected** (runs [16, 26]): kmseg + Gemma block 0.403 vs kmaves + Gemma block 0.451; onset p50 46 ms (best median so far) but plateau 0.527 vs 0.559 - segment units trade semantics for a small median gain, tail p90 379 ms unchanged
  - 2026-10-18 **gate-failed** (runs []): teacher-free spectral-flux change points at start rates 0.15/0.25 hit event boundaries with P=0.33/0.44 (inside 0.14/0.24) - below the 0.6 gate, so no new segment-unit run was launched. Boundary-aligned segmentation needs a learned signal (student self-iteration on its own onset predictions), not a hand-crafted one

**H6** [REJECTED (timing artefact)] A small student distilled into discrete 20 ms units localises events better than its frozen teacher  
  contract: AVES-bio frozen linear probe vs kmaves student (15M); expected: student tau50 < teacher tau50 at equal or higher D
  - 2026-10-18 **confirmed** (runs [11, 15]): onset p50 211 ms (teacher) vs 56 ms (student); F1cls@0.2 0.171 vs 0.426; F1any@1.0 0.667 vs 0.752
  - 2026-10-18 **confirmed (3 seeds)** (runs []): kmaves 3 seeds: onset p50 50±7 ms vs frozen AVES 211 ms
  - 2026-10-18 **under-reaudit** (runs []): Review found that frozen-AVES features were concatenated across 10 s chunks with 499 tokens per chunk on a 500-token grid: a 20 ms drift per chunk (6149 vs 6161 tokens on a 123 s recording, up to 1.7 s on the longest). The 211 ms teacher onset median is therefore contaminated. probe.py now pads each chunk to the exact grid; re-probe running. The student numbers (mel front-end) are unaffected; kmaves pretraining targets were correctly truncated per 5 s sequence
  - 2026-10-18 **rejected** (runs []): With chunk features padded to the exact 20 ms grid, frozen AVES-bio probe: F1cls@0.2 0.497 (was 0.171), plateau 0.598, any@0.2 0.619, onset p50/p90 34/274 ms (was 211/644). On the common matched set (142 events): aves 36 ms vs mel 51, kmaves 53, d2v_modern 47. The frozen teacher localises BETTER than every student probe. The earlier 4x claim was entirely the 499-vs-500 tokens drift. Student lines recover 88% (km(AVES) 0.437) and 94% (d2v+Gemma 0.466) of the teacher's frozen F1 at 1/6 of the parameters; that is the honest statement

**H7** [confirmed] Ranking of models under the screening operating point (max recall at precision>=0.5, tau 0.5) matches the F1@0.2 ranking, i.e. D and tau50 are separable axes  
  contract: 6 models, thresholds selected on 2 dev recordings; expected: same ranking
  - 2026-10-18 **confirmed** (runs [11, 13, 15, 17, 19, 20]): kmaves best on both (recall 0.68 / precision 0.54 / 287 FA/h); dev-selected precision floor did not transfer to test for 5 of 6 models - select by FAR budget next

**H8** [rejected] Two codebooks (mel units + AVES units, MT4SSL-style) raise the plateau D over single AVES units  
  contract: kmboth vs kmaves, vanilla block, 1 seed; expected: D up, tau50 unchanged
  - 2026-10-18 **rejected** (runs [15, 24]): F1cls@0.2 0.410 vs 0.426, plateau 0.536 vs 0.579, onset p50 58 vs 56 ms; the low-PNMI mel codebook dilutes the objective rather than adding information

**H9** [rejected (3 seeds): not additive] The two winning lines are complementary: discrete AVES units (CE) + continuous EMA targets (MSE) on the Gemma block beat either alone  
  contract: hybrid variant, modern block, 20k steps, seeds 0-1; expected: F1cls@0.2 > 0.47 and onset p90 < 250 ms
  - 2026-10-18 **inconclusive** (runs []): hybrid (CE AVES units + d2v MSE, Gemma block): seed0 0.503 / seed1 0.429 F1cls@0.2 (mean 0.466±0.052) = d2v_modern 0.466±0.031 and kmaves_modern 0.448±0.004; seed0 is the best single run of the study (plateau 0.631, onset p50 40 ms) but the spread is twice that of the single-objective lines. Not additive on average; third seed queued
  - 2026-10-18 **rejected** (runs []): 3 seeds 0.459±0.039 (0.503/0.429/0.445) vs d2v_modern 0.466±0.031 and kmaves_modern 0.448±0.003; onset p50 44±4 ms equal, p90 360±35 worse than d2v_modern 254±44. Summing the two objectives neither adds plateau nor tightens the tail; it inherits the variance of the EMA line

**H10** [confirmed with corrected attribution] Full fine-tuning preserves the ordering found with frozen probes and closes most of the gap to event boundaries  
  contract: finetune.py: 1500 steps, bs 8 x 10 s chunks, enc lr 5e-5 / head lr 1e-3, same folds/dev/decoder/scorer; seed 0; expected: FT > probe for all; ordering d2v_modern >= kmaves >> scratch
  - 2026-10-18 **confirmed** (runs []): F1cls@0.2: scratch 0.311, kmaves 0.620, d2v_modern 0.650 (probe: 0.187 / 0.426 / 0.449); onset p50 77 / 33 / 31 ms, p90 336 / 180 / 162 ms; any@0.2 0.458 / 0.700 / 0.711. Pretraining gain survives FT (+0.31 over scratch). For reference the ICLR paper's full FT of AVES-bio under its own protocol was 41.2
  - 2026-10-18 **confirmed** (runs []): 3 seeds: ft_d2v_modern 0.647±0.012, ft_kmaves 0.628±0.014 F1cls@0.2; plateau 0.750±0.007 vs 0.728±0.023; onset p50 32±2 vs 34±2 ms, p90 194±31 vs 195±23. FT halves seed variance vs probes; the no-teacher line stays ahead by ~0.02 (P~0.86 bootstrap)
  - 2026-10-18 **re-audit** (runs []): Review: ft_scratch used the vanilla block while ft_d2v_modern uses the modern block, so +0.31 mixes pretraining and architecture. ft_scratch_modern (same block, same budget) running; the vanilla-vs-vanilla pair km(AVES) 0.628 vs scratch 0.311 already isolates pretraining for that line
  - 2026-10-18 **corrected** (runs []): Matched-block scratch (modern, same 1500-step budget): F1cls@0.2 0.463, plateau 0.571, onset p50/p90 37/254 ms. Attribution of the earlier +0.31: block +0.15 (0.311->0.463), pretraining +0.18 (0.463->0.647). Pretraining still helps substantially under a fair comparison, but by half of what was claimed
  - 2026-10-18 **teacher-FT-added** (runs []): AVES-bio full FT under our protocol (lr 2e-5, same folds/decoder/scorer, seed 0): F1cls@0.2 0.637, plateau 0.714, any@0.2 0.743, onset p50/p90 24/171 ms. Versus our d2v+modern FT 0.647±0.012 / 0.750±0.007 / 0.712 / 32/194. The 94M teacher fine-tuned is equal on class-aware F1, better on class-agnostic detection and localisation, lower on the plateau. Our 18M model is therefore a parameter-efficient match, not a better model
  - 2026-10-18 **bootstrap** (runs []): paired bootstrap ft_aves - ft_d2v_modern: tau .05 +0.050 [-0.005,+0.113]; tau .2 -0.011 [-0.052,+0.040]; tau 1 -0.036 [-0.073,+0.010]. ft_d2v_modern - ft_scratch_modern: +0.187 [+0.126,+0.254] at tau .2, P>0.999. Wording changed from 'parity' to 'comparable observed F1'
  - 2026-10-18 **teacher-FT 2 seeds** (runs []): ft_aves seeds 0/1: F1cls@0.2 0.637/0.640 (0.638±0.002), plateau 0.719±0.007, any@0.2 0.747±0.005, onset p50 24/22 ms. vs ft_d2v_modern 0.647±0.012 / 0.750±0.007 / 0.712±0.014 / 32±2 ms. Comparable class-aware F1; teacher better on class-agnostic detection and onset precision; student better on plateau; 5.3x fewer parameters

**H11** [rejected (1 seed, confounded)] After full FT the onset error (p50 ~32 ms) approaches the 20 ms grid ceiling (g/2 = 10 ms), so a 10 ms token stride now improves boundaries where it did not for frozen probes  
  contract: d2v + Gemma block pretrained at 10 ms tokens (20k steps), full FT, vs ft_d2v_modern at 20 ms; expected: onset p50 < 28 ms, F1cls@0.05 up
  - 2026-10-18 **rejected** (runs []): d2v+Gemma at 10 ms: probe 0.406 (20 ms: 0.466±0.031), FT 0.616 (0.647±0.012), onset p50 30 ms (32±2), F1cls@0.05 0.468 (0.486). No gain even under FT. Confound: the local-attention window is 64 tokens, i.e. 0.64 s at 10 ms vs 1.28 s at 20 ms, so context shrank with the stride; a window-matched rerun (128 tokens) would be needed before calling the grid irrelevant. Decision: keep 20 ms as the default

**H12** [weakly supported (inductive, 1 seed)] The framework (same encoder, data2vec objective, decoder, collar scorer) transfers to multichannel telemetry: SSL pretraining on unlabeled laps improves boundary-precise event detection in BeamNG  
  contract: 63 laps, 2.07 h, 15 ch @30 Hz, 10 overlapping event classes, lap-held-out 3 folds, multi-label per-class decoding; d2v+Gemma 12k steps; seed 0; expected: FT(pretrained) > FT(scratch) at tau<=0.2
  - 2026-10-18 **confirmed** (runs []): F1cls@0.2: scratch probe 0.116, d2v probe 0.128, scratch FT 0.566, d2v FT 0.669 (+0.10); any@0.2 0.586 -> 0.682; F1cls@1.0 0.716 -> 0.750. Frozen probes are weak in this domain (linear head on 256-d features cannot separate 10 overlapping classes); FT is the relevant regime. Per-class recall@0.2 at FT: braking 0.95, abs 0.82, tcs 0.10, esc 0.13 (rare classes 29/86 events). Seeds queued
  - 2026-10-18 **confirmed** (runs []): 3 seeds, full FT: pretrained 0.653±0.016 vs scratch 0.558±0.008 F1cls@0.2 (+0.095); plateau 0.737±0.011 vs 0.708±0.008; any@0.2 0.669±0.014 vs 0.580±0.006. At tau=33 ms (one frame) both are 0.18: the gain is in plateau and mid-collar detection, not at single-frame precision
  - 2026-10-18 **under-reaudit** (runs []): Review: pretraining and normalisation used all laps incl. test laps; braking/full_throttle/corner/gear_shift are thresholds of input channels. Protocol made inductive (per-fold pretraining and normalisation on non-test laps), scratch now matched-block, headline moved to non-trivial classes (abs/tcs/esc/impact/airtime); rerun queued
  - 2026-10-18 **downgraded** (runs []): Inductive protocol (per-fold pretraining and normalisation on non-test laps, scratch on the same modern block): all 10 classes F1cls@0.2 0.647 (pretrained) vs 0.560 (scratch), +0.087. NON-TRIVIAL classes (abs/tcs/esc/impact/airtime): 0.560 vs 0.534, +0.026, single seed, inside expected noise; per-class recall abs 0.79 vs 0.80, impact 0.16 vs 0.26 (worse), tcs/esc ~0.1 both. Most of the earlier gain sat on classes defined by thresholds of input channels. Frozen probes 0.116 / 0.098. Honest claim: SSL helps on telemetry events that are deterministic functions of inputs; for the non-trivial flags the evidence is not there yet

**H13** [resolved: outcome 4 (alignment > random, but coarse units < frame units)] Boundary-aligned units are causally useful: units merged over TRUE event segments (oracle, synthetic labels known by construction) improve downstream localisation/plateau over frame units, while units merged over random segments with the same length statistics do not  
  contract: vanilla block, 20k steps, same probe/FT protocol, 2 seeds per arm; frame-unit arm = kmaves (3 seeds). Diagnostic: uses construction labels of the synthetic corpus, never test labels; expected: oracle > kmaves on tau50 and/or plateau; randseg <= kmaves
  - 2026-10-18 **outcome-4** (runs []): 2 seeds/arm, vanilla block, same compute. Probe F1cls@0.2: frame kmaves 0.437±0.027 | oracle 0.361±0.009 | randseg 0.303±0.033; onset p50 50 | 64 | 74 ms. Full FT: 0.628±0.014 | 0.542±0.033 | 0.447±0.023; onset p50 34 | 44 | 56 ms. Paired bootstrap (seed 0, FT, tau .2): oracle - randseg +0.137 [+0.079,+0.206] P>0.999; frame - oracle +0.055 [-0.004,+0.105] P=0.968. Reading (pre-registered row 4): event-alignment recovers a large part of the loss caused by coarsening units to event length, but one-unit-per-event targets lose more than alignment returns, on plateau AND on boundaries. Causal claim limited to targets built by averaging AVES units over segments. Implication: an architecture that REPLACES fine units by event-level units is not supported; any boundary mechanism must be additive to fine units (hierarchy or auxiliary loss), and the simple competitor (frame units + auxiliary boundary loss) is now the first thing to test, not a learned segmenter
  - 2026-10-18 **wording** (runs []): Clarification requested by the supervisor: the seed-mean gap frame-minus-oracle at FT is 0.086 (0.628 vs 0.542); the paired bootstrap +0.055 [-0.004,+0.105] is the SEED-0 pair only and its CI includes zero. Sufficient to stop investing in this construction; not a categorical ban on event-level units

**H14** [cancelled (scope decision 2026-10-18): only BEST-RQ-on-modern and HuBERT-iter-2 kept; TS-JEPA deferred] Fixed comparison pool on one block and protocol: HuBERT-km(teacher), data2vec, BEST-RQ, TS-JEPA-style (predictor + block masking), supervised scratch; 3 seeds, probe and FT  
  contract: modern block, 20k steps, 3 seeds; expected: ordering stable across probe and FT

**H15** [rejected at screening (block masks)] A JEPA formulation (separate 2-layer predictor, EMA target encoder, context encoder without masked tokens, block masks ~0.8 s) on the modern block beats modern data2vec at equal data and budget  
  contract: modern block, 20k steps x 32 x 5 s, seed 0 screening; compare to d2v_modern 0.466±0.031 probe / 0.647±0.012 FT and ft_scratch_modern 0.463; expected: jepa >= d2v_modern
  - 2026-10-18 **rejected** (runs []): Probe F1cls@0.2 0.276 (plateau 0.389, onset p50 76 ms) = vanilla data2vec level; FT 0.392, BELOW scratch-modern FT 0.463: block-mask JEPA pretraining harms initialisation. Hypothesised cause: 0.8 s blocks remove whole calls, which are not predictable from surrounding noise, so the predictor's task degenerates; short-span control (p .08, span 10, same predictor) queued to separate masking granularity from the predictor design
  - 2026-10-18 **implementation-audit** (runs []): Rebuilt the run from the checkpoint and executed one step: predictor (3.7M, 27 tensors) is in the optimizer and receives gradients (norm 0.14 vs encoder 0.07); target encoder has no gradients; EMA moves the teacher after each step; masked context positions are exactly zero and unmasked are unchanged; targets instance-normed (std 1.00); loss computed on masked positions only (unmasked MSE 2.87 untrained, by design). Measured mask fraction 0.36 with mean block 45 tokens (0.9 s), not the ~45% I had assumed. Mechanically correct. Differences from our data2vec are THREE at once: (1) separate predictor with learned mask token instead of a linear head on the student output; (2) context encoder sees zeros at masked positions instead of a learned mask embedding; (3) target = last layer only instead of mean of top-4 layers. The screening therefore cannot attribute the loss to any one of them; the short-span control isolates only masking granularity. No further JEPA runs after that control; next hypothesis to be chosen jointly. Saved args lack jepa_p/jepa_span (flags added after the run); defaults 0.015/40 were used
  - 2026-10-19 **short-span control** (runs []): same predictor/targets, HuBERT-style spans (p .08, span 10): probe 0.415 (plateau 0.537, onset 60 ms), FT 0.612 (onset 37 ms) vs block-mask 0.276 / 0.392 and d2v_modern 0.466±0.031 / 0.649±0.014. Masking granularity explains most of the block-mask loss (1 seed); with short spans the predictor formulation is still below data2vec but inside the pool. Supports the hypothesis that events are not predictable from context; does not establish that the predictor itself is harmful. No further JEPA runs

**H16** [rejected (3 seeds): no gain at 2x compute] HuBERT iteration 2 (k-means on layer 6 of the iteration-1 student) improves over iteration 1  
  contract: vanilla block, 20k steps per iteration (iteration 2 costs 2x total pretraining compute), 3 seeds, same probe/FT protocol; expected: iter2 >= iter1
  - 2026-10-18 **same-seed comparison** (runs []): seed 0: probe iter1 0.426 / iter2 0.421 (plateau 0.579 / 0.561); FT iter1 0.620 / iter2 0.647 (plateau 0.716 / 0.746, any 0.700 / 0.687, onset 33 / 37 ms). Promising at FT, neutral at probe, within seed spread; unit-prediction accuracy 0.46 -> 0.68 is not evidence of better representations because the targets changed. Seeds 1-2 running; iteration 2 also uses 2x pretraining compute, so a compute-matched reading needs iteration 1 at 40k steps
  - 2026-10-19 **rejected** (runs []): 3 seeds: probe iter2 0.422±0.007 vs iter1 0.437±0.027; FT 0.633±0.014 vs 0.627±0.015; plateau 0.732 vs 0.729; any@0.2 0.677 vs 0.690. The seed-0 FT gain (0.647) did not replicate; iteration 2 costs 2x pretraining compute. Self-reclustering does not help on this corpus

**H17** [rejected (3 seeds)] BEST-RQ (random quantiser) on the modern block is competitive with teacher units / data2vec  
  contract: modern block, 3 seeds, probe and FT; expected: rq_modern >= kmaves_modern
  - 2026-10-19 **rejected** (runs []): probe 0.348±0.038 (plateau 0.458), FT 0.590±0.009 (plateau 0.678) vs kmaves_modern 0.448±0.003 / FT n/a and d2v_modern 0.466±0.031 / 0.649±0.014. Teacher-free random targets trail both teacher units and EMA targets by 0.06 at FT; still +0.12 over scratch-modern 0.467±0.010

**H18** [NOT MET (3 seeds)] Main criterion of the fixed route: modern HuBERT-family student beats the fine-tuned teacher and prior detectors on class-agnostic recall at fixed false-alarm budgets (50/100/300 FA per hour)  
  contract: recall_far.py: thresholds chosen on 2 dev recordings per fold s.t. dev FAR<=budget, test pooled; tau 0.5, class-agnostic; seed-0 FT models; expected: student recall >= teacher FT recall at each budget
  - 2026-10-19 **not met** (runs []): recall at FA<=50/100/300 per h: AVES FT (94M) 0.755/0.752/0.759 (test FAR 71/99/123); data2vec modern 0.715/0.706/0.723 (67/79/107); HuBERT km(AVES) vanilla 0.695/0.695/0.715; km(AVES) modern 0.698/0.698/0.708; iter2 0.664/0.659/0.675; scratch modern 0.644/0.644/0.653 (FAR 116-216); frozen AVES 0.729/0.721/0.729 but FAR 142-203 (budget not met). Teacher FT leads by ~0.04 recall at every budget; dev-selected thresholds overshoot the budget on test by 1.3-2x; recall is nearly flat across budgets, i.e. ~25% of calls never cross threshold (K12 recall 0.42-0.61 at 300/h). Levers in order: 32 kHz front-end (K12), DAPT on train-fold field audio, hybrid with lab classifier; seeds 1-2 for the recall table queued
  - 2026-10-19 **miss decomposition** (runs []): At the dev-100 FA/h point (test FAR 99 AVES / 79 d2v): AVES FT detected 75%, boundary-only 14% (84 events found but failing collar/offset tolerance), merged 6% (34), decoder 1%, below-threshold 2% (15), absent 2% (12); 'visible to Olga' (any overlap) 94.5%. data2vec modern: detected 71%, boundary-only 14%, merged 6%, below-threshold 5% (30), absent 4% (24); visible 90.5%. So (a) my claim 'a quarter of calls never cross threshold' was wrong: true non-detections are 5% (AVES) and 10% (student); the rest of the strict-scorer misses are found-but-mis-bounded or merged; (b) K12 is 18/152 = 12% of AVES misses, not the majority; (c) the AVES-student gap is in sensitivity (absent+below-threshold 27 vs 54 events), not in the decoder; (d) test recall-FAR curves are flat (AVES 0.75-0.77 over FAR 64-202/h; student 0.70-0.73), interpolated test recall at FAR 100: 0.761 vs 0.724. Examples exported for Olga
  - 2026-10-19 **measured cause** (runs []): Boundary-only misses: 81% have a neighbouring reference within 0.3 s (median gap -0.06 s, i.e. overlapping annotations); merged: 100% (median gap -0.35 s); detected events: only 15%. Of the dense-bout misses ~40% have a different-class neighbour (splittable with class-aware tracks), ~60% same class (needs an explicit onset/overlap model). Identical pattern for AVES FT and the student. The class-agnostic call track saturates across abutting/overlapping calls and the two-threshold decoder emits one event; the strict scorer then counts all but one as misses. This, not sensitivity, is the dominant strict-recall loss (20% of references); sensitivity explains the AVES-student gap (27 vs 54 events)
  - 2026-10-19 **not met, seeds** (runs []): recall @dev-budget 50/100/300 (mean±sd, test FAR in brackets): AVES FT n=2 0.744±0.016 (69) / 0.742±0.014 (84) / 0.753±0.008 (103); d2v_modern n=3 0.716±0.022 (63) / 0.709±0.011 (80) / 0.728±0.005 (114); km(AVES) modern n=3 0.698±0.030 / 0.702±0.025 / 0.715±0.022; km(AVES) vanilla n=3 0.697±0.002 / 0.702±0.010 / 0.719±0.011; scratch modern n=3 0.648±0.004 / 0.647 / 0.651 (FAR 116-204, budget not met). Teacher FT ahead of the best student by 0.03 at every budget; modern block gives no recall gain over the vanilla HuBERT line (0.698 vs 0.697); pretraining +0.07 over scratch at matched budget

**H19** [intermediate by pre-set criterion (+0.03..+0.08); kept as a detector component] A series-splitting decoder (class-track argmax splits and/or an auxiliary onset head trained at FT) raises strict recall at matched false-alarm rate by closing the dense-bout misses, for both the student and the fine-tuned teacher  
  contract: FT with --onset_head (BCE pos_weight 20, w=1) for d2v_modern (3 seeds) and AVES (3 seeds); 4 decoder variants tuned on dev only (max strict recall s.t. dev FAR<=100/h); report test strict recall, actual test FAR, visible(>=50% coverage) recall, boundary-only/merged counts, and FAR-matched recall from a detect sweep (interpolated 50/100/300); expected: continue if >= +0.08 strict recall at matched test FAR for at least one model; drop if < +0.03
  - 2026-10-19 **positive (onset)** (runs []): dev-budget 100 FA/h, 3 seeds per model. Student d2v+Gemma: base strict R 0.715±0.012 (test FAR 71, FAR-matched R@100 0.730) -> onset 0.755±0.012 (FAR 96, R@100 0.777, +0.047) ; class 0.745 but FAR 128 (+57/h), visible -0.007; both 0.772 but FAR 133, R@300 +0.075. Teacher AVES: base 0.754±0.008 (FAR 76, R@100 0.759) -> onset 0.821±0.004 (FAR 84, R@100 0.836, +0.077); class 0.804 at FAR 139; both 0.830 at FAR 140, R@300 +0.080. Onset split keeps or raises visible(>=50%) recall (+0.015 / +0.007) and halves boundary-only misses (70->54, 77->43); merged misses barely change (35->33, 38->34), i.e. onset split fixes abutting calls, not true overlaps. Class split raises FAR by ~60/h and lowers visibility, so its gain is not at matched FAR. Auxiliary onset loss did not change the base decoder (0.703 vs 0.706; 0.752 vs 0.752). Reading: stable improvement at comparable FAR with a clear source (dense-bout splitting); the pre-set +0.08 is met at FAR 300 for the teacher (both) and nearly at FAR 100 (onset, +0.077), +0.047 for the student. Mechanism is model-agnostic (FT-stage head + decoder); not yet run on modern HuBERT (km(AVES) modern) - porting is a 6-min FT per seed and is the open decision
  - 2026-10-19 **paired bootstrap by recording** (runs []): dev-100 operating points, 2000 resamples of 17 recordings, seeds pooled. Student (3 seeds): onset vs base +0.039 [+0.012,+0.079] strict recall, visible(>=50%) +0.015 [+0.007,+0.025]; onset vs plain FT (no aux head) +0.045 [-0.003,+0.107]; aux loss alone +0.006 [-0.022,+0.035]. Teacher (seeds 0,1 matched): onset vs base +0.071 [+0.028,+0.111], visible +0.010 [+0.004,+0.016]; onset vs plain FT +0.078 [+0.025,+0.130]; aux alone +0.007 [-0.012,+0.023]. False alarms rise with onset split (student 261->350 summed over 3 seeds, teacher 198->221 over 2), so operating-point deltas are not FAR-matched; the FAR-matched curve gives +0.047 / +0.077 at 100 FA/h. The gain is in splitting found series (strict recall), coverage gains are 0.01-0.015. Nearest prior work: Voxaboxen (Mahon, Hoffman et al., DCASE 2025) - onset/offset boxes with duration regression on an AVES backbone, graph-matched, robust to overlap, already 65.5 sed_eval F1 on these recordings; YOHO (2022) start/end regression; SEBBs (Ebbers 2024). Our mechanism is a one-logit auxiliary onset head used only to split two-threshold events: simpler, keeps the frame-wise operating point, but is a subset of what Voxaboxen does. Missing comparison: Voxaboxen under our recall-at-FAR protocol

**H20** [evaluated] Voxaboxen (published onset/offset box detector, BEATs backbone, trained only on synthetic recordings from lab clips) under our recall-at-FAR protocol, from its stored per-recording predictions (no training)  
  contract: stored fwd/bck boxes with Detection Prob >= 0.5 (tool pre-filter), fusion approximated as fwd + non-overlapping bck (IoU<0.5); threshold sweep only upward from 0.5; same 17 recordings, class-agnostic matching tau 0.5; expected: reference point for the onset-split component
  - 2026-10-19 **Voxaboxen leads at matched FAR** (runs []): fused @0.5: test FAR 61/h, strict recall 0.816, coverage>=50% 0.938, boundary-only 33, merged 56; sed_eval 62.6 with my fusion vs 65.5 reported with the tool's graph matching, so the real system is somewhat stronger. Our best onset-split models: teacher 0.821 at FAR 84, student 0.755 at FAR 96; at ~60 FA/h our teacher base is ~0.75. Voxaboxen never saw field audio while our FT models used train-fold field labels. Its curve below thr 0.5 is unavailable (pre-filtered), so dev-budget rows at <=50 FA/h are truncation artefacts. Its remaining misses are dominated by true overlaps (merged 56) rather than boundary failures (33). Conclusion: the published onset-based detector beats the one-head split at matched FAR; the contribution of our section is the miss decomposition, not a competitive detector

**H21** [evaluated] Prototype: modern HuBERT that keeps frame-level units and adds an event-instance branch (presence, onset, log-duration; overlapping instances allowed) trained already during pretraining on synthetic mixtures whose inserted-call positions are known by construction, then fine-tuned on train-fold field labels. Joint training of units and event structure reduces absorptions/boundary failures more than adding the instance head only at FT  
  contract: modern block, 20k steps; regimes R1 units only (existing kmaves_modern ckpts), R2 units + event branch (ev_w 1: BCE presence + BCE onset pw10 + 0.5 L1 log-duration), R3 event branch only (control); identical FT with the same instance head (1.5k steps) and identical instance decoder (onset peaks + predicted duration, overlaps allowed) tuned on dev under FAR<=100/h; 3 seeds each. Explicitly extra supervision from the synthetic construction, not pure SSL; expected: R2 > R1 on strict recall at matched test FAR and on the overlapping-reference subset; R3 shows whether the unit loss matters at all
  - 2026-10-19 **outcome: joint objective NOT better than the event-only control; event-instance pretraining and the instance decoder carry the gain** (runs ['fte_R1_seed0', 'fte_R1_seed1', 'fte_R1_seed2', 'fte_R2_seed0', 'fte_R2_seed1', 'fte_R2_seed2', 'fte_R3_seed0', 'fte_R3_seed1', 'fte_R3_seed2']): Instance decoder, dev<=100 FA/h, mean+-sd over 3 seeds: R1 (units only, head at FT) strict R 0.799+-0.005 at test FAR 90; R2 (units + event branch) 0.871+-0.012 at FAR 106; R3 (event branch only, no unit loss) 0.887+-0.029 at FAR 128. FAR-matched at 100/h: 0.801 / 0.873 / 0.885. Overlapping-reference subset (n=119): 0.638 / 0.723 / 0.709. Paired recording-level bootstrap (B=2000, recording carries all seeds), dR at FAR 100: R2-R1 +0.065 [+0.038,+0.095]; R3-R1 +0.083 [+0.043,+0.124]; R3-R2 +0.015 [-0.007,+0.038] (overlap subset -0.014 [-0.045,+0.011]). So (i) pretraining the instance branch on construction labels beats adding it only at FT (+0.065), (ii) the HuBERT unit loss adds nothing measurable on top of it, the event-only control is at least as good, also with the frame decoder (R3-R1 frame +0.030 [+0.006,+0.060]). (iii) The instance decoder itself is the largest effect: vs the frame decoder on the same runs +0.11 (R1) to +0.17 (R2/R3) at FAR 100, overlap-subset recall 0.24 -> 0.72. Voxaboxen (stored boxes, thr>=0.5 only, never saw field audio): at its maximal FAR 61/h R 0.816; R2 at 61/h 0.847, R3 0.849; paired dR +0.044 [-0.046,+0.166] / +0.039 [-0.051,+0.150], not resolved (Voxaboxen's single low-FAR point, wide CI); at the dev-100 operating point it reaches only 0.697 at FAR 43. Caveats: R2/R3 operating FARs overshoot the dev budget (106/128 vs 90), so matched-FAR numbers are the primary ones; R3 FAR-matched R@100 defined in all bootstrap samples, R2 in 66%; the event branch uses construction labels (supervised pretraining on synthetic mixtures), so the result says synthetic instance supervision > masked-unit SSL for this detector, not 'a better HuBERT'. Files: results/bootstrap_instances_b100.json, results/decode_instances_fte_R*_seed*_b100.json. UPDATE (wider threshold grid, th_on down to 0.005, R@100 now defined in >=99% of bootstrap samples): R2-R1 +0.066 [+0.038,+0.096]; R3-R1 +0.086 [+0.045,+0.128]; R3-R2 +0.019 [-0.004,+0.042]; instance-frame +0.157 (R2) / +0.102 (R1). Wording per supervisor: 'additional benefit of the HuBERT objective not detected in H21', not 'unnecessary'. 0.072 (seed means at dev operating points, FAR 106 vs 90) vs 0.066 (pooled FAR-matched curve, bootstrap mean) = different quantity and aggregation (check_aggregation.py); the pooled full-data estimate is 0.068.

**H22** [evaluated] Does the modern (Gemma-style) block matter for the event-only detector (R3)? R3 on the vanilla block vs R3 modern, same FT and instance decoder.  
  contract: pretrain.py --variant kmaves --block vanilla --event_head --no_hubert, seed 0 only (screening); finetune --block vanilla --event_head; decode_instances dev<=100; compare with fte_R3_seed0 (modern) via paired recording bootstrap; repeats only if the difference deserves it; expected: screening: if vanilla matches modern, the modern block has no measurable role in the event detector under this budget (this component combination only); if vanilla is worse, the block contributes and component attribution becomes a question
  - 2026-10-19 **screening (seed 0): vanilla block worse than modern in the event-only detector** (runs ['fte_R3van_seed0', 'fte_R3_seed0']): Instance decoder, dev<=100: vanilla op R 0.869 at FAR 146 vs modern 0.905 at FAR 132; R@100 from pooled test curve 0.847 vs 0.890. Paired recording bootstrap (seed 0 only): dR op -0.036 [-0.062,-0.011]; dR@100 -0.063 [-0.161,-0.022]; dR@150 -0.054 [-0.107,-0.024]; overlapping subset +0.017 [-0.045,+0.073] (no difference). Frame decoder: -0.022 [-0.067,+0.018], unresolved. Caveat: modern seed 0 is the best of its three seeds (0.905; others 0.902, 0.853), vanilla 0.869 lies inside the modern seed range, so one seed cannot settle it; seeds 1,2 queued after the Voxaboxen run (run_r3van_seeds.sh). Pretraining event loss at 20k: vanilla 0.52 vs modern 0.33 (same data), consistent with the block mattering for the event branch.
  - 2026-10-19 **3 seeds: vanilla block worse than modern in the event-only detector (every seed lower)** (runs ['fte_R3van_seed0', 'fte_R3van_seed1', 'fte_R3van_seed2', 'fte_R3_seed0', 'fte_R3_seed1', 'fte_R3_seed2']): Instance decoder, dev<=100: vanilla per-seed op recall 0.869/0.825/0.786 (pooled 0.827 at FAR 101) vs modern 0.905/0.902/0.853 (0.887 at FAR 128); pooled test-curve R@61/100/150: vanilla 0.792/0.844/0.853 vs modern 0.849/0.885/0.903. Paired recording bootstrap (recording carries all seeds): dR op -0.060 [-0.079,-0.039]; dR@100 -0.050 [-0.114,-0.018]; dR@150 -0.049 [-0.076,-0.026]; dR@61 -0.061 [-0.162,+0.001]; overlapping subset -0.068 [-0.103,-0.036]. Frame decoder: op -0.021 [-0.037,-0.005], R@100 -0.021 [-0.043,+0.000] (borderline). Reading: the modern (Gemma-style) block contributes measurably to the event-instance detector, more than it did to the frame detector with HuBERT units (H1x: ~0); closes only this component combination (RMSNorm+GeGLU+RoPE+QK-norm+local/global attention+gated conv together) at this compute budget; which component matters is not attributed. Files: results/bootstrap_h22_b100.json.

**H23** [evaluated] Field-fine-tuned Voxaboxen (BEATs, from its synthetic-trained detector) under our protocol: same folds/dev, same access to field labels, staff decoder (soft-NMS + fwd/bck bipartite matching), threshold under FA budgets; vs R2/R3 instance detectors  
  contract: voxaboxen 0.2.0 (commit 20cf966) with 4 local inference patches; per fold: train-model from final-model.pt, April-2026 hyperparameters, n_epochs 50 early stopping on dev (patience 10); inference det 0.05, comb_discard 0, Detection Prob swept; seed 0 only; paired recording bootstrap vs R1/R2/R3 at 61/100/150/300 FA/h; expected: decides competitiveness of the instance detector against the nearest published method at equal data access
  - 2026-10-19 **budget-dependent: ours ahead at <=100 FA/h and on overlapping calls; parity/behind at >=150 FA/h** (runs ['vox_ft_fold0', 'vox_ft_fold1', 'vox_ft_fold2']): VOXFT dev<=100: strict R 0.773 at test FAR 91 (thresholds 0.7/0.75/0.8), overlapping-subset 0.479, coverage 0.830; FAR-matched R@61/100/150/300 = 0.695/0.790/0.868/0.945. Paired: R2-VOXFT op +0.096 [+0.050,+0.135]; R@61 +0.142 [+0.046,+0.262]; R@100 +0.076 [-0.014,+0.176] (not resolved); R@150 +0.019 [-0.059,+0.112]; R@300 -0.031 [-0.077,+0.019] (R2 curve rarely reaches 300). R3-VOXFT: op +0.112 [+0.063,+0.153]; R@61 +0.141 [+0.053,+0.247]; R@100 +0.095 [+0.010,+0.184]; R@150 +0.038 [-0.034,+0.128]; R@300 -0.024 [-0.058,+0.019]. Overlapping subset: +0.244 [+0.168,+0.344] (R2), +0.230 [+0.167,+0.306] (R3). R1-VOXFT: R@100 +0.009 n.s., R@300 -0.121 [-0.174,-0.062] (units-only student loses at high budgets). VOXFT vs synthetic-only VOX at 61 FA/h: -0.121 [-0.204,-0.044] (field FT lowered its low-FAR recall) while raising high-FAR recall. Caveats: one Voxaboxen seed; its training ran ~48-50 epochs with the best model found late (fold0/1/2 best at epochs see log), i.e. capped by the matched budget, possibly under-trained; encoder 90M vs our 17.8M; its dev-chosen comb_discard irrelevant here because the curve is swept. Reading: at Olga's budgets (<=100 FA/h) and for overlapping calls the instance detector leads with CIs excluding zero (R3 at 100; both at 61); at >=150 FA/h Voxaboxen-FT catches up and is nominally ahead at 300 (unresolved). Files: results/voxft_b100.json, results/bootstrap_instances_b100.json.
  - 2026-10-20 **best-val checkpoint variant is weaker than the staff last-epoch model; conclusion unchanged** (runs ['vox_ftb_fold0', 'vox_ftb_fold1', 'vox_ftb_fold2']): Second pass with a local patch saving best-model.pt whenever val comb-F1 improves (best epochs 19/42/25): op 0.685 at FAR 76, overlap 0.445, coverage 0.737; R@61/100/150/300 = 0.653/0.764/0.842/0.935, i.e. below the last-epoch model everywhere (VOXFTB-VOXFT R@61 -0.043 [-0.089,+0.001], R@100 -0.024 [-0.057,+0.006]). The 'overfit last epoch' objection is dismissed: val-F1 selection picks a precision-oriented epoch that is worse under recall-at-FAR. Paired vs best-val variant: R2 R@61 +0.185 [+0.092,+0.286], R@100 +0.101 [+0.019,+0.199], R@150 +0.039 [-0.034,+0.119], R@300 -0.011 [-0.061,+0.054]; R3 R@100 +0.119 [+0.043,+0.204]; overlap +0.27/+0.28. Headline for the paper uses the stronger Voxaboxen variant (last epoch, staff pipeline). Files: results/voxftb_b100.json, results/bootstrap_instances_b100.json.
  - 2026-10-20 **wording corrected after supervisor review; overlap recall at matched FAR added** (runs []): Supervisor: field FT degraded Voxaboxen at <=61 FA/h (0.695 vs synthetic 0.816), so '+0.141' is over a degraded configuration; vs synthetic checkpoint R3 +0.033 (CI includes 0) → claim = 'beats the tested field-FT configuration of Voxaboxen', not 'beats the method'; a recall-FAR-appropriate FT regime for Voxaboxen was not found. Bipartite fwd/bck matching does not forbid overlapping outputs; soft-NMS is the candidate; explanation remains a hypothesis. Overlap recall at matched FAR (new): R3 vs VOXFT @100 +0.195 [+0.020,+0.304], @61 +0.248 [+0.139,+0.352]; vs VOX @61 +0.160 [+0.060,+0.248]; R2-R1 @100 +0.068 [+0.025,+0.112]; inst-frame @100 +0.415. Synthetic overlap fraction corrected: 57% of calls overlap another call (event-level, 1500 clips), not 37%.

**H24** [evaluated] Does overlap in the synthetic construction explain the instance detector's gain on overlapping field calls? R3 (event branch only, modern) pretrained on a no-overlap synthetic corpus vs the default corpus  
  contract: TSSSL_NOOVERLAP=1 in common.synth: same clip ids, gains and call-count draws as the default corpus, calls placed only in free segments (min gap 0.1 s), unplaceable calls dropped; measured on 1500 clips: calls/clip 2.38 vs 2.51, overlapping-call fraction 0.00 vs 0.57 (event-level definition), call-time 0.44 vs 0.38; pretraining 20k steps, 3 seeds; identical FT (field labels unchanged) and instance decoder; compare with fte_R3_seed0-2 via paired recording bootstrap; primary readout = recall on the overlapping-reference subset (n=119), secondary = R@100 and isolated-subset recall; expected: if the overlapping-subset gain shrinks on the no-overlap corpus, the overlap-rich construction is a measured difference from Voxaboxen's sequential synthetic data; if not, the gain is attributable to the decoder/branch alone
  - 2026-10-20 **3 seeds: without overlaps in the synthetic pretraining corpus the overlapping-subset recall drops by 0.11 at matched FAR, isolated recall unchanged; the instance-decoder advantage on overlaps persists at about two thirds of its size** (runs ['fte_R3noov_seed0', 'fte_R3noov_seed1', 'fte_R3noov_seed2', 'fte_R3_seed0', 'fte_R3_seed1', 'fte_R3_seed2']): Instance decoder, dev<=100: no-overlap per-seed op recall 0.876/0.840/0.825 (pooled 0.847 at FAR 111) vs default 0.905/0.902/0.853 (0.887 at FAR 128); pooled test curve R@61/100/150: 0.810/0.854/0.872 vs 0.849/0.885/0.903. Overlapping subset (n=119) at matched FAR 100: 0.600 vs 0.708; at 61: 0.518 vs 0.633; isolated subset at FAR 100: 0.916 vs 0.928. Paired recording bootstrap (B=2000, recording carries all seeds), no-overlap minus default: overlap@100 -0.110 [-0.169,-0.057], overlap@61 -0.110 [-0.154,-0.058], overlap@150 -0.109 [-0.170,-0.064]; isolated@100 -0.015 [-0.042,+0.004] (no difference); R@100 -0.033 [-0.066,-0.010]; op recall -0.040 [-0.052,-0.027]. Within the no-overlap model the instance decoder still beats the frame decoder on overlaps: +0.270 [+0.160,+0.366] at FAR 100 (default corpus: +0.422 [+0.273,+0.528]). Frame decoder of the same models is slightly BETTER on the no-overlap corpus (+0.038 [+0.010,+0.069] at FAR 100), so the loss is specific to the instance pathway, not to representation quality. Reading: overlaps in the synthetic construction are a measured contributor to overlap recall (about a quarter of the instance-vs-frame overlap advantage, 0.11 of 0.42), specific to overlapping calls; the larger part of the advantage comes from the event branch + instance decoder (and field FT labels, which contain overlaps in both arms). Explains the result; not a novelty claim (published Voxaboxen also trains on overlap-rich synthetic sets). Limits: corpus density differs slightly (-5% calls, +6 pp call time); one architecture (R3 modern); the frame-decoder gain on the no-overlap corpus shows the corpora are not perfectly matched.

**H25** [evaluated] Does the one-token/one-instance limitation of the event branch (one onset peak and one duration per 20 ms token; two calls starting in the same token cannot both be represented) explain a notable share of the remaining misses on overlapping calls? Analysis only, no training.  
  contract: analyze_onset_proximity.py on fte_R3_seed0-2 at the dev<=100 instance operating points: for each missed overlapping reference, distance from its onset to the nearest other reference onset; categories same-token (<=20 ms), within the fold's chosen d_min (peak suppressed by minimum peak distance), other (split by whether an onset peak >=0.05 exists within +-1 token of the reference onset); expected: if same-token misses are a large share, a multi-instance-per-token output (slots/queries) is a motivated architectural direction; if not, that direction is not supported by the error structure
  - 2026-10-20 **not supported: same-token misses are 6% of missed overlapping calls (2 of 34.7 per run)** (runs ['fte_R3_seed0', 'fte_R3_seed1', 'fte_R3_seed2']): Per run: 119 overlapping refs, 34.7 missed (recall 0.709). Nearest other onset for the missed ones: <=20 ms (same token) 2.0 (6%); within the chosen d_min (0.2-0.7 s depending on fold/seed) 10.0 (29%): the second onset peak is suppressed by the minimum peak distance, which the dev selection set to 0.2-0.3 s under the FA budget although 0.1 s was in the grid, i.e. a precision trade-off, not a representational limit; other 22.7 (65%), of which 12.7 have an onset peak >=0.05 at the reference onset (peak present but instance failed score/boundary matching) and 10.0 have no onset evidence (sensitivity). Only 3% of all overlapping references have another onset within 20 ms (6% of the missed ones), so a multi-instance-per-token output could recover at most ~2 references per run. Reading: the slot/query architecture direction is not motivated by the measured error structure; the remaining overlap misses are peak suppression under the FA budget, boundary/score failures and sensitivity. No architecture work follows from this.

## 5. Open items
- second seed for d2v / d2v_modern (queued)
- compute-matched 10 ms run (queued)
- bootstrap CIs over recordings
- boundary-aligned segmentation (spectral flux / self-iteration) with an alignment gate before training
- 32 kHz front-end for K12
- FAR-budget threshold selection for the screening mode
