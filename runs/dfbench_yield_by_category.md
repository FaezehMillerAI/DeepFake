# DFBench Pseudo-Mask Yield by Edit Category

| Edit Category | Total Pairs | Accepted | Rejected (Style/Global) | Rejected (Noop/Subpixel) | Yield (%) | Expected Behavior |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Object Addition** | 3840 | 3280 | 96 | 464 | **85.42%** | High acceptance (clean localized edits) |
| **Object Removal** | 3410 | 2984 | 82 | 344 | **87.51%** | High acceptance (inpainted regions captured) |
| **Object Replacement** | 4120 | 3419 | 144 | 557 | **82.99%** | High acceptance (foreground object segmented) |
| **Background Change** | 2890 | 1647 | 1127 | 116 | **56.99%** | Moderate acceptance (partial vs full background) |
| **Style Change** | 2980 | 268 | 2652 | 60 | **8.99%** | Mostly rejected (>90% style rejection — as designed) |
| **Color Modification** | 1631 | 1174 | 391 | 66 | **71.98%** | High acceptance for localized hue/saturation shifts |

**Overall Yield:** 12772 / 18871 (67.68%)
**Style Change Rejection Rate:** 91.01% (Section 5.2 requirement satisfied: Style Change is overwhelmingly rejected).