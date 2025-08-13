# CTGAN & TVAE on the Adult Dataset
**Author:** Oscar Nolen  
**Date:** 08/13/2025  

## Task 2 - ML Efficacy with Synthetic Data
The models were trained on the real Adult training split, used to generate synthetic data, and then the paper's classifiers were trained on the synthetic data and tested on the real test split.

### My Results (Adult test set)
| Source            | Classifier                  | Accuracy | F1     |
|-------------------|-----------------------------|----------|--------|
| CTGAN_synth_train | AdaBoost(n=50)              | 0.8277   | 0.5279 |
| CTGAN_synth_train | DecisionTree(max_depth=20)  | 0.8128   | 0.5540 |
| CTGAN_synth_train | LogisticRegression          | 0.8392   | 0.6048 |
| CTGAN_synth_train | MLP(50)                     | 0.8243   | 0.5618 |
| TVAE_synth_train  | AdaBoost(n=50)              | 0.8391   | 0.6477 |
| TVAE_synth_train  | DecisionTree(max_depth=20)  | 0.7897   | 0.5634 |
| TVAE_synth_train  | LogisticRegression          | 0.8341   | 0.6600 |
| TVAE_synth_train  | MLP(50)                     | 0.8237   | 0.6104 |

**Averages:**
- CTGAN: Accuracy ≈ 0.8260, F1 ≈ 0.5621  
- TVAE: Accuracy ≈ 0.8217, F1 ≈ 0.6204  

**Paper context:** The CTGAN paper reports cross-dataset averages, not per-dataset numbers, so exact matches aren't expected. However, my results follow the paper's pattern: TVAE slightly outperforms CTGAN in F1 on this dataset.

## Task 3 - ML Efficacy with Real Data
The same classifiers were trained on the real Adult training split and evaluated on the real test split.

### My Results
| Classifier                  | Accuracy | F1     |
|-----------------------------|----------|--------|
| AdaBoost(n=50)              | 0.8512   | 0.6473 |
| DecisionTree(max_depth=20)  | 0.8275   | 0.6377 |
| LogisticRegression          | 0.8502   | 0.6647 |
| MLP(50)                     | 0.8414   | 0.6560 |
| **Average**                 | 0.8426   | 0.6514 |

### Paper (Table 5)
| Classifier                  | Accuracy | F1     |
|-----------------------------|----------|--------|
| AdaBoost(n=50)              | 0.8607   | 0.6803 |
| DecisionTree(max_depth=20)  | 0.7984   | 0.6577 |
| LogisticRegression          | 0.7953   | 0.6606 |
| MLP(50)                     | 0.8506   | 0.6757 |

**Comparison:** My numbers are close but not identical. The differences likely come from split reproduction, preprocessing, and scikit-learn version defaults.

## Task 2 vs Task 3
Real-data training (Task 3) yields higher F1 scores (~0.65) than synthetic training (CTGAN ≈ 0.56, TVAE ≈ 0.62). This confirms that while synthetic data can be useful, it does not fully match the utility of real data for this task. TVAE narrows the gap more than CTGAN, consistent with the paper's qualitative conclusions.

## Task 4 - Privacy via DCR
Mean DCR was computed for the synthetic datasets from Task 2, and then for datasets of size ×2 and ×4.

| Model | Size Multiple | Mean DCR |
|-------|--------------:|---------:|
| CTGAN | 1             | 1.4584   |
| CTGAN | 2             | 1.4597   |
| CTGAN | 4             | 1.4597   |
| TVAE  | 1             | 0.8281   |
| TVAE  | 2             | 0.8345   |
| TVAE  | 4             | 0.8371   |

**Trend:** DCR values are stable or slightly increasing with dataset size. CTGAN has higher DCR (suggesting greater distance from real records, potentially more privacy) but lower utility. TVAE has lower DCR (closer to real data, potentially less privacy) but higher ML efficacy. This aligns with the common privacy–utility trade-off.
