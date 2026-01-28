# Feature Diagnostics Guide

## Overview

This document explains the three diagnostic analyses used to evaluate feature quality and multicollinearity in our AirBnB rating prediction model.

---

## 1. Permutation Importance

### What It Is

Permutation importance measures how much the model's performance decreases when a single feature's values are randomly shuffled. Unlike impurity-based importance (default in Random Forest), it is:

- **Unbiased**: Not affected by feature cardinality or scale
- **Model-agnostic**: Works with any model type
- **Reliable**: Measures actual predictive power on test data

### Interpretation

| Importance Value | Meaning |
|------------------|---------|
| **Positive (high)** | Feature is important - shuffling it hurts predictions |
| **Near zero** | Feature has little predictive value |
| **Negative** | Feature may be adding noise - consider removing |

### Why Impurity-Based Importance Can Be Misleading

Random Forest's default `.feature_importances_` is biased toward:
- High-cardinality features (e.g., `host_id` with thousands of unique values)
- Continuous variables (vs. binary features)
- Features with many possible split points

**Example**: `host_id` might show high impurity-based importance simply because it has many unique values, not because it generalizes well.

---

## 2. Correlation Matrix

### What It Is

A correlation matrix shows pairwise Pearson correlation coefficients (r) between all numeric features.

### Interpretation

| Correlation (|r|) | Meaning |
|-------------------|---------|
| **> 0.9** | Very high - near-perfect linear relationship (problematic) |
| **0.7 - 0.9** | High - strong relationship (concerning) |
| **0.4 - 0.7** | Moderate - some relationship |
| **< 0.4** | Low - weak or no relationship |

### Why High Correlation Is Problematic

When two features are highly correlated (|r| > 0.9):

1. **Redundancy**: They carry the same information
2. **Importance dilution**: Random Forest splits importance between them
3. **Instability**: Small data changes can swap which feature gets used
4. **Interpretability**: Hard to know which feature actually matters

### Action

For pairs with |r| > 0.9, consider dropping one:
- Keep the feature that is more interpretable
- Keep the feature with lower VIF
- Keep the feature with higher permutation importance

---

## 3. VIF Analysis (Variance Inflation Factor)

### What It Is

VIF measures how much the variance of a regression coefficient is inflated due to multicollinearity with other features.

**Formula**: VIF = 1 / (1 - R²), where R² is from regressing the feature on all other features.

### Interpretation

| VIF | Multicollinearity Level | Action |
|-----|-------------------------|--------|
| **1** | None | Perfect |
| **1 - 5** | Low | Acceptable |
| **5 - 10** | Moderate | Concerning - monitor |
| **> 10** | Severe | Problematic - consider removal |
| **> 100** | Extreme | Feature is almost perfectly predicted by others |

### Why High VIF Is Problematic

1. **Unstable coefficients**: Small changes in data cause large coefficient swings
2. **Inflated standard errors**: Harder to determine statistical significance
3. **Redundant information**: The feature can be predicted from other features

### Common High-VIF Scenarios

- `price` and `price_per_person` (one is derived from the other)
- `beds` and `bedrooms` (highly correlated)
- `availability_30` and `availability_365` (one is subset of other)

---

## How to Use the Diagnostic Script

```bash
python feature_diagnostics.py
```

### Output Files

| File | Description |
|------|-------------|
| `results/diagnostics/correlation_matrix.png` | Visual heatmap of feature correlations |
| `results/diagnostics/permutation_importance.csv` | All features ranked by permutation importance |
| `results/diagnostics/permutation_importance.png` | Bar chart of top 30 features |
| `results/diagnostics/vif_analysis.csv` | VIF values for all numeric features |
| `results/diagnostics/diagnostics_summary.json` | Combined JSON with all results |

---

## Decision Framework

After running diagnostics, follow this process:

### Step 1: Check High Correlations

```
If |r| > 0.9 between features A and B:
  → Compare permutation importance
  → Keep the one with higher importance
  → Drop the other
```

### Step 2: Check VIF

```
If VIF > 10:
  → Check if feature is in a correlated pair
  → Consider if feature is derived from others (e.g., ratios)
  → Drop if redundant
```

### Step 3: Check Permutation Importance

```
If importance <= 0:
  → Feature is likely adding noise
  → Safe to remove (may improve generalization)
```

---

## Example Analysis

**Scenario**: You find these results:

| Feature Pair | Correlation |
|--------------|-------------|
| `beds` / `bedrooms` | r = 0.92 |

| Feature | VIF |
|---------|-----|
| `beds` | 8.5 |
| `bedrooms` | 7.2 |

| Feature | Permutation Importance |
|---------|------------------------|
| `beds` | 0.002 |
| `bedrooms` | 0.005 |

**Decision**: Keep `bedrooms` (higher importance, lower VIF). Drop `beds`.

---

## References

1. Breiman, L. (2001). "Random Forests" - Original RF paper
2. Strobl, C. et al. (2007). "Bias in random forest variable importance measures"
3. James, G. et al. (2013). "An Introduction to Statistical Learning" - VIF explanation
