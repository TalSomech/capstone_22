# Model Improvements

## Problem Statement

With `review_scores_rating` as target (mean=4.754, std=0.445, median=4.9), the model was achieving R²=0.134. The vast majority of ratings cluster between 4.5 and 5.0, creating a ceiling effect that makes regression difficult.

This document describes the improvements implemented to address this.

---

## 1. Target Transformation (`--target-transform`)

### Why

The target distribution is heavily left-skewed with a ceiling at 5.0. Most ratings fall in a narrow 0.3-point band (4.7-5.0). Standard regression treats the difference between 4.8 and 4.9 the same as 2.0 and 2.1, but the former is much harder to distinguish.

### Available Transforms

| Transform | Description |
|-----------|-------------|
| `none` (default) | No transformation |
| `quantile_normal` | QuantileTransformer maps target to normal distribution. Spreads the dense 4-5 region across the full Gaussian range. Sklearn handles fit/inverse natively. |
| `log1p` | Simple `log(1 + y)` transform. Mild spreading of values. |

### Why `quantile_normal` (not `reflected_log`)

The original `reflected_log` transform (`-log(5.01 - y)`) was tested but produced **negative R²** (worse than predicting the mean). The nonlinear inverse `5.01 - exp(-x)` amplified prediction errors in the dense region.

`quantile_normal` uses sklearn's `QuantileTransformer` which:
- Has a well-conditioned inverse (piecewise linear interpolation)
- Maps the target to a standard normal distribution
- Spreads the dense 4.5-5.0 region across the full range
- Is fitted on training data only (no leakage)

### Inverse Transform

Predictions are always inverse-transformed back to the original rating scale before computing metrics (RMSE, MAE, R²). All plots and wandb logs use the original scale for fair comparison.

### Usage

```bash
# Without transform (baseline)
python train.py --model-type gradient_boosting

# With quantile normal transform
python train.py --model-type gradient_boosting --target-transform quantile_normal

# With log1p transform
python train.py --model-type gradient_boosting --target-transform log1p
```

---

## 2. Feature Set Control (`--feature-set`)

### Why

22 features were excluded for having negative permutation importance. However, many of these are **domain-meaningful** (e.g., `accommodates`, `bedrooms`, `bathrooms`, `is_entire_home`). They may have had negative importance because:

1. The untransformed target had too little variance for them to matter
2. Feature interactions were masking their individual contributions
3. The model wasn't powerful enough to leverage them

### Experiment Result

`--feature-set full` (58 features) did **not** improve over `lean` (36 features):
- lean: R²=0.1374, RMSE=0.4251
- full: R²=0.1285, RMSE=0.4273

The lean feature set exclusions are validated. More features added noise.

### Two Modes

| Mode | Features | Use Case |
|------|----------|----------|
| `lean` (default) | ~36 features | Parsimonious model, best generalization |
| `full` | ~58 features | All domain-meaningful features included |

### Features Re-included in `full` Mode

Accommodation: `accommodates`, `bedrooms`, `beds`, `bathrooms`
Amenities: `has_washer`, `has_kitchen`, `has_dryer`, `has_self_checkin`, `has_pets_allowed`, `has_free_parking`, `has_carbon_alarm`, `has_tv`, `has_ac`, `has_pool`, `has_fire_extinguisher`
Room type: `is_entire_home`, `is_private_room`
Host/Text: `host_about_length`, `host_acceptance_rate`, `name_length`, `description_word_count`, `has_neighborhood_overview`

### Usage

```bash
# Lean feature set (default, best performance)
python train.py --model-type gradient_boosting --feature-set lean

# Full feature set
python train.py --model-type gradient_boosting --feature-set full
```

---

## 3. HistGradientBoosting (`--model-type hist_gradient_boosting`)

### Why

`HistGradientBoostingRegressor` is scikit-learn's implementation of histogram-based gradient boosting (similar to LightGBM). Compared to `GradientBoostingRegressor`:

| Property | GradientBoosting | HistGradientBoosting |
|----------|-----------------|---------------------|
| Speed | Slow (exact splits) | Fast (histogram binning) |
| Missing values | Requires imputation | Handles natively |
| Large datasets | Struggles | Designed for |
| Categorical features | Needs encoding | Can handle natively |

### Experiment Result

Best performer so far: **R²=0.1411, RMSE=0.4242** (lean features, no transform).

### Hyperparameter Search Space

```
max_iter: [100, 200, 300, 500]
max_depth: [3, 5, 7, None]
learning_rate: [0.01, 0.05, 0.1]
min_samples_leaf: [5, 10, 20]
max_leaf_nodes: [15, 31, 63, None]
```

### Usage

```bash
python train.py --model-type hist_gradient_boosting
```

---

## 4. Expanded Hyperparameter Search

### Changes

**Random Forest**: Added `n_estimators=500`

**Gradient Boosting**: Added `n_estimators=500` and `subsample=[0.8, 0.9, 1.0]`
- Stochastic gradient boosting (`subsample < 1.0`) adds regularization by training each tree on a random subset of the data, reducing overfitting.

---

## Experiment Results (Round 1)

| Run | Model | Transform | Features | Test R² | Test RMSE | vs Dummy |
|-----|-------|-----------|----------|---------|-----------|----------|
| cerulean-elevator-6 | gradient_boosting | none | lean | 0.1374 | 0.4251 | +7.1% |
| dry-water-8 | gradient_boosting | none | full | 0.1285 | 0.4273 | +6.6% |
| **deft-wind-11** | **hist_gradient_boosting** | **none** | **lean** | **0.1411** | **0.4242** | **+7.3%** |
| dulcet-snow-7 | gradient_boosting | reflected_log | lean | -0.0692 | 0.4732 | -3.4% |
| dulcet-water-9 | gradient_boosting | reflected_log | full | -0.0634 | 0.4720 | -3.1% |
| fallen-surf-12 | hist_gradient_boosting | reflected_log | full | -0.0706 | 0.4736 | -3.5% |

### Key Findings

1. **`reflected_log` hurts** - Replaced with `quantile_normal` (sklearn QuantileTransformer)
2. **`hist_gradient_boosting` is best** - Marginal but consistent improvement
3. **`lean` beats `full`** - Feature exclusions validated

---

## Recommended Next Experiments

```bash
# Best model + quantile transform
python train.py --model-type hist_gradient_boosting --target-transform quantile_normal

# Best model + log1p
python train.py --model-type hist_gradient_boosting --target-transform log1p
```

---

## Note on `neighbourhood_cleansed`

`TARGET_ENCODE_COLS` references `neighbourhood_cleansed`, but this column does not exist in the raw data files (`listingsLA.csv`, `listingsNYC.csv`). The raw data only has `neighborhood_overview` (free text), which is already used for the binary feature `has_neighborhood_overview`. Location is represented through `latitude` and `longitude` coordinates.
