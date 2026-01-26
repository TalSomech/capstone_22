# Feature Exclusions for Model Parsimony

## Overview

With 62 features and an R² of only 0.145, the model was drowning signal in noise. In Random Forests, collinear features "dilute" feature importance, making it harder to identify what actually matters.

This document explains which features were excluded and why, following biostatistics principles of **parsimonious modeling** (Occam's razor).

---

## Excluded Features

### 1. The "Identifier" Trap (High Cardinality)

| Feature | Reason |
|---------|--------|
| `host_id` | Nominal identifier, not a feature. The model tries to "memorize" specific hosts rather than learning generalizable traits. Leads to massive overfitting. |

---

### 2. Perfect Multicollinearity (Redundancy)

| Feature | Reason |
|---------|--------|
| `room_type` | Already encoded as `is_entire_home` and `is_private_room` binary features. Including the categorical version adds no new information. |
| `calculated_host_listings_count_entire_homes` | Breakdown of `host_listings_count`. Rarely adds value for predicting rating quality. |
| `calculated_host_listings_count_private_rooms` | Same as above. |
| `calculated_host_listings_count_shared_rooms` | Same as above. |
| `host_total_listings_count` | Usually duplicate of `host_listings_count` (correlation > 0.99). |

---

### 3. The "Availability" Bloat

**Problem**: 8 variables just describing "minimum nights" - extreme collinearity.

| Feature | Reason |
|---------|--------|
| `minimum_minimum_nights` | Subtle differences between "minimum_minimum" and "average_minimum" are negligible for guest satisfaction. |
| `maximum_minimum_nights` | Same as above. |
| `minimum_maximum_nights` | Same as above. |
| `maximum_maximum_nights` | Same as above. |
| `minimum_nights_avg_ntm` | Same as above. |
| `maximum_nights_avg_ntm` | Same as above. |

**Kept**: `minimum_nights` and `maximum_nights` only.

---

### 4. Temporal Redundancy

| Feature | Reason |
|---------|--------|
| `availability_30` | Subset of `availability_365`. If a place is available for 30 days, it's likely available for 60. |
| `availability_60` | Same as above. |
| `availability_90` | Same as above. |
| `number_of_reviews_ltm` | Correlates heavily with `reviews_per_month`. |
| `number_of_reviews_l30d` | Same as above. |

**Kept**: `availability_365` (general busyness), `number_of_reviews` (total social proof), `reviews_per_month` (current velocity).

---

### 5. Low Variance / Irrelevant Signals

| Feature | Reason |
|---------|--------|
| `host_has_profile_pic` | Nearly always 1 (True) on Airbnb. Variables with zero or near-zero variance offer no splitting power. |

---

## Benefits of Exclusion

By reducing the number of predictors (p), we:

1. **Reduce Variance**: The model is less likely to chase noise.
2. **Increase Interpretability**: Feature importance is no longer split among redundant variables.
3. **Improve Generalization**: Less overfitting to training data quirks.
4. **Faster Training**: Fewer features = faster computation.

---

## Summary

| Category | Features Dropped | Count |
|----------|------------------|-------|
| High Cardinality ID | `host_id` | 1 |
| Multicollinearity | `room_type`, host listing breakdowns | 5 |
| Availability Bloat | min/max night variants | 6 |
| Temporal Redundancy | availability subsets, review subsets | 5 |
| Low Variance | `host_has_profile_pic` | 1 |
| **Total** | | **18** |

**Before**: 62 features
**After**: ~44 features

---

## Next Steps

After re-running the model with excluded features, consider:

1. **Permutation Importance**: More reliable than impurity-based importance (which is biased toward high-cardinality and continuous variables).
2. **Correlation Matrix**: Verify remaining features don't have |r| > 0.9.
3. **VIF Analysis**: Check Variance Inflation Factor for multicollinearity.
