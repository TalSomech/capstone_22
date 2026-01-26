# Feature Engineering Changes

## Overview

This document explains the feature engineering improvements made to improve model performance for predicting Airbnb review scores.

## Problem

Initial model performance was low:
- **R² = 0.135** (only 13.5% of variance explained)
- **Test RMSE = 0.4265**
- **Improvement vs baseline = 7%**

The main issue: features didn't capture what actually makes guests rate listings highly.

---

## Original Features (Before)

The original `preprocess.py` extracted **8 features**:

| Feature | Description |
|---------|-------------|
| `amenities_count` | Total number of amenities |
| `description_length` | Character count of description |
| `description_word_count` | Word count of description |
| `name_length` | Character count of listing name |
| `has_neighborhood_overview` | Binary: has neighborhood text |
| `has_host_about` | Binary: has host bio |
| `host_about_length` | Character count of host bio |
| `host_verifications_count` | Number of host verifications |

**Limitation**: These features only captured *quantity* (counts, lengths) but not *quality* (which amenities, host responsiveness, value for money).

---

## New Features (After)

### 1. Key Amenities (19 binary features)

Instead of just counting amenities, we now check for specific high-impact amenities:

| Feature | Description |
|---------|-------------|
| `has_wifi` | WiFi/wireless internet available |
| `has_kitchen` | Kitchen available |
| `has_ac` | Air conditioning |
| `has_heating` | Heating system |
| `has_washer` | Washer available |
| `has_dryer` | Dryer available |
| `has_free_parking` | Free parking on premises/street |
| `has_workspace` | Dedicated workspace |
| `has_tv` | TV/HDTV |
| `has_hot_water` | Hot water |
| `has_self_checkin` | Self check-in (lockbox/keypad/smart lock) |
| `has_smoke_alarm` | Smoke alarm/detector |
| `has_carbon_alarm` | Carbon monoxide alarm |
| `has_fire_extinguisher` | Fire extinguisher |
| `has_first_aid` | First aid kit |
| `has_pool` | Pool access |
| `has_hot_tub` | Hot tub/jacuzzi |
| `has_gym` | Gym/fitness center |
| `has_pets_allowed` | Pets allowed |

**Why it matters**: Guests care about *which* amenities exist, not just how many. WiFi and AC have much bigger impact than "extra pillows".

---

### 2. Host Experience

| Feature | Description |
|---------|-------------|
| `host_experience_days` | Days since host joined Airbnb |
| `response_time_score` | Numeric encoding of response time (1=within hour, 5=unknown) |

**Why it matters**: Experienced hosts and quick responders typically get better reviews.

---

### 3. Value & Comfort Ratios

| Feature | Description |
|---------|-------------|
| `price_per_person` | Price / accommodates (value for money) |
| `bedrooms_per_person` | Bedrooms / accommodates (space comfort) |
| `beds_per_person` | Beds / accommodates (sleeping comfort) |
| `bathrooms_per_person` | Bathrooms / accommodates |

**Why it matters**: A $200/night listing for 2 people is different from $200/night for 8 people.

---

### 4. Room Type Encoding

| Feature | Description |
|---------|-------------|
| `is_entire_home` | Binary: entire home/apartment |
| `is_private_room` | Binary: private room |

**Why it matters**: Room type significantly affects guest expectations and satisfaction.

---

### 5. Safety Score

| Feature | Description |
|---------|-------------|
| `safety_amenities_count` | Sum of safety amenities (smoke alarm, carbon alarm, fire extinguisher, first aid) |

**Why it matters**: Safety-conscious listings may attract more careful hosts who maintain quality.

---

## Summary of Changes

| Category | Before | After |
|----------|--------|-------|
| Total features | 8 | 37 |
| Amenity features | 1 (count only) | 20 (count + 19 specific) |
| Host features | 1 | 3 |
| Ratio features | 0 | 4 |
| Room type features | 0 | 2 |
| Safety features | 0 | 1 |

---

## How to Use

1. **Re-run preprocessing** to generate new features:
   ```bash
   python preprocess.py
   ```

2. **Re-train the model** with new features:
   ```bash
   python train.py
   ```

3. **Compare results** - expect improvement in R² and RMSE.

---

## Expected Impact

These features should improve model performance because they capture:

1. **What guests actually care about** (specific amenities, not just counts)
2. **Value for money** (price relative to capacity)
3. **Host quality signals** (experience, responsiveness)
4. **Comfort factors** (space per person)
5. **Trust signals** (safety equipment)

The R² should improve from ~0.135 to potentially 0.20-0.30+ depending on data quality.

---

## Classification Approach (Alternative)

Since regression R² remains low (~0.145), we also provide a **classification approach**.

### Why Classification?

| Problem | Regression | Classification |
|---------|------------|----------------|
| Task | Predict exact score (4.2, 4.7, 4.95) | Predict category (excellent vs not) |
| Difficulty | Hard - small differences are random | Easier - distinguish two groups |
| Metrics | R², RMSE (hard to interpret) | Accuracy, Precision, Recall (intuitive) |
| Business value | Limited | Actionable (filter top listings) |

### How to Use

```bash
# Train classifier (default: rating >= 4.8 is "excellent")
python train_classifier.py

# Try different threshold
python train_classifier.py --threshold 4.9

# Try different model
python train_classifier.py --model-type gradient_boosting
```

### Metrics Explained

| Metric | Meaning |
|--------|---------|
| **Accuracy** | % of correct predictions overall |
| **Precision** | When we predict "excellent", how often are we right? |
| **Recall** | Of all truly excellent listings, how many do we catch? |
| **F1 Score** | Balance between precision and recall |
| **AUC** | Model's ability to distinguish classes (0.5 = random, 1.0 = perfect) |
