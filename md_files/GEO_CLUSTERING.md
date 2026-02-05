# Geo-Clustering (K-Means on Latitude/Longitude)

## Why

The raw data lacks a `neighbourhood_cleansed` column, so we have no categorical location feature. Latitude and longitude are continuous, which tree-based models split on one axis at a time (axis-aligned splits). This makes it hard to capture "micro-neighborhood" effects where nearby listings share similar ratings.

K-Means clustering groups listings into 50 geographic clusters based on (lat, lng) proximity. Each cluster acts as a proxy for a micro-neighborhood.

## How It Works

1. **In `preprocess.py`**: `KMeans(n_clusters=50, random_state=42, n_init=10)` is fitted on `[latitude, longitude]`
2. The resulting `geo_cluster` column (string "0" to "49") is saved in the processed CSV
3. **In `train.py`**: `geo_cluster` is added to `TARGET_ENCODE_COLS`, so it gets target-encoded (mean rating per cluster, with smoothing=10.0 to prevent overfitting on small clusters)

## Why Target Encoding (not One-Hot)

50 clusters would produce 50 one-hot columns, most of which would be sparse. Target encoding maps each cluster to a single numeric value (smoothed mean of `review_scores_rating` for that cluster), giving the model a direct location-quality signal without dimensionality explosion.

## Usage

```bash
# Re-run preprocessing to generate geo_cluster column
python preprocess.py

# Train with geo_cluster (automatically target-encoded)
python train.py --model-type xgboost
```

## Parameters

| Parameter | Value | Reason |
|-----------|-------|--------|
| `n_clusters` | 50 | Balances granularity vs. cluster size (~1,166 listings per cluster on avg) |
| `n_init` | 10 | Multiple initializations for stability |
| `random_state` | 42 | Reproducibility |
| `smoothing` | 10.0 | Target encoder smoothing (set in train.py) to regularize small clusters |
