# Streamlit Dashboard for Airbnb Capstone

## Overview

A single-page Streamlit web application that provides interactive exploration of the Airbnb data, model performance visualization, and real-time rating predictions.

## Files Created/Modified

| File | Action | Description |
|------|--------|-------------|
| `app/app.py` | Created | Main Streamlit dashboard (~500 lines) |
| `requirements.txt` | Modified | Added `streamlit` and `plotly` |

## How to Run

```bash
# Install dependencies
pip install streamlit plotly

# Run the app
cd /Users/tawfeek/Desktop/DS/classical_ML/AirBnb/capstone_22
streamlit run app/app.py
```

The app will open in your browser at `http://localhost:8501`.

## App Structure

### Page 1: EDA Explorer

Interactive exploratory data analysis of the 58,303 Airbnb listings.

**Features:**
- **Key Stats Row**: Total listings, average rating, average price, superhost percentage
- **Rating Distribution**: Histogram showing the concentration of ratings (4.5-5.0)
- **City Comparison**: Box plot and stats table comparing LA vs NYC
- **Price vs Rating**: Scatter plot with adjustable price cap slider
- **Listings Map**: Interactive map colored by rating (sampled to 5,000 points)
- **Room Type Distribution**: Grouped bar chart by city

### Page 2: Model Performance

Visualization of the trained XGBoost model's performance.

**Features:**
- **Metrics Cards**: Test RMSE (0.4258), MAE (0.2276), R² (0.1346), improvement vs dummy (7.0%)
- **Model vs Dummy**: Bar chart comparing model RMSE to baseline
- **Model Configuration**: Displays hyperparameters and feature counts
- **Feature Importance**: Horizontal bar chart of top 20 features
- **Predicted vs Actual**: Scatter plot with perfect prediction line
- **Residual Distribution**: Histogram of prediction errors

### Page 3: Predict a Rating

Two input methods available:

#### Option A: Single Listing (Interactive)

**User Inputs (14 features):**
- Host: Superhost status, response time, response rate, experience days
- Property: Type, price, accommodates, bedrooms, beds, bathrooms
- Booking: Instant bookable, minimum nights
- Amenities: Count + 8 specific amenities (WiFi, heating, etc.)
- Location: City (sets lat/lon defaults)

**Output:**
- Predicted rating with delta vs average
- Rating category (Excellent/Very Good/Good/Below Average)
- Visual gauge chart
- Top 10 feature importances

#### Option B: Batch Upload (CSV)

Upload a CSV file to get predictions for multiple listings at once.

**Required columns:**
- `price`, `accommodates`, `bedrooms`, `beds`, `bathrooms`
- `host_is_superhost` (1/0 or True/False)
- `property_type`

**Optional columns** (uses defaults if missing):
- `latitude`, `longitude`, `minimum_nights`, `maximum_nights`
- `host_response_rate`, `host_experience_days`, `instant_bookable`
- `amenities_count`, `has_wifi`, `has_heating`, etc.

**Output:**
- Summary statistics (count, mean, min, max)
- Distribution histogram of predictions
- Results table with `predicted_rating` and `rating_category` columns
- Download button to export results as CSV

## Technical Details

### Data Loading (Cached)
- `@st.cache_data` for CSV (58K rows) and metrics JSON
- `@st.cache_resource` for model.joblib (sklearn Pipeline)
- Data loads once and stays in memory across page switches

### Column Alignment
The prediction page builds a row matching the exact columns the model expects:
- Replicates `COLS_TO_EXCLUDE` and `NOISE_FEATURES` from `train.py`
- Uses dataset medians as defaults for non-interactive features
- Computes derived features (log transforms, ratios, min_stay_cost)

### Charts
- **Plotly**: Interactive charts (hover, zoom, pan)
- **Sampling**: Maps and scatters limited to 5,000 points for performance
- **Mapbox**: Uses `open-street-map` style (no API key needed)

## Dependencies Added

```
streamlit
plotly
```

Both work with the existing dependencies (pandas, numpy, scikit-learn, xgboost, joblib, category_encoders).

## App Structure Code

```
app/app.py
├── Constants (paths, exclusion lists, city coords)
├── Cached loading functions
│   ├── load_data()
│   ├── load_model()
│   ├── load_metrics()
│   └── prepare_model_data()
├── page_eda()
├── page_model()
├── page_predict()
└── main() with sidebar navigation
```

## Screenshots

Run the app to see:
1. EDA page with interactive charts and a listings map
2. Model performance with feature importance and predicted vs actual
3. Prediction tool with sliders, checkboxes, and a rating gauge
