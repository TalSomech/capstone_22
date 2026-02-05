# app/app.py — Streamlit Dashboard for Airbnb Capstone
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import json
from pathlib import Path
from sklearn.model_selection import train_test_split

# =============================================================================
# CONSTANTS
# =============================================================================
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "processed" / "listings_combined_clean.csv"
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
METRICS_PATH = BASE_DIR / "results" / "metrics.json"

TARGET_COL = "review_scores_rating"

# Same exclusions as train.py (lean feature set)
COLS_TO_EXCLUDE = [
    "city", "host_id", "room_type",
    "minimum_minimum_nights", "maximum_minimum_nights",
    "minimum_maximum_nights", "maximum_maximum_nights",
    "minimum_nights_avg_ntm", "maximum_nights_avg_ntm",
    "availability_30", "availability_60", "availability_90",
    "number_of_reviews_ltm", "number_of_reviews_l30d",
    "calculated_host_listings_count_entire_homes",
    "calculated_host_listings_count_private_rooms",
    "calculated_host_listings_count_shared_rooms",
    "host_total_listings_count",
    "host_has_profile_pic",
    "number_of_reviews", "host_listings_count",
]

NOISE_FEATURES = [
    "has_washer", "description_word_count", "has_kitchen", "is_entire_home",
    "has_dryer", "has_self_checkin", "has_pets_allowed", "has_free_parking",
    "is_private_room", "has_neighborhood_overview", "has_carbon_alarm",
    "host_about_length", "has_tv", "bedrooms", "host_acceptance_rate",
    "bathrooms", "has_fire_extinguisher", "name_length", "has_pool",
    "accommodates", "has_ac", "beds",
]

# Response time encoding (same as train.py)
RESPONSE_TIME_MAP = {
    "within an hour": 5,
    "within a few hours": 4,
    "within a day": 3,
    "a few days or more": 2,
}

# City coordinates for map defaults
CITY_COORDS = {
    "LA": {"lat": 34.05, "lon": -118.25},
    "NYC": {"lat": 40.73, "lon": -73.99},
}


# =============================================================================
# CACHED DATA LOADING
# =============================================================================
@st.cache_data
def load_data():
    """Load the processed listings CSV."""
    return pd.read_csv(DATA_PATH)


@st.cache_resource
def load_model():
    """Load the trained sklearn pipeline."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics():
    """Load the metrics JSON."""
    with open(METRICS_PATH, "r") as f:
        return json.load(f)


@st.cache_data
def prepare_model_data(_df):
    """Prepare X, y and train/test split matching train.py exactly."""
    df = _df.copy()

    # Apply same exclusions as train.py (lean mode)
    exclude_cols = [c for c in COLS_TO_EXCLUDE if c in df.columns]
    exclude_cols += [c for c in NOISE_FEATURES if c in df.columns and c not in exclude_cols]

    X = df.drop(columns=[TARGET_COL] + exclude_cols, errors="ignore")
    y = df[TARGET_COL]

    # Same split parameters as train.py
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    return X_train, X_test, y_train, y_test


# =============================================================================
# PAGE 1: EDA EXPLORER
# =============================================================================
def page_eda():
    st.header("Exploratory Data Analysis")

    df = load_data()

    # Key Stats Row
    st.subheader("Key Statistics")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Listings", f"{len(df):,}")
    with col2:
        st.metric("Average Rating", f"{df[TARGET_COL].mean():.2f}")
    with col3:
        st.metric("Average Price", f"${df['price'].mean():.0f}")
    with col4:
        superhost_pct = df["host_is_superhost"].mean() * 100
        st.metric("Superhost %", f"{superhost_pct:.1f}%")

    st.markdown("---")

    # Rating Distribution
    st.subheader("Rating Distribution")
    fig_rating = px.histogram(
        df, x=TARGET_COL, nbins=50,
        color_discrete_sequence=["steelblue"],
        labels={TARGET_COL: "Review Score Rating"},
    )
    fig_rating.add_vline(
        x=df[TARGET_COL].mean(), line_dash="dash", line_color="red",
        annotation_text=f"Mean: {df[TARGET_COL].mean():.2f}"
    )
    fig_rating.update_layout(showlegend=False)
    st.plotly_chart(fig_rating, use_container_width=True)

    st.info("Most ratings are concentrated between 4.5 and 5.0, creating a ceiling effect that limits model performance.")

    st.markdown("---")

    # City Comparison
    st.subheader("City Comparison: LA vs NYC")
    col1, col2 = st.columns(2)

    with col1:
        fig_box = px.box(
            df, x="city", y=TARGET_COL, color="city",
            labels={TARGET_COL: "Rating", "city": "City"},
            color_discrete_map={"LA": "coral", "NYC": "steelblue"},
        )
        fig_box.update_layout(showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    with col2:
        city_stats = df.groupby("city")[TARGET_COL].agg(["count", "mean", "median", "std"]).round(3)
        city_stats.columns = ["Count", "Mean", "Median", "Std Dev"]
        st.dataframe(city_stats, use_container_width=True)

    st.markdown("---")

    # Price vs Rating
    st.subheader("Price vs Rating")
    max_price = st.slider("Max price to display ($)", 50, 2000, 500)

    df_filtered = df[df["price"] <= max_price]
    sample_size = min(5000, len(df_filtered))
    df_sample = df_filtered.sample(n=sample_size, random_state=42)

    fig_scatter = px.scatter(
        df_sample, x="price", y=TARGET_COL, color="city",
        opacity=0.4,
        labels={"price": "Price ($)", TARGET_COL: "Rating"},
        color_discrete_map={"LA": "coral", "NYC": "steelblue"},
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("---")

    # Map of Listings
    st.subheader("Listings Map (colored by rating)")

    df_map = df.dropna(subset=["latitude", "longitude", TARGET_COL])
    df_map_sample = df_map.sample(n=min(5000, len(df_map)), random_state=42)

    fig_map = px.scatter_mapbox(
        df_map_sample,
        lat="latitude", lon="longitude",
        color=TARGET_COL,
        color_continuous_scale="RdYlGn",
        range_color=[3.5, 5.0],
        zoom=9,
        height=500,
        opacity=0.6,
        labels={TARGET_COL: "Rating"},
    )
    fig_map.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("---")

    # Room Type Distribution
    st.subheader("Room Type Distribution")
    fig_room = px.histogram(
        df, x="room_type", color="city", barmode="group",
        color_discrete_map={"LA": "coral", "NYC": "steelblue"},
    )
    st.plotly_chart(fig_room, use_container_width=True)


# =============================================================================
# PAGE 2: MODEL PERFORMANCE
# =============================================================================
def page_model():
    st.header("Model Performance")

    metrics_data = load_metrics()
    metrics = metrics_data["metrics"]
    feature_importance = metrics_data["feature_importance"]

    # Metrics Cards
    st.subheader("Test Set Metrics")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        delta = metrics["dummy_rmse"] - metrics["test_rmse"]
        st.metric("Test RMSE", f"{metrics['test_rmse']:.4f}", delta=f"-{delta:.4f} vs dummy")
    with col2:
        st.metric("Test MAE", f"{metrics['test_mae']:.4f}")
    with col3:
        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
    with col4:
        st.metric("Improvement vs Dummy", f"{metrics['rmse_improvement_vs_dummy']:.1f}%")

    st.info(f"R² = {metrics['test_r2']:.3f} means the model explains ~{metrics['test_r2']*100:.1f}% of rating variance. This is modest but expected given the tight rating distribution (most are 4.5-5.0).")

    st.markdown("---")

    # Dummy vs Model Comparison
    st.subheader("Model vs Dummy Baseline")
    col1, col2 = st.columns(2)

    with col1:
        fig_compare = go.Figure(go.Bar(
            x=["Dummy (Mean)", "XGBoost Model"],
            y=[metrics["dummy_rmse"], metrics["test_rmse"]],
            marker_color=["#EF553B", "#00CC96"],
            text=[f"{metrics['dummy_rmse']:.4f}", f"{metrics['test_rmse']:.4f}"],
            textposition="outside",
        ))
        fig_compare.update_layout(
            yaxis_title="RMSE (lower is better)",
            showlegend=False,
            height=400,
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    with col2:
        st.markdown("**Model Configuration:**")
        st.markdown(f"- **Model:** {metrics['model_type']}")
        st.markdown(f"- **Features:** {metrics['n_features_numeric']} numeric + {metrics['n_features_categorical']} categorical")
        st.markdown(f"- **CV Folds:** {metrics['cv_folds']}")
        if metrics.get("best_params"):
            st.markdown("**Best Hyperparameters:**")
            for k, v in metrics["best_params"].items():
                param_name = k.replace("model__", "")
                st.markdown(f"- {param_name}: {v}")

    st.markdown("---")

    # Feature Importance
    st.subheader("Top 20 Feature Importances")

    # Clean feature names and sort
    fi_clean = {}
    for k, v in feature_importance.items():
        clean_name = k.replace("num__", "").replace("target__", "target_encoded_").replace("cat__", "")
        fi_clean[clean_name] = v

    fi_df = pd.DataFrame({
        "Feature": list(fi_clean.keys()),
        "Importance": list(fi_clean.values()),
    }).sort_values("Importance", ascending=True)

    fig_fi = px.bar(
        fi_df, x="Importance", y="Feature", orientation="h",
        color="Importance", color_continuous_scale="Blues",
    )
    fig_fi.update_layout(height=600, showlegend=False)
    st.plotly_chart(fig_fi, use_container_width=True)

    st.markdown("---")

    # Predicted vs Actual
    st.subheader("Predicted vs Actual Ratings")

    df = load_data()
    model = load_model()
    _, X_test, _, y_test = prepare_model_data(df)

    y_pred = model.predict(X_test)

    # Sample for visualization
    sample_size = min(5000, len(y_test))
    indices = np.random.RandomState(42).choice(len(y_test), sample_size, replace=False)
    y_test_sample = y_test.iloc[indices].values
    y_pred_sample = y_pred[indices]

    fig_pred = px.scatter(
        x=y_test_sample, y=y_pred_sample, opacity=0.3,
        labels={"x": "Actual Rating", "y": "Predicted Rating"},
    )
    fig_pred.add_trace(go.Scatter(
        x=[1, 5], y=[1, 5], mode="lines",
        line=dict(color="red", dash="dash"),
        name="Perfect Prediction",
    ))
    fig_pred.update_layout(height=500)
    st.plotly_chart(fig_pred, use_container_width=True)

    st.info("The model predicts within a narrow band (4.2-5.0) because most actual ratings are concentrated there. The diagonal red line shows perfect prediction.")

    st.markdown("---")

    # Residual Distribution
    st.subheader("Residual Distribution")
    residuals = y_test.values - y_pred

    fig_resid = px.histogram(
        x=residuals, nbins=50,
        labels={"x": "Residual (Actual - Predicted)"},
        color_discrete_sequence=["steelblue"],
    )
    fig_resid.add_vline(x=0, line_dash="dash", line_color="red")
    st.plotly_chart(fig_resid, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Residual", f"{residuals.mean():.4f}")
    with col2:
        st.metric("Std Residual", f"{residuals.std():.4f}")


# =============================================================================
# PAGE 3: PREDICT A RATING
# =============================================================================
def page_predict():
    st.header("Predict a Listing Rating")

    df = load_data()
    model = load_model()

    # Choose input method
    input_method = st.radio(
        "Choose input method:",
        ["Single Listing (Interactive)", "Batch Upload (CSV)"],
        horizontal=True,
    )

    if input_method == "Batch Upload (CSV)":
        page_predict_batch(df, model)
    else:
        page_predict_single(df, model)


def page_predict_batch(df, model):
    """Handle CSV batch predictions."""
    st.subheader("Upload CSV for Batch Predictions")

    st.markdown("""
    Upload a CSV file with listing data. The model will predict ratings for each row.

    **Required columns** (minimum):
    - `price`, `accommodates`, `bedrooms`, `beds`, `bathrooms`
    - `host_is_superhost` (1/0 or True/False)
    - `property_type`

    **Optional columns** (will use defaults if missing):
    - `latitude`, `longitude`, `minimum_nights`, `maximum_nights`
    - `host_response_rate`, `host_experience_days`, `instant_bookable`
    - `amenities_count`, `has_wifi`, `has_heating`, etc.
    """)

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        try:
            user_df = pd.read_csv(uploaded_file)
            st.success(f"Loaded {len(user_df):,} rows")

            # Show preview
            st.markdown("**Preview of uploaded data:**")
            st.dataframe(user_df.head(10), use_container_width=True)

            if st.button("Generate Predictions", type="primary"):
                with st.spinner("Preparing data and generating predictions..."):
                    # Get model's expected columns
                    _, X_template, _, _ = prepare_model_data(df)
                    expected_cols = X_template.columns.tolist()

                    # Build prediction DataFrame
                    pred_df = prepare_batch_for_prediction(user_df, X_template, df)

                    # Generate predictions
                    predictions = model.predict(pred_df)

                    # Add predictions to original data
                    result_df = user_df.copy()
                    result_df["predicted_rating"] = predictions

                    # Add rating category
                    def get_category(rating):
                        if rating >= 4.8:
                            return "Excellent"
                        elif rating >= 4.5:
                            return "Very Good"
                        elif rating >= 4.0:
                            return "Good"
                        else:
                            return "Below Average"

                    result_df["rating_category"] = result_df["predicted_rating"].apply(get_category)

                st.markdown("---")
                st.subheader("Prediction Results")

                # Summary stats
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Predictions", f"{len(predictions):,}")
                with col2:
                    st.metric("Average Predicted", f"{predictions.mean():.2f}")
                with col3:
                    st.metric("Min Predicted", f"{predictions.min():.2f}")
                with col4:
                    st.metric("Max Predicted", f"{predictions.max():.2f}")

                # Distribution chart
                fig_dist = px.histogram(
                    result_df, x="predicted_rating", nbins=30,
                    color_discrete_sequence=["steelblue"],
                    labels={"predicted_rating": "Predicted Rating"},
                )
                fig_dist.update_layout(title="Distribution of Predicted Ratings")
                st.plotly_chart(fig_dist, use_container_width=True)

                # Results table
                st.markdown("**Results Table:**")
                display_cols = ["predicted_rating", "rating_category"]
                # Add original columns that exist
                for col in ["price", "accommodates", "property_type", "host_is_superhost"]:
                    if col in result_df.columns:
                        display_cols.append(col)
                display_cols = [c for c in display_cols if c in result_df.columns]

                st.dataframe(
                    result_df[display_cols + [c for c in result_df.columns if c not in display_cols]].head(100),
                    use_container_width=True,
                )

                # Download button
                csv_output = result_df.to_csv(index=False)
                st.download_button(
                    label="Download Results as CSV",
                    data=csv_output,
                    file_name="predictions.csv",
                    mime="text/csv",
                )

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


def prepare_batch_for_prediction(user_df, X_template, reference_df):
    """Prepare uploaded CSV data for model prediction."""
    # Get template with median values
    template = X_template.median(numeric_only=True).to_dict()

    # Get mode for categorical columns
    for col in X_template.select_dtypes(include=["object"]).columns:
        mode_vals = X_template[col].mode()
        template[col] = mode_vals.iloc[0] if len(mode_vals) > 0 else ""

    # Build prediction rows
    rows = []
    for idx, row in user_df.iterrows():
        pred_row = template.copy()

        # Map user columns to model columns
        # Direct mappings (same name)
        direct_cols = [
            "price", "accommodates", "bedrooms", "beds", "bathrooms",
            "latitude", "longitude", "minimum_nights", "maximum_nights",
            "host_response_rate", "host_experience_days", "instant_bookable",
            "amenities_count", "host_is_superhost", "property_type",
            "has_wifi", "has_heating", "has_workspace", "has_hot_water",
            "has_smoke_alarm", "has_first_aid", "has_hot_tub", "has_gym",
            "host_response_time", "geo_cluster",
        ]

        # Numeric columns that need type conversion
        numeric_cols = [
            "price", "accommodates", "bedrooms", "beds", "bathrooms",
            "latitude", "longitude", "minimum_nights", "maximum_nights",
            "host_response_rate", "host_experience_days", "amenities_count",
        ]

        for col in direct_cols:
            if col in row.index and pd.notna(row[col]):
                val = row[col]
                # Convert numeric columns to float
                if col in numeric_cols:
                    try:
                        val = float(val)
                    except (ValueError, TypeError):
                        val = pred_row[col]  # Keep default
                pred_row[col] = val

        # Handle boolean conversions
        if "host_is_superhost" in row.index:
            val = row["host_is_superhost"]
            if isinstance(val, str):
                pred_row["host_is_superhost"] = 1 if val.lower() in ["true", "t", "yes", "1"] else 0
            else:
                pred_row["host_is_superhost"] = int(val) if pd.notna(val) else 0

        if "instant_bookable" in row.index:
            val = row["instant_bookable"]
            if isinstance(val, str):
                pred_row["instant_bookable"] = 1 if val.lower() in ["true", "t", "yes", "1"] else 0
            else:
                pred_row["instant_bookable"] = int(val) if pd.notna(val) else 0

        # Compute derived features (ensure numeric types)
        price = float(pred_row.get("price", 100) or 100)
        accommodates = max(float(pred_row.get("accommodates", 2) or 2), 1)
        bedrooms = float(pred_row.get("bedrooms", 1) or 1)
        beds = float(pred_row.get("beds", 1) or 1)
        bathrooms = float(pred_row.get("bathrooms", 1) or 1)
        minimum_nights = float(pred_row.get("minimum_nights", 2) or 2)

        pred_row["log_price"] = np.log1p(price)
        pred_row["log_minimum_nights"] = np.log1p(minimum_nights)
        pred_row["price_per_person"] = price / accommodates
        pred_row["bedrooms_per_person"] = bedrooms / accommodates
        pred_row["beds_per_person"] = beds / accommodates
        pred_row["bathrooms_per_person"] = bathrooms / accommodates
        pred_row["min_stay_cost"] = price * minimum_nights

        # Response time score
        host_response_time = pred_row.get("host_response_time", "")
        pred_row["response_time_score"] = RESPONSE_TIME_MAP.get(host_response_time, 0)

        # Safety amenities count
        safety_cols = ["has_smoke_alarm", "has_first_aid"]
        pred_row["safety_amenities_count"] = sum(
            1 for col in safety_cols if pred_row.get(col, 0) == 1
        )

        # Missing indicators (set to 0 for uploaded data)
        for col in ["price_missing", "beds_missing", "bedrooms_missing",
                    "bathrooms_missing", "host_response_rate_missing", "host_acceptance_rate_missing"]:
            pred_row[col] = 0

        rows.append(pred_row)

    # Create DataFrame and ensure correct column order
    result = pd.DataFrame(rows)
    result = result[X_template.columns]
    return result


def page_predict_single(df, model):
    """Handle single listing interactive prediction."""
    st.markdown("Adjust the listing features below to predict its rating.")

    # Get common property types for dropdown
    top_property_types = df["property_type"].value_counts().head(20).index.tolist()

    # Input Form
    st.subheader("Listing Features")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Host Information**")
        is_superhost = st.selectbox("Superhost?", ["Yes", "No"])
        host_response_time = st.selectbox(
            "Host Response Time",
            ["within an hour", "within a few hours", "within a day", "a few days or more", "Unknown"]
        )
        host_response_rate = st.slider("Host Response Rate (%)", 0, 100, 90)
        host_experience_days = st.slider("Host Experience (days)", 0, 6000, 2000)

        st.markdown("**Property Details**")
        property_type = st.selectbox("Property Type", top_property_types)
        price = st.number_input("Price per Night ($)", 10, 10000, 150)
        accommodates = st.slider("Accommodates (guests)", 1, 16, 4)
        bedrooms = st.slider("Bedrooms", 0, 10, 1)
        beds = st.slider("Beds", 0, 20, 2)
        bathrooms = st.slider("Bathrooms", 0.0, 10.0, 1.0, step=0.5)

    with col2:
        st.markdown("**Booking Settings**")
        instant_bookable = st.checkbox("Instant Bookable", value=True)
        minimum_nights = st.slider("Minimum Nights", 1, 365, 2)

        st.markdown("**Amenities**")
        amenities_count = st.slider("Total Amenities Count", 0, 100, 30)
        has_wifi = st.checkbox("WiFi", value=True)
        has_heating = st.checkbox("Heating", value=True)
        has_workspace = st.checkbox("Workspace", value=False)
        has_hot_water = st.checkbox("Hot Water", value=True)
        has_smoke_alarm = st.checkbox("Smoke Alarm", value=True)
        has_first_aid = st.checkbox("First Aid Kit", value=False)
        has_hot_tub = st.checkbox("Hot Tub", value=False)
        has_gym = st.checkbox("Gym", value=False)

        st.markdown("**Location**")
        city = st.selectbox("City", ["LA", "NYC"])

    # Build prediction row
    if st.button("Predict Rating", type="primary"):
        # Prepare model data to get the exact column structure
        _, X_test, _, _ = prepare_model_data(df)

        # Start with median values as template
        template = X_test.median(numeric_only=True).to_dict()

        # Get mode for categorical columns
        for col in X_test.select_dtypes(include=["object"]).columns:
            template[col] = X_test[col].mode().iloc[0] if len(X_test[col].mode()) > 0 else ""

        # Override with user inputs
        template["host_is_superhost"] = 1 if is_superhost == "Yes" else 0
        template["host_response_rate"] = host_response_rate
        template["host_experience_days"] = host_experience_days
        template["price"] = price
        template["log_price"] = np.log1p(price)
        template["minimum_nights"] = minimum_nights
        template["log_minimum_nights"] = np.log1p(minimum_nights)
        template["maximum_nights"] = 365
        template["instant_bookable"] = 1 if instant_bookable else 0
        template["amenities_count"] = amenities_count
        template["has_wifi"] = 1 if has_wifi else 0
        template["has_heating"] = 1 if has_heating else 0
        template["has_workspace"] = 1 if has_workspace else 0
        template["has_hot_water"] = 1 if has_hot_water else 0
        template["has_smoke_alarm"] = 1 if has_smoke_alarm else 0
        template["has_first_aid"] = 1 if has_first_aid else 0
        template["has_hot_tub"] = 1 if has_hot_tub else 0
        template["has_gym"] = 1 if has_gym else 0
        template["property_type"] = property_type
        template["geo_cluster"] = "20"  # Common cluster

        # Compute derived features
        safe_accommodates = max(accommodates, 1)
        template["price_per_person"] = price / safe_accommodates
        template["bedrooms_per_person"] = bedrooms / safe_accommodates
        template["beds_per_person"] = beds / safe_accommodates
        template["bathrooms_per_person"] = bathrooms / safe_accommodates
        template["min_stay_cost"] = price * minimum_nights

        # Response time score
        template["response_time_score"] = RESPONSE_TIME_MAP.get(host_response_time, 0)
        template["host_response_time"] = host_response_time if host_response_time != "Unknown" else np.nan

        # Location
        template["latitude"] = CITY_COORDS[city]["lat"]
        template["longitude"] = CITY_COORDS[city]["lon"]

        # Safety amenities count
        safety_sum = sum([
            1 if has_smoke_alarm else 0,
            1 if has_first_aid else 0,
        ])
        template["safety_amenities_count"] = safety_sum

        # Missing indicators (all 0 for user input)
        for col in ["price_missing", "beds_missing", "bedrooms_missing",
                    "bathrooms_missing", "host_response_rate_missing", "host_acceptance_rate_missing"]:
            if col in template:
                template[col] = 0

        # Create DataFrame with correct columns
        pred_row = pd.DataFrame([template])
        pred_row = pred_row[X_test.columns]  # Ensure correct column order

        # Predict
        prediction = model.predict(pred_row)[0]

        # Display result
        st.markdown("---")
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        avg_rating = df[TARGET_COL].mean()
        delta = prediction - avg_rating

        with col1:
            st.metric(
                "Predicted Rating",
                f"{prediction:.2f}",
                delta=f"{delta:+.2f} vs avg ({avg_rating:.2f})"
            )

        with col2:
            # Rating category
            if prediction >= 4.8:
                rating_cat = "Excellent"
                color = "green"
            elif prediction >= 4.5:
                rating_cat = "Very Good"
                color = "blue"
            elif prediction >= 4.0:
                rating_cat = "Good"
                color = "orange"
            else:
                rating_cat = "Below Average"
                color = "red"
            st.markdown(f"**Category:** :{color}[{rating_cat}]")

        with col3:
            # Visual gauge
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=prediction,
                domain={"x": [0, 1], "y": [0, 1]},
                gauge={
                    "axis": {"range": [1, 5]},
                    "bar": {"color": "steelblue"},
                    "steps": [
                        {"range": [1, 3], "color": "#FFCDD2"},
                        {"range": [3, 4], "color": "#FFF9C4"},
                        {"range": [4, 4.5], "color": "#C8E6C9"},
                        {"range": [4.5, 5], "color": "#81C784"},
                    ],
                },
            ))
            fig_gauge.update_layout(height=200, margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge, use_container_width=True)

        # Top factors
        st.markdown("---")
        st.subheader("Key Factors Affecting Ratings")
        st.markdown("Based on feature importance, these factors matter most:")

        metrics_data = load_metrics()
        top_features = list(metrics_data["feature_importance"].items())[:10]

        factors_df = pd.DataFrame(top_features, columns=["Feature", "Importance"])
        factors_df["Feature"] = factors_df["Feature"].str.replace("num__", "").str.replace("target__", "")

        fig_factors = px.bar(
            factors_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale="Blues",
        )
        fig_factors.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_factors, use_container_width=True)


# =============================================================================
# MAIN
# =============================================================================
def main():
    st.set_page_config(
        page_title="Airbnb Rating Predictor",
        page_icon="🏠",
        layout="wide",
    )

    st.sidebar.title("🏠 Airbnb Capstone")
    st.sidebar.markdown("Predicting Review Scores Rating")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Navigate",
        ["EDA Explorer", "Model Performance", "Predict a Rating"],
    )

    if page == "EDA Explorer":
        page_eda()
    elif page == "Model Performance":
        page_model()
    elif page == "Predict a Rating":
        page_predict()

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Data:** Inside Airbnb (LA + NYC)")
    st.sidebar.markdown("**Model:** XGBoost Regression")
    st.sidebar.markdown("**Features:** 37 (lean set)")


if __name__ == "__main__":
    main()
