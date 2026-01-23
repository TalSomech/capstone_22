# End-to-End Airbnb Machine Learning Pipeline

This repository contains a **complete, end-to-end machine learning pipeline** for training and running predictions on Airbnb listing data.  
The project is designed to reflect **production-grade ML practices**, where each stage of the pipeline is clearly separated, reproducible, and driven by explicit artifacts.

The pipeline starts from raw CSV files and ends with model predictions and summarized results.

---

## What This Repository Does (High-Level)

The workflow implemented in this repository follows this sequence:

1. Take raw Airbnb listing data from multiple cities
2. Clean and merge the datasets into a single processed dataset
3. Train a machine learning model using a robust preprocessing + modeling pipeline
4. Save the trained model as a reusable artifact
5. Use the trained model to generate predictions on data
6. Summarize prediction outputs into final results

Each step is implemented as a **standalone Python script** that can be run independently.

---

## Repository Structure

```text
.
├── data/
│   ├── raw/
│   │   ├── listingsLA.csv           # Raw Los Angeles Airbnb data
│   │   └── listingsNYC.csv          # Raw New York City Airbnb data
│   │
│   └── processed/
│       ├── listings_combined_clean.csv
│       └── cleaning_summary.json
│
├── models/
│   └── model.joblib                 # Trained ML pipeline
│
├── results/
│   ├── metrics.json                 # Training evaluation metrics
│   ├── predictions.csv              # Model predictions
│   └── prediction_summary.json      # Aggregated prediction results
│
├── preprocess.py                    # Data preprocessing script
├── train.py                         # Model training script
├── predict.py                       # Batch prediction script
├── results.py                       # Prediction aggregation script
├── requirements.txt
└── README.md
