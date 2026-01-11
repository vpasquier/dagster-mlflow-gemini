# AFib Detection Pipeline with Dagster + MLflow

> Trains XGBoost classifiers to detect atrial fibrillation from activity data with automated hyperparameter tuning and
> MLflow tracking.

## Repository Structure

```
├── src/dagster_tutorial/          # Main Dagster pipeline code
│   ├── assets/                    # Dagster asset definitions
│   │   ├── data_loading.py       # Load AFib data from BigQuery/GCS
│   │   ├── feature_engineering.py # Feature extraction & train/test split
│   │   ├── training_model.py     # XGBoost model training (partitioned)
│   │   ├── model_registry.py     # MLflow model registration & best model selection
│   │   └── gemini_analysis.py    # AI-powered insights with Gemini
│   ├── resources/                 # Dagster resources (GCS, MLflow configs)
│   └── definitions.py             # Main Dagster definitions
└── pyproject.toml                 # Python dependencies & project config
```

## Pipeline Architecture

```
┌──────────────────┐
│   BigQuery /     │
│   GCS Bucket     │
│  (Data Source)   │
└────────┬─────────┘
         │
         ▼
┌─────────────────────────────┐
│    afib_raw_data            │
│    (data_loading.py)        │
│  Load & cache AFib dataset  │
│  from BigQuery/GCS          │
└─────────────┬───────────────┘
              │
              │ GCS: gs://<YOUR-BUCKET>/
              │      dagster-ml/afib_raw_data.parquet
              │
              ▼
┌─────────────────────────────┐
│    afib_features            │
│ (feature_engineering.py)    │
│  Engineer rolling features  │
│  & split train/test         │
└─────────────┬───────────────┘
              │
              │ Features: activity_trend_5min,
              │          activity_std_5min, etc.
              │
              ▼
┌─────────────────────────────────────────────────────┐
│    afib_model_training (Partitioned)                │
│    (training_model.py)                              │
├──────────────────┬──────────────────┬───────────────┤
│  hp_0_maxdepth3  │  hp_1_maxdepth5  │  hp_2_maxdepth7 │
│  _lr0.1          │  _lr0.01         │  _lr0.2         │
└────────┬─────────┴────────┬─────────┴────────┬──────┘
         │                  │                  │
         │ Train 3 XGBoost models with different hyperparams
         │ Log each run to MLflow (metrics, params, artifacts)
         │ Save models to /tmp/dagster_models/
         │
         └──────────┬───────┴─────────┬────────┘
                    │                 │
                    ▼                 ▼
         ┌────────────────────────────────────────┐
         │   afib_model_visualization             │
         │   (model_visualization.py)             │
         │  Compare all models & create plots     │
         └────────────────┬───────────────────────┘
                          │
                          │ Queries training materializations
                          │
         ┌────────────────┴───────────────────────┐
         │                                        │
         ▼                                        ▼
┌─────────────────────┐           ┌────────────────────────┐
│register_best_model_ │           │  afib_gemini_analysis  │
│to_mlflow            │           │  (gemini_analysis.py)  │
│(model_registry.py)  │           │  AI-powered insights   │
│  Select best model  │           │  using Gemini API      │
│  by accuracy/PR-AUC │           └────────────────────────┘
│  Register to MLflow │
│  Registry as:       │
│  afib_xgboost_      │
│  classifier         │
└─────────────────────┘
```

## Quick Start

```bash
# 1. Start MLflow
mlflow server --host 127.0.0.1 --port 5000

# 2. Run Dagster
dagster dev

# 3. Open tools
open http://127.0.0.1:5000
open http://localhost:3000

Materialize All in Dagster and see results in each assets + mlflow!
```

## What Gets Logged to MLflow

- **3 Training Runs**: Each hyperparameter configuration with metrics
- **1 Registry Run**: Best model (highest accuracy) with GCS dataset reference
- **Registered Model**: `afib_xgboost_classifier` ready for deployment
- **Gemini Analysis**: AI-generated insights on model performance (separate from MLflow)
