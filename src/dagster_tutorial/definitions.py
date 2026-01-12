from dagster import (
    Definitions,
    AssetSelection,
    define_asset_job,
    materialize,
    EnvVar,
)
from dagster_tutorial.assets.data_loading import afib_raw_data
from dagster_tutorial.assets.feature_engineering import afib_features
from dagster_tutorial.assets.training_model import (
    afib_model_training,
    HYPERPARAM_CONFIGS,
)
from dagster_tutorial.assets.model_visualization import afib_model_visualization
from dagster_tutorial.assets.gemini_analysis import afib_gemini_analysis
from dagster_tutorial.assets.model_registry import register_best_model_to_mlflow
from dagster_tutorial.resources.resources import BigQueryResource, GCSResource
from dagster_gemini import GeminiResource

all_assets = [
    afib_raw_data,
    afib_features,
    afib_model_training,
    afib_model_visualization,
    afib_gemini_analysis,
    register_best_model_to_mlflow,
]

# Job 1: Data preparation (non-partitioned assets)
prepare_data_job = define_asset_job(
    name="prepare_data",
    selection=AssetSelection.assets(afib_raw_data, afib_features),
    description="Load and prepare data for model training",
)

# Job 2: Model training (partitioned asset - use backfill in UI for parallel execution)
train_models_job = define_asset_job(
    name="train_models",
    selection=AssetSelection.assets(afib_model_training),
    description="Train models with different hyperparameters (run as backfill for all partitions)",
)

# Job 3: Model evaluation and registration (non-partitioned downstream assets)
evaluate_models_job = define_asset_job(
    name="evaluate_models",
    selection=AssetSelection.assets(
        afib_model_visualization,
        afib_gemini_analysis,
        register_best_model_to_mlflow
    ),
    description="Compare models, analyze with Gemini, and register best model to MLflow",
)

from dagster_mlflow import mlflow_tracking

resources_dict = {
    "bigquery": BigQueryResource(),
    "gcs": GCSResource(),
    "mlflow": mlflow_tracking.configured({
        "experiment_name": "afib_detection",
        "mlflow_tracking_uri": "http://127.0.0.1:5000",
    }),
    "gemini": GeminiResource(
        api_key=EnvVar("GEMINI_API_KEY"),
        generative_model_name="gemini-2.5-flash",
    ),
}

defs = Definitions(
    assets=all_assets,
    jobs=[prepare_data_job, train_models_job, evaluate_models_job],
    resources=resources_dict,
)

if __name__ == "__main__":
    """
    This demonstrates the proper workflow for the partitioned pipeline:
    
    Step 1: Prepare data (runs once)
    Step 2: Train models (runs 3 times in parallel - one per partition)
    Step 3: Evaluate models (runs once after all training completes)
    
    For production, use the Dagster UI to:
    1. Run the 'prepare_data' job
    2. Launch a BACKFILL for 'train_models' job (materializes all partitions in parallel)
    3. Run the 'evaluate_models' job
    """
    print("=" * 70)
    print("AFib Detection Pipeline - Full Workflow")
    print("=" * 70)
    
    # Step 1: Prepare data
    print("\n[1/3] Preparing data (non-partitioned)...")
    materialize(
        assets=[afib_raw_data, afib_features],
        resources=resources_dict
    )
    print("✓ Data preparation complete\n")
    
    # Step 2: Train models with different hyperparameters (partitioned)
    print("[2/3] Training models with different hyperparameters (partitioned)...")
    for partition_key in HYPERPARAM_CONFIGS.keys():
        print(f"  → Training partition: {partition_key}")
        materialize(
            assets=[afib_model_training],
            resources=resources_dict,
            partition_key=partition_key
        )
    print("✓ All model training complete\n")
    
    # Step 3: Evaluate and register best model (non-partitioned)
    print("[3/3] Evaluating models and registering best model (non-partitioned)...")
    materialize(
        assets=[afib_model_visualization, afib_gemini_analysis, register_best_model_to_mlflow],
        resources=resources_dict
    )
    print("✓ Model evaluation and registration complete\n")
    
    print("=" * 70)
    print("Pipeline execution complete!")
    print("=" * 70)
