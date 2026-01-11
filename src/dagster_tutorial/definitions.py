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
    hyperparams_partitions,
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

train_all_models_job = define_asset_job(
    name="train_all_models_parallel",
    selection=AssetSelection.all(),
    description="Full pipeline: load data, train models in parallel, compare results",
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
    jobs=[train_all_models_job],
    resources=resources_dict,
)

if __name__ == "__main__":
    print("Training models with different hyperparameters...\n")
    for partition_key in HYPERPARAM_CONFIGS.keys():
        print(f"Training partition: {partition_key}")
        materialize(assets=all_assets, resources=resources_dict, partition_key=partition_key)
    print("\nAll models trained successfully!")
