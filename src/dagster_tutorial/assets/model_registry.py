from dagster import asset, Output, MetadataValue as MV, AssetExecutionContext, AssetKey
import pandas as pd
import mlflow
import os
from dagster_tutorial.assets.model_visualization import afib_model_visualization
from dagster_tutorial.assets.training_model import HYPERPARAM_CONFIGS
import io


@asset(
    compute_kind="mlflow",
    required_resource_keys={"mlflow", "gcs"}
)
def register_best_model_to_mlflow(
    context: AssetExecutionContext,
    afib_model_visualization: dict,
) -> Output[dict]:
    """
    Register the best model to MLflow.
    Uses comparison data and best model selection from the afib_model_visualization asset.
    """
    context.log.info("Received model comparison data from visualization asset...")
    registry_model_name = "afib_xgboost_classifier"
    
    # Extract the comparison DataFrame and best model info from visualization asset
    df = afib_model_visualization["comparison_df"]
    best_partition = afib_model_visualization["best_partition"]
    
    if df.empty or best_partition is None:
        context.log.error("No valid model results found! Cannot register to MLflow.")
        return Output(
            value={"status": "failed", "reason": "No model results available"},
            metadata={"status": MV.text("No models to register")}
        )
    
    # Get the best model configuration from the DataFrame
    best_config = df[df["partition"] == best_partition].iloc[0]
    
    # Build model path for the best model
    model_paths = {pk: f"/tmp/dagster_models/model_{pk}.pkl" for pk in HYPERPARAM_CONFIGS.keys()}
    best_model_path = model_paths.get(best_partition)
    
    context.log.info(f"Best model found: {best_partition}")
    context.log.info(f"   Accuracy: {best_config['accuracy']:.4f}")
    context.log.info(f"   PR-AUC: {best_config['pr_auc']:.4f}" if pd.notna(best_config['pr_auc']) else "   PR-AUC: N/A")
    context.log.info(f"   ROC-AUC: {best_config['roc_auc']:.4f}" if pd.notna(best_config['roc_auc']) else "   ROC-AUC: N/A")
    context.log.info(f"   Model path: {best_model_path}")
    
    gcs = context.resources.gcs
    gcs_data_path = os.getenv("GCS_DATA_PATH", "dagster-ml/afib_raw_data.parquet")
    gcs_dataset_path = f"gs://{gcs.bucket_name}/{gcs_data_path}"
    
    # Check if model file exists
    if not os.path.exists(best_model_path):
        context.log.error(f"Model file not found at {best_model_path}")
        return Output(
            value={"status": "failed", "reason": f"Model file not found: {best_model_path}"},
            metadata={"status": MV.text("Model file not found")}
        )
    
    # Load the best model
    import pickle
    with open(best_model_path, 'rb') as f:
        best_model = pickle.load(f)
    
    context.log.info("Best model loaded successfully")
    
    # Get MLflow resource
    mlf = context.resources.mlflow
    
    # Set a descriptive run name
    custom_run_name = f"register_best_model_{best_partition}_acc{best_config['accuracy']:.3f}"
    mlf.set_tag("mlflow.runName", custom_run_name)
    context.log.info(f"MLflow run name set to: {custom_run_name}")
    
    # Log hyperparameters
    mlf.log_param("max_depth", best_config["max_depth"])
    mlf.log_param("learning_rate", best_config["learning_rate"])
    mlf.log_param("partition_key", best_partition)
    
    # Log metrics
    mlf.log_metric("accuracy", best_config["accuracy"])
    if pd.notna(best_config["pr_auc"]):
        mlf.log_metric("pr_auc", best_config["pr_auc"])
    if pd.notna(best_config["roc_auc"]):
        mlf.log_metric("roc_auc", best_config["roc_auc"])
    
    # Log dataset reference information as parameters AND as a dataset
    mlf.log_param("dataset_location", gcs_dataset_path)
    mlf.log_param("dataset_bucket", gcs.bucket_name)
    mlf.log_param("dataset_path", gcs_data_path)
    
    # Add tags for better organization
    mlf.set_tag("model_type", "xgboost")
    mlf.set_tag("task", "afib_detection")
    mlf.set_tag("best_model", "true")
    mlf.set_tag("dagster_partition", best_partition)
    mlf.set_tag("source", "dagster_pipeline")
    mlf.set_tag("dagster_run_id", context.run_id)
    
    # IMPORTANT: Create MLflow dataset reference
    # Use native MLflow API (not the dagster wrapper) to log dataset inputs
    context.log.info("Creating MLflow dataset reference...")
    
    # Load the actual dataset from GCS to log it properly
    dataset_logged = False
    actual_dataset = None  # Initialize to None
    active_run = mlflow.active_run()
    
    try:
        # Use Google Cloud Storage SDK (same as data_loading.py)
        bucket = gcs.get_bucket()
        blob = bucket.blob(gcs_data_path)
        
        context.log.info(f"Reading dataset from: {gcs_dataset_path}")
        
        # Download the parquet file as bytes and read into DataFrame
        parquet_bytes = blob.download_as_bytes()
        actual_dataset = pd.read_parquet(io.BytesIO(parquet_bytes))
        
        context.log.info(f"Dataset loaded: {len(actual_dataset)} rows, {len(actual_dataset.columns)} columns")
        context.log.info(f"Dataset columns: {list(actual_dataset.columns)}")
        
        # Get the active run to verify we're in a run context
        if active_run:
            context.log.info(f"Active MLflow run ID: {active_run.info.run_id}")
        else:
            context.log.warning("No active MLflow run found!")
        
        # Log the actual dataset with MLflow's native dataset API
        # Use mlflow directly (not the dagster wrapper) for dataset logging
        dataset = mlflow.data.from_pandas(
            actual_dataset,
            source=gcs_dataset_path,
            name="afib_training_data",
            targets="avg_afib_prob"  # Specify the target column
        )
        
        context.log.info(f"Dataset object created: {dataset}")
        
        # Use native MLflow API to log input
        mlflow.log_input(dataset, context="training")
        dataset_logged = True
        
        context.log.info(f"Dataset properly logged to MLflow with {len(actual_dataset)} rows and full schema")
        context.log.info(f"   Dataset name: afib_training_data")
        context.log.info(f"   Dataset source: {gcs_dataset_path}")
        context.log.info(f"   Target column: avg_afib_prob")
        
    except Exception as e:
        context.log.error(f"Failed to load/log dataset from GCS: {e}")
        context.log.error(f"   Error type: {type(e).__name__}")
        import traceback
        context.log.error(f"   Traceback: {traceback.format_exc()}")
        
        context.log.info("Attempting fallback to URI-only dataset reference...")
        
        # Fallback: Log just the URI reference
        try:
            dataset = mlflow.data.from_pandas(
                pd.DataFrame({"dataset_uri": [gcs_dataset_path]}),
                source=gcs_dataset_path,
                name="afib_training_data"
            )
            mlflow.log_input(dataset, context="training")
            dataset_logged = True
            context.log.info("Logged URI-only dataset reference as fallback")
        except Exception as e2:
            context.log.error(f"Fallback also failed: {e2}")
            dataset_logged = False
    
    if dataset_logged:
        context.log.info(f"Dataset logging completed successfully for: {gcs_dataset_path}")
    else:
        context.log.warning(f"Dataset logging failed - model will be registered without dataset reference")
    
    # Log model to MLflow with dataset reference
    context.log.info("Logging model to MLflow...")
    
    # Create input example from the dataset for model signature
    input_example = actual_dataset.drop(columns=['avg_afib_prob']).head(5) if dataset_logged and actual_dataset is not None else None
    
    # Create datasets list to associate with the model (for UI visibility)
    datasets = None
    if dataset_logged and actual_dataset is not None:
        try:
            # Create the dataset object for model metadata
            model_dataset = mlflow.data.from_pandas(
                actual_dataset,
                source=gcs_dataset_path,
                name="afib_training_data",
                targets="avg_afib_prob"
            )
            datasets = [model_dataset]
            context.log.info(f"Dataset will be associated with model: {gcs_dataset_path}")
        except Exception as e:
            context.log.warning(f"Could not create dataset for model association: {e}")
    
    mlf.xgboost.log_model(
        best_model,
        artifact_path="model",
        registered_model_name=registry_model_name,
        input_example=input_example,  # This adds dataset schema to the model
        metadata={"dataset_source": gcs_dataset_path} if dataset_logged else None  # Add dataset metadata
    )
    context.log.info(f"Model logged to MLflow registry: {registry_model_name}")
    
    # Add dataset metadata to the registered model version
    try:
        # Get the latest version that was just created
        client = mlflow.tracking.MlflowClient()
        latest_versions = client.get_latest_versions(registry_model_name, stages=["None"])
        
        if latest_versions:
            latest_version = latest_versions[0]
            model_version = latest_version.version
            
            context.log.info(f"Adding dataset metadata to model version {model_version}...")
            
            # Set dataset tags on the model version
            client.set_model_version_tag(
                name=registry_model_name,
                version=model_version,
                key="dataset_source",
                value=gcs_dataset_path
            )
            client.set_model_version_tag(
                name=registry_model_name,
                version=model_version,
                key="dataset_name",
                value="afib_training_data"
            )
            client.set_model_version_tag(
                name=registry_model_name,
                version=model_version,
                key="dataset_rows",
                value=str(len(actual_dataset)) if dataset_logged else "unknown"
            )
            client.set_model_version_tag(
                name=registry_model_name,
                version=model_version,
                key="dataset_columns",
                value=str(len(actual_dataset.columns)) if dataset_logged else "unknown"
            )
            client.set_model_version_tag(
                name=registry_model_name,
                version=model_version,
                key="training_run_id",
                value=active_run.info.run_id if active_run else "unknown"
            )
            
            # Update model version description with dataset info
            dataset_rows_str = f"{len(actual_dataset):,}" if dataset_logged and actual_dataset is not None else 'N/A'
            dataset_cols_str = str(len(actual_dataset.columns)) if dataset_logged and actual_dataset is not None else 'N/A'
            run_id_str = active_run.info.run_id if active_run else 'N/A'
            
            dataset_description = f"""
This model was trained on the AFib detection dataset.

**Dataset Information:**
- Source: `{gcs_dataset_path}`
- Rows: {dataset_rows_str}
- Columns: {dataset_cols_str}
- Target: avg_afib_prob
- Features: gross_activity, activity_confidence, and engineered features

**Training Run:** {run_id_str}

See the training run for full dataset details and lineage.
"""
            
            client.update_model_version(
                name=registry_model_name,
                version=model_version,
                description=dataset_description
            )
            
            context.log.info(f"Dataset metadata added to model version {model_version}")
        else:
            context.log.warning("Could not find latest model version to add dataset metadata")
            
    except Exception as e:
        context.log.warning(f"Failed to add dataset metadata to model version: {e}")
    
    # Format metric values for display
    pr_auc_str = f"{best_config['pr_auc']:.4f}" if pd.notna(best_config['pr_auc']) else 'N/A'
    roc_auc_str = f"{best_config['roc_auc']:.4f}" if pd.notna(best_config['roc_auc']) else 'N/A'
    
    # Create detailed dataset info artifact
    dataset_info = f"""Dataset Location: {gcs_dataset_path}
Bucket: {gcs.bucket_name}
Path: {gcs_data_path}

This model was trained on AFib detection data.
Best hyperparameters:
  - max_depth: {best_config['max_depth']}
  - learning_rate: {best_config['learning_rate']}

Metrics:
  - Accuracy: {best_config['accuracy']:.4f}
  - PR-AUC: {pr_auc_str}
  - ROC-AUC: {roc_auc_str}

Dataset Schema:
  - ext_account: Patient identifier
  - collected_at: Timestamp of measurement
  - avg_afib_prob: Target variable (AFib probability)
  - afib_confidence: Confidence score for AFib measurement
  - gross_activity: Patient activity level
  - activity_confidence: Confidence score for activity measurement
"""
    with open("/tmp/dataset_info.txt", "w") as f:
        f.write(dataset_info)
    mlf.log_artifact("/tmp/dataset_info.txt", "dataset")
    
    # Get active run information
    active_run = mlf.active_run()
    run_id = active_run.info.run_id if active_run else "unknown"
    
    context.log.info(f"MLflow run created: {run_id}")
    context.log.info(f"   Registered Model: {registry_model_name}")
    context.log.info(f"   Dataset Reference: {gcs_dataset_path}")
    
    # Prepare result
    result = {
        "status": "success",
        "best_partition": best_partition,
        "best_accuracy": float(best_config["accuracy"]),
        "best_pr_auc": float(best_config["pr_auc"]) if pd.notna(best_config["pr_auc"]) else None,
        "best_roc_auc": float(best_config["roc_auc"]) if pd.notna(best_config["roc_auc"]) else None,
        "hyperparameters": {
            "max_depth": int(best_config["max_depth"]),
            "learning_rate": float(best_config["learning_rate"]),
        },
        "mlflow_run_id": run_id,
        "mlflow_model_name": registry_model_name,
        "dataset_location": gcs_dataset_path,
    }
    
    # Create summary table
    summary_md = f"""
## Best Model Registered to MLflow

**Partition:** {best_partition}

**Metrics:**
- Accuracy: {best_config['accuracy']:.4f}
- PR-AUC: {pr_auc_str}
- ROC-AUC: {roc_auc_str}

**Hyperparameters:**
- max_depth: {best_config['max_depth']}
- learning_rate: {best_config['learning_rate']}

**MLflow Details:**
- Run ID: `{run_id}`
- Registered Model: `{registry_model_name}`
- Experiment: `afib_detection`

**Dataset Reference:**
- Location: `{gcs_dataset_path}`
- Bucket: `{gcs.bucket_name}`
- Logged as MLflow Dataset: Yes

**All Models Comparison:**

{df[['partition', 'accuracy', 'pr_auc', 'roc_auc', 'max_depth', 'learning_rate']].to_markdown(index=False)}
"""
    
    return Output(
        value=result,
        metadata={
            "summary": MV.md(summary_md),
            "mlflow_run_id": str(run_id),
            "mlflow_model_name": str(registry_model_name),
            "best_partition": str(best_partition),
            "best_accuracy": float(best_config["accuracy"]),
            "dataset_location": str(gcs_dataset_path),
        }
    )
