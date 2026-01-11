from dagster import asset, Output, MetadataValue as MV, AssetExecutionContext, AssetKey
import pandas as pd
import mlflow
import mlflow.xgboost
import os
from dagster_mlflow import mlflow_tracking
from dagster_tutorial.assets.model_visualization import afib_model_visualization
from dagster_tutorial.assets.training_model import HYPERPARAM_CONFIGS
from dagster_tutorial.resources.resources import GCSResource
import io


@asset(
    deps=[afib_model_visualization],
    compute_kind="mlflow",
    required_resource_keys={"mlflow", "gcs"}
)
def register_best_model_to_mlflow(
    context: AssetExecutionContext,
) -> Output[dict]:
    """
    Find the best model across all hyperparameter configurations and register it to MLflow.
    Uses accuracy as the primary metric and properly references the GCS dataset location.
    This is a non-partitioned asset that runs after all partitioned training is complete.
    
    Uses the official dagster-mlflow integration for proper MLflow tracking.
    """
    context.log.info("Collecting all model results to find best model...")
    
    # The mlflow resource automatically handles tracking URI and experiment setup
    registry_model_name = "afib_xgboost_classifier"
    
    # Collect all metrics from all partitions
    comparison_data = []
    model_paths = {}
    
    for pk in HYPERPARAM_CONFIGS.keys():
        # Get the training asset materializations to fetch the model
        training_asset_key = AssetKey(["afib_model_training"])
        records = context.instance.fetch_materializations(
            training_asset_key,
            limit=10
        )
        
        partition_materialization = None
        if records.records:
            for record in records.records:
                if record.event_log_entry.dagster_event.partition == pk:
                    partition_materialization = record.asset_materialization
                    break
        
        if partition_materialization:
            metadata = partition_materialization.metadata
            
            # Extract metrics
            roc_auc_meta = metadata.get("roc_auc")
            pr_auc_meta = metadata.get("pr_auc")
            accuracy_meta = metadata.get("accuracy")
            
            partition_roc_auc = roc_auc_meta.value if roc_auc_meta else None
            partition_pr_auc = pr_auc_meta.value if pr_auc_meta else None
            partition_accuracy = accuracy_meta.value if accuracy_meta else None
            
            # Handle NaN values
            if partition_roc_auc is not None and (partition_roc_auc != partition_roc_auc):
                partition_roc_auc = None
            if partition_pr_auc is not None and (partition_pr_auc != partition_pr_auc):
                partition_pr_auc = None
            
            hp = HYPERPARAM_CONFIGS[pk]
            
            comparison_data.append({
                "partition": pk,
                "max_depth": hp["max_depth"],
                "learning_rate": hp["learning_rate"],
                "roc_auc": partition_roc_auc,
                "pr_auc": partition_pr_auc,
                "accuracy": partition_accuracy,
            })
            
            # Store model path
            model_paths[pk] = f"/tmp/dagster_models/model_{pk}.pkl"
    
    if not comparison_data:
        context.log.error("No model results found! Cannot register to MLflow.")
        return Output(
            value={"status": "failed", "reason": "No model results available"},
            metadata={"status": MV.text("No models to register")}
        )
    
    df = pd.DataFrame(comparison_data)
    
    # Find best model - prioritize accuracy as primary metric
    # Filter out models with invalid metrics
    df_valid = df[df["accuracy"].notna()].copy()
    
    if df_valid.empty:
        context.log.error("No valid models found (all have NaN accuracy)!")
        return Output(
            value={"status": "failed", "reason": "All models have invalid metrics"},
            metadata={"status": MV.text("No valid models")}
        )
    
    # Sort by accuracy (primary), then by PR-AUC (secondary for imbalanced data)
    df_sorted = df_valid.sort_values(
        by=["accuracy", "pr_auc"],
        ascending=[False, False],
        na_position='last'
    )
    
    best_config = df_sorted.iloc[0]
    best_partition = best_config["partition"]
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
        
        context.log.info(f"✅ Dataset properly logged to MLflow with {len(actual_dataset)} rows and full schema")
        context.log.info(f"   Dataset name: afib_training_data")
        context.log.info(f"   Dataset source: {gcs_dataset_path}")
        context.log.info(f"   Target column: avg_afib_prob")
        
    except Exception as e:
        context.log.error(f"❌ Failed to load/log dataset from GCS: {e}")
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
            context.log.info("⚠️  Logged URI-only dataset reference as fallback")
        except Exception as e2:
            context.log.error(f"❌ Fallback also failed: {e2}")
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
            
            context.log.info(f"✅ Dataset metadata added to model version {model_version}")
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
- Logged as MLflow Dataset: ✅

**All Models Comparison:**

{df_sorted[['partition', 'accuracy', 'pr_auc', 'roc_auc', 'max_depth', 'learning_rate']].to_markdown(index=False)}
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
