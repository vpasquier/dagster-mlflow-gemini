from dagster import asset, Output, MetadataValue as MV, StaticPartitionsDefinition
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import pickle
import os

# Define hyperparameter configurations
HYPERPARAM_CONFIGS = {
    "hp_0_maxdepth3_lr0.1": {"max_depth": 3, "learning_rate": 0.1},
    "hp_1_maxdepth5_lr0.01": {"max_depth": 5, "learning_rate": 0.01},
    "hp_2_maxdepth7_lr0.2": {"max_depth": 7, "learning_rate": 0.2},
}

# Create static partitions for hyperparameters
hyperparams_partitions = StaticPartitionsDefinition(
    list(HYPERPARAM_CONFIGS.keys())
)


@asset(partitions_def=hyperparams_partitions, required_resource_keys={"mlflow"})
def afib_model_training(
    context,
    afib_features: dict,
) -> Output[dict]:
    """Train XGBoost model with given hyperparameters from partition.
    
    Each training run is logged to MLflow with:
    - Hyperparameters
    - Metrics (accuracy, PR-AUC, ROC-AUC)
    - Feature importance plot
    - Model artifact (saved locally for best model selection)
    """
    # Get hyperparameters based on partition key
    partition_key = context.partition_key
    hyperparams = HYPERPARAM_CONFIGS[partition_key]
    
    context.log.info(f"Training model with partition: {partition_key}")
    context.log.info(f"Hyperparameters: {hyperparams}")
    
    X_train, X_test, y_train, y_test = (
        afib_features["X_train"],
        afib_features["X_test"],
        afib_features["y_train"],
        afib_features["y_test"],
    )
    
    # Calculate scale_pos_weight to handle class imbalance
    n_negative = (y_train == 0).sum()
    n_positive = (y_train == 1).sum()
    scale_pos_weight = n_negative / n_positive if n_positive > 0 else 1
    
    context.log.info(f"Training class distribution - AFib: {n_positive}, No AFib: {n_negative}")
    context.log.info(f"Using scale_pos_weight: {scale_pos_weight:.2f} to handle class imbalance")

    model = xgb.XGBClassifier(
        **hyperparams,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric="logloss",
        use_label_encoder=False,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    accuracy = accuracy_score(y_test, y_pred)
    
    # Check if we can compute ROC AUC (need at least 2 classes)
    unique_classes = len(set(y_test))
    if unique_classes < 2:
        context.log.warning(f"Only {unique_classes} class(es) in test set. ROC AUC is not defined. Setting to NaN.")
        roc_auc = float('nan')
        pr_auc = float('nan')
    else:
        roc_auc = roc_auc_score(y_test, y_proba)
        # PR-AUC (Precision-Recall AUC) - better for imbalanced datasets
        pr_auc = average_precision_score(y_test, y_proba)
    
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    # Log to MLflow using the dagster-mlflow resource
    # The resource automatically manages the run lifecycle
    mlf = context.resources.mlflow
    
    # Log hyperparameters
    mlf.log_param("max_depth", hyperparams["max_depth"])
    mlf.log_param("learning_rate", hyperparams["learning_rate"])
    mlf.log_param("scale_pos_weight", scale_pos_weight)
    mlf.log_param("random_state", 42)
    mlf.log_param("partition_key", partition_key)
    
    # Log training data info
    mlf.log_param("train_size", len(X_train))
    mlf.log_param("test_size", len(X_test))
    mlf.log_param("n_features", X_train.shape[1])
    mlf.log_param("afib_cases_train", int(n_positive))
    mlf.log_param("no_afib_cases_train", int(n_negative))
    mlf.log_metric("accuracy", accuracy)
    if not (isinstance(roc_auc, float) and roc_auc != roc_auc):  # Check for NaN
        mlf.log_metric("roc_auc", roc_auc)
        mlf.log_metric("pr_auc", pr_auc)
    
    # Log run
    run_name = f"train_{partition_key}_acc{accuracy:.3f}"
    mlf.set_tag("mlflow.runName", run_name)
    context.log.info(f"MLflow run name set to: {run_name}")
    
    # Log classification report metrics
    for class_label in ['0', '1']:
        if class_label in report:
            class_name = "no_afib" if class_label == '0' else "afib"
            mlf.log_metric(f"{class_name}_precision", report[class_label]['precision'])
            mlf.log_metric(f"{class_name}_recall", report[class_label]['recall'])
            mlf.log_metric(f"{class_name}_f1", report[class_label]['f1-score'])
    
    # Add tags
    mlf.set_tag("model_type", "xgboost")
    mlf.set_tag("task", "afib_detection")
    mlf.set_tag("stage", "training")
    mlf.set_tag("dagster_partition", partition_key)
    mlf.set_tag("dagster_run_id", context.run_id)
    
    # Evaluate different thresholds
    thresholds_to_test = [0.1, 0.2, 0.3, 0.4, 0.5]
    threshold_results = []
    for threshold in thresholds_to_test:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        thresh_accuracy = accuracy_score(y_test, y_pred_thresh)
        thresh_precision = precision_score(y_test, y_pred_thresh, zero_division=0)
        thresh_recall = recall_score(y_test, y_pred_thresh, zero_division=0)
        thresh_f1 = f1_score(y_test, y_pred_thresh, zero_division=0)
        threshold_results.append({
            'threshold': threshold,
            'accuracy': thresh_accuracy,
            'precision': thresh_precision,
            'recall': thresh_recall,
            'f1': thresh_f1
        })
    
    # Find best threshold based on F1 score (balance of precision and recall)
    best_threshold_result = max(threshold_results, key=lambda x: x['f1'])
    best_threshold = best_threshold_result['threshold']
    
    # Feature Importance
    feature_names = afib_features.get("features", [f"f{i}" for i in range(X_train.shape[1])])
    feature_importance = dict(zip(feature_names, model.feature_importances_))
    
    # Save model to temporary location for MLflow registration
    model_dir = "/tmp/dagster_models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"model_{partition_key}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    context.log.info(f"Model saved to {model_path} for MLflow registration")

    # Plot feature importance
    plt.figure(figsize=(10, 6))
    xgb.plot_importance(model)
    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    feature_importance_plot = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()

    # Prepare metadata
    metadata = {
        "feature_importance": MV.md(f"![Feature Importance](data:image/png;base64,{feature_importance_plot})"),
        "accuracy": accuracy,
    }
    
    # Only add ROC AUC and PR-AUC to metadata if valid
    if not (roc_auc is None or (isinstance(roc_auc, float) and roc_auc != roc_auc)):  # Check for NaN
        metadata["roc_auc"] = roc_auc
        metadata["pr_auc"] = pr_auc
    else:
        metadata["roc_auc_status"] = "undefined (single class in test set)"
    
    return Output(
        value={
            "model": model,
            "model_path": model_path,
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "report": report,
            "hyperparams": hyperparams,
            "feature_importance": feature_importance,
            "y_proba": y_proba,
            "y_test": y_test,
            "threshold_results": threshold_results,
            "best_threshold": best_threshold,
        },
        metadata=metadata,
    )
