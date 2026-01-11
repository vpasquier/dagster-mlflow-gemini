from dagster import asset, Output, MetadataValue as MV, AssetExecutionContext, AssetKey
import pandas as pd
from dagster_gemini import GeminiResource
from dagster_tutorial.assets.model_visualization import afib_model_visualization
from dagster_tutorial.assets.training_model import HYPERPARAM_CONFIGS


@asset(
    deps=[afib_model_visualization],
    compute_kind="gemini"
)
def afib_gemini_analysis(
    context: AssetExecutionContext, 
    gemini: GeminiResource
) -> Output[str]:
    """Use Gemini to analyze all model metrics and provide insights."""
    
    # Collect all metrics from all partitions
    comparison_data = []
    
    for pk in HYPERPARAM_CONFIGS.keys():
        # Get the training asset materializations
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
    
    # Format metrics into a readable prompt for Gemini
    if not comparison_data:
        context.log.warning("No metrics data available yet")
        return Output(
            value="No metrics available for analysis yet. Please run the model training first.",
            metadata={"status": MV.text("pending")}
        )
    
    df = pd.DataFrame(comparison_data)
    
    # Create a detailed prompt with all metrics
    prompt = f"""
You are an expert machine learning engineer analyzing AFib (Atrial Fibrillation) detection model results.

I have trained {len(comparison_data)} XGBoost models with different hyperparameter configurations to detect AFib from patient data.
This is an imbalanced classification problem where AFib cases are rare but critical to detect.

Here are the results:

{df.to_string(index=False)}

Key Metrics Explained:
- **PR-AUC (Precision-Recall AUC)**: Most important for imbalanced data. Shows trade-off between precision and recall.
- **ROC-AUC**: Area under ROC curve. Good for balanced datasets.
- **Accuracy**: Overall correctness, but can be misleading with class imbalance.
- **max_depth**: Maximum tree depth in XGBoost.
- **learning_rate**: Step size for gradient boosting.

Please provide a comprehensive analysis covering:

1. **Best Model Identification**: Which hyperparameter configuration performs best and why?
2. **Metric Interpretation**: How good are these scores for AFib detection? What do they mean clinically?
3. **Hyperparameter Insights**: What patterns do you see in how max_depth and learning_rate affect performance?
4. **Recommendations**: 
   - Should we try different hyperparameter ranges?
   - Are there any concerning patterns (overfitting, underfitting)?
   - What would you suggest for production deployment?
5. **Overall Assessment**: Summary of model readiness and confidence level.

Keep your analysis concise but insightful (2-3 paragraphs max).
"""
    
    context.log.info("Sending metrics to Gemini for analysis...")
    
    # Call Gemini API
    with gemini.get_model(context) as model:
        response = model.generate_content(prompt)
        analysis = response.text
    
    context.log.info(f"Gemini analysis complete: {len(analysis)} characters")
    
    # Prepare metadata - convert numpy types to Python types for serialization
    metadata_dict = {
        "analysis": MV.md(analysis),
        "models_analyzed": len(comparison_data),
    }
    
    # Add best metrics if available, converting numpy types to native Python types
    if not df["pr_auc"].isna().all():
        metadata_dict["best_pr_auc"] = float(df["pr_auc"].max())
        metadata_dict["best_config"] = str(df.loc[df["pr_auc"].idxmax(), "partition"])
    
    # Return the analysis with markdown formatting
    return Output(
        value=analysis,
        metadata=metadata_dict
    )
