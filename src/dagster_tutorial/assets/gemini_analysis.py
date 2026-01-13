from dagster import asset, Output, MetadataValue as MV, AssetExecutionContext
import pandas as pd
from dagster_gemini import GeminiResource


@asset(
    compute_kind="gemini"
)
def afib_gemini_analysis(
    context: AssetExecutionContext, 
    gemini: GeminiResource,
    afib_model_visualization: dict
) -> Output[str]:
    """Use Gemini to analyze all model metrics and provide insights."""
    
    # Reuse the comparison DataFrame from the visualization asset
    df = afib_model_visualization["comparison_df"]
    
    # Format metrics into a readable prompt for Gemini
    if df.empty:
        context.log.warning("No metrics data available yet")
        return Output(
            value="No metrics available for analysis yet. Please run the model training first.",
            metadata={"status": MV.text("pending")}
        )
    
    # Create a detailed prompt with all metrics
    prompt = f"""
You are an expert machine learning engineer analyzing AFib (Atrial Fibrillation) detection model results.

I have trained {len(df)} XGBoost models with different hyperparameter configurations to detect AFib from patient data.
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
        "models_analyzed": len(df),
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
