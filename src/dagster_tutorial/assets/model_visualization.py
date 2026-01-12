import base64
from io import BytesIO

from dagster import AssetExecutionContext, AssetKey, MetadataValue as MV, Output, asset
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from dagster_tutorial.assets.training_model import (
    HYPERPARAM_CONFIGS,
    afib_model_training,
)


@asset(
    deps=[afib_model_training],
    compute_kind="matplotlib"
)
def afib_model_visualization(context: AssetExecutionContext) -> Output[dict]:
    """
    Generate comprehensive visualizations comparing all trained models across hyperparameters.
    Returns a dict with the comparison DataFrame and best model information for downstream consumption.
    """
    context.log.info("Generating consolidated visualization for all trained models...")
    
    # Collect metrics from all partitions
    comparison_data = []
    
    for pk in HYPERPARAM_CONFIGS.keys():
        asset_key = AssetKey(["afib_model_training"])
        records = context.instance.fetch_materializations(
            asset_key,
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
    
    if not comparison_data:
        context.log.warning("No model data found yet. Run afib_model_training first.")
        return Output(
            value={"comparison_df": pd.DataFrame(), "best_partition": None},
            metadata={"status": MV.text("No model data available")}
        )
    
    df = pd.DataFrame(comparison_data)
    
    # Create comprehensive comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("AFib Model Training - Comprehensive Results Comparison", fontsize=16, fontweight='bold')
    
    # Plot 1: Accuracy Comparison
    ax1 = axes[0, 0]
    df_valid_acc = df[df["accuracy"].notna()].copy()
    if not df_valid_acc.empty:
        colors = plt.cm.viridis(range(len(df_valid_acc)))
        bars = ax1.bar(df_valid_acc["partition"], df_valid_acc["accuracy"], color=colors)
        ax1.set_xlabel("Hyperparameter Configuration")
        ax1.set_ylabel("Accuracy")
        ax1.set_title("Accuracy Comparison Across Models")
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df_valid_acc["accuracy"]):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax1.text(0.5, 0.5, "No accuracy data available", 
                ha='center', va='center', transform=ax1.transAxes, color='red')
    
    # Plot 2: ROC-AUC Comparison
    ax2 = axes[0, 1]
    df_valid_roc = df[df["roc_auc"].notna()].copy()
    if not df_valid_roc.empty:
        colors = plt.cm.plasma(range(len(df_valid_roc)))
        bars = ax2.bar(df_valid_roc["partition"], df_valid_roc["roc_auc"], color=colors)
        ax2.set_xlabel("Hyperparameter Configuration")
        ax2.set_ylabel("ROC-AUC Score")
        ax2.set_title("ROC-AUC Comparison")
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df_valid_roc["roc_auc"]):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
    else:
        ax2.text(0.5, 0.5, "ROC-AUC undefined\n(single class in test set)", 
                ha='center', va='center', transform=ax2.transAxes, color='red')
    
    # Plot 3: PR-AUC Comparison (better for imbalanced data)
    ax3 = axes[1, 0]
    df_valid_pr = df[df["pr_auc"].notna()].copy()
    if not df_valid_pr.empty:
        colors = plt.cm.cividis(range(len(df_valid_pr)))
        bars = ax3.bar(df_valid_pr["partition"], df_valid_pr["pr_auc"], color=colors)
        ax3.set_xlabel("Hyperparameter Configuration")
        ax3.set_ylabel("PR-AUC Score")
        ax3.set_title("PR-AUC Comparison (Best for Imbalanced Data)")
        ax3.set_ylim(0, 1)
        ax3.tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars, df_valid_pr["pr_auc"]):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.4f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Highlight the best model
        best_idx = df_valid_pr["pr_auc"].idxmax()
        best_partition = df_valid_pr.loc[best_idx, "partition"]
        best_pr_auc = df_valid_pr.loc[best_idx, "pr_auc"]
        ax3.text(0.5, 0.95, f"Best: {best_partition} (PR-AUC: {best_pr_auc:.4f})",
                ha='center', transform=ax3.transAxes, fontsize=10, 
                bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
    else:
        ax3.text(0.5, 0.5, "PR-AUC undefined", 
                ha='center', va='center', transform=ax3.transAxes, color='red')
    
    # Plot 4: Hyperparameter Impact - Grouped Bar Chart
    ax4 = axes[1, 1]
    if not df[df["accuracy"].notna()].empty:
        # Create grouped bar chart showing metrics by hyperparameters
        metrics_df = df[["partition", "accuracy", "roc_auc", "pr_auc"]].set_index("partition")
        metrics_df = metrics_df.fillna(0)  # Fill NaN with 0 for plotting
        
        x = range(len(metrics_df))
        width = 0.25
        
        ax4.bar([i - width for i in x], metrics_df["accuracy"], width, label='Accuracy', color='steelblue')
        ax4.bar(x, metrics_df["roc_auc"], width, label='ROC-AUC', color='coral')
        ax4.bar([i + width for i in x], metrics_df["pr_auc"], width, label='PR-AUC', color='lightgreen')
        
        ax4.set_xlabel("Hyperparameter Configuration")
        ax4.set_ylabel("Score")
        ax4.set_title("All Metrics Comparison")
        ax4.set_xticks(x)
        ax4.set_xticklabels(metrics_df.index, rotation=45)
        ax4.set_ylim(0, 1)
        ax4.legend()
        ax4.grid(axis='y', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "Insufficient data for comparison", 
                ha='center', va='center', transform=ax4.transAxes, color='red')
    
    plt.tight_layout()
    buffer = BytesIO()
    plt.savefig(buffer, format="png", dpi=150, bbox_inches='tight')
    buffer.seek(0)
    comparison_plot = base64.b64encode(buffer.read()).decode("utf-8")
    plt.close()
    
    # Create summary table
    summary_table = df.to_markdown(index=False, floatfmt=".4f")
    
    # Find best model using robust filtering and sorting
    # Filter out models with invalid metrics
    df_valid = df[df["accuracy"].notna()].copy()
    
    if not df_valid.empty:
        # Sort by accuracy (primary), then by PR-AUC (secondary for imbalanced data)
        df_sorted = df_valid.sort_values(
            by=["accuracy", "pr_auc"],
            ascending=[False, False],
            na_position='last'
        )
        best_model = df_sorted.iloc[0]
        best_partition = best_model["partition"]
    else:
        best_model = None
        best_partition = None
    
    summary_md = f"""
## Model Training Results Summary

**Total Models Trained:** {len(comparison_data)}

### Best Model by Accuracy:
{f"- **{best_model['partition']}**" if best_model is not None else "N/A"}
{f"- Accuracy: {best_model['accuracy']:.4f}" if best_model is not None and pd.notna(best_model['accuracy']) else ""}
{f"- PR-AUC: {best_model['pr_auc']:.4f}" if best_model is not None and pd.notna(best_model['pr_auc']) else ""}
{f"- ROC-AUC: {best_model['roc_auc']:.4f}" if best_model is not None and pd.notna(best_model['roc_auc']) else ""}
{f"- Hyperparameters: max_depth={best_model['max_depth']}, learning_rate={best_model['learning_rate']}" if best_model is not None else ""}

### All Models Comparison:

{summary_table}

**Note:** PR-AUC (Precision-Recall AUC) is the most reliable metric for imbalanced AFib detection.
"""
    
    metadata_dict = {
        "comparison_plot": MV.md(f"![Model Comparison](data:image/png;base64,{comparison_plot})"),
        "summary": MV.md(summary_md),
        "models_trained": len(comparison_data),
    }
    
    if best_model is not None:
        metadata_dict["best_model"] = str(best_model["partition"])
        if pd.notna(best_model["accuracy"]):
            metadata_dict["best_accuracy"] = float(best_model["accuracy"])
    
    context.log.info(f"Generated comprehensive visualization comparing {len(comparison_data)} models")
    if best_partition:
        context.log.info(f"   Best model identified: {best_partition}")
    
    return Output(
        value={
            "comparison_df": df,
            "best_partition": best_partition
        },
        metadata=metadata_dict,
    )
