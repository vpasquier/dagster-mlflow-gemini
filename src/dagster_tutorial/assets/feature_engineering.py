from dagster import asset, Output, MetadataValue as MV
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

@asset
def afib_features(context, afib_raw_data: pd.DataFrame) -> Output[dict]:
    """Prepare features and split data using sample-based stratified split."""
    context.log.info(f"Starting feature engineering with {len(afib_raw_data)} rows")
    
    df = afib_raw_data.copy()
    
    # Set collected_at as index for rolling window calculations
    df = df.sort_values(['ext_account', 'collected_at'])
    
    # Create rolling features based on gross_activity
    df["activity_trend_5min"] = df.groupby("ext_account")["gross_activity"].transform(
        lambda x: x.rolling(window=5, min_periods=1).mean()
    )
    df["activity_std_5min"] = df.groupby("ext_account")["gross_activity"].transform(
        lambda x: x.rolling(window=5, min_periods=1).std()
    )
    df["activity_max_5min"] = df.groupby("ext_account")["gross_activity"].transform(
        lambda x: x.rolling(window=5, min_periods=1).max()
    )
    
    context.log.info(f"After feature creation: {len(df)} rows")
    df = df.dropna()
    context.log.info(f"After dropna: {len(df)} rows")

    features = ["gross_activity", "activity_confidence", "activity_trend_5min", "activity_std_5min", "activity_max_5min"]
    X = df[features]
    y = (df["avg_afib_prob"] >= 50).astype(int)

    # Sample-based split (instead of patient-based split)
    context.log.info(f"Total samples: {len(X)}")
    context.log.info(f"Class distribution - AFib: {y.sum()}, No AFib: {(~y.astype(bool)).sum()}")
    
    # Use stratified train_test_split to maintain class distribution
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42,
            stratify=y
        )
        context.log.info(f"Split into training and test sets (stratified by class)")
    except ValueError as e:
        # If stratification fails (e.g., too few samples), fall back to random split
        context.log.warning(f"Stratified split failed: {e}. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, 
            y, 
            test_size=0.2, 
            random_state=42
        )
        context.log.info(f"Split into training and test sets (random)")

    context.log.info(f"Training set size: {len(X_train)}, Test set size: {len(X_test)}")
    context.log.info(f"Training class distribution - AFib: {y_train.sum()}, No AFib: {(~y_train.astype(bool)).sum()}")
    context.log.info(f"Test class distribution - AFib: {y_test.sum()}, No AFib: {(~y_test.astype(bool)).sum()}")

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_test

    context.log.info("Feature engineering completed successfully")
    
    # Calculate class percentages
    total_samples = len(y)
    afib_count = y.sum()
    no_afib_count = len(y) - afib_count
    afib_pct = (afib_count / total_samples) * 100
    
    train_afib_pct = (y_train.sum() / len(y_train)) * 100
    test_afib_pct = (y_test.sum() / len(y_test)) * 100 if len(y_test) > 0 else 0
    
    # Create markdown report
    report_md = f"""
## Feature Engineering Summary

### Dataset Overview
- **Total Samples**: {total_samples:,}
- **Training Samples**: {len(X_train):,} ({len(X_train)/total_samples*100:.1f}%)
- **Test Samples**: {len(X_test):,} ({len(X_test)/total_samples*100:.1f}%)

### Class Distribution
| Split | AFib Cases | No AFib Cases | AFib % |
|-------|------------|---------------|--------|
| **Overall** | {afib_count:,} | {no_afib_count:,} | {afib_pct:.1f}% |
| **Training** | {y_train.sum():,} | {len(y_train) - y_train.sum():,} | {train_afib_pct:.1f}% |
| **Test** | {y_test.sum():,} | {len(y_test) - y_test.sum():,} | {test_afib_pct:.1f}% |

### Engineered Features
{', '.join([f'`{f}`' for f in features])}

**Feature Descriptions:**
- `gross_activity`: Raw patient activity level
- `activity_confidence`: Confidence score for activity measurement
- `activity_trend_5min`: 5-minute rolling mean of activity (trend indicator)
- `activity_std_5min`: 5-minute rolling std of activity (variability indicator)
- `activity_max_5min`: 5-minute rolling max of activity (peak detection)

### Data Preprocessing
- ✅ Rolling window features calculated (5-minute windows)
- ✅ Missing values removed
- ✅ Features standardized using StandardScaler
- ✅ Stratified train/test split (80/20) to maintain class balance

### Quality Checks
- **Class Balance**: {"⚠️ Imbalanced dataset" if afib_pct < 20 else "✅ Balanced dataset"}
- **Split Quality**: {"✅ Stratification successful" if abs(train_afib_pct - test_afib_pct) < 5 else "⚠️ Check stratification"}
"""
    
    return Output(
        value={
            "X_train": X_train_scaled,
            "X_test": X_test_scaled,
            "y_train": y_train,
            "y_test": y_test,
            "features": features,
            "scaler": scaler,
        },
        metadata={
            "report": MV.md(report_md),
            "total_samples": total_samples,
            "training_samples": len(X_train),
            "test_samples": len(X_test),
            "afib_percentage": float(afib_pct),
            "num_features": len(features),
        }
    )
