from dagster import asset, Output, MetadataValue as MV
import pandas as pd
import io
import os
from dagster_tutorial.resources.resources import BigQueryResource, GCSResource

@asset
def afib_raw_data(context, bigquery: BigQueryResource, gcs: GCSResource) -> Output[pd.DataFrame]:
    """
    Loads joined AFib-related data from GCS parquet file if available, otherwise fetches from BigQuery.
    """
    blob_name = os.getenv("GCS_DATA_PATH", "dagster-ml/afib_raw_data.parquet")
    
    # Check if parquet file exists in GCS
    bucket = gcs.get_bucket()
    blob = bucket.blob(blob_name)
    
    if blob.exists():
        # Load from GCS
        context.log.info(f"Loading data from gs://{gcs.bucket_name}/{blob_name}")
        parquet_bytes = blob.download_as_bytes()
        df = pd.read_parquet(io.BytesIO(parquet_bytes))
        context.log.info(f"Loaded {len(df)} rows from GCS parquet file")
        source = f"gs://{gcs.bucket_name}/{blob_name}"
    else:
        # Fetch from BigQuery and save to GCS
        context.log.info("Parquet file not found in GCS, fetching from BigQuery...")
        afib_table = os.getenv("BIGQUERY_AFIB_TABLE", "")
        activity_table = os.getenv("BIGQUERY_ACTIVITY_TABLE", "")

        query = f"""
        WITH
        -- AFib probability data (labels)
        afib_data AS (
          SELECT
            ext_account,
            collected_at,
            avg_afib_prob,
            confidence AS afib_confidence
          FROM
            `{afib_table}`
        ),

        -- Activity data (1-min averages)
        activity_data AS (
          SELECT
            ext_account,
            collected_at,
            gross_activity,
            confidence AS activity_confidence
          FROM
            `{activity_table}`
        )

        -- Join the two tables
        SELECT
          f.ext_account,
          f.collected_at,
          f.avg_afib_prob,
          f.afib_confidence,
          a.gross_activity,
          a.activity_confidence
        FROM
          afib_data f
        INNER JOIN
          activity_data a
          ON f.ext_account = a.ext_account
          AND f.collected_at = a.collected_at
        ORDER BY
          f.ext_account,
          f.collected_at;
        """

        # Execute query and get results
        client = bigquery.get_client()
        df = client.query(query).to_dataframe()
        context.log.info(f"Fetched {len(df)} rows from BigQuery")
        
        # Save to GCS as parquet
        context.log.info(f"Uploading data to gs://{gcs.bucket_name}/{blob_name}")
        parquet_buffer = df.to_parquet(index=False)
        blob.upload_from_string(parquet_buffer, content_type='application/octet-stream')
        context.log.info(f"Successfully uploaded {len(df)} rows to GCS")
        source = "BigQuery (saved to GCS)"

    return Output(
        value=df,
        metadata={
            "num_rows": len(df),
            "preview": MV.md(df.head(5).to_markdown()),
            "columns": list(df.columns),
            "source": source,
        }
    )
