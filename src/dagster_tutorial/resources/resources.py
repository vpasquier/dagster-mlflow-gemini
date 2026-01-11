from dagster import ConfigurableResource, EnvVar
from google.cloud import bigquery, storage


class BigQueryResource(ConfigurableResource):
    """Resource for BigQuery client."""
    project: str = EnvVar("GCP_BIGQUERY_PROJECT")
    
    def get_client(self) -> bigquery.Client:
        return bigquery.Client(project=self.project)


class GCSResource(ConfigurableResource):
    """Resource for Google Cloud Storage client."""
    project: str = EnvVar("GCP_GCS_PROJECT")
    bucket_name: str = EnvVar("GCS_BUCKET_NAME")
    
    def get_client(self) -> storage.Client:
        return storage.Client(project=self.project)
    
    def get_bucket(self) -> storage.Bucket:
        client = self.get_client()
        return client.bucket(self.bucket_name)
