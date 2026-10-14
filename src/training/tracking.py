"""Optional MLflow integration that does not require a tracking server or credentials."""
from __future__ import annotations
from pathlib import Path


def log_run(settings: dict, report: dict, tracking_uri: str | None = None) -> str:
    """Log locally to MLflow when installed; otherwise return an explicit no-op status."""
    try:
        import mlflow
    except ImportError:
        return "MLflow not installed (optional dependency); JSON report retained locally."
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri((Path.cwd() / "mlruns").as_uri())
    mlflow.set_experiment("intelligent-recommendation-engine")
    with mlflow.start_run():
        mlflow.log_params(settings)
        for model, values in report["metrics"].items():
            for metric, value in values.items(): mlflow.log_metric(f"{model}.{metric}", value)
        for metric, value in report["latency"].items():
            if isinstance(value, (int, float)): mlflow.log_metric(f"latency.{metric}", value)
        return mlflow.active_run().info.run_id
