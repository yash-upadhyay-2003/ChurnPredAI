from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd

BUNDLE_DIR = Path(__file__).resolve().parent

def load_bundle():
    """Load the saved fitted model and matching metadata."""
    model = joblib.load(BUNDLE_DIR / "churn_model.pkl")
    with open(
        BUNDLE_DIR / "churn_model_metadata.json",
        "r",
        encoding="utf-8"
    ) as file:
        metadata = json.load(file)
    return model, metadata

def prepare_features(records, metadata):
    """Prepare raw customer records in the model's expected feature format."""
    if isinstance(records, dict):
        records = [records]

    frame = (
        records.copy()
        if isinstance(records, pd.DataFrame)
        else pd.DataFrame(records)
    )

    required = metadata["raw_input_columns"]
    missing = [column for column in required if column not in frame.columns]

    if missing:
        raise ValueError(
            f"Missing required prediction input columns: {missing}"
        )

    # Keep only the original predictor columns, in the required order.
    frame = frame[required].copy()

    # Preserve DeviceType category representation matching training data (literal '-')
    if "DeviceType" in frame.columns:
        frame["DeviceType"] = frame["DeviceType"].fillna("Missing (null)")

    # Only coerce truly numeric columns; preserve StartMonth and StartWeek as categorical strings
    numeric_cols = [
        "BaseFee", "AdditionalFeatureSpend", "NumberOfPenalties",
        "CurrentPeriodUsage", "PriorPeriodUsage", "CustomerTenure",
        "ServiceRating", "ServiceTickets"
    ]
    for col in numeric_cols:
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    # Reproduce the exact feature-engineering formulas used during training.
    frame["TotalCustomerSpend"] = (
        frame["BaseFee"] + frame["AdditionalFeatureSpend"]
    )
    frame["UsageDifference"] = (
        frame["CurrentPeriodUsage"] - frame["PriorPeriodUsage"]
    )
    frame["AdditionalSpendRatio"] = (
        frame["AdditionalFeatureSpend"] / (frame["BaseFee"] + 1)
    )
    frame["Penalties_per_Tenure"] = (
        frame["NumberOfPenalties"] / (frame["CustomerTenure"] + 1)
    )
    frame["Tickets_per_Tenure"] = (
        frame["ServiceTickets"] / (frame["CustomerTenure"] + 1)
    )

    # Preserve the exact column names and order expected by the fitted pipeline.
    return frame[metadata["model_input_columns_after_feature_engineering"]]

def predict_customers(records, model=None, metadata=None):
    """Return churn probabilities and threshold-based predictions."""
    if model is None or metadata is None:
        loaded_model, loaded_metadata = load_bundle()
        model = loaded_model if model is None else model
        metadata = loaded_metadata if metadata is None else metadata

    features = prepare_features(records, metadata)
    probability = model.predict_proba(features)[:, 1]
    threshold = float(metadata["decision_threshold"])
    prediction = (probability >= threshold).astype(int)

    return pd.DataFrame({
        "churn_probability": probability,
        "predicted_class": prediction,
        "predicted_label": np.where(prediction == 1, "yes", "no"),
        "threshold_used": threshold
    })
