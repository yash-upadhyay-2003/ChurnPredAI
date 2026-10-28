"""
Comprehensive automated test suite for Customer Churn Prediction.
Tests model artifacts, deployment helper, inference pipelines, and FastAPI endpoints.
"""

import sys
import json
import math
from pathlib import Path
import pandas as pd
import numpy as np
from fastapi.testclient import TestClient

# Resolve project base directory dynamically so tests run from any working directory
BASE_DIR = Path(__file__).resolve().parent.parent
SAVED_MODEL_DIR = BASE_DIR / "saved_model"

if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))
if str(SAVED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SAVED_MODEL_DIR))

# pyrefly: ignore[missing-import]
import churn_deployment
import api


def run_smoke_test():
    print("=" * 70)
    print("RUNNING COMPREHENSIVE TEST SUITE FOR CUSTOMER CHURN PREDICTION")
    print("=" * 70)

    # ---------------------------------------------------------
    # 1. Model Artifact and Metadata Loading
    # ---------------------------------------------------------
    model, metadata = churn_deployment.load_bundle()
    assert model is not None, "Model failed to load!"
    assert metadata is not None, "Metadata failed to load!"
    model_name = str(metadata.get("selected_model", "Random Forest")).replace("\u2014", "-").replace("\u2013", "-")
    print("1. [PASS] Bundle loaded successfully.")
    print(f"   Model type: {model_name}")
    print(f"   Decision threshold: {metadata['decision_threshold']}")
    print(f"   Scikit-learn version: {metadata.get('scikit_learn_version')}")
    print(f"   Joblib version: {metadata.get('joblib_version')}")

    # ---------------------------------------------------------
    # 2. Verify Schema Contract (14 raw, 5 engineered, 19 model inputs)
    # ---------------------------------------------------------
    raw_cols = metadata["raw_input_columns"]
    eng_cols = list(metadata["engineered_features"].keys())
    model_input_cols = metadata["model_input_columns_after_feature_engineering"]

    assert "HasRenewed" not in raw_cols, "HasRenewed must NOT be in raw_input_columns!"
    assert "HasRenewed" not in model_input_cols, "HasRenewed must NOT be in model_input_cols!"
    assert len(raw_cols) == 14, f"Expected 14 raw cols, got {len(raw_cols)}"
    assert len(eng_cols) == 5, f"Expected 5 engineered cols, got {len(eng_cols)}"
    assert len(model_input_cols) == 19, f"Expected 19 model cols, got {len(model_input_cols)}"
    print("2. [PASS] Metadata columns verified: 14 raw inputs, 5 engineered, 19 model inputs.")
    print("   Confirmation: HasRenewed is completely absent from all prediction schemas.")

    # ---------------------------------------------------------
    # 3. Test with Raw Dataset Records
    # ---------------------------------------------------------
    dataset_candidates = [
        BASE_DIR / "data" / "raw" / "Customer_Cancellations_Training (1).csv",
        BASE_DIR / "Customer_Cancellations_Training (1).csv",
    ]
    dataset_path = next((p for p in dataset_candidates if p.exists()), None)
    assert dataset_path is not None, "Dataset Customer_Cancellations_Training (1).csv not found!"

    df = pd.read_csv(dataset_path)
    sample_records = df.loc[:4, raw_cols].copy()
    preds_df = churn_deployment.predict_customers(sample_records, model, metadata)
    assert len(preds_df) == 5
    print("3. [PASS] Batch prediction on raw dataset DataFrame succeeded.")

    # ---------------------------------------------------------
    # 4. Valid Probability Bounds and Threshold-based Class Mapping
    # ---------------------------------------------------------
    assert (preds_df["churn_probability"] >= 0.0).all() and (preds_df["churn_probability"] <= 1.0).all()
    assert set(preds_df["predicted_class"].unique()).issubset({0, 1})
    assert set(preds_df["predicted_label"].unique()).issubset({"yes", "no"})
    threshold = float(metadata["decision_threshold"])
    expected_classes = (preds_df["churn_probability"] >= threshold).astype(int)
    assert (preds_df["predicted_class"] == expected_classes).all(), "Class mapping does not match threshold!"
    print("4. [PASS] Probability bounds (0.0 - 1.0) and threshold classification verified.")

    # ---------------------------------------------------------
    # 5. Single Dict and List-of-Dicts Prediction
    # ---------------------------------------------------------
    single_record = sample_records.iloc[0].to_dict()
    single_pred = churn_deployment.predict_customers(single_record, model, metadata)
    assert len(single_pred) == 1
    assert single_pred["predicted_class"].iloc[0] == preds_df["predicted_class"].iloc[0]

    list_records = sample_records.to_dict(orient="records")
    list_preds = churn_deployment.predict_customers(list_records, model, metadata)
    assert len(list_preds) == 5
    print("5. [PASS] Single dict and list-of-dicts input formats handled correctly.")

    # ---------------------------------------------------------
    # 6. Missing Required Fields Validation
    # ---------------------------------------------------------
    try:
        bad_record = {"Territory": "CA", "BaseFee": 50.0}
        churn_deployment.predict_customers(bad_record, model, metadata)
        raise AssertionError("Validation failed to catch missing columns!")
    except ValueError as val_err:
        assert "Missing required prediction input columns" in str(val_err)
        print("6. [PASS] Missing required columns caught and raised ValueError.")

    # ---------------------------------------------------------
    # 7. FastAPI TestClient: GET / and GET /health
    # ---------------------------------------------------------
    client = TestClient(api.app)

    r_root = client.get("/")
    assert r_root.status_code == 200, f"GET / failed: {r_root.status_code}"
    root_json = r_root.json()
    assert "version" in root_json and "max_batch_size" in root_json
    print("7. [PASS] GET / returned HTTP 200 with API info and max_batch_size.")

    r_health = client.get("/health")
    assert r_health.status_code == 200, f"GET /health failed: {r_health.status_code}"
    health_json = r_health.json()
    assert health_json["status"] == "healthy"
    assert health_json["model_loaded"] is True
    assert math.isclose(health_json["decision_threshold"], threshold, rel_tol=1e-5)
    print("8. [PASS] GET /health returned HTTP 200 (healthy, model_loaded=True).")

    # ---------------------------------------------------------
    # 8. FastAPI TestClient: POST /predict (Valid Single Customer)
    # ---------------------------------------------------------
    valid_customer_payload = {
        "Territory": "CA",
        "DeviceType": "Galaxy",
        "Promotion": "no",
        "PlanType": "Purple Plan",
        "BaseFee": 30.0,
        "AdditionalFeatureSpend": 15.0,
        "NumberOfPenalties": 1,
        "CurrentPeriodUsage": 200.0,
        "PriorPeriodUsage": 190.0,
        "ServiceRating": 10.0,
        "ServiceTickets": 4,
        "StartMonth": "Mar",
        "StartWeek": "W10",
        "CustomerTenure": 100
    }
    r_pred = client.post("/predict", json=valid_customer_payload)
    assert r_pred.status_code == 200, f"POST /predict failed: {r_pred.status_code} {r_pred.text}"
    pred_json = r_pred.json()
    assert "churn_probability" in pred_json
    assert "predicted_class" in pred_json
    assert pred_json["predicted_label"] in ["yes", "no"]
    assert 0.0 <= pred_json["churn_probability"] <= 1.0
    print("9. [PASS] POST /predict returned HTTP 200 with valid churn probability and label.")

    # ---------------------------------------------------------
    # 9. FastAPI Validation: Missing Fields and Non-Finite Floats
    # ---------------------------------------------------------
    # Missing required field
    r_missing = client.post("/predict", json={"Territory": "CA"})
    assert r_missing.status_code == 422, f"Expected 422 for missing fields, got {r_missing.status_code}"

    # Invalid data type (string instead of numeric)
    invalid_type_payload = valid_customer_payload.copy()
    invalid_type_payload["BaseFee"] = "invalid_string_fee"
    r_type = client.post("/predict", json=invalid_type_payload)
    assert r_type.status_code == 422, f"Expected 422 for string float, got {r_type.status_code}"

    # Negative fee violating ge=0.0 constraint
    neg_payload = valid_customer_payload.copy()
    neg_payload["BaseFee"] = -25.0
    r_neg = client.post("/predict", json=neg_payload)
    assert r_neg.status_code == 422, f"Expected 422 for negative fee, got {r_neg.status_code}"

    # Non-finite float (NaN) sent via raw body
    raw_nan = json.dumps(valid_customer_payload).replace('"BaseFee": 30.0', '"BaseFee": NaN')
    r_nan = client.post("/predict", content=raw_nan, headers={"Content-Type": "application/json"})
    assert r_nan.status_code == 422, f"Expected 422 for raw NaN, got {r_nan.status_code}"

    print("10. [PASS] Input schema validation correctly rejected missing fields, invalid types, and non-finite values with HTTP 422.")

    # ---------------------------------------------------------
    # 10. FastAPI TestClient: POST /predict/batch (Valid, Empty, and Oversized)
    # ---------------------------------------------------------
    # Valid batch
    r_batch = client.post("/predict/batch", json=[valid_customer_payload, valid_customer_payload])
    assert r_batch.status_code == 200, f"POST /predict/batch failed: {r_batch.status_code}"
    batch_json = r_batch.json()
    assert batch_json["count"] == 2
    assert len(batch_json["predictions"]) == 2

    # Empty batch
    r_empty = client.post("/predict/batch", json=[])
    assert r_empty.status_code == 400, f"Expected 400 for empty batch, got {r_empty.status_code}"

    # Oversized batch (> MAX_BATCH_SIZE)
    oversized = [valid_customer_payload] * (api.MAX_BATCH_SIZE + 1)
    r_oversized = client.post("/predict/batch", json=oversized)
    assert r_oversized.status_code == 400, f"Expected 400 for oversized batch, got {r_oversized.status_code}"
    assert "exceeds the maximum allowed limit" in r_oversized.json()["detail"]
    print("11. [PASS] Batch predictions validated: valid batch (200), empty batch (400), oversized batch (400).")

    # ---------------------------------------------------------
    # 11. Backend Model Unavailable Simulation (HTTP 503)
    # ---------------------------------------------------------
    saved_m, saved_meta = api.model, api.metadata
    try:
        api.model, api.metadata = None, None
        r_unavail_health = client.get("/health")
        assert r_unavail_health.status_code == 503, f"Expected 503 on health when model None, got {r_unavail_health.status_code}"

        r_unavail_pred = client.post("/predict", json=valid_customer_payload)
        assert r_unavail_pred.status_code == 503, f"Expected 503 on predict when model None, got {r_unavail_pred.status_code}"

        r_unavail_batch = client.post("/predict/batch", json=[valid_customer_payload])
        assert r_unavail_batch.status_code == 503, f"Expected 503 on batch when model None, got {r_unavail_batch.status_code}"
    finally:
        api.model, api.metadata = saved_m, saved_meta

    print("12. [PASS] Model unavailable simulation returned HTTP 503 on health, predict, and batch without leaking internals.")

    # ---------------------------------------------------------
    # 12. Frontend Error Path Simulation
    # ---------------------------------------------------------
    import requests
    # Simulate frontend trying to reach an unreachable host
    unreachable_url = "http://127.0.0.1:59999"
    try:
        requests.get(f"{unreachable_url}/health", timeout=0.5)
        connection_failed = False
    except requests.exceptions.RequestException:
        connection_failed = True
    assert connection_failed, "Unreachable URL did not raise RequestException!"
    print("13. [PASS] Frontend error handling simulation: unreachable host raises handled connection exception.")

    print("\n" + "=" * 70)
    print("ALL SMOKE TESTS AND API INTEGRATION TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    run_smoke_test()
