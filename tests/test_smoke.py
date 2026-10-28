import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np

# Resolve project base directory dynamically so it works from any working directory
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = BASE_DIR.parent / "saved_model" if (BASE_DIR.parent / "saved_model").exists() else BASE_DIR / "saved_model"

# Add saved_model to sys.path
sys.path.insert(0, str(SAVED_MODEL_DIR))
# pyrefly: ignore[missing-import]
import churn_deployment

def run_smoke_test():
    print("=" * 60)
    print("RUNNING FINAL SMOKE TEST FOR CHURN DEPLOYMENT (No HasRenewed)")
    print("=" * 60)

    # 1. Load saved model and metadata
    model, metadata = churn_deployment.load_bundle()
    print("1. [PASS] Bundle loaded successfully.")
    model_name = str(metadata.get('selected_model', 'Random Forest')).replace('\u2014', '-').replace('\u2013', '-')
    print(f"   Model type: {model_name}")
    print(f"   Decision threshold: {metadata['decision_threshold']}")
    print(f"   Scikit-learn version: {metadata.get('scikit_learn_version')}")
    print(f"   Joblib version: {metadata.get('joblib_version')}")

    # 2. Verify metadata contract consistency
    raw_cols = metadata["raw_input_columns"]
    eng_cols = list(metadata["engineered_features"].keys())
    model_input_cols = metadata["model_input_columns_after_feature_engineering"]

    # HasRenewed must NOT be in raw columns, engineered columns, or model inputs
    assert "HasRenewed" not in raw_cols, "HasRenewed must not be in raw_input_columns!"
    assert "HasRenewed" not in model_input_cols, "HasRenewed must not be in model_input_columns!"
    assert len(raw_cols) == 14, f"Expected 14 raw cols, got {len(raw_cols)}"
    assert len(eng_cols) == 5, f"Expected 5 engineered cols, got {len(eng_cols)}"
    assert len(model_input_cols) == 19, f"Expected 19 model cols, got {len(model_input_cols)}"
    print("2. [PASS] Metadata columns verified: 14 raw inputs, 5 engineered features, 19 model inputs.")
    print("   Confirmation: HasRenewed is completely absent from all input schemas.")

    # 3. Test with raw training dataset records
    candidate_data = [
        BASE_DIR.parent / "data" / "raw" / "Customer_Cancellations_Training (1).csv",
        BASE_DIR / "data" / "raw" / "Customer_Cancellations_Training (1).csv",
        BASE_DIR.parent / "Customer_Cancellations_Training (1).csv",
        BASE_DIR / "Customer_Cancellations_Training (1).csv"
    ]
    dataset_path = next((p for p in candidate_data if p.exists()), None)
    assert dataset_path is not None, "Dataset Customer_Cancellations_Training (1).csv not found!"
    
    df = pd.read_csv(dataset_path)
    sample_records = df.loc[:4, raw_cols].copy()
    
    # 4. Run predict_customers on DataFrame
    preds_df = churn_deployment.predict_customers(sample_records, model, metadata)
    print("3. [PASS] Predict on DataFrame returned successfully.")
    print(preds_df)

    # Assertions
    assert len(preds_df) == 5
    assert (preds_df["churn_probability"] >= 0.0).all() and (preds_df["churn_probability"] <= 1.0).all()
    assert set(preds_df["predicted_class"].unique()).issubset({0, 1})
    assert set(preds_df["predicted_label"].unique()).issubset({"yes", "no"})
    assert (preds_df["predicted_class"] == (preds_df["churn_probability"] >= metadata["decision_threshold"]).astype(int)).all()
    print("4. [PASS] Probability bounds and threshold logic verified.")

    # 5. Test with single dict input
    single_record = sample_records.iloc[0].to_dict()
    assert "HasRenewed" not in single_record
    single_pred = churn_deployment.predict_customers(single_record, model, metadata)
    assert len(single_pred) == 1
    assert single_pred["predicted_class"].iloc[0] == preds_df["predicted_class"].iloc[0]
    print("5. [PASS] Single dict input handled correctly.")

    # 6. Test with list of dicts
    list_records = sample_records.to_dict(orient="records")
    list_preds = churn_deployment.predict_customers(list_records, model, metadata)
    assert len(list_preds) == 5
    print("6. [PASS] List of dicts input handled correctly.")

    # 7. Test validation on missing columns
    try:
        bad_record = {"Territory": "Central", "BaseFee": 50.0}
        churn_deployment.predict_customers(bad_record, model, metadata)
        raise AssertionError("Validation failed to catch missing columns!")
    except ValueError as e:
        print("7. [PASS] Missing required column safely caught and raised ValueError:")
        print(f"   {e}")

    print("\nALL SMOKE TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    run_smoke_test()
