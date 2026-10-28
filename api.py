"""
FastAPI application for Customer Churn Prediction.
Provides endpoints for health checking, single-customer predictions, and batch predictions.
Reuses the saved Random Forest model bundle and deployment helper without retraining.
"""

import sys
import math
import logging
from pathlib import Path
from typing import List
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator

# Maximum allowed records per batch prediction request to prevent resource exhaustion
MAX_BATCH_SIZE = 100

# Set up logging for API diagnostics and cloud monitoring
logger = logging.getLogger("api")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# Ensure the saved_model directory is accessible for helper import
BASE_DIR = Path(__file__).resolve().parent
SAVED_MODEL_DIR = BASE_DIR / "saved_model"
if str(SAVED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(SAVED_MODEL_DIR))

# Import the existing prediction helper
# pyrefly: ignore[missing-import]
import churn_deployment

# Initialize the FastAPI application
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Production-ready FastAPI backend for customer churn probability and class prediction.",
    version="1.0.0"
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Safely serialize validation errors, converting non-finite numbers (NaN/Inf)
    to strings so JSONResponse never crashes with a ValueError.
    """
    errors = []
    for err in exc.errors():
        clean_err = {
            "type": err.get("type"),
            "loc": err.get("loc"),
            "msg": err.get("msg")
        }
        raw_val = err.get("input")
        if isinstance(raw_val, float) and not math.isfinite(raw_val):
            clean_err["input"] = str(raw_val)
        elif isinstance(raw_val, dict):
            clean_err["input"] = {
                k: str(v) if isinstance(v, float) and not math.isfinite(v) else v
                for k, v in raw_val.items()
            }
        else:
            clean_err["input"] = raw_val
        errors.append(clean_err)
    return JSONResponse(status_code=422, content={"detail": errors})


# Load the model and metadata once at startup to keep requests fast and lightweight
try:
    model, metadata = churn_deployment.load_bundle()
    logger.info(
        "Successfully loaded model bundle and metadata (model: %s, threshold: %s)",
        metadata.get("selected_model", "Unknown"),
        metadata.get("decision_threshold")
    )
except Exception as err:
    logger.error("Failed to load model bundle at startup: %s", err, exc_info=True)
    model, metadata = None, None


# Define schema for the 14 raw prediction input fields
class CustomerInput(BaseModel):
    Territory: str = Field(..., description="Customer territory or state abbreviation, e.g. CA, NY, TX")
    DeviceType: str = Field(..., description="Customer device type: Galaxy, iPhone, or -")
    Promotion: str = Field(..., description="Promotion received: yes or no")
    PlanType: str = Field(..., description="Subscription plan type, e.g. Purple Plan, Red Plan, Blue Plan, Green Plan")
    BaseFee: float = Field(..., ge=0.0, description="Base monthly subscription fee")
    AdditionalFeatureSpend: float = Field(..., ge=0.0, description="Monthly spend on additional features")
    NumberOfPenalties: int = Field(..., ge=0, description="Number of penalty infractions incurred")
    CurrentPeriodUsage: float = Field(..., ge=0.0, description="Usage in the current billing period")
    PriorPeriodUsage: float = Field(..., ge=0.0, description="Usage in the prior billing period")
    ServiceRating: float = Field(..., ge=0.0, description="Customer service rating score")
    ServiceTickets: int = Field(..., ge=0, description="Number of customer service tickets raised")
    StartMonth: str = Field(..., description="Contract start month, e.g. Jan, Feb, Mar")
    StartWeek: str = Field(..., description="Contract start week code, e.g. W01, W10, W52")
    CustomerTenure: int = Field(..., ge=0, description="Customer tenure in months")

    @field_validator("BaseFee", "AdditionalFeatureSpend", "CurrentPeriodUsage", "PriorPeriodUsage", "ServiceRating")
    @classmethod
    def validate_finite_numeric(cls, v: float, info):
        if not math.isfinite(v):
            raise ValueError(f"{info.field_name} must be a finite numeric value, got {v}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


@app.get("/")
def get_root():
    """Return basic API information and list of available endpoints."""
    return {
        "message": "Customer Churn Prediction API is online",
        "version": "1.0.0",
        "max_batch_size": MAX_BATCH_SIZE,
        "endpoints": {
            "GET /": "API information",
            "GET /health": "Model load status and health check",
            "POST /predict": "Predict churn for a single customer",
            "POST /predict/batch": "Predict churn for a batch of customers"
        }
    }


@app.get("/health")
def get_health():
    """Indicate whether the saved model and metadata bundle are loaded and ready."""
    if model is None or metadata is None:
        logger.warning("Health check failed: model bundle is not loaded.")
        raise HTTPException(
            status_code=503,
            detail="Model bundle is not loaded or unavailable."
        )
    return {
        "status": "healthy",
        "model_loaded": True,
        "selected_model": metadata.get("selected_model", "Random Forest"),
        "decision_threshold": metadata.get("decision_threshold")
    }


@app.post("/predict")
def predict_single(customer: CustomerInput):
    """
    Predict churn probability and classification for a single customer.
    Reuses existing preprocessing, feature engineering, and decision threshold.
    """
    if model is None or metadata is None:
        logger.error("Predict endpoint rejected request: model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Model is currently unavailable."
        )

    try:
        # Pass the input dictionary to the existing helper
        record = customer.model_dump()
        result_df = churn_deployment.predict_customers(record, model, metadata)
        row = result_df.iloc[0]

        return {
            "churn_probability": round(float(row["churn_probability"]), 4),
            "predicted_class": int(row["predicted_class"]),
            "predicted_label": str(row["predicted_label"]),
            "threshold_used": round(float(row["threshold_used"]), 4)
        }
    except ValueError as val_err:
        logger.warning("Invalid input received in /predict: %s", val_err)
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        logger.error("Unexpected error in /predict: %s", err, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the prediction."
        )


@app.post("/predict/batch")
def predict_batch(customers: List[CustomerInput]):
    """
    Predict churn for a batch of customers in a single request.
    Returns list of prediction outputs with probabilities and class labels.
    """
    if model is None or metadata is None:
        logger.error("Batch predict endpoint rejected request: model is unavailable.")
        raise HTTPException(
            status_code=503,
            detail="Model is currently unavailable."
        )

    if not customers:
        raise HTTPException(
            status_code=400,
            detail="Batch list of customers cannot be empty."
        )

    if len(customers) > MAX_BATCH_SIZE:
        logger.warning(
            "Batch request rejected: %d records exceeds maximum limit of %d",
            len(customers),
            MAX_BATCH_SIZE
        )
        raise HTTPException(
            status_code=400,
            detail=f"Batch size of {len(customers)} exceeds the maximum allowed limit of {MAX_BATCH_SIZE} customers."
        )

    try:
        records = [c.model_dump() for c in customers]
        results_df = churn_deployment.predict_customers(records, model, metadata)

        predictions = []
        for _, row in results_df.iterrows():
            predictions.append({
                "churn_probability": round(float(row["churn_probability"]), 4),
                "predicted_class": int(row["predicted_class"]),
                "predicted_label": str(row["predicted_label"]),
                "threshold_used": round(float(row["threshold_used"]), 4)
            })

        return {
            "count": len(predictions),
            "predictions": predictions
        }
    except ValueError as val_err:
        logger.warning("Invalid input received in /predict/batch: %s", val_err)
        raise HTTPException(status_code=400, detail=str(val_err))
    except Exception as err:
        logger.error("Unexpected error in /predict/batch: %s", err, exc_info=True)
        raise HTTPException(
            status_code=500,
            detail="An error occurred while processing the batch predictions."
        )
