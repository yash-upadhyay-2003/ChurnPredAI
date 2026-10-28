# Customer Churn Prediction

A Customer Churn Prediction application featuring an end-to-end Machine Learning pipeline developed in Jupyter Notebook, a lightweight FastAPI backend for real-time inference, and an intuitive Streamlit frontend.

---

## 1. Project Structure

```text
Churn prediction model ML/
├── api.py                                           # FastAPI backend application
├── streamlit_app.py                                 # Streamlit frontend user interface
├── requirements.txt                                 # Production dependency pins
├── README.md                                        # Project documentation & guide
├── .gitignore                                       # Standard Git ignore rules
├── .streamlit/
│   └── config.toml                                  # Streamlit server & UI configuration
├── data/
│   └── raw/
│       └── Customer_Cancellations_Training (1).csv  # Raw subscription dataset (for validation & notebook)
├── saved_model/
│   ├── churn_model.pkl                              # Serialized Random Forest pipeline (6.38 MB)
│   ├── churn_model_metadata.json                    # Model contract, threshold (0.3810), metadata
│   └── churn_deployment.py                          # Preprocessing & inference helper
├── tests/
│   └── test_smoke.py                                # Automated 7-stage test suite
└── notebooks/
    └── Customer_Cancellations_Model_Selection_Evaluation.ipynb  # Primary, verified ML pipeline
```

---

## 2. Machine Learning Workflow & Model

- **Canonical Notebook:** `notebooks/Customer_Cancellations_Model_Selection_Evaluation.ipynb`
- **Selected Model:** `Random Forest — selected features` (`SelectFromModel` + `RandomForestClassifier` with 200 estimators).
- **Target Variable:** `Churned` (`yes` / `no`).
- **Data Leakage Precaution:** `HasRenewed` is strictly excluded from all predictors and pipeline stages.
- **Categorical Handling:** Preserves literal `DeviceType = "-"` as its own explicit category while handling genuine missing values with `"Missing (null)"`.
- **Optimal Decision Threshold:** `0.3810` (chosen using training-only out-of-fold F1 optimization).
- **Test Performance (Holdout Evaluation):**
  - **Accuracy:** 93.0%
  - **Churn Recall:** 78.1%
  - **Churn Precision:** 73.2%
  - **Churn F1-score:** 75.6%
  - **ROC-AUC:** 93.1%

---

## 3. Running Locally

### Step 1: Install Dependencies
Create a virtual environment (recommended) and install the requirements:
```bash
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
# or: .venv\Scripts\activate    # On Windows

pip install -r requirements.txt
```

### Step 2: Run Tests
Verify that the model bundle, schemas, and inference pipelines are operational:
```bash
python tests/test_smoke.py
```

### Step 3: Start the FastAPI Backend
Launch the backend server using Uvicorn:
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
- Interactive Swagger UI: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### Step 4: Start the Streamlit Frontend
In a separate terminal, launch the Streamlit application:
```bash
streamlit run streamlit_app.py
```
The web interface will automatically open in your default browser at [http://localhost:8501](http://localhost:8501).

---

## 4. Manual Cloud Deployment Guide

### A. Deploy Backend on Render
1. Create a new **Web Service** on [Render](https://render.com).
2. Connect your GitHub repository: `https://github.com/yash-upadhyay-2003/ChurnPredAI`.
3. Configure service settings:
   - **Name:** `churn-prediction-api` (or preferred name)
   - **Branch:** `main`
   - **Language / Runtime:** `Python 3`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
4. Click **Deploy Web Service** and wait for the build to finish.
5. Copy your live Render URL (e.g., `https://churn-prediction-api.onrender.com`).

### B. Deploy Frontend on Streamlit Community Cloud
1. Sign in to [Streamlit Community Cloud](https://share.streamlit.io).
2. Click **New app** and configure:
   - **Repository:** `yash-upadhyay-2003/ChurnPredAI`
   - **Branch:** `main`
   - **Main file path:** `streamlit_app.py`
3. Expand **Advanced settings** -> **Secrets**, and enter your Render URL:
   ```toml
   BACKEND_URL = "https://YOUR-RENDER-SERVICE.onrender.com"
   ```
   *(Important: Replace `https://YOUR-RENDER-SERVICE.onrender.com` with your actual live Render service URL from Step A).*
4. Click **Deploy**.

---

## 5. Post-Deployment Verification

After deploying on Render, test your live backend service before pointing Streamlit to it:

### 1. Verify Health Endpoint
```bash
curl -X GET "https://YOUR-RENDER-SERVICE.onrender.com/health"
```
Expected response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "selected_model": "Random Forest — selected features",
  "decision_threshold": 0.3809723161404831
}
```

### 2. Verify Prediction Endpoint
```bash
curl -X POST "https://YOUR-RENDER-SERVICE.onrender.com/predict" \
  -H "Content-Type: application/json" \
  -d '{
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
  }'
```
Expected response:
```json
{
  "churn_probability": 0.0,
  "predicted_class": 0,
  "predicted_label": "no",
  "threshold_used": 0.381
}
```
