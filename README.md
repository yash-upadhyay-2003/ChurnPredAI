# Customer Churn Prediction

A production-ready Customer Churn Prediction application featuring an end-to-end Machine Learning pipeline developed in Jupyter Notebook, a lightweight FastAPI backend for real-time inference, and an intuitive Streamlit frontend.

---

## 1. Project Structure

```text
DDFFFF/
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

### Step 2: Run Smoke Tests
Verify that the model bundle and preprocessing helper are operational:
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

## 4. Production Deployment

### Backend (Render)
- **Service Type:** Web Service (Python)
- **Build Command:** `pip install -r requirements.txt`
- **Start Command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`

### Frontend (Streamlit Community Cloud)
- **Main file path:** `streamlit_app.py`
- **Secrets Configuration:**
  ```toml
  BACKEND_URL = "https://<your-render-service-name>.onrender.com"
  ```
