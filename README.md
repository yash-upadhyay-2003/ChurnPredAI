# Customer Churn Prediction

An end-to-end Machine Learning project that predicts whether a customer is likely to churn. The project includes a Jupyter Notebook for model development and evaluation, a FastAPI backend for prediction requests, and a Streamlit frontend for user interaction.

---

## Table of Contents
- [Project Overview](#project-overview)
- [Project Structure](#project-structure)
- [Machine Learning Workflow](#machine-learning-workflow)
- [Model Evaluation](#model-evaluation)
- [Technology Stack](#technology-stack)
- [Run Locally](#run-locally)
- [API Reference](#api-reference)
- [Deploy to the Cloud](#deploy-to-the-cloud)
- [Post-Deployment Verification](#post-deployment-verification)
- [Important Notes](#important-notes)
- [License](#license)

---

## Project Overview

Customer churn occurs when a customer stops using a service or cancels a subscription. This project uses a trained classification model to estimate churn probability and classify a customer as **yes** or **no** for churn.

The application has three main components:
1. **Machine Learning:** Model selection, training, evaluation, feature engineering, and decision-threshold selection in Jupyter Notebook.
2. **FastAPI backend:** Loads the saved model and exposes endpoints for health checks, individual predictions, and batch predictions.
3. **Streamlit frontend:** Provides a web interface for submitting customer information and viewing predictions.

The deployed application uses the saved model artifact. It does not retrain the model for every prediction request.

---

## Project Structure

```text
ChurnPredAI/
├── api.py                                           # FastAPI application and prediction endpoints
├── streamlit_app.py                                 # Streamlit user interface
├── requirements.txt                                 # Python dependencies
├── README.md                                        # Project documentation & guide
├── .gitignore                                       # Git ignore rules
├── .streamlit/
│   └── config.toml                                  # Streamlit server & UI configuration
├── data/
│   └── raw/
│       └── Customer_Cancellations_Training (1).csv  # Dataset used by the project and smoke tests
├── saved_model/
│   ├── churn_model.pkl                              # Serialized trained model
│   ├── churn_model_metadata.json                    # Model metadata, feature definitions, decision threshold
│   └── churn_deployment.py                          # Model loading, preprocessing, feature engineering logic
├── tests/
│   └── test_smoke.py                                # Smoke tests for the saved model and prediction helper
└── notebooks/
    └── Customer_Cancellations_Model_Selection_Evaluation.ipynb  # Model development & evaluation notebook
```

### Important Files

| File | Purpose |
| :--- | :--- |
| `api.py` | FastAPI application and prediction endpoints |
| `streamlit_app.py` | Streamlit user interface |
| `requirements.txt` | Python dependencies |
| `saved_model/churn_model.pkl` | Serialized trained model |
| `saved_model/churn_model_metadata.json` | Model metadata, feature definitions, and decision threshold |
| `saved_model/churn_deployment.py` | Model loading, preprocessing, feature engineering, and prediction logic |
| `tests/test_smoke.py` | Smoke tests for the saved model and prediction helper |
| `notebooks/Customer_Cancellations_Model_Selection_Evaluation.ipynb` | Model development and evaluation notebook |
| `data/raw/Customer_Cancellations_Training (1).csv` | Dataset used by the project and smoke tests |

---

## Machine Learning Workflow

The project follows this workflow:
1. Load and inspect the customer dataset.
2. Perform data preparation and preprocessing.
3. Engineer additional customer-behaviour features.
4. Compare candidate classification models.
5. Select the model using the notebook's evaluation procedure.
6. Select a decision threshold using training-only out-of-fold predictions.
7. Evaluate the selected model on the holdout test set.
8. Save the fitted model and its metadata.
9. Load the saved model through the deployment helper.
10. Serve predictions through FastAPI and display them through Streamlit.

### Selected Model
The selected model recorded in the model metadata is **Random Forest — selected features**, using feature selection with `SelectFromModel` and a `RandomForestClassifier` configured with 200 estimators.

### Target Variable
- **Target column:** `Churned`
- **Target labels:** `yes` and `no`
- **Prediction class mapping:** `0 = no`, `1 = yes`

### Data Leakage Precaution
`HasRenewed` is excluded from the model's prediction inputs. This prevents the deployment pipeline from using that column as a predictor.

### Feature Engineering
The deployment helper derives these features from the raw customer inputs:
- `TotalCustomerSpend`
- `UsageDifference`
- `AdditionalSpendRatio`
- `Penalties_per_Tenure`
- `Tickets_per_Tenure`

The formulas and expected input columns are defined in `saved_model/churn_model_metadata.json`.

### Decision Threshold
The saved model metadata specifies a decision threshold of approximately **`0.3810`**.
The application uses this threshold to convert the predicted churn probability into a class:
- Probability greater than or equal to the threshold → **`yes` (class 1)**
- Probability below the threshold → **`no` (class 0)**

The threshold is not the same as the predicted probability. It is the cutoff used to classify that probability.

---

## Model Evaluation

The holdout evaluation results for the selected model are:

| Metric | Reported Result |
| :--- | :--- |
| **Accuracy** | 93.0% |
| **Churn Recall** | 78.1% |
| **Churn Precision** | 73.2% |
| **Churn F1-score** | 75.6% |
| **ROC-AUC** | 93.1% |

These are the results documented for the selected model in the project. They should be interpreted as reported evaluation results, not as a guarantee of performance on new, real-world customers.

---

## Technology Stack

- **Language:** Python
- **Machine Learning:** scikit-learn
- **Data processing:** Pandas, NumPy
- **Model serialization:** Joblib
- **Backend:** FastAPI
- **ASGI server:** Uvicorn
- **Frontend:** Streamlit
- **API communication:** HTTP requests
- **Development and evaluation:** Jupyter Notebook
- **Version control:** Git and GitHub
- **Deployment:** Render (Backend) and Streamlit Community Cloud (Frontend)

---

## Run Locally

### Prerequisites
- Python installed on your computer (Python 3.10+ recommended)
- Git
- A terminal or command prompt

### Step 1: Clone the repository
```bash
git clone https://github.com/yash-upadhyay-2003/ChurnPredAI.git
cd ChurnPredAI
```

### Step 2: Create a virtual environment
```bash
python -m venv .venv
```
Activate it on Windows (PowerShell):
```powershell
.venv\Scripts\Activate.ps1
```
Or, using Command Prompt:
```cmd
.venv\Scripts\activate.bat
```
On Linux or macOS:
```bash
source .venv/bin/activate
```

### Step 3: Install dependencies
```bash
python -m pip install -r requirements.txt
```
If dependency installation fails, review the error and confirm that the pinned `scikit-learn` and `joblib` versions are available for your Python environment.

### Step 4: Run the smoke tests
From the repository root:
```bash
python tests/test_smoke.py
```
The test script checks model loading, metadata consistency, feature inputs, prediction outputs, threshold logic, and handling of missing required columns.
A successful run ends with:
```text
ALL SMOKE TESTS PASSED SUCCESSFULLY!
```

### Step 5: Start the FastAPI backend
```bash
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```
Available local endpoints:
- API information: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Interactive API documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- Health check: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Keep this terminal running while using the frontend.

### Step 6: Start the Streamlit frontend
Open a second terminal, activate the same virtual environment, and run:
```bash
streamlit run streamlit_app.py
```
Streamlit opens in your browser at:
[http://localhost:8501](http://localhost:8501/)

For local development, the frontend connects to the backend at `http://127.0.0.1:8000`. For cloud deployment, configure the backend URL as described below.

---

## API Reference

The FastAPI backend exposes the following endpoints:

| Method | Endpoint | Purpose |
| :--- | :--- | :--- |
| `GET` | `/` | Returns basic API information |
| `GET` | `/health` | Checks whether the model bundle is loaded |
| `POST` | `/predict` | Predicts churn for one customer |
| `POST` | `/predict/batch` | Predicts churn for multiple customers |

### Required Prediction Fields
A prediction request must include the following 14 fields:
```json
{
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
```

### Test a Single Prediction
With the backend running locally, send a request using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
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

### Prediction Response Format
A successful request returns these fields:
```json
{
  "churn_probability": 0.0,
  "predicted_class": 0,
  "predicted_label": "no",
  "threshold_used": 0.381
}
```
*Note: Probabilities and the threshold are rounded to four decimal places in the API response.*

### Batch Predictions
Send a JSON array of customer objects to `POST /predict/batch`. Each object must contain the same required fields as a single-customer request.
A successful response contains a count and a predictions array, with one prediction result per submitted customer.

---

## Deploy to the Cloud

The application can be deployed as two services:
- **Backend:** Render Web Service
- **Frontend:** Streamlit Community Cloud

Deploy the backend first so you can configure the frontend with its actual public URL.

### A. Deploy the FastAPI Backend on Render
1. Open [Render](https://render.com/) and sign in.
2. Create a new **Web Service**.
3. Connect the GitHub repository: `yash-upadhyay-2003/ChurnPredAI`
4. Configure the service:
   - **Service name:** `churn-prediction-api` (or preferred name)
   - **Branch:** `main`
   - **Runtime:** `Python 3`
   - **Build command:** `pip install -r requirements.txt`
   - **Start command:** `uvicorn api:app --host 0.0.0.0 --port $PORT`
5. Start the deployment and inspect the build logs.
6. Copy the assigned public Render service URL (e.g., `https://your-service-name.onrender.com`).

### B. Verify the Backend
Open the health check in your browser or terminal:
```bash
curl -X GET "https://YOUR-RENDER-SERVICE.onrender.com/health"
```
A successful health check returns HTTP 200:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "selected_model": "Random Forest — selected features",
  "decision_threshold": 0.3809723161404831
}
```
You can also visit the interactive documentation at:
`https://YOUR-RENDER-SERVICE.onrender.com/docs`

### C. Deploy the Streamlit Frontend
1. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
2. Select **New app**.
3. Choose the repository `yash-upadhyay-2003/ChurnPredAI`, branch `main`, and main file path:
   ```text
   streamlit_app.py
   ```
4. Open **Advanced settings** -> **Secrets**.
5. Add your Render backend URL:
   ```toml
   BACKEND_URL = "https://YOUR-RENDER-SERVICE.onrender.com"
   ```
   *(Use only the base URL without trailing slashes, `/health`, or `/predict`).*
6. Click **Deploy**.

---

## Post-Deployment Verification

After both services are deployed, verify the following:
- [ ] The Render service finishes building and starts successfully.
- [ ] The `/health` endpoint returns HTTP 200 and `"model_loaded": true`.
- [ ] The `/docs` page loads interactive Swagger UI.
- [ ] A valid request to `/predict` returns a probability, class, label, and threshold.
- [ ] The Streamlit app opens without startup errors.
- [ ] The Streamlit app successfully communicates with the deployed backend.
- [ ] A prediction submitted through the frontend returns a result.
- [ ] Invalid requests produce a handled error rather than an unhandled application failure.

---

## Important Notes

- **Model compatibility:** The metadata records `scikit-learn 1.9.1` and `joblib 1.6.0`. These versions are pinned in `requirements.txt` to match the saved model.
- **Model loading:** The backend loads the saved model and metadata at startup. It does not retrain the model during prediction requests.
- **Input validation:** Prediction requests must include all 14 required input fields. The API validates types and numeric constraints defined in its request schema.
- **Prediction interpretation:** A churn probability is a statistical estimate, not a certainty. The decision threshold controls the returned class label.
- **Dataset permissions:** Before making the repository public or sharing the dataset, confirm that you have permission to publish and redistribute the data.
- **Deployment status:** The instructions in this README describe how to deploy the project. They do not, by themselves, establish that the application has been deployed or verified on a live cloud service.

---

## License

Add a license if you intend to distribute this project under specific reuse terms. Do not assume that a public GitHub repository automatically grants permission to reuse its dataset or model.
