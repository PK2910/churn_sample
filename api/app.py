import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel

# Not used directly below, but required: the pickled pipeline references
# these functions by module path ('preprocessing.clean_total_revenue', etc.),
# so this import must succeed before joblib.load can reconstruct it.
import preprocessing  # noqa: F401

# Loaded once when the server starts, not on every request — loading a model
# from disk is relatively slow, so doing it per-request would make the API
# unnecessarily sluggish.
xgb_pipeline = joblib.load("xgb_pipeline.joblib")
xgb_best_model = joblib.load("xgb_best_model.joblib")

app = FastAPI(title="Churn Prediction API")

# Chosen deliberately, not the sklearn default (0.5) — found via a
# precision-recall threshold search on the test set (recall ~0.866,
# precision ~0.592), leaning toward catching more churners at the cost of
# more false alarms.
CHURN_THRESHOLD = 0.038


# Mirrors the raw columns the training data had (minus 'Churn', the target).
# FastAPI uses this to validate incoming requests automatically — a request
# missing a field, or with the wrong type, gets rejected before your code
# ever runs.
class CustomerData(BaseModel):
    customerID: str
    gender: str
    SeniorCitizen: int
    MaritalStatus: str
    Dependents: str
    tenure: int
    PhoneService: str
    MultipleLines: str
    InternetService: str
    OnlineSecurity: str
    OnlineBackup: str
    DeviceProtection: str
    TechSupport: str
    StreamingTV: str
    StreamingMovies: str
    Contract: str
    PaperlessBilling: str
    PaymentMethod: str
    InternationalPlan: str
    VoiceMailPlan: str
    NumbervMailMessages: int
    TotalDayMinutes: float
    TotalDayCalls: int
    TotalEveMinutes: float
    TotalEveCalls: int
    TotalNightMinutes: float
    TotalNightCalls: int
    TotalIntlMinutes: float
    TotalIntlCalls: int
    CustomerServiceCalls: int
    TotalCall: int
    TotalRevenue: str

    # Shown as the pre-filled example on the /docs page, so clicking "Try it
    # out" then "Execute" works immediately, instead of Swagger's default
    # generic placeholder ("string" for every text field) that isn't a real
    # category the model has ever seen.
    model_config = {
        "json_schema_extra": {
            "example": {
                "customerID": "TEST-001",
                "gender": "Female",
                "SeniorCitizen": 0,
                "MaritalStatus": "Yes",
                "Dependents": "Yes",
                "tenure": 9,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "DSL",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "Yes",
                "StreamingTV": "Yes",
                "StreamingMovies": "No",
                "Contract": "One year",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Mailed check",
                "InternationalPlan": "No",
                "VoiceMailPlan": "No",
                "NumbervMailMessages": 0,
                "TotalDayMinutes": 168.8,
                "TotalDayCalls": 137,
                "TotalEveMinutes": 241.4,
                "TotalEveCalls": 107,
                "TotalNightMinutes": 204.8,
                "TotalNightCalls": 106,
                "TotalIntlMinutes": 15.5,
                "TotalIntlCalls": 4,
                "CustomerServiceCalls": 0,
                "TotalCall": 354,
                "TotalRevenue": "593.3",
            }
        }
    }


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(customer: CustomerData):
    # The pipeline expects a DataFrame shaped like the original training
    # data — one row here, built from the request body.
    row = pd.DataFrame([customer.model_dump()])

    processed = xgb_pipeline.transform(row)
    churn_probability = xgb_best_model.predict_proba(processed)[0, 1]

    return {
        "customerID": customer.customerID,
        "churn_probability": float(churn_probability),
        "churn_prediction": bool(churn_probability >= CHURN_THRESHOLD),
    }
