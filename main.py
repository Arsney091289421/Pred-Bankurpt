from fastapi import FastAPI
from fastapi.responses import JSONResponse
import pandas as pd
import cloudpickle

app = FastAPI()

model_path = "pipeline!.pkl"
with open(model_path, "rb") as f:
    model = cloudpickle.load(f)
print(f"Model loaded from {model_path}")

# def selected features
selected_features = [
    " Net Income to Stockholder's Equity",
    " Interest-bearing debt interest rate",
    " Persistent EPS in the Last Four Seasons",
    " Quick Ratio",
    " ROA(B) before interest and depreciation after tax",
    " Allocation rate per person",
    " Accounts Receivable Turnover",
    " Non-industry income and expenditure/revenue",
    " Average Collection Days",
    " Cash/Total Assets",
    " Operating Expense Rate",
    " Net worth/Assets",
    " Revenue per person",
    " Total Asset Growth Rate",
    " Quick Assets/Current Liability",
    " Net Value Growth Rate",
    " Total debt/Total net worth",
    " Interest Coverage Ratio (Interest expense to EBIT)",
    " Current Asset Turnover Rate",
    " Inventory/Working Capital",
    " Cash Turnover Rate",
    " Inventory/Current Liability",
    " Degree of Financial Leverage (DFL)",
    " Retained Earnings to Total Assets",
    " Inventory Turnover Rate (times)",
    " Total assets to GNP price",
    " Research and development expense rate"
]


@app.post("/predict-json/")
async def predict_json_endpoint(data: dict):
    """
    get input and give predictions
    """
    # transfer to df
    df = pd.DataFrame([data])

    # check features
    missing_features = [feature for feature in selected_features if feature not in df.columns]
    if missing_features:
        return JSONResponse(
            content={"error": "Missing required features", "missing_features": missing_features}, status_code=400
        )

    
    X = df[selected_features]

    # pred
    
    probabilities = model.predict_proba(X)
    threshold = 0.3
    predictions = (probabilities[:, 1] >= threshold).astype(int)

    # result
    results = {
        "Probability_Normal (%)": round(probabilities[0][0] * 100, 2),
        "Probability_Bankrupt (%)": round(probabilities[0][1] * 100, 2),
        "Predicted_Class": int(predictions[0]),
        "Prediction_Label": "Bankrupt" if predictions[0] == 1 else "Normal"
    }

    return JSONResponse(content=results)
