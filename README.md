
# **Bankruptcy Prediction API**

This project implements a **CatBoost** model to predict bankruptcy based on financial indicators. The model is deployed as an API using **FastAPI**, allowing users to input financial data and receive predictions about bankruptcy probabilities and classifications.

---

## **1. Dataset**
This project uses the [Company Bankruptcy Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/company-bankruptcy-prediction), sourced from Kaggle. The dataset contains various financial features designed to assess company health and predict potential bankruptcy risks.

---

## **2. Model Overview**
- The model leverages the **CatBoostClassifier**, which is highly efficient for datasets with a large number of features and class imbalances.
- The model is optimized for performance and integrated with **FastAPI** for easy deployment and usage.

---

## **3. Dataset Features**
The dataset contains 96 financial features. Some key features include:
- **Net Income to Stockholder's Equity**
- **Interest-bearing Debt Interest Rate**
- **Persistent EPS in the Last Four Seasons**
- **ROA(B) Before Interest and Depreciation After Tax**
- **Quick Ratio**
- **Cash/Total Assets**
- **Operating Expense Rate**
- **Net Worth/Assets**
- **Revenue per Person**
- **Inventory Turnover Rate (times)**

These features provide a comprehensive view of a company's financial health, which helps in effectively predicting bankruptcy risks.

---

## **4. Data Splitting**
The dataset was split into three parts:
- **Out-of-Sample Data (OOS):** 10% of the dataset was reserved as out-of-sample data for endpoint testing. A small sample size is sufficient for this purpose.
- **Training and Validation Split:** The remaining 90% of the data was split into **80% training / 20% validation**, a standard and effective ratio.

---

## **5. Feature Selection**
- **Feature Importance Calculation:**
  - Both CatBoost and XGBoost models were trained to compute feature importance scores.
  - The importance scores were scaled to a common range, averaged, and the top 30 most important features were selected.
- **Multicollinearity Removal:**
  - Features with a correlation coefficient above **0.8** were removed to reduce redundancy and improve model robustness.

---

## **6. Feature Generation**
No additional feature generation was performed. The original dataset, with 96 features, was sufficient for effective modeling.
While new feature generation could marginally improve performance in some cases, it is unlikely to provide significant benefits given the current dataset.

---

## **7. Hyperparameter Optimization**
- **Bayesian Optimization** was used to fine-tune the model's hyperparameters efficiently.
- The best hyperparameters were saved directly into the pipeline for easy reproducibility.

---

## **8. Pipeline**
The pipeline includes:
- **Optimized hyperparameters**
- **Class weights** to handle class imbalance
- **CatBoost classifier**
The pipeline is ready to use without requiring further adjustments.

---

## **9. API Implementation**
The model is deployed using **FastAPI**, providing a simple and user-friendly interface for predictions.

### **API Features:**
1. **Input:**
   - Users can submit financial data in JSON format.
2. **Processing:**
   - The API processes the data through the model pipeline.
3. **Output:**
   - Returns probabilities for bankruptcy and non-bankruptcy.
   - Provides the final classification (Bankrupt/Normal).

---

## **10. Requirements**
- Python 3.8+
- FastAPI
- CatBoost
- All dependencies are listed in the `requirements.txt` file.

---

## **11. How to Run**

### **Step 1: Clone the Repository**
```bash
git clone https://github.com/Arsney091289421/Pred-Bankurpt.git
cd Pred-Bankurpt
```

### **Step 2: Install Dependencies**
```bash
pip install -r requirements.txt
```

### **Step 3: Start the API**
```bash
uvicorn main:app --reload
```

### **Step 4: Access the API**
Open your browser and navigate to [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs) to use the interactive documentation for testing the API.

---

## **12. Output Explanation**
- **Probability_Normal (%):** The probability that the input data represents a non-bankrupt company.
- **Probability_Bankrupt (%):** The probability that the input data represents a bankrupt company.
- **Predicted Class:** The predicted classification: `Normal` or `Bankrupt`.

---

## **13. Model Strengths and Weaknesses**

### **Strengths:**
- **Simplicity and Efficiency:**
  - Avoids complex custom classes, making it user-friendly.
  - The optimized pipeline enables quick predictions.
- **Feature Selection:**
  - Effectively identifies the most relevant financial features, improving focus and model performance.

### **Weaknesses:**
- **Accuracy:**
  - While the model performs well, further improvement is possible, particularly for predicting minority classes (e.g., bankrupt companies).

---

## **14. Challenges**

### **Initial Issues:**
- Initial attempts used complex custom classes for feature selection, which caused numerous errors during API deployment.

### **Solution:**
- Simplified the process by calculating feature importance directly, removing redundant features, and passing parameters directly into the pipeline. This approach was simpler and eliminated errors.


