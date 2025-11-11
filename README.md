# Credit Card Fraud Detection App

A fully interactive **Streamlit web application** that detects fraudulent credit card transactions using a **Random Forest Classifier** trained with **SMOTE** for class balancing.

This project lets you upload transaction data, visualize predictions, evaluate model metrics, and even test single transactions — all inside one smooth, scrollable interface.

---

## Overview

Credit card fraud is a growing problem in the digital economy.  
This app demonstrates how **machine learning** can be used to identify potentially fraudulent transactions based on anonymized features (`V1–V28`) and the `Amount` column.

Built using:
-  **Python**
-  **Scikit-Learn**
-  **Plotly**
-  **Streamlit**
-  **Random Forest Model**
-  **SMOTE** for handling imbalanced data

---

## Features

| Feature | Description |
|----------|--------------|
| **Upload Custom CSV** | Upload your dataset with columns `V1–V28`, `Amount`, and optionally `Class`. |
| **Batch Prediction** | Run fraud detection across all uploaded transactions. |
| **Evaluation Metrics** | Auto-calculates Precision, Recall, and F1 Score (if `Class` column present). |
| **Confusion Matrix & ROC Curve** | Interactive Plotly visualizations to analyze model performance. |
| **Predict Single Transaction** | Enter or auto-load a single transaction’s values and see instant prediction results. |
| **SHAP Explanations (Optional)** | Enable interpretable ML with SHAP (if available locally). |
| **Feature Importance Chart** | View top features contributing to model predictions. |

---

## UI Layout

The interface is clean and scrollable:

1. **Upload Transactions / Load Sample**
   - Upload your own CSV or load the sample dataset.
   - View top 10 rows in preview.
   - Clear dataset easily with one click.

2. **Run Batch Prediction**
   - Predict and download results.
   - Visualize confusion matrix + ROC Curve.
   - View Precision, Recall, F1 metrics.

3. **Predict Single Transaction**
   - Fill values manually or auto-load a random transaction.
   - Instantly see whether it’s **🚨 Fraud** or **✅ Legit**.
   - Optional SHAP explanation for model transparency.

4. **Model Insights**
   - Bar chart showing top 12 feature importances.

---

## Input Format

Your uploaded CSV must contain:
`V1, V2, V3, ..., V28, Amount, [optional: Time, Class]`

**Example:**
```csv
Time,V1,V2,V3,...,V28,Amount,Class
0,-1.3598071336738,-0.0727811733098497,...,0.0,149.62,0
```

## Installation & Setup

### 1 Clone this Repository
```bash
git clone https://github.com/your-username/creditcard-fraud-detection-app.git
cd creditcard-fraud-detection-app
```

### 2 Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # (Mac/Linux)
venv\Scripts\activate     # (Windows)
```

### 3 Install Dependencies
```bash
pip install -r requirements.txt
```


### 4 Run the App 
```bash
streamlit run app/app.py
```

### 5 Access it Locally
```bash
🌐 Visit: http://localhost:8501
```

## Project Structure

```bash
creditcard-fraud-detection-app/
├── app/
│   ├── app.py                 # Main Streamlit app
│   ├── img/                   # Images & icons
│   └── data/
│       └── sample_creditcard.csv
│
├── models/
│   ├── rf_fraud_model.joblib  # Trained Random Forest model
│   └── scaler.joblib          # Fitted StandardScaler
│
├── requirements.txt
└── README.md
```
## Model Details

- **Algorithm:** Random Forest Classifier  
- **Preprocessing:** StandardScaler  
- **Balancing Technique:** SMOTE (Synthetic Minority Oversampling)  
- **Evaluation Metrics:** Precision, Recall, F1, ROC AUC  
- **Dataset:** Trained on the Kaggle Credit Card Fraud Detection dataset  

## Tech Stack

| **Layer** | **Technology** |
|------------|----------------|
| **Frontend** | Streamlit, Plotly |
| **Backend** | Scikit-learn, Pandas, NumPy |
| **Model** | Random Forest Classifier |
| **Visualization** | Plotly, SHAP |
| **Deployment** | Streamlit Cloud / Render |

## Author

**Shivam Markanday**  
**B.Tech CSE (AI/ML)** | Adani University  

Building intelligent, user-focused apps that bridge data science & usability.  
Reach me on [LinkedIn](https://www.linkedin.com/in/shivammarkanday/)
