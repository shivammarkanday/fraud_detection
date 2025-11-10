# app/app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from io import StringIO
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score
)
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Credit Card Fraud Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Helpers ----------
@st.cache_resource
def load_model_and_scaler():
    model_path = os.path.join("models", "rf_fraud_model.joblib")
    scaler_path = os.path.join("models", "scaler.joblib")
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model, scaler

def preprocess_input(df_in, scaler):
    # assume df_in columns are V1..V28, Amount (no Time)
    cols_expected = [f"V{i}" for i in range(1,29)] + ["Amount"]
    df = df_in.copy()
    # if Time present drop it
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
    # keep only expected columns (in order), fill missing with 0
    df = df.reindex(columns=cols_expected, fill_value=0)
    X_scaled = scaler.transform(df.values)
    return X_scaled, df

def predict_batch(df_in, model, scaler):
    X, df_clean = preprocess_input(df_in, scaler)
    probs = model.predict_proba(X)[:,1]
    preds = model.predict(X)
    out = df_clean.copy()
    out["predicted"] = preds
    out["fraud_prob"] = np.round(probs, 4)
    return out

def pretty_label(prob, thresh=0.5):
    return "FRAUD" if prob >= thresh else "LEGIT"

# ---------- Load model ----------
with st.spinner("Loading model..."):
    model, scaler = load_model_and_scaler()

# ---------- Layout ----------
st.markdown(
    """
    <style>
      .big-title {font-size:32px; font-weight:700;}
      .muted {color:#6c757d;}
    </style>
    """, unsafe_allow_html=True
)

col1, col2 = st.columns([3,1])
with col1:
    st.markdown('<div class="big-title">🚨 Credit Card Fraud Detection</div>', unsafe_allow_html=True)
    st.write("Interactive demo — upload transactions or test single records. Model: Random Forest.")
with col2:
    st.image("img/scrooge_mcduck_s_face_by_adrianapendleton_detahh0.png", width=60)

st.markdown("---")

# ---------- Sidebar controls ----------
st.sidebar.header("App Controls")
show_shap = st.sidebar.checkbox("Enable SHAP explanations (if available)", value=False)
prob_thresh = st.sidebar.slider("Fraud probability threshold", 0.01, 0.99, 0.5, 0.01)
st.sidebar.markdown("**Model info**")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Trained with SMOTE on training set")

# ---------- Main: Upload / Sample / Manual ----------
left, right = st.columns([2,1])

with left:
    st.subheader("1) Upload transactions (CSV) or test with sample data")
    uploaded = st.file_uploader("Upload CSV file (must include V1..V28 and Amount, optional Time, optional Class)", type=["csv"])
    sample_btn = st.button("Load random sample from data/creditcard.csv")
    if uploaded:
        try:
            df_user = pd.read_csv(uploaded)
        except Exception as e:
            st.error("Failed to read CSV: " + str(e))
            df_user = None
    elif sample_btn:
        try:
            df_user = pd.read_csv("data/sample_creditcard.csv")
            df_user = df_user.sample(200, random_state=42).reset_index(drop=True)
            st.success("Loaded random sample (200 rows) from data/creditcard.csv")
        except Exception as e:
            st.error("Could not load sample dataset. Make sure data/creditcard.csv exists in repo root.")
            df_user = None
    else:
        df_user = None

    if df_user is not None:
        st.write("Preview uploaded data:")
        st.dataframe(df_user.head())

        if st.button("Run batch prediction"):
            out = predict_batch(df_user, model, scaler)
            st.success("Predictions ready")
            st.dataframe(out.head())
            csv = out.to_csv(index=False)
            st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

            # If ground truth present, show evaluation
            if "Class" in df_user.columns:
                y_true = df_user["Class"].values
                y_pred = out["predicted"].values
                cm = confusion_matrix(y_true, y_pred)
                p = precision_score(y_true, y_pred, zero_division=0)
                r = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                probs = out["fraud_prob"].values
                fpr, tpr, _ = roc_curve(y_true, probs)
                roc_auc = auc(fpr, tpr)

                st.subheader("Batch evaluation (using uploaded Class column)")
                c1, c2, c3 = st.columns(3)
                c1.metric("Precision", f"{p:.4f}")
                c2.metric("Recall", f"{r:.4f}")
                c3.metric("F1", f"{f1:.4f}")

                # Confusion matrix heatmap (plotly)
                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Pred 0","Pred 1"],
                    y=["True 0","True 1"],
                    colorscale="Blues",
                    hoverongaps=False,
                ))
                fig_cm.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_cm, use_container_width=True)

                # ROC curve
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.4f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)

with right:
    st.subheader("2) Predict single transaction")
    st.info("Fill values or press ‘Load random sample’ to auto-fill fields.")

    # Build dynamic form fields for V1..V28 + Amount
    with st.form("single_form"):
        cols = st.columns([1,1])
        sample_single = st.form_submit_button("Load random single sample from dataset")
        # create placeholders for inputs
        inputs = {}
        for i in range(1,29):
            key = f"V{i}"
            inputs[key] = st.number_input(key, value=0.0, format="%.6f", step=0.1, key=key)
        amount = st.number_input("Amount", value=0.0, format="%.2f", key="Amount")
        submit_single = st.form_submit_button("Predict")

    # allow sample auto-fill (non-blocking)
    if sample_single:
        try:
            df_all = pd.read_csv("data/sample_creditcard.csv")

            row = df_all.sample(1, random_state=42).iloc[0]
            for i in range(1,29):
                st.session_state[f"V{i}"] = float(row[f"V{i}"])
            st.session_state["Amount"] = float(row["Amount"])
            st.success("Loaded random sample into the form. Re-run Predict.")
        except Exception:
            st.warning("Could not load sample dataset.")

    if submit_single:
        # build DataFrame from inputs
        data = {f"V{i}": st.session_state.get(f"V{i}", 0.0) for i in range(1,29)}
        data["Amount"] = st.session_state.get("Amount", 0.0)
        df_single = pd.DataFrame([data])
        Xs, df_clean = preprocess_input(df_single, scaler)
        prob = model.predict_proba(Xs)[:,1][0]
        pred = model.predict(Xs)[0]
        label = pretty_label(prob, prob_thresh)

        # Verdict display
        if prob >= prob_thresh:
            st.markdown(f"### 🚨 Prediction: **{label}**  — probability **{prob:.4f}**", unsafe_allow_html=True)
        else:
            st.markdown(f"### ✅ Prediction: **{label}**  — probability **{prob:.4f}**", unsafe_allow_html=True)

        # SHAP explanation (if user enabled)
        if show_shap:
            try:
                import shap
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(Xs)
                st.subheader("SHAP explanation (approx.)")
                st.pyplot(shap.plots.force(explainer.expected_value[1], shap_vals[1][0], feature_names=df_clean.columns))
            except Exception as e:
                st.warning("SHAP not available or failed: " + str(e))

st.markdown("---")
# ---------- Footer: Feature importances ----------
st.subheader("Model insights")
try:
    fi = model.feature_importances_
    feat_names = [f"V{i}" for i in range(1,29)] + ["Amount"]
    df_fi = pd.DataFrame({"feature": feat_names, "importance": fi})
    df_fi = df_fi.sort_values("importance", ascending=False).reset_index(drop=True)
    fig = px.bar(df_fi.head(12), x="importance", y="feature", orientation="h", title="Top 12 feature importances")
    st.plotly_chart(fig, use_container_width=True)
except Exception as e:
    st.write("Feature importances not available: " + str(e))

st.markdown("Made with ❤️  •  Deploy with Streamlit Cloud / Render")
