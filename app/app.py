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
    cols_expected = [f"V{i}" for i in range(1,29)] + ["Amount"]
    df = df_in.copy()
    if "Time" in df.columns:
        df = df.drop(columns=["Time"])
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
    try:
        model, scaler = load_model_and_scaler()
    except Exception as e:
        st.error("Failed to load model or scaler. Make sure models are in /models folder.")
        st.stop()

# ---------- Page header ----------
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
    img_path = "img/scrooge_mcduck_s_face_by_adrianapendleton_detahh0.png"
with col2:
    # optional image if present
    # img_path = "/Users/shivammarkanday/Library/CloudStorage/OneDrive-AdaniUniversity/SEMESTERS/SEM 5 (2025)/Machine Learning (ML)/Project/fraud_detection/img/scrooge_mcduck_s_face_by_adrianapendleton_detahh0.png"
    if os.path.exists(img_path):
        st.image(img_path, width=120)

st.markdown("---")

# ---------- Sidebar controls ----------
st.sidebar.header("App Controls")
show_shap = st.sidebar.checkbox("Enable SHAP explanations (if available)", value=False)
prob_thresh = st.sidebar.slider("Fraud probability threshold", 0.01, 0.99, 0.5, 0.01)
st.sidebar.markdown("**Model info**")
st.sidebar.write("Model: Random Forest")
st.sidebar.write("Trained with SMOTE on training set")

# ensure session_state keys exist
if "df_user" not in st.session_state:
    st.session_state.df_user = None
for i in range(1,29):
    if f"V{i}" not in st.session_state:
        st.session_state[f"V{i}"] = 0.0
if "Amount" not in st.session_state:
    st.session_state["Amount"] = 0.0

# ---------- Main: Upload / Sample / Manual (STACKED LAYOUT) ----------
st.subheader("1) Upload transactions (CSV) or test with sample data")

uploaded = st.file_uploader(
    "Upload CSV (must include V1..V28 and Amount; optional Time, Class)",
    type=["csv"],
    key="uploader_left"
)

# If user uploaded a custom file, save into session_state immediately
if uploaded is not None:
    try:
        st.session_state.df_user = pd.read_csv(uploaded)
        st.success("Uploaded file loaded and stored.")
    except Exception as e:
        st.error("Failed to read CSV: " + str(e))
        st.session_state.df_user = None

# Load sample button: put dataframe into session_state (persistent)
if st.button("Load random data from sample_creditcard.csv", key="load_sample_left"):
    try:
        df_sample = pd.read_csv("data/sample_creditcard.csv")
        # keep sample small for UI responsiveness
        if len(df_sample) > 200:
            df_sample = df_sample.sample(200, random_state=42).reset_index(drop=True)
        st.session_state.df_user = df_sample
        st.success("Loaded sample dataset and saved to session.")
    except Exception as e:
        st.error("Could not load sample dataset: " + str(e))
        st.session_state.df_user = None

# Preview current session dataset (if any)
if st.session_state.df_user is not None:
    st.write("### 🔍 Data Preview (first 10 rows):")
    st.dataframe(st.session_state.df_user.head(10))

    # --- Clear loaded dataset button ---
    if st.button("Clear loaded dataset", key="clear_loaded"):
        # clear the stored dataframe
        st.session_state.df_user = None

        # also clear the file_uploader widget state (so the upload disappears)
        if "uploader_left" in st.session_state:
            try:
                st.session_state["uploader_left"] = None
            except Exception:
                # fallback: ignore if we can't clear it
                pass

        st.success("Cleared loaded dataset and upload.")
        # refresh UI so preview and upload control update immediately
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass

    # ---------- Batch prediction block (now under preview) ----------
    st.markdown("")  # spacing
    st.subheader(" Run batch prediction (on uploaded / loaded CSV)")
    st.write(f"Dataset loaded: {len(st.session_state.df_user)} rows")
    if st.button("Run batch prediction", key="run_batch_left"):
        try:
            out = predict_batch(st.session_state.df_user, model, scaler)
            st.success("✅ Batch predictions completed.")
            st.dataframe(out.head(10))
            csv = out.to_csv(index=False)
            st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

            # evaluation if 'Class' present
            if "Class" in st.session_state.df_user.columns:
                y_true = st.session_state.df_user["Class"].values
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

                fig_cm = go.Figure(data=go.Heatmap(
                    z=cm,
                    x=["Pred 0","Pred 1"],
                    y=["True 0","True 1"],
                    colorscale="Blues",
                    hoverongaps=False,
                ))
                fig_cm.update_layout(title="Confusion Matrix", xaxis_title="", yaxis_title="")
                st.plotly_chart(fig_cm, use_container_width=True)

                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.4f}"))
                fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", line=dict(dash="dash"), name="Random"))
                fig_roc.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
                st.plotly_chart(fig_roc, use_container_width=True)
        except Exception as e:
            st.error("Batch prediction failed: " + str(e))
else:
    st.info("No dataset loaded. Upload a CSV or load the sample.")

st.markdown("---")

# ---------- Predict single transaction (STACKED, below batch section) ----------
st.subheader("2) Predict single transaction")
st.info("Fill values or press ‘Load random single sample’ to auto-fill fields.")

# ---- Load single sample button (OUTSIDE form) ----
if st.button("Load random single sample from data/sample_creditcard.csv", key="load_single"):
    try:
        df_all = pd.read_csv("data/sample_creditcard.csv")
        row = df_all.sample(1).iloc[0]

        # write into both fallback keys and widget keys so inputs update reliably
        for i in range(1, 29):
            val = float(row[f"V{i}"])
            st.session_state[f"V{i}"] = val
            st.session_state[f"V{i}_input"] = val
        amt = float(row["Amount"])
        st.session_state["Amount"] = amt
        st.session_state["Amount_input"] = amt

        st.success("Loaded random sample into the form. Press Predict.")
        # rerun to reflect session_state values in the form inputs
        try:
            st.rerun()
        except Exception:
            try:
                st.experimental_rerun()
            except Exception:
                pass
    except Exception as e:
        st.error("Could not load sample: " + str(e))

# Single transaction form (uses session_state stored values)
with st.form("single_form"):
    for i in range(1, 29):
        key = f"V{i}"
        # use separate widget keys so we can read form values distinctly
        st.number_input(key, value=st.session_state.get(key, 0.0),
                        format="%.6f", step=0.1, key=f"{key}_input")
    st.number_input("Amount", value=st.session_state.get("Amount", 0.0),
                    format="%.2f", key="Amount_input")

    submit_single = st.form_submit_button("Predict")

# handle predict action
if submit_single:
    # read values from form keys first, fallback to session_state if needed
    data = {}
    for i in range(1,29):
        form_key = f"V{i}_input"
        data[f"V{i}"] = st.session_state.get(form_key, st.session_state.get(f"V{i}", 0.0))
    data["Amount"] = st.session_state.get("Amount_input", st.session_state.get("Amount", 0.0))

    df_single = pd.DataFrame([data])
    Xs, df_clean = preprocess_input(df_single, scaler)
    prob = model.predict_proba(Xs)[:,1][0]
    pred = model.predict(Xs)[0]
    label = pretty_label(prob, prob_thresh)

    if prob >= prob_thresh:
        st.markdown(f"### 🚨 Prediction: **{label}**  — probability **{prob:.4f}**", unsafe_allow_html=True)
    else:
        st.markdown(f"### ✅ Prediction: **{label}**  — probability **{prob:.4f}**", unsafe_allow_html=True)

    if show_shap:
        try:
            import shap
            explainer = shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(Xs)
            st.subheader("SHAP explanation (approx.)")
            # fallback plotting; SHAP may require matplotlib backend
            st.pyplot(shap.plots.force(explainer.expected_value[1], shap_vals[1][0], feature_names=df_clean.columns))
        except Exception as e:
            st.warning("SHAP not available or failed: " + str(e))

# ---------- Footer: Feature importances ----------
st.markdown("---")
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
