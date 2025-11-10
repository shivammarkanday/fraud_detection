# ---------- Main: Upload / Sample / Manual (REPLACEMENT BLOCK) ----------
col_left, col_right = st.columns([2, 1])

# LEFT: Upload / sample preview only
with col_left:
    st.subheader("1) Upload or Load Sample Data")
    if "df_user" not in st.session_state:
        st.session_state.df_user = None

    uploaded = st.file_uploader(
        "Upload CSV (must include V1..V28 and Amount; optional Time, Class)",
        type=["csv"],
        key="uploader"
    )

    if uploaded is not None:
        try:
            st.session_state.df_user = pd.read_csv(uploaded)
            st.success("✅ CSV uploaded and stored.")
        except Exception as e:
            st.error("❌ Failed to read CSV: " + str(e))
            st.session_state.df_user = None

    if st.button("📂 Load random sample from data/sample_creditcard.csv", key="load_sample"):
        try:
            df_sample = pd.read_csv("data/sample_creditcard.csv")
            # sample for UI responsiveness if file is large
            if len(df_sample) > 200:
                df_sample = df_sample.sample(200, random_state=42).reset_index(drop=True)
            st.session_state.df_user = df_sample
            st.success("✅ Loaded sample dataset")
        except Exception as e:
            st.error("❌ Could not load sample dataset: " + str(e))
            st.session_state.df_user = None

    if st.session_state.df_user is not None:
        st.write("### 🔍 Data Preview (first 10 rows):")
        st.dataframe(st.session_state.df_user.head(10))
    else:
        st.info("Upload a CSV or load sample to preview data.")


# RIGHT: Single-transaction form + Batch prediction (moved here)
with col_right:
    st.subheader("2) Predict single transaction")
    st.info("Fill values or press ‘Load random single sample’ to auto-fill fields.")

    # Load a random single sample (button in right column)
    if st.button("Load random single sample from data/sample_creditcard.csv", key="load_single_sample"):
        try:
            df_all = pd.read_csv("data/sample_creditcard.csv")
            row = df_all.sample(1, random_state=42).iloc[0]
            for i in range(1, 29):
                st.session_state[f"V{i}"] = float(row[f"V{i}"])
            st.session_state["Amount"] = float(row["Amount"])
            st.success("Loaded random sample into the form. Press Predict.")
        except Exception as e:
            st.error("Could not load sample: " + str(e))

    # Single transaction form (uses session_state stored values)
    with st.form("single_form"):
        for i in range(1, 29):
            key = f"V{i}"
            st.number_input(key, value=st.session_state.get(key, 0.0),
                            format="%.6f", step=0.1, key=key + "_input")
        st.number_input("Amount", value=st.session_state.get("Amount", 0.0),
                        format="%.2f", key="Amount_input")

        submit_single = st.form_submit_button("Predict single transaction")

    if submit_single:
        # read values from session_state (note keys used above)
        data = {f"V{i}": st.session_state.get(f"V{i}_input", st.session_state.get(f"V{i}", 0.0)) for i in range(1,29)}
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
                st.pyplot(shap.plots.force(explainer.expected_value[1], shap_vals[1][0], feature_names=df_clean.columns))
            except Exception as e:
                st.warning("SHAP not available or failed: " + str(e))

    st.markdown("---")
    # Batch prediction button + output (moved below single-form)
    st.subheader("3) Run batch prediction (on uploaded / loaded CSV)")
    if st.session_state.df_user is None:
        st.info("Upload or load a sample dataset from the left column first.")
    else:
        st.write(f"Dataset loaded: {len(st.session_state.df_user)} rows")
        if st.button("Run batch prediction", key="run_batch"):
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
