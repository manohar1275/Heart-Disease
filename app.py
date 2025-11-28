import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Heart Disease Prediction (XAI Enabled)", layout="wide")

# Load model
with open("heart_model.pkl", "rb") as f:
    model, scaler, feature_names = pickle.load(f)

st.title("❤️ Heart Disease Prediction System (XAI Enabled)")

feature_values = {}

col1, col2, col3 = st.columns(3)

with col1:
    feature_values["age"] = st.number_input("Age", min_value=0, max_value=95, step=1, format="%d")
    feature_values["sex"] = st.selectbox("Sex", [0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    feature_values["cp"] = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3])
    feature_values["trestbps"] = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, step=1, format="%d")
    feature_values["chol"] = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, step=1, format="%d")

with col2:
    feature_values["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl", [0, 1])
    feature_values["restecg"] = st.selectbox("Rest ECG Results (0–2)", [0, 1, 2])
    feature_values["thalach"] = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, step=1, format="%d")
    feature_values["exang"] = st.selectbox("Exercise Induced Angina", [0, 1])
    feature_values["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, format="%.1f")

with col3:
    feature_values["slope"] = st.selectbox("Slope of ST Segment (0–2)", [0, 1, 2])
    feature_values["ca"] = st.number_input("Number of Major Vessels (0–4)", min_value=0, max_value=4, step=1, format="%d")
    feature_values["thal"] = st.selectbox(
        "Thalassemia",
        [0, 1, 2],
        format_func=lambda x: ["Normal", "Fixed Defect", "Reversible Defect"][x]
    )

# ---------------------------------------------------
#               PREDICTION BUTTON
# ---------------------------------------------------
if st.button("🔍 Predict Heart Disease"):
    input_values = np.array([list(feature_values.values())])
    scaled_values = scaler.transform(input_values)

    prediction = model.predict(scaled_values)[0]
    probability = model.predict_proba(scaled_values)[0][1]

    if prediction == 1:
        st.error(f"⚠️ High Chance of Heart Disease ({probability*100:.2f}% probability)")
    else:
        st.success(f"💚 Low Chance of Heart Disease ({(1-probability)*100:.2f}% probability)")

    # ---------------- SHAP EXPLANATION AFTER PREDICTION ONLY ----------------
    st.subheader("📊 SHAP Explanation for this Prediction")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(scaled_values)

    # Normalize shap_values to a local per-sample, per-feature array for class=1
    if isinstance(shap_values, np.ndarray):
        # shape could be (n_samples, n_features, n_classes)
        if shap_values.ndim == 3:
            local_shap = shap_values[0, :, 1]
        elif shap_values.ndim == 2:
            local_shap = shap_values[0]
        else:
            local_shap = np.ravel(shap_values)[0:len(feature_names)]
    else:
        # older SHAP returns list of arrays, one per class
        try:
            local_shap = shap_values[1][0]
        except Exception:
            local_shap = shap_values[0][0]

    # expected value for class 1
    try:
        expected = explainer.expected_value[1]
    except Exception:
        # fallback if expected_value is scalar or differently shaped
        expected = np.array(explainer.expected_value).ravel()[-1]

    # Try JS force plot; if IPython/shap JS unavailable, fall back to matplotlib bar chart
    st.write("### Local Explanation (Single Prediction)")
    try:
        shap.initjs()
        fp = shap.force_plot(expected, local_shap, scaled_values[0], feature_names=feature_names, matplotlib=False)
        import tempfile, os
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
        tmp.close()
        shap.save_html(tmp.name, fp)
        html = open(tmp.name, 'r', encoding='utf-8').read()
        st.components.v1.html(html, height=350)
        os.remove(tmp.name)
    except Exception:
        # fallback: simple matplotlib horizontal bar chart
        st.write("(Falling back to static matplotlib chart)")
        fig, ax = plt.subplots(figsize=(6, 4))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, local_shap)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel('SHAP value')
        fig.tight_layout()
        st.pyplot(fig)

    st.write("### Feature Importance for This Prediction")
    try:
        # Try SHAP bar plot (modern API)
        exp = shap.Explanation(values=local_shap, base_values=expected, data=scaled_values[0], feature_names=feature_names)
        shap.plots.bar(exp, show=False)
        st.pyplot(plt.gcf())
    except Exception:
        # Simple matplotlib fallback
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        ax2.barh(feature_names, local_shap)
        ax2.set_xlabel('SHAP value')
        fig2.tight_layout()
        st.pyplot(fig2)
