import streamlit as st
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import tempfile
import os

st.set_page_config(page_title="Heart Disease Prediction (XAI Enabled) - Debug", layout="wide")

# --------------------------- CONFIG: expected features ---------------------------
# Use the canonical feature order your model was trained with (13 features)
EXPECTED_FEATURE_ORDER = [
    "age", "sex", "cp", "trestbps", "chol",
    "fbs", "restecg", "thalach", "exang",
    "oldpeak", "slope", "ca", "thal"
]

# --------------------------- LOAD MODEL & SCALER ---------------------------
with open("heart_model.pkl", "rb") as f:
    model, scaler, feature_names_from_pickle = pickle.load(f)

# If pickle mistakenly included target or wrong list, override or clean it.
feature_names = list(feature_names_from_pickle) if feature_names_from_pickle is not None else []
if "target" in feature_names:
    feature_names.remove("target")

# If lengths mismatch or feature_names suspicious, force EXPECTED_FEATURE_ORDER
if len(feature_names) != len(EXPECTED_FEATURE_ORDER):
    st.warning(
        "feature_names length from pickle doesn't match expected. "
        "Forcing canonical EXPECTED_FEATURE_ORDER to avoid misalignment."
    )
    feature_names = EXPECTED_FEATURE_ORDER.copy()
else:
    # Ensure names match expected order; if not, use expected order (safer).
    if feature_names != EXPECTED_FEATURE_ORDER:
        st.info("Reordering features to canonical EXPECTED_FEATURE_ORDER to match model training.")
        feature_names = EXPECTED_FEATURE_ORDER.copy()

# --------------------------- Basic diagnostics visible in UI ---------------------------
st.title("❤️ Heart Disease Prediction — (Robust SHAP + Diagnostics)")
st.markdown("This app enforces feature order and prints diagnostics to avoid identical SHAP explanations.")

st.subheader("Model / Scaler diagnostics (visible for debugging)")
try:
    n_in = model.n_features_in_
except Exception:
    # some models may not expose this attribute
    n_in = None

st.write("**Model n_features_in_**:", n_in)
st.write("**Scaler shape (if available)**:", getattr(scaler, "mean_", "scaler has no mean_ attribute"))
st.write("**Using feature order**:", feature_names)
st.write("**Expected feature order**:", EXPECTED_FEATURE_ORDER)

# Warn if model expected features mismatch with our feature list
if n_in is not None and n_in != len(feature_names):
    st.error(
        f"Mismatch: model expects {n_in} features but feature_names has {len(feature_names)}. "
        "This will cause wrong SHAP values. Check training pipeline & pickle file."
    )

# --------------------------- INPUT UI ---------------------------
feature_values = {}
col1, col2, col3 = st.columns(3)

with col1:
    feature_values["age"] = st.number_input("Age", min_value=0, max_value=95, step=1, value=55)
    feature_values["sex"] = st.selectbox("Sex (0=Female, 1=Male)", [0, 1], index=1)
    feature_values["cp"] = st.selectbox("Chest Pain Type (0–3)", [0, 1, 2, 3], index=1)
    feature_values["trestbps"] = st.number_input("Resting BP (mm Hg)", min_value=80, max_value=200, value=140)

with col2:
    feature_values["chol"] = st.number_input("Cholesterol (mg/dl)", min_value=100, max_value=600, value=240)
    feature_values["fbs"] = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1=True)", [0, 1], index=0)
    feature_values["restecg"] = st.selectbox("Rest ECG Results (0–2)", [0, 1, 2], index=0)
    feature_values["thalach"] = st.number_input("Max Heart Rate Achieved", min_value=60, max_value=220, value=150)

with col3:
    feature_values["exang"] = st.selectbox("Exercise Induced Angina", [0, 1], index=0)
    feature_values["oldpeak"] = st.number_input("Oldpeak (ST Depression)", min_value=0.0, max_value=10.0, step=0.1, value=1.0)
    feature_values["slope"] = st.selectbox("Slope (0–2)", [0, 1, 2], index=1)
    feature_values["ca"] = st.number_input("Major Vessels (0–4)", min_value=0, max_value=4, step=1, value=0)
    feature_values["thal"] = st.selectbox("Thal (0=Normal,1=Fixed,2=Reversible)", [0, 1, 2], index=2)

# --------------------------- PREDICTION BUTTON ---------------------------
if st.button("🔍 Predict Heart Disease"):

    # Build input array in the enforced order
    try:
        ordered_input = [feature_values[f] for f in feature_names]
    except KeyError as e:
        st.error(f"Input building failed — missing feature: {e}. Make sure UI keys match feature_names.")
        st.stop()

    input_values = np.array([ordered_input], dtype=np.float64)

    st.write("### Input vector sent to scaler/model (ordered):")
    st.write(dict(zip(feature_names, input_values[0])))

    # Scale input
    try:
        scaled_values = scaler.transform(input_values)
    except Exception as e:
        st.error(f"Scaler.transform failed: {e}")
        st.stop()

    # Predict
    try:
        prediction = model.predict(scaled_values)[0]
        proba = model.predict_proba(scaled_values)[0] if hasattr(model, "predict_proba") else None
    except Exception as e:
        st.error(f"Model prediction failed: {e}")
        st.stop()

    probability = proba[1] if proba is not None else None

    if probability is not None:
        if prediction == 1:
            st.error(f"⚠️ High Chance of Heart Disease ({probability * 100:.2f}%)")
        else:
            st.success(f"💚 Low Chance of Heart Disease ({(1 - probability) * 100:.2f}%)")
    else:
        st.write("Prediction (no probability available):", prediction)

    # --------------------------- SHAP EXPLANATION ---------------------------
    st.subheader("📊 SHAP Explanation (local)")

    # Create or reuse explainer cache to avoid re-creation cost
    if "shap_explainer_cached" not in st.session_state:
        try:
            # Use TreeExplainer if available; fallback to generic Explainer
            explainer = shap.TreeExplainer(model)
        except Exception:
            explainer = shap.Explainer(model, masker=shap.maskers.Independent(scaled_values))
        st.session_state["shap_explainer_cached"] = explainer
    else:
        explainer = st.session_state["shap_explainer_cached"]

    # compute shap_values
    try:
        shap_values = explainer.shap_values(scaled_values)
    except Exception as e:
        st.error(f"shap_values computation failed: {e}")
        # Try generic call
        try:
            shap_values = explainer(scaled_values)
        except Exception as e2:
            st.error(f"Alternative shap call also failed: {e2}")
            st.stop()

    # Determine predicted class and extract local shap correctly
    predicted_class = int(prediction) if prediction is not None else 1

    # Robust extraction for different SHAP outputs:
    try:
        if isinstance(shap_values, list):
            # old API: list of arrays per class
            local_shap = np.array(shap_values[predicted_class][0])
        elif hasattr(shap_values, "values") and getattr(shap_values, "values") is not None:
            # new shap.Explanation object
            vals = np.array(shap_values.values)
            if vals.ndim == 3:
                # (samples, features, classes)
                local_shap = vals[0, :, predicted_class]
            else:
                # (samples, features)
                local_shap = vals[0, :]
        else:
            # plain ndarray
            arr = np.array(shap_values)
            if arr.ndim == 3:
                local_shap = arr[0, :, predicted_class]
            else:
                local_shap = arr[0]
    except Exception as e:
        st.error(f"Failed to extract local_shap: {e}")
        st.stop()

    # expected value per class (robust)
    try:
        expected_value = explainer.expected_value[predicted_class]
    except Exception:
        try:
            expected_value = np.array(explainer.expected_value).ravel()[predicted_class]
        except Exception:
            expected_value = None

    # Show numeric SHAP values so you can verify they change between runs
    st.write("#### Numeric SHAP values for this prediction (feature : shap_value):")
    shap_pairs = list(zip(feature_names, local_shap.tolist()))
    st.dataframe({"feature": [p[0] for p in shap_pairs], "shap_value": [p[1] for p in shap_pairs]})

    # Force plot (interactive) with fallback
    st.write("### Force plot (interactive if supported)")
    try:
        shap.initjs()
        # For force_plot we need base value. If not available, pass scalar 0 (visual only)
        base = expected_value if expected_value is not None else 0.0

        # Using shap.force_plot; save to temp HTML
        fp = shap.force_plot(base, local_shap, scaled_values[0], feature_names=feature_names, matplotlib=False)
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".html")
        tmp.close()
        shap.save_html(tmp.name, fp)
        html = open(tmp.name, 'r', encoding='utf-8').read()
        st.components.v1.html(html, height=400)
        os.remove(tmp.name)
    except Exception as e:
        st.warning(f"Interactive force plot failed: {e}. Showing static bar chart.")
        fig, ax = plt.subplots(figsize=(7, 5))
        y_pos = np.arange(len(feature_names))
        ax.barh(y_pos, local_shap)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(feature_names)
        ax.set_xlabel("SHAP value")
        fig.tight_layout()
        st.pyplot(fig)

    # Feature importance bar (SHAP)
    st.write("### SHAP Feature importance (bar)")
    try:
        exp = shap.Explanation(values=local_shap, base_values=expected_value, data=scaled_values[0], feature_names=feature_names)
        shap.plots.bar(exp, show=False)
        st.pyplot(plt.gcf())
    except Exception as e:
        st.warning(f"shap.plots.bar failed: {e}. Using matplotlib fallback.")
        fig2, ax2 = plt.subplots(figsize=(7, 5))
        ax2.barh(feature_names, local_shap)
        ax2.set_xlabel("SHAP value")
        fig2.tight_layout()
        st.pyplot(fig2)

    st.success("SHAP computed and displayed. Change inputs and run again — numeric SHAP table above should change per-input.")
