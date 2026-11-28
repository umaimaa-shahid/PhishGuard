# Import libraries for the Streamlit interface, model, and SHAP explanations.
import os
import joblib
import shap
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from src.url_features import extract_url_features


# Define paths to the saved model and cleaned dataset.
MODEL_PATH = "models/final_model.joblib"
DATA_PATH = "data/cleaned_with_domains.csv"


# Configure the Streamlit page.
st.set_page_config(
    page_title="PhishGuard",
    page_icon="🛡️",
    layout="centered"
)


# Load the trained model, dataset, and SHAP explainer once.
@st.cache_resource
def load_resources():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Saved model was not found.")

    saved = joblib.load(MODEL_PATH)

    model = saved["model"]
    expected_features = saved["features"]

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError("Cleaned dataset was not found.")

    dataset = pd.read_csv(DATA_PATH)

    prep, clf = model.named_steps["prep"], model.named_steps["clf"]
    names = [n.split("__", 1)[1] for n in prep.get_feature_names_out()]
    explainer = shap.TreeExplainer(clf)
    
    return model, expected_features, dataset, explainer, prep, names


# Display the application introduction.
st.title("🛡️ PhishGuard")
st.subheader("Explainable Phishing URL Detection")

st.write(
    "Analyze a website URL using a trained machine learning model "
    "and understand which URL features influenced its prediction."
)

st.info(
    "PhishGuard analyzes URL text only. It does not visit websites "
    "or execute website content."
)

# Accept the URL entered by the user.
url = st.text_input(
    "Website URL",
    placeholder="https://example.com/login"
)


# Run URL analysis when the user clicks the button.
if st.button("Analyze URL", type="primary", use_container_width=True):

    if not url.strip():
        st.warning("Please enter a URL.")

    else:
        try:
            with st.spinner("Analyzing URL..."):

                # Load the model and extract the URL features.
                model, expected_features, dataset, explainer, prep, names = load_resources()
                extracted, extraction_source = extract_url_features(
                    url,
                    dataset
                )

                # Arrange features in the exact order used during training.
                model_input = extracted.reindex(
                    columns=expected_features
                )

                if model_input.isnull().any().any():
                    missing = model_input.columns[
                        model_input.isnull().any()
                    ].tolist()

                    raise ValueError(
                        f"Missing model features: {missing}"
                    )

                # Predict the URL class and calculate class probabilities.
                prediction = int(model.predict(model_input)[0])

                probabilities = model.predict_proba(model_input)[0]
                
                classes = list(model.classes_)
                phishing_index = classes.index(1)
                legitimate_index = classes.index(0)

                phishing_prob = float(probabilities[phishing_index])
                legitimate_prob = float(probabilities[legitimate_index])

                # Generate a SHAP explanation for the phishing class.
                xi = pd.DataFrame(prep.transform(model_input), columns=names)
                phishing_shap = explainer(xi)[0]   # positive = pushes toward phishing

            # Display the prediction and probability scores.
            st.divider()
            st.subheader("Result")
            if prediction == 1:
                st.error("⚠️ Prediction: PHISHING")
            else:
                st.success("✅ Prediction: LEGITIMATE")

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Phishing probability",
                    f"{phishing_prob * 100:.2f}%"
                )

            with col2:
                st.metric(
                    "Legitimate probability",
                    f"{legitimate_prob * 100:.2f}%"
                )

            # Disclose whether the URL used original or calculated features.
            st.caption(f"Feature source: {extraction_source}")

            if extraction_source != "Original dataset features":
                st.warning(
                    "This URL was not found in the original dataset. "
                    "Some extracted features are approximations. "
                    "Treat this prediction as experimental."
                )

            # Display the SHAP feature-contribution explanation.
            st.divider()
            st.subheader("SHAP Explanation")

            st.write(
                "The waterfall plot shows how individual URL features "
                "contribute to the model's phishing-class output."
            )

            plt.figure(figsize=(10, 6))
            shap.plots.waterfall(phishing_shap, show=False)
            st.pyplot(plt.gcf())
            plt.close()

            # Display the values supplied to the trained model.
            st.divider()
            st.subheader("Extracted URL Features")

            st.dataframe(
                model_input.T.rename(columns={0: "Value"}),
                use_container_width=True
            )

        except Exception as error:
            st.error(f"Could not analyze this URL: {error}")


# Display the application footer.
st.divider()

st.caption(
    "PhishGuard | Explainable Phishing-URL Detection | "
    "Academic project prototype"
)
