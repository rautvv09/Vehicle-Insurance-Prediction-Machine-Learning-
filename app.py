import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ============================
# 1. PAGE CONFIGURATION
# ============================
st.set_page_config(
    page_title="Vehicle Insurance Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================
# 2. LOAD MODEL BUNDLE
# ============================
@st.cache_resource
def load_model():
    data = joblib.load("insurance_bundle.pkl")
    return data["model"], data["scaler"], data["feature_columns"]

@st.cache_data
def load_dataset():
    return pd.read_csv("dataset.csv")

# ============================
# 3. PREPROCESS INPUT (MATCH TRAINING)
# ============================
def preprocess_input(user_input, feature_columns, scaler):
    input_df = pd.DataFrame([user_input])

    # One-Hot Encoding (same as training)
    input_df = pd.get_dummies(input_df)

    # Align with training features
    input_df = input_df.reindex(columns=feature_columns, fill_value=0)

    # Scaling
    input_df[:] = scaler.transform(input_df)

    return input_df

# ============================
# 4. LOAD RESOURCES
# ============================
model, scaler, feature_columns = load_model()
dataset = load_dataset()

# Identify columns for UI
numerical_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
numerical_cols.remove("is_claim")

categorical_cols = dataset.select_dtypes(include=["object"]).columns.tolist()

# ============================
# 5. SIDEBAR
# ============================
st.sidebar.title("📊 Project Information")

st.sidebar.markdown("""
### About This Application
This web application predicts whether an insurance customer is likely to make a claim based on various vehicle and policy features.

### Model Information
- **Algorithm**: Decision Tree Classifier
- **Training**: Pre-trained model loaded from `insurance_bundle.pkl`
- **Features**: Vehicle specifications and derived engineering features

### How to Use
1. Enter customer and vehicle details in the form
2. Click the **Predict** button
3. View the prediction result

### Feature Engineering
- Torque to RPM Ratio  
- Power to RPM Ratio
""")

st.sidebar.markdown("---")
st.sidebar.info("💡 **Tip**: All fields are required for accurate predictions")

# ============================
# 6. MAIN HEADER
# ============================
st.title("🚗 Vehicle Insurance Prediction")
st.markdown("""
This application uses a **Decision Tree model** to predict the likelihood of insurance claims based on customer and vehicle data.
Enter the details below to get a prediction.
""")
st.markdown("---")

# ============================
# 7. USER INPUT FORM
# ============================
st.subheader("📝 Enter Customer & Vehicle Details")

with st.form("prediction_form"):
    user_input = {}

    col1, col2, col3 = st.columns(3)
    split = len(numerical_cols) // 3 + 1

    with col1:
        for col in numerical_cols[:split]:
            user_input[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(dataset[col].mean())
            )

    with col2:
        for col in numerical_cols[split:2*split]:
            user_input[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(dataset[col].mean())
            )

    with col3:
        for col in numerical_cols[2*split:]:
            user_input[col] = st.number_input(
                col.replace("_", " ").title(),
                value=float(dataset[col].mean())
            )

    st.markdown("---")
    st.markdown("### Categorical Features")

    cat_cols = st.columns(4)
    for idx, col in enumerate(categorical_cols):
        with cat_cols[idx % 4]:
            user_input[col] = st.selectbox(
                col.replace("_", " ").title(),
                dataset[col].unique()
            )

    submit = st.form_submit_button("🔮 Predict Insurance Claim",use_container_width=True)

# ============================
# 8. PREDICTION
# ============================
if submit:
    with st.spinner("Predicting..."):
        processed_input = preprocess_input(
            user_input,
            feature_columns,
            scaler
        )

        prediction = model.predict(processed_input)[0]

    st.markdown("---")
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.warning("⚠️ Customer is **likely to claim insurance (1)**")
    else:
        st.success("✅ Customer is **not likely to claim insurance(0)**")

    with st.expander("📋 View Input Summary"):
        st.dataframe(pd.DataFrame([user_input]).T)

# ============================
# 9. FOOTER
# ============================
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:gray;">
Vehicle Insurance Prediction | Decision Tree Model <br>
Built with Streamlit ⚡
</div>
""", unsafe_allow_html=True)