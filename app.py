import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# ─── Load Model and Preprocessors ───────────────────────────────────────────

@st.cache_resource
def load_model_and_encoders():
    try:
        model = tf.keras.models.load_model('model.keras')
        with open('label_encoder_gender.pkl', 'rb') as f:
            label_encoder_gender = pickle.load(f)
        with open('onehot_encoder_geo.pkl', 'rb') as f:
            onehot_encoder_geo = pickle.load(f)
        with open('scaler.pkl', 'rb') as f:
            scaler = pickle.load(f)
        return model, label_encoder_gender, onehot_encoder_geo, scaler
    except Exception as e:
        st.error(f"Error loading model/encoders: {e}")
        st.stop()

model, label_encoder_gender, onehot_encoder_geo, scaler = load_model_and_encoders()

# ─── Streamlit UI ────────────────────────────────────────────────────────────

st.title('Customer Churn Prediction')
st.markdown("Fill in the customer details below and click **Predict** to see the churn probability.")

col1, col2 = st.columns(2)

with col1:
    geography       = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender          = st.selectbox('Gender', label_encoder_gender.classes_)
    age             = st.slider('Age', 18, 92, 35)
    balance         = st.number_input('Balance', min_value=0.0, value=0.0)
    credit_score    = st.number_input('Credit Score', min_value=300, max_value=850, value=650)

with col2:
    estimated_salary  = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    tenure            = st.slider('Tenure', 0, 10, 5)
    num_of_products   = st.slider('Number of Products', 1, 4, 1)
    has_cr_card       = st.selectbox('Has Credit Card', [0, 1])
    is_active_member  = st.selectbox('Is Active Member', [0, 1])

# ─── Prediction (only on button click) ──────────────────────────────────────

if st.button('Predict Churn'):

    # Step 1: One-hot encode Geography
    geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=onehot_encoder_geo.get_feature_names_out(['Geography'])
    )

    # Step 2: Label encode Gender
    gender_encoded = label_encoder_gender.transform([gender])[0]

    # Step 3: Build input DataFrame in the EXACT same column order as training
    # Training order: CreditScore, Geography_*, Gender, Age, Tenure, Balance,
    #                 NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary
    input_data = pd.DataFrame({
        'CreditScore':     [credit_score],
        'Gender':          [gender_encoded],
        'Age':             [age],
        'Tenure':          [tenure],
        'Balance':         [balance],
        'NumOfProducts':   [num_of_products],
        'HasCrCard':       [has_cr_card],
        'IsActiveMember':  [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    # Step 4: Insert Geography OHE columns right after CreditScore (index 1)
    # This matches the column order the scaler was trained on
    input_data = pd.concat(
        [input_data.iloc[:, :1], geo_encoded_df, input_data.iloc[:, 1:]],
        axis=1
    )

    # Step 5: Scale
    input_data_scaled = scaler.transform(input_data)

    # Step 6: Predict
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    # ─── Display Results ─────────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Prediction Result")

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(label="Churn Probability", value=f"{prediction_proba:.2%}")
    with col_b:
        if prediction_proba > 0.5:
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **not likely to churn**.")

    # Confidence bar
    st.progress(prediction_proba, text=f"Churn Risk: {prediction_proba:.2%}")
