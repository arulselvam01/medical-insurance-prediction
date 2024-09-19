import streamlit as st
import pickle
import numpy as np

# Set page title
st.title("Medical Insurance Charges Prediction App")

# Sidebar for file uploads
st.sidebar.header("Upload Model")

# Upload the pickle file for the linear regression model
uploaded_pickle_file = st.sidebar.file_uploader("Upload Linear Model Pickle File", type=["pkl"])

# Function to load the linear model
def load_model(pickle_file):
    if pickle_file is not None:
        loaded_model = pickle.load(pickle_file)
        return loaded_model
    return None

# Load the linear model
model = load_model(uploaded_pickle_file)

# Check if model is loaded
if model is not None:
    # User input for all features
    st.header("Enter Details to Predict Insurance Charges")

    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    sex = st.selectbox("Sex", options=["male", "female"])
    bmi = st.number_input("BMI", min_value=0.0, max_value=60.0, value=25.0)
    children = st.number_input("Children", min_value=0, max_value=10, value=0)
    smoker = st.selectbox("Smoker", options=["yes", "no"])
    region = st.selectbox("Region", options=["northwest", "northeast", "southeast", "southwest"])

    # Convert categorical variables to numeric
    sex_numeric = 0 if sex == "male" else 1
    smoker_numeric = 1 if smoker == "yes" else 0

    # One-hot encoding for region
    region_northwest = 1 if region == "northwest" else 0
    region_northeast = 1 if region == "northeast" else 0
    region_southeast = 1 if region == "southeast" else 0
    region_southwest = 1 if region == "southwest" else 0

    # Predict button
    if st.button("Make Prediction"):
        # Prepare input for the model with all 8 features
        user_input = np.array([[age, sex_numeric, bmi, children, smoker_numeric,
                                region_northwest, region_northeast, region_southeast]])

        # Make prediction
        prediction = model.predict(user_input)

        # Convert prediction from USD to INR (assuming 1 USD = 83 INR)
        conversion_rate = 83
        prediction_in_inr = prediction[0] * conversion_rate

        # Display prediction in INR
        st.subheader(f"Predicted Insurance Charges: ₹{prediction_in_inr:,.2f} INR")

else:
    st.warning("Please upload the linear model to proceed.")