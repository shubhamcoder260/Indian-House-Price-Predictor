import streamlit as st
import pandas as pd
import pickle
import joblib
import numpy as np

# Load model, scaler, PCA, and feature names once at start
model_path = "/home/shubh-k-pc/Documents/AIML/house-price-prediction-model-main/models/house_price_model.pkl"
scaler_path = "/home/shubh-k-pc/Documents/AIML/house-price-prediction-model-main/models/scaler.pkl"
pca_path = "/home/shubh-k-pc/Documents/AIML/house-price-prediction-model-main/models/pca.pkl"
input_columns_path = "/home/shubh-k-pc/Documents/AIML/house-price-prediction-model-main/models/input_columns.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)

with open(input_columns_path, "rb") as f:
    input_columns = pickle.load(f)

st.set_page_config(page_title="House Price Predictor", layout="centered")
st.title("🏡 Indian House Price Predictor")
st.markdown("Enter property details below to estimate the price.")

# User inputs
bedrooms = st.number_input("number of bedrooms", min_value=1, max_value=10, value=5)

bathrooms = st.number_input("number of bathrooms", min_value=1, max_value=10, value=2)

living_area = st.number_input("living area", min_value=100, max_value=10000, value=1500)

lot_area = st.number_input("lot area", min_value=100, max_value=20000, value=2000)

number_of_floors = st.number_input("number of floors", min_value=1, max_value=10, value=6)

waterfront_present = st.selectbox("waterfront present", [0, 1])

condition_of_the_house = st.number_input("condition of the house", min_value=1, max_value=10, value=6)

grade_of_the_house = st.number_input("grade of the house", min_value=1, max_value=13, value=9)

Area_of_the_house_excluding_basement = st.number_input("Area of the house(excluding basement)", min_value=100, max_value=10000, value=1000)

Built_Year = st.number_input("Built Year", min_value=1800, max_value=2025, value=2021)

Renovation_Year = st.number_input("Renovation Year", min_value=0, max_value=2025, value=2013)

Number_of_schools_nearby = st.number_input("Number of schools nearby", min_value=0, max_value=20, value=2)

number_of_views = st.number_input("Number of views", min_value=0, max_value=100000, value=0)

area_of_basement = st.number_input("Area of the basement", min_value=0, max_value=5000, value=0)

latitude = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=28.61)  # example India lat

longitude = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=77.23)  # example India long

distance_from_airport = st.number_input("Distance from the airport (km)", min_value=0, max_value=500, value=20)



if st.button("Predict Price"):
    input_df = pd.DataFrame([{
        'number of bedrooms': bedrooms,
        'number of bathrooms': bathrooms,
        'living area': living_area,
        'lot area': lot_area,
        'number of floors': number_of_floors,
        'waterfront present': waterfront_present,
        'condition of the house': condition_of_the_house,
        'grade of the house': grade_of_the_house,
        'Area of the house(excluding basement)': Area_of_the_house_excluding_basement,
        'Built Year': Built_Year,
        'Renovation Year': Renovation_Year,
        'Number of schools nearby': Number_of_schools_nearby,
        'number of views': number_of_views,
        'Area of the basement': area_of_basement,
        'Lattitude': latitude,
        'Longitude': longitude,
        'Distance from the airport': distance_from_airport,    }])

    # Ensure columns are in the correct order as model expects
    input_df = input_df[input_columns]

    # Scale and apply PCA transform
    scaled_input = scaler.transform(input_df)
    pca_input = pca.transform(scaled_input)

    # Predict log(price) and then convert to original scale
    log_price = model.predict(pca_input)[0]
    predicted_price = np.exp(log_price)

    st.write("Input Features:")
    st.write(input_df)
    st.success(f"🏷️ Estimated House Price: ₹{predicted_price:,.2f}")
