## Main streamlit app entry point & requests for API data

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocess import load_data_from_api, transform_for_visuals
import joblib
import os

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="NYC Yellow Taxi Dashboard")
st.title("🚕 NYC Yellow Taxi - January 2023 (Live from NYC Open Data API)")

st.sidebar.header("Data Date Range")

from datetime import datetime

start = st.sidebar.date_input("Start Date", datetime(2023, 1, 1))
end = st.sidebar.date_input("End Date", datetime(2023, 1, 3))

# Check valid range
if start > end:
    st.error("Start date must be before end date.")
    st.stop()

# --- Load Data ---
df_raw = load_data_from_api(start.isoformat(), end.isoformat(), max_rows=10000)

if df_raw is None:
    st.error("Failed to load data from the API. Please try a different date range.")
    st.stop()

df_viz = transform_for_visuals(df_raw)

if df_viz.empty:
    st.warning("No data available for the selected date range. Please try different dates.")
    st.stop()

# Load model and features if they exist
try:
    model = joblib.load("models/fare_model.pkl")
    model_features = joblib.load("models/fare_model_features.pkl")
    model_loaded = True
except FileNotFoundError:
    st.warning("Fare prediction model not found. Please run the model training script first.")
    model_loaded = False

# --- Sidebar Filters ---
st.sidebar.header("🔧 Filters")

# Get unique passenger counts and remove any null/zero values
valid_passenger_counts = sorted(df_viz['passenger_count'].dropna().unique())
valid_passenger_counts = [x for x in valid_passenger_counts if x > 0]

passenger_filter = st.sidebar.selectbox(
    "Passenger Count",
    options=valid_passenger_counts
)

payment_filter = st.sidebar.selectbox(
    "Payment Type",
    options=df_viz['payment_type'].dropna().unique()
)

# --- Apply Filters ---
filtered_df = df_viz[
    (df_viz['tpep_pickup_datetime'].dt.date >= start) &
    (df_viz['tpep_pickup_datetime'].dt.date <= end) &
    (df_viz['passenger_count'] == passenger_filter) &
    (df_viz['payment_type'] == payment_filter)
].copy()

# --- KPIs ---
total_trips = len(filtered_df)
avg_fare = filtered_df['fare_amount'].mean()
avg_distance = filtered_df['trip_distance'].mean()

col1, col2, col3 = st.columns(3)
col1.metric("Total Trips", f"{total_trips:,}")
col2.metric("Avg Fare", f"${avg_fare:.2f}")
col3.metric("Avg Distance", f"{avg_distance:.2f} mi")

# --- Sample Table ---
st.subheader("Sample Taxi Trips")
st.dataframe(filtered_df.head(10))

# --- Avg Fare by Location
st.subheader("Avg Fare by Location")
avg_fare_by_borough = (
    filtered_df.groupby('pulocation_borough')['fare_amount'].mean()
    .reset_index()
    .sort_values(by='fare_amount', ascending=False)
)

fig_fare = px.bar(
    avg_fare_by_borough,
    x='pulocation_borough', 
    y='fare_amount',
    title='Average Fare by Borough',
    labels={'pulocation_borough': 'Borough', 'fare_amount': 'Avg Fare ($)'}
)

st.plotly_chart(fig_fare, use_container_width=True)

# --- Avg Passenger Count by Pickup Borough ---
st.subheader("🧍 Average Passenger Count by Pickup Borough")

if 'pulocation_borough' in filtered_df.columns and not filtered_df.empty:
    filtered_df['passenger_count'] = pd.to_numeric(filtered_df['passenger_count'], errors='coerce')
    filtered_df = filtered_df[filtered_df['passenger_count'] > 0]

    avg_passengers = (
        filtered_df.groupby('pulocation_borough')['passenger_count']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values(by='mean', ascending=False)
    )

    if not avg_passengers.empty:
        fig_passengers = px.bar(
            avg_passengers,
            x='pulocation_borough',
            y='mean',
            title="Average Passenger Count by Pickup Borough",
            labels={'pulocation_borough': 'Borough', 'mean': 'Avg Passengers'},
            hover_data=['count']
        )
        st.plotly_chart(fig_passengers, use_container_width=True)
    else:
        st.info("No passenger data available after filtering.")
else:
    st.warning("Missing borough or passenger count data.")

# --- Trip Distance Distribution ---
st.subheader("Trip Distance Distribution")
fig2 = px.histogram(
    filtered_df,
    x='trip_distance',
    nbins=50,
    title="Trip Distance Distribution",
    labels={'trip_distance': 'Miles'}
)
st.plotly_chart(fig2, use_container_width=True)

# --- Top Pickup Boroughs ---
st.subheader("🏙️ Top 10 Pickup Boroughs")
pickup_borough_counts = (
    filtered_df['pulocation_borough']
    .value_counts()
    .nlargest(10)
    .reset_index()
)
pickup_borough_counts.columns = ['Borough', 'Trips']

fig3 = px.bar(
    pickup_borough_counts,
    x='Borough',
    y='Trips',
    title="Top 10 Pickup Boroughs"
)
st.plotly_chart(fig3, use_container_width=True)

# Only show fare prediction if model is loaded
if model_loaded:
    st.header("Predict Taxi Fare")

    with st.expander("Click to view Data Dictionary"):
        st.markdown("""
                #### Categorical Values
                   ** Payment Type **
                   - 1: Credit Card
                   - 2: Cash
                   - 3: No Charge
                   - 4: Dispute
                   - 5: Unknown
                   - 6: Voided Trip
                    
                ** Rate Code **
                - 1: Standard Rate
                - 2: JFK
                - 3: Newark
                - 4: Nassau or Westchester
                - 5: Negotiated Fare
                - 6: Group Ride
                    
                ** Location ID
                - 132: JFK Airport
                - 138: LaGuardia Airport
                - Use the download link below to the view the full location dictionary
                
                ** Day of Week **
                - 0: Monday
                - 1: Tuesday
                - 2: Wednesday
                - 3: Thursday
                - 4: Friday
                - 5: Saturday
                - 6: Sunday
                """)
        try: 
            with open("data/locationid.csv", "rb") as f:
                st.download_button("Download Full Location Dictionary (CSV)", f, file_name="locationid.csv")
        except FileNotFoundError:
            st.error("File not found. Please check the file path.")

    with st.form("fare_form"):
        col1, col2, col3 = st.columns(3)

        trip_distance = col1.number_input("Trip Distance (miles)", min_value=0.1, max_value=100.0, value=1.5)
        passenger_count = col2.selectbox("Passenger Count", options=[1, 2, 3, 4, 5, 6])
        ratecodeid = col3.selectbox("Rate Code", options=[1, 2, 3, 4, 5, 6])
        
        col4, col5, col6 = st.columns(3)
        payment_type = col4.selectbox("Payment Type", options=[1, 2, 3, 4, 5, 6])
        pulocationid = col5.number_input("Pickup Location ID", min_value=1, max_value=265, value=132)
        dolocationid = col6.number_input("Dropoff Location ID", min_value=1, max_value=265, value=237)

        col7, col8 = st.columns(2)
        pickup_hour = col7.slider("Pickup Hour (0–23)", 0, 23, 14)
        pickup_dayofweek = col8.slider("Day of Week (0=Mon, 6=Sun)", 0, 6, 3)

        submitted = st.form_submit_button("Predict Fare")

        if submitted:
            input_data = {
                'trip_distance': trip_distance,
                'passenger_count': passenger_count,
                'ratecodeid': ratecodeid,
                'payment_type': payment_type,
                'pulocationid': pulocationid,
                'dolocationid': dolocationid,
                'pickup_hour': pickup_hour,
                'pickup_dayofweek': pickup_dayofweek
            }

            # Convert to DataFrame with same column order
            input_df = pd.DataFrame([input_data])
            input_df = pd.get_dummies(input_df)

            # Align to model's expected columns
            for col in model_features:
                if col not in input_df.columns:
                    input_df[col] = 0
            input_df = input_df[model_features]

            predicted_fare = model.predict(input_df)[0]
            st.success(f"💵 Estimated Fare: ${predicted_fare:.2f}")