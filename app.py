## Main streamlit app entry point & requests for API data

import streamlit as st
import pandas as pd
import plotly.express as px
from utils.preprocess import load_data_from_api, transform_for_visuals

# --- Page Setup ---
st.set_page_config(layout="wide", page_title="NYC Yellow Taxi Dashboard")
st.title("🚕 NYC Yellow Taxi - January 2023 (Live from NYC Open Data API)")

# --- Load Data ---
df_raw = load_data_from_api()
df_viz = transform_for_visuals(df_raw)

# --- Sidebar Filters ---
st.sidebar.header("🔧 Filters")

min_date = df_viz['tpep_pickup_datetime'].min()
max_date = df_viz['tpep_pickup_datetime'].max()

date_range = st.sidebar.date_input(
    "Select date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)

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
    (df_viz['tpep_pickup_datetime'].dt.date >= date_range[0]) &
    (df_viz['tpep_pickup_datetime'].dt.date <= date_range[1]) &
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

# Debug information
st.write("Debug Info:")
st.write(f"Columns in DataFrame: {filtered_df.columns.tolist()}")
st.write(f"Sample passenger counts: {filtered_df['passenger_count'].head()}")
st.write(f"Unique passenger counts: {filtered_df['passenger_count'].unique()}")

if 'pulocation_borough' in filtered_df.columns and not filtered_df.empty:
    # Ensure passenger count is numeric and remove any zero/null values
    filtered_df['passenger_count'] = pd.to_numeric(filtered_df['passenger_count'], errors='coerce')
    filtered_df = filtered_df[filtered_df['passenger_count'] > 0]

    avg_passengers = (
        filtered_df.groupby('pulocation_borough')['passenger_count']
        .agg(['mean', 'count'])
        .reset_index()
        .sort_values(by='mean', ascending=False)
    )

    # Debug the grouped data
    st.write("Grouped data:")
    st.write(avg_passengers)

    if not avg_passengers.empty:
        fig_passengers = px.bar(
            avg_passengers,
            x='pulocation_borough',
            y='mean',
            title="Average Passenger Count by Pickup Borough",
            labels={'pulocation_borough': 'Borough', 'mean': 'Avg Passengers'},
            hover_data=['count']  # Show number of trips in hover
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