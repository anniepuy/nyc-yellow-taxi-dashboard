## Main streamlit app entry point & requests for API data

import streamlit as st
import pandas as pd
import requests
from io import StringIO


st.set_page_config(layout="wide", page_title="NYC Yellow Taxi Dashboard")

st.title("NYC Yellow Taxi - January 2023")

#load data from the NYC SOAP api
@st.cache_data
def fetch_data():
    API_URL = "https://data.cityofnewyork.us/resource/4b4i-vvec.csv"

    query = (
        "?$query=SELECT "
        "`vendorid`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, "
        "`passenger_count`, `trip_distance`, `ratecodeid`, `store_and_fwd_flag`, "
        "`pulocationid`, `dolocationid`, `payment_type`, `fare_amount`, `extra`, "
        "`mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, "
        "`total_amount`, `congestion_surcharge`, `airport_fee` "
        "WHERE `tpep_pickup_datetime` BETWEEN '2023-01-01T09:01:30'::floating_timestamp "
        "AND '2023-01-31T09:01:30'::floating_timestamp"
    )

    response = requests.get(API_URL + query)
    if response.status_code != 200:
        st.error("Error fetching data from the API")
        return pd.DataFrame()
    
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)
    return df

#load data
df = fetch_data()

st.subheader("Sample Taxi Trips")
st.dataframe(df.head(10))