# Preprocess data for the model with helper functions
import pandas as pd
import os
import requests
from io import StringIO


# Load raw data and parse datetimes
def load_data_from_api(start_date: str, end_date: str, max_rows: int = 10000, batch_size: int = 1000) -> pd.DataFrame:
    API_BASE = "https://data.cityofnewyork.us/resource/4b4i-vvec.csv"
    all_data = []

    try:
        for offset in range(0, max_rows, batch_size):
            query = (
                "?$query=SELECT "
                "`vendorid`, `tpep_pickup_datetime`, `tpep_dropoff_datetime`, "
                "`passenger_count`, `trip_distance`, `ratecodeid`, `store_and_fwd_flag`, "
                "`pulocationid`, `dolocationid`, `payment_type`, `fare_amount`, `extra`, "
                "`mta_tax`, `tip_amount`, `tolls_amount`, `improvement_surcharge`, "
                "`total_amount`, `congestion_surcharge`, `airport_fee` "
                f"WHERE `tpep_pickup_datetime` BETWEEN '{start_date}T00:00:00'::floating_timestamp "
                f"AND '{end_date}T23:59:59'::floating_timestamp "
                f"LIMIT {batch_size} OFFSET {offset}"
            )

            response = requests.get(API_BASE + query)
            if response.status_code != 200:
                print(f"Failed to fetch data: {response.status_code}")
                print(f"Response: {response.text}")
                return None

            if not response.text.strip():
                break

            chunk = pd.read_csv(StringIO(response.text))
            if chunk.empty:
                break

            all_data.append(chunk)

        if not all_data:
            print("No data retrieved from API.")
            return None

        df = pd.concat(all_data, ignore_index=True)
        df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
        df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

        return df

    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return None

# Clean copy for visualizations
def transform_for_visuals(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        raise ValueError("No data available to transform")
        
    df_viz = df.copy()

    # Vendor ID Mapping
    vendor_map = {
        1: "Creative Mobile Technologies (CMT)",
        2: "Curb Mobility",
        6: "Myle Technologies",
        7: "Helix", 
    }
    df_viz['vendor_name'] = df_viz['vendorid'].map(vendor_map).fillna("Other")

    #Rate code mapping
    rate_map = {
        1: "Standard Rate",
        2: "JFK Airport",
        3: "Newark Airport",
        4: "Nassau or Westchester County",
        5: "Negotiated Fare",
        6: "Group Ride",
        99: "Private Hire",
    }
    df_viz['rate_name'] = df_viz['ratecodeid'].map(rate_map).fillna("Other")

    #Store and forward flag mapping
    sfw_map = {
        "Y": "Store and forward",
        "N": "Not a store and forward trip",
    }
    df_viz['store_and_fwd_flag'] = df_viz['store_and_fwd_flag'].map(sfw_map).fillna("Unknown")

    #Payment type mapping
    payment_map = {
        1: "Credit Card",
        2: "Cash",
        3: "No Charge",
        4: "Dispute",
        5: "Unknown",
        6: "Voided Trip",
    }
    df_viz['payment_type'] = df_viz['payment_type'].map(payment_map).fillna("Unknown")

    # Load borough lookup csv
    borough_path = os.path.join("data", "locationid.csv")
    borough_df = pd.read_csv(borough_path)

    # Map pickup and dropoff boroughs
    borough_map = borough_df.set_index("LocationID")["Borough"].to_dict()
    df_viz['pulocation_borough'] = df_viz['pulocationid'].map(borough_map).fillna("Unknown")
    df_viz['dolocation_borough'] = df_viz['dolocationid'].map(borough_map).fillna("Unknown")

    return df_viz