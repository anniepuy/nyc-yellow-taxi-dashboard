# Preprocess data for the model with helper functions
import pandas as pd
import os


# Load raw data and parse datetimes
def load_data_from_api():
    import requests
    from io import StringIO

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
    csv_data = StringIO(response.text)
    df = pd.read_csv(csv_data)

    # Parse datetime columns once here
    df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
    df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

    return df

# Clean copy for visualizations
def transform_for_visuals(df: pd.DataFrame) -> pd.DataFrame:
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