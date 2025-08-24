import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, hour, count
import urllib.request
import os

# Initialize Spark session
spark = SparkSession.builder.appName("NYCTaxiAnalysis").getOrCreate()

# Streamlit app title
st.title("NYC Taxi Trip Analytics Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Pickup Heatmap", "Trip Trends"])

# Function to download dataset
@st.cache_data
def load_data():
    url = "https://data.cityofnewyork.us/api/views/23ev-vb22/rows.csv?accessType=DOWNLOAD"
    file_path = "nyc_taxi_2023.csv"
    if not os.path.exists(file_path):
        urllib.request.urlretrieve(url, file_path)
    return file_path

# Preprocess data with Spark
@st.cache_resource
def preprocess_data():
    file_path = load_data()
    # Load CSV into Spark DataFrame
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    # Clean data: Filter out invalid coordinates and fares
    df = df.filter((col("pickup_latitude").isNotNull()) & 
                   (col("pickup_longitude").isNotNull()) & 
                   (col("fare_amount") > 0))
    
    # Aggregate by pickup location and hour
    agg_df = df.groupBy(
        col("pickup_longitude"),
        col("pickup_latitude"),
        hour(col("tpep_pickup_datetime")).alias("hour")
    ).agg(count("*").alias("trip_count"))
    
    # Convert to Pandas for Streamlit
    result_df = agg_df.toPandas()
    return result_df

# Load preprocessed data
result_df = preprocess_data()

# Page: Pickup Heatmap
if page == "Pickup Heatmap":
    st.header("Taxi Pickup Heatmap")
    st.write("Visualize taxi pickup locations in NYC.")
    
    fig = px.density_mapbox(
        result_df,
        lat="pickup_latitude",
        lon="pickup_longitude",
        z="trip_count",
        radius=10,
        center={"lat": 40.7128, "lon": -74.0060},
        zoom=10,
        mapbox_style="open-street-map",
        title="NYC Taxi Pickup Heatmap"
    )
    st.plotly_chart(fig)

# Page: Trip Trends
elif page == "Trip Trends":
    st.header("Trip Count Trends by Hour")
    st.write("Select an hour to view trip count trends.")
    
    hours = sorted(result_df["hour"].unique())
    selected_hour = st.selectbox("Select Hour", hours)
    
    filtered_df = result_df[result_df["hour"] == selected_hour]
    fig = px.scatter(
        filtered_df,
        x="pickup_longitude",
        y="pickup_latitude",
        size="trip_count",
        title=f"Taxi Pickups at Hour {selected_hour}"
    )
    st.plotly_chart(fig)

# Stop Spark session
spark.stop()
