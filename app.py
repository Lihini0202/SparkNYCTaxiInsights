import streamlit as st
import pandas as pd
import plotly.express as px
from pyspark.sql import SparkSession
from pyspark.ml.classification import RandomForestClassificationModel
from pyspark.ml.feature import VectorAssembler
import os

# Initialize Spark session
spark = SparkSession.builder.appName("NYCTaxiAnalysis").getOrCreate()

# Streamlit app title
st.title("NYC Taxi Trip Analytics Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Trip Trends", "Demand Predictions"])

# Load preprocessed data and model
@st.cache_resource
def load_data_and_model():
    data_path = "agg_data"
    if not os.path.exists(data_path):
        st.error("Aggregated data not found. Please run train_model.py first.")
        return None, None, None
    
    try:
        agg_df = spark.read.parquet(data_path)
        result_df = agg_df.toPandas()
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None
    
    model_path = "rf_model"
    assembler_path = "assembler"
    if not os.path.exists(model_path) or not os.path.exists(assembler_path):
        st.error("Model or assembler not found. Please run train_model.py first.")
        return result_df, None, None
    
    try:
        model = RandomForestClassificationModel.load(model_path)
        assembler = VectorAssembler.load(assembler_path)
    except Exception as e:
        st.error(f"Error loading model or assembler: {e}")
        return result_df, None, None
    
    return result_df, model, assembler

# Load data and model
result_df, model, assembler = load_data_and_model()

# Check if data/model loaded successfully
if result_df is None or model is None or assembler is None:
    st.stop()

# Page: Trip Trends
if page == "Trip Trends":
    st.header("Trip Count Trends by Hour")
    st.write("Select an hour to view trip count trends by taxi zone.")
    
    hours = sorted(result_df["hour"].unique())
    selected_hour = st.selectbox("Select Hour", hours)
    
    filtered_df = result_df[result_df["hour"] == selected_hour]
    fig = px.bar(
        filtered_df,
        x="PULocationID",
        y="trip_count",
        title=f"Taxi Pickups at Hour {selected_hour}"
    )
    st.plotly_chart(fig)

# Page: Demand Predictions
elif page == "Demand Predictions":
    st.header("High-Demand Zone Predictions")
    st.write("Predict whether a taxi zone is high-demand (trip count > 50) based on zone ID and hour.")
    
    # User input for prediction
    st.subheader("Input Zone Details")
    pulocation_id = st.number_input("Pickup Zone ID", value=236, min_value=1, max_value=263)  # NYC taxi zones: 1-263
    hour_input = st.selectbox("Hour of Day", list(range(24)))
    
    # Prepare input for prediction
    try:
        input_data = spark.createDataFrame([(float(pulocation_id), hour_input)], ["PULocationID", "hour"])
        input_transformed = assembler.transform(input_data)
        
        # Make prediction
        prediction = model.transform(input_transformed).select("prediction").collect()[0][0]
        prediction_label = "High-Demand" if prediction == 1.0 else "Low-Demand"
        
        st.write(f"Prediction: **{prediction_label}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
    
    # Show feature importance
    st.subheader("Feature Importance")
    importance = model.featureImportances.toArray()
    features = ["PULocationID", "hour"]
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    
    fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance for Demand Prediction")
    st.plotly_chart(fig)

# Stop Spark session
spark.stop()
