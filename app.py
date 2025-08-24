import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
import os
import urllib.request

# Streamlit app title
st.title("NYC Taxi Trip Analytics Dashboard")

# Sidebar for navigation
st.sidebar.header("Navigation")
page = st.sidebar.selectbox("Choose a page", ["Trip Trends", "Demand Predictions"])

# Load or generate data
@st.cache_data
def load_data_and_model():
    data_path = "agg_data.parquet"
    if not os.path.exists(data_path):
        try:
            url = "https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2023-01.parquet"
            urllib.request.urlretrieve(url, "temp.parquet")
            df = pd.read_parquet("temp.parquet")
            df = df.sample(frac=0.01, random_state=42)  # Smaller sample to reduce memory
            df = df[df['PULocationID'].notnull() & (df['fare_amount'] > 0)]
            df['hour'] = pd.to_datetime(df['tpep_pickup_datetime']).dt.hour
            agg_df = df.groupby(['PULocationID', 'hour']).size().reset_index(name='trip_count')
            agg_df.to_parquet(data_path)
            os.remove("temp.parquet")
        except Exception as e:
            st.error(f"Error downloading or processing data: {e}")
            return None, None
    
    try:
        result_df = pd.read_parquet(data_path)
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None
    
    model_path = "model.pkl"
    if not os.path.exists(model_path):
        st.error("Model not found. Ensure model.pkl is present.")
        return result_df, None
    
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return result_df, None
    
    return result_df, model

# Load data and model
result_df, model = load_data_and_model()

# Check if data/model loaded successfully
if result_df is None or model is None:
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
        input_data = pd.DataFrame([[pulocation_id, hour_input]], columns=["PULocationID", "hour"])
        prediction = model.predict(input_data)[0]
        prediction_label = "High-Demand" if prediction == 1 else "Low-Demand"
        
        st.write(f"Prediction: **{prediction_label}**")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
    
    # Show feature importance
    st.subheader("Feature Importance")
    importance = model.feature_importances
    features = ["PULocationID", "hour"]
    importance_df = pd.DataFrame({"Feature": features, "Importance": importance})
    
    fig = px.bar(importance_df, x="Feature", y="Importance", title="Feature Importance for Demand Prediction")
    st.plotly_chart(fig)
