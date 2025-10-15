
import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Dutsinma Crime Prediction System", layout="wide")

APP_TITLE = "Dutsinma Crime Prediction System"
SUPERVISOR = "Supervised by Mr. Abdulmuminu Yusuf (FUDMA)"
DEVELOPER = "Developed by Ahmad Nasir (B.Sc. Computer Science & IT, FUDMA)"

USERNAME = "Binnaseer1"
PASSWORD = "An@25787238"

def check_credentials(user, pwd):
    return (user == USERNAME and pwd == PASSWORD)

@st.cache_data
def load_data():
    return pd.read_csv("dutsinma_crime_mock.csv", parse_dates=['Date'])

def load_model_safe():
    try:
        data = joblib.load("model.joblib")
        return data.get('model', None), data.get('features', None)
    except Exception as e:
        return None, None

st.title(APP_TITLE)
st.write(SUPERVISOR)
st.write(DEVELOPER)
st.markdown("---")

with st.sidebar:
    st.header("Login")
    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")
    if st.button("Sign in"):
        if check_credentials(user, pwd):
            st.success("Login successful")
            st.session_state['auth'] = True
        else:
            st.error("Invalid credentials")

if not st.session_state.get('auth', False):
    st.info("Please login from the sidebar to access the system demo.")
    st.stop()

df = load_data()
model, features = load_model_safe()

left, right = st.columns([1,2])

with left:
    st.subheader("Controls")
    st.write("Quick dataset summary:")
    st.write(f"Records: {len(df)}")
    crime_select = st.multiselect("Filter by crime type", options=sorted(df['Crime_Type'].unique()), default=sorted(df['Crime_Type'].unique()))
    loc_select = st.multiselect("Filter by location", options=sorted(df['Location_Desc'].unique()), default=sorted(df['Location_Desc'].unique()))
    date_range = st.date_input("Date range", value=(df['Date'].min(), df['Date'].max()))
    if st.button("Apply Filters"):
        df = df[(df['Crime_Type'].isin(crime_select)) & (df['Location_Desc'].isin(loc_select)) & (df['Date'] >= pd.to_datetime(date_range[0])) & (df['Date'] <= pd.to_datetime(date_range[1]))]
        st.success(f"Filtered to {len(df)} records")

    st.markdown("---")
    st.subheader("Prediction Panel")
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=20)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=6)
    day = st.number_input("Day (1-31)", min_value=1, max_value=31, value=15)
    lat = st.number_input("Latitude", value=12.5340, format="%.6f")
    lon = st.number_input("Longitude", value=7.5600, format="%.6f")
    if st.button("Predict Crime Type"):
        if model is None:
            st.error("Model not available. The app includes a demo dataset and pre-trained model; if the model file is missing or incompatible, training is required.")
        else:
            x = dict.fromkeys(features, 0.0)
            x['Hour'] = hour
            x['Month'] = month
            x['Day'] = day
            x['Latitude'] = lat
            x['Longitude'] = lon
            input_vec = [x[f] for f in features]
            pred = model.predict([input_vec])[0]
            st.success(f"Predicted Crime Type: {pred}")
            st.info("Note: This is a prototype using mock data.")

with right:
    st.subheader("Dashboard")
    col1, col2 = st.columns(2)
    type_counts = df['Crime_Type'].value_counts()
    fig1, ax1 = plt.subplots(figsize=(5,3))
    type_counts.plot(kind='bar', ax=ax1)
    ax1.set_title("Crimes by Type")
    ax1.set_xlabel("Crime Type")
    ax1.set_ylabel("Count")
    st.pyplot(fig1)

    loc_counts = df['Location_Desc'].value_counts()
    fig2, ax2 = plt.subplots(figsize=(5,3))
    ax2.pie(loc_counts, labels=loc_counts.index, autopct='%1.1f%%', startangle=140)
    ax2.set_title("Crimes by Location")
    st.pyplot(fig2)

    df_month = df.copy()
    df_month['Month'] = df_month['Date'].dt.to_period('M').dt.to_timestamp()
    monthly = df_month.groupby('Month').size()
    fig3, ax3 = plt.subplots(figsize=(10,3))
    ax3.plot(monthly.index, monthly.values, marker='o')
    ax3.set_title("Monthly Crime Trend")
    ax3.set_xlabel("Month")
    ax3.set_ylabel("Number of Crimes")
    plt.xticks(rotation=45)
    st.pyplot(fig3)

    st.markdown("---")
    st.subheader("Data Preview")
    st.dataframe(df.head(10))

st.markdown("---")
st.write("Notes: This is a demo prototype for academic purposes using mock data for Dutsinma LGA.")