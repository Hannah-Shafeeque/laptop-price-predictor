import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np
import base64

from utils.data_transformer import DataTransformer   


def set_bg(image_file):
    with open(image_file, "rb") as img:
        encoded = base64.b64encode(img.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{encoded}");
        background-size: cover;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


set_bg("assets/background.jpg")

data = pd.read_csv('data/laptop_prices.csv')


def price():
    st.title("LAPTOP 💻 PRICE ")
    st.info('Know the Right Price Before You Pay!')

    st.subheader("Select the Features")


    col1, spacer, col2 = st.columns([1, 0.2, 1])

    with col1:
        st.markdown('<div class="left-align">', unsafe_allow_html=True)

        Brand = st.selectbox("**Brand**", data['Brand'].unique())
        RAM_GB = st.selectbox("**RAM (GB)**", data['RAM (GB)'].sort_values().unique())
        GPU = st.selectbox("**GPU**", data['GPU'].unique())
        Resolution = st.selectbox("**Resolution**", data['Resolution'].unique())
        Battery_Life = st.slider("**Battery Life (hours)**", 4.0, 12.0, 7.0, step=0.5)

    with col2:
        st.markdown('<div class="right-align">', unsafe_allow_html=True)

        Processor = st.selectbox("**Processor**", data['Processor'].unique())
        Storage = st.selectbox("**Storage**", data['Storage'].unique())
        Screen_Size = st.selectbox("**Screen Size (inch)**", data['Screen Size (inch)'].sort_values().unique())
        Operating_System = st.selectbox("**Operating System**", data['Operating System'].unique())
        Weight = st.slider("**Weight (kg)**", 1.2, 3.5, 2.0, step=0.1)

    df = pd.DataFrame([[Brand, Processor, RAM_GB, Storage, GPU, Screen_Size, Resolution, Battery_Life, Weight, Operating_System]],
                      columns=['Brand', 'Processor', 'RAM (GB)', 'Storage', 'GPU', 'Screen Size (inch)', 'Resolution', 'Battery Life (hours)', 'Weight (kg)', 'Operating System'])

    model = joblib.load('artifacts/rfmodel.sav')
    scaler = joblib.load('artifacts/rfscaler.sav')

    data_transformer = DataTransformer(df).transform()

    Brand_ohe = joblib.load('artifacts/Brand.sav')
    Operating_System_ohe = joblib.load('artifacts/Operating_System.sav')
    
    Brand = Brand_ohe.transform(df[['Brand']])
    dfBrand = pd.DataFrame(Brand, columns=Brand_ohe.get_feature_names_out(['Brand']))

    Operating_System = Operating_System_ohe.transform(df[['Operating System']])
    dfOperating_System = pd.DataFrame(Operating_System, columns=Operating_System_ohe.get_feature_names_out(['Operating System']))

    features = pd.concat([df, dfBrand, dfOperating_System], axis=1)
    features.drop(columns={'Brand', 'Operating System'}, inplace=True)


    pred = st.button('**GET  PRICE**')

    if pred:
        feature = scaler.transform(features)
        prediction = model.predict(feature)
        st.subheader('💰 **Estimated Price**')
        st.title(f'$ {prediction[0]:,.2f}')


price()
