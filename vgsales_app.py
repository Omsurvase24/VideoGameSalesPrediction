import streamlit as st
import pickle
import sklearn
import pandas as pd
import numpy as np

model = pickle.load(open('vgmodel.sav', 'rb'))

st.title('Global Sales Prediction')
st.sidebar.header('Data')


def user_report():
    NA_Sales = st.sidebar.slider('NA_Sales', 0, 40, 1)
    EU_Sales = st.sidebar.slider('EU_Sales', 0, 40, 1)
    JP_Sales = st.sidebar.slider('JP_Sales', 0, 40, 1)

    user_report_data = {
        'NA_Sales': NA_Sales,
        'EU_Sales': EU_Sales,
        'JP_Sales': JP_Sales
    }

    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


user_data = user_report()
st.header('Data')
st.write(user_data)

Global_Sales = model.predict(user_data)
st.subheader('NA_Sales')
st.subheader('$'+str(np.round(Global_Sales[0], 2)))
