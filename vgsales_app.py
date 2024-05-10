import streamlit as st
import pickle
import pandas as pd
import numpy as np

model = pickle.load(open('vgmodel.sav', 'rb'))

st.title('Global Sales Prediction')
st.sidebar.header('Data')


def user_report():
    NA_Sales = st.sidebar.slider('NA_Sales', 0, 40, 1)
    EU_Sales = st.sidebar.slider('EU_Sales', 0, 40, 1)
    JP_Sales = st.sidebar.slider('JP_Sales', 0, 40, 1)
    Other_Sales = st.sidebar.slider('Other_Sales', 0, 40, 1)

    # Create a dictionary with dummy values for unused features
    dummy_data = {
        # Assuming the model expects 63 features
        'Unused_' + str(i): 0 for i in range(59)
    }

    # Update the dictionary with the user input for relevant features
    dummy_data.update({
        'NA_Sales': NA_Sales,
        'EU_Sales': EU_Sales,
        'JP_Sales': JP_Sales,
        'Other_Sales': Other_Sales
    })

    # Create a DataFrame from the dictionary
    report_data = pd.DataFrame(dummy_data, index=[0])
    return report_data


user_data = user_report()
st.header('Data')
columns_to_display = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
st.write(user_data[columns_to_display])

Global_Sales = model.predict(user_data)
st.subheader('Predicted Global Sales')
st.subheader('$' + str(np.round(np.exp(Global_Sales[0]), 2)))
