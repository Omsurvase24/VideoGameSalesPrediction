import streamlit as st
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import pandas as pd

data = pd.read_csv('vgsales.csv')

st.title("Video Game Global Sales Prediction")

nav = st.sidebar.radio("Navigation", ["Home", "Prediction"])
if nav == "Home":
    st.write("home")
    if st.checkbox("Show Table"):
        st.table(data)

    graph = st.selectbox("What kind of graph ?", [
                         "Non-Interactive", "Interactive"])

    val = st.slider("Filer data using years", 1985, 2020)
    data = data.loc[data["Year"] >= val]
    if graph == "Non-Interactive":
        plt.figure(figsize=(10, 5))
        plt.scatter(data["NA_Sales"], data["Global_Sales"])
        plt.ylim(0)
        plt.xlabel("North America Sales")
        plt.ylabel("Global Sales")
        plt.tight_layout()
        st.pyplot(plt)
    if graph == "Interactive":
        layout = go.Layout(
            xaxis=dict(),  # Can set range if want to
            yaxis=dict()
        )
        fig = go.Figure(data=go.Scatter(
            x=data['NA_Sales'], y=data['Global_Sales'], mode="markers"), layout=layout)
        st.plotly_chart(fig)

if nav == "Prediction":
    pass
