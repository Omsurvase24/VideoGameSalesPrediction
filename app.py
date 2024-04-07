import streamlit as st
from matplotlib import pyplot as plt
from plotly import graph_objs as go
import pandas as pd
# split
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import WordNetLemmatizer
from scipy.sparse import hstack, csr_matrix
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from joblib import dump, load


data = pd.read_csv('vgsales.csv')
swords = stopwords.words('english')


def clean_text(sent):
    tokens = word_tokenize(sent)
    wnl = WordNetLemmatizer()
    clean = ' '.join([wnl.lemmatize(word)
                     for word in tokens if word.isalpha()])
    return clean


tfidf = TfidfVectorizer(analyzer=clean_text)
text_columns = ['Name', 'Platform', 'Genre', 'Publisher']
nltk.download('wordnet')
X_text = tfidf.fit_transform(data[text_columns].apply(''.join, axis=1))
X_numerical = data[['Year', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']]
X_combined = hstack([X_text, csr_matrix(X_numerical)])
data['Global_Sales'] = np.log1p(data['Global_Sales'])
y = data['Global_Sales']
X_train, X_test, y_train, y_test = train_test_split(
    X_combined, y, test_size=0.25, random_state=42)
rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
rf_regressor.fit(X_train, y_train)
y_pred = rf_regressor.predict(X_test)

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
    st.header("Predict Global Sales")
    val = st.number_input("Enter NA Sales:", 0.00, 40.00, step=0.25)
    val = X_test
    y_pred = rf_regressor.predict(X_test)[0]

    if st.button("Predict"):
        st.success(f"Predicted Global Sale is: {y_pred}")
