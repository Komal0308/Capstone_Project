import pandas as pd   
import numpy as np    
from sklearn import *
import pickle
import streamlit as st

# Load the model and dataset
model = pickle.load(open('rf.pkl','rb'))
df = pickle.load(open('car_data.pkl','rb'))

st.title('car Price Prediction')
st.header('Fill the details to predict Car Price')

Year=st.selectbox('year',df['year'].unique())
Km_driven=st.selectbox('km_driven',sorted(df['km_driven'].unique()))
Fuel=st.selectbox('fuel',df['fuel'].unique())
Seller_type=st.selectbox('seller_type',df['seller_type'].unique())
Transmission=st.selectbox('transmission',df['transmission'].unique())
Owner=st.selectbox('owner',df['owner'].unique())
Car_name=st.selectbox('car_name',df['car_name'].unique())

if st.button('Predict car price'):
    prediction=model.predict([[Year,Km_driven,Fuel,Seller_type,Transmission,Owner,Car_name]])
    output=round(prediction[0],2)
    st.success('You can Sell Your car for {}'.format(output))