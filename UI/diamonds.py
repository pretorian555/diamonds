
import numpy as np
import pandas as pd
import streamlit as st
import json
import requests

DATA_BUCKET ='gs://diamonds_data/diamonds.csv'

url_serving = 'https://diamonds-image-bfpumxj2xa-uc.a.run.app/v1/models/diamonds:predict'

st.title("Diamond price prediction tool")

st.image('diamond-434.gif')

st.subheader("Please, select your diamond characteristics:")

##load the diamond dataframe


df = pd.read_csv(DATA_BUCKET)
df.pop('price')

##selection.append(st.slider(column, min_value=float(red_df[column].min()),max_value=float(red_df[column].max()),value=float(red_df[column].mean()),format='%2f'))

def rest_request(text, url=None):
    
    if url is None:
        url='https://diamonds-image-bfpumxj2xa-uc.a.run.app/v1/models/diamonds:predict'
        
    payload = json.dumps({"instances":[text]})
    print(payload)
    response = requests.post(url,data=payload)
        
    return response
    

diamond_desc = {
    'carat':'Weight of the diamond',
    'cut': "Quality of the cut",
    'color':'Diamond color',
    'clarity':'Clarity of the diamond',
    'x':'Length in mm',
    'y':'Width in mm',
    'z': 'Depth in mm',
    'depth':'Total depth percentage',
    'table':'Width of top of the diamond relative to widest point'
}

diamond_values = {
    'cut':['Fair', 'Good', 'Very Good', 'Premium', 'Ideal'],
    'color':['D(best)','E','F','G','H','I','J(worst)'],
    'clarity':['I1 (worst)', 'SI2', 'SI1', 'VS2', 'VS1', 'VVS2', 'VVS1', 'IF (best)']  
}

NUMERIC_COLS = ['carat','x','y','z','depth','table']

CAT_COLS = ['cut','color','clarity']

selection = {}

for column in df:
    
    if column in NUMERIC_COLS:
        
        selection[column] = [st.slider(diamond_desc[column], min_value=float(df[column].min()),max_value=float(df[column].max()),value=float(round(df[column].mean(),2)))]

    if column in CAT_COLS:
        
        selection[column] = [st.selectbox(diamond_desc[column],diamond_values[column])]
        
        
##st.text(selection)

if st.button("Predict the price!"):
    rs = rest_request(selection, url_serving)
    st.subheader("The estimated price of the diamond is ${0:,}".format(int(rs.json()['predictions'][0][0]), unsafe_allow_html=True))

##rs = rest_request(selection)
##st.write(rs.json()['predictions'][0][0])
