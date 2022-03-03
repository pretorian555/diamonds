# Diamond price prediction tool

## Overview

The goal of this project is to develop a machine learning model that would predict a price of a diamond based on several characteristics. The model was trained using a 50K record [dataset](https://www.kaggle.com/shivam2503/diamonds)

The model was trained using **TensorFlow2** framework and stored on google Cloud Storage for easier access. 

Trained model was deployed as a REST endpoing microservice on Google Cloud Run using **Tensorflow serving** docker image. 

The model predictions can be queried using a simple streamlit web interface hosted on Google Cloud Run service. 

Please, click below to try it!

<a href="https://diamonds-ui-image-bfpumxj2xa-uc.a.run.app/" target="_blank"> Diamond Price Prediction Tool</a> 

## Repo organization

Initial notebook where a tf model was developed

UI folder contains a streamlit python model for model querying

serving folder contains a Dockerfile to build a tensorflow serving image

TODO: Notebook with tfx pipeline code 


