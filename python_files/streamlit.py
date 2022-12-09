import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from dotenv import dotenv_values
from sqlalchemy import create_engine
import sklearn
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import pairwise
from sklearn.metrics import accuracy_score
import statsmodels.formula.api as sm
from scipy.sparse import csr_matrix
import os
import streamlit as st
import pickle
from thefuzz import process

# Function to select moiveids from the dataset
def title_id (title_list):
    movieid_list = movies.index[movies["title"].isin(title_list)].tolist()
    return movieid_list


st.header('Netflix Recommendation System')

# import all Data
movies = pd.read_csv("C:/Users/harsh/OneDrive/Documents/GitHub/thymestamps-working-folder/data/selected_movies.csv")
    # change the below file to filtered_ratings file
ratings = pd.read_csv("C:/Users/harsh/OneDrive/Documents/GitHub/thymestamps-working-folder/data/selected_ratings.csv")

# loading the model
loaded_model = pickle.load(open('../../final-project/notebooks/finalized_model.sav', 'rb'))
model = loaded_model.fit(user)

# try cache 

# creating list for dropdown menu
movie_list = movies['title'].values
movie_list

# User selecti0n as Dropdown menu
liked_items = st.multiselect( "Type or select a movie from the dropdown", movie_list, max_selection = 3 )

# creating vector for the user
user_vec = np.repeat(0, 17770)
user_vec[liked_items] = 5

# creating pivot table for user mapping
user = csr_matrix((ratings['rating'], (ratings['customerid'], ratings['movieid'])))

if st.button('Show Recommendation'):
      
    #running model

    model.kneighbors(user[1,:], n_neighbors=20)

    distances, user_ids = model.kneighbors([user_vec], n_neighbors=10)

    neighborhood = ratings.set_index('customerid').loc[user_ids[0]]

    recommendations = neighborhood.groupby('movieid')['rating'].mean().sort_values(ascending=False)

    item_filter = ~recommendations.index.isin(liked_items)
    recommendations = recommendations.loc[item_filter]

    recommended = movies.loc[recommendations.head(10).index]
    st.write(recommended)
    
st.button("Reset")




    
    
