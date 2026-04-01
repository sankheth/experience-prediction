#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load dataset (for reference if needed)
path = r"C:\Users\gopin\OneDrive\Desktop\ML PROJECT\Data\Salary_Data.csv"
data = pd.read_csv(path)

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# UI
st.title("💼 Experience Predictor")

salary = st.number_input("Enter Salary", min_value=1000, step=1000)

if st.button("Predict"):
    result = model.predict(np.array([[salary]]))
    st.success(f"Required Experience: {result[0]:.2f} years")


# In[ ]:




