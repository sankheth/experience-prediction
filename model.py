#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
from sklearn.linear_model import LinearRegression
import pickle

# Load dataset using full path
data = pd.read_csv("Salary_Data.csv")

# Reverse mapping (Salary → Experience)
X = data[["Salary"]]
y = data["YearsExperience"]

# Train model
model = LinearRegression()
model.fit(X, y)

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully!")

