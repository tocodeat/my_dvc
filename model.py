#!/usr/bin/env python
# coding: utf-8

# In[32]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Assuming 'data' is your dataframe
X = data.drop('total_lift', axis=1)  # All columns except the dependent variable
y = data['total_lift']

# Splitting the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating a Random Forest Regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Training the model
rf.fit(X_train, y_train)

# Predicting on the test data
y_pred = rf.predict(X_test)

# Evaluating the model using Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[33]:


from sklearn.metrics import r2_score
# Calculating the R^2 score
r2 = r2_score(y_test, y_pred)
print(f"R^2 Score: {r2:.2f}")


# In[36]:


import numpy as np

rmse = np.sqrt(mse)
print(f"Root Mean Squared Error: {rmse}")


# In[ ]:




