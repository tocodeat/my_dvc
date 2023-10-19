#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


import gdown
import pandas as pd

file_id = "1qcRjG8YEcGEF2W8_K718VNU_VnLUvJ1O"
url = f'https://drive.google.com/uc?id={file_id}'
output = 'temporary.csv'
gdown.download(url, output, quiet=False)

data = pd.read_csv(output)


# In[3]:


data.shape


# In[4]:


data.info()


# In[5]:


# Remove not relevant columns
data = data.dropna(subset=['region','age','weight','height','howlong','gender','eat', \
                               'train','background','experience','schedule','howlong', \
                               'deadlift','candj','snatch','backsq','experience',\
                               'background','schedule','howlong'])
data = data.drop(columns=['affiliate','team','name','athlete_id','fran','helen','grace',\
                              'filthy50','fgonebad','run400','run5k','pullups','train'])


# In[6]:


# Clean Survey Data

decline_dict = {'Decline to answer|': np.nan}
data = data.replace(decline_dict)
data = data.dropna(subset=['background','experience','schedule','howlong','eat'])


# In[7]:


data['gender'].value_counts()


# In[8]:


data = data[data['gender'] != '--']


# In[9]:


data.columns


# In[10]:


data.shape


# In[11]:


import matplotlib.pyplot as plt

# Define the list of categorical variables you want to plot
categorical_variables = ['gender', 'region']  # Add other variables as needed

# Create subplots based on the number of categorical variables
num_variables = len(categorical_variables)
fig, axes = plt.subplots(1, num_variables, figsize=(5 * num_variables, 5))

# Iterate through the categorical variables and plot a bar chart for each variable
for i, variable in enumerate(categorical_variables):
    ax = axes[i]
    data[variable].value_counts().plot(kind='bar', ax=ax)
    ax.set_title(variable)
    ax.set_xlabel(variable)
    ax.set_ylabel('Count')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the list of variables you want to plot
variables = ['age', 'height', 'weight']

# Use the "Set1" color palette for brighter colors
color_palette = "Set2"
color_gradient = sns.color_palette(color_palette, n_colors=len(variables))

# Create subplots for the four variables in a single row
fig, axes = plt.subplots(1, 3, figsize=(20, 5))

# Iterate through the variables and plot distribution plots for each variable with a color from the gradient
for i, (variable, color) in enumerate(zip(variables, color_gradient)):
    ax = axes[i]
    sns.distplot(data[variable], color=color, ax=ax)
    ax.set_title(variable)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


# In[13]:


import matplotlib.pyplot as plt
import seaborn as sns

# Define the list of variables you want to plot
variables = ['candj', 'snatch', 'deadlift', 'backsq']

# Use the "Set1" color palette for brighter colors
color_palette = "Set1"
color_gradient = sns.color_palette(color_palette, n_colors=len(variables))

# Calculate the number of rows needed
num_rows = (len(variables) + 2) // 3  # Rounds up to the nearest integer

# Create subplots with the calculated number of rows and three columns
fig, axes = plt.subplots(num_rows, 3, figsize=(18, 5 * num_rows))

# Iterate through the variables and plot distribution plots for each variable with a color from the gradient
for i, (variable, color) in enumerate(zip(variables, color_gradient)):
    row = i // 3  # Calculate the row index
    col = i % 3   # Calculate the column index
    ax = axes[row, col]
    sns.distplot(data[variable], color=color, ax=ax)
    ax.set_title(variable)
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')

# If there are empty subplots in the last row, remove them
if len(variables) % 3 != 0:
    for i in range(len(variables) % 3, 3):
        fig.delaxes(axes[-1, i])

# Adjust spacing between subplots
plt.tight_layout()

# Show the plots
plt.show()


# In[14]:


data['total_lift'] = data['candj'] + data['snatch'] + data['deadlift'] + data['backsq']


# In[15]:


sns.distplot(data['total_lift'])


# In[16]:


data.drop(['candj','snatch','deadlift','backsq'], axis=1, inplace=True)


# In[17]:


data = pd.get_dummies(data, columns=['gender'], drop_first=True)


# In[18]:


data = pd.get_dummies(data, columns=['region'], drop_first=True)


# In[19]:


# Creating binary columns for eat
data['eat_quality_foods'] = data['eat'].str.contains("I eat quality foods but don't measure the amount").astype(int)
data['eat_cheat_meals'] = data['eat'].str.contains("I eat 1-3 full cheat meals per week").astype(int)
data['eat_whatever_is_convenient'] = data['eat'].str.contains("I eat whatever is convenient").astype(int)
data['weigh_and_measure_food'] = data['eat'].str.contains("I weigh and measure my food").astype(int)
data['eat_strict_paleo'] = data['eat'].str.contains("I eat strict Paleo").astype(int)
data.drop('eat', axis=1, inplace=True)


# In[20]:


# Creating binary columns for background
data['played_youth_highschool_sports'] = data['background'].str.contains("I played youth or high school level sports").astype(int)
data['regularly_play_recreational_sports'] = data['background'].str.contains("I regularly play recreational sports").astype(int)
data['no_athletic_background'] = data['background'].str.contains("I have no athletic background besides CrossFit").astype(int)
data['played_college_sports'] = data['background'].str.contains("I played college sports").astype(int)
data['played_professional_sports'] = data['background'].str.contains("I played professional sports").astype(int)
data.drop('background', axis=1, inplace=True)


# In[21]:


# Creating binary columns for each category in the 'experience' feature

data['began_with_coach'] = data['experience'].str.contains("I began CrossFit with a coach \(e.g. at an affiliate\)").astype(int)
data['life_changing_experience'] = data['experience'].str.contains("I have had a life changing experience due to CrossFit").astype(int)
data['began_alone'] = data['experience'].str.contains("I began CrossFit by trying it alone \(without a coach\)").astype(int)
data['completed_level_1'] = data['experience'].str.contains("I have completed the CrossFit Level 1 certificate course").astype(int)
data['attended_specialty_courses'] = data['experience'].str.contains("I have attended one or more specialty courses").astype(int)
data['trains_others'] = data['experience'].str.contains("I train other people").astype(int)
data.drop('experience', axis=1, inplace=True)


# In[22]:


# Creating binary columns for each category in the 'schedule' feature
data['1_workout_a_day'] = data['schedule'].str.contains("I usually only do 1 workout a day").astype(int)
data['multi_workouts_3plus_times'] = data['schedule'].str.contains("I do multiple workouts in a day 3+ times a week").astype(int)
data['multi_workouts_2x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 2x a week").astype(int)
data['multi_workouts_1x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 1x a week").astype(int)
data['strictly_schedule_rest'] = data['schedule'].str.contains("I strictly schedule my rest days").astype(int)
data['rest_4_or_more_days'] = data['schedule'].str.contains("I typically rest 4 or more days per month").astype(int)
data['rest_fewer_than_4_days'] = data['schedule'].str.contains("I typically rest fewer than 4 days per month").astype(int)
data.drop('schedule', axis=1, inplace=True)



# In[23]:


# Creating binary columns for each category in the 'howlong' feature
data['2_4_years'] = data['howlong'].str.contains("2-4 years").astype(int)
data['1_2_years'] = data['howlong'].str.contains("1-2 years").astype(int)
data['4plus_years'] = data['howlong'].str.contains("4+ years").astype(int)
data['6_12_months'] = data['howlong'].str.contains("6-12 months").astype(int)
data['less_than_6_months'] = data['howlong'].str.contains("Less than 6 months").astype(int)
data.drop('howlong', axis=1, inplace=True)


# In[24]:


data.info()


# In[25]:


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import numpy as np

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

# Evaluating the model using various metrics:

# Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")

# Root Mean Squared Error (RMSE)
rmse = np.sqrt(mse)
print(f"Root Mean Squared Error (RMSE): {rmse}")

# Mean Absolute Error (MAE)
mae = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error (MAE): {mae}")

# R-squared (R^2)
r2 = r2_score(y_test, y_pred)
print(f"R-squared (R^2): {r2}")

# Explained Variance Score (EVS)
evs = explained_variance_score(y_test, y_pred)
print(f"Explained Variance Score (EVS): {evs}")

