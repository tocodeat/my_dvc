#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[2]:


data = pd.read_csv('/Users/kothasravani/my_dvc_project/athletes_raw.csv')


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


# Remove Outliers

data = data[data['weight'] < 1500]
data = data[data['age'] >= 18]
data = data[(data['height'] < 96) & (data['height'] > 48)]

data = data[(data['deadlift'] > 0) & (data['deadlift'] <= 1105)|((data['gender'] == 'Female') \
             & (data['deadlift'] <= 636))]
data = data[(data['candj'] > 0) & (data['candj'] <= 395)]
data = data[(data['snatch'] > 0) & (data['snatch'] <= 496)]
data = data[(data['backsq'] > 0) & (data['backsq'] <= 1069)]


# In[7]:


# Clean Survey Data

decline_dict = {'Decline to answer|': np.nan}
data = data.replace(decline_dict)
data = data.dropna(subset=['background','experience','schedule','howlong','eat'])


# In[8]:


data['gender'].value_counts()


# In[9]:


data = data[data['gender'] != '--']


# In[10]:


data.columns


# In[11]:


data.shape


# In[12]:


data['gender'].value_counts().plot(kind='bar')


# In[13]:


data = pd.get_dummies(data, columns=['gender'], drop_first=True)


# In[14]:


data['region'].value_counts().plot(kind='bar')


# In[15]:


data = pd.get_dummies(data, columns=['region'], drop_first=True)


# In[16]:


sns.distplot(data['age'])


# In[17]:


sns.distplot(data['height'])


# In[18]:


sns.distplot(data['weight'])


# In[19]:


sns.distplot(data['candj'])


# In[20]:


sns.distplot(data['snatch'])


# In[21]:


sns.distplot(data['deadlift'])


# In[22]:


sns.distplot(data['backsq'])


# In[23]:


data['total_lift'] = data['candj'] + data['snatch'] + data['deadlift'] + data['backsq']


# In[24]:


sns.distplot(data['total_lift'])


# In[25]:


data.drop(['candj','snatch','deadlift','backsq'], axis=1, inplace=True)


# In[26]:


## Creating new features


# In[27]:


# 1. Calculate BMI
data['bmi'] = data['weight'] / (data['height']/100)**2
data.drop(['height'], axis=1, inplace=True)


# In[28]:


# 2. Lift Efficiency
data['lift_efficiency'] = data['total_lift'] / data['weight']
data.drop(['weight'], axis=1, inplace=True)


# In[29]:


bins = [17, 27, 37, 47, 56]  # Define age bins
labels = ['Young Adults', 'Adults', 'Middle-Aged', 'Senior Adults']  # Corresponding labels for the bins

data['age_group'] = pd.cut(data['age'], bins=bins, labels=labels, right=True)
data = pd.get_dummies(data, columns=['age_group'], drop_first=True)
data.drop(['age'], axis=1, inplace=True)


# In[30]:


mean_lift = data['total_lift'].mean()
std_lift = data['total_lift'].std()

conditions = [
    data['total_lift'] < mean_lift - std_lift,
    (data['total_lift'] >= mean_lift - std_lift) & (data['total_lift'] <= mean_lift + std_lift),
    data['total_lift'] > mean_lift + std_lift
]

categories = ['Below average', 'Average', 'Above average']

data['performance_category'] = np.select(conditions, categories, default='Unknown')

data = pd.get_dummies(data, columns=['performance_category'], drop_first=True)


# In[31]:


# Creating binary columns for eat
data['eat_quality_foods'] = data['eat'].str.contains("I eat quality foods but don't measure the amount").astype(int)
data['eat_cheat_meals'] = data['eat'].str.contains("I eat 1-3 full cheat meals per week").astype(int)
data['eat_whatever_is_convenient'] = data['eat'].str.contains("I eat whatever is convenient").astype(int)
data['weigh_and_measure_food'] = data['eat'].str.contains("I weigh and measure my food").astype(int)
data['eat_strict_paleo'] = data['eat'].str.contains("I eat strict Paleo").astype(int)
data.drop('eat', axis=1, inplace=True)


# In[32]:


# Creating binary columns for background
data['played_youth_highschool_sports'] = data['background'].str.contains("I played youth or high school level sports").astype(int)
data['regularly_play_recreational_sports'] = data['background'].str.contains("I regularly play recreational sports").astype(int)
data['no_athletic_background'] = data['background'].str.contains("I have no athletic background besides CrossFit").astype(int)
data['played_college_sports'] = data['background'].str.contains("I played college sports").astype(int)
data['played_professional_sports'] = data['background'].str.contains("I played professional sports").astype(int)
data.drop('background', axis=1, inplace=True)


# In[33]:


# Creating binary columns for each category in the 'experience' feature

data['began_with_coach'] = data['experience'].str.contains("I began CrossFit with a coach \(e.g. at an affiliate\)").astype(int)
data['life_changing_experience'] = data['experience'].str.contains("I have had a life changing experience due to CrossFit").astype(int)
data['began_alone'] = data['experience'].str.contains("I began CrossFit by trying it alone \(without a coach\)").astype(int)
data['completed_level_1'] = data['experience'].str.contains("I have completed the CrossFit Level 1 certificate course").astype(int)
data['attended_specialty_courses'] = data['experience'].str.contains("I have attended one or more specialty courses").astype(int)
data['trains_others'] = data['experience'].str.contains("I train other people").astype(int)
data.drop('experience', axis=1, inplace=True)


# In[34]:


# Creating binary columns for each category in the 'schedule' feature
data['1_workout_a_day'] = data['schedule'].str.contains("I usually only do 1 workout a day").astype(int)
data['multi_workouts_3plus_times'] = data['schedule'].str.contains("I do multiple workouts in a day 3+ times a week").astype(int)
data['multi_workouts_2x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 2x a week").astype(int)
data['multi_workouts_1x_a_week'] = data['schedule'].str.contains("I do multiple workouts in a day 1x a week").astype(int)
data['strictly_schedule_rest'] = data['schedule'].str.contains("I strictly schedule my rest days").astype(int)
data['rest_4_or_more_days'] = data['schedule'].str.contains("I typically rest 4 or more days per month").astype(int)
data['rest_fewer_than_4_days'] = data['schedule'].str.contains("I typically rest fewer than 4 days per month").astype(int)
data.drop('schedule', axis=1, inplace=True)



# In[35]:


# Creating binary columns for each category in the 'howlong' feature
data['2_4_years'] = data['howlong'].str.contains("2-4 years").astype(int)
data['1_2_years'] = data['howlong'].str.contains("1-2 years").astype(int)
data['4plus_years'] = data['howlong'].str.contains("4+ years").astype(int)
data['6_12_months'] = data['howlong'].str.contains("6-12 months").astype(int)
data['less_than_6_months'] = data['howlong'].str.contains("Less than 6 months").astype(int)
data.drop('howlong', axis=1, inplace=True)


# In[36]:


data.info()


# In[37]:


from sklearn.preprocessing import StandardScaler

# Features to scale
features_to_scale = ['total_lift']

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler to the data and transform
data[features_to_scale] = scaler.fit_transform(data[features_to_scale])


# In[38]:


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


# In[39]:


# Extracting feature importances and names
features = X.columns
feature_importances = rf.feature_importances_

# Create a dataframe for feature importances
feature_df = pd.DataFrame({
    'Feature': features,
    'Importance': feature_importances
})

# Sort the dataframe based on importance
feature_df = feature_df.sort_values(by='Importance', ascending=False)

# Display the sorted dataframe
print(feature_df)



# In[40]:


# Define a list of columns to keep
columns_to_keep = [
    'gender_Male',
    'lift_efficiency',
    'performance_category_Average',
    'bmi',
    'performance_category_Below average',
    'played_college_sports',
    'total_lift' # You need to keep the dependent variable for now
]

# Drop all other columns
data = data[columns_to_keep]

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

