# -*- coding: utf-8 -*-
"""
Created on Sun Jun 24 12:13:44 2023

@author: JonBest
"""

# Jon Best
# 6/25/2023
# CS379 - Machine Learning
# Machine Learning Algorithm Evaluation 
# The purpose of this Python code is to use the Logistic Regression algorithm 
# to evaluate statistical information about gun related deathes in the US 2006-2020, 
# and to predict the most likely race, sex, and intent based on the dataset information.
 
#***************************************************************************************
# Title: Classification with Naive Bayes
# Author: Hasdemir, B.
# Date: 2020
# Availability: https://www.kaggle.com/code/barishasdemir/classification-with-naive-bayes
#
# Title: How to build Naive Bayes classifiers using Python Scikit-learn?
# Author: Leekha, G. 
# Date: 2022
# Availability: https://www.tutorialspoint.com/how-to-build-naive-bayes-classifiers-using-python-scikit-learn
#
# Title: Naive Bayes Classification Tutorial using Scikit-learn
# Author: DataCamp
# Date: 2023
# Availability: https://www.datacamp.com/tutorial/naive-bayes-scikit-learn
#
# Title: Titanic Survival Prediction Using Machine Learning
# Author: randerson112358
# Date: 2019
# Availability: https://betterprogramming.pub/titanic-survival-prediction-using-machine-learning-4c5ff1e3fa16
#
# Title: Visualize data from CSV file in Python
# Author: greeshmanalla
# Date: 2021
# Availability: https://www.geeksforgeeks.org/visualize-data-from-csv-file-in-python/
#
#***************************************************************************************

# Imported libraries include: pandas to develop dataframes, numpy to calculate complex math, 
# sklearn for machine learning functions, and matplotlib plus seasborn for graphic representation of data.
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Reading CSV file to retrieve the required data.
data = pd.read_csv('gun_related_deaths.csv')

# Explores data analysis of dataset. 
print(data.shape)
print(data.head())
print(data.dtypes)

# Creates a bar graph that displays the count of gun deaths by race.
sns.set(style="darkgrid")
race_counts = data['race'].value_counts()
plt.figure(figsize=(8, 6))
sns.barplot(x=race_counts.index, y=race_counts.values, palette="rocket")
plt.title("Gun Deaths by Race")
plt.xlabel("Race")
plt.ylabel("Count")
plt.show()

# Creates histogram graph that displays age distribution.
plt.figure(figsize=(8, 6))
sns.histplot(data['age'], bins=20, kde=True)
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Age Distribution')
plt.show()

# Creates bar graph that shows the total for each intent category.
plt.figure(figsize=(8, 6))
intent_counts = data['intent'].value_counts()
sns.barplot(x=intent_counts.index, y=intent_counts.values)
plt.xlabel('Intent')
plt.ylabel('Count')
plt.title('Intent Categories')
plt.show()

# Filters homicides from the intent column
filtered_data = data[data['intent'] == 'Homicide']

# Groups data by year and calculates the count of homicides
homicide_counts = filtered_data.groupby('year').size()

# Creates a line graph that displays the increase of homicides from 2006 to 2020.
plt.figure(figsize=(10, 6))
homicide_counts.plot()
plt.xlabel('Year')
plt.ylabel('Homicide Count')
plt.title('Trend of Homicide Counts Over the Years')
plt.show()

# Filters the data based on specific intent words.
intent_words = ['Homicide', 'Suicide', 'Accidental', 'Undetermined']
filtered_data = data[data['intent'].isin(intent_words)]

# Creates a scatter plot graph to show a overview of the gun death intents per year.
plt.figure(figsize=(10, 6))

# Plot points for each intent category.
for intent in intent_words:
    category_data = filtered_data[filtered_data['intent'] == intent]
    plt.scatter(category_data['age'], category_data['year'], label=intent)

plt.xlabel('Age')
plt.ylabel('Year')
plt.title('Scatter Plot of Gun Deaths by Intent')
plt.legend()
plt.show()
plt.close()


# Filters the data based on specific intent words.
intent_words = ['Homicide', 'Suicide', 'Accidental', 'Undetermined']
filtered_data = data[data['intent'].isin(intent_words)]

# Creates line graph to show a different overview of the gun death intents per year.
plt.figure(figsize=(10, 6))

intent_counts = filtered_data['intent'].value_counts()

# Plot line graph for each intent category.
for intent in intent_counts.index:
    intent_data = filtered_data[filtered_data['intent'] == intent]
    intent_data_counts = intent_data.groupby('year').size()
    plt.plot(intent_data_counts.index, intent_data_counts.values, label=intent)

plt.xlabel('Time')
plt.ylabel('Count')
plt.title('Intent Count Over Time')
plt.legend()

# Display the line graph.
plt.show()
plt.close()

# Ensure that dataset is void of null values.
print(data.isna().sum())

# Display unique objects before conversion.
print(data['intent'].unique())
print(data['sex'].unique())
print(data['race'].unique())
print(data['place'].unique())
print(data['education'].unique())

#Convert object datatypes to integers for all applicable features.
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()

# Transform the 'intent' column to integers.
data.iloc[:,2]= labelencoder.fit_transform(data.iloc[:,2].values)

# Transform the 'sex' column to integers.
data.iloc[:,4]= labelencoder.fit_transform(data.iloc[:,4].values)

# Transform the 'age' column to integers.
data.iloc[:,5]= labelencoder.fit_transform(data.iloc[:,5].values)

# Transform the 'race' column to integers.
data.iloc[:,6]= labelencoder.fit_transform(data.iloc[:,6].values)

# Transform the 'place' column to integers.
data.iloc[:,8]= labelencoder.fit_transform(data.iloc[:,8].values)

# Transform the 'education' column to integers.
data.iloc[:,9]= labelencoder.fit_transform(data.iloc[:,9].values)

# Display unique objects after conversion.
print(data['intent'].unique())
print(data['sex'].unique())
print(data['race'].unique())
print(data['place'].unique())
print(data['education'].unique())

# Ensure that dataset is void of null values.
print(data.isna().sum())

# Set feature selections and target variables for each prediction.
intent_features = ['age', 'sex', 'race', 'place', 'education'] 
sex_features = ['age', 'sex', 'race', 'place', 'education'] 
race_features = ['age', 'sex', 'race', 'place', 'education'] 

# Set X and Y variables for each prediction task.
X_intent = data[intent_features]
y_intent = data['intent']

X_sex = data[sex_features]
y_sex = data['sex']

X_race = data[race_features]
y_race = data['race']

# Splits the data into training and testing sets for each prediction task.
X_intent_train, X_intent_test, y_intent_train, y_intent_test = train_test_split(X_intent, y_intent, test_size=0.2, random_state=42)
X_sex_train, X_sex_test, y_sex_train, y_sex_test = train_test_split(X_sex, y_sex, test_size=0.2, random_state=42)
X_race_train, X_race_test, y_race_train, y_race_test = train_test_split(X_race, y_race, test_size=0.2, random_state=42)

# Initiates model training for each prediction task.
intent_model = GaussianNB()
intent_model.fit(X_intent_train, y_intent_train)

sex_model = GaussianNB()
sex_model.fit(X_sex_train, y_sex_train)

race_model = GaussianNB()
race_model.fit(X_race_train, y_race_train)

# Initiates model evaluation and displays accuracy for each prediction task.
intent_pred = intent_model.predict(X_intent_test)
intent_accuracy = accuracy_score(y_intent_test, intent_pred)
print("Intent Accuracy:", intent_accuracy)

sex_pred = sex_model.predict(X_sex_test)
sex_accuracy = accuracy_score(y_sex_test, sex_pred)
print("Sex Accuracy:", sex_accuracy)

race_pred = race_model.predict(X_race_test)
race_accuracy = accuracy_score(y_race_test, race_pred)
print("Race Accuracy:", race_accuracy)

# Initiates model deployment and displays prediction for each prediction task.
new_intent_instance = pd.DataFrame([[30, '1', '4', '1', '3']], columns=intent_features)
intent_prediction = intent_model.predict(new_intent_instance)
print("Predicted Intent:", intent_prediction)

new_sex_instance = pd.DataFrame([[30, '1', '4', '1', '3']], columns=sex_features)
sex_prediction = sex_model.predict(new_sex_instance)
print("Predicted Sex:", sex_prediction)

new_race_instance = pd.DataFrame([[30, '1', '4', '1', '3']], columns=race_features)
race_prediction = race_model.predict(new_race_instance)
print("Predicted Race:", race_prediction)


