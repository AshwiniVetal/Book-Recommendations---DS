#!/usr/bin/env python
# coding: utf-8

# # Project 2 - Book Recommendations 

# ### Step 1 EDA

# In[1]:


#step1-import important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns 


# In[2]:


#step2-load the datas
books=pd.read_csv("Books.csv", low_memory=False)
users=pd.read_csv("Users.csv")
ratings=pd.read_csv("Ratings.csv")


# In[3]:


#step3- explore data
books.head()


# In[4]:


users.head()


# In[5]:


ratings.head()


# In[6]:


books.tail()


# In[7]:


users.tail()


# In[8]:


ratings.tail()


# In[9]:


books.info()


# In[10]:


users.info()


# In[11]:


ratings.info()


# In[12]:


books.describe()


# In[13]:


users.describe()


# In[14]:


ratings.describe()


# In[15]:


print(books.shape)
print(users.shape)
print(ratings.shape)


# In[16]:


#step4-merge/join DataFrames
data=pd.merge(ratings,users, on='User-ID')
data


# In[17]:


df=pd.merge(data,books, on='ISBN')
df


# In[18]:


df.info()


# In[19]:


df.describe()


# In[20]:


df.shape


# In[21]:


# Unique values and counts for categorical columns
print(df['Book-Title'].value_counts())
print(df['Book-Author'].value_counts())


# In[22]:


# step5- fill missing values
df.isnull().sum()


# In[23]:


df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)


# In[24]:


df.isnull().sum()


# In[25]:


duplicates=data.duplicated().sum()


# In[26]:


duplicates


# In[27]:


df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce')


# ### Step 2 Data Visualisation

# In[28]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set plot style
sns.set(style="whitegrid")


# In[29]:


import warnings
warnings.filterwarnings('ignore')


# In[30]:


# Histogram of book ratings
sns.histplot(df['Book-Rating'], bins=10, kde=True)
plt.title('Distribution of Book Ratings')
plt.xlabel('Book Rating')
plt.ylabel('Frequency')
plt.show()


# In[31]:


# Histogram of user ages
sns.histplot(df['Age'].dropna(), bins=20, kde=True)
plt.title('Distribution of User Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()


# In[32]:


# Bar plot of most rated books
top_books = df['Book-Title'].value_counts().head(10)
sns.barplot(x=top_books.values, y=top_books.index, palette='viridis')
plt.title('Top 10 Most Rated Books')
plt.xlabel('Number of Ratings')
plt.ylabel('Book Title')
plt.show()


# In[33]:


# Pie chart of most popular authors
top_authors = df['Book-Author'].value_counts().head(10)
plt.figure(figsize=(8,8))
top_authors.plot.pie(autopct='%1.1f%%', colors=sns.color_palette('viridis', 10))
plt.title('Top 10 Most Popular Authors')
plt.ylabel('')
plt.show()


# In[34]:


correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()


# ## Step 3: Data Cleaning and Preprocessing

# In[35]:


# Fill missing ages with median age
df['Age'].fillna(df['Age'].median(), inplace=True)

# Drop rows with missing book titles or authors
df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)


# In[36]:


# Convert Year-Of-Publication to integer
df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce').fillna(0).astype(int)


# In[37]:


from sklearn.preprocessing import MinMaxScaler

# Normalize Age and Book-Rating
scaler = MinMaxScaler()
df[['Age', 'Book-Rating']] = scaler.fit_transform(df[['Age', 'Book-Rating']])


# ## Step 4: Feature Engineering

# In[38]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


# In[39]:


# Handle missing values and data types
df['Age'].fillna(df['Age'].median(), inplace=True)
df.dropna(subset=['Book-Title', 'Book-Author'], inplace=True)
df['Year-Of-Publication'] = pd.to_numeric(df['Year-Of-Publication'], errors='coerce').fillna(0).astype(int)


# In[40]:


from sklearn.preprocessing import LabelEncoder
# Initialize label encoders
user_encoder = LabelEncoder()
book_encoder = LabelEncoder()

# Apply label encoders
df['User-ID'] = user_encoder.fit_transform(df['User-ID'])
df['Book-Title'] = book_encoder.fit_transform(df['Book-Title'])


# In[41]:


# Create features and target variable
X = df[['User-ID', 'Book-Title', 'Age', 'Year-Of-Publication']]
y = df['Book-Rating']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[42]:


X_train


# In[43]:


y_train


# ## Model Building and Evaluation

# In[44]:


# Initialize and train the model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)


# In[45]:


# Make predictions
y_pred_lr = lr_model.predict(X_test)


# In[46]:


# Evaluate the model
mse_lr = mean_squared_error(y_test, y_pred_lr)
print(f'Linear Regression MSE: {mse_lr}')


# In[47]:


# Initialize and train the model
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)


# In[48]:


# Make predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_dt


# In[49]:


# Evaluate the model
mse_dt = mean_squared_error(y_test, y_pred_dt)
print(f'Decision Tree MSE: {mse_dt}')


# In[50]:


# Initialize and train the model
knn_model = KNeighborsRegressor(n_neighbors=5)
knn_model.fit(X_train, y_train)


# In[51]:


# Make predictions
y_pred_knn = knn_model.predict(X_test)


# In[52]:


# Evaluate the model
mse_knn = mean_squared_error(y_test, y_pred_knn)
print(f'KNN MSE: {mse_knn}')


# In[53]:


# Initialize and train the model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)


# In[54]:


# Make predictions
y_pred_rf = rf_model.predict(X_test)


# In[55]:


# Evaluate the model
mse_rf = mean_squared_error(y_test, y_pred_rf)
print(f'Random Forest MSE: {mse_rf}')


# In[ ]:


# Initialize and train the model
svm_model = SVR()
svm_model.fit(X_train, y_train)


# In[ ]:


# Make predictions
y_pred_svm = svm_model.predict(X_test)


# In[ ]:


# Evaluate the model
mse_svm = mean_squared_error(y_test, y_pred_svm)
print(f'SVM MSE: {mse_svm}')


# In[ ]:


from flask import Flask, request, jsonify
import pandas as pd
import joblib


# In[ ]:


# Load the trained model and encoders
lr_model = joblib.load('lr_model.pkl')
user_encoder = joblib.load('user_encoder.pkl')
book_encoder = joblib.load('book_encoder.pkl')


# In[ ]:


def recommend_books(user_id, model, df, user_encoder, book_encoder, n_recommendations=5):
    try:
        # Encode the user ID
        user_encoded = user_encoder.transform([user_id])[0]
    except ValueError:
        return ["User ID not found."]

    # Prepare potential books
    all_books = df['Book-Title'].unique()
    potential_books = pd.DataFrame({
        'User-ID': user_encoded,
        'Book-Title': all_books,
        'Age': df['Age'].mean(),
        'Year-Of-Publication': df['Year-Of-Publication'].median()
    })
    
    # Filter out titles not in encoder
    valid_titles = potential_books['Book-Title'][potential_books['Book-Title'].isin(book_encoder.classes_)]
    
    # Encode book titles
    potential_books['Book-Title'] = book_encoder.transform(valid_titles)

    # Make predictions
    predicted_ratings = model.predict(potential_books)

    # Get top N book titles
    top_n_indices = predicted_ratings.argsort()[-n_recommendations:][::-1]
    top_n_books = valid_titles.iloc[top_n_indices]
    
    return top_n_books.tolist()

# Example usage
user_id = user_encoder.inverse_transform([df['User-ID'].iloc[0]])[0]
recommendations = recommend_books(user_id, lr_model, df, user_encoder, book_encoder)
print(f'Recommended books for user {user_id}: {recommendations}')


# In[ ]:




