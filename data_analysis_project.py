### Data Analyst Project: Comprehensive Analysis from Scratch

# **Project Overview:**
# This project involves analyzing a dataset step-by-step, covering every major topic required for a data analyst role.

# **Step 1: Import Libraries**
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# **Step 2: Load Dataset**
# For this project, we'll use the 'House Prices' dataset from Kaggle.
# Ensure you have downloaded the dataset and placed it in the working directory.
dataset_path = "house_prices.csv"  # Replace with your file path
data = pd.read_csv(dataset_path)

# **Step 3: Data Overview**
print("First 5 rows of the dataset:")
print(data.head())
print("\nDataset information:")
print(data.info())
print("\nSummary statistics:")
print(data.describe())

# **Step 4: Data Cleaning**
# 4.1 Check for missing values
print("\nMissing values per column:")
print(data.isnull().sum())

# 4.2 Handle missing values
# - Drop columns with a high percentage of missing values
# - Impute missing values for numeric and categorical columns
threshold = 0.3  # Drop columns with >30% missing values
missing_fraction = data.isnull().sum() / len(data)
data = data.loc[:, missing_fraction <= threshold]

# Fill missing numeric values with the median
for col in data.select_dtypes(include=[np.number]).columns:
    data[col] = data[col].fillna(data[col].median())

# Fill missing categorical values with the mode
for col in data.select_dtypes(include=["object"]).columns:
    data[col] = data[col].fillna(data[col].mode()[0])

print("\nMissing values after cleaning:")
print(data.isnull().sum())

# **Step 5: Exploratory Data Analysis (EDA)**
# 5.1 Univariate Analysis
plt.figure(figsize=(10, 6))
sns.histplot(data['price'], kde=True, color='blue')
plt.title("Distribution of Sale Prices")
plt.xlabel("Sale Price")
plt.ylabel("Frequency")
plt.show()

# 5.2 Correlation Matrix
correlation_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

# 5.3 Pairplot for Selected Features
selected_features = ['SalePrice', 'GrLivArea', 'OverallQual', 'GarageCars', 'TotalBsmtSF']
sns.pairplot(data[selected_features])
plt.show()

# **Step 6: Feature Engineering**
# 6.1 Encoding Categorical Variables
categorical_columns = data.select_dtypes(include=["object"]).columns
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)

# 6.2 Feature Selection
correlation_target = correlation_matrix['SalePrice'].sort_values(ascending=False)
print("\nTop correlated features with SalePrice:")
print(correlation_target.head(10))

# Select top features based on correlation
important_features = correlation_target.index[:10]
data = data[important_features]

# **Step 7: Model Building**
# Define features (X) and target variable (y)
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# **Step 8: Model Evaluation**
# Predict on test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

# **Step 9: Visualize Model Performance**
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.7, color='green')
plt.title("Actual vs Predicted Sale Prices")
plt.xlabel("Actual Sale Prices")
plt.ylabel("Predicted Sale Prices")
plt.show()

# **Step 10: Conclusion**
# - Summarize findings from the EDA and model evaluation.
# - Highlight the features that most influence house prices.
# - Suggest potential next steps for improving the model, such as using advanced algorithms or feature engineering.
