import pandas as pd
from sklearn.preprocessing import StandardScaler

# Loading the dataset
data_path = "C:/Users/tjrom/Desktop/ml_wine/winequality-red.csv"
wine_data = pd.read_csv(data_path, delimiter=';')

# Initial Data Exploration
print(wine_data.head()) # Display the first few rows of the dataset (testing)
print(wine_data.describe()) #Display statistical summaries of the data.
print(wine_data.info()) # View data types and check for null values.

# Check for missing values
print(wine_data.isnull().sum())

# If any missing values, use common method of filling them with the mean of the column.
# wine_data.fillna(wine_data.mean(), inplace=True)

# Handle Outliers using IQR
Q1 = wine_data.quantile(0.25)
Q3 = wine_data.quantile(0.75)
IQR = Q3 - Q1

wine_data = wine_data[~((wine_data < (Q1 - 1.5 * IQR)) | (wine_data > (Q3 + 1.5 * IQR))).any(axis=1)]

# Feature Engineering
# For this dataset, major feature engineering might not be required, but this is where you'd perform such operations.

# Normalization using Z-score normalization
scaler = StandardScaler()
wine_data_scaled = pd.DataFrame(scaler.fit_transform(wine_data), columns=wine_data.columns)

# Save the Cleaned and Preprocessed Data
wine_data_scaled.to_csv("C:/Users/tjrom/Desktop/ml_wine/cleaned_winequality-red.csv", index=False)

print("Data cleaning and preprocessing completed!")