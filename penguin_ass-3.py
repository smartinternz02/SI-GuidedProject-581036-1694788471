# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load the dataset
df = pd.read_csv('penguin_dataset.csv')

# 3. Perform Univariate Analysis
# Plot histograms for numerical attributes
df.hist(figsize=(12, 8))
plt.show()

# 3. Perform Bi-Variate Analysis
# Pairplot to visualize relationships between numerical attributes
sns.pairplot(df, hue='Species')
plt.show()

# 3. Perform Multi-Variate Analysis
# You can use seaborn's heatmap to visualize the correlation matrix
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.show()

# 4. Perform descriptive statistics
desc_stats = df.describe()
print(desc_stats)

# 5. Check for Missing values and deal with them
missing_values = df.isnull().sum()
print(missing_values)

# If missing values are found, you can fill them with appropriate values or drop rows/columns as needed.

# 6. Find and replace outliers (if necessary)
# You can use various methods like z-scores, IQR, or domain knowledge to identify and handle outliers.

# 7. Check the correlation of independent variables with the target
correlation_with_target = df.corr()['Species'].sort_values(ascending=False)
print(correlation_with_target)

# 8. Check for Categorical columns and perform encoding (if needed)
# You can use one-hot encoding for categorical columns like 'Island' and 'Sex'.

# Example:
# df_encoded = pd.get_dummies(df, columns=['Island', 'Sex'], drop_first=True)

# 9. Split the data into dependent and independent variables
X = df.drop('Species', axis=1)
y = df['Species']

# 10. Scaling the data (if needed)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 11. Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 12. Check the training and testing data shape
print("Training data shape:", X_train.shape, y_train.shape)
print("Testing data shape:", X_test.shape, y_test.shape)
