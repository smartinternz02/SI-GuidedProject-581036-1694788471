import pandas as pd

# Load the datasets
red_wine = pd.read_csv('winequality-red.csv', delimiter=';')
white_wine = pd.read_csv('winequality-white.csv', delimiter=';')
# Explore the data
print(red_wine.head())
print(white_wine.head())

# Check for missing values
print(red_wine.isnull().sum())
print(white_wine.isnull().sum())

# Visualize data using histograms, box plots, or pair plots
import seaborn as sns
import matplotlib.pyplot as plt

# Example: Histogram of alcohol content in red wine
plt.figure(figsize=(8, 6))
sns.histplot(red_wine['alcohol'], bins=20, kde=True)
plt.xlabel('Alcohol Content')
plt.ylabel('Frequency')
plt.title('Distribution of Alcohol Content in Red Wine')
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Prepare the data and split into train and test sets
X = red_wine.drop('quality', axis=1)
y = red_wine['quality']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the Random Forest classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)
# Test with random observations
random_observation = X.sample(n=1, random_state=42)
predicted_quality = clf.predict(random_observation)
print("Predicted Quality:", predicted_quality[0])
