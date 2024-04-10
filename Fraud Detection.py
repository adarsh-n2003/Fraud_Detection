# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
credit_card_data = pd.read_csv('credit_card_dataset.csv')

# Exploratory Data Analysis (EDA)
# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(credit_card_data.head())

# Display information about the dataset
print("\nInformation about the dataset:")
print(credit_card_data.info())

# Summary statistics of the dataset
print("\nSummary statistics of the dataset:")
print(credit_card_data.describe())

# Check for missing values
print("\nMissing values in the dataset:")
print(credit_card_data.isnull().sum())

# Check the distribution of the target variable
print("\nDistribution of target variable:")
print(credit_card_data['Class'].value_counts())

# Visualize the distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=credit_card_data)
plt.title('Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

# Correlation heatmap with enhanced visibility
plt.figure(figsize=(10, 8))
sns.heatmap(credit_card_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 10})
plt.title('Correlation Heatmap with Enhanced Visibility', fontsize=8)
plt.show()

# Split the data into features and target variable
X = credit_card_data.drop('Class', axis=1)  # Features
y = credit_card_data['Class']  # Target variable

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Now, you can use this model to predict fraud transactions in real-time data.
