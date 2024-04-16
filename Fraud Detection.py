import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

credit_card_data = pd.read_csv('credit_card_dataset.csv')

print("First few rows of the dataset:")
print(credit_card_data.head())

print("\nInformation about the dataset:")
print(credit_card_data.info())

print("\nSummary statistics of the dataset:")
print(credit_card_data.describe())

print("\nMissing values in the dataset:")
print(credit_card_data.isnull().sum())

print("\nDistribution of target variable:")
print(credit_card_data['Class'].value_counts())

plt.figure(figsize=(8, 6))
sns.countplot(x='Class', data=credit_card_data)
plt.title('Distribution of Target Variable')
plt.xlabel('Class')
plt.ylabel('Count')
plt.show()

plt.figure(figsize=(10, 8))
sns.heatmap(credit_card_data.corr(), annot=True, cmap='coolwarm', fmt='.2f', annot_kws={'size': 10})
plt.title('Correlation Heatmap with Enhanced Visibility', fontsize=8)
plt.show()

X = credit_card_data.drop('Class', axis=1)  # Features
y = credit_card_data['Class']  # Target variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("\nModel Evaluation:")
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
