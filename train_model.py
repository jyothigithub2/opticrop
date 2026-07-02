# ==========================================
# Import Libraries
# ==========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
# ==========================================
# Load Dataset
# ==========================================
df = pd.read_csv("dataset/Crop_recommendation.csv")
# ==========================================
# Dataset Exploration
# ==========================================
print("========== First 5 Rows ==========")
print(df.head())
print("\n========== Dataset Information ==========")
df.info()
print("\n========== Statistical Summary ==========")
print(df.describe())
print("\n========== Missing Values ==========")
print(df.isnull().sum())
print("\n========== Crop Distribution ==========")
print(df['label'].value_counts())
# ==========================================
# Crop Distribution (Bar Chart)
# ==========================================
plt.figure(figsize=(12,6))
sns.countplot(x='label', data=df)
plt.xticks(rotation=90)
plt.title("Crop Distribution")
plt.show()
# ==========================================
# Univariate Analysis
# ==========================================
plt.figure(figsize=(15,10))
columns = ['N', 'P', 'K', 'temperature',
           'humidity', 'ph', 'rainfall']
for i, col in enumerate(columns):
    plt.subplot(2,4,i+1)
    sns.histplot(df[col], kde=True)
    plt.title(col)
plt.tight_layout()
plt.show()
# ==========================================
# Bivariate Analysis
# ==========================================
plt.figure(figsize=(10,8))
sns.heatmap(df.drop('label', axis=1).corr(),
            annot=True,
            cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
# ==========================================
# Multivariate Analysis
# ==========================================
sns.pairplot(df, hue='label')
plt.show()
# ==========================================
# Outlier Detection
# ==========================================
plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop('label', axis=1))
plt.xticks(rotation=45)
plt.title("Outlier Detection")
plt.show()
# ==========================================
# Extracting Seasonal Crops
# ==========================================
print("========== Summer Crops ==========")
summer_crops = df[(df['temperature'] > 30) &
                  (df['humidity'] > 50)]['label'].unique()
print(summer_crops)
print("========== Winter Crops ==========")
winter_crops = df[(df['temperature'] < 20) &
                  (df['humidity'] > 30)]['label'].unique()
print(winter_crops)
print("========== Rainy Crops ==========")
rainy_crops = df[(df['rainfall'] > 200) &
                (df['humidity'] > 50)]['label'].unique()
print(rainy_crops)
# ==========================================
# Create Features and Target Variable
# ==========================================
X = df.drop('label', axis=1)
y = df['label']
print("\n========== Input Features ==========")
print(X.head())
print("\n========== Target Variable ==========")
print(y.head())
# ==========================================
# Train Test Split
# ==========================================
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)
print("\n========== Train Test Split ==========")
print("Training Data Shape :", X_train.shape)
print("Testing Data Shape  :", X_test.shape)
print("Training Labels Shape :", y_train.shape)
print("Testing Labels Shape  :", y_test.shape)
# ==========================================
# KNN
# ==========================================
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
print("\nKNN Model Trained Successfully")
prediction = knn.predict(X_test)
print("First 10 Predictions")
print(prediction[:10])
accuracy = accuracy_score(y_test, prediction)
print("KNN Accuracy :", accuracy)
# ==========================================
# Logistic Regression
# ==========================================
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)
print("\nLogistic Regression Model Trained Successfully")
lr_prediction = lr.predict(X_test_scaled)
print("First 10 Predictions")
print(lr_prediction[:10])
lr_accuracy = accuracy_score(y_test, lr_prediction)
print("Logistic Regression Accuracy :", lr_accuracy)
# ==========================================
# Decision Tree
# ==========================================
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("\nDecision Tree Model Trained Successfully")
dt_prediction = dt.predict(X_test)
print("First 10 Predictions")
print(dt_prediction[:10])
dt_accuracy = accuracy_score(y_test, dt_prediction)
print("Decision Tree Accuracy :", dt_accuracy)
# ==========================================
# Random Forest
# ==========================================
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
print("\nRandom Forest Model Trained Successfully")
rf_prediction = rf.predict(X_test)
print("First 10 Predictions")
print(rf_prediction[:10])
rf_accuracy = accuracy_score(y_test, rf_prediction)
print("Random Forest Accuracy :", rf_accuracy)
# ==========================================
# K-Means Clustering
# ==========================================
from sklearn.cluster import KMeans
kmeans = KMeans(
    n_clusters=22,
    random_state=42,
    n_init=10
)
kmeans.fit(X)
print("\nK-Means Model Trained Successfully")
# ==========================================
# Model Comparison
# ==========================================
print("\n===================================")
print("Model Accuracy Comparison")
print("===================================")
print("KNN Accuracy                 :", accuracy)
print("Logistic Regression Accuracy :", lr_accuracy)
print("Decision Tree Accuracy       :", dt_accuracy)
print("Random Forest Accuracy       :", rf_accuracy)
# ==========================================
# Save Best Model
# ==========================================
joblib.dump(rf, "model/crop_model.pkl")
print("\nRandom Forest Model Saved Successfully!")