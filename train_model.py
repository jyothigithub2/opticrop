# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import seaborn as sns
import joblib
sns.set_style("white")
# Read the Dataset
df = pd.read_csv("dataset/Crop_recommendation.csv")
# First 5 Rows
print("=====First 5 Rows =====")
print(df.head())
# Univariate Analysis
# Create figure
plt.figure(figsize=(16, 9))
plt.suptitle("Distribution of agricultural conditions", fontsize=22, fontweight='bold')
# Nitrogen
plt.subplot(2,4,1)
sns.histplot(df["N"], kde=True, color="orange")
plt.xlabel("Ratio of Nitrogen")
plt.ylabel("Density")
# Phosphorous
plt.subplot(2,4,2)
sns.histplot(df["P"], kde=True, color="blue")
plt.xlabel("Ratio of Phosphorous")
plt.ylabel("Density")
# Potassium
plt.subplot(2,4,3)
sns.histplot(df["K"], kde=True, color="violet")
plt.xlabel("Ratio of Potassium")
plt.ylabel("Density")
# Temperature
plt.subplot(2,4,4)
sns.histplot(df["temperature"], kde=True, color="green")
plt.xlabel("Ratio of Temperature")
plt.ylabel("Density")
# Humidity
plt.subplot(2,4,5)
sns.histplot(df["humidity"], kde=True, color="deepskyblue")
plt.xlabel("Ratio of Humidity")
plt.ylabel("Density")
# pH
plt.subplot(2,4,6)
sns.histplot(df["ph"], kde=True, color="crimson")
plt.xlabel("Ratio of PH")
plt.ylabel("Density")
# Rainfall
plt.subplot(2,4,7)
sns.histplot(df["rainfall"], kde=True, color="gold")
plt.xlabel("Ratio of Rainfall")
plt.ylabel("Density")
plt.tight_layout(rect=[0,0,1,0.95])
# Bivariate Analysis
plt.figure(figsize=(15,8))
plt.subplot(2,4,1)
sns.scatterplot(x=df['N'], y=df['label'])
plt.subplot(2,4,2)
sns.scatterplot(x=df['P'], y=df['label'])
plt.subplot(2,4,3)
sns.scatterplot(x=df['K'], y=df['label'])
plt.subplot(2,4,4)
sns.scatterplot(x=df['temperature'], y=df['label'])
plt.subplot(2,4,5)
sns.scatterplot(x=df['humidity'], y=df['label'])
plt.subplot(2,4,6)
sns.scatterplot(x=df['ph'], y=df['label'])
plt.subplot(2,4,7)
sns.scatterplot(x=df['rainfall'], y=df['label'])
plt.tight_layout()
plt.show()
# Multivariate Analysis
plt.figure(figsize=(10,6))
feature_counts = df.drop('label', axis=1).count()
feature_counts.index = [
    'nitrogen',
    'phosphorous',
    'potassium',
    'temperature',
    'humidity',
    'ph',
    'rainfall'
]
feature_counts.plot(
    kind='bar',
    color=['blue', 'orange', 'green', 'red', 'slateblue', 'brown', 'violet']
)
plt.title("Feature Counts", fontsize=16, fontweight='bold')
plt.xlabel("Features", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
print("Descriptive Analysis")
print(df.describe())
print(df.isnull().sum())

# Dataset Information
print("\n===== Dataset Information =====")
df.info()
# Outlier Detection
plt.figure(figsize=(12,6))
sns.boxplot(data=df.drop('label', axis=1))
plt.xticks(rotation=45)
plt.title("Outlier Detection")
plt.show()
# Extracting Seasonal Crops
print("=====Summer Crops =====")
summer_crops = df[(df['temperature'] > 30) &
                  (df['humidity'] > 50)]['label'].unique()
print(summer_crops)
print("=====Winter Crops =====")
winter_crops = df[(df['temperature'] < 20) &
                  (df['humidity'] > 30)]['label'].unique()
print(winter_crops)
print("=====Rainy Crops =====")
rainy_crops = df[(df['rainfall'] > 200) &
                (df['humidity'] > 50)]['label'].unique()
print(rainy_crops)
# Create Features and Target Variable
y = df['label']
X = df.drop(['label'], axis=1)
print("Shape of X :", X.shape)
print("Shape of y :", y.shape)
# Train-Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=0
)
print("\nThe shape of X_train :", X_train.shape)
print("The shape of X_test  :", X_test.shape)
print("The shape of y_train :", y_train.shape)
print("The shape of y_test  :", y_test.shape)
# K-Means Clustering
# Elbow Method
wcss = []
for i in range(1, 11):
    km = KMeans(
        n_clusters=i,
        init='k-means++',
        max_iter=300,
        n_init=10,
        random_state=42
    )
    km.fit(X)
    wcss.append(km.inertia_)
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.title("The Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    n_init=10,
    random_state=42
)
cluster = kmeans.fit_predict(X)
cluster_df = X.copy()
cluster_df['cluster'] = cluster
cluster_df['label'] = y.values
print("\nLet's check the results after applying K-Means Clustering\n")
for i in range(4):
    print(f"Crops in Cluster {i+1} :")
    print(cluster_df[cluster_df['cluster']==i]['label'].unique())
    print("-"*60)
# Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# Feature Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
# Train Model
lr = LogisticRegression(max_iter=5000)
lr.fit(X_train_scaled, y_train)
print("Logistic Regression")
print("Logistic Regression Model Trained Successfully")
# Prediction
lr_prediction = lr.predict(X_test_scaled)
print("\nFirst 10 Predictions")
print(lr_prediction[:10])
# Accuracy
lr_accuracy = accuracy_score(y_test, lr_prediction)
print("\nLogistic Regression Accuracy :", lr_accuracy)
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
# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
print("\nDecision Tree Model Trained Successfully")
dt_prediction = dt.predict(X_test)
print("First 10 Predictions")
print(dt_prediction[:10])
dt_accuracy = accuracy_score(y_test, dt_prediction)
print("Decision Tree Accuracy :", dt_accuracy)
# Random Forest
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
import joblib
# Create Random Forest Model
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
# Train Model
rf.fit(X_train, y_train)
print("Random Forest")
print("Random Forest Model Trained Successfully")
# Predict on Test Data
rf_prediction = rf.predict(X_test)
print("\nFirst 10 Predictions")
print(rf_prediction[:10])
# Accuracy
rf_accuracy = accuracy_score(y_test, rf_prediction)
print("\nRandom Forest Accuracy :", rf_accuracy)
# Classification Report
print("\nClassification Report")
print(classification_report(y_test, rf_prediction))
# Save Best Model
joblib.dump(rf, "model/crop_model.pkl")
print("\nRandom Forest Model Saved Successfully!")
# Predict Crop for Sample Input
sample_data = pd.DataFrame(
    [[105, 35, 40, 25, 64, 7, 160]],
    columns=[
        'N',
        'P',
        'K',
        'temperature',
        'humidity',
        'ph',
        'rainfall'
    ]
)
prediction = rf.predict(sample_data)
print("Crop Prediction Using Random Forest")
print("Recommended Crop :", prediction[0])
print(prediction)