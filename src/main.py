import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report

file_path = 'data/fleet_train.csv'
df = pd.read_csv(file_path)

irrelevant_columns = ['record_id', 'fleetid', 'Region', 'GPS_Longitude', 'GPS_Latitude', 
                      'GPS_Bearing', 'GPS_Altitude', 'Measurement_timestamp', 'truckid']
df = df.drop(columns=irrelevant_columns)

target = df['Maintenance_flag']
inputs = df.drop('Maintenance_flag', axis=1)

x_train, x_test, y_train, y_test = train_test_split(inputs, target, test_size=0.2, random_state=42)

features_to_drop = [col for col in x_train.columns if x_train[col].nunique() < 200] # drop low-uniqueness features
x_train = x_train.drop(columns=features_to_drop)
x_test = x_test.drop(columns=features_to_drop)

var_thres = VarianceThreshold(threshold=10) # variance threshold
var_thres.fit(x_train)

const_cols = [col for col in x_train.columns if col not in x_train.columns[var_thres.get_support()]] # drop low-variance features
x_train = x_train.drop(columns=const_cols)
x_test = x_test.drop(columns=const_cols)

mutual_info = pd.Series(mutual_info_classif(x_train, y_train), index=x_train.columns)
print("Mutual Information Scores:")
print(mutual_info.sort_values(ascending=False))

corrDf = x_train.corr()

plt.figure(figsize=(10, 10))
sns.heatmap(corrDf, annot=False, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# highly correlated features
corr_pairs = [(corrDf.columns[i], corrDf.columns[j]) 
              for i in range(len(corrDf.columns)) 
              for j in range(i) if corrDf.iloc[i, j] > 0.80]

features_to_drop = list(set([pair[0] for pair in corr_pairs])) # drop highly correlated features
x_train = x_train.drop(columns=features_to_drop)
x_test = x_test.drop(columns=features_to_drop)

print("Remaining features after EDA:", x_train.columns.tolist())

def evaluate_model(model, x_test, y_test, title):
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"{title} Accuracy: {accuracy:.2f}")
    print(classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Maintenance', 'Maintenance'], 
                yticklabels=['No Maintenance', 'Maintenance'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix ({title})')
    plt.show()

# Logistic Regression
print("\n--- Logistic Regression ---")
lr_model = LogisticRegression(random_state=42)
lr_model.fit(x_train, y_train)
evaluate_model(lr_model, x_test, y_test, "Logistic Regression")

# Random Forest Classifier
print("\n--- Random Forest Classifier ---")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)
evaluate_model(rf_model, x_test, y_test, "Random Forest Classifier")

# Gradient Boosting Classifier
print("\n--- Gradient Boosting Classifier ---")
gb_model = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_model.fit(x_train, y_train)
evaluate_model(gb_model, x_test, y_test, "Gradient Boosting Classifier")
