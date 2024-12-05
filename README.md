# Predictive Fleet Care
The goal is to predict maintenance needs for trucks using ML techniques and historical data, aiming to reduce unplanned downtimes and improve operational efficiency.

## Data Description
The dataset (fleet_train.csv) contains information about truck operations. Each record includes:
1. Features: Operational and environmental data (e.g., GPS, timestamps, fleet ID).
2. Target Variable: Maintenance_flag (binary: 1 for maintenance needed, 0 for no maintenance).

![Dataset_Distribution](https://github.com/user-attachments/assets/9716e0f8-8e90-48f4-91c8-b05732f5e46e)

## Pipeline
1. EDA
    * Step 1: Drop irrelevant columns (e.g., identifiers, GPS data).
    * Step 2: Remove low-variance features.
    * Step 3: Perform mutual information analysis to assess feature importance.
    * Step 4: Analyze correlations and drop highly correlated features (correlation > 0.80).
2. Model Training & Evaluation
    * Train three machine learning models:
      * Logistic Regression
      * Random Forest Classifier
      * Gradient Boosting Classifier
3. Evaluate models using:
    * Accuracy Score
    * Classification Report (Precision, Recall, F1-score)
    * Confusion Matrix

## 

## Installation
Clone the repository:
   ```bash
   git clone https://github.com/your-username/predictive-fleet-care.git
