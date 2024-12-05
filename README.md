# Predictive Fleet Care
The goal is to predict maintenance needs for trucks using ML techniques and historical data, aiming to reduce unplanned downtimes and improve operational efficiency.

## Project Overview
* Removing irrelevant and low-variance features.
  * Using mutual information and correlation analysis to identify and drop redundant features.
* Machine Learning Models:
  * Comparing Logistic Regression, Random Forest, and Gradient Boosting classifiers.
* Performance Metrics:
  * Accuracy, Precision, Recall, F1-score, and a visualized Confusion Matrix.
 
## Data Description
The dataset (fleet_train.csv) contains information about truck operations. Each record includes:
1. Features: Operational and environmental data (e.g., GPS, timestamps, fleet ID).
2. Target Variable: Maintenance_flag (binary: 1 for maintenance needed, 0 for no maintenance).

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
