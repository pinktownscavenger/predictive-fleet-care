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
   
      ![corr](https://github.com/user-attachments/assets/aaec6a37-95b5-453d-9323-2e3decd08527)
      
2. Model Training & Evaluation
    * Train three machine learning models:
      * Logistic Regression        
      * Random Forest Classifier        
      * Gradient Boosting Classifier        
3. Evaluate models using:
    * Accuracy Score
    * Classification Report (Precision, Recall, F1-score)
    * Confusion Matrix

## Confusion Matrices
1. Logistic Regression
![cm_logit](https://github.com/user-attachments/assets/406479d7-53a8-4ad0-a464-00a34c795d76)
2. Random Forest Classifier 
![cm_rfc](https://github.com/user-attachments/assets/cde3ae13-781d-482e-9a71-69a21cec4b13)
3. Gradient Boosting Classifier 
![cm_gbc](https://github.com/user-attachments/assets/8c18275b-f8b9-4d60-abf0-503faea34f6a)

## Installation
Clone the repository:
   ```bash
   git clone https://github.com/your-username/predictive-fleet-care.git
