# KD34403-ML-Project

## Milestone 1: Data Pipeline (Preprocessing)

In this milestone, we prepared the Telco Customer Churn dataset for machine learning.

### Steps Completed
- Loaded and inspected the dataset (7043 rows, 21 columns)
- Performed exploratory data analysis (EDA)
- Removed irrelevant column (`customerID`)
- Converted `TotalCharges` to numeric format
- Handled missing values (removed 11 rows)
- Encoded categorical variables using one-hot encoding
- Converted target variable `Churn` into binary (0/1)
- Split dataset into training and testing sets (80/20)
- Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`)

### Output
- Final dataset: 7032 rows, 30 features
- Data is ready for model training in the next milestone
