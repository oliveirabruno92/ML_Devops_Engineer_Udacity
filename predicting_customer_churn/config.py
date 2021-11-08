class Config:

    CATEGORICAL_COLS = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]
    
    COLS_TO_PLOT = [
        "Churn", 
        "Customer_Age", 
        "Marital_Status", 
        "Total_Trans", 
        "Heatmap"
    ]
    
    COLS_TO_KEEP = [
        'Customer_Age', 
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio',
        'Gender_Churn', 
        'Education_Level_Churn', 
        'Marital_Status_Churn', 
        'Income_Category_Churn', 
        'Card_Category_Churn'
    ]
    
    PARAM_GRID = { 
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    RESULTS_IMG = [
        "logistic_results",
        "rf_results",
        "feature_importance",
        "roc_curve_results"
    ]

    DATA_PATH = r"./data/bank_data.csv"

    RESPONSE = 'Churn'
