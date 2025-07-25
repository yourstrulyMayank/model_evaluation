def preprocess(df):
    import pandas as pd
    import numpy as np

    # Load expected columns
    trained_columns_path = 'trained_columns.txt'
    with open(trained_columns_path, 'r') as f:
        trained_columns = [line.strip() for line in f.readlines()]

    # Remove ID columns if present
    if 'Loan_ID' in df.columns:
        df = df.drop('Loan_ID', axis=1)

    # Define features to exclude (same as training)
    exclude_features = ['Loan_ID', 'PD', 'LGD', 'EAD', 'Expected_Loss',
                      'Origination_Date', 'Risk_Weight', 'RWA', 'Regulatory_Capital']

    # Handle datetime columns if present
    if 'Origination_Date' in df.columns:
        df['Origination_Date'] = pd.to_datetime(df['Origination_Date'])
        # Extract useful datetime features
        df['Origination_Year'] = df['Origination_Date'].dt.year
        df['Origination_Month'] = df['Origination_Date'].dt.month
        df['Origination_Quarter'] = df['Origination_Date'].dt.quarter

    # Prepare features (exclude target and intermediate calculations)
    X = df.drop(exclude_features, axis=1, errors='ignore')
    
    # Get target variable
    y = df['Expected_Loss']

    # Handle categorical variables (same as training)
    categorical_features = ['Industry_Sector', 'Geographic_Region', 'Loan_Purpose']
    categorical_features = [col for col in categorical_features if col in X.columns]

    if categorical_features:
        X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=True)
    else:
        X_encoded = X

    # Handle boolean columns
    bool_columns = X_encoded.select_dtypes(include=['bool']).columns
    X_encoded[bool_columns] = X_encoded[bool_columns].astype(int)

    # Add any missing columns (from training) with 0
    for col in trained_columns:
        if col not in X_encoded.columns:
            X_encoded[col] = 0

    # Ensure columns are in the same order as training
    X_encoded = X_encoded[trained_columns]

    return X_encoded, y