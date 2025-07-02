def preprocess(df):
    import pandas as pd

    # Load expected columns
    trained_columns_path = r'uploads\\Linear Regression - Credit Rating Prediction\\trained_columns.txt'
    with open(trained_columns_path, 'r') as f:
        trained_columns = [line.strip() for line in f.readlines()]

    # Remove ID if present
    if 'ID' in df.columns:
        df = df.drop('ID', axis=1)

    # Extract features
    features = ['WC_TA', 'RE_TA', 'EBIT_TA', 'MVE_BVTD', 'S_TA']

    # Handle 'Industry' as categorical
    industry_categories = [int(col.split('_')[1]) for col in trained_columns if col.startswith('Industry_')]
    df['Industry'] = pd.Categorical(df['Industry'], categories=industry_categories)

    # One-hot encode 'Industry'
    industry_ohe = pd.get_dummies(df['Industry'], prefix='Industry', drop_first=True)

    # Combine features
    X = pd.concat([df[features], industry_ohe], axis=1)

    # Add any missing columns (from training) with 0
    for col in trained_columns:
        if col not in X.columns:
            X[col] = 0

    # Ensure columns are in the same order
    X = X[trained_columns]

    # Get y as ordinal-encoded (numerical) target
    # Assume last column is the target (as in your test.csv)
    target_col = df.columns[-1]
    ordered_ratings = ['C', 'CC', 'CCC', 'B', 'BB', 'BBB', 'A', 'AA', 'AAA']
    rating_to_num = {r: i for i, r in enumerate(ordered_ratings)}
    y = df[target_col].map(rating_to_num)

    return X, y