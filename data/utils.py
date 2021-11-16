"""Utility functions for project."""

from sklearn.ensemble import RandomForestRegressor

def fill_missing_values_rfr(df):
    """
    Use random forest regression to fill in missing values.
    ASSUMPTION: df only contains numeric data.
    """

    # Identify columns needed to be filled in
    columns = []
    X = df.copy()
    for col in df.columns:
        if len(X[X[col].isnull()]) > 0:
            columns += [col]

    # Let X = df with no rows/cols with nulls
    X = X.dropna(axis=0, how='any')
    X = X.drop(columns, axis=1)

    # Let Z = df with no rows with nulls
    Z = df.copy()
    Z = Z.dropna(axis=0, how='any')

    print("Number of columns to fit:", len(columns))

    # Perform RFR on each column with missing values
    predicted_cols = {}
    for col in columns:
        print("Currently fitting column", col)

        y = Z[col].to_numpy()
        
        rfr = RandomForestRegressor(random_state=0,
                                    n_estimators=500,
                                    verbose=1,
                                    n_jobs=-1)
        rfr.fit(X, y)

        X_in = df[df[col].isnull()]
        X_in = X_in.drop(columns=columns)
        X_in = X_in.dropna(axis=1,how='any').to_numpy()
        predicted = rfr.predict(X_in)
        predicted_cols[col] = predicted
    
    for col in columns:
        df.loc[(df[col].isnull()), col] = predicted_cols[col]

    return df
