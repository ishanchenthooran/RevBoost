import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def train_logistic_model(df, features, target='Churn'):
    """
    Train Logistic Regression model and return the model and evaluation report.
    
    Parameters:
        df (pd.DataFrame): Cleaned and clustered dataset
        features (list): List of features to use for prediction
        target (str): Target column name (default: 'Churn')
        
    Returns:
        LogisticRegression: Trained model 
        dict: Classification report as a dictionary
    """

    # Encoding categorical features
    for col in features:
        if df[col].dtype == 'object':
            df[col] = df[col].astype('category').cat.codes

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    df = df.copy()
    df['Churn_Prob'] = model.predict_proba(X)[:, 1]

    return df, model
