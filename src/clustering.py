import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def perform_kmeans_clustering(df, features, k=3):
    """
    Apply K-Means clustering on selected features and return dataframe with cluster labels.
    
    Parameters:
        df (pd.DataFrame): Cleaned telco dataset
        features (list): List of numeric column names to use for clustering
        k (int): Number of clusters (default=3)
        
    Returns:
        pd.DataFrame: Original DataFrame with added 'Segment' column
        KMeans: Trained KMeans model
    """
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])
    
    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    df['Segment'] = kmeans.fit_predict(X_scaled)

    return df, kmeans
