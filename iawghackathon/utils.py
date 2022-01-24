import sklearn.metrics
import sklearn.cluster
import pandas as pd


def score(y_pred, csv_path='Trainingdata_label.csv'):
    """
    Normalized Mutual Information Score
    """
    y_true = pd.read_csv(csv_path)['Ligand_Label']
    return sklearn.metrics.normalized_mutual_info_score(y_true, y_pred)

def clustering(X, n_clusters):
    """
    K-Means Clustering
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans.labels_