import pandas as pd
from tqdm import tqdm
from kmodes.kprototypes import KPrototypes
from kmodes.kmodes import KModes
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score


def calculate_metric(data_matrix, clustering_info):
    results = {}
    for cluster, clustering in tqdm(clustering_info.items()):
        labels = clustering['labels']
        results[f'silhouette_{cluster}'] = silhouette_score(data_matrix, labels)
        results[f'davies_bouldin_{cluster}'] = davies_bouldin_score(data_matrix, labels)
        results[f'calinski_harabasz_{cluster}'] = calinski_harabasz_score(data_matrix, labels) 
    return results


def get_metrics(dic_metrics):
    data = pd.DataFrame(list(dic_metrics.items()), columns=['key', 'value'])
    
    data_score_silhouette = data[data['key'].str.contains('silhouette')].copy()
    data_score_davies = data[data['key'].str.contains('davies_bouldin')].copy()
    data_score_calinski = data[data['key'].str.contains('calinski_harabasz')].copy()
    
    data_score_silhouette['cluster'] = data_score_silhouette['key'].apply(lambda string: string.split('_')[2])
    data_score_davies['cluster'] = data_score_davies['key'].apply(lambda string: string.split('_')[3])
    data_score_calinski['cluster'] = data_score_calinski['key'].apply(lambda string: string.split('_')[3])
    
    
    data_score_silhouette = data_score_silhouette[['cluster', 'value']].reset_index(drop=True)
    data_score_davies = data_score_davies[['cluster', 'value']].reset_index(drop=True)
    data_score_calinski = data_score_calinski[['cluster', 'value']].reset_index(drop=True)
    
    return data_score_silhouette, data_score_davies, data_score_calinski


def create_clustering_kprototypes(data):
    data_matrix = data.values
    data_index = data.dtypes.reset_index()
    data_index.columns = ['variable', 'type']
    data_index['type'] = data_index['type'].astype(str)
    categorical_index = list(data_index[data_index['type'] == 'object'].index) 
    
    clustering_info = {}
    for cluster in tqdm(range(2, 12)): 
        clustering = {}
        kprototypes = KPrototypes(
            n_jobs = -1, 
            n_clusters = cluster, 
            init = 'Huang', 
            random_state = 42
        ) 
        clustering['labels'] = kprototypes.fit_predict(data_matrix, categorical = categorical_index)
        clustering['centroids'] = kprototypes.cluster_centroids_   
        clustering['algorithm'] = kprototypes
        clustering_info[f'cluster_{cluster}'] = clustering

    return clustering_info


def create_clustering_kmodes(data):
    data_matrix = data.values
  
    clustering_info = {}
    for cluster in tqdm(range(2, 12)): 
        clustering = {}
        kmodes = KModes(
            n_jobs=-1, 
            n_clusters=cluster,
            init='Cao', 
            random_state=42
        )
        clustering['labels'] = kmodes.fit_predict(data_matrix)
        clustering['centroids'] = kmodes.cluster_centroids_
        clustering['algorithm'] = kmodes
        clustering_info[f'cluster_{cluster}'] = clustering
    return clustering_info