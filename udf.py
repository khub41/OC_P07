import time

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler
import pandas as pd

from contextlib import contextmanager

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - done in {:.2f}s".format(title, time.time() - t0))


def scale_data(raw_features):
    scaler = StandardScaler()

    data_scale = pd.DataFrame(scaler.fit_transform(raw_features),
                              index=raw_features.index,
                              columns=raw_features.columns)

    return data_scale, scaler


def scree_plot(data_scale, max_comp, savefig=False):
    pca_scree = PCA(n_components=max_comp)
    pca_scree.fit(data_scale)
    plt.style.use('fivethirtyeight')
    plt.xlabel('Nb de Composantes Principales')
    plt.ylabel('Pourcentage de variance expliquée (cumulée)')
    plt.title('Scree plot PCA')
    plt.plot(np.arange(1, pca_scree.n_components_ + 1), pca_scree.explained_variance_ratio_.cumsum() * 100,
             color='#8c704d')
    if savefig:
     plt.savefig('plots/{}.png'.format(savefig), bbox_inches='tight', dpi=720)
    plt.show()


def reduce_dim_pca(data_scale, n_comp):
    pca = PCA(n_components=n_comp)
    data_scale_decomp = pca.fit_transform(data_scale)
    return data_scale_decomp, pca


def tuning_kmeans(data, range_n_clust, experiment, show=True, savefig=False):

    slh_scores = {}
    db_scores = {}
    mlflow.set_experiment(experiment)
    for n_clust in range_n_clust:
        with mlflow.start_run(run_name=f"tuning ({n_clust} clusters)"):
            start = time.time()
            model_k_means = KMeans(n_clusters=n_clust)
            print(f"{model_k_means} start training")
            labels = model_k_means.fit_predict(data)
            slh = silhouette_score(data, labels)
            db = davies_bouldin_score(data, labels)
            elapsed = time.time() - start
            mlflow.log_param("n_clusters", n_clust)
            mlflow.log_metric('train time', elapsed)
            mlflow.log_metric("silh score", slh)
            mlflow.log_metric("DB score", db)
            print(f"{model_k_means} trained in {elapsed}s")
            print(f"Slh score : {slh}\nDB score : {db}")
            slh_scores[n_clust] = slh
            db_scores[n_clust] = db
            mlflow.end_run()