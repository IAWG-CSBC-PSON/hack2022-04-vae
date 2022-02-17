# hack2022-04-vae
Challenge 4: Variational Autoencoder for single cell image feature extraction

Base ME-VAE Architecture Implementation (reference: https://github.com/GelatinFrogs/ME-VAE_Architecture): This is a base implementation of the Multi-Encoder Variational Autoencoder (ME-VAE). Uninformative features, especially transformational image features, present a problem for feature extraction in single cell images. It can be hard to disentangle these uninformative features from the output of a conventional VAE. This architecture allows the analyst to remove several uninformative/undesired features simultaneously to improve downstream analysis. This implementation is published and citable in the BioRxiv here: https://www.biorxiv.org/content/10.1101/2021.04.22.441005v2

The two image datasets for this challenge are as follows:

syn26850670 - a minimal example set for method development and benchmarking
syn26850728 - training dataset (syn26850670) label (ground truth) for scoring
syn26850727 - a large-scale test image set for scoring

We provide a small python module for clustering and scoring for the trainings dataset. The results from the test dataset will be scored via the same scoring function. 
The required libraries for the functions are:
- scikit-learn
- pandas

You can install them via pip:
```
pip install scikit-learn pandas
```

The two functions are:
```
def clustering(X, n_clusters):
    """
    K-Means Clustering
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X)
    return kmeans.labels_
```
and
```
def score(y_pred, csv_path='Trainingdata_label.csv'):
    """
    Normalized Mutual Information Score
    """
    y_true = pd.read_csv(path)['Ligand_Label']
    return sklearn.metrics.normalized_mutual_info_score(y_true, y_pred)
```
 You can import them via 
```
from iawghackathon.utils import clustering, score
```

Example:
```
import numpy as np
from iawghackathon.utils import clustering, score

x = np.random.random((15898, 2))
y = clustering(x , 2)
nmi = score(y)
print(nmi)
```
```
0.00011703560856498105
```
