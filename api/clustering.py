import hashlib
import time
import json
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from sklearn import metrics, preprocessing
import scipy
from scipy.spatial.distance import cdist
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.metrics import silhouette_samples, silhouette_score, pairwise_distances
from sklearn.feature_extraction.image import grid_to_graph
import matplotlib
import seaborn as sns
from math import ceil

from distutils.version import LooseVersion
from scipy.ndimage.filters import gaussian_filter
from scipy import sparse
import scipy.cluster.hierarchy as sch
from scipy.cluster import hierarchy
from scipy.cluster.hierarchy import dendrogram, linkage, inconsistent
import skimage
from skimage.transform import rescale

matplotlib.use('Agg')

# Methods


def kmeans(args):
    clustersNumber = int(args.get('clustersNumber'))
    datasetMatrix = args.get('datasetMatrix')
    datasetDf = args.get('datasetDf')
    plotingColumns = int(args.get('plotingColumns'))

    model = KMeans(n_clusters=clustersNumber)
    model.fit(datasetMatrix)

    centers = model.cluster_centers_
    centers = json.dumps(centers.tolist())

    # remove some features
    # datasetDf = datasetDf.drop(['Age'],axis=1)
    return {'reutl': 'done'}
    visual, labeledDataset = variablesSprendingPlot(model, datasetDf, plotingColumns)
    
    return {'visual': visual, 'data': {'labeledDataset': labeledDataset.drop(['Constant'],axis=1).to_json(), 'centers': centers}}


def hierarchical(args):
    clustersNumber = int(args.get('clustersNumber'))
    datasetMatrix = args.get('datasetMatrix')
    datasetDf = args.get('datasetDf')
    plotingColumns = int(args.get('plotingColumns'))
    linkageMethod = args.get('linkageMethod')

    # normalization
    if (True):
        scaler = preprocessing.MinMaxScaler()
        datasetMatrix = scaler.fit_transform(datasetDf)

    # D = pairwise_distances(datasetMatrix)
    # print('=========')
    # print(max(map(max, D)))
    # print('=========')

    model = AgglomerativeClustering(n_clusters=clustersNumber, affinity='euclidean', linkage=linkageMethod)
    model.fit_predict(datasetMatrix)

    labels = pd.DataFrame(model.labels_)
    labeledDataset = pd.concat((datasetDf,labels),axis=1).rename({0:'labels'},axis=1)

    linked = linkage(datasetMatrix, linkageMethod)
    # print(linked)
    print(max(map(max, linked)))


    matplotlib.rcParams['lines.linewidth'] = 4

    plt.figure(figsize=(100, 60))
    plt.grid()
    plt.yticks(fontsize=50)
    fancy_dendrogram(linked,
                orientation='top',
                distance_sort='descending',
                truncate_mode='lastp',  # show only the last p merged clusters
                p=clustersNumber,  # show only the last p merged clusters
                color_threshold=0.1*2,
                leaf_rotation=90.,
                leaf_font_size=60.,
                show_contracted=True,
                annotate_above=1,
                max_d=1)

    # avoid circular imports (error)
    from api.helpers import getPltImage

    dendrogram = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'
    dendrogram = '<div class="row">' + dendrogram + '</div>'

    # visual, labeledDataset = variablesSprendingPlot(model, datasetDf, plotingColumns)

    visual = dendrogram

    # return {'visual': visual, 'data': {'labeledDataset': labeledDataset.drop(['Constant'],axis=1).to_json()}}
    return {'visual': visual, 'data': {'labeledDataset': 100}}


def spectral(args):
    clustersNumber = int(args.get('clustersNumber'))
    datasetMatrix = args.get('datasetMatrix')
    datasetDf = args.get('datasetDf')
    plotingColumns = int(args.get('plotingColumns'))

    D = pairwise_distances(datasetMatrix)  # Distance matrix
    print('=========')
    print(D)
    print('=========')
    S = np.max(D) - D  # Similarity matrix
    print('=========')
    print(S)
    print('=========')
    S = sparse.coo_matrix(S)
    print('=========')
    print(S)
    print('=========')

    model = SpectralClustering(n_clusters=clustersNumber, affinity='precomputed')
    model.fit(S)

    centers = model.cluster_centers_
    centers = json.dumps(centers.tolist())
    print(centers)

    visual, labeledDataset = variablesSprendingPlot(model, datasetDf, plotingColumns)

    return {'visual': visual, 'data': {'labeledDataset': labeledDataset.drop(['Constant'],axis=1).to_json()}}

# private 
def variablesSprendingPlot(model, datasetDf, plotingColumns):
    labels = pd.DataFrame(model.labels_)
    labeledDataset = pd.concat((datasetDf,labels),axis=1)
    labeledDataset = labeledDataset.rename({0:'labels'},axis=1)

    labeledDataset['Constant'] = "Data"

    featuresNumber = len(list(labeledDataset))
    plotColumns = plotingColumns
    plotRows = ceil((featuresNumber  - 2) / plotColumns)
    
    f, axes = plt.subplots(plotRows, plotColumns, sharex=False) 
    f.set_size_inches(18, 9 * plotRows)
    f.subplots_adjust(hspace=0.2, wspace=0.7)

    for i in range(0,plotRows):
        for j in range(0, plotColumns):
            featuresNumber -= 1
            if featuresNumber > 1:
                col = labeledDataset.columns[j + (i * plotColumns)]
                
                try:
                    ax = sns.swarmplot(x=labeledDataset['Constant'],y=labeledDataset[col].values,hue=labeledDataset['labels'],ax=axes[i, j])
                    ax.set_title(col)
                except Exception:
                    ax = sns.swarmplot(x=labeledDataset['Constant'],y=labeledDataset[col].values,hue=labeledDataset['labels'],ax=axes[j])
                    ax.set_title(col)

    # avoid circular imports (error)
    from api.helpers import getPltImage

    visual = '<div class="col-sm-12">' + getPltImage(plt) + '</div>'
    visual = '<div class="row">' + visual + '</div>'

    plt.close("all")

    return visual, labeledDataset


def fancy_dendrogram(*args, **kwargs):
    max_d = kwargs.pop('max_d', None)
    if max_d and 'color_threshold' not in kwargs:
        kwargs['color_threshold'] = max_d
    annotate_above = kwargs.pop('annotate_above', 0)

    ddata = dendrogram(*args, **kwargs)

    if not kwargs.get('no_plot', False):
        # plt.title('Hierarchical Clustering Dendrogram (truncated)')
        # plt.xlabel('sample index or (cluster size)')
        # plt.ylabel('distance')
        for i, d, c in zip(ddata['icoord'], ddata['dcoord'], ddata['color_list']):
            x = 0.5 * sum(i[1:3])
            y = d[1]
            if y > annotate_above:
                plt.plot(x, y, 'o', c=c)
                plt.annotate("%.3g" % y, (x, y), xytext=(0, -1),
                             textcoords='offset points',
                             fontsize=50,
                             va='top', ha='center')
        if max_d:
            plt.axhline(y=max_d, linewidth=7, color='#f00')
    return ddata