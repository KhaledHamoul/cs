import io
import base64
from app.templatetags.template_filters import get_item
from api.optimumClustersNumber import elbow, silhouette, gapStatistic
from api.clustering import kmeans, hierarchical, spectral
import pandas as pd

# get image element from matplotlib plt
def getPltImage(plt):
    f = io.BytesIO()
    plt.savefig(f, format="png", facecolor=(0.95, 0.95, 0.95))
    encoded_img = base64.b64encode(f.getvalue()).decode('utf-8').replace('\n', '')
    f.close()
    return '<img id="result-visual" src="data:image/png;base64,%s" />' % encoded_img

# build dataset matrix from dataset records
def buildDatasetMatrix(dataset):
    datasetMatrix = []
    for record in dataset.records.all():
        temp = []
        for attribute in dataset.attributes.all():
            temp.append(ord(get_item(record.data, attribute.name)[0]) if type(get_item(record.data, attribute.name)) == str else get_item(record.data, attribute.name))

        datasetMatrix.append(temp)

    return datasetMatrix

def buildDataFrame(dataset):
    dataDict = {}
    index = 0
    for record in dataset.records.all():
        temp = {}
        for attribute in dataset.attributes.all():
            temp[attribute.name] = ord(get_item(record.data, attribute.name)[0] if type(get_item(record.data, attribute.name)) == str else get_item(record.data, attribute.name))
        
        dataDict[index] = temp
        index += 1

    datasetDf = pd.DataFrame.from_dict(dataDict, orient='index')

    return datasetDf

# methods switcher
def switcher(method, args):
    # Optimum clusters number algorithms
    if method == 'elbow':
        return elbow(args)

    if method == 'silhouette':
        return silhouette(args)

    if method == 'gap_statistic':
        return gapStatistic(args)

    # clustering algorithms
    if method == 'kmeans':
        return kmeans(args)

    if method == 'hierarchical':
        return hierarchical(args)
    
    if method == 'spectral':
        return spectral(args) 
    
    else:
        return "Invalid method"

