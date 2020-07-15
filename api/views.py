from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import csv
import json
from app.models import Dataset, Record, Attribute
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from api.helpers import buildDatasetMatrix
from api.optimumClustersNumber import switcher
from api.pre_processing import pre_processing
import pandas as pd

# upload & validate dataset


@csrf_exempt
def upload_dataset(request):
    csvfile = request.FILES['dataset']

    # TODO return validation errors from the pre_proccesing class if any
    # link : https://www.kaggle.com/smohubal/market-customer-segmentation-clustering/notebook
    # csvfile = pd.read_csv(csvfile)
    # datasetInstance = pre_processing(csvfile)
    # result = datasetInstance.missing_percent_plot()
    # print(result)
    # return JsonResponse({'result': result})

    decoded_file = csvfile.read().decode('utf-8').splitlines()
    reader = csv.reader(decoded_file, delimiter=',')

    attributes = {}
    attributes = next(reader)

    reader = csv.DictReader(decoded_file, delimiter=',')
    records = []
    for row in reader:
        records.append(row)

    # create the dataset
    dataset = Dataset(title='Dataset_name')
    dataset.save()

    # save the attributes
    for attributeName in attributes:
        attributeInstance = Attribute(name=attributeName, label=attributeName)
        attributeInstance.save()
        dataset.attributes.add(attributeInstance)

    # save the records
    for data in records:
        record = Record(data=data)
        record.save()
        dataset.records.add(record)

    return JsonResponse({'attributes': attributes, 'records_count': len(records)})

# optimum_clusters_number


@csrf_exempt
def optimum_clusters_number(request):
    datasetId = request.POST.get('datasetId')
    method = request.POST.get('method')
    maxIterationsNumeber = request.POST.get('maxIterationsNumeber')

    dataset = Dataset.objects.get(id=datasetId)
    datasetMatrix = buildDatasetMatrix(dataset=dataset)

    result = switcher(method=method, args={
                      'datasetMatrix': datasetMatrix, 'maxIterationsNumeber': maxIterationsNumeber})

    return JsonResponse(result)


def calculate_WSS(points, kmax):
    distortions = []
    K = range(1, 10)
    for k in K:
        kmeanModel = KMeans(n_clusters=k)
        kmeanModel.fit(df)
        distortions.append(kmeanModel.inertia_)
    sse = []
    for k in range(1, kmax+1):
        kmeans = KMeans(n_clusters=k).fit(points)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(points)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(points)):
            curr_center = centroids[pred_clusters[i]]
            curr_sse += (points[i, 0] - curr_center[0]) ** 2 + \
                (points[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
    return sse
