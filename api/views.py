from django.http import HttpResponse, JsonResponse
from django.views.decorators.csrf import csrf_exempt
import csv
import json
from app.models import Dataset, Record, Attribute
from api.models import Result, ExecutionLog
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from api.helpers import switcher, buildDatasetMatrix, buildDataFrame
from api.pre_processing import pre_processing
import pandas as pd
from sklearn.decomposition import PCA
import time
from django.template.loader import render_to_string

# upload & validate dataset


@csrf_exempt
def upload_dataset(request):
    start_time = time.time()

    result = {}
    executionStatus = True
    try:
        csvfile = request.FILES['dataset']
        datasetName = request.POST.get('dataset_name')
        datasetDescription = request.POST.get('dataset_description') if request.POST.get(
            'dataset_description') != None else ''
        allowMissinValues = request.POST.get('Allow_missing_values')

        # link : https://www.kaggle.com/smohubal/market-customer-segmentation-clustering/notebook
        try:
            if allowMissinValues == 'false':
                tmpCsvfile = pd.read_csv(csvfile)
                datasetInstance = pre_processing(tmpCsvfile)
                visual, data = datasetInstance.missing_percent()

                executionTime = "{:.4f}".format(time.time() - start_time)
                execLog = ExecutionLog(method='Dataset Uploading', dataset_id=None,
                                       exec_time=executionTime, status=False)
                execLog.save()

                result['data'] = data
                result['visual'] = visual
                result['status'] = False

                return JsonResponse(result)
            else:
                csvfile.seek(0)
        except Exception as e:
            csvfile.seek(0)

        decoded_file = csvfile.read().decode('utf-8').splitlines()
        reader = csv.reader(decoded_file, delimiter=',')

        attributes = {}
        attributes = next(reader)

        reader = csv.DictReader(decoded_file, delimiter=',')
        records = []
        for row in reader:
            records.append(row)

        # create the dataset
        dataset = Dataset(title=datasetName, description=datasetDescription)
        dataset.save()

        # save the attributes
        for attributeName in attributes:
            try:
                attributeInstance = Attribute.objects.get(name=attributeName)
            except Attribute.DoesNotExist:
                attributeInstance = Attribute(
                    name=attributeName, label=attributeName)
                attributeInstance.save()

            dataset.attributes.add(attributeInstance)

        # save the records
        for data in records:
            record = Record(data=data)
            record.save()
            dataset.records.add(record)

        result['attributes'] = attributes
        result['records_count'] = len(records)
    except Exception as e:
        executionStatus = False
        result['message'] = str(e)
        pass

    executionTime = "{:.4f}".format(time.time() - start_time)
    execLog = ExecutionLog(method='Dataset Uploading', dataset_id=(dataset.id if executionStatus else None), error=('' if executionStatus else result['message']),
                           exec_time=executionTime, status=executionStatus)
    execLog.save()

    result['status'] = executionStatus

    return JsonResponse(result)

@csrf_exempt
def clone_dataset(request):
    start_time = time.time()

    result = {}
    executionStatus = True
    try:
        datasetId = request.POST.get('datasetId')
        datasetName = request.POST.get('dataset_name')
        datasetDescription = request.POST.get('dataset_description') if request.POST.get(
            'dataset_description') != None else ''
        allowMissinValues = request.POST.get('Allow_missing_values')


        # create the dataset
        dataset = Dataset(title=datasetName, description=datasetDescription)
        dataset.save()

        # save the attributes
        for attributeName in attributes:
            try:
                attributeInstance = Attribute.objects.get(name=attributeName)
            except Attribute.DoesNotExist:
                attributeInstance = Attribute(
                    name=attributeName, label=attributeName)
                attributeInstance.save()

            dataset.attributes.add(attributeInstance)

        # save the records
        for data in records:
            record = Record(data=data)
            record.save()
            dataset.records.add(record)

        result['attributes'] = attributes
        result['records_count'] = len(records)
    except Exception as e:
        executionStatus = False
        result['message'] = str(e)
        pass

    executionTime = "{:.4f}".format(time.time() - start_time)
    execLog = ExecutionLog(method='Dataset Uploading', dataset_id=(dataset.id if executionStatus else None), error=('' if executionStatus else result['message']),
                           exec_time=executionTime, status=executionStatus)
    execLog.save()

    result['status'] = executionStatus

    return JsonResponse(result)

# optimum_clusters_number


@csrf_exempt
def optimum_clusters_number(request):
    start_time = time.time()

    result = {}
    executionStatus = True
    try:
        datasetId = request.POST.get('datasetId')
        method = request.POST.get('method')
        maxIterationsNumeber = request.POST.get('maxIterationsNumeber')
        pcaComponents = request.POST.get('pcaComponents')

        dataset = Dataset.objects.get(id=datasetId)
        datasetMatrix = buildDatasetMatrix(dataset=dataset)

        if pcaComponents != "" and pcaComponents != 0:
            pca = PCA(n_components=float(pcaComponents) if float(
                pcaComponents) < 1 else int(pcaComponents))
            datasetMatrix = pca.fit_transform(datasetMatrix)

        result = switcher(method=method, args={
            'datasetMatrix': datasetMatrix, 'maxIterationsNumeber': maxIterationsNumeber})
        pass
    except Exception as e:
        executionStatus = False
        result['message'] = str(e)
        pass

    executionTime = "{:.4f}".format(time.time() - start_time)
    execLog = ExecutionLog(method=method, dataset_id=datasetId, error=('' if executionStatus else result['message']),
                           exec_time=executionTime, status=executionStatus)
    execLog.save()

    result['status'] = executionStatus

    return JsonResponse(result)

# clustering


@csrf_exempt
def clustering(request):
    start_time = time.time()

    result = {}
    executionStatus = True
    contributions = False
    try:
        datasetId = request.POST.get('datasetId')
        method = request.POST.get('method')
        clustersNumber = request.POST.get('clustersNumber')
        samplingPoints = request.POST.get('samplingPoints')
        linkageMethod = request.POST.get('linkageMethod')
        pcaComponents = request.POST.get('pcaComponents')

        dataset = Dataset.objects.get(id=datasetId)
        datasetDf = buildDataFrame(dataset=dataset)

        if pcaComponents == '':
            pcaComponents = 0

        nPca = float(pcaComponents) if float(
            pcaComponents) < 1 else int(pcaComponents)

        if pcaComponents != "" and pcaComponents != 0 and (nPca > 2 or nPca < 1):
            print('========= PCA Applied =========')
            pca = PCA(n_components=float(pcaComponents) if float(
                pcaComponents) < 1 else int(pcaComponents))
            datasetDf = pca.fit_transform(datasetDf)
            contributions = pca.components_

            columns = ['PC'+str(i) for i in range(datasetDf.shape[1])]
            datasetDf = pd.DataFrame(datasetDf, columns=columns)
            clustersNumber = datasetDf.shape[1]

        args = {
            'datasetDf': datasetDf,
            'clustersNumber': clustersNumber,
            'samplingPoints': samplingPoints,
            'linkageMethod': linkageMethod
        }

        result = switcher(method=method, args=args)

        # used for saving later on
        result['method'] = method
        result['datasetId'] = datasetId

        if contributions is not False:
            result['contributions'] = json.dumps(contributions.tolist())

        pass
    except Exception as e:
        executionStatus = False
        result['message'] = str(e)
        pass

    executionTime = "{:.4f}".format(time.time() - start_time)
    execLog = ExecutionLog(method=method, dataset_id=datasetId, error=('' if executionStatus else result['message']),
                           exec_time=executionTime, status=executionStatus)
    execLog.save()

    result['status'] = executionStatus

    return JsonResponse(result)


@csrf_exempt
def save_result(request):
    datasetId = int(request.POST.get('datasetId'))
    method = request.POST.get('method')
    indexes = request.POST.get('indexes')
    pca3d = request.POST.get('pca3d')
    parallelCoord = request.POST.get('parallelCoord')
    parallelCentroids = request.POST.get('parallelCentroids')

    try:
        result = Result(
            dataset_id=datasetId,
            method=method,
            indexes=str(indexes),
            matplt_3d=pca3d,
            parallel_coord=parallelCoord,
            parallel_centroids=parallelCentroids,
            tsne_3d=render_to_string('3d_tsne.html'),
            pca_3d=render_to_string('3d_pca.html'),
        )
        result.save()
        return JsonResponse({'status': True})
    except Exception as e:
        return JsonResponse({'status': False, 'message': str(e)}, status=400)
