from django.db import models
from app.models import Dataset

class Result(models.Model):
    method = models.CharField(max_length=60)
    indexes = models.TextField(default="")
    
    matplt_3d = models.TextField(default="")
    parallel_coord = models.TextField(default="")
    parallel_centroids = models.TextField(default="")
    
    pca_3d = models.TextField(default="")
    tsne_3d = models.TextField(default="")
    
    # foreign keys
    dataset = models.ForeignKey(Dataset, on_delete=models.DO_NOTHING)

class ExecutionLog(models.Model):
    method = models.CharField(max_length=60)
    exec_time = models.TextField()
    status = models.BooleanField()
    created_at = models.DateTimeField(auto_now_add=True)
    # foreign keys
    dataset = models.ForeignKey(Dataset, on_delete=models.CASCADE, null=True)
    result = models.ForeignKey(Result, on_delete=models.DO_NOTHING, null=True)


