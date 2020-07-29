from django.db import models
from app.models import Dataset

class Result(models.Model):
    visual = models.TextField()
    data = models.TextField()
    method = models.CharField(max_length=60)
    # foreign keys
    dataset = models.ForeignKey(Dataset, on_delete=models.DO_NOTHING)


