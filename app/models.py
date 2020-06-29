from django.db import models
from django.contrib.auth.models import User

class Attribute(models.Model):
    name = models.CharField(max_length=200, unique=True)
    label = models.CharField(max_length=200)
    description = models.TextField()
    attribute_type = models.CharField(max_length=30)

class Record(models.Model):
    data = models.TextField()

class Dataset(models.Model):
    title = models.CharField(max_length=200)
    description = models.CharField(max_length=200)
    # foreign keys
    parent = models.ForeignKey('Dataset', on_delete=models.DO_NOTHING, null=True)
    attributes = models.ManyToManyField(Attribute)
    records = models.ManyToManyField(Record)



