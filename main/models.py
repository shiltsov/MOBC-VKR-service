from django.db import models
import uuid

class Article(models.Model):
    number = models.IntegerField()
    title = models.CharField(max_length=100)
    url = models.CharField(max_length=100)
    summary = models.CharField(max_length=5000)


# ADDED: таблица для ведения тасков.
class TaskStatus(models.Model):
    task_id = models.CharField(max_length=255, unique=True, default=uuid.uuid4, editable=False)
    status = models.CharField(max_length=50, choices=[
        ("pending", "Pending"),
        ("running", "Running"),
        ("completed", "Completed"),
        ("failed", "Failed"),
    ])
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
