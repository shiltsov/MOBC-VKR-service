import os
import uuid
from django.db import models
from django.utils import timezone

class LogEntry(models.Model):
    created_at = models.DateTimeField(default=timezone.now)
    user_input = models.TextField()
    result_json = models.JSONField()

    def __str__(self):
        return f"{self.created_at.strftime('%Y-%m-%d %H:%M')} — {self.user_input[:30]}..."

class UploadedDatasetFile(models.Model):
    file = models.FileField(upload_to='dataset/')

    def __str__(self):
        return self.file.name

    def delete(self, *args, **kwargs):
        # Сохраняем путь перед удалением объекта
        file_path = self.file.path
        super().delete(*args, **kwargs)
        # Удаляем файл с диска
        if os.path.isfile(file_path):
            os.remove(file_path)

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
