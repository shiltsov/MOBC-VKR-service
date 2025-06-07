import os
from celery import Celery

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'extractor.settings')

app = Celery('extractor')
app.config_from_object('django.conf:settings', namespace='CELERY')

app.autodiscover_tasks()
