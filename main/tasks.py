import os
from heapq import heappush, heappop
import pickle
import requests
import wikipedia
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import pandas as pd
import bs4

from celery import shared_task
from main.models import Article, TaskStatus

@shared_task
def train_model(task_id):
    """обучение модели"""
    task = TaskStatus.objects.get(task_id=task_id)
    task.status = "Running"
    task.save()

    max_articles_train = int(os.environ.get('num_articles', 10000))

    Article.objects.all().delete()
    try:
        data = pd.read_csv('wiki_movie_plots_deduped.csv').sample(max_articles_train)
    except Exception as e:
        task = TaskStatus.objects.get(task_id=task_id)
        task.status = "Failed"
        task.save()

        return f"Model training FAILED for task {task_id}"

    text_corpus = list(data.Plot)

    articles = [Article(number=i, title=data.iloc[i].Title[:100], url=data.iloc[i]['Wiki Page'][:100], summary=data.iloc[i].Plot[:4000])
                for i in range(data.shape[0])]

    Article.objects.bulk_create(articles)

    model = TfidfVectorizer(analyzer='word', stop_words='english', strip_accents='ascii')
    param_matrix = model.fit_transform(text_corpus)

    if os.path.exists("model/model.pickle"):
        os.remove("model/model.pickle")

    if os.path.exists("model/data.npz"):
        os.remove("model/data.npz")

    with open('model/model.pickle', 'wb') as f:
        pickle.dump(model, f)
    scipy.sparse.save_npz('model/data.npz', param_matrix)

    task.status = "Completed"
    task.save()
    return f"Model training completed for task {task_id}"

