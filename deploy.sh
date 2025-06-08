#!/bin/bash

# 1. Запускаем Minikube
minikube start --cpus=6 --memory=16g --driver=docker
minikube addons enable ingress

# 2. Используем внутренний Docker-демон Minikube
eval $(minikube docker-env)

# 3. Собираем Docker-образ

docker build -t django-app:latest .

# 4. Применяем манифесты Kubernetes
kubectl apply -f configmap.yaml
kubectl apply -f secret.yaml

kubectl apply -f static-pv-pvc.yaml
kubectl apply -f model-pv-pvc.yaml


kubectl apply -f postgres-statefulset.yaml
kubectl apply -f postgres-service.yaml

kubectl apply -f django-collectstatic-job.yaml
kubectl apply -f django-migrate-job.yaml
kubectl apply -f createsuperuser-job.yaml
kubectl apply -f model-loader-job.yaml

kubectl apply -f django-deployment.yaml
kubectl apply -f static-deployment.yaml

kubectl apply -f celery-deployment.yaml
kubectl apply -f rabbitmq-deployment.yaml

# содержимое конфига через configmap передадим
kubectl create configmap nginx-proxy-conf --from-file=nginx.conf=nginx.conf.proxy
kubectl apply -f proxy-deployment.yaml

# экспортеры для прометеуса
kubectl apply -f rabbitmq-exporter.yaml
kubectl apply -f postgres-exporter.yaml 

# Джанго не требует отдельного контейнера, встраивается в джанго
kubectl apply -f prometheus.yaml
kubectl apply -f grafana.yaml

minikube service grafana --url
minikube service prometheus --url

# прописыаем в хостc
echo "$(minikube ip) django.local" | sudo tee -a /etc/hosts
echo "$(minikube ip) rabbitmq.local" | sudo tee -a /etc/hosts

kubectl apply -f ingress.yaml

# 5. Выводим URL для доступа к Django
echo "Django доступен по адресу: $(minikube service django --url)"
echo "Кроме того по http://django.local"

echo "Django доступен по адресу: $(minikube service rabbitmq --url)"
echo "Кроме того по http://rabbitmq.local"


