FROM python:3.10-slim

RUN apt update && rm -rf /var/lib/apt/lists
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000

CMD sh -c "gunicorn --bind 0.0.0.0:8000 review2.wsgi:application"
