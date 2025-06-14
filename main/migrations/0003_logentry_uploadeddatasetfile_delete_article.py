# Generated by Django 4.2.21 on 2025-05-26 11:12

from django.db import migrations, models
import django.utils.timezone


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_article_number'),
    ]

    operations = [
        migrations.CreateModel(
            name='LogEntry',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created_at', models.DateTimeField(default=django.utils.timezone.now)),
                ('user_input', models.TextField()),
                ('result_json', models.JSONField()),
            ],
        ),
        migrations.CreateModel(
            name='UploadedDatasetFile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('file', models.FileField(upload_to='dataset/')),
            ],
        ),
        migrations.DeleteModel(
            name='Article',
        ),
    ]
