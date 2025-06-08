# импорты джанго
import os
from django.contrib import admin, messages
from .models import LogEntry, UploadedDatasetFile
from django.utils.html import format_html
from django.urls import reverse, path
from django.http import HttpResponseRedirect
from django.conf import settings


# импорты для тренировки модели
import json
import torch
import os
import sys
import warnings
import random

import numpy as np

from pathlib import Path
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from transformers import TrainerCallback

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig,PeftModel

# либы для валидации
#lib_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
#sys.path.append(lib_path)

#from lib.final_metrics import evaluate_obj_attr_metrics
from lib.pseudotext import json_to_pseudo_text, pseudo_text_to_json, text_to_triplets, triplets_to_text, postprocess_text

from celery import shared_task
from main.models import TaskStatus


MODEL_NAME = "sberbank-ai/ruT5-base"
DEVICE = "cpu" # у нас будет хост на CPU

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
DATA_DIR = Path(os.path.join(settings.MEDIA_ROOT, 'dataset'))

per_device_train_batch_size = 1

lora_rank = 8
lora_alpha = 16
lora_target_modules=["q", "v"]   # в какие слои делаем инъекции
lora_dropout=0.1

start_learning_rate = 5e-4 # стартовый
lr_scheduler_type="cosine"
warmup_steps=10 # прогрев (меньше чем эпоха в нашем случае)

INPUT_SEQ_LENGTH = 1700
OUTPUT_SEQ_LENGTH = 512

# параметры генерации
NUM_BEAMS = 6

PROMPT = """
Ты должен проанализировать описание сцены и вернуть ответ в специальном псевдоформате.

Твоя задача:
- Найди все объекты, упомянутые в описании, и их признаки.
- Верни результат строго в псевдоформате вывода объектов — одной строкой.
- Найди все пространственные связи между объектами
- Верни результат строго в псевдоформате вывода связей — одной новой строкой.

Формат вывода объектов:
объект1 (признак1 признак2) объект2 () объект3 (признак)

Формат вывода связей:
(объект1, связь, объект2) (объект1, связь, объект3)


Требования:
- Каждый объект в выводе списка объектов указывается один раз.
- В названии объекта может быть число (например "ваза 1") - тогда нужно указывать в таком же виде.
- Признаки пишутся через пробел внутри круглых скобок.
- Если признаки отсутствуют, используй пустые скобки ().
- Не добавляй объектов или признаков, которых нет в описании.
- Вывод объектов и вывод связей должны быть на двух разные строках
- В ответе не должно быть никаких пояснений, комментариев или заголовков — только две строки с результатом.

Примеры:

Описание: Маленький красный стол стоит у окна.
Ответ:
стол (маленький красный) окно ()
(стол, у, окна)

Описание: Синяя ваза 1 стоит рядом с вазой 2.
Ответ:
ваза 1 (синяя) ваза 2 ()
(ваза 1, рядом с, ваза 2)

Описание: {description}
Ответ:
"""



@shared_task
def train_model(task_id, epochs):
    """
    обучение модели
    """


    """пихаем в статус что идет обучение модели"""
    task = TaskStatus.objects.get(task_id=task_id)
    task.status = "Running"
    task.save()


    num_train_epochs = epochs
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    def preprocess(example):
        inputs = tokenizer(example["input"], padding="max_length", truncation=True, max_length=INPUT_SEQ_LENGTH)
        targets = tokenizer(example["target"], padding="max_length", truncation=True, max_length=OUTPUT_SEQ_LENGTH)
        inputs["labels"] = targets["input_ids"]

        # сохраняем оригинал обратно в пример
        # inputs["target_raw"] = example["target_raw"]
        return inputs


    def make_target(scene_objects, scene_triplets):
        objects_dict = {}
        for obj in scene_objects:
            for name, attrs in obj.items():
                objects_dict[name] = attrs
        ps_text = json_to_pseudo_text([objects_dict]) + "\n" + triplets_to_text(scene_triplets)
        print(ps_text)
        return ps_text


    print(os.path.join(settings.MEDIA_ROOT, 'dataset'))
    print(f"train {epochs} {DATA_DIR}")


    data = []
    for path in sorted(DATA_DIR.glob("*.jsonl")):
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                print("\n" + line + "\n")
                try:
                    item = json.loads(line)
                    description = item["description"]
                    target = make_target(item["scene"]["objects"], item["scene"]["relations"])
                    data.append({
                        "input": PROMPT.format(description=description),
                        "target": target,
                    })
                except:
                    pass

    # Делаем датасет
    dataset = Dataset.from_list(data).shuffle(seed=42) # у нас разные по типу бачи - лучше перемешать
    #dataset = dataset.train_test_split(test_size=0.02, seed=42)
    #train_ds, val_ds = dataset["train"], dataset["test"]
    train_ds = dataset

    train_ds = train_ds.map(preprocess, batched=False)
    #val_ds = val_ds.map(preprocess, batched=False)

    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

    lora_config = LoraConfig(
        r=lora_rank, # ранг низкоранговой матрицы
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        # target_modules=["q", "k", "v", "o"]
        lora_dropout=lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(model, lora_config)
    model.to(DEVICE)

    # Обучение
    training_args = TrainingArguments(
        output_dir = MODEL_DIR,
        per_device_train_batch_size=per_device_train_batch_size,
        #per_device_eval_batch_size=4,
        num_train_epochs=num_train_epochs,
        logging_dir="../logs",
        logging_steps=1,
        learning_rate=start_learning_rate,
        lr_scheduler_type=lr_scheduler_type,
        warmup_steps=warmup_steps,
        #eval_strategy="epoch",
        #eval_accumulation_steps=10, # для маленькой памяти GPU - ск бачей одновременно грузить
        save_strategy="epoch",
        save_total_limit=5,
        #load_best_model_at_end=True,
        #report_to="wandb",
        fp16=True,
        no_cuda=True,
    )


    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        #eval_dataset=val_ds,
        tokenizer=tokenizer,
        #compute_metrics=compute_metrics,
    )

    # грузим уже обученную
    #trainer.train(resume_from_checkpoint=True)

    trainer.train()
    model.save_pretrained(MODEL_DIR)

    task.status = "Completed"
    task.save()

    return f"Model training completed for task {task_id}" 

