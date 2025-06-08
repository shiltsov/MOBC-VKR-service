import transformers
import datasets
import huggingface_hub
import torch
import json
import torch
import os
import sys
import warnings
import random
import io
import base64

import numpy as np

from pathlib import Path
from datasets import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, TrainingArguments, Trainer
from transformers import TrainerCallback

from peft import LoraConfig, get_peft_model, TaskType, PeftConfig,PeftModel
from tqdm import tqdm

import matplotlib.pyplot as plt

# логируем обращения
from .models import LogEntry


# отключаем их все чтобы картинку не портили
warnings.filterwarnings("ignore", category=FutureWarning)

MODEL_NAME = "sberbank-ai/ruT5-base"

INPUT_SEQ_LENGTH = 1500
OUTPUT_SEQ_LENGTH = 512
NUM_BEAMS = 6

lib_path = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(lib_path)


# перевод в псевдотекст и обратно
from lib.pseudotext import postprocess_text
from lib.graph_vizualization import scene_to_graph_sp, draw_scene_graph_sp


MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'model'))
config = PeftConfig.from_pretrained(MODEL_DIR)

DEVICE = "cpu" # у нас будет хост на CPU

base_model = T5ForConditionalGeneration.from_pretrained(config.base_model_name_or_path)
tokenizer = T5Tokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(base_model, MODEL_DIR)
model = model.to(DEVICE)
model.eval()

from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.http import HttpResponseBadRequest




PROMPT = """
Ты должен проанализировать описание сцены и вернуть ответ в специальном псевдоформате.

Твоя задача:
- Найди все объекты, упомянутые в описании, и их признаки.
- Найди все пространственные связи между объектами
- Верни результат строго в псевдоформате одной новой строкой.

Формат вывода объектов:
объект1 (признак1 признак2) объект2 () объект3 (признак) : (объект1, связь, объект2) (объект1, связь, объект3)


Требования:
- Каждый объект в выводе списка объектов указывается один раз.
- В названии объекта может быть число (например "ваза 1") - тогда нужно указывать в таком же виде.
- Признаки пишутся через пробел внутри круглых скобок.
- Если признаки отсутствуют, используй пустые скобки ().
- Не добавляй объектов или признаков, которых нет в описании.
- Вывод объектов и вывод связей должныразделяться символом :
- В ответе не должно быть никаких пояснений, комментариев или заголовков — только две строки с результатом.

Примеры:

Описание: Маленький красный стол стоит у окна.
Ответ:
стол (маленький красный) окно () : (стол, у, окна)

Описание: Синяя ваза 1 стоит рядом с вазой 2.
Ответ:
ваза 1 (синяя) ваза 2 () : (ваза 1, рядом с, ваза 2)

Описание: {description}
Ответ:
"""



# Генерация
def predict(description, max_length=OUTPUT_SEQ_LENGTH):
    prompt = PROMPT.format(description=description)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=INPUT_SEQ_LENGTH
    ).to(DEVICE)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            num_beams=NUM_BEAMS, # попробовать меньше
            #temperature=TEMPERATURE, # параметризовать
            early_stopping=True
        )

    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    #print(output_text)
    try:
        parsed_json = postprocess_text(output_text)
    except Exception as e:
        print(f"Ошибка парсинга JSON: {e}")
        print("Сырые данные:", output_text)
        parsed_json = None

    LogEntry.objects.create(
        user_input=description,
        result_json=parsed_json
    )

    return output_text, parsed_json


def index(request):
    return render(request, 'main/index.html')

@csrf_exempt
def get_predict(request):
    """Обработка POST-запроса и вывод результата"""

    text = request.POST.get("description", "")
    if len(text) == 0:
        return render(request, "main/error.html", {"message": "пустой запрос"})
    

    # Предсказание
    output_text, result = predict(text)
    json_text = json.dumps(result, indent=2, ensure_ascii=False)

    # Рисуем граф и кодируем его в base64
    G = scene_to_graph_sp(result.get("scene", {}))
    buf = io.BytesIO()
    plt.figure(figsize=(8, 6))
    draw_scene_graph_sp(G)
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)
    image_base64 = base64.b64encode(buf.read()).decode("utf-8")

    # Возвращаем HTML с текстом и картинкой

    context = {
        "description": text,
        "output_text": output_text,
        "json_text": json_text,
        "image_base64": image_base64
    }
    return render(request, "main/predict.html", context)








