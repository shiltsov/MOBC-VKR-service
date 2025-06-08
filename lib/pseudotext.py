import re

def json_to_pseudo_text(json_data):
    """
    Преобразует JSON-объекты в строку вида:
    'стол 1 (большой) стол 2 () ваза ()'
    """
    objects = []
    for obj_dict in json_data:
        for obj_name, attrs in obj_dict.items():
            attr_text = " ".join(attrs) if attrs else ""
            objects.append(f"{obj_name} ({attr_text})")
    return " ".join(objects)


def pseudo_text_to_json(text):
    """
    Парсит строку вида 'стол 1 (большой) стол 2 () ваза ()' в JSON-формат.
    Корректно обрабатывает многословные имена объектов.
    """
    try:
        result = []
        # регулярка: захватывает имя объекта и строку атрибутов
        pattern = re.finditer(r'(.*?)\s*\(([^()]*)\)', text)
        for match in pattern:
            obj_name = match.group(1).strip()
            attrs_str = match.group(2).strip()
            attrs = [attr for attr in attrs_str.split() if attr]
            result.append({obj_name: attrs})
        return result
    except Exception as e:
        print(f"Ошибка при разборе псевдокода: {e}")
        return []

def text_to_triplets(text):
    """
    Преобразует строку вида:
    "(книга 1, на, стол) (вилка, рядом с, тарелка)"
    в список троек: [("книга 1", "на", "стол"), ...]
    """
    triplets = []
    pattern = re.finditer(r'\((.*?),(.*?),(.*?)\)', text)
    for match in pattern:
        obj1 = match.group(1).strip()
        rel = match.group(2).strip()
        obj2 = match.group(3).strip()
        triplets.append((obj1, rel, obj2))
    return triplets


def triplets_to_text(triplets):
    """
    Преобразует список троек в строку:
    "(книга 1, на, стол) (вилка, рядом с, тарелка)"
    """
    return " ".join(f"({obj1}, {rel}, {obj2})" for obj1, rel, obj2 in triplets)


def complene_scene_from_objects_and_triplets(objects: list, triplets: list, ) -> dict:
    return {
        "scene": {
            "location": "",
            "objects": objects,
            "relations": triplets
        }
    }

def postprocess_text(pred):
    try:
        pred_obj, pred_triplets =  pred.split(":")
        pred_obj_json = pseudo_text_to_json(pred_obj.strip())
        pred_triplets_json = text_to_triplets(pred_triplets.strip())

    except Exception as e:
        #print("Ошибка парсинга:", pred_strs, "|", e)
        pred_scene = []
        label_scene = []
    pred_scene = complene_scene_from_objects_and_triplets(pred_obj_json, pred_triplets_json)

    # выдает батчами - либо распознаную сцену, либо пустышки
    return pred_scene


# проверка
if __name__ == "__main__":
    text = "стол 1 (большой) стол 2 () ваза ()"
    js = pseudo_text_to_json(text)
    text_recuperated = json_to_pseudo_text(js)

    print(js)
    print(text_recuperated)

    text = "стол    (большой) стул  () ваза (зеленая)"
    js = pseudo_text_to_json(text)
    text_recuperated = json_to_pseudo_text(js)

    print(js)
    print(text_recuperated)

    # триплеты
    text = "(книга 1, на, стол) (вилка, рядом с, тарелка)"
    
    triplets = text_to_triplets(text)
    print("Parsed triplets:", triplets)

    text_reconstructed = triplets_to_text(triplets)
    print("Reconstructed text:", text_reconstructed)


