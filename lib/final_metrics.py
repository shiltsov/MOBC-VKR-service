# все сцены имеют формат
#
#{
#    "scene": {
#        "location": "автосервис",
#        "objects": [
#            {"гаечный ключ": ["металлический"]},
#            {"домкрат": ["металлический", "тяжелый", "прочный"]},
#            {"аккумулятор": []}
#        ],
#        "relations": [
#            ["аккумулятор", "рядом с", "гаечный ключ"],
#            ["гаечный ключ", "на", "домкрат"]
#        ]
#    }
#}

import spacy
import smatch
import networkx as nx

from smatch import compute_f
from typing import List, Dict, Tuple
from collections import Counter
from networkx.algorithms.similarity import optimize_graph_edit_distance
from typing import Dict

nlp = spacy.load("ru_core_news_sm")

def lemmatize_scene(scene: Dict, nlp) -> Dict:
    def lemmatize(text: str) -> str:
        doc = nlp(text)
        return " ".join([token.lemma_ for token in doc])

    new_scene = {
        "scene": {
            "location": scene["scene"].get("location", "неизвестно"),
            "objects": [],
            "relations": []
        }
    }

    # Лемматизируем объекты и их признаки
    for obj in scene["scene"].get("objects", []):
        for obj_name, attributes in obj.items():
            obj_lemma = lemmatize(obj_name)
            attrs_lemma = [lemmatize(attr) for attr in attributes]
            new_scene["scene"]["objects"].append({obj_lemma: attrs_lemma})

    # Лемматизируем связи
    for subj, rel, obj in scene["scene"].get("relations", []):
        subj_lemma = lemmatize(subj)
        rel_lemma = lemmatize(rel)
        obj_lemma = lemmatize(obj)
        new_scene["scene"]["relations"].append([subj_lemma, rel_lemma, obj_lemma])

    return new_scene

def precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def extract_object_attribute_pairs(scene_obj_list):
    """
    scene_obj_list: список словарей вида [{'объект 1': [признаки]}, ...]
    Возвращает множество пар (объект, признак)
    """
    pairs = set()
    for obj_dict in scene_obj_list:
        for obj, attrs in obj_dict.items():
            for attr in attrs:
                pairs.add((obj, attr))
    return pairs


# сравнивает объекты и признаки по куче метрик
def evaluate_obj_attr_metrics(pred, label, normalize = True) -> Dict[str, float]:
    
    # лемматизируем если нужно
    if normalize:
        nlp = spacy.load("ru_core_news_sm")
        pred = lemmatize_scene(pred,nlp)
        label = lemmatize_scene(label,nlp)

    # F1 по объектам
    #print(pred["scene"]["objects"], label["scene"]["objects"])
    
    pred_objects = set([list(descr.keys())[0] for descr in  pred["scene"]["objects"]])
    label_objects = set([list(descr.keys())[0] for descr in  label["scene"]["objects"]])
    
    #print(pred_objects, label_objects)
    # до сих пор ок
    
    tp_obj = len(pred_objects & label_objects)
    fp_obj = len(pred_objects - label_objects)
    fn_obj = len(label_objects - pred_objects)
    _, _, f1_objects = precision_recall_f1(tp_obj, fp_obj, fn_obj)

    # F1 по признакам по каждому объекту (усреднение по объектам, macro)
    f1_per_object = []
    total_attrs = 0
    weighted_sum = 0

    for obj in label_objects | pred_objects:
        # по объекту из пересечения извлекаем атрибуты 
        # ну так себе конечно, можно и переписать но в целом ок, хотя и легаст
        # суть в том что в одном из объекта может не быть - тогда и атрибутов нет
        try:
            label_attrs  = set([obj_dict[obj] for obj_dict in label["scene"]["objects"] if obj in obj_dict.keys()][0])
        except:
            label_attrs = set()
        try:    
            pred_attrs  = set([obj_dict[obj] for obj_dict in pred["scene"]["objects"] if obj in obj_dict.keys()][0])
        except:
            pred_attrs = set()
                    
        tp = len(label_attrs & pred_attrs)
        fp = len(pred_attrs - label_attrs)
        fn = len(label_attrs - pred_attrs)
        _, _, f1 = precision_recall_f1(tp, fp, fn)
        f1_per_object.append(f1)
        weighted_sum += f1 * len(label_attrs)
        total_attrs += len(label_attrs)

    f1_attributes_macro = sum(f1_per_object) / len(f1_per_object) if f1_per_object else 0.0
    f1_attributes_weighted = weighted_sum / total_attrs if total_attrs > 0 else 0.0

    # Глобальный F1 по парам (obj, attr)
    pred_pairs = extract_object_attribute_pairs(pred["scene"]["objects"])
    label_pairs = extract_object_attribute_pairs(label["scene"]["objects"])    

    tp_pairs = len(pred_pairs & label_pairs)
    fp_pairs = len(pred_pairs - label_pairs)
    fn_pairs = len(label_pairs - pred_pairs)
    _, _, f1_global_pairs = precision_recall_f1(tp_pairs, fp_pairs, fn_pairs)

    # Объединённые метрики
    f1_combined_simple = (f1_objects + f1_attributes_macro) / 2
    total_obj = len(label_objects)
    f1_combined_weighted = ((total_obj * f1_objects) + (total_attrs * f1_attributes_weighted)) / (total_obj + total_attrs) if (total_obj + total_attrs) > 0 else 0.0

    return {
        "f1_objects": round(f1_objects, 4),
        "f1_attributes_macro": round(f1_attributes_macro, 4),
        "f1_attributes_weighted": round(f1_attributes_weighted, 4),
        "f1_global_obj_attr_pairs": round(f1_global_pairs, 4),
        "f1_combined_simple": round(f1_combined_simple, 4),
        "f1_combined_weighted": round(f1_combined_weighted, 4)
    }

## метрики для spacial relations когда делаем все за один проход
def evaluate_ordered_triplet_f1(predicted_json, gold_json, normalize = True):
    """
    Строгое сравнение триплетов (obj1, rel, obj2) из описания сцены с учётом порядка.
    
    Parameters:
        predicted: List[Tuple[str, str, str]] — предсказанные триплеты
        gold:      List[Tuple[str, str, str]] — эталонные (истинные) триплеты
        
    Returns:
        precision, recall, f1
    """
    # лемматизируем если нужно
    if normalize:
        nlp = spacy.load("ru_core_news_sm")
        predicted_json = lemmatize_scene(predicted_json, nlp)
        gold_json = lemmatize_scene(gold_json, nlp)

    predicted = predicted_json["scene"]["relations"]
    gold = gold_json["scene"]["relations"]

    predicted_set = set(predicted)
    gold_set = set(gold)

    tp = len(predicted_set & gold_set)
    fp = len(predicted_set - gold_set)
    fn = len(gold_set - predicted_set)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "precision_triplets": precision, 
        "recall_triplets": recall, 
        "f1_triplets": f1
    }


## метрики для spacial relations
## F1-binary - есть связь или нет
## F1-strict - не просто факт но еще и тип
def evaluate_relation_predictions(data):
    """
    Эта метрика только для обучения угадывать связи между объектами
    """

    true_binary = [0 if item["target"] == "нет связи" else 1 for item in data]
    pred_binary = [0 if item["predicted_target"] == "нет связи" else 1 for item in data]

    # Binary F1 вручную чтобы не грузить громоздкий sklearn
    TP = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 1)
    FP = sum(1 for t, p in zip(true_binary, pred_binary) if t == 0 and p == 1)
    FN = sum(1 for t, p in zip(true_binary, pred_binary) if t == 1 and p == 0)

    precision_binary = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    recall_binary = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    f1_binary = (2 * precision_binary * recall_binary) / (precision_binary + recall_binary) if (precision_binary + recall_binary) > 0 else 0.0

    # Strict F1 (точное совпадение метки связи)
    TP_strict = sum(1 for item in data if item["target"] == item["predicted_target"])
    FP_strict = len(data) - TP_strict  # каждая пара — один шанс на TP
    FN_strict = FP_strict  # аналогично

    precision_strict = TP_strict / (TP_strict + FP_strict) if (TP_strict + FP_strict) > 0 else 0.0
    recall_strict = TP_strict / (TP_strict + FN_strict) if (TP_strict + FN_strict) > 0 else 0.0
    f1_strict = (2 * precision_strict * recall_strict) / (precision_strict + recall_strict) if (precision_strict + recall_strict) > 0 else 0.0

    return {
        "F1binary": round(f1_binary, 4),
        "F1strict": round(f1_strict, 4),
    }



def scene_to_graph_with_attrs(scene: Dict) -> nx.DiGraph:
    G = nx.DiGraph()

    # Вершины: объекты и атрибуты
    for obj in scene["scene"]["objects"]:
        for obj_name, attributes in obj.items():
            G.add_node(obj_name, type="object")
            for attr in attributes:
                G.add_node(attr, type="attribute")
                G.add_edge(obj_name, attr, relation=None)

    # Пространственные отношения
    for subj, rel, obj in scene["scene"]["relations"]:
        G.add_edge(subj, obj, relation=rel)

    return G

# Функция сопоставления меток узлов
# узлы совпадают если в вершине одинаковые объекты
def node_match(n1, n2):
    return n1.get("type") == n2.get("type")


# Функция сопоставления меток рёбер
# если у обоих relation=None — считается совпадением (нормально)
# то есть ребра объект-атрибут в обоих гафах считаются совпадающими даже при условии что нет метки
def edge_match(e1, e2):
    return e1.get("relation") == e2.get("relation")


def evaluate_ged_score(scene1: Dict, scene2: Dict, normalize = True) -> float:
    if normalize:
        nlp = spacy.load("ru_core_news_sm")
        scene1 = lemmatize_scene(scene1, nlp)
        scene2 = lemmatize_scene(scene2, nlp)    
    
    G1 = scene_to_graph_with_attrs(scene1)
    G2 = scene_to_graph_with_attrs(scene2)

    
    # Оптимизируем GED с учётом атрибутов
    # optimize_graph_edit_distance использует итеративный эвристический поиск 
    # по возможным сопоставлениям графов, первая выдача как правило оптимальная
    # (но NP сложная задача и на больших графах работает медленно)
    ged_iter = optimize_graph_edit_distance(G1, G2, node_match=node_match, edge_match=edge_match)

    try:
        edit_distance = next(ged_iter)
    except StopIteration:
        return 0.0

    # Нормализация: максимум — если ни одной вершины и ребра не совпало
    max_size = max(G1.number_of_nodes() + G1.number_of_edges(),
                   G2.number_of_nodes() + G2.number_of_edges())

    if max_size == 0:
        return 1.0  # два пустых графа считаем полностью совпадающими

    similarity = 1.0 - (edit_distance / max_size)
    return {
        "GED_score" : round(similarity, 4)
    }

### smatch

## переводит в json более удобный для постраения пенманн графов
# для конвертации в пенман - с этим работать удобнее
def convert_object_dict_to_list(object_list):
    
    result_list = []
    
    for object_dict in object_list:
        for name, attrs in object_dict.items():
            result_list.append( {"name": name, "attributes": attrs})
            
    
    return result_list

def scene_to_penman(scene: dict) -> str:
    """
    Преобразует сцену в валидную PENMAN-строку.
    Объекты → узлы; атрибуты → :has-attribute "значение";
    Пространственные отношения → узлы вида (r / <relation> :from ... :to ...).
    """
    lines = ["(r / root"]
    var_map = {}  # имя объекта → переменная

    # Объекты и их атрибуты
    
    objects = convert_object_dict_to_list(scene["scene"]["objects"])
    for i, obj in enumerate(objects):
        name = obj["name"]
        var = f"o{i}"
        var_map[name] = var
        lines.append(f'    :object ({var} / {name}')
        for attr in obj.get("attributes", []):
            lines.append(f'        :has-attribute "{attr}"')
        lines.append('    )')  # Закрываем объект

    # Пространственные связи
    for j, (subj, rel, obj) in enumerate(scene.get("relations", [])):
        if subj in var_map and obj in var_map:
            rvar = f"r{j}"
            lines.append(f'    :relation ({rvar} / {rel}')
            lines.append(f'        :from {var_map[subj]}')
            lines.append(f'        :to {var_map[obj]}')
            lines.append('    )')

    lines.append(")")  # Закрываем root
    return "\n".join(lines)

def evaluate_scene_graph_with_smatch(scene_gt: dict, scene_pred: dict):
    penman_gt = scene_to_penman(scene_gt)
    penman_pred = scene_to_penman(scene_pred)

    gold_lines = penman_gt.strip().splitlines()
    pred_lines = penman_pred.strip().splitlines()

    print(gold_lines)
    print(pred_lines)

    
    F1, precision, recall = smatch.main(gold_lines, pred_lines)
    print(f"Smatch F1={F1:.3f}  P={precision:.3f}  R={recall:.3f}")    
    
    F1, precision, recall = smatch.match_amr_lines(pred_lines, gold_lines)
    return precision, recall, F1

def calculate_smatch(refs_penman: List[str], preds_penman: List[str]):
    total_match_num = total_test_num = total_gold_num = 0
    n_invalid = 0

    for sentid, (ref_penman, pred_penman) in enumerate(zip(refs_penman, preds_penman), 1):
        best_match_num, test_triple_num, gold_triple_num = smatch.get_amr_match(
            ref_penman, pred_penman, sent_num=sentid
        )

        total_match_num += best_match_num
        total_test_num += test_triple_num
        total_gold_num += gold_triple_num
        # clear the matching triple dictionary for the next AMR pair
        smatch.match_triple_dict.clear()

    score = smatch.compute_f(total_match_num, total_test_num, total_gold_num)

    return {
        "smatch_precision": score[0],
        "smatch_recall": score[1],
        "smatch_fscore": score[2],
        "ratio_invalid_amrs": n_invalid / len(preds_penman) * 100,
    }

def evaluate_smatch(scene_gt: Dict, scene_pred: Dict):
    penman_gt = scene_to_penman(scene_gt)
    penman_pred = scene_to_penman(scene_pred)
    
    return calculate_smatch([penman_gt], [penman_pred])



