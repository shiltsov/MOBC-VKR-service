import networkx as nx
import matplotlib.pyplot as plt
import json

# каждый признак уникален в контексте объекта, т.е. даже если у двух объектов один и тот же признак ("большой")
# то это 2 разных узла графа
def scene_to_graph(scene: dict) -> nx.DiGraph:
    G = nx.DiGraph()
    
    for obj_dict in scene.get("objects", []):
        for obj, attrs in obj_dict.items():
            G.add_node(obj, type='object')
            for attr in attrs:
                attr_node = f"{obj}_{attr}"
                G.add_node(attr_node, type='attribute')
                G.add_edge(obj, attr_node, relation_type='attribute') 

    return G

def draw_scene_graph(G, figsize=(12, 8)):
    """
    Визуализирует граф сцены с цветовой разметкой:
    - Объекты — синие узлы
    - Признаки — зелёные узлы
    - Связи объект-признак — серые стрелки
    - Пространственные связи между объектами — красные стрелки
    """
    #pos = nx.spring_layout(G, seed=42)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    node_colors = []

    for node in G.nodes(data=True):
        if node[1].get("type") == "object":
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgreen")

    #plt.figure(figsize=figsize)
    plt.figure(figsize=figsize, dpi=120)    

    # Разделяем рёбра по типам
    attr_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation_type") == "attribute"]

    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Рисуем рёбра
    nx.draw_networkx_edges(
        G, pos,
        edgelist=attr_edges,
        edge_color='red',
        arrows=True,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.05',
        min_source_margin=20,
        min_target_margin=20, 
        width=2
    )

    plt.title("Scene Graph")
    plt.axis('off')
    plt.show()

# граф со spacial связями
def scene_to_graph_sp(scene: dict) -> nx.DiGraph:
   
    G = nx.DiGraph()
    for obj_dict in scene.get("objects", []):
        for obj, attrs in obj_dict.items():
            G.add_node(obj, type='object')
            for attr in attrs:
                attr_node = f"{obj}_{attr}"
                G.add_node(attr_node, type='attribute')
                G.add_edge(obj, attr_node, relation_type='attribute') 

    # Добавляем пространственные связи между объектами если они заданы
    for subj, prep, obj in scene.get("relations", []):
        label = prep
        G.add_edge(subj, obj, relation_type="spatial", label=label)


    return G

def draw_scene_graph_sp(G, figsize=(12, 8)):
    """
    Визуализирует граф сцены с цветовой разметкой:
    - Объекты — синие узлы
    - Признаки — зелёные узлы
    - Связи объект-признак — серые стрелки
    - Пространственные связи между объектами — красные стрелки
    """
    #pos = nx.spring_layout(G, seed=42)
    pos = nx.spring_layout(G, seed=42, k=1.5)
    
    node_colors = []

    for node in G.nodes(data=True):
        if node[1].get("type") == "object":
            node_colors.append("skyblue")
        else:
            node_colors.append("lightgreen")

    #plt.figure(figsize=figsize)
    plt.figure(figsize=figsize, dpi=120)    

    # Разделяем рёбра по типам
    attr_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation_type") == "attribute"]
    spatial_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation_type") == "spatial"]
    spatial_labels = {(u, v): d.get("label", "") for u, v, d in G.edges(data=True) if d.get("relation_type") == "spatial"}

    # Рисуем узлы
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=1800)
    nx.draw_networkx_labels(G, pos, font_size=8)

    # Рисуем рёбра пространственных связей
    nx.draw_networkx_edges(
        G, pos,
        edgelist=spatial_edges,
        edge_color='red',
        arrows=True,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.15',
        min_source_margin=20,
        min_target_margin=20,        
        width=2
    )    
    
    # Рёбра объект-признак
    nx.draw_networkx_edges(
        G, pos,
        edgelist=attr_edges,
        edge_color='gray',
        arrows=True,
        arrowstyle='-|>',
        connectionstyle='arc3,rad=0.05',
        min_source_margin=20,
        min_target_margin=20, 
        width=2
    )

    # Подписи для пространственных рёбер
    nx.draw_networkx_edge_labels(G, pos, edge_labels=spatial_labels, font_color='red')

    plt.title("Scene Graph")
    plt.axis('off')
    plt.show()


