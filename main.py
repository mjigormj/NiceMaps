import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import osmnx as ox
import networkx as nx
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import numpy as np
import random
import re
from cooordenadas import Coordenadas
import utils

# Função de treinamento do modelo de tráfego
def train_traffic_model():
    
    # Largura da rua em metros, Velocidade média em km/h, Densidade de tráfego
    features = [
        [10, 90, 10],    
        [8, 60, 15],   
        [5, 30, 20]     
    ]
     
    labels = ['livre', 'moderado', 'pesado']

    # Pré-processamento
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)
    
    # Divisão dos dados em conjuntos de treinamento e teste
    features_train, features_test, labels_train, labels_test = train_test_split(
        features, labels_encoded, test_size=0.2, random_state=42
    )

    # Treinar um modelo (Random Forest, por exemplo)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(features_train, labels_train)

    # Avaliar a precisão do modelo nos dados de teste
    predictions_test = model.predict(features_test)
    accuracy = accuracy_score(labels_test, predictions_test)
    
    return model, label_encoder
    

def predict_traffic_conditions(model, label_encoder, features):
    # Fazer previsões usando o modelo treinado
    predictions = model.predict(features)
    predicted_classes = label_encoder.inverse_transform(predictions)
    return predicted_classes

    
#def calculate_route():
    chosen_edge_features = np.random.randint(5, 90, size=(1, 3)) 
    chosen_edge = edges[combo.get()]
    chosen_node = chosen_edge[0]
    chosen_route = nx.shortest_path(G, chosen_node, utils.get_nearest_node(G, coordenadasFixas[1]))
    yellow = nx.shortest_path(G, utils.get_nearest_node(G, coordenadasFixas[1]), utils.get_nearest_node(G, coordenadasFixas[1]))

    # Verificação de rota válida
    is_route_valid = True
    for node1, node2 in zip(chosen_route[:-1], chosen_route[1:]):
        edge_str = str((node1, node2))
        if clusters.get(edge_str, -1) == -1:
            is_route_valid = False
            break

    if not is_route_valid:
        # Rota alternativa para evitar tráfego pesado
        alternative_route = nx.shortest_path(G, chosen_node, utils.get_nearest_node(G, coordenadasFixas[1]))
        is_alternative_valid = True
        for node1, node2 in zip(alternative_route[:-1], alternative_route[1:]):
            edge_str = str((node1, node2))
            if clusters.get(edge_str, -1) != -1:
                is_alternative_valid = False
                break

        if is_alternative_valid:
            chosen_route = alternative_route

    # Lógica de cores para tráfego pesado
    predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, np.array(chosen_edge_features).reshape(1, -1))

    # Lógica de cores para tráfego moderado
    if predicted_traffic[0] == 'moderado':
        edge_colors = 'Yellow'
    # Lógica de cores para tráfego livre
    elif predicted_traffic[0] == 'livre':
        edge_colors = 'Green'
    # Lógica de cores para tráfego pesado
    elif predicted_traffic[0] == 'pesado':
        edge_colors = 'Red'
    
    print("Chosen Edge Features:", chosen_edge_features)
    print("Predicted Traffic:", predicted_traffic)
    print("Edge Colors:", edge_colors)
    
    # Plotar cada rota separadamente
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Defina a cor de fundo
    ax.set_facecolor('Black')

    # Adicione a plotagem das ruas
    ox.plot_graph(G, ax=ax, node_size=0, edge_color='White', edge_linewidth=0.5, show=False, close=False)
    
    ox.plot_graph_route(G, chosen_route, ax=ax, node_size=0, route_color=edge_colors, route_linewidth=2, show=False, close=False)

    buildings.plot(ax=ax, color="Gray", alpha=0.5)
    green_areas.plot(ax=ax, color='LightSeaGreen', alpha=0.5)
    
    plt.show()

def show_all_edges():
    # Iterar sobre todas as arestas do grafo e mostrar individualmente
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('Black')

    ox.plot_graph(G, ax=ax, node_size=0, edge_color='White', edge_linewidth=0.5, show=False, close=False)
    buildings.plot(ax=ax, color="Gray", alpha=0.5)
    green_areas.plot(ax=ax, color='LightSeaGreen', alpha=0.5)

    for edge in G.edges():
        edge_str = str(edge)
        edge_features = np.random.randint(5, 90, size=(1, 3))  # Substitua por seus próprios dados
        predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, edge_features.reshape(1, -1))

        if predicted_traffic[0] == 'moderado':
            edge_color = 'Yellow'
        elif predicted_traffic[0] == 'livre':
            edge_color = 'Green'
        elif predicted_traffic[0] == 'pesado':
            edge_color = 'Red'
        else:
            edge_color = 'Purple'  # Adicione uma cor padrão para outras condições

        ox.plot_graph_route(G, [edge[0], edge[1]], ax=ax, node_size=0, route_color=edge_color, route_linewidth=2, show=False, close=False)

    plt.show()

def calculate_route():
    chosen_edge_features = np.random.randint(5, 90, size=(1, 3)) 
    chosen_edge = edges[combo.get()]
    chosen_node = chosen_edge[0]
    chosen_route = nx.shortest_path(G, chosen_node, utils.get_nearest_node(G, coordenadasFixas[1]))

    # Verificação de rota válida
    is_route_valid = True
    for node1, node2 in zip(chosen_route[:-1], chosen_route[1:]):
        edge_str = str((node1, node2))
        if clusters.get(edge_str, -1) == -1:
            is_route_valid = False
            break

    if not is_route_valid:
        # Rota alternativa para evitar tráfego pesado
        alternative_route = nx.shortest_path(G, chosen_node, utils.get_nearest_node(G, coordenadasFixas[1]))
        is_alternative_valid = True
        for node1, node2 in zip(alternative_route[:-1], alternative_route[1:]):
            edge_str = str((node1, node2))
            if clusters.get(edge_str, -1) != -1:
                is_alternative_valid = False
                break

        if is_alternative_valid:
            chosen_route = alternative_route

    # Lógica de cores para tráfego pesado
    predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, np.array(chosen_edge_features).reshape(1, -1))

    # Iterar sobre cada trecho da rota e plotar individualmente
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('Black')

    ox.plot_graph(G, ax=ax, node_size=0, edge_color='White', edge_linewidth=0.5, show=False, close=False)
    buildings.plot(ax=ax, color="Gray", alpha=0.5)
    green_areas.plot(ax=ax, color='LightSeaGreen', alpha=0.5)

    for node1, node2 in zip(chosen_route[:-1], chosen_route[1:]):
        edge_str = str((node1, node2))
        edge_features = np.random.randint(5, 90, size=(1, 3))  
        predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, edge_features.reshape(1, -1))

        if predicted_traffic[0] == 'moderado':
            edge_color = 'Yellow'
        elif predicted_traffic[0] == 'livre':
            edge_color = 'Green'
        elif predicted_traffic[0] == 'pesado':
            edge_color = 'Red'
        else:
            edge_color = 'purple'

        ox.plot_graph_route(G, [node1, node2], ax=ax, node_size=0, route_color=edge_color, route_linewidth=2, show=False, close=False)

    plt.show()

# Dados iniciais
coordenadasFixas = [
    Coordenadas("Casa do Ator", -23.591563, -46.682362, ""),
    Coordenadas("anhembi", -23.599899, -46.676735, ""),
]

G = utils.create_graph_from_point(coordenadasFixas[0])
green_areas = utils.get_features(coordenadasFixas[0], {'leisure': 'park'})
buildings = utils.get_features(coordenadasFixas[0], {'building': True})

edges_raw = list(G.edges.data('name'))
clean_edges = {}

for i, edge in enumerate(edges_raw):
    clean_edge_name = re.sub(r'[\d(),\'\"]', '', str(edge[2]))
    clean_edges[f"{i}: {clean_edge_name}"] = edge

edges = dict(sorted(clean_edges.items(), key=lambda item: item[1]))

traffic_data = {str(edge): random.randint(0, 10000) for edge in edges.values()}
X = np.array(list(traffic_data.values())).reshape(-1, 1)
db = DBSCAN(eps=5, min_samples=5).fit(X)
labels = db.labels_
clusters = {str(edge): label for edge, label in zip(traffic_data.keys(), labels)}

# Treinar o modelo de tráfego
traffic_model, label_encoder = train_traffic_model()

# Interface Gráfica
root = tk.Tk()
root.title("Escolha uma rua")

frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

# Adicione uma label para exibir a precisão
#accuracy_label = ttk.Label(frame, text="Accuracy on test set: N/A")
#accuracy_label.grid(column=1, row=2, columnspan=3)

ttk.Label(frame, text="Escolha uma rua:").grid(column=1, row=1, sticky=tk.W)

combo = ttk.Combobox(frame, width=40, values=list(edges.keys()))
combo.grid(column=2, row=1)
combo.current(0)

ttk.Button(frame, text="Calcular rota", command=calculate_route).grid(column=3, row=1)

# Adicione um botão para mostrar todas as arestas
ttk.Button(frame, text="Mostrar Todas as Ruas", command=show_all_edges).grid(column=4, row=1)


root.mainloop()
