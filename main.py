import random
import re
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import osmnx as ox
from sklearn.cluster import DBSCAN
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import utils
from coordenadas import Coordenadas
import matplotlib.patches as mpatches

# Função de treinamento do modelo de tráfego
def train_traffic_model():
    # Largura da rua em metros, Velocidade média em km/h, Densidade de tráfego
    features = [
        [10, 90, 5],
        [8, 60, 7],
        [5, 30, 20]
    ]

    labels = ['livre', 'moderado', 'pesado']

    # Criando um conjunto de dados balanceado
    num_samples = 300  # Total de amostras desejadas
    samples_per_class = num_samples // len(labels)  # Número de amostras por classe

    balanced_features = []
    balanced_labels = []

    # Gerando dados balanceados
    for label_idx, label in enumerate(labels):
        # Criando dados para cada classe de forma balanceada
        for _ in range(samples_per_class):
            # Criando uma amostra com 33% de chance para cada valor
            random_feature = [
                np.random.choice([5, 8, 10]),
                np.random.choice([30, 60, 90]),
                np.random.choice([5, 7, 20])
            ]
            balanced_features.append(random_feature)
            balanced_labels.append(label)

    # Embaralhando os dados
    balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels, random_state=42)

    # Pré-processamento
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(balanced_labels)

    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(balanced_features, labels_encoded)

    return model, label_encoder

def predict_traffic_conditions(model, label_encoder, features):
    # Fazer previsões usando o modelo treinado
    predictions = model.predict(features)
    predicted_classes = label_encoder.inverse_transform(predictions)
    return predicted_classes

def analyze_traffic(G, traffic_model, label_encoder):
    # Coleta dados de tráfego para cada aresta do grafo
    edge_features = [
        [random.uniform(5, 20), random.uniform(60, 120), random.uniform(5, 25)],
        [random.uniform(5, 20), random.uniform(60, 120), random.uniform(5, 25)],
        [random.uniform(5, 20), random.uniform(60, 120), random.uniform(5, 25)]
    ]
    for u, v, data in G.edges(data=True):
        edge_data = [
            random.uniform(5, 20),  # Largura da rua
            random.uniform(60, 120),  # Velocidade média
            random.uniform(5, 25)  # Densidade de tráfego
        ]
        edge_features.append(edge_data)

    # Previsões de tráfego para cada aresta com o modelo
    predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, np.array(edge_features))

    # Atribuição de pesos e cores com base nas previsões de tráfego
    edge_colors = []
    edge_weights = {}
    for (u, v, data), traffic in zip(G.edges(data=True), predicted_traffic):
        edge_key = (u, v, 0)  # Chave da aresta
        if traffic == 'moderado':
            edge_colors.append('Yellow')
            edge_weights[edge_key] = 1.5  # Atribui um peso médio para tráfego moderado
        elif traffic == 'livre':
            edge_colors.append('Green')
            edge_weights[edge_key] = 1  # Atribui um peso baixo para tráfego livre
        elif traffic == 'pesado':
            edge_colors.append('Red')
            edge_weights[edge_key] = 3  # Atribui um peso alto para tráfego pesado
        else:
            edge_colors.append('Purple')  # Cor padrão para outras condições
            edge_weights[edge_key] = 1  # Peso padrão para outras condições

    # Define atributos de peso e cor para as arestas
    nx.set_edge_attributes(G, edge_weights, 'weight')
    nx.set_edge_attributes(G, dict(zip(G.edges(keys=True), edge_colors)), 'edge_color')

    return G

def calculate_route_with_traffic():
    chosen_edge = edges[combo.get()]
    # Obter o nó inicial da aresta escolhida
    chosen_node = chosen_edge[0]

    # Analisar o tráfego ao longo da rota calculada
    G_with_traffic = analyze_traffic(G.copy(), traffic_model, label_encoder)

    # Calcular o nó mais próximo das coordenadas de destino (apenas para fins de exemplo)
    destination_node = ox.nearest_nodes(G_with_traffic, -46.676660, -23.599918)

    # Calcular a rota mais curta considerando o tráfego
    route_with_traffic = nx.shortest_path(G_with_traffic, chosen_node, destination_node, weight='weight')

    # Cria uma lista de cores para todas as arestas do grafo
    edge_colors = [G_with_traffic[u][v][0].get('edge_color', 'Purple') for u, v, k in G_with_traffic.edges(keys=True)]

    # Agora, passe essa lista para a função
    fig, ax = ox.plot_graph_route(G_with_traffic, route_with_traffic, edge_color=edge_colors, route_color="blue", node_size=0,
                                  route_linewidth=5, show=False, close=False)

    # Cria as legendas
    red_patch = mpatches.Patch(color='red', label='Pesado')
    yellow_patch = mpatches.Patch(color='yellow', label='Moderado')
    green_patch = mpatches.Patch(color='green', label='Livre')

    # Adiciona as legendas ao gráfico
    plt.legend(handles=[red_patch, yellow_patch, green_patch])

    # Adicionar informações visuais ao mapa
    ax.set_title('Rota com Base no Tráfego')
    buildings.plot(ax=ax, color="Gray", alpha=0.5)
    green_areas.plot(ax=ax, color='LightSeaGreen', alpha=0.5)
    ox.plot_footprints(buildings, ax=ax, color='gray', alpha=0.5)

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

combo = ttk.Combobox(frame, width=40, values=list(edges.keys()))
combo.grid(column=2, row=1)
combo.current(0)

# Adicione um botão para mostrar todas as arestas
ttk.Button(frame, text="Calcular rota", command=calculate_route_with_traffic).grid(column=4, row=1)

# Inicie o loop principal após configurar todos os elementos da interface gráfica
root.mainloop()
