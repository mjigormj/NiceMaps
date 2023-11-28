import random
import re
import tkinter as tk
from tkinter import ttk

import numpy as np
from sklearn.cluster import DBSCAN

from map_operations import calculate_route_with_traffic
from traffic_model import train_traffic_model
from utils import Coordenadas, create_graph_from_point, get_features

# Dados iniciais
coordenadasFixas = [
    Coordenadas("Casa do Ator", -23.591563, -46.682362, ""),
    Coordenadas("anhembi", -23.599899, -46.676735, ""),
]

G = create_graph_from_point(coordenadasFixas[0])
green_areas = get_features(coordenadasFixas[0], {'leisure': 'park'})
buildings = get_features(coordenadasFixas[0], {'building': True})

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
ttk.Button(frame, text="Calcular rota", command=lambda: calculate_route_with_traffic(G, edges, combo, buildings,
                                                                                     green_areas, traffic_model,
                                                                                     label_encoder)).grid(column=4, row=1)

# Inicie o loop principal após configurar todos os elementos da interface gráfica
root.mainloop()