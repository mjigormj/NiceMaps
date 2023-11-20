import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
import osmnx as ox
import networkx as nx
from sklearn.cluster import DBSCAN
import numpy as np
import random
from coordenadas import Coordenadas
import utils
import re

# Funções
def calculate_route():
    chosen_edge = edges[combo.get()]
    chosen_node = chosen_edge[0]
    route_estacao = utils.calculate_shortest_path(G, coordenadasFixas[2], coordenadasFixas[1])
    route_quata = utils.calculate_shortest_path(G, coordenadasFixas[3], coordenadasFixas[1])
    chosen_route = nx.shortest_path(G, chosen_node, utils.get_nearest_node(G, coordenadasFixas[1]))
    red = nx.shortest_path(G, utils.get_nearest_node(G, coordenadasFixas[1]), utils.get_nearest_node(G, coordenadasFixas[1]))

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
    edge_colors = ['blue' if label == -1 else 'red' for label in labels]
    edge_weights = {(u, v, k): 1 if label == -1 else 3 for (u, v, k, _), label in zip(G.edges(data=True, keys=True), labels)}

    fig, ax = ox.plot_graph_routes(G, [route_estacao, route_quata, chosen_route], node_size=0, edge_color=edge_colors,
                                   route_colors=['purple', 'purple', 'green'], route_linewidth=6, show=False, close=False)
    ox.plot_graph_route(G, red, route_color='red', route_linewidth=3, ax=ax, show=False, close=False)
    buildings.plot(ax=ax, color="silver", alpha=0.7)
    green_areas.plot(ax=ax, color='w', alpha=0.2)

    plt.show()

def continue_with_existing_routes():
    try:
        route_estacao = utils.calculate_shortest_path(G, coordenadasFixas[2], coordenadasFixas[1])
        route_quata = utils.calculate_shortest_path(G, coordenadasFixas[3], coordenadasFixas[1])
        red = nx.shortest_path(G, utils.get_nearest_node(G, coordenadasFixas[1]), utils.get_nearest_node(G, coordenadasFixas[1]))

        fig, ax = ox.plot_graph_routes(G, [route_estacao, route_quata], node_size=0, edge_color='gray',
                                       route_colors=['purple', 'green'], route_linewidth=6, show=False, close=False)
        ox.plot_graph_route(G, red, route_color='red', route_linewidth=3, ax=ax, show=False, close=False)
        buildings.plot(ax=ax, color="silver", alpha=0.7)
        green_areas.plot(ax=ax, color='w', alpha=0.2)

        plt.show()
    except Exception as e:
        messagebox.showerror("Erro", str(e))

# Dados iniciais
coordenadasFixas = [
    Coordenadas("Casa do Ator", -23.591563, -46.682362, ""),
    Coordenadas("anhembi", -23.599899, -46.676735, ""),
    Coordenadas("estacao", -23.595314, -46.689486, ""),
    Coordenadas("quata", -23.599317, -46.675556, "")
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

# Interface Gráfica
root = tk.Tk()
root.title("Escolha uma rua")

frame = ttk.Frame(root, padding="3 3 12 12")
frame.grid(column=0, row=0, sticky=(tk.N, tk.W, tk.E, tk.S))
frame.columnconfigure(0, weight=1)
frame.rowconfigure(0, weight=1)

ttk.Label(frame, text="Escolha uma rua:").grid(column=1, row=1, sticky=tk.W)

combo = ttk.Combobox(frame, width=100, values=list(edges.keys()))
combo.grid(column=2, row=1)
combo.current(0)

ttk.Button(frame, text="Calcular rota", command=calculate_route).grid(column=3, row=3)
ttk.Button(frame, text="Continuar com rotas existentes", command=continue_with_existing_routes).grid(column=3, row=4)

root.mainloop()
