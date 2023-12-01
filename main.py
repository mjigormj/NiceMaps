import re
import tkinter as tk
from tkinter import ttk

from map_operations import calculate_route_with_traffic
from traffic_model import train_traffic_model
from utils import Coordenadas, create_graph_from_point, get_features
"""O código começa importando as bibliotecas necessárias"""

# Dados iniciais
coordenadasFixas = [
    Coordenadas("Casa do Ator", -23.591563, -46.682362, ""),
    Coordenadas("anhembi", -23.599899, -46.676735, ""),
]
"""São definidas algumas coordenadas fixas, representando pontos no mapa"""

G = create_graph_from_point(coordenadasFixas[0])
"""Cria um grafo OpenStreetMap a partir das coordenadas do primeiro ponto fixo"""

green_areas = get_features(coordenadasFixas[0], {'leisure': 'park'})
"""Obtém características das áreas verdes no mapa"""

buildings = get_features(coordenadasFixas[0], {'building': True})
"""Obtém características dos prédios no mapa"""

# Listando nome das ruas
edges_raw = list(G.edges.data('name'))
"""Obtém informações brutas das arestas do grafo, incluindo nomes de ruas"""
clean_edges = {}

# Formatando nome das ruas
for i, edge in enumerate(edges_raw):
    clean_edge_name = re.sub(r'[\d(),\'\"]', '', str(edge[2]))
    clean_edges[f"{i}: {clean_edge_name}"] = edge

edges = dict(sorted(clean_edges.items(), key=lambda item: item[1]))
"""O loop for formata os nomes das ruas removendo caracteres indesejados e cria um dicionário chamado edges"""

# Treinar o modelo de tráfego
traffic_model, label_encoder = train_traffic_model()
"""Treina o modelo de tráfego utilizando dados do arquivo dados.csv"""

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
"""Configuração da interface gráfica usando Tkinter"""

# Inicia o mapa
ttk.Button(frame, text="Calcular rota", command=lambda: calculate_route_with_traffic(G, edges, combo, buildings,
                                                                                     green_areas, traffic_model,
                                                                                     label_encoder)).grid(column=4, row=1)
"""Adiciona um botão à interface que chama a função calculate_route_with_traffic quando pressionado.
Inicia o loop principal da interface gráfica"""

# Inicie o loop principal após configurar todos os elementos da interface gráfica
root.mainloop()
