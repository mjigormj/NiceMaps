import osmnx as ox
"""Importa a biblioteca OSMnx"""

class Coordenadas:
    def __init__(self, nome, latitude, longitude, descricao):
        self.nome = nome
        self.latitude = latitude
        self.longitude = longitude
        self.descricao = descricao
"""Define uma classe para representar coordenadas com nome, latitude, longitude e descrição"""

def create_graph_from_point(coordenada):
    return ox.graph_from_point((coordenada.latitude, coordenada.longitude), dist=1500, network_type='drive')
"""Cria um grafo OpenStreetMap a partir de um ponto específico"""

def get_features(coordenada, tags):
    return ox.features_from_point((coordenada.latitude, coordenada.longitude), tags=tags, dist=1500)
"""Obtém características do mapa (como edifícios, áreas verdes) a partir de um ponto específico"""
