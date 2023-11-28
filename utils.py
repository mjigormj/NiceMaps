import osmnx as ox


class Coordenadas:
    def __init__(self, nome, latitude, longitude, descricao):
        self.nome = nome
        self.latitude = latitude
        self.longitude = longitude
        self.descricao = descricao


def create_graph_from_point(coordenada):
    return ox.graph_from_point((coordenada.latitude, coordenada.longitude), dist=1500, network_type='drive')


def get_features(coordenada, tags):
    return ox.features_from_point((coordenada.latitude, coordenada.longitude), tags=tags, dist=1500)
