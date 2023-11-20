import networkx as nx
import osmnx as ox

def create_graph_from_point(coordinate, dist=1500, network_type='drive'):
    return ox.graph_from_point((coordinate.latitude, coordinate.longitude), dist=dist, network_type=network_type)

def get_features(coordinate, tags, dist=1500):
    return ox.features_from_point((coordinate.latitude, coordinate.longitude), tags=tags, dist=dist)

def get_nearest_node(G, coordinate):
    return ox.nearest_nodes(G, coordinate.longitude, coordinate.latitude)

def calculate_shortest_path(G, start_coordinate, end_coordinate):
    start_node = get_nearest_node(G, start_coordinate)
    end_node = get_nearest_node(G, end_coordinate)
    return nx.shortest_path(G, start_node, end_node)
