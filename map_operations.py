import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import osmnx as ox

from traffic_model import analyze_traffic


def calculate_route_with_traffic(G, edges, combo, buildings, green_areas, traffic_model, label_encoder):
    """
    Calcula a rota com base no tráfego e exibe no mapa.

    Args:
    - G: Grafo do OpenStreetMap.
    - edges: Dicionário de arestas.
    - combo: Combobox para seleção de arestas.
    - buildings: Características dos prédios no mapa.
    - green_areas: Áreas verdes no mapa.
    - traffic_model: Modelo treinado para previsão de tráfego.
    - label_encoder: Codificador de rótulos.

    Returns:
    - Exibe o mapa com a rota calculada e informações visuais.
    """

    chosen_edge = edges[combo.get()]
    # Obter o nó inicial da aresta escolhida
    chosen_node = chosen_edge[0]

    # Analisar o tráfego ao longo da rota calculada
    G_with_traffic = analyze_traffic(G.copy(), traffic_model, label_encoder)

    # Calcular o nó mais próximo das coordenadas de destino (apenas para fins de exemplo)
    destination_node = ox.nearest_nodes(G_with_traffic, -46.67674, -23.59984)

    # Calcular a rota mais curta considerando o tráfego
    route_with_traffic = nx.shortest_path(G_with_traffic, chosen_node, destination_node, weight='weight')

    # Cria uma lista de cores para todas as arestas do grafo
    edge_colors = [G_with_traffic[u][v][0].get('edge_color', 'Purple') for u, v, k in G_with_traffic.edges(keys=True)]

    # Passando lista para a função
    fig, ax = ox.plot_graph_route(G_with_traffic, route_with_traffic, edge_color=edge_colors, route_color="blue",
                                  node_size=0,
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
