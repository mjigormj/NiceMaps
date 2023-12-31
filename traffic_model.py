import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle


def train_traffic_model():
    labels = ['livre', 'moderado', 'pesado']

    # Criando um conjunto de dados balanceado
    num_samples = 300  # Total de amostras desejadas
    samples_per_class = num_samples // len(labels)  # Número de amostras por classe

    balanced_features = []
    balanced_labels = []

    # Lendo os dados do arquivo CSV
    data = pd.read_csv('dados.csv')

    for label_idx, label in enumerate(labels):
        # Filtrando os dados por classe
        class_data = data[data['Label'] == label]  # Supondo que 'classe' seja a coluna que contém as classes

        # Pegando amostras para cada classe de forma balanceada
        for _ in range(samples_per_class):
            # Pegando uma linha aleatória do dataframe filtrado
            random_feature = class_data.sample(1).values[0]
            balanced_features.append(random_feature[:-1])  # Removendo a coluna de classe das features
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
        [np.random.uniform(4, 16), np.random.uniform(5, 60), np.random.uniform(1, 27)]
        for _ in range(3)
    ]
    for u, v, data in G.edges(data=True):
        edge_data = [
            np.random.uniform(4, 16),  # Largura da rua
            np.random.uniform(5, 60),  # Velocidade média
            np.random.uniform(1, 27)  # Densidade de tráfego
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
