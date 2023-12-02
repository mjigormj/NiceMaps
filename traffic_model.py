import networkx as nx
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
"""Importa bibliotecas necessárias"""

def train_traffic_model():
    labels = ['livre', 'moderado', 'pesado']
    """Os rótulos representam as condições de tráfego possíveis"""

    # Criando um conjunto de dados balanceado
    num_samples = 300  # Total de amostras desejadas
    samples_per_class = num_samples // len(labels)  # Número de amostras por classe
    """Define o número total de amostras desejadas e calcula o número de amostras por classe para criar um conjunto de dados balanceado"""

    balanced_features = []
    balanced_labels = []
    """Essas listas serão preenchidas com as características e rótulos das amostras balanceadas"""
    
    # Lendo os dados do arquivo CSV
    data = pd.read_csv('dados.csv', sep=";")
    """Carrega os dados do arquivo CSV chamado 'dados.csv' usando a biblioteca pandas"""
    
    for label_idx, label in enumerate(labels):
        # Filtrando os dados por classe
        class_data = data[data['Label'] == label]  # Supondo que 'classe' seja a coluna que contém as classes
        """Itera sobre cada rótulo e filtra os dados do DataFrame original para obter apenas as amostras da classe correspondente"""

        # Pegando amostras para cada classe de forma balanceada
        for _ in range(samples_per_class):
            # Pegando uma linha aleatória do dataframe filtrado
            random_feature = class_data.sample(1).values[0]
            balanced_features.append(random_feature[:-1])  # Removendo a coluna de classe das features
            balanced_labels.append(label)
            """Para cada classe, é feita uma amostragem balanceada, onde uma linha aleatória é escolhida do DataFrame filtrado e suas características são adicionadas às listas balanced_features e balanced_labels"""

    # Embaralhando os dados
    balanced_features, balanced_labels = shuffle(balanced_features, balanced_labels, random_state=42)
    """As listas de características e rótulos são embaralhadas para garantir uma distribuição aleatória"""
    
    # Pré-processamento
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(balanced_labels)
    """Usa LabelEncoder para transformar rótulos de strings em números inteiros"""

    # Treinamento do modelo
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(balanced_features, labels_encoded)
    """Cria e treina um modelo de Random Forest usando balanced_features e labels_encoded"""

    return model, label_encoder
    """Retorna o modelo treinado e o LabelEncoder """

def predict_traffic_conditions(model, label_encoder, features):
    # Fazer previsões usando o modelo treinado
    predictions = model.predict(features)
    """faz previsões usando o modelo treinado nas características fornecidas"""
    predicted_classes = label_encoder.inverse_transform(predictions)
    """reverte as previsões de volta para rótulos originais"""
    return predicted_classes
"""Usa o modelo treinado para fazer previsões com base nas características fornecidas"""

def analyze_traffic(G, traffic_model, label_encoder):
    # Coleta dados de tráfego para cada aresta do grafo
    edge_features = []

    for u, v, data in G.edges(data=True):
        edge_data = [
            np.random.uniform(4, 16),  # Largura da rua
            np.random.uniform(5, 60),  # Velocidade média
            np.random.uniform(1, 27), # Densidade de tráfego
            np.random.uniform(1, 4), # Clima
            np.random.uniform(1, 3)  # Periodo
        ]
        edge_features.append(edge_data)
    """coleta características para todas as arestas reais no grafo"""
    
    # Previsões de tráfego para cada aresta com o modelo
    predicted_traffic = predict_traffic_conditions(traffic_model, label_encoder, np.array(edge_features))
    """usado para fazer previsões de tráfego com base nas características coletadas"""

    # Atribuição de pesos e cores com base nas previsões de tráfego
    edge_colors = []
    edge_weights = {}
    """são listas que armazenam cores e pesos para cada aresta, respectivamente"""
    
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
    """Examina arestas e previsões com base nas previsões, atribui cores e pesos às arestas"""
    
    # Define atributos de peso e cor para as arestas
    nx.set_edge_attributes(G, edge_weights, 'weight')
    nx.set_edge_attributes(G, dict(zip(G.edges(keys=True), edge_colors)), 'edge_color')
    """usado para atribuir os pesos e cores às arestas do grafo"""
    
    return G
