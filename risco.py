import pickle

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

# Importando a base de dados
base_risco_credito = pd.read_csv('risco_credito.csv')

# Separação em preditores e classes
x_risco_credito = base_risco_credito.iloc[:, 0:4].values
y_risco_credito = base_risco_credito.iloc[:, 4].values

# Pre processamento dos dados
label_encoder_historia = LabelEncoder()
label_encoder_divida = LabelEncoder()
label_encoder_garantias = LabelEncoder()
label_encoder_renda = LabelEncoder()

x_risco_credito[:, 0] = label_encoder_historia.fit_transform(x_risco_credito[:, 0])
x_risco_credito[:, 1] = label_encoder_divida.fit_transform(x_risco_credito[:, 1])
x_risco_credito[:, 2] = label_encoder_garantias.fit_transform(x_risco_credito[:, 2])
x_risco_credito[:, 3] = label_encoder_renda.fit_transform(x_risco_credito[:, 3])

# Salvando os dados pre proccessados em um arquivo
with open('risco_credito.pkl', 'wb') as f:
    pickle.dump([x_risco_credito, y_risco_credito], f)

# Gerando as tabelas de probabilodades
naive_risco_crdito = GaussianNB()
naive_risco_crdito.fit(x_risco_credito, y_risco_credito)

# Fazendo a previsão com dados fictícios
# História = boa(0), divida = alta(0), garantias = nenhuma(1), renda > 35(2)
# História = ruim(2), divida = alta(0), garantias = adequada(0), renda < 15(0)
previsao = naive_risco_crdito.predict([[0, 0, 1, 2], [2, 0, 0, 0]])
print(previsao)