import pickle

import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Importando dados
with open('census.pkl', 'rb') as f:
    [x_census_treinamento, y_census_treinamento, x_census_teste, y_census_teste] = pickle.load(f)

# Gerando tabela de estatísticas
naive_census_data = GaussianNB()
naive_census_data.fit(x_census_treinamento, y_census_treinamento)

# Fazendo previsões e comparando com o gabarito do teste
# Baixa precisão
previsoes = naive_census_data.predict(x_census_teste)
print(accuracy_score(previsoes, y_census_teste))
print(confusion_matrix(y_census_teste, previsoes))
