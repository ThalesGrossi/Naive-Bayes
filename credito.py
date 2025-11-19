import pickle

import pandas as pd
import numpy as np

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from yellowbrick.classifier import ConfusionMatrix

# Importando dados
with open('credit.pkl', 'rb') as f:
    [x_credit_treinamento, y_credit_treinamento,
    x_credit_teste, y_credit_teste] = pickle.load(f)

# Gerando tabela de estatísticas
naive_credit_data = GaussianNB()
naive_credit_data.fit(x_credit_treinamento, y_credit_treinamento)

# Fazendo previsões e comparando com o gabarito do teste
previsoes = naive_credit_data.predict(x_credit_teste)
print(accuracy_score(y_credit_teste, previsoes))
print(confusion_matrix(y_credit_teste, previsoes))

cm = ConfusionMatrix(naive_credit_data)
cm.fit(x_credit_treinamento, y_credit_treinamento)
print(cm.score(x_credit_teste, y_credit_teste))

print(classification_report(y_credit_teste, previsoes))