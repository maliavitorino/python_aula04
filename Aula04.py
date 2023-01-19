#!/usr/bin/env python
# coding: utf-8

# In[6]:


get_ipython().system('pip install matplotlib')
get_ipython().system('pip install seaborn ')
get_ipython().system('pip install scikit-learn')


# In[26]:


# Passo 1: Entendimento do Desafio
# Passo 2: Entendimento da Área/Empresa

# Passo 3: Importar a base de dados

import pandas as pd

tabela = pd.read_csv("advertising.csv")

# Passo 4: Ajuste de Dados (Tratamento/Limpeza)
# Passo 5: Análise Exploratório
import matplotlib.pyplot as plt
import seaborn as sns 

# criar um grafico
# sns.heatmap(tabela.corr(), cmap="Reds", annot=True)

# Passo 6: Modelagem + Algoritmos (Aqui entra a IA, se necessário)

y = tabela["Vendas"] # quem você quer prever
x = tabela[["TV", "Radio", "Jornal"]]

# separar os dados de treino e de teste

from sklearn.model_selection import train_test_split
x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

#importar a IA

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# criar a IA

modelo_regressaolinear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

# treinar a IA

modelo_regressaolinear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)

# pedir para fazer a previsão

previsao_regressaolinear = modelo_regressaolinear.predict(x_teste)
previsao_arvoredecisao = modelo_arvoredecisao.predict(x_teste)

# comparar

from sklearn.metrics import r2_score

print(r2_score(y_teste, previsao_regressaolinear))
print(r2_score(y_teste, previsao_arvoredecisao))

# visualização grafica

tabela_auxiliar = pd.DataFrame()
tabela_auxiliar["y_teste"] = y_teste
tabela_auxiliar["Previsão Árvore Decisão"] = previsao_arvoredecisao
tabela_auxiliar["Previsão Regressão Linear"] = previsao_regressaolinear

sns.lineplot(data=tabela_auxiliar)
plt.show()

# Como fazer uma nova previsão

nova_tabela = pd.read_csv("novos.csv")
display(nova_tabela)

previsao = modelo_arvoredecisao.predict(nova_tabela)
print(previsao)

