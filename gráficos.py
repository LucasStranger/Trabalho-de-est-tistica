#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import statsmodels.stats.api as sms
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.graphics.gofplots import qqplot
from statsmodels.compat import lzip

# Carregar dados
df = pd.read_csv("Consumo_cerveja.csv")

# Explorar dados
print(df.head())
df.info()
print(df.describe())

# Correlação
print(df.corr(numeric_only=True))

# Visualização
plt.figure(figsize=(15,7))
sns.countplot(x='Final de Semana', data=df)
plt.xlabel('Final de Semana')
plt.ylabel('')
plt.show()

df[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)']].plot(figsize=(15,7))
plt.title('Séries de temperaturas máximas, Médias e Mínimas', size=15)
plt.show()

df['Precipitacao (mm)'].plot(figsize=(15,7))
plt.title('Precipitação em mm', size=15)
plt.show()

df['Consumo de cerveja (litros)'].plot(figsize=(15,7), color='black')
plt.title('Consumo de cerveja', size=15)
plt.show()

plt.figure(figsize=(15,7))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="RdYlGn")
plt.title('Correlação de Pearson', size=15)
plt.show()

plt.figure(figsize=(15,7))
plt.title('Correlação de Spearman', size=15)
sns.heatmap(df.corr('spearman', numeric_only=True), annot=True, cmap="coolwarm")
plt.show()

df[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']].plot.box(figsize=(15,7))
plt.show()

df[['Temperatura Media (C)', 'Temperatura Minima (C)', 'Temperatura Maxima (C)', 'Precipitacao (mm)', 'Consumo de cerveja (litros)']].hist(figsize=(15,7), bins=50)
plt.show()

fig, ax = plt.subplots(2, 2, figsize=(15,7))
sns.scatterplot(x='Consumo de cerveja (litros)', y='Temperatura Media (C)', data=df, ax=ax[0,0])
sns.scatterplot(x='Consumo de cerveja (litros)', y='Temperatura Minima (C)', data=df, ax=ax[0,1])
sns.scatterplot(x='Consumo de cerveja (litros)', y='Temperatura Maxima (C)', data=df, ax=ax[1,0])
sns.scatterplot(x='Consumo de cerveja (litros)', y='Precipitacao (mm)', data=df, ax=ax[1,1])
plt.show()

# Separação entre variável dependente e independentes
X = df.drop(['Consumo de cerveja (litros)', 'Data'], axis=1)
y = df['Consumo de cerveja (litros)']

# Regressão Linear com Scikit-Learn
modelo = LinearRegression()
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)
modelo.fit(X_treino, y_treino)

# Coeficiente de Determinação (R²)
print("R²:", modelo.score(X_teste, y_teste))
# Intercepto
print("Intercepto:", modelo.intercept_)
# Coeficientes
print("Coeficientes:", modelo.coef_)

# Erros
y_pred = modelo.predict(X_teste)
print("MAE:", mean_absolute_error(y_teste, y_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_teste, y_pred)))

# Regressão Linear com Statsmodels
modelo1 = sm.OLS(y, sm.add_constant(X)).fit()
print(modelo1.summary(title='Sumário do modelo com intercepto'))

modelo2 = sm.OLS(y, X).fit()
print(modelo2.summary(title='Sumário do modelo sem intercepto'))

# Resíduos e Diagnósticos
residuos1 = modelo1.resid
fig, ax = plt.subplots(2, 2, figsize=(15, 6))
residuos1.plot(title="Resíduos do modelo 1", ax=ax[0,0])
sns.histplot(residuos1, kde=True, ax=ax[0,1])
plot_acf(residuos1, lags=40, ax=ax[1,0])
qqplot(residuos1, line='s', ax=ax[1,1])
plt.show()

residuos2 = modelo2.resid
fig, ax = plt.subplots(2, 2, figsize=(15, 6))
residuos2.plot(title="Resíduos do modelo 2", ax=ax[0,0])
sns.histplot(residuos2, kde=True, ax=ax[0,1])
plot_acf(residuos2, lags=40, ax=ax[1,0])
qqplot(residuos2, line='s', ax=ax[1,1])
plt.show()

# Teste Omnibus
nome1 = ['Estatística', 'Probabilidade']
teste1 = sms.omni_normtest(modelo1.resid)
print(list(zip(nome1, teste1)))

nome2 = ['Estatística', 'Probabilidade']
teste2 = sms.omni_normtest(modelo2.resid)
print(list(zip(nome2, teste2)))

# Multicolinearidade
print('Número condição do modelo 1:', np.linalg.cond(modelo1.model.exog))
print('Número condição do modelo 2:', np.linalg.cond(modelo2.model.exog))

# Resíduos plot
df1 = df.copy()
df1['residuos1'] = modelo1.resid
df1['residuos2'] = modelo2.resid

fig, ax = plt.subplots(1, 2, figsize=(20, 7))
sns.regplot(x='Consumo de cerveja (litros)', y='residuos1', data=df1, ax=ax[0])
sns.regplot(x='Consumo de cerveja (litros)', y='residuos2', data=df1, ax=ax[1])
plt.show()

