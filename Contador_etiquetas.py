import pandas as pd

# Cargar el dataset
df = pd.read_csv('emotions.csv')

# Visualizar las primeras filas para verificar la estructura
print("Primeras filas del conjunto de datos:")
print(df.head())

# Verificar distribución de clases
print("\nDistribución de clases:")
print(df['sentiment'].value_counts())

# Verificar valores nulos por columna
print("\nValores nulos por columna:")
print(df.isnull().sum())
