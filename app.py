# Simulação simples no Colab (sem Streamlit)
# Testando o agente via input()

# 1) Importa dependências mínimas
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# 2) Carrega um CSV de exemplo
df = pd.DataFrame({
    "time": np.arange(100),
    "amount": np.random.normal(50, 10, 100),
    "v1": np.random.randn(100),
    "v2": np.random.randn(100)
})
print("DataFrame de teste criado com colunas:", df.columns.tolist())

# 3) Função exemplo: histograma
def generate_histogram(column):
    if column not in df.columns:
        return f"Coluna {column} não encontrada."
    fig = px.histogram(df, x=column, title=f"Histograma de {column}")
    fig.show()
    return f"Histograma gerado para '{column}'"

# 4) Simulação de pergunta do usuário
pergunta = input("Digite sua pergunta (ex: histograma de amount): ")

if "histograma" in pergunta.lower():
    col = pergunta.split()[-1]
    print(generate_histogram(col))
else:
    print("Função ainda não implementada para essa pergunta.")
