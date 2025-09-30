# -*- coding: utf-8 -*-
"""
Streamlit app: Agente de Análise Exploratória de Dados (EDA)
Versão: simplificada e robusta para rodar em Streamlit Cloud (Python 3.12)
Objetivo: responder perguntas listadas pelo usuário e gerar gráficos (Plotly + fallback Matplotlib).

Principais mudanças/observações:
- Simplifiquei a criação de tools e a interface para reduzir reruns e loops.
- Quando um gráfico é gerado, SALVA-SE tanto um objeto Plotly (plotly_figure)
  quanto um objeto Matplotlib de fallback (grafico_para_exibir) com os comandos
  solicitados: "fig = plt.figure(figsize=(10, 6))" e "st.session_state.grafico_para_exibir = fig".
- Tornei o carregamento do arquivo cacheável e seguro para Streamlit Cloud.
- Mantive as funções de análise essenciais pedidas no enunciado.

Dependências (requirements.txt):
streamlit>=1.36.0
numpy
pandas
matplotlib
seaborn
plotly
scikit-learn
tabulate
langchain>=0.2.0
langchain-core>=0.2.0
langchain-google-genai>=0.0.6

Uso rápido:
1) Adicione a chave do Google AI em st.secrets['google_ai']['google_api_key']
2) Faça upload de CSV ou ZIP contendo CSV
3) Inicialize agente e faça perguntas como "Gere um histograma de amount" ou
   "Quais são os tipos de dados?". O agente escolherá a ferramenta apropriada.

"""

import io
import zipfile
import re
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory

# -----------------------------
# Configuração e estado inicial
# -----------------------------
st.set_page_config(page_title="Agente EDA - Gemini", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA) - Versão Estável")
st.markdown(
    "Envie um CSV (ou ZIP) e pergunte ao agente. Ele tenta responder e, quando aplicável, gera gráficos interativos."
)

# Secrets: chave do Google
try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except Exception:
    st.error("Chave de API do Google não encontrada em st.secrets['google_ai']['google_api_key'].")
    st.stop()

# Estado da sessão
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "messages" not in st.session_state:
    st.session_state.messages = []
# Figure objects: plotly and matplotlib fallback
if "figure_to_display" not in st.session_state:
    st.session_state.figure_to_display = None
if "grafico_para_exibir" not in st.session_state:
    st.session_state.grafico_para_exibir = None

# -----------------------------
# Utilitários de carregamento
# -----------------------------
@st.cache_data(show_spinner=False)
def load_and_extract_data(uploaded_file_bytes, filename):
    if uploaded_file_bytes is None:
        return {"status": "error", "message": "Nenhum arquivo recebido."}
    bio = io.BytesIO(uploaded_file_bytes)

    try:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(bio, "r") as z:
                # procura pelo primeiro csv
                csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_names:
                    return {"status": "error", "message": "ZIP não contém CSV."}
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(bio)
        else:
            return {"status": "error", "message": "Formato não suportado. Envie CSV ou ZIP."}

        # Normaliza colunas
        df.columns = [c.lower() for c in df.columns]
        return {"status": "success", "df": df, "message": f"Arquivo {filename} carregado."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar arquivo: {e}"}

# -----------------------------
# Ferramentas / Tools do agente
# -----------------------------
# Observação: cada ferramenta retorna um dicionário com 'status' e 'message'.


def show_descriptive_stats(*args):
    """Gera estatísticas descritivas (tipos, contagem, média, mediana, std, min, max)"""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}

    types = pd.DataFrame(df.dtypes, columns=["dtype"]).astype(str)
    numeric = df.select_dtypes(include=[np.number]).describe().T
    top_values = {}
    for col in df.columns:
        top = df[col].value_counts(dropna=False).head(5).to_dict()
        top_values[col] = top

    out = "**Tipos de dados (dtype):**\n\n" + types.to_markdown(tablefmt="pipe")
    out += "\n\n**Estatísticas (colunas numéricas):**\n\n" + numeric.to_markdown(tablefmt="pipe")
    # adicionar top values (resumido)
    out += "\n\n**Valores mais frequentes (top 5 por coluna):**\n"
    for col, tv in top_values.items():
        out += f"\n- **{col}**: {tv}\n"

    return {"status": "success", "message": out}


def generate_histogram(column: str, *args):
    """Gera histograma Plotly (interativo) para coluna numérica e salva fallback Matplotlib."""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    col = column.lower().strip()
    if col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}
    if not pd.api.types.is_numeric_dtype(df[col]):
        return {"status": "error", "message": f"Coluna '{col}' não é numérica."}

    # Plotly
    fig = px.histogram(df, x=col, title=f"Histograma de {col}")
    st.session_state.figure_to_display = fig

    # --- Matplotlib fallback (as linhas solicitadas) ---
    fig_mpl = plt.figure(figsize=(10, 6))
    st.session_state.grafico_para_exibir = fig_mpl
    # plot no matplotlib (simples)
    ax = fig_mpl.add_subplot(1, 1, 1)
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Histograma (mpl) de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Contagem")

    message = f"Histograma gerado para '{col}'. Ver gráfico interativo e fallback Matplotlib salvo."
    return {"status": "success", "message": message}


def generate_correlation_heatmap(*args):
    """Gera mapa de calor da correlação (Plotly) entre colunas numéricas e salva fallback Matplotlib."""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Não há colunas numéricas suficientes."}

    corr = df[numeric_cols].corr()
    fig = px.imshow(corr, text_auto='.2f', aspect='auto', title='Matriz de Correlação')
    fig.update_xaxes(side='top')
    st.session_state.figure_to_display = fig

    # Matplotlib fallback
    fig_mpl = plt.figure(figsize=(10, 6))
    st.session_state.grafico_para_exibir = fig_mpl
    ax = fig_mpl.add_subplot(1, 1, 1)
    cax = ax.matshow(corr, cmap='RdBu')
    fig_mpl.colorbar(cax)
    ax.set_xticks(range(len(numeric_cols)))
    ax.set_yticks(range(len(numeric_cols)))
    ax.set_xticklabels(numeric_cols, rotation=90)
    ax.set_yticklabels(numeric_cols)
    ax.set_title('Matriz de Correlação (mpl)')

    return {"status": "success", "message": "Mapa de calor da correlação gerado (Plotly + Matplotlib fallback)."}


def generate_scatter_plot(columns_str: str, *args):
    """Gera scatter Plotly entre duas colunas e salva fallback Matplotlib."""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    col_names = re.split(r"[,\s]+", columns_str.lower())
    col_names = [c for c in col_names if c and c != 'e']
    if len(col_names) < 2:
        return {"status": "error", "message": "Forneça duas colunas separadas por vírgula ou espaço."}
    x_col, y_col = col_names[0], col_names[1]
    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Colunas '{x_col}' ou '{y_col}' não encontradas."}

    fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")
    st.session_state.figure_to_display = fig

    # Matplotlib fallback
    fig_mpl = plt.figure(figsize=(10, 6))
    st.session_state.grafico_para_exibir = fig_mpl
    ax = fig_mpl.add_subplot(1, 1, 1)
    ax.scatter(df[x_col], df[y_col], alpha=0.6)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col} (mpl)")

    return {"status": "success", "message": f"Gráfico de dispersão gerado para '{x_col}' vs '{y_col}'."}


def detect_outliers_isolation_forest(*args):
    """Detecta outliers com IsolationForest nas colunas v* + time + amount (se existirem)."""
    try:
        df = st.session_state.df
        if df is None:
            return {"status": "error", "message": "Nenhum DataFrame carregado."}
        feature_cols = [c for c in df.columns if c.startswith('v')] + ['time', 'amount']
        existing = [c for c in feature_cols if c in df.columns]
        if not existing:
            return {"status": "error", "message": "Nenhuma coluna v*, time ou amount encontrada."}

        X = df[existing].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.01, random_state=42)
        labels = iso.fit_predict(Xs)
        # join back to original index
        out_idx = X.index[labels == -1]
        sample = df.loc[out_idx].head(10)
        msg = f"Foram detectados {len(out_idx)} outliers (amostra até 10 linhas):\n\n" + sample.to_markdown(tablefmt="pipe")
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": f"Erro: {e}"}


def find_clusters_kmeans(n_clusters: str, *args):
    """Agrupamento KMeans nas colunas v* + time + amount."""
    try:
        n = int(n_clusters)
    except Exception:
        return {"status": "error", "message": "Número de clusters inválido."}
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    feature_cols = [c for c in df.columns if c.startswith('v')] + ['time', 'amount']
    existing = [c for c in feature_cols if c in df.columns]
    if not existing:
        return {"status": "error", "message": "Nenhuma coluna v*, time ou amount encontrada."}

    X = df[existing].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    km = KMeans(n_clusters=n, random_state=42, n_init='auto')
    labels = km.fit_predict(Xs)
    df_copy = df.copy()
    df_copy['cluster'] = labels
    summary = df_copy.groupby('cluster').agg({'amount': ['mean','min','max'], 'time': ['min','max']}).to_markdown(tablefmt='pipe')
    return {"status": "success", "message": f"KMeans executado com {n} clusters.\n\n{summary}"}

# Converte funções para Tools
tool_functions = [
    show_descriptive_stats,
    generate_histogram,
    generate_correlation_heatmap,
    generate_scatter_plot,
    detect_outliers_isolation_forest,
    find_clusters_kmeans,
]

# -----------------------------
# Inicializa Agente LangChain
# -----------------------------

def initialize_agent(tools_list, system_prompt_text):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=google_api_key, temperature=0.0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools_list, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools_list, verbose=False, memory=memory, max_iterations=10)
    return agent_executor

# -----------------------------
# Interface do Streamlit
# -----------------------------
with st.sidebar:
    st.header("Upload de dados")
    uploaded_file = st.file_uploader("CSV ou ZIP (com CSV)", type=["csv","zip"])

    if st.button("Carregar e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando arquivo..."):
            data = load_and_extract_data(uploaded_file.getvalue(), uploaded_file.name)
        if data['status'] == 'success':
            st.session_state.df = data['df']

            # cria tools
            tools_with_df = [Tool(name=fn.__name__, description=fn.__doc__, func=fn) for fn in tool_functions]

            system_prompt = (
                "Você é um agente de EDA que deve, sempre que possível, usar as ferramentas disponíveis para responder às perguntas do usuário. "
                "Se o usuário pedir um gráfico, execute a ferramenta adequada e gere a figura interativa. Responda em Português e forneça insights simples e diretos."
            )
            try:
                st.session_state.agent_executor = initialize_agent(tools_with_df, system_prompt)
                st.success("Agente inicializado. Faça suas perguntas na caixa de chat abaixo.")
            except Exception as e:
                st.error(f"Erro ao inicializar agente: {e}")
        else:
            st.error(data['message'])

    if st.session_state.df is not None:
        st.success(f"DataFrame carregado: {len(st.session_state.df)} linhas, {len(st.session_state.df.columns)} colunas")
        st.subheader("Amostra")
        st.dataframe(st.session_state.df.head())

# Chat input e exibição de respostas
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

prompt = st.chat_input("Pergunte algo (ex: 'Gere um histograma de amount' ou 'Quais são os tipos de dados?')")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.agent_executor is None:
        with st.chat_message("assistant"):
            st.error("Agente não inicializado. Faça upload do arquivo e clique em 'Carregar e Inicializar Agente' na barra lateral.")
    else:
        with st.chat_message("assistant"):
            resp_container = st.container()
            try:
                full = st.session_state.agent_executor.invoke({"input": prompt})
                response_content = full.get('output')

                # Se a figura Plotly foi salva na sessão, exibir agora
                if st.session_state.figure_to_display is not None:
                    resp_container.plotly_chart(st.session_state.figure_to_display, use_container_width=True)
                    st.session_state.figure_to_display = None

                # Se houver fallback matplotlib, exibimos como imagem
                if st.session_state.grafico_para_exibir is not None:
                    # Converte matplotlib figure para imagem via streamlit
                    resp_container.pyplot(st.session_state.grafico_para_exibir)
                    st.session_state.grafico_para_exibir = None

                # Tratamento da resposta da ferramenta
                if isinstance(response_content, dict) and response_content.get('status') in ['success','error']:
                    if response_content.get('status') == 'error':
                        resp_container.error(response_content.get('message'))
                    else:
                        resp_container.markdown(response_content.get('message'))
                    st.session_state.messages.append({"role": "assistant", "content": response_content.get('message')})
                else:
                    # Texto retornado diretamente pelo LLM
                    resp_container.markdown(str(response_content))
                    st.session_state.messages.append({"role": "assistant", "content": str(response_content)})

            except Exception as e:
                err = f"Erro na execução do agente: {e}"
                resp_container.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# Pequena ajuda com as perguntas pedidas no desafio (exemplos de prompts)
with st.expander("Exemplos de perguntas (copie e cole)"):
    st.markdown("- Quais são os tipos de dados (numéricos, categóricos)?\n- Qual a distribuição de amount?\n- Gere um histograma de v1\n- Gere mapa de calor da correlação\n- Detecte outliers\n- Faça KMeans com 4 clusters\n- Gere um scatter plot entre time e amount")

# Fim do arquivo
