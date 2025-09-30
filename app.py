# -*- coding: utf-8 -*-
"""
Streamlit app: Agente de Análise Exploratória de Dados (EDA)
Versão: Otimizada (Python 3.12)
Objetivo: responder perguntas listadas pelo usuário e gerar gráficos (Plotly + fallback Matplotlib).
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

# Módulos LangChain
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
# figure_to_display é o Plotly Figure
if "figure_to_display" not in st.session_state:
    st.session_state.figure_to_display = None
# grafico_para_exibir é o Matplotlib Figure
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

def show_descriptive_stats(*args):
    """
    Gera estatísticas descritivas do DataFrame, incluindo:
    - Tipos de dados (dtype/Tipos de dados)
    - Contagem de valores
    - Estatísticas para colunas numéricas (média, mediana, desvio padrão, min, max, count)
    - Amostra de valores mais frequentes.
    
    Use esta ferramenta para perguntas sobre **estrutura**, **tipos de dados** (numéricos, categóricos), 
    **contagem de linhas/colunas**, **valores** ou **resumo** do DataFrame.
    """
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

    # Matplotlib fallback
    fig_mpl = plt.figure(figsize=(10, 6))
    st.session_state.grafico_para_exibir = fig_mpl
    ax = fig_mpl.add_subplot(1, 1, 1)
    ax.hist(df[col].dropna(), bins=30)
    ax.set_title(f"Histograma (mpl) de {col}")
    ax.set_xlabel(col)
    ax.set_ylabel("Contagem")

    message = f"Histograma gerado para '{col}'. Verifique os gráficos interativos e o fallback."
    return {"status": "success", "message": message, "plotly_figure": fig, "matplotlib_figure": fig_mpl}


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

    return {"status": "success", "message": "Mapa de calor da correlação gerado (Plotly + Matplotlib fallback).", "plotly_figure": fig, "matplotlib_figure": fig_mpl}


def generate_scatter_plot(columns_str: str, *args):
    """Gera scatter Plotly entre duas colunas e salva fallback Matplotlib."""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    # Processa as colunas, separando por vírgula, espaço ou 'e'
    col_names = re.split(r"[,\s]+", columns_str.lower().replace(' e ', ','))
    col_names = [c for c in col_names if c]
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

    return {"status": "success", "message": f"Gráfico de dispersão gerado para '{x_col}' vs '{y_col}'.", "plotly_figure": fig, "matplotlib_figure": fig_mpl}


def detect_outliers_isolation_forest(*args):
    """Detecta outliers com IsolationForest nas colunas v* + time + amount (se existirem)."""
    try:
        df = st.session_state.df
        if df is None:
            return {"status": "error", "message": "Nenhum DataFrame carregado."}
        
        # Colunas candidatas: 'v' seguidas por números, 'time' e 'amount'
        feature_cols = [c for c in df.columns if re.match(r'v\d+', c)] + ['time', 'amount']
        existing = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
        
        if not existing:
            return {"status": "error", "message": "Nenhuma coluna numérica 'v*', 'time' ou 'amount' encontrada para análise de outliers."}

        X = df[existing].dropna()
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        iso = IsolationForest(contamination=0.01, random_state=42)
        labels = iso.fit_predict(Xs)
        
        out_idx = X.index[labels == -1]
        sample = df.loc[out_idx].head(10)
        
        msg = f"Foram detectados {len(out_idx)} outliers (amostra até 10 linhas):\n\n" + sample.to_markdown(tablefmt="pipe")
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": f"Erro na detecção de outliers: {e}"}


def find_clusters_kmeans(n_clusters: str, *args):
    """Agrupamento KMeans nas colunas v* + time + amount (se existirem)."""
    try:
        n = int(n_clusters)
        if n <= 1:
             return {"status": "error", "message": "Número de clusters deve ser maior que 1."}
    except Exception:
        return {"status": "error", "message": "Número de clusters inválido."}
        
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
        
    feature_cols = [c for c in df.columns if re.match(r'v\d+', c)] + ['time', 'amount']
    existing = [c for c in feature_cols if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    
    if not existing:
        return {"status": "error", "message": "Nenhuma coluna numérica 'v*', 'time' ou 'amount' encontrada para agrupamento."}

    X = df[existing].fillna(0)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    
    if len(X) < n:
        return {"status": "error", "message": f"O DataFrame tem menos linhas ({len(X)}) do que o número de clusters ({n})."}
        
    km = KMeans(n_clusters=n, random_state=42, n_init='auto')
    labels = km.fit_predict(Xs)
    
    df_copy = df.copy()
    df_copy['cluster'] = labels
    summary = df_copy.groupby('cluster').agg({'amount': ['mean','min','max'], 'time': ['min','max']}).to_markdown(tablefmt='pipe')
    return {"status": "success", "message": f"KMeans executado com {n} clusters. O resumo dos clusters na variável 'amount' e 'time' é:\n\n{summary}"}

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
                "Quando uma pergunta for sobre tipos de dados, estrutura do DataFrame ou contagem, use a ferramenta 'show_descriptive_stats'."
            )
            try:
                st.session_state.agent_executor = initialize_agent(tools_with_df, system_prompt)
                # Limpa figuras na inicialização
                st.session_state.figure_to_display = None
                st.session_state.grafico_para_exibir = None
                st.success("Agente inicializado. Faça suas perguntas na caixa de chat abaixo.")
            except Exception as e:
                st.error(f"Erro ao inicializar agente: {e}")
        else:
            st.error(data['message'])

    if st.session_state.df is not None:
        st.success(f"DataFrame carregado: {len(st.session_state.df)} linhas, {len(st.session_state.df.columns)} colunas")
        st.subheader("Amostra")
        st.dataframe(st.session_state.df.head())

# -----------------------------
# Lógica de Chat
# -----------------------------

# Exibir histórico de mensagens (Garante a persistência)
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        # Se há um Plotly Figure salvo no histórico, exibe-o (Plotly é serializável)
        if 'plotly_figure' in msg:
            st.plotly_chart(msg['plotly_figure'], use_container_width=True)
        # Se há um Matplotlib Figure (improvável, mas mantido por segurança), exibe-o
        if 'grafico_para_exibir' in msg:
             # Nota: Matplotlib figures não devem ser salvas no histórico, mas exibidas.
             st.pyplot(msg['grafico_para_exibir']) 
             
        # Exibe o texto
        st.markdown(msg['content'])


# Input do usuário (inicia novo ciclo)
prompt = st.chat_input("Pergunte algo (ex: 'Gere um histograma de amount' ou 'Quais são os tipos de dados?')")
if prompt:
    # 1. Salva e exibe o prompt do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # 2. Executa o agente
    if st.session_state.agent_executor is None:
        with st.chat_message("assistant"):
            st.error("Agente não inicializado. Faça upload do arquivo e clique em 'Carregar e Inicializar Agente' na barra lateral.")
    else:
        # A nova bolha do assistente
        with st.chat_message("assistant"):
            # O container é importante para garantir a ordem (figura antes do texto)
            resp_container = st.container() 
            final_message = ""
            assistant_response = {"role": "assistant"} # Objeto para salvar no histórico
            
            try:
                # Invoca o agente
                full = st.session_state.agent_executor.invoke({"input": prompt})
                response_content = full.get('output')
                
                # 3. Processa a resposta
                if isinstance(response_content, dict) and response_content.get('status') in ['success','error']:
                    # Resposta de Ferramenta
                    final_message = response_content.get('message', 'Erro ao obter mensagem da ferramenta.')

                    # Figuras geradas são salvas na Session State, exibimos e as movemos para o Histórico
                    if st.session_state.figure_to_display is not None:
                         # Exibe o gráfico Plotly
                        resp_container.plotly_chart(st.session_state.figure_to_display, use_container_width=True)
                        # Salva o objeto Plotly no histórico para persistência
                        assistant_response['plotly_figure'] = st.session_state.figure_to_display
                        st.session_state.figure_to_display = None # Limpa para o próximo ciclo
                        
                    if st.session_state.grafico_para_exibir is not None:
                        # Exibe o gráfico Matplotlib
                        resp_container.pyplot(st.session_state.grafico_para_exibir)
                        # O Matplotlib não é salvo no histórico por ser difícil de serializar
                        st.session_state.grafico_para_exibir = None # Limpa para o próximo ciclo
                        
                else:
                    # Resposta Direta do LLM
                    final_message = str(response_content)

                # 4. Exibe a mensagem de texto FINAL
                if final_message:
                    resp_container.markdown(final_message)
                    assistant_response['content'] = final_message
                    
                    # 5. Salva no histórico (com ou sem figura)
                    st.session_state.messages.append(assistant_response)

            except Exception as e:
                err = f"❌ Erro na execução do agente: {e}"
                resp_container.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})

# Pequena ajuda com as perguntas pedidas no desafio (exemplos de prompts)
with st.expander("Exemplos de perguntas (copie e cole)"):
    st.markdown("- **Quais são os tipos de dados (numéricos, categóricos)?** (Agora deve funcionar)\n- Qual a distribuição de amount?\n- Gere um histograma de v1\n- Gere mapa de calor da correlação\n- Detecte outliers\n- Faça KMeans com 4 clusters\n- Gere um scatter plot entre time e amount")
