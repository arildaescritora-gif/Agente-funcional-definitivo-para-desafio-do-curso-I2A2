# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import zipfile
import io
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re 
import plotly.express as px 
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferWindowMemory
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate 

# --------------------------------------------------------------------------------------
# --- CONFIGURAÇÃO INICIAL ---
# --------------------------------------------------------------------------------------

# --- Configuração da Chave de API do Google ---
try:
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error("Chave de API do Google não encontrada. Certifique-se de adicioná-la nos 'Secrets' da sua aplicação.")
    st.stop()

# --------------------------------------------------------------------------------------
# --- FERRAMENTAS (TOOLS) PARA O AGENTE ---
# --------------------------------------------------------------------------------------

def show_descriptive_stats(*args):
    """Gera estatísticas descritivas para todas as colunas do DataFrame."""
    df = st.session_state.df
    stats = df.describe(include='all').to_markdown(tablefmt="pipe")
    return {"status": "success", "data": stats, "message": "Estatísticas descritivas geradas com sucesso."}

def generate_histogram(column: str, *args):
    """Gera um histograma interativo para uma coluna numérica específica."""
    df = st.session_state.df
    column = column.lower()
    
    if column not in df.columns:
        return {"status": "error", "message": f"Erro: Coluna '{column}' não encontrada."}
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {"status": "error", "message": f"Erro: Coluna '{column}' não é numérica."}
    
    fig = px.histogram(df, x=column, title=f'Distribuição de {column}')
    return {"status": "success", "plotly_figure": fig, "message": f"Histograma da coluna '{column}' gerado com sucesso."}

def generate_correlation_heatmap(*args):
    """Gera mapa de calor da correlação entre variáveis numéricas."""
    df = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Erro: Não há colunas numéricas suficientes."}
    
    correlation_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title='Mapa de Calor da Correlação',
        color_continuous_scale='RdBu_r'
    )
    fig.update_xaxes(side="top")
    return {"status": "success", "plotly_figure": fig, "message": "Mapa de calor de correlação gerado com sucesso."}

def generate_scatter_plot(columns_str: str, *args):
    """Gera gráfico de dispersão entre duas colunas numéricas."""
    df = st.session_state.df
    
    col_names = re.split(r'[,\s]+', columns_str.lower())
    col_names = [col for col in col_names if col and col != 'e'] 
    
    if len(col_names) < 2:
        return {"status": "error", "message": "Erro: Forneça duas colunas para o gráfico de dispersão."}

    x_col, y_col = col_names[0], col_names[1]

    if x_col not in df.columns or y_col not in df.columns:
        return {"status": "error", "message": f"Erro: Coluna(s) não encontrada(s)."}

    fig = px.scatter(df, x=x_col, y=y_col, title=f'Dispersão: {x_col} vs {y_col}')
    return {"status": "success", "plotly_figure": fig, "message": f"Gráfico de dispersão gerado com sucesso."}

def detect_outliers_isolation_forest(*args):
    """Detecta anomalias usando Isolation Forest."""
    try:
        df = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        existing_features = [col for col in feature_cols if col in df.columns]
        
        if not existing_features:
            return {"status": "error", "message": "Erro: Colunas V*, 'time' ou 'amount' não encontradas."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        model = IsolationForest(contamination=0.01, random_state=42)
        df['anomaly_score'] = model.fit_predict(df_scaled)
        outliers = df[df['anomaly_score'] == -1]
        
        message = f"Isolation Forest detectou {len(outliers)} outliers."
        if not outliers.empty:
            message += "\nAmostra dos outliers:\n" + outliers.head().to_markdown(tablefmt="pipe")
            
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao detectar anomalias: {e}"}

def find_clusters_kmeans(n_clusters: str, *args):
    """Realiza agrupamento usando K-Means."""
    try:
        n_clusters = int(n_clusters)
    except ValueError:
        return {"status": "error", "message": "Erro: Número de clusters deve ser inteiro."}

    try:
        df = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith('v')] + ['time', 'amount']
        existing_features = [col for col in feature_cols if col in df.columns]
        
        if not existing_features:
            return {"status": "error", "message": "Erro: Colunas V*, 'time' ou 'amount' não encontradas."}

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        df['cluster'] = kmeans.fit_predict(df_scaled)
        
        cluster_summary = df.groupby('cluster').agg({
            'amount': ['mean', 'min', 'max'],
            'time': ['min', 'max']
        }).to_markdown(tablefmt="pipe")
        
        message = f"K-Means com {n_clusters} clusters concluído.\nResumo:\n" + cluster_summary
        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro no K-Means: {e}"}

# --------------------------------------------------------------------------------------
# --- FUNÇÕES DE CARREGAMENTO DE DADOS E AGENTE ---
# --------------------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_and_extract_data(uploaded_file):
    """Carrega e prepara o DataFrame a partir de um arquivo CSV ou ZIP."""
    if uploaded_file is None:
        return {"status": "error", "message": "Nenhum arquivo enviado."}

    try:
        if uploaded_file.name.endswith('.zip'):
            with zipfile.ZipFile(uploaded_file, 'r') as z:
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        elif uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            return {"status": "error", "message": "Formato não suportado. Use CSV ou ZIP."}

        df.columns = [col.lower() for col in df.columns]
        return {"status": "success", "df": df, "message": f"Arquivo '{uploaded_file.name}' carregado com sucesso."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar arquivo: {e}"}

def initialize_agent(tools_list, system_prompt_text):
    """Inicializa e configura o LangChain Agent."""
    
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Usando Flash para melhor performance
        google_api_key=google_api_key,
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferWindowMemory(k=5, memory_key="chat_history", return_messages=True)
    agent = create_tool_calling_agent(llm, tools_list, prompt)

    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools_list,
        verbose=True,
        memory=memory,
        max_iterations=10,  # Reduzido para melhor performance
        handle_parsing_errors=True
    )
    return agent_executor

# --------------------------------------------------------------------------------------
# --- INTERFACE DO STREAMLIT ---
# --------------------------------------------------------------------------------------

st.set_page_config(page_title="Agente de Análise de Dados", layout="wide")
st.title("🤖 Agente de Análise de Dados (EDA)")
st.markdown("Envie um arquivo CSV/ZIP e faça perguntas sobre análise exploratória de dados.")

# Inicializa o estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
if "current_figure" not in st.session_state:
    st.session_state.current_figure = None

# Sidebar para upload
with st.sidebar:
    st.header("📁 Upload de Dados")
    uploaded_file = st.file_uploader("Escolha CSV ou ZIP", type=["csv", "zip"])
    
    if st.button("🚀 Carregar Dados e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando dados..."):
            load_result = load_and_extract_data(uploaded_file)

        if load_result["status"] == "success":
            st.session_state.df = load_result["df"]
            
            # Criar ferramentas
            tools = [
                Tool(name=show_descriptive_stats.__name__, description=show_descriptive_stats.__doc__, func=show_descriptive_stats),
                Tool(name=generate_histogram.__name__, description=generate_histogram.__doc__, func=generate_histogram),
                Tool(name=generate_correlation_heatmap.__name__, description=generate_correlation_heatmap.__doc__, func=generate_correlation_heatmap),
                Tool(name=generate_scatter_plot.__name__, description=generate_scatter_plot.__doc__, func=generate_scatter_plot),
                Tool(name=detect_outliers_isolation_forest.__name__, description=detect_outliers_isolation_forest.__doc__, func=detect_outliers_isolation_forest),
                Tool(name=find_clusters_kmeans.__name__, description=find_clusters_kmeans.__doc__, func=find_clusters_kmeans)
            ]

            system_prompt = """Você é um especialista em Análise Exploratória de Dados. 
SEMPRE use as ferramentas disponíveis para responder às perguntas do usuário.
Priorize gerar visualizações quando solicitado.
Responda em português com insights claros e práticos.
As colunas estão em minúsculas: 'v1'-'v28', 'time', 'amount'."""

            st.session_state.agent_executor = initialize_agent(tools, system_prompt)
            st.success("✅ Dados carregados e agente inicializado!")

        else:
            st.error(load_result["message"])

    if st.session_state.df is not None:
        st.info(f"📊 Dataset: {len(st.session_state.df)} linhas × {len(st.session_state.df.columns)} colunas")
        if st.checkbox("🔍 Ver amostra dos dados"):
            st.dataframe(st.session_state.df.head(5))

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], pd.DataFrame):
            st.dataframe(message["content"])
        else:
            st.markdown(message["content"])

# Exibir gráfico atual se existir
if st.session_state.current_figure is not None:
    st.plotly_chart(st.session_state.current_figure, use_container_width=True)
    st.session_state.current_figure = None

# Input do usuário
if prompt_input := st.chat_input("Faça sua pergunta sobre os dados..."):
    
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})
    
    if st.session_state.agent_executor is not None:
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                response = st.session_state.agent_executor.invoke({"input": prompt_input})
                response_content = response["output"]

                # Processar resposta
                if isinstance(response_content, dict) and response_content.get("status") in ["success", "error"]:
                    
                    # Exibir gráfico se existir
                    if "plotly_figure" in response_content:
                        st.session_state.current_figure = response_content["plotly_figure"]
                    
                    # Exibir mensagem
                    if "message" in response_content:
                        response_placeholder.markdown(response_content["message"])
                        st.session_state.messages.append({"role": "assistant", "content": response_content["message"]})
                    
                    # Exibir dados tabulares
                    if "data" in response_content:
                        st.markdown("**Estatísticas Descritivas:**")
                        st.markdown(response_content["data"])
                        st.session_state.messages.append({"role": "assistant", "content": response_content["data"]})
                        
                else:
                    # Resposta direta do LLM
                    response_placeholder.markdown(str(response_content))
                    st.session_state.messages.append({"role": "assistant", "content": str(response_content)})

            except Exception as e:
                error_msg = f"❌ Erro na análise: {str(e)}"
                response_placeholder.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.warning("⚠️ Por favor, carregue os dados primeiro usando o botão na sidebar.")

# Adicionar exemplos de perguntas
with st.expander("💡 Exemplos de perguntas para fazer"):
    st.markdown("""
    **Análise Descritiva:**
    - "Quais são os tipos de dados das colunas?"
    - "Mostre estatísticas descritivas do dataset"
    - "Qual a distribuição da coluna 'amount'?"
    
    **Visualizações:**
    - "Gere um histograma para 'time'"
    - "Mostre a correlação entre as variáveis"
    - "Crie um gráfico de dispersão entre v1 e amount"
    
    **Análise Avançada:**
    - "Detecte outliers nos dados"
    - "Agrupe os dados em 3 clusters"
    - "Analise a relação entre time e amount"
    """)
