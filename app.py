import io
import re
import zipfile

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.memory import ConversationBufferWindowMemory
from langchain.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------------------------------------------
# --- CONFIGURAÇÃO INICIAL ---
# --------------------------------------------------------------------------------------

# --- Configuração da Chave de API do Google ---
try:
    # A chave deve estar configurada nos "Secrets" do Streamlit Cloud
    google_api_key = st.secrets["google_ai"]["google_api_key"]
except KeyError:
    st.error(
        "Chave de API do Google não encontrada. Certifique-se de adicioná-la nos 'Secrets' da sua aplicação."
    )
    st.stop()


# --------------------------------------------------------------------------------------
# --- FERRAMENTAS (TOOLS) PARA O AGENTE ---
# --------------------------------------------------------------------------------------


def show_descriptive_stats(*args):
    """
    Gera estatísticas descritivas para todas as colunas de um DataFrame.
    Retorna um dicionário com o resumo estatístico.
    """
    df: pd.DataFrame = st.session_state.df
    stats = df.describe(include="all").to_markdown()
    return {
        "status": "success",
        "data": stats,
        "message": "Estatísticas descritivas geradas.",
    }


def generate_histogram(column: str, *args):
    """
    Gera um histograma interativo Plotly para uma coluna numérica específica do DataFrame.
    A entrada deve ser o nome da coluna (ex: 'amount', 'v5', 'time').
    """
    df: pd.DataFrame = st.session_state.df
    column = column.lower()

    if column not in df.columns:
        return {
            "status": "error",
            "message": f"Erro: A coluna '{column}' não foi encontrada no DataFrame. Por favor, verifique se o nome está correto.",
        }
    if not pd.api.types.is_numeric_dtype(df[column]):
        return {
            "status": "error",
            "message": f"Erro: A coluna '{column}' não é numérica. Forneça uma coluna numérica para gerar um histograma.",
        }

    # Usando Plotly Express
    fig = px.histogram(df, x=column, title=f"Distribuição de {column}")
    st.session_state.figure_to_display = fig  # Salva a figura na sessão
    return {
        "status": "success",
        "message": f"O histograma da coluna '{column}' foi gerado com sucesso. Analise a distribuição dos dados e procure por assimetrias ou picos.",
    }


def generate_correlation_heatmap(*args):
    """
    Calcula a matriz de correlação entre as variáveis numéricas do DataFrame
    e gera um mapa de calor (heatmap) interativo Plotly.
    """
    df: pd.DataFrame = st.session_state.df
    numeric_cols = df.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        return {
            "status": "error",
            "message": "Erro: O DataFrame não tem colunas numéricas suficientes para calcular a correlação.",
        }

    correlation_matrix = df[numeric_cols].corr()

    # Usando Plotly Express
    fig = px.imshow(
        correlation_matrix,
        text_auto=".2f",
        aspect="auto",
        title="Mapa de Calor da Matriz de Correlação",
        color_continuous_scale="RdBu_r",
    )
    fig.update_xaxes(side="top")
    st.session_state.figure_to_display = fig  # Salva a figura na sessão
    return {
        "status": "success",
        "message": "O mapa de calor da correlação interativo foi gerado. Analise o padrão de cores para identificar relações fortes (vermelho/azul escuro) ou fracas (cinza claro).",
    }


def generate_scatter_plot(columns_str: str, *args):
    """
    Gera um gráfico de dispersão (scatter plot) interativo Plotly para visualizar
    a relação entre duas colunas numéricas.
    A entrada DEVE ser uma string contendo os nomes das duas colunas SEPARADAS por um espaço,
    vírgula ou 'e' (ex: 'time, amount' ou 'v1 e v2').
    """
    df: pd.DataFrame = st.session_state.df

    col_names = re.split(r"[,\s]+", columns_str.lower())
    col_names = [col for col in col_names if col and col != "e"]

    if len(col_names) < 2:
        return {
            "status": "error",
            "message": f"Erro de Argumentos: O agente precisa de pelo menos DOIS nomes de coluna para o gráfico de dispersão. Foi encontrado apenas: {col_names}",
        }

    x_col = col_names[0]
    y_col = col_names[1]

    if x_col not in df.columns or y_col not in df.columns:
        return {
            "status": "error",
            "message": f"Erro: Uma ou ambas as colunas ('{x_col}', '{y_col}') não existem no DataFrame.",
        }

    # Usando Plotly Express
    fig = px.scatter(
        df, x=x_col, y=y_col, title=f"Gráfico de Dispersão: {x_col} vs {y_col}"
    )
    st.session_state.figure_to_display = fig  # Salva a figura na sessão
    return {
        "status": "success",
        "message": f"O gráfico de dispersão interativo para '{x_col}' vs '{y_col}' foi gerado. Use-o para visualizar a forma e a densidade da relação entre essas variáveis.",
    }


def detect_outliers_isolation_forest(*args):
    """
    Detecta anomalias (outliers) no DataFrame usando o algoritmo Isolation Forest.
    A análise é aplicada às colunas V1 a V28, 'time' e 'amount'.
    Retorna o número de anomalias detectadas e uma amostra dos outliers.
    """
    try:
        df: pd.DataFrame = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith("v")] + [
            "time",
            "amount",
        ]

        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
            return {
                "status": "error",
                "message": "Erro ao detectar anomalias: Não foram encontradas colunas V*, 'time' ou 'amount' no DataFrame.",
            }

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)
        model = IsolationForest(contamination=0.01, random_state=42)
        df["anomaly_score"] = model.fit_predict(df_scaled)
        outliers = df[df["anomaly_score"] == -1]

        message = f"O algoritmo Isolation Forest detectou {len(outliers)} transações atípicas (outliers)."
        if not outliers.empty:
            message += (
                "\nAmostra das transações detectadas como anomalias:\n"
                + outliers.head().to_markdown(tablefmt="pipe")
            )

        return {"status": "success", "message": message}
    except Exception as e:
        return {"status": "error", "message": f"Erro ao detectar anomalias: {e}"}


def find_clusters_kmeans(n_clusters: str, *args):
    """
    Realiza agrupamento (clustering) nos dados usando o algoritmo K-Means.
    A análise é aplicada às colunas V1 a V28, 'time' e 'amount'.
    A entrada DEVE ser o número de clusters desejado (como string, ex: "5").
    Retorna uma descrição dos clusters encontrados.
    """
    try:
        n_clusters = int(n_clusters)
    except ValueError:
        return {
            "status": "error",
            "message": f"O número de clusters deve ser um número inteiro, mas o valor recebido foi '{n_clusters}'.",
        }

    try:
        df: pd.DataFrame = st.session_state.df
        feature_cols = [col for col in df.columns if col.startswith("v")] + [
            "time",
            "amount",
        ]

        existing_features = [col for col in feature_cols if col in df.columns]
        if not existing_features:
            return {
                "status": "error",
                "message": "Erro ao encontrar clusters: Não foram encontradas colunas V*, 'time' ou 'amount' no DataFrame.",
            }

        df_features = df[existing_features]
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_features)

        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
        df["cluster"] = kmeans.fit_predict(df_scaled)

        cluster_summary = (
            df.groupby("cluster")
            .agg({"amount": ["mean", "min", "max"], "time": ["min", "max"]})
            .to_markdown(tablefmt="pipe")
        )

        message = f"O agrupamento K-Means com {n_clusters} clusters foi concluído."
        message += "\nCaracterísticas dos Clusters:\n" + cluster_summary

        return {"status": "success", "message": message}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Erro ao realizar o agrupamento com K-Means: {e}",
        }


# --------------------------------------------------------------------------------------
# --- FUNÇÕES DE CARREGAMENTO DE DADOS E AGENTE ---
# --------------------------------------------------------------------------------------


# Otimização de Cache: A função é decorada com st.cache_data para evitar reprocessar o mesmo arquivo.
# Passamos os bytes do arquivo (file_bytes) em vez do objeto UploadedFile, pois os bytes são "hashable"
# e garantem que o cache funcione corretamente.
@st.cache_data(show_spinner="Processando arquivo...")
def load_and_extract_data(file_bytes, filename):
    """Carrega e prepara o DataFrame a partir de um arquivo CSV ou ZIP."""
    if file_bytes is None:
        return {"status": "error", "message": "Bytes do arquivo não recebidos."}
    uploaded_file = io.BytesIO(file_bytes)

    try:
        if filename.endswith(".zip"):
            with zipfile.ZipFile(uploaded_file, "r") as z:
                # Assume que o CSV é o primeiro arquivo dentro do ZIP
                with z.open(z.namelist()[0]) as f:
                    df = pd.read_csv(f)
        elif filename.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            return {
                "status": "error",
                "message": "Formato de arquivo não suportado. Por favor, envie um arquivo ZIP ou CSV.",
            }

        # Padroniza nomes de colunas para minúsculas
        df.columns = [col.lower() for col in df.columns]

        return {
            "status": "success",
            "df": df,
            "message": f"Arquivo '{filename}' carregado com sucesso. DataFrame pronto para análise.",
        }

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar o arquivo: {e}"}


def initialize_agent(tools_list, system_prompt_text):
    """Inicializa e configura o LangChain Agent com o modelo Gemini Pro."""

    # Modelo alterado para Gemini 2.5 Flash, que é mais rápido e tem cota gratuita maior que o Pro.
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",  # MUDANÇA: Usando Flash para evitar o erro 429 de cota (mais rápido e maior limite)
        google_api_key=google_api_key,
        temperature=0.0,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt_text),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True
    )

    agent = create_tool_calling_agent(llm, tools_list, prompt)

    # Cria o executor do agente
    agent_executor = AgentExecutor(
        agent=agent, tools=tools_list, verbose=True, memory=memory, max_iterations=15
    )
    return agent_executor


# --------------------------------------------------------------------------------------
# --- INTERFACE DO STREAMLIT ---
# --------------------------------------------------------------------------------------

st.set_page_config(
    page_title="Agente de Análise de Dados (Gemini Flash)", layout="wide"
)

st.title("🤖 Agente de Análise de Dados (EDA) com Gemini Flash")
st.markdown(
    "Envie um arquivo CSV (ou ZIP com CSV) e pergunte ao agente para realizar análises, como correlação, estatísticas descritivas ou detecção de anomalias."
)

# Inicializa o estado da sessão
if "messages" not in st.session_state:
    st.session_state.messages = []
if "df" not in st.session_state:
    st.session_state.df = None
if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = None
# Variável de sessão para armazenar a figura a ser exibida
if "figure_to_display" not in st.session_state:
    st.session_state.figure_to_display = None

# Sidebar para upload de arquivo
with st.sidebar:
    st.header("Upload do Arquivo de Dados")
    uploaded_file = st.file_uploader(
        "Escolha um arquivo CSV ou ZIP", type=["csv", "zip"]
    )

    if st.button("Carregar Dados e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando e preparando dados..."):
            # Passa os bytes do arquivo para a função cacheada
            file_bytes = uploaded_file.getvalue()
            load_result = load_and_extract_data(file_bytes, uploaded_file.name)

        if load_result["status"] == "success":
            st.session_state.df = load_result["df"]

            # Lista de funções a serem convertidas em ferramentas
            tool_functions = [
                show_descriptive_stats,
                generate_histogram,
                generate_correlation_heatmap,
                generate_scatter_plot,
                detect_outliers_isolation_forest,
                find_clusters_kmeans,
            ]

            # Cria a lista de ferramentas LangChain de forma programática
            tools_with_df = [
                Tool(name=func.__name__, description=func.__doc__, func=func)
                for func in tool_functions
            ]

            system_prompt = (
                "Você é um agente de Análise Exploratória de Dados (EDA) altamente proficiente. "
                "Sua *PRIMEIRA PRIORIDADE* é sempre tentar responder à pergunta do usuário usando uma das ferramentas disponíveis. "
                "*SEMPRE* que o usuário solicitar uma análise de dados (ex: 'correlação', 'distribuição', 'relação', 'gráfico'), você *DEVE* selecionar a ferramenta apropriada e executá-la. "
                "Quando uma ferramenta de gráfico é usada, a figura é salva e exibida na interface. Você só precisa fornecer os insights em texto. "
                "Sua resposta final deve sempre ser em Português e oferecer insights."
            )

            st.session_state.agent_executor = initialize_agent(
                tools_with_df, system_prompt
            )
            st.success(
                "Dados carregados e agente inicializado! Você pode começar a perguntar."
            )

        else:
            st.error(load_result["message"])

    if st.session_state.df is not None:
        st.success(
            f"DataFrame carregado com {len(st.session_state.df)} linhas e {len(st.session_state.df.columns)} colunas."
        )
        st.subheader("Visualização dos Dados (Amostra)")
        st.dataframe(st.session_state.df.head())


# --- EXIBIÇÃO DE MENSAGENS E GRÁFICOS ---

# Exibir histórico de mensagens
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        # CORREÇÃO: Renderiza apenas strings. Evita renderizar DataFrames diretamente do histórico,
        # o que causa o loop infinito de reruns. A exibição de tabelas e gráficos
        # acontece apenas uma vez, no momento da resposta do agente.
        st.markdown(message["content"])

# Tratamento de entrada do usuário
if prompt_input := st.chat_input(
    "Qual análise você gostaria de fazer? (Ex: 'Gere um mapa de calor da correlação')"
):
    with st.chat_message("user"):
        st.markdown(prompt_input)
    st.session_state.messages.append({"role": "user", "content": prompt_input})

    if st.session_state.agent_executor is not None:
        with st.chat_message("assistant"):
            response_container = st.container()

            try:
                full_response = st.session_state.agent_executor.invoke(
                    {"input": prompt_input}
                )
                response_content = full_response["output"]

                # 1. Verifica se há um gráfico na sessão para exibir
                if st.session_state.figure_to_display is not None:
                    response_container.plotly_chart(
                        st.session_state.figure_to_display, use_container_width=True
                    )
                    st.session_state.figure_to_display = (
                        None  # Limpa a sessão após exibir
                    )

                # Se a resposta for um dicionário, a ferramenta foi usada.
                if isinstance(response_content, dict) and response_content.get(
                    "status"
                ) in ["success", "error"]:
                    # 2. Exibe a mensagem de texto (status, insights, etc.)
                    if "message" in response_content:
                        # Trata mensagens de erro de forma diferente
                        if response_content.get("status") == "error":
                            response_container.error(response_content["message"])
                        else:
                            response_container.markdown(response_content["message"])
                        st.session_state.messages.append(
                            {
                                "role": "assistant",
                                "content": response_content["message"],
                            }
                        )

                    # 3. Exibe dados tabulares, se existirem
                    if "data" in response_content:
                        # SIMPLIFICAÇÃO: A ferramenta já retorna uma string formatada como Markdown.
                        # Exibimos essa string diretamente em vez de convertê-la de volta para um DataFrame.
                        # Isso é mais simples e evita problemas de formatação.
                        table_markdown = response_content["data"]
                        response_container.markdown(table_markdown)
                        # Salva a mesma string no histórico para consistência.
                        st.session_state.messages.append(
                            {"role": "assistant", "content": table_markdown}
                        )
                else:
                    # Se for uma resposta de texto simples do LLM (sem usar ferramenta)
                    response_container.markdown(str(response_content))
                    st.session_state.messages.append(
                        {"role": "assistant", "content": str(response_content)}
                    )

            except Exception as e:
                # O erro 429 (Quota Exceeded) não é mais comum com o Gemini Flash, mas mantemos o tratamento.
                error_message = f"Desculpe, ocorreu um erro inesperado na análise: {e}. Por favor, recarregue a página ou simplifique sua última pergunta."
                response_container.error(error_message)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_message}
                )
