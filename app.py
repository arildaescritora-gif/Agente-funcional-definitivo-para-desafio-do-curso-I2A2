# -*- coding: utf-8 -*-

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
st.title("🤖 Agente de Análise de Dados (EDA) - Versão Corrigida")
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
                csv_names = [n for n in z.namelist() if n.lower().endswith('.csv')]
                if not csv_names:
                    return {"status": "error", "message": "ZIP não contém CSV."}
                with z.open(csv_names[0]) as f:
                    df = pd.read_csv(f)
        elif filename.lower().endswith(".csv"):
            df = pd.read_csv(bio)
        else:
            return {"status": "error", "message": "Formato não suportado. Envie CSV ou ZIP."}

        df.columns = [c.lower() for c in df.columns]
        return {"status": "success", "df": df, "message": f"Arquivo {filename} carregado."}

    except Exception as e:
        return {"status": "error", "message": f"Erro ao processar arquivo: {e}"}

# -----------------------------
# Ferramentas / Tools do agente - CORRIGIDAS
# -----------------------------

def show_descriptive_stats(*args):
    """Gera estatísticas descritivas (tipos, contagem, média, mediana, std, min, max)"""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}

    try:
        # Tipos de dados
        types_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            null_count = df[col].isnull().sum()
            types_info.append(f"- **{col}**: {dtype} (Valores únicos: {unique_count}, Nulos: {null_count})")
        
        types_output = "**Tipos de dados:**\n" + "\n".join(types_info)
        
        # Estatísticas numéricas
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            stats_output = "\n\n**Estatísticas descritivas (variáveis numéricas):**\n\n"
            stats_output += df[numeric_cols].describe().to_markdown(tablefmt="pipe")
        else:
            stats_output = "\n\n**Não há variáveis numéricas para estatísticas descritivas.**"
        
        # Valores mais frequentes
        freq_output = "\n\n**Valores mais frequentes (top 3 por coluna):**\n"
        for col in df.columns:
            if df[col].dtype == 'object' or df[col].nunique() < 20:
                top_values = df[col].value_counts().head(3)
                freq_output += f"\n**{col}**: " + ", ".join([f"'{k}': {v}" for k, v in top_values.items()])
            else:
                freq_output += f"\n**{col}**: (muitos valores únicos)"
        
        full_output = types_output + stats_output + freq_output
        return {"status": "success", "message": full_output}
        
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar estatísticas: {e}"}

def generate_histogram(column: str, *args):
    """Gera histograma Plotly para coluna numérica"""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}
    
    col = column.lower().strip()
    if col not in df.columns:
        return {"status": "error", "message": f"Coluna '{col}' não encontrada."}
    if not pd.api.types.is_numeric_dtype(df[col]):
        return {"status": "error", "message": f"Coluna '{col}' não é numérica."}

    try:
        # Plotly
        fig = px.histogram(df, x=col, title=f"Histograma de {col}")
        
        # Matplotlib fallback
        fig_mpl, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df[col].dropna(), bins=30, alpha=0.7)
        ax.set_title(f"Histograma de {col}")
        ax.set_xlabel(col)
        ax.set_ylabel("Frequência")
        plt.tight_layout()

        return {
            "status": "success", 
            "message": f"Histograma gerado para '{col}'.",
            "plotly_figure": fig,
            "matplotlib_figure": fig_mpl
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar histograma: {e}"}

def generate_correlation_heatmap(*args):
    """Gera mapa de calor da correlação entre colunas numéricas"""
    df = st.session_state.df
    if df is None:
        return {"status": "error", "message": "Nenhum DataFrame carregado."}

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) < 2:
        return {"status": "error", "message": "Não há colunas numéricas suficientes."}

    try:
        corr = df[numeric_cols].corr()
        fig = px.imshow(corr, text_auto='.2f', aspect='auto', 
                       title='Matriz de Correlação', color_continuous_scale='RdBu_r')
        fig.update_xaxes(side='top')

        # Matplotlib fallback
        fig_mpl, ax = plt.subplots(figsize=(12, 8))
        im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
        ax.set_xticks(range(len(numeric_cols)))
        ax.set_yticks(range(len(numeric_cols)))
        ax.set_xticklabels(numeric_cols, rotation=45, ha='right')
        ax.set_yticklabels(numeric_cols)
        ax.set_title('Matriz de Correlação')
        
        # Adicionar valores na matriz
        for i in range(len(numeric_cols)):
            for j in range(len(numeric_cols)):
                text = ax.text(j, i, f'{corr.iloc[i, j]:.2f}',
                              ha="center", va="center", color="black", fontsize=8)
        
        plt.colorbar(im, ax=ax)
        plt.tight_layout()

        return {
            "status": "success", 
            "message": "Mapa de calor da correlação gerado.",
            "plotly_figure": fig,
            "matplotlib_figure": fig_mpl
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar mapa de calor: {e}"}

def generate_scatter_plot(columns_str: str, *args):
    """Gera scatter Plotly entre duas colunas"""
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

    try:
        fig = px.scatter(df, x=x_col, y=y_col, title=f"{x_col} vs {y_col}")

        # Matplotlib fallback
        fig_mpl, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df[x_col], df[y_col], alpha=0.6)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        ax.set_title(f"{x_col} vs {y_col}")
        plt.tight_layout()

        return {
            "status": "success", 
            "message": f"Gráfico de dispersão gerado para '{x_col}' vs '{y_col}'.",
            "plotly_figure": fig,
            "matplotlib_figure": fig_mpl
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro ao gerar scatter plot: {e}"}

def detect_outliers_isolation_forest(*args):
    """Detecta outliers com IsolationForest nas colunas v* + time + amount"""
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
        
        out_idx = X.index[labels == -1]
        sample = df.loc[out_idx].head(10)
        
        msg = f"Foram detectados {len(out_idx)} outliers (amostra de 10 linhas):\n\n"
        msg += sample.to_markdown(tablefmt="pipe")
        
        return {"status": "success", "message": msg}
    except Exception as e:
        return {"status": "error", "message": f"Erro: {e}"}

def find_clusters_kmeans(n_clusters: str, *args):
    """Agrupamento KMeans nas colunas v* + time + amount"""
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

    try:
        X = df[existing].fillna(0)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        km = KMeans(n_clusters=n, random_state=42, n_init='auto')
        labels = km.fit_predict(Xs)
        
        df_copy = df.copy()
        df_copy['cluster'] = labels
        summary = df_copy.groupby('cluster').agg({
            'amount': ['mean','min','max'], 
            'time': ['min','max']
        }).to_markdown(tablefmt='pipe')
        
        return {
            "status": "success", 
            "message": f"KMeans executado com {n} clusters.\n\n{summary}"
        }
    except Exception as e:
        return {"status": "error", "message": f"Erro no K-Means: {e}"}

# Lista de ferramentas
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
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",  # Usando flash para melhor performance
        google_api_key=google_api_key, 
        temperature=0.0
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt_text),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])

    memory = ConversationBufferWindowMemory(
        k=5, 
        memory_key="chat_history", 
        return_messages=True
    )
    
    agent = create_tool_calling_agent(llm, tools_list, prompt)

    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools_list, 
        verbose=False, 
        memory=memory, 
        max_iterations=10,
        handle_parsing_errors=True
    )
    return agent_executor

# -----------------------------
# Interface do Streamlit - CORRIGIDA
# -----------------------------
with st.sidebar:
    st.header("📁 Upload de dados")
    uploaded_file = st.file_uploader("CSV ou ZIP (com CSV)", type=["csv","zip"])

    if st.button("🚀 Carregar e Inicializar Agente") and uploaded_file is not None:
        with st.spinner("Carregando arquivo..."):
            data = load_and_extract_data(uploaded_file.getvalue(), uploaded_file.name)
        
        if data['status'] == 'success':
            st.session_state.df = data['df']

            # Cria tools
            tools_with_df = [
                Tool(
                    name=fn.__name__, 
                    description=fn.__doc__, 
                    func=fn
                ) for fn in tool_functions
            ]

            system_prompt = (
                "Você é um agente de EDA especializado. SEMPRE use as ferramentas disponíveis para responder às perguntas. "
                "Quando o usuário pedir gráficos ou análises, execute a ferramenta adequada. "
                "Responda em Português com insights claros. "
                "As colunas estão em minúsculas: v1-v28, time, amount, class."
            )
            
            try:
                st.session_state.agent_executor = initialize_agent(tools_with_df, system_prompt)
                st.success("✅ Agente inicializado! Faça suas perguntas no chat.")
            except Exception as e:
                st.error(f"Erro ao inicializar agente: {e}")
        else:
            st.error(data['message'])

    if st.session_state.df is not None:
        st.success(f"📊 DataFrame: {len(st.session_state.df)} linhas, {len(st.session_state.df.columns)} colunas")
        if st.checkbox("👀 Ver amostra dos dados"):
            st.dataframe(st.session_state.df.head())

# Exibir histórico de mensagens
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        # Se a mensagem contém figuras, exibe-as
        if 'plotly_figure' in msg:
            st.plotly_chart(msg['plotly_figure'], use_container_width=True)
        if 'matplotlib_figure' in msg:
            st.pyplot(msg['matplotlib_figure'])
        if 'message' in msg:
            st.markdown(msg['message'])

# Input do usuário
prompt = st.chat_input("Pergunte algo sobre os dados...")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)

    if st.session_state.agent_executor is None:
        with st.chat_message("assistant"):
            st.error("❌ Agente não inicializado. Faça upload do arquivo e clique em 'Carregar e Inicializar Agente'.")
    else:
        with st.chat_message("assistant"):
            try:
                full_response = st.session_state.agent_executor.invoke({"input": prompt})
                response_content = full_response.get('output', '')

                # Preparar resposta para salvar no histórico
                assistant_response = {"role": "assistant"}

                # Processar resposta baseada no tipo
                if isinstance(response_content, dict) and response_content.get('status') in ['success', 'error']:
                    
                    # Adicionar gráficos Plotly se existirem
                    if 'plotly_figure' in response_content:
                        st.plotly_chart(response_content['plotly_figure'], use_container_width=True)
                        assistant_response['plotly_figure'] = response_content['plotly_figure']
                    
                    # Adicionar gráficos Matplotlib se existirem
                    if 'matplotlib_figure' in response_content:
                        st.pyplot(response_content['matplotlib_figure'])
                        assistant_response['matplotlib_figure'] = response_content['matplotlib_figure']
                    
                    # Adicionar mensagem de texto
                    if 'message' in response_content:
                        if response_content.get('status') == 'error':
                            st.error(response_content['message'])
                        else:
                            st.markdown(response_content['message'])
                        assistant_response['message'] = response_content['message']
                        
                else:
                    # Resposta direta do LLM
                    st.markdown(str(response_content))
                    assistant_response['message'] = str(response_content)

                # Salvar resposta no histórico
                st.session_state.messages.append(assistant_response)

            except Exception as e:
                error_msg = f"❌ Erro na execução: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "message": error_msg
                })

# Exemplos de perguntas
with st.expander("💡 Exemplos de perguntas para testar"):
    st.markdown("""
    **Perguntas do desafio I2A2:**
    
    - *Quais são os tipos de dados (numéricos, categóricos)?*
    - *Qual a distribuição de cada variável?* (peça histogramas específicos)
    - *Gere um histograma de amount*
    - *Qual o intervalo de cada variável (mínimo, máximo)?*
    - *Quais são as medidas de tendência central?*
    - *Qual a variabilidade dos dados?*
    - *Gere mapa de calor da correlação*
    - *Existem padrões ou tendências temporais?*
    - *Quais os valores mais frequentes?*
    - *Existem agrupamentos nos dados?* (peça KMeans com 2, 3, 4 clusters)
    - *Detecte outliers nos dados*
    - *Como as variáveis se relacionam?* (scatter plots)
    - *Gere um scatter plot entre time e amount*
    """)
