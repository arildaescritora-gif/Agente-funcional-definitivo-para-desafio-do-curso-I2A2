# ... (código antes do "if prompt") ...

# Input do usuário
prompt = st.chat_input("Pergunte algo sobre os dados...")
if prompt:
    # Adicionar prompt do usuário ao histórico
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
                
                # O LangChain Tool Calling Agent (Executor) deve retornar a resposta da ferramenta se ela for chamada.
                
                # Tenta extrair diretamente se o output for uma resposta da ferramenta (dicionário)
                if isinstance(response_content, dict) and response_content.get('status') in ['success', 'error']:
                    
                    # Adicionar gráficos Plotly (interativos e persistentes)
                    if 'plotly_figure' in response_content:
                        # LINHA REMOVIDA: st.plotly_chart(response_content['plotly_figure'], use_container_width=True)
                        # Salva a figura Plotly no histórico (serializável pelo Streamlit)
                        assistant_response['plotly_figure'] = response_content['plotly_figure']
                    
                    # Adicionar gráficos Matplotlib (apenas exibição imediata, não persistirá)
                    if 'matplotlib_figure' in response_content:
                        # LINHA REMOVIDA: st.pyplot(response_content['matplotlib_figure'])
                        pass # A figura Matplotlib não é salva no histórico para evitar erros de serialização

                    # Adicionar mensagem de texto
                    message_text = response_content.get('message', 'Erro desconhecido.')
                    if response_content.get('status') == 'error':
                        st.error(message_text)
                    else:
                        st.markdown(message_text)
                    
                    assistant_response['message'] = message_text
                
                else:
                    # Resposta direta do LLM (após Tool call ou se nenhuma ferramenta foi usada)
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

# ... (restante do código) ...
