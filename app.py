# Chat input e exibição de respostas
# REMOVER (ou deixar) A EXIBIÇÃO DO HISTÓRICO AQUI É O PONTO CHAVE.

# Exibir histórico de mensagens ANTES do chat_input, SEMPRE
for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        # Se a figura Plotly foi salva no histórico, exibe-a
        if 'plotly_figure' in msg:
            st.plotly_chart(msg['plotly_figure'], use_container_width=True)
        # Se a figura Matplotlib foi salva no histórico (pode ser o caso do seu código), exibe-a
        if 'grafico_para_exibir' in msg:
            st.pyplot(msg['grafico_para_exibir'])
            
        st.markdown(msg['content'])
        
# Onde começa a nova interação:
prompt = st.chat_input("Pergunte algo (ex: 'Gere um histograma de amount' ou 'Quais são os tipos de dados?')")
if prompt:
    # 1. Salva e exibe o prompt do usuário
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Inicializa o assistente
    if st.session_state.agent_executor is None:
        with st.chat_message("assistant"):
            st.error("Agente não inicializado. Faça upload do arquivo e clique em 'Carregar e Inicializar Agente' na barra lateral.")
    else:
        # A nova bolha do assistente
        with st.chat_message("assistant"):
            # O container é importante para garantir a ordem (figura antes do texto)
            resp_container = st.container() 
            final_message = ""
            
            try:
                # 3. Invoca o agente
                full = st.session_state.agent_executor.invoke({"input": prompt})
                response_content = full.get('output')
                
                # Prepara o objeto para salvar no histórico
                assistant_response = {"role": "assistant"}
                
                # 4. Processa a resposta
                if isinstance(response_content, dict) and response_content.get('status') in ['success','error']:
                    # Resposta de Ferramenta
                    final_message = response_content.get('message', 'Erro ao obter mensagem da ferramenta.')

                    # Salva figuras para exibição (Plotly deve ser serializável)
                    if st.session_state.figure_to_display is not None:
                         # Exibe e salva o gráfico Plotly no histórico
                        resp_container.plotly_chart(st.session_state.figure_to_display, use_container_width=True)
                        assistant_response['plotly_figure'] = st.session_state.figure_to_display
                        st.session_state.figure_to_display = None
                        
                    # O Matplotlib não será persistido (evita erros), mas pode ser exibido
                    if st.session_state.grafico_para_exibir is not None:
                        resp_container.pyplot(st.session_state.grafico_para_exibir)
                        # NÃO SALVA Matplotlib no histórico
                        st.session_state.grafico_para_exibir = None
                        
                else:
                    # Resposta Direta do LLM (aqui está a correção principal)
                    final_message = str(response_content)

                # 5. Exibe a mensagem de texto FINAL
                if final_message:
                    resp_container.markdown(final_message)
                    assistant_response['content'] = final_message
                    
                    # 6. Salva no histórico APENAS UMA VEZ
                    st.session_state.messages.append(assistant_response)

            except Exception as e:
                err = f"Erro na execução do agente: {e}"
                resp_container.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
