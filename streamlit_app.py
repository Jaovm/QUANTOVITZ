# streamlit_app.py
import streamlit as st
import pandas as pd
import numpy as np
import financial_analyzer
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(layout="wide")
st.title("Painel de Otimização de Carteira de Investimentos")
# Importar funções do financial_analyzer.py
# Supondo que financial_analyzer.py está no mesmo diretório ou no PYTHONPATH
try:
    from financial_analyzer import (
        obter_dados_fundamentalistas,
        calcular_quant_value,
        simular_dados_historicos_retornos, 
        calcular_probabilidade_retorno,
        otimizar_portfolio_markowitz,
        calcular_metricas_portfolio,
        sugerir_alocacao_novo_aporte
    )
except ImportError:
    st.error("Erro ao importar o módulo 'financial_analyzer'. Certifique-se de que o arquivo está no diretório correto.")
    # Mock functions if import fails, to allow UI to load for development
    def obter_dados_fundamentalistas(ativos):
        return pd.DataFrame()
    def calcular_quant_value(ativos, pesos, dados):
        return pd.DataFrame(columns=["Ticker", "Quant Score"])
    def simular_dados_historicos_retornos(ativos, periodos=252):
        return pd.DataFrame(np.random.randn(periodos, len(ativos)), columns=ativos)
    def calcular_probabilidade_retorno(retornos_list):
        return 0.5
    def otimizar_portfolio_markowitz(ativos, df_retornos, taxa_livre_risco=0.02):
        return None, None, []
    def sugerir_alocacao_novo_aporte(current_portfolio_values, new_capital, target_weights):
        return {}, 0.0


# --- Entradas do Usuário na Sidebar ---
st.sidebar.header("Entradas do Usuário")

# 1. Carteira Atual
st.sidebar.subheader("1. Carteira Atual")
ativos_input_str = st.sidebar.text_input("Ativos da carteira (ex: PETR4,VALE3,ITUB4)", "PETR4,VALE3,ITUB4")
pesos_input_str = st.sidebar.text_input("Pesos percentuais (ex: 40,30,30)", "40,30,30")
valor_total_carteira_atual = st.sidebar.number_input("Valor total da carteira atual (R$)", min_value=0.0, value=100000.0, step=1000.0)

# 2. Novo Aporte (Opcional)
st.sidebar.subheader("2. Novo Aporte (Opcional)")
novo_capital_input = st.sidebar.number_input("Novo capital a ser aportado (R$)", min_value=0.0, value=10000.0, step=100.0)

# 3. Ativos Candidatos (Opcional)
st.sidebar.subheader("3. Ativos Candidatos (Opcional)")
candidatos_input_str = st.sidebar.text_input("Ativos candidatos à entrada (ex: MGLU3,VIIA3)", "MGLU3,VIIA3")

# 4. Modelo de Otimização
st.sidebar.subheader("4. Modelo de Otimização")
modelo_selecionado = st.sidebar.selectbox(
    "Escolha o modelo",
    ("Somente Quant-Value", "Quant-Value + Fronteira Eficiente", "Quant-Value + Fronteira Eficiente + Econometria")
)

# 5. Pesos para Métricas Fundamentalistas (Opcional)
st.sidebar.subheader("5. Pesos das Métricas Fundamentalistas (Opcional)")
# Adicionar mais métricas conforme a disponibilidade no `obter_dados_fundamentalistas`
pesos_metricas = {}
pesos_metricas['ROE'] = st.sidebar.slider("Peso ROE (%)", 0, 100, 25) / 100.0
pesos_metricas['EV/EBIT'] = st.sidebar.slider("Peso EV/EBIT (%)", 0, 100, 25) / 100.0
pesos_metricas['P/L'] = st.sidebar.slider("Peso P/L (%)", 0, 100, 25) / 100.0
pesos_metricas['DY'] = st.sidebar.slider("Peso DY (%)", 0, 100, 25) / 100.0

# Normalizar pesos das métricas para somarem 1 (opcional, ou avisar usuário)
soma_pesos_metricas = sum(pesos_metricas.values())
if soma_pesos_metricas == 0: # Evitar divisão por zero se todos os pesos forem 0
    st.sidebar.warning("Defina pesos para as métricas fundamentalistas para usar o Quant-Value.")
elif abs(soma_pesos_metricas - 1.0) > 1e-6:
    st.sidebar.warning(f"A soma dos pesos das métricas ({soma_pesos_metricas*100:.0f}%) não é 100%. Eles serão normalizados ou pode levar a resultados inesperados.")
    # Para normalizar: pesos_metricas = {k: v / soma_pesos_metricas for k, v in pesos_metricas.items()} 

# 6. Limites de Alocação (Opcional)
st.sidebar.subheader("6. Limites de Alocação por Ativo (Opcional)")
min_aloc_ativo = st.sidebar.slider("Alocação Mínima por Ativo (%)", 0, 100, 0) / 100.0
max_aloc_ativo = st.sidebar.slider("Alocação Máxima por Ativo (%)", 0, 100, 100) / 100.0

# Botão para executar a análise
run_analysis = st.sidebar.button("Executar Análise")

# --- Funções de Plotagem ---
def plot_quant_value_scores(df_scores):
    if df_scores.empty:
        st.write("Não há dados de Score Quant-Value para exibir.")
        return
    fig = px.bar(df_scores, x='Ticker', y='Quant Score', title='Score Quant-Value por Ativo',
                 labels={'Quant Score': 'Score (0-1)', 'Ticker': 'Ativo'},
                 color='Quant Score', color_continuous_scale=px.colors.sequential.Viridis)
    fig.update_layout(xaxis_title="Ativo", yaxis_title="Score Quant-Value")
    st.plotly_chart(fig, use_container_width=True)

def plot_asset_return_probabilities(probabilities_dict):
    if not probabilities_dict:
        st.write("Não há dados de Probabilidade de Retorno para exibir.")
        return
    df_probs = pd.DataFrame.from_dict(probabilities_dict, orient='index', columns=['Probabilidade de Retorno Positivo'])
    df_probs['Ativo'] = df_probs.index
    df_probs['Probabilidade de Retorno Positivo'] = df_probs['Probabilidade de Retorno Positivo'] * 100 # Para percentual
    fig = px.bar(df_probs, x='Ativo', y='Probabilidade de Retorno Positivo',
                 title='Probabilidade de Retorno Positivo por Ativo (%)',
                 labels={'Probabilidade de Retorno Positivo': 'Probabilidade (%)', 'Ativo': 'Ativo'},
                 color='Probabilidade de Retorno Positivo', color_continuous_scale=px.colors.sequential.Viridis,
                 range_color=[0,100])
    fig.update_layout(yaxis_range=[0,100])
    st.plotly_chart(fig, use_container_width=True)

def plot_efficient_frontier(fronteira_pontos, portfolio_max_sharpe, portfolio_max_retorno, carteira_atual_metricas=None):
    if not fronteira_pontos:
        st.write("Não foi possível gerar a Fronteira Eficiente.")
        return

    df_fronteira = pd.DataFrame(fronteira_pontos)
    
    fig = go.Figure()
    # Pontos da fronteira eficiente simulados
    fig.add_trace(go.Scatter(x=df_fronteira['volatilidade']*100, y=df_fronteira['retorno']*100,
                             mode='markers', name='Portfólios Simulados',
                             marker=dict(color=df_fronteira['sharpe'], colorscale='Viridis', showscale=True, size=5, line=dict(width=0),
                                         colorbar=dict(title='Sharpe Ratio')),
                             text=[f"Sharpe: {s:.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in p.items()})[:100]}..." for s, p in zip(df_fronteira['sharpe'], df_fronteira['pesos'])]))

    # Portfólio de Máximo Sharpe
    if portfolio_max_sharpe:
        fig.add_trace(go.Scatter(x=[portfolio_max_sharpe['volatilidade']*100], y=[portfolio_max_sharpe['retorno_esperado']*100],
                                 mode='markers', name='Max Sharpe Ratio',
                                 marker=dict(color='red', size=12, symbol='star'),
                                 text=f"Max Sharpe: {portfolio_max_sharpe['sharpe_ratio']:.2f}<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in portfolio_max_sharpe['pesos'].items()})}"))

    # Portfólio de Máximo Retorno (entre os simulados)
    if portfolio_max_retorno:
        fig.add_trace(go.Scatter(x=[portfolio_max_retorno['volatilidade']*100], y=[portfolio_max_retorno['retorno_esperado']*100],
                                 mode='markers', name='Max Retorno (Simulado)',
                                 marker=dict(color='green', size=12, symbol='diamond'),
                                 text=f"Max Retorno: {portfolio_max_retorno['retorno_esperado']*100:.2f}%<br>Pesos: {str({k: f'{v*100:.1f}%' for k,v in portfolio_max_retorno['pesos'].items()})}"))
    
    # Carteira Atual
    if carteira_atual_metricas:
        fig.add_trace(go.Scatter(x=[carteira_atual_metricas['volatilidade']*100], y=[carteira_atual_metricas['retorno_esperado']*100],
                                 mode='markers', name='Carteira Atual',
                                 marker=dict(color='blue', size=12, symbol='circle'),
                                 text=f"Carteira Atual<br>Sharpe: {carteira_atual_metricas['sharpe_ratio']:.2f}"))

    fig.update_layout(title='Fronteira Eficiente de Markowitz',
                      xaxis_title='Volatilidade Anualizada (%)',
                      yaxis_title='Retorno Esperado Anualizado (%)',
                      legend_title_text='Portfólios',
                      coloraxis_colorbar_x=0.85)
    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_pie_chart(weights_dict, title):
    if not weights_dict or sum(weights_dict.values()) == 0:
        st.write(f"Não há dados para o gráfico de pizza: {title}")
        return
    df_pie = pd.DataFrame(list(weights_dict.items()), columns=['Ativo', 'Peso'])
    df_pie = df_pie[df_pie['Peso'] > 1e-4] # Filtrar pesos muito pequenos para melhor visualização
    fig = px.pie(df_pie, values='Peso', names='Ativo', title=title,
                 hole=0.3)
    fig.update_traces(textposition='inside', textinfo='percent+label')
    st.plotly_chart(fig, use_container_width=True)

def display_comparative_table(carteiras_data):
    """ Exibe uma tabela comparativa das carteiras. 
        carteiras_data é uma lista de dicionários, cada um representando uma carteira.
        Ex: [{'Nome': 'Atual', 'Retorno (%)': 10, 'Volatilidade (%)': 15, 'Sharpe': 0.5, 'Pesos': {'A':0.5,...}}, ...]
    """
    if not carteiras_data:
        st.write("Não há dados de carteiras para comparar.")
        return
    
    # Preparar dados para a tabela principal
    df_comparativo = pd.DataFrame(carteiras_data)
    df_comparativo_display = df_comparativo[['Nome', 'Retorno Esperado (%)', 'Volatilidade (%)', 'Sharpe Ratio']].copy()
    df_comparativo_display = df_comparativo_display.set_index('Nome')
    st.subheader("Tabela Comparativa de Carteiras")
    st.dataframe(df_comparativo_display.style.format("{:.2f}"))

    # Mostrar pesos detalhados
    st.subheader("Composição Detalhada das Carteiras (%)")
    # Coletar todos os ativos envolvidos para colunas da tabela de pesos
    todos_ativos_pesos = set()
    for c in carteiras_data:
        if 'Pesos' in c and isinstance(c['Pesos'], dict):
            todos_ativos_pesos.update(c['Pesos'].keys())
    
    pesos_data_list = []
    for c in carteiras_data:
        row = {'Nome': c['Nome']}
        if 'Pesos' in c and isinstance(c['Pesos'], dict):
            for ativo in todos_ativos_pesos:
                row[ativo] = c['Pesos'].get(ativo, 0) * 100 # Em percentual
        pesos_data_list.append(row)
    
    df_pesos_detalhados = pd.DataFrame(pesos_data_list).set_index('Nome')
    st.dataframe(df_pesos_detalhados.style.format("{:.2f}"))

# --- Lógica Principal da Aplicação ---
if run_analysis:
    st.header("Resultados da Análise")

    # Processar entradas
    ativos_carteira_lista = [s.strip().upper() for s in ativos_input_str.split(',') if s.strip()]
    try:
        pesos_carteira_lista_pct = [float(p.strip()) for p in pesos_input_str.split(',') if p.strip()]
        if len(ativos_carteira_lista) != len(pesos_carteira_lista_pct):
            st.error("O número de ativos e pesos na carteira atual deve ser o mesmo.")
            st.stop()
        if abs(sum(pesos_carteira_lista_pct) - 100.0) > 1e-2:
            st.warning(f"A soma dos pesos da carteira atual ({sum(pesos_carteira_lista_pct):.2f}%) não é 100%. Ajuste para prosseguir ou os resultados podem ser inconsistentes.")
            # Poderia normalizar ou parar
        pesos_carteira_decimal = [p/100.0 for p in pesos_carteira_lista_pct]
        carteira_atual_composicao_valores = {ativo: peso_dec * valor_total_carteira_atual for ativo, peso_dec in zip(ativos_carteira_lista, pesos_carteira_decimal)}

    except ValueError:
        st.error("Os pesos da carteira atual devem ser números.")
        st.stop()

    ativos_candidatos_lista = [s.strip().upper() for s in candidatos_input_str.split(',') if s.strip()]
    todos_ativos_analise = list(set(ativos_carteira_lista + ativos_candidatos_lista))

    if not todos_ativos_analise:
        st.error("Nenhum ativo fornecido para análise.")
        st.stop()

    # 1. Coleta de Dados Fundamentalistas
    with st.spinner("Coletando dados fundamentalistas..."):
        df_fundamentalistas = obter_dados_fundamentalistas(todos_ativos_analise)
        if df_fundamentalistas.empty and 'Quant-Value' in modelo_selecionado:
            st.warning("Não foi possível obter dados fundamentalistas para os ativos. A análise Quant-Value pode ser afetada.")
        elif not df_fundamentalistas.empty:
            df_fundamentalistas.index.name = 'Ticker' # Assegurar que o índice tem nome

    # 2. Cálculo do Score Quant-Value
    quant_scores_df = pd.DataFrame(columns=["Ticker", "Quant Score"])
    if 'Quant-Value' in modelo_selecionado:
        with st.spinner("Calculando Score Quant-Value..."):
            if soma_pesos_metricas > 1e-6: # Se há pesos definidos
                quant_scores_df = calcular_quant_value(todos_ativos_analise, pesos_metricas, df_fundamentalistas)
                st.subheader("Análise Quant-Value")
                plot_quant_value_scores(quant_scores_df)
            else:
                st.warning("Pesos das métricas fundamentalistas não definidos. Score Quant-Value não calculado.")
    
    # 3. Coleta de Dados Históricos e Cálculo de Probabilidade de Retorno
    df_retornos_historicos = pd.DataFrame()
    probabilidades_retorno = {}
    if 'Fronteira Eficiente' in modelo_selecionado or 'Econometria' in modelo_selecionado:
        with st.spinner("Coletando dados históricos e calculando probabilidades de retorno..."):
            # Usar os ativos que serão considerados na otimização (todos ou um subconjunto)
            ativos_para_otimizacao = todos_ativos_analise # Pode ser ajustado por uma seleção do usuário
            df_retornos_historicos = simular_dados_historicos_retornos(ativos_para_otimizacao, periodos=252*3) # 3 anos de dados simulados
            
            if not df_retornos_historicos.empty:
                for ativo in ativos_para_otimizacao:
                    if ativo in df_retornos_historicos.columns:
                        probabilidades_retorno[ativo] = calcular_probabilidade_retorno(df_retornos_historicos[ativo].tolist())
                st.subheader("Probabilidade de Retorno")
                plot_asset_return_probabilities(probabilidades_retorno)
            else:
                st.warning("Não foi possível obter dados históricos de retorno. Otimização e cálculo de probabilidade afetados.")

    # 4. Otimização com Fronteira Eficiente
    portfolio_otimizado_sharpe = None
    portfolio_otimizado_retorno = None
    fronteira_eficiente_data = []
    carteiras_para_comparacao = []
    taxa_livre_risco_anual = 0.02 # Exemplo, pode ser input do usuário

    # Calcular métricas da carteira atual para plotar na fronteira
    carteira_atual_plot_metrics = None
    if ativos_carteira_lista and not df_retornos_historicos.empty and all(a in df_retornos_historicos.columns for a in ativos_carteira_lista):
        retornos_medios_atuais = df_retornos_historicos[ativos_carteira_lista].mean() * 252
        matriz_cov_atuais = df_retornos_historicos[ativos_carteira_lista].cov() * 252
        ret_atual, vol_atual, sharpe_atual = calcular_metricas_portfolio(np.array(pesos_carteira_decimal), retornos_medios_atuais, matriz_cov_atuais, taxa_livre_risco_anual)
        carteira_atual_plot_metrics = {'retorno_esperado': ret_atual, 'volatilidade': vol_atual, 'sharpe_ratio': sharpe_atual}
        carteiras_para_comparacao.append({
            'Nome': 'Carteira Atual',
            'Retorno Esperado (%)': ret_atual * 100,
            'Volatilidade (%)': vol_atual * 100,
            'Sharpe Ratio': sharpe_atual,
            'Pesos': dict(zip(ativos_carteira_lista, pesos_carteira_decimal))
        })
        st.subheader("Carteira Atual")
        plot_portfolio_pie_chart(dict(zip(ativos_carteira_lista, pesos_carteira_decimal)), "Composição da Carteira Atual")

    if 'Fronteira Eficiente' in modelo_selecionado:
        with st.spinner("Otimizando portfólio (Fronteira Eficiente)..."):
            # A otimização pode ser feita com todos os ativos ou apenas os da carteira + candidatos
            # Por enquanto, usando todos_ativos_analise
            if not df_retornos_historicos.empty and todos_ativos_analise:
                portfolio_otimizado_sharpe, portfolio_otimizado_retorno, fronteira_eficiente_data = \
                    otimizar_portfolio_markowitz(todos_ativos_analise, df_retornos_historicos[todos_ativos_analise], taxa_livre_risco_anual)
                
                st.subheader("Otimização de Carteira (Fronteira Eficiente)")
                plot_efficient_frontier(fronteira_eficiente_data, portfolio_otimizado_sharpe, portfolio_otimizado_retorno, carteira_atual_plot_metrics)

                if portfolio_otimizado_sharpe:
                    st.write("**Carteira com Maior Sharpe Ratio (Otimizada):**")
                    plot_portfolio_pie_chart(portfolio_otimizado_sharpe['pesos'], "Carteira Otimizada (Max Sharpe)")
                    carteiras_para_comparacao.append({
                        'Nome': 'Carteira Max Sharpe',
                        'Retorno Esperado (%)': portfolio_otimizado_sharpe['retorno_esperado'] * 100,
                        'Volatilidade (%)': portfolio_otimizado_sharpe['volatilidade'] * 100,
                        'Sharpe Ratio': portfolio_otimizado_sharpe['sharpe_ratio'],
                        'Pesos': portfolio_otimizado_sharpe['pesos']
                    })
                
                if portfolio_otimizado_retorno:
                    # st.write("**Carteira com Maior Retorno (Otimizada):**") # Opcional mostrar esta
                    # plot_portfolio_pie_chart(portfolio_otimizado_retorno['pesos'], "Carteira Otimizada (Max Retorno)")
                    carteiras_para_comparacao.append({
                        'Nome': 'Carteira Max Retorno (Sim.)',
                        'Retorno Esperado (%)': portfolio_otimizado_retorno['retorno_esperado'] * 100,
                        'Volatilidade (%)': portfolio_otimizado_retorno['volatilidade'] * 100,
                        'Sharpe Ratio': portfolio_otimizado_retorno['sharpe_ratio'],
                        'Pesos': portfolio_otimizado_retorno['pesos']
                    })
            else:
                st.warning("Não foi possível realizar a otimização da fronteira eficiente devido à falta de dados de retorno ou ativos.")

    # 5. Sugestão de Compra com Novo Aporte
    if novo_capital_input > 0 and portfolio_otimizado_sharpe: # Basear sugestão na carteira de max sharpe
        with st.spinner("Sugerindo alocação para novo aporte..."):
            st.subheader("Sugestão de Alocação para Novo Aporte")
            # Target é a carteira de máximo Sharpe
            target_weights_novo_aporte = portfolio_otimizado_sharpe['pesos'] 
            
            compras_sugeridas, capital_excedente = sugerir_alocacao_novo_aporte(
                carteira_atual_composicao_valores,
                novo_capital_input,
                target_weights_novo_aporte
            )
            
            if compras_sugeridas:
                st.write("**Valores a comprar por ativo (R$):**")
                for ativo, valor in compras_sugeridas.items():
                    st.write(f"- {ativo}: {valor:.2f}")
                # Pie chart da alocação do novo capital
                plot_portfolio_pie_chart(compras_sugeridas, f"Alocação Sugerida do Novo Capital (R$ {novo_capital_input:.2f})")
            else:
                st.write("Nenhuma compra específica sugerida para o novo aporte com base na carteira de máximo Sharpe. O capital pode ser distribuído conforme os pesos da carteira alvo ou mantido como excedente.")
            
            if capital_excedente > 0.01:
                st.write(f"**Capital excedente após alocação sugerida:** R$ {capital_excedente:.2f}")
            
            # Mostrar como ficaria a carteira final após o aporte
            carteira_final_valores = carteira_atual_composicao_valores.copy()
            for ativo, valor_compra in compras_sugeridas.items():
                carteira_final_valores[ativo] = carteira_final_valores.get(ativo, 0) + valor_compra
            if capital_excedente > 0.01 and sum(target_weights_novo_aporte.values()) > 1e-6 : # se sobrou e tem onde alocar
                 # Distribuir o excedente conforme pesos da carteira alvo
                 for ativo, peso_alvo in target_weights_novo_aporte.items():
                     carteira_final_valores[ativo] = carteira_final_valores.get(ativo, 0) + (capital_excedente * peso_alvo)
            
            soma_carteira_final = sum(carteira_final_valores.values())
            if soma_carteira_final > 1e-6:
                pesos_carteira_final = {k: v / soma_carteira_final for k, v in carteira_final_valores.items()}
                plot_portfolio_pie_chart(pesos_carteira_final, "Composição da Carteira Projetada Após Novo Aporte")
                # Adicionar à tabela comparativa
                ret_final, vol_final, sharpe_final = calcular_metricas_portfolio(np.array(list(pesos_carteira_final.values())), 
                                                                                df_retornos_historicos[list(pesos_carteira_final.keys())].mean()*252, 
                                                                                df_retornos_historicos[list(pesos_carteira_final.keys())].cov()*252, 
                                                                                taxa_livre_risco_anual)
                carteiras_para_comparacao.append({
                    'Nome': 'Carteira Pós-Aporte (Alvo Max Sharpe)',
                    'Retorno Esperado (%)': ret_final * 100,
                    'Volatilidade (%)': vol_final * 100,
                    'Sharpe Ratio': sharpe_final,
                    'Pesos': pesos_carteira_final
                })

    # Exibir tabela comparativa de todas as carteiras geradas
    if carteiras_para_comparacao:
        display_comparative_table(carteiras_para_comparacao)

    # 6. Explicações no Painel (Placeholder)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Glossário e Explicações")
    with st.sidebar.expander("Quant-Value"):
        st.write("O modelo Quant-Value combina métricas fundamentalistas para atribuir um score a cada ativo, ajudando a identificar os mais atrativos com base em valor.")
    with st.sidebar.expander("Fronteira Eficiente"):
        st.write("A Fronteira Eficiente de Markowitz mostra o conjunto de portfólios ótimos que oferecem o maior retorno esperado para um dado nível de risco, ou o menor risco para um dado retorno.")
    with st.sidebar.expander("Índice de Sharpe"):
        st.write("O Índice de Sharpe mede o retorno de um investimento ajustado pelo risco. Um Sharpe mais alto indica melhor desempenho por unidade de risco.")
    with st.sidebar.expander("Econometria (Prob. Retorno)"):
        st.write("Modelos econométricos (aqui simplificados por simulação histórica) podem ser usados para projetar retornos esperados e a probabilidade de valorização de ativos.")

else:
    st.info("Ajuste os parâmetros na barra lateral e clique em 'Executar Análise' para ver os resultados.")


