
import pandas as pd
import numpy as np
# from scipy.optimize import minimize # Importação comentada pois não será usada na simulação inicial

# Simulação de dados fundamentalistas (substituir por dados reais ou API)
def obter_dados_fundamentalistas(ativos):
    # Esta função simula a obtenção de dados. Em um cenário real, você usaria uma API.
    dados = {
        'PETR4': {'ROE': 0.15, 'EV/EBIT': 8, 'P/L': 10, 'DY': 0.05},
        'VALE3': {'ROE': 0.20, 'EV/EBIT': 6, 'P/L': 8, 'DY': 0.06},
        'ITUB4': {'ROE': 0.18, 'EV/EBIT': 7, 'P/L': 9, 'DY': 0.04},
        'MGLU3': {'ROE': -0.05, 'EV/EBIT': -15, 'P/L': -12, 'DY': 0.01},
        'VIIA3': {'ROE': -0.10, 'EV/EBIT': -20, 'P/L': -10, 'DY': 0.005}
    }
    data_for_ativos = {ativo: dados.get(ativo, {})
                       for ativo in ativos}
    return pd.DataFrame.from_dict(data_for_ativos, orient='index')

def calcular_quant_value(ativos_selecionados, pesos_metricas, dados_fundamentalistas):
    # Assegura que dados_fundamentalistas é um DataFrame e tem os ativos como índice
    if not isinstance(dados_fundamentalistas, pd.DataFrame) or not dados_fundamentalistas.index.name == 'Ticker':
        # Tenta converter se for um dict simples como o da simulação, ou reindexar
        if isinstance(dados_fundamentalistas, dict):
            dados_fundamentalistas = pd.DataFrame.from_dict(dados_fundamentalistas, orient='index')
        if dados_fundamentalistas.empty or not all(col in dados_fundamentalistas.columns for col in pesos_metricas.keys()):
             # Se ainda vazio ou faltando colunas de métricas, retorna df vazio
            return pd.DataFrame(columns=['Ticker', 'Quant Score'])
        # Se o índice não for 'Ticker', mas os tickers estiverem em uma coluna, ajuste necessário (não coberto aqui)

    # Filtra os dados para incluir apenas os ativos selecionados e que estão no DataFrame
    ativos_presentes_em_dados = [ativo for ativo in ativos_selecionados if ativo in dados_fundamentalistas.index]
    if not ativos_presentes_em_dados:
        return pd.DataFrame(columns=['Ticker', 'Quant Score'])
    dados_relevantes = dados_fundamentalistas.loc[ativos_presentes_em_dados, list(pesos_metricas.keys())].copy()
    
    if dados_relevantes.empty:
        return pd.DataFrame(columns=['Ticker', 'Quant Score'])

    # Normalização dos dados (exemplo: ranking)
    dados_normalizados = dados_relevantes.rank(pct=True, axis=0) # rank within each metric column

    # Cálculo do Score Quant-Value
    score_quant = pd.Series(0.0, index=dados_relevantes.index)
    for metrica, peso in pesos_metricas.items():
        if metrica in dados_normalizados.columns:
            score_quant += dados_normalizados[metrica] * peso
    
    resultado_df = pd.DataFrame({'Ticker': score_quant.index, 'Quant Score': score_quant.values})
    resultado_df = resultado_df.sort_values(by='Quant Score', ascending=False).reset_index(drop=True)
    return resultado_df

def calcular_probabilidade_retorno(retornos_historicos_list):
    if not retornos_historicos_list:
        return 0.5 # Retorno neutro se não houver histórico
    positivos = sum(1 for r in retornos_historicos_list if r > 0)
    return positivos / len(retornos_historicos_list)

def simular_dados_historicos_retornos(ativos, periodos=252):
    np.random.seed(42)
    retornos_simulados = {}
    for ativo in ativos:
        media_retorno_diario = np.random.uniform(-0.001, 0.002)
        desvio_padrao_diario = np.random.uniform(0.01, 0.03)
        retornos_simulados[ativo] = np.random.normal(media_retorno_diario, desvio_padrao_diario, periodos)
    return pd.DataFrame(retornos_simulados)

def calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco=0.02):
    retorno_portfolio = np.sum(retornos_medios_anuais * pesos)
    volatilidade_portfolio = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia_anual, pesos)))
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / volatilidade_portfolio if volatilidade_portfolio != 0 else -np.inf
    return retorno_portfolio, volatilidade_portfolio, sharpe_ratio

def otimizar_portfolio_markowitz(ativos, df_retornos_historicos, taxa_livre_risco=0.02):
    if df_retornos_historicos.empty or len(ativos) == 0 or not all(a in df_retornos_historicos.columns for a in ativos):
        # Retorna None para todos se os dados de entrada não forem válidos
        return None, None, [] 

    # Considerar apenas os ativos solicitados que estão no dataframe de retornos
    retornos_considerados = df_retornos_historicos[ativos]

    retornos_medios_diarios = retornos_considerados.mean()
    matriz_covariancia_diaria = retornos_considerados.cov()
    num_ativos = len(ativos)

    retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_anual = matriz_covariancia_diaria * 252

    num_portfolios_simulados = 5000 # Reduzido para performance em simulação
    resultados_lista = [] # Lista para armazenar dicionários de resultados
    pesos_portfolios = []

    for i in range(num_portfolios_simulados):
        pesos = np.random.random(num_ativos)
        pesos /= np.sum(pesos)
        pesos_portfolios.append(pesos)
        retorno, volatilidade, sharpe = calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco)
        resultados_lista.append({'retorno': retorno, 'volatilidade': volatilidade, 'sharpe': sharpe, 'pesos': dict(zip(ativos, pesos))})
    
    if not resultados_lista:
        return None, None, []

    # Portfólio com maior Sharpe Ratio
    portfolio_max_sharpe_dict = max(resultados_lista, key=lambda x: x['sharpe'])
    portfolio_max_sharpe = {
        'pesos': portfolio_max_sharpe_dict['pesos'],
        'retorno_esperado': portfolio_max_sharpe_dict['retorno'],
        'volatilidade': portfolio_max_sharpe_dict['volatilidade'],
        'sharpe_ratio': portfolio_max_sharpe_dict['sharpe']
    }

    # Portfólio com maior Retorno Esperado
    portfolio_max_retorno_dict = max(resultados_lista, key=lambda x: x['retorno'])
    portfolio_max_retorno = {
        'pesos': portfolio_max_retorno_dict['pesos'],
        'retorno_esperado': portfolio_max_retorno_dict['retorno'],
        'volatilidade': portfolio_max_retorno_dict['volatilidade'],
        'sharpe_ratio': portfolio_max_retorno_dict['sharpe']
    }
    
    return portfolio_max_sharpe, portfolio_max_retorno, resultados_lista

def sugerir_alocacao_novo_aporte(
    current_portfolio_composition_values: dict, # {'PETR4': 40000, 'VALE3': 60000}
    new_capital: float,
    target_portfolio_weights_decimal: dict, # {'PETR4': 0.5, 'VALE3': 0.3, 'MGLU3': 0.2}
    # quant_scores_df: pd.DataFrame = None # Para futura priorização
):
    if new_capital <= 1e-6: # Consider small new_capital as no aport
        return {}, 0.0 # No allocation, no surplus

    current_portfolio_value = sum(current_portfolio_composition_values.values())
    final_portfolio_value = current_portfolio_value + new_capital

    purchases_to_reach_target = {}
    # Consider all assets present in current portfolio or target portfolio
    all_involved_assets = set(current_portfolio_composition_values.keys()) | set(target_portfolio_weights_decimal.keys())

    for asset_ticker in all_involved_assets:
        current_asset_value = current_portfolio_composition_values.get(asset_ticker, 0.0)
        target_asset_final_value = target_portfolio_weights_decimal.get(asset_ticker, 0.0) * final_portfolio_value
        
        amount_to_buy = max(0, target_asset_final_value - current_asset_value)
        if amount_to_buy > 1e-6: # Use a small epsilon for float comparisons
            purchases_to_reach_target[asset_ticker] = amount_to_buy

    total_capital_needed_for_target = sum(purchases_to_reach_target.values())
    
    actual_purchases = {}
    surplus_capital = 0.0

    if total_capital_needed_for_target <= 1e-6: 
        # No specific purchases needed to reach target weights (e.g., already optimal or overweighted in needed assets)
        # Distribute all new_capital according to target_portfolio_weights_decimal for assets in target_portfolio_weights_decimal.
        relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
        sum_target_weights_for_distribution = sum(relevant_target_assets.values())

        if sum_target_weights_for_distribution > 1e-6:
            for asset_ticker, weight in relevant_target_assets.items():
                actual_purchases[asset_ticker] = (weight / sum_target_weights_for_distribution) * new_capital
        else:
            # If no positive target weights, all new capital is surplus (or could be distributed equally if desired)
            surplus_capital = new_capital
    elif total_capital_needed_for_target <= new_capital:
        # Enough capital to make all desired purchases to reach target.
        actual_purchases = purchases_to_reach_target.copy()
        surplus_capital_after_reaching_target = new_capital - total_capital_needed_for_target
        
        if surplus_capital_after_reaching_target > 1e-6:
            # Distribute surplus according to target_weights_decimal
            relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
            sum_target_weights_for_distribution = sum(relevant_target_assets.values())
            if sum_target_weights_for_distribution > 1e-6:
                for asset_ticker, weight in relevant_target_assets.items():
                    additional_buy = (weight / sum_target_weights_for_distribution) * surplus_capital_after_reaching_target
                    actual_purchases[asset_ticker] = actual_purchases.get(asset_ticker, 0) + additional_buy
                surplus_capital = 0.0 # Surplus has been distributed
            else:
                surplus_capital = surplus_capital_after_reaching_target # Surplus remains
        else:
            surplus_capital = 0.0
    else: # total_capital_needed_for_target > new_capital
        # Not enough capital. Distribute available new_capital proportionally to needed amounts.
        # TODO: Consider Quant-Value for prioritization here if quant_scores_df is provided.
        for asset_ticker, needed_amount in purchases_to_reach_target.items():
            actual_purchases[asset_ticker] = (needed_amount / total_capital_needed_for_target) * new_capital
        surplus_capital = 0.0

    actual_purchases = {k: v for k, v in actual_purchases.items() if v > 0.01} # Filter to amounts > 1 cent

    return actual_purchases, surplus_capital


if __name__ == '__main__':
    # Exemplo de uso das funções:
    ativos_carteira_exemplo = ['PETR4', 'VALE3', 'ITUB4']
    ativos_candidatos_exemplo = ['MGLU3', 'VIIA3']
    todos_ativos_exemplo = list(set(ativos_carteira_exemplo + ativos_candidatos_exemplo))

    # Dados Fundamentalistas e Quant-Value
    pesos_metricas_exemplo = {'ROE': 0.4, 'EV/EBIT': 0.3, 'P/L': 0.2, 'DY': 0.1}
    df_fundamentalistas_exemplo = obter_dados_fundamentalistas(todos_ativos_exemplo)
    df_fundamentalistas_exemplo.index.name = 'Ticker' # Definindo nome do índice
    quant_value_scores_exemplo = calcular_quant_value(todos_ativos_exemplo, pesos_metricas_exemplo, df_fundamentalistas_exemplo)
    print("Score Quant-Value dos Ativos (Exemplo):")
    print(quant_value_scores_exemplo)

    # Simular dados de retornos históricos para todos os ativos relevantes
    df_retornos_historicos_exemplo = simular_dados_historicos_retornos(todos_ativos_exemplo)

    # Calcular probabilidade de retorno
    probabilidades_retorno_exemplo = {}
    for ativo in todos_ativos_exemplo:
        if ativo in df_retornos_historicos_exemplo.columns:
            probabilidades_retorno_exemplo[ativo] = calcular_probabilidade_retorno(df_retornos_historicos_exemplo[ativo].tolist())
    print("\nProbabilidade de Retorno Positivo (Exemplo):")
    for ativo, prob in probabilidades_retorno_exemplo.items():
        print(f"- {ativo}: {prob:.2%}")

    # Otimizar portfólio (usando todos os ativos para a otimização, por exemplo)
    # A escolha de quais ativos entram na otimização depende da estratégia do usuário
    portfolio_sharpe, portfolio_retorno, _ = otimizar_portfolio_markowitz(todos_ativos_exemplo, df_retornos_historicos_exemplo)

    if portfolio_sharpe:
        print("\n--- Portfólio com Maior Sharpe Ratio (Simulado) ---")
        for ativo, peso in portfolio_sharpe['pesos'].items():
            print(f"- {ativo}: {peso*100:.2f}%")
        print(f"Retorno Esperado: {portfolio_sharpe['retorno_esperado']*100:.2f}%")
        print(f"Volatilidade: {portfolio_sharpe['volatilidade']*100:.2f}%")
        print(f"Sharpe Ratio: {portfolio_sharpe['sharpe_ratio']:.2f}")

        # Exemplo de Sugestão de Novo Aporte
        current_portfolio_values_ex = {'PETR4': 60000, 'VALE3': 40000} # Exemplo de carteira atual em valores
        new_capital_ex = 20000
        # Usar os pesos do portfólio de max Sharpe como alvo
        target_weights_decimal_ex = portfolio_sharpe['pesos'] 

        compras_sugeridas, capital_excedente = sugerir_alocacao_novo_aporte(
            current_portfolio_values_ex,
            new_capital_ex,
            target_weights_decimal_ex
        )
        print("\n--- Sugestão de Alocação para Novo Aporte (R$) ---")
        if compras_sugeridas:
            for ativo, valor_compra in compras_sugeridas.items():
                print(f"- Comprar {ativo}: R$ {valor_compra:.2f}")
        else:
            print("Nenhuma compra específica sugerida para o novo aporte com base nos critérios.")
        if capital_excedente > 0.01:
            print(f"Capital excedente após alocação: R$ {capital_excedente:.2f}")
        elif not compras_sugeridas and new_capital_ex > 0:
             print(f"Todo o novo capital (R$ {new_capital_ex:.2f}) é excedente ou não há alvos claros.")
```
