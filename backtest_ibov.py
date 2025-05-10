import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def obter_dados_historicos_retornos_yf(ativos, start_date="2015-01-01", end_date=None):
    if end_date is None:
        end_date = datetime.today().strftime('%Y-%m-%d')
    df_prices = yf.download(ativos, start=start_date, end=end_date)['Adj Close']
    if isinstance(df_prices, pd.Series):
        df_prices = df_prices.to_frame()
    df_retornos = df_prices.pct_change().dropna()
    return df_prices, df_retornos

def calcular_cagr(valor_final, valor_inicial, anos):
    return (valor_final / valor_inicial) ** (1 / anos) - 1

if __name__ == "__main__":
    # Defina os ativos e pesos da carteira simulada
    ativos_carteira = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA']
    pesos = np.array([0.4, 0.4, 0.2])  # Exemplo: 40% PETR4, 40% VALE3, 20% ITUB4

    # Inclui o índice Ibovespa (^BVSP)
    ativos_yf = ativos_carteira + ['^BVSP']

    # Baixa os preços ajustados
    df_prices, df_retornos = obter_dados_historicos_retornos_yf(ativos_yf, start_date="2015-01-01")

    # Sincroniza datas e remove NaN
    df_prices = df_prices.dropna()
    df_retornos = df_retornos.loc[df_prices.index]

    # Calcula o valor acumulado da carteira simulada
    retornos_carteira = (df_retornos[ativos_carteira] * pesos).sum(axis=1)
    valor_inicial = 1000
    valor_carteira = valor_inicial * (1 + retornos_carteira).cumprod()

    # Calcula o valor acumulado do Ibovespa
    valor_ibov = valor_inicial * (1 + df_retornos['^BVSP']).cumprod()

    # Calcula o CAGR
    anos = (df_prices.index[-1] - df_prices.index[0]).days / 365.25
    cagr_carteira = calcular_cagr(valor_carteira.iloc[-1], valor_carteira.iloc[0], anos)
    cagr_ibov = calcular_cagr(valor_ibov.iloc[-1], valor_ibov.iloc[0], anos)

    # Resultados
    print(f"Período: {df_prices.index[0].strftime('%Y-%m-%d')} até {df_prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"Valor final carteira simulada: R$ {valor_carteira.iloc[-1]:.2f}")
    print(f"Valor final Ibovespa: R$ {valor_ibov.iloc[-1]:.2f}")
    print(f"CAGR Carteira: {cagr_carteira:.2%}")
    print(f"CAGR Ibovespa: {cagr_ibov:.2%}")

    # (Opcional) Plot
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12,6))
        plt.plot(valor_carteira, label="Carteira Simulada")
        plt.plot(valor_ibov, label="Ibovespa")
        plt.title("Backtest Carteira vs Ibovespa (desde 2015)")
        plt.ylabel("Valor (R$)")
        plt.legend()
        plt.grid()
        plt.show()
    except ImportError:
        pass