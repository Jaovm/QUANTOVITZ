#!/usr/bin/python3.11
# financial_analyzer_enhanced.py

import pandas as pd
import numpy as np
import requests
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
from arch import arch_model
import statsmodels.api as sm
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize

# --- Configuration & Constants ---
DATA_DIR = "/home/ubuntu/collected_data"
RISK_FREE_RATE_DEFAULT = 0.02 # Annual risk-free rate
MIN_YEARS_DATA = 3 # Minimum years of data for some calculations

# --- Data Loading and Preprocessing ---

def load_historical_data_from_csv(ticker):
    """Loads historical price data from a CSV file in DATA_DIR."""
    safe_ticker_name = ticker.replace('^', 'INDEX_')
    file_path = os.path.join(DATA_DIR, f"{safe_ticker_name}_chart_data.csv")
    if os.path.exists(file_path):
        try:
            df = pd.read_csv(file_path, parse_dates=["Date"])
            df.set_index("Date", inplace=True)
            # Ensure Adj Close is present and numeric
            if 'Adj Close' in df.columns and pd.api.types.is_numeric_dtype(df['Adj Close']):
                df.dropna(subset=['Adj Close'], inplace=True) # Drop rows where Adj Close is NaN
                return df[['Adj Close']]
            else:
                print(f"Warning: 'Adj Close' not found or not numeric in {file_path} for {ticker}.")
                return pd.DataFrame() # Return empty DataFrame
        except Exception as e:
            print(f"Error loading or processing CSV {file_path} for {ticker}: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

def obter_dados_historicos_yf(ativos, start_date_str, end_date_str=None):
    """Obtém retornos diários dos ativos, prioritizing CSV then Yahoo Finance API."""
    if end_date_str is None:
        end_date_str = datetime.today().strftime('%Y-%m-%d')
    
    all_adj_close = pd.DataFrame()
    for ativo in ativos:
        df_ativo = load_historical_data_from_csv(ativo)
        if df_ativo.empty:
            print(f"Fetching {ativo} from yfinance API as CSV not found or invalid.")
            try:
                data = yf.download(ativo, start=start_date_str, end=end_date_str, progress=False)
                if not data.empty and 'Close' in data.columns: # Check for 'Close'
                    df_ativo = data[['Close']].copy() # Use 'Close'
                    df_ativo.rename(columns={'Close': 'Adj Close'}, inplace=True) # Rename to 'Adj Close' for consistency
                    df_ativo.dropna(inplace=True)
                else:
                    print(f"Warning: No 'Adj Close' data from yfinance API for {ativo}.")
                    continue # Skip this asset if no data
            except Exception as e:
                print(f"Error fetching {ativo} from yfinance API: {e}")
                continue
        
        if not df_ativo.empty:
            # Filter by date range after loading
            df_ativo = df_ativo[(df_ativo.index >= pd.to_datetime(start_date_str)) & (df_ativo.index <= pd.to_datetime(end_date_str))]
            df_ativo.rename(columns={'Adj Close': ativo}, inplace=True)
            if all_adj_close.empty:
                all_adj_close = df_ativo
            else:
                all_adj_close = all_adj_close.join(df_ativo, how='outer')
        else:
            print(f"Warning: No data loaded for {ativo}.")

    if all_adj_close.empty:
        return pd.DataFrame() # Return empty if no data for any asset
        
    df_retornos = all_adj_close.pct_change().dropna(how='all') # Drop rows with all NaNs after pct_change
    return df_retornos

def load_insights_data_from_json(ticker):
    """Loads insights data from a JSON file in DATA_DIR."""
    safe_ticker_name = ticker.replace('^', 'INDEX_')
    file_path = os.path.join(DATA_DIR, f"{safe_ticker_name}_insights_data.json")
    if os.path.exists(file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading JSON {file_path} for {ticker}: {e}")
    return None

def get_yfinance_ticker_info(ticker_symbol):
    """Safely fetches yfinance.Ticker object and its info."""
    try:
        ticker_obj = yf.Ticker(ticker_symbol)
        return ticker_obj.info, ticker_obj
    except Exception as e:
        print(f"Error fetching yfinance.Ticker info for {ticker_symbol}: {e}")
        return None, None

# --- Fundamental Metrics --- 
def obter_dados_fundamentalistas_detalhados_br(ativos):
    """
    Obtém dados fundamentalistas detalhados para ativos da B3, usando yfinance,
    e complementando com APIs nacionais (exemplo: Brapi) se necessário.
    Retorna DataFrame padronizado para análise fundamentalista avançada.
    """
    import requests
    dados_fund = {}
    for ativo in ativos:
        temp_data = {'ticker': ativo}
        # Tenta yfinance
        info, yf_ticker_obj = get_yfinance_ticker_info(ativo)
        if info:
            temp_data['marketCap'] = info.get('marketCap')
            temp_data['sharesOutstanding'] = info.get('sharesOutstanding')
            temp_data['dividendYield'] = info.get('dividendYield')
            temp_data['trailingPE'] = info.get('trailingPE')
            temp_data['priceToBook'] = info.get('priceToBook')
            temp_data['returnOnEquity'] = info.get('returnOnEquity')
            temp_data['totalRevenue'] = info.get('totalRevenue')
            temp_data['operatingCashflow'] = info.get('operatingCashflow')
            temp_data['totalDebt'] = info.get('totalDebt')
            temp_data['grossMargins'] = info.get('grossMargins')
            temp_data['netIncomeToCommon'] = info.get('netIncomeToCommon')
            temp_data['sector'] = info.get('sector')
            temp_data['industry'] = info.get('industry')
        # Dados detalhados yfinance (finance, balance_sheet, cashflow)
        if yf_ticker_obj:
            try:
                financials = yf_ticker_obj.financials
                balance_sheet = yf_ticker_obj.balance_sheet
                cashflow = yf_ticker_obj.cashflow
                # Padronização para campos brasileiros
                if not financials.empty and len(financials.columns) >= 1:
                    col = financials.columns[0]
                    temp_data['lucro_liquido_atual'] = financials.loc['Net Income', col] if 'Net Income' in financials.index else None
                    temp_data['lucro_bruto_atual'] = financials.loc['Gross Profit', col] if 'Gross Profit' in financials.index else None
                    temp_data['receita_liquida_atual'] = financials.loc['Total Revenue', col] if 'Total Revenue' in financials.index else None
                if not balance_sheet.empty and len(balance_sheet.columns) >= 1:
                    col = balance_sheet.columns[0]
                    temp_data['ativos_totais_atual'] = balance_sheet.loc['Total Assets', col] if 'Total Assets' in balance_sheet.index else None
                    temp_data['divida_lp_atual'] = balance_sheet.loc['Long Term Debt', col] if 'Long Term Debt' in balance_sheet.index else None
                    temp_data['ativos_circulantes_atual'] = balance_sheet.loc['Total Current Assets', col] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['passivos_circulantes_atual'] = balance_sheet.loc['Total Current Liabilities', col] if 'Total Current Liabilities' in balance_sheet.index else None
                if not cashflow.empty and len(cashflow.columns) >= 1:
                    col = cashflow.columns[0]
                    temp_data['cfo_atual'] = cashflow.loc['Total Cash From Operating Activities', col] if 'Total Cash From Operating Activities' in cashflow.index else None
                # Dados do ano anterior (para F-Score)
                if len(financials.columns) >= 2:
                    col_prev = financials.columns[1]
                    temp_data['lucro_liquido_anterior'] = financials.loc['Net Income', col_prev] if 'Net Income' in financials.index else None
                    temp_data['lucro_bruto_anterior'] = financials.loc['Gross Profit', col_prev] if 'Gross Profit' in financials.index else None
                    temp_data['receita_liquida_anterior'] = financials.loc['Total Revenue', col_prev] if 'Total Revenue' in financials.index else None
                if len(balance_sheet.columns) >= 2:
                    col_prev = balance_sheet.columns[1]
                    temp_data['ativos_totais_anterior'] = balance_sheet.loc['Total Assets', col_prev] if 'Total Assets' in balance_sheet.index else None
                    temp_data['divida_lp_anterior'] = balance_sheet.loc['Long Term Debt', col_prev] if 'Long Term Debt' in balance_sheet.index else None
                    temp_data['ativos_circulantes_anterior'] = balance_sheet.loc['Total Current Assets', col_prev] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['passivos_circulantes_anterior'] = balance_sheet.loc['Total Current Liabilities', col_prev] if 'Total Current Liabilities' in balance_sheet.index else None
            except Exception as e:
                print(f"Erro ao buscar dados detalhados para {ativo}: {e}")

        # INTEGRAÇÃO (mock) com Brapi para complementar dados de empresas da B3
        try:
            # Exemplo: https://brapi.dev/api/quote/PETR4.SA?fundamental=true
            url_brapi = f"https://brapi.dev/api/quote/{ativo}?token=2D29LijXrSGRJAQ7De5bUh"
            r = requests.get(url_brapi, timeout=3)
            if r.status_code == 200:
                brapi_data = r.json()['results'][0].get('fundamental', {})
                temp_data['roic'] = brapi_data.get('roic')
                temp_data['ebit_margin'] = brapi_data.get('ebitMargin')
                temp_data['net_debt_ebitda'] = brapi_data.get('netDebtEBITDA')
                temp_data['cagr_receita_5a'] = brapi_data.get('revenueCAGR5Y')
                temp_data['cagr_lucro_5a'] = brapi_data.get('netIncomeCAGR5Y')
        except Exception as e:
            print(f"Falha ao buscar dados Brapi para {ativo}: {e}")

        dados_fund[ativo] = temp_data

    df_fund = pd.DataFrame.from_dict(dados_fund, orient='index')
    # Conversão numérica
    for col in df_fund.columns:
        df_fund[col] = pd.to_numeric(df_fund[col], errors='ignore')
    return df_fund


def calcular_piotroski_f_score_br(row, verbose=False):
    """
    Calcula o Piotroski F-Score com validações robustas para empresas brasileiras.
    """
    # Extração segura de campos
    lucro_liquido_atual = pd.to_numeric(row.get('lucro_liquido_atual'), errors='coerce')
    lucro_liquido_anterior = pd.to_numeric(row.get('lucro_liquido_anterior'), errors='coerce')
    receita_liquida_atual = pd.to_numeric(row.get('receita_liquida_atual'), errors='coerce')
    receita_liquida_anterior = pd.to_numeric(row.get('receita_liquida_anterior'), errors='coerce')
    ativos_totais_atual = pd.to_numeric(row.get('ativos_totais_atual'), errors='coerce')
    ativos_totais_anterior = pd.to_numeric(row.get('ativos_totais_anterior'), errors='coerce')
    cfo_atual = pd.to_numeric(row.get('cfo_atual'), errors='coerce')
    divida_lp_atual = pd.to_numeric(row.get('divida_lp_atual'), errors='coerce')
    divida_lp_anterior = pd.to_numeric(row.get('divida_lp_anterior'), errors='coerce')
    ativos_circulantes_atual = pd.to_numeric(row.get('ativos_circulantes_atual'), errors='coerce')
    passivos_circulantes_atual = pd.to_numeric(row.get('passivos_circulantes_atual'), errors='coerce')
    ativos_circulantes_anterior = pd.to_numeric(row.get('ativos_circulantes_anterior'), errors='coerce')
    passivos_circulantes_anterior = pd.to_numeric(row.get('passivos_circulantes_anterior'), errors='coerce')
    acoes_emitidas_atual = pd.to_numeric(row.get('acoes_emitidas_atual'), errors='coerce')
    acoes_emitidas_anterior = pd.to_numeric(row.get('acoes_emitidas_anterior'), errors='coerce')
    lucro_bruto_atual = pd.to_numeric(row.get('lucro_bruto_atual'), errors='coerce')
    lucro_bruto_anterior = pd.to_numeric(row.get('lucro_bruto_anterior'), errors='coerce')

    # Margens brutas calculadas, se necessário
    margem_bruta_atual = lucro_bruto_atual / receita_liquida_atual if pd.notna(lucro_bruto_atual) and pd.notna(receita_liquida_atual) and receita_liquida_atual != 0 else np.nan
    margem_bruta_anterior = lucro_bruto_anterior / receita_liquida_anterior if pd.notna(lucro_bruto_anterior) and pd.notna(receita_liquida_anterior) and receita_liquida_anterior != 0 else np.nan

    # Rotatividade de ativos
    rot_ativos_atual = receita_liquida_atual / ativos_totais_atual if pd.notna(receita_liquida_atual) and pd.notna(ativos_totais_atual) and ativos_totais_atual != 0 else np.nan
    rot_ativos_anterior = receita_liquida_anterior / ativos_totais_anterior if pd.notna(receita_liquida_anterior) and pd.notna(ativos_totais_anterior) and ativos_totais_anterior != 0 else np.nan

    # ROA
    roa_atual = lucro_liquido_atual / ativos_totais_atual if pd.notna(lucro_liquido_atual) and pd.notna(ativos_totais_atual) and ativos_totais_atual != 0 else np.nan
    roa_anterior = lucro_liquido_anterior / ativos_totais_anterior if pd.notna(lucro_liquido_anterior) and pd.notna(ativos_totais_anterior) and ativos_totais_anterior != 0 else np.nan

    # Liquidez Corrente
    try:
        if pd.notna(ativos_circulantes_atual) and pd.notna(passivos_circulantes_atual) and passivos_circulantes_atual != 0:
            liquidez_corrente_atual = ativos_circulantes_atual / passivos_circulantes_atual
        else:
            liquidez_corrente_atual = np.nan
        if pd.notna(ativos_circulantes_anterior) and pd.notna(passivos_circulantes_anterior) and passivos_circulantes_anterior != 0:
            liquidez_corrente_anterior = ativos_circulantes_anterior / passivos_circulantes_anterior
        else:
            liquidez_corrente_anterior = np.nan
    except:
        liquidez_corrente_atual = liquidez_corrente_anterior = np.nan

    # Critérios
    criterios = {
        'lucro_liquido_positivo': pd.notna(lucro_liquido_atual) and lucro_liquido_atual > 0,
        'cfo_positivo': pd.notna(cfo_atual) and cfo_atual > 0,
        'roa_crescente': pd.notna(roa_atual) and pd.notna(roa_anterior) and roa_atual > roa_anterior,
        'cfo_maior_que_lucro': pd.notna(cfo_atual) and pd.notna(lucro_liquido_atual) and cfo_atual > lucro_liquido_atual,
        'queda_divida_lp': pd.notna(divida_lp_atual) and pd.notna(divida_lp_anterior) and divida_lp_atual < divida_lp_anterior,
        'aumento_liquidez_corrente': pd.notna(liquidez_corrente_atual) and pd.notna(liquidez_corrente_anterior) and liquidez_corrente_atual > liquidez_corrente_anterior,
        'aumento_margem_bruta': pd.notna(margem_bruta_atual) and pd.notna(margem_bruta_anterior) and margem_bruta_atual > margem_bruta_anterior,
        'aumento_rotatividade_ativos': pd.notna(rot_ativos_atual) and pd.notna(rot_ativos_anterior) and rot_ativos_atual > rot_ativos_anterior,
        'nao_emitiu_acoes': pd.notna(acoes_emitidas_atual) and pd.notna(acoes_emitidas_anterior) and acoes_emitidas_atual <= acoes_emitidas_anterior
    }

    score = sum(criterios.values())
    return (score, criterios) if verbose else score
    
def calcular_altman_z_score(row):
    """
    Calcula o Altman Z-Score (versão para empresas não financeiras brasileiras).
    Retorna float.
    """
    # Padronização dos campos (ajuste conforme disponibilidade dos dados)
    ativos_totais = row.get('ativos_totais_atual') or row.get('TotalAssets_curr')
    passivo_circulante = row.get('passivos_circulantes_atual') or row.get('CurrentLiab_curr')
    capital_giro = (row.get('ativos_circulantes_atual') or row.get('CurrentAssets_curr')) - passivo_circulante if pd.notna(passivo_circulante) else np.nan
    receita_liquida = row.get('receita_liquida_atual') or row.get('Revenue_curr')
    ebit = row.get('ebit') or row.get('EBIT') or row.get('ebit_margin') * receita_liquida if 'ebit_margin' in row and pd.notna(row['ebit_margin']) else np.nan
    patrimonio_liquido = row.get('patrimonio_liquido') or row.get('Equity') or np.nan
    lucro_retido = row.get('lucro_retido') or np.nan
    valor_mercado = row.get('marketCap')
    total_passivo = row.get('totalDebt') or np.nan

    try:
        A = capital_giro / ativos_totais
        B = lucro_retido / ativos_totais if pd.notna(lucro_retido) and pd.notna(ativos_totais) else 0
        C = ebit / ativos_totais if pd.notna(ebit) and pd.notna(ativos_totais) else 0
        D = valor_mercado / total_passivo if pd.notna(valor_mercado) and pd.notna(total_passivo) and total_passivo > 0 else 0
        E = receita_liquida / ativos_totais if pd.notna(receita_liquida) and pd.notna(ativos_totais) else 0
        z_score = 1.2*A + 1.4*B + 3.3*C + 0.6*D + 1.0*E
        return z_score
    except Exception:
        return np.nan

def calcular_beneish_m_score(row):
    """
    Calcula o Beneish M-Score para detectar manipulação contábil.
    Retorna float.
    """
    # Padronize os campos conforme disponíveis na base
    # Atenção: pode exigir vários anos de dados, implemente versões simplificadas se necessário
    try:
        dsri = row.get('DSRI', 1.0)    # Days Sales in Receivables Index
        gmi = row.get('GMI', 1.0)      # Gross Margin Index
        aqi = row.get('AQI', 1.0)      # Asset Quality Index
        sgi = row.get('SGI', 1.0)      # Sales Growth Index
        depi = row.get('DEPI', 1.0)    # Depreciation Index
        sgai = row.get('SGAI', 1.0)    # SGA Expense Index
        lvgi = row.get('LVGI', 1.0)    # Leverage Index
        tata = row.get('TATA', 0.0)    # Total Accruals to Total Assets

        m_score = (
            -4.84 + 0.92*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi +
            0.115*depi - 0.172*sgai + 4.679*lvgi - 0.327*tata
        )
        return m_score
    except Exception:
        return np.nan

def calcular_score_setorial(row, media_setor: dict, campos_multiplo: list):
    """
    Compara múltiplos da empresa com a média do setor.
    Retorna score (quanto menor, melhor).
    Exemplo de uso:
        calcular_score_setorial(row, media_setor={'trailingPE': 10, ...}, campos_multiplo=['trailingPE','priceToBook'])
    """
    score = 0
    for campo in campos_multiplo:
        valor = row.get(campo)
        media = media_setor.get(campo)
        if pd.notna(valor) and pd.notna(media):
            score += (valor / media)
    return score / len(campos_multiplo) if campos_multiplo else np.nan
    
def calcular_value_composite_score(df_fundamental, metrics_config):
    if df_fundamental.empty:
        return pd.Series(dtype=float)
    ranked_df = pd.DataFrame(index=df_fundamental.index)
    for metric, ranking_type in metrics_config.items():
        if metric in df_fundamental.columns:
            numeric_metric = pd.to_numeric(df_fundamental[metric], errors='coerce')
            if ranking_type == 'lower_is_better':
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True) 
            elif ranking_type == 'higher_is_better':
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=False)
            else:
                 ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True)
        else:
            print(f"Warning: Metric {metric} not found in fundamental data for Value Composite Score.")
    rank_cols = [col for col in ranked_df.columns if '_rank' in col]
    if not rank_cols:
        return pd.Series(dtype=float, index=df_fundamental.index)
    composite_score = ranked_df[rank_cols].mean(axis=1)
    return composite_score

def calcular_volatilidade_garch(returns_series, p=1, q=1):
    if returns_series.empty or len(returns_series) < (p + q + 20):
        print(f"Not enough data points for GARCH model on series: {returns_series.name}")
        return np.nan
    try:
        scaled_returns = returns_series * 100
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        forecasted_std_dev_daily_scaled = np.sqrt(forecast.variance.iloc[-1,0])
        forecasted_std_dev_daily = forecasted_std_dev_daily_scaled / 100.0
        return forecasted_std_dev_daily * np.sqrt(252)
    except Exception as e:
        print(f"Error fitting GARCH for {returns_series.name}: {e}")
        return np.nan

def get_fama_french_factors(start_date, end_date, risk_free_rate_series=None):
    print("Fetching Fama-French factor proxies...")
    factor_tickers = {
        'MKT_PROXY': '^GSPC',
        'SMB_SMALL': '^RUT',
        'SMB_LARGE': '^GSPC',
        'HML_VALUE': 'IVE',
        'HML_GROWTH':'IVW',
        'WML_MOM': 'MTUM'
    }
    factor_data_all_cols = yf.download(list(factor_tickers.values()), start=start_date, end=end_date, progress=False)
    if factor_data_all_cols.empty:
        print("Could not download factor proxy data (empty main DataFrame).")
        return pd.DataFrame()
    try:
        factor_data_raw = factor_data_all_cols['Close']
    except KeyError:
        try:
            factor_data_raw = factor_data_all_cols['Adj Close']
        except KeyError:
            print("Error: Neither 'Close' nor 'Adj Close' found as a primary column key for factor data.")
            available_keys = factor_data_all_cols.columns.levels[0] if isinstance(factor_data_all_cols.columns, pd.MultiIndex) else factor_data_all_cols.columns
            print(f"Available primary column keys: {available_keys}")
            return pd.DataFrame()
    if factor_data_raw.empty:
        print("Extracted factor price data ('Close' or 'Adj Close') is empty.")
        return pd.DataFrame()
    factor_returns = factor_data_raw.pct_change().dropna()
    factors_df = pd.DataFrame(index=factor_returns.index)
    if risk_free_rate_series is None:
        rf_data_raw = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        if not rf_data_raw.empty:
            try:
                rf_data_series = rf_data_raw['Close']
            except KeyError:
                try:
                    rf_data_series = rf_data_raw['Adj Close']
                except KeyError:
                    print("Warning: Neither 'Close' nor 'Adj Close' found for ^IRX. RF rate might be inaccurate.")
                    rf_data_series = pd.Series(dtype=float)
            if not rf_data_series.empty and pd.api.types.is_numeric_dtype(rf_data_series):
                daily_rf = (rf_data_series / 100) / 252 
                factors_df['RF'] = daily_rf.reindex(factors_df.index, method='ffill')
            else:
                print("Warning: ^IRX data for risk-free rate is empty or not numeric after extraction. Using default constant / 252.")
                factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
        else:
            print("Warning: Could not fetch ^IRX for risk-free rate (empty DataFrame). Using default constant / 252.")
            factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
    else:
        factors_df['RF'] = risk_free_rate_series
    factors_df.ffill(inplace=True)  # <<<< CORREÇÃO DE WARNING (era fillna(method='ffill'))
    factors_df.dropna(inplace=True)
    if factor_tickers['MKT_PROXY'] in factor_returns.columns and 'RF' in factors_df.columns:
        mkt_rf_series = factor_returns[factor_tickers['MKT_PROXY']] - factors_df['RF']
        factors_df['Mkt-RF'] = mkt_rf_series.loc[factors_df.index]
    if factor_tickers['SMB_SMALL'] in factor_returns.columns and factor_tickers['SMB_LARGE'] in factor_returns.columns:
        smb_series = factor_returns[factor_tickers['SMB_SMALL']] - factor_returns[factor_tickers['SMB_LARGE']]
        factors_df['SMB'] = smb_series.loc[factors_df.index]
    if factor_tickers['HML_VALUE'] in factor_returns.columns and factor_tickers['HML_GROWTH'] in factor_returns.columns:
        hml_series = factor_returns[factor_tickers['HML_VALUE']] - factor_returns[factor_tickers['HML_GROWTH']]
        factors_df['HML'] = hml_series.loc[factors_df.index]
    if factor_tickers['WML_MOM'] in factor_returns.columns and 'RF' in factors_df.columns:
        wml_series = factor_returns[factor_tickers['WML_MOM']] - factors_df['RF']
        factors_df['WML'] = wml_series.loc[factors_df.index]
    factors_df.dropna(inplace=True)
    return factors_df[['Mkt-RF', 'SMB', 'HML', 'WML', 'RF']].copy()

def estimar_fatores_alpha_beta(asset_returns_series, factor_df):
    if asset_returns_series.empty or factor_df.empty:
        return np.nan, {}
    common_index = asset_returns_series.index.intersection(factor_df.index)
    if len(common_index) < 20:
        print(f"Not enough common data points for factor regression on {asset_returns_series.name}")
        return np.nan, {}
    y = asset_returns_series.loc[common_index] - factor_df.loc[common_index, 'RF']
    X = factor_df.loc[common_index, ['Mkt-RF', 'SMB', 'HML', 'WML']]
    X = sm.add_constant(X)
    X.dropna(inplace=True)
    y = y.loc[X.index]
    if len(y) < 20:
        print(f"Not enough data points after alignment for factor regression on {asset_returns_series.name}")
        return np.nan, {}
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        alpha = model.params.get('const', np.nan) * 252
        betas = model.params.drop('const', errors='ignore').to_dict()
        return alpha, betas
    except Exception as e:
        asset_name_for_log = asset_returns_series.name if isinstance(asset_returns_series, pd.Series) else str(asset_returns_series.columns[0]) if isinstance(asset_returns_series, pd.DataFrame) else "unknown"
        print(f"Error in OLS regression for {asset_name_for_log}: {e}")
        return np.nan, {}

def prever_retornos_arima(returns_series, order=(5,1,0)):
    if returns_series.empty or len(returns_series) < (sum(order) + 20):
        print(f"Not enough data points for ARIMA model on series: {returns_series.name}")
        return np.nan
    try:
        model = ARIMA(returns_series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0]
    except Exception as e:
        print(f"Error fitting ARIMA for {returns_series.name}: {e}. Trying auto_arima or simpler model.")
        try:
            model = ARIMA(returns_series, order=(1,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e2:
            print(f"Fallback ARIMA failed for {returns_series.name}: {e2}")
            return np.nan

def calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco):
    retorno_portfolio = np.sum(retornos_medios_anuais * pesos)
    volatilidade_portfolio = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia_anual, pesos)))
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / volatilidade_portfolio if volatilidade_portfolio != 0 else -np.inf
    return retorno_portfolio, volatilidade_portfolio, sharpe_ratio

def ajustar_retornos_esperados(base_retornos_medios_anuais, df_fundamental, alphas, arima_forecasts, quant_value_scores, piotroski_scores):
    adjusted_retornos = base_retornos_medios_anuais.copy()
    for ativo in adjusted_retornos.index:
        adjustment_factor = 1.0
        if piotroski_scores is not None and ativo in piotroski_scores and pd.notna(piotroski_scores[ativo]):
            pscore = piotroski_scores[ativo]
            if pscore <= 2: adjustment_factor *= 0.9
            elif pscore >=7: adjustment_factor *= 1.1
        if quant_value_scores is not None and ativo in quant_value_scores and pd.notna(quant_value_scores[ativo]):
            qscore = quant_value_scores[ativo]
            adjustment_factor *= (1 + (qscore - 0.5) * 0.2)
        adjusted_retornos[ativo] *= adjustment_factor
        if alphas is not None and ativo in alphas and pd.notna(alphas[ativo]):
            adjusted_retornos[ativo] += alphas[ativo]
        if arima_forecasts is not None and ativo in arima_forecasts and pd.notna(arima_forecasts[ativo]):
            annualized_arima_forecast = arima_forecasts[ativo] * 252
            adjusted_retornos[ativo] = (adjusted_retornos[ativo] * 0.8) + (annualized_arima_forecast * 0.2)
    return adjusted_retornos

def preencher_campos_faltantes_brapi(row):
    """
    Preenche campos críticos do Piotroski usando a API Brapi se estiverem ausentes.
    Espera um campo 'ticker' no row.
    """
    ticker = row.get('ticker')
    if not ticker:
        return row  # Sem ticker, não faz nada

    try:
        url = f"https://brapi.dev/api/quote/{ticker}?fundamental=true"
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return row
        data = resp.json()
        if 'results' not in data or not data['results']:
            return row
        dados = data['results'][0]

        # CFO (Cash Flow from Operations)
        if pd.isna(row.get('cfo_atual')) or row.get('cfo_atual') is None:
            cfo = None
            if 'cashFlowStatement' in dados and 'operatingCashFlow' in dados['cashFlowStatement']:
                cfo = dados['cashFlowStatement']['operatingCashFlow']
            elif 'operatingCashFlow' in dados:
                cfo = dados['operatingCashFlow']
            if cfo is not None:
                try:
                    row['cfo_atual'] = float(str(cfo).replace('.', '').replace(',', '.'))
                except:
                    pass

        # Ações emitidas (sharesOutstanding)
        if (pd.isna(row.get('acoes_emitidas_atual')) or row.get('acoes_emitidas_atual') is None):
            shares = dados.get('shares') or dados.get('numberOfShares')
            if shares is not None:
                try:
                    row['acoes_emitidas_atual'] = float(str(shares).replace('.', '').replace(',', '.'))
                except:
                    pass

        # Ativos/Passivos Circulantes para Liquidez Corrente
        if (pd.isna(row.get('ativos_circulantes_atual')) or row.get('ativos_circulantes_atual') is None) and 'currentAssets' in dados:
            try:
                row['ativos_circulantes_atual'] = float(str(dados['currentAssets']).replace('.', '').replace(',', '.'))
            except:
                pass
        if (pd.isna(row.get('passivos_circulantes_atual')) or row.get('passivos_circulantes_atual') is None) and 'currentLiabilities' in dados:
            try:
                row['passivos_circulantes_atual'] = float(str(dados['currentLiabilities']).replace('.', '').replace(',', '.'))
            except:
                pass

        # (Opcional) Futuramente, inclua lógica para buscar valores do ano anterior, se conseguir fonte.

    except Exception as e:
        print(f"[Brapi] Falha para {ticker}: {e}")

    return row

def converter_campos_piotroski(df):
    """
    Converte todos os campos usados no Piotroski F-Score para float, se existirem no DataFrame.
    """
    piotroski_fields = [
        'lucro_liquido_atual', 'lucro_liquido_anterior', 'NI_curr', 'NI_prev', 'NetIncome_curr', 'NetIncome_prev',
        'receita_liquida_atual', 'receita_liquida_anterior', 'Revenue_curr', 'Revenue_prev', 'TotalRevenue_curr', 'TotalRevenue_prev',
        'ativos_totais_atual', 'ativos_totais_anterior', 'TotalAssets_curr', 'TotalAssets_prev',
        'cfo_atual', 'CFO_curr', 'operatingCashflow',
        'divida_lp_atual', 'divida_lp_anterior', 'divida_longo_prazo_atual', 'divida_longo_prazo_anterior', 'LongTermDebt_curr', 'LongTermDebt_prev',
        'ativos_circulantes_atual', 'ativos_circulantes_anterior', 'CurrentAssets_curr', 'CurrentAssets_prev',
        'passivos_circulantes_atual', 'passivos_circulantes_anterior', 'CurrentLiab_curr', 'CurrentLiab_prev',
        'acoes_emitidas_atual', 'acoes_emitidas_anterior', 'sharesOutstanding', 'sharesOutstanding_prev',
        'margem_bruta_atual', 'margem_bruta_anterior', 'GrossMargin_curr', 'GrossMargin_prev',
        'lucro_bruto_atual', 'lucro_bruto_anterior', 'GrossProfit_curr', 'GrossProfit_prev'
    ]
    for col in piotroski_fields:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df
    
def otimizar_portfolio_scipy(
    ativos,
    df_retornos_historicos,
    df_fundamental_completo=None,
    fama_french_factors=None,
    taxa_livre_risco=RISK_FREE_RATE_DEFAULT,
    pesos_atuais=None,
    restricoes_pesos_min_max=None,
    objetivo='max_sharpe'
):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    retornos_considerados = df_retornos_historicos[ativos].copy()
    if retornos_considerados.shape[1] != len(ativos) or retornos_considerados.isnull().values.any():
        print("Warning: Missing return data for some assets or all assets. Dropping NaNs.")
        retornos_considerados.dropna(axis=1, how='all', inplace=True)
        retornos_considerados.dropna(axis=0, how='any', inplace=True)
        ativos = list(retornos_considerados.columns)
        if not ativos:
            print("Error: No valid asset data left after cleaning for optimization.")
            return None, None, []
    retornos_medios_diarios = retornos_considerados.mean()
    base_retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_diaria = retornos_considerados.cov()
    matriz_covariancia_anual_historica = matriz_covariancia_diaria * 252
    num_ativos = len(ativos)
    adj_retornos_esperados = base_retornos_medios_anuais.copy()
    adj_matriz_covariancia = matriz_covariancia_anual_historica.copy()
    alphas = {}
    arima_forecasts = {}
    garch_volatilities = {}
    if df_fundamental_completo is not None and not df_fundamental_completo.empty:
        if 'Piotroski_F_Score' not in df_fundamental_completo.columns:
            df_fundamental_completo['Piotroski_F_Score'] = df_fundamental_completo.apply(calcular_piotroski_f_score_br, axis=1)
        vc_metrics = {
            'trailingPE': 'lower_is_better', 
            'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better',
            'dividendYield': 'higher_is_better', 
            'returnOnEquity': 'higher_is_better',
            'netMargin': 'higher_is_better'
        }
        if 'Quant_Value_Score' not in df_fundamental_completo.columns:
            df_fundamental_completo['Quant_Value_Score'] = calcular_value_composite_score(df_fundamental_completo, vc_metrics)
        piotroski_scores_series = df_fundamental_completo['Piotroski_F_Score'].reindex(ativos)
        quant_value_scores_series = df_fundamental_completo['Quant_Value_Score'].reindex(ativos)
        for ativo in ativos:
            asset_returns = retornos_considerados[ativo].dropna()
            if fama_french_factors is not None and not fama_french_factors.empty:
                alpha, _ = estimar_fatores_alpha_beta(asset_returns, fama_french_factors)
                alphas[ativo] = alpha
            arima_forecasts[ativo] = prever_retornos_arima(asset_returns)
            garch_vol = calcular_volatilidade_garch(asset_returns)
            if pd.notna(garch_vol) and garch_vol > 0:
                if ativo in adj_matriz_covariancia.index:
                    adj_matriz_covariancia.loc[ativo, ativo] = garch_vol**2
                    garch_volatilities[ativo] = garch_vol
        adj_retornos_esperados = ajustar_retornos_esperados(
            base_retornos_medios_anuais, 
            df_fundamental_completo.reindex(ativos),
            alphas, 
            arima_forecasts, 
            quant_value_scores_series,
            piotroski_scores_series
        )
    else:
        print("No fundamental data or factors provided for advanced optimization. Using historical mean/covariance.")
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    def portfolio_return(weights, expected_returns):
        return np.sum(expected_returns * weights)
    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_return = portfolio_return(weights, expected_returns)
        p_volatility = portfolio_volatility(weights, cov_matrix)
        if p_volatility == 0: return -np.inf
        return -(p_return - risk_free_rate) / p_volatility
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    bounds = []
    default_min_b, default_max_b = 0.0, 1.0
    if isinstance(restricoes_pesos_min_max, tuple) and len(restricoes_pesos_min_max) == 2:
        default_min_b, default_max_b = restricoes_pesos_min_max[0], restricoes_pesos_min_max[1]
    for ativo in ativos:
        min_b, max_b = default_min_b, default_max_b
        if isinstance(restricoes_pesos_min_max, dict) and ativo in restricoes_pesos_min_max:
            min_b, max_b = restricoes_pesos_min_max[ativo]
        elif pesos_atuais and ativo in pesos_atuais:
            pass
        bounds.append((min_b, max_b))
    initial_weights = np.array([1/num_ativos] * num_ativos)
    if pesos_atuais and len(pesos_atuais) == num_ativos:
        current_weights_array = np.array([pesos_atuais.get(a, 0) for a in ativos])
        if np.isclose(np.sum(current_weights_array), 1.0):
            initial_weights = current_weights_array
        else:
            print("Warning: Sum of provided current weights is not 1. Using equal initial weights.")
    if objetivo == 'max_sharpe':
        opt_args = (adj_retornos_esperados.values, adj_matriz_covariancia.values, taxa_livre_risco)
        opt_func = neg_sharpe_ratio
    elif objetivo == 'min_volatility':
        opt_args = (adj_matriz_covariancia.values,)
        opt_func = lambda w, cov: portfolio_volatility(w, cov)
    else:
        raise ValueError("Invalid objective function specified.")
    optimized_result = minimize(opt_func, initial_weights, args=opt_args,
                                method='SLSQP', bounds=bounds, constraints=constraints)
    if not optimized_result.success:
        print(f"Optimization failed: {optimized_result.message}")
    optimal_weights = optimized_result.x
    optimal_weights[optimal_weights < 1e-4] = 0
    optimal_weights = optimal_weights / np.sum(optimal_weights)
    ret_opt, vol_opt, sharpe_opt = calcular_metricas_portfolio(optimal_weights, adj_retornos_esperados, adj_matriz_covariancia, taxa_livre_risco)
    portfolio_otimizado = {
        'pesos': dict(zip(ativos, optimal_weights)),
        'retorno_esperado': ret_opt,
        'volatilidade': vol_opt,
        'sharpe_ratio': sharpe_opt,
        'garch_volatilities': garch_volatilities if garch_volatilities else None,
        'alphas': alphas if alphas else None,
        'arima_forecasts': arima_forecasts if arima_forecasts else None
    }
    fronteira_pontos_simulados = []
    return portfolio_otimizado, fronteira_pontos_simulados

def otimizar_portfolio_markowitz_mc(ativos, df_retornos_historicos, taxa_livre_risco=RISK_FREE_RATE_DEFAULT, num_portfolios_simulados=10000):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    retornos_considerados = df_retornos_historicos[ativos].copy()
    retornos_considerados.dropna(axis=1, how='all', inplace=True)
    retornos_considerados.dropna(axis=0, how='any', inplace=True)
    ativos = list(retornos_considerados.columns)
    if not ativos:
        return None, None, []
    retornos_medios_diarios = retornos_considerados.mean()
    matriz_covariancia_diaria = retornos_considerados.cov()
    num_ativos_calc = len(ativos)
    retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_anual = matriz_covariancia_diaria * 252
    resultados_lista = []
    for _ in range(num_portfolios_simulados):
        pesos = np.random.random(num_ativos_calc)
        pesos /= np.sum(pesos)
        retorno, volatilidade, sharpe = calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco)
        resultados_lista.append({'retorno': retorno, 'volatilidade': volatilidade, 'sharpe': sharpe, 'pesos': dict(zip(ativos, pesos))})
    if not resultados_lista:
        return None, None, []
    portfolio_max_sharpe_dict = max(resultados_lista, key=lambda x: x['sharpe'])
    portfolio_max_sharpe = {
        'pesos': portfolio_max_sharpe_dict['pesos'],
        'retorno_esperado': portfolio_max_sharpe_dict['retorno'],
        'volatilidade': portfolio_max_sharpe_dict['volatilidade'],
        'sharpe_ratio': portfolio_max_sharpe_dict['sharpe']
    }
    return portfolio_max_sharpe, resultados_lista

def sugerir_alocacao_novo_aporte(
    current_portfolio_composition_values: dict, 
    new_capital: float,
    target_portfolio_weights_decimal: dict,
):
    if new_capital <= 1e-6: return {}, 0.0
    current_portfolio_value = sum(current_portfolio_composition_values.values())
    final_portfolio_value = current_portfolio_value + new_capital
    purchases_to_reach_target = {}
    all_involved_assets = set(current_portfolio_composition_values.keys()) | set(target_portfolio_weights_decimal.keys())
    for asset_ticker in all_involved_assets:
        current_asset_value = current_portfolio_composition_values.get(asset_ticker, 0.0)
        target_asset_final_value = target_portfolio_weights_decimal.get(asset_ticker, 0.0) * final_portfolio_value
        amount_to_buy = max(0, target_asset_final_value - current_asset_value)
        if amount_to_buy > 1e-6: purchases_to_reach_target[asset_ticker] = amount_to_buy
    total_capital_needed_for_target = sum(purchases_to_reach_target.values())
    actual_purchases = {}
    surplus_capital = 0.0
    if total_capital_needed_for_target <= 1e-6:
        relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
        sum_target_weights_for_distribution = sum(relevant_target_assets.values())
        if sum_target_weights_for_distribution > 1e-6:
            for asset_ticker, weight in relevant_target_assets.items():
                actual_purchases[asset_ticker] = (weight / sum_target_weights_for_distribution) * new_capital
        else: surplus_capital = new_capital
    elif total_capital_needed_for_target <= new_capital:
        actual_purchases = purchases_to_reach_target.copy()
        surplus_capital_after_reaching_target = new_capital - total_capital_needed_for_target
        if surplus_capital_after_reaching_target > 1e-6:
            relevant_target_assets = {t: w for t, w in target_portfolio_weights_decimal.items() if w > 1e-6}
            sum_target_weights_for_distribution = sum(relevant_target_assets.values())
            if sum_target_weights_for_distribution > 1e-6:
                for asset_ticker, weight in relevant_target_assets.items():
                    additional_buy = (weight / sum_target_weights_for_distribution) * surplus_capital_after_reaching_target
                    actual_purchases[asset_ticker] = actual_purchases.get(asset_ticker, 0) + additional_buy
                surplus_capital = 0.0
            else: surplus_capital = surplus_capital_after_reaching_target
        else: surplus_capital = 0.0
    else:
        for asset_ticker, needed_amount in purchases_to_reach_target.items():
            actual_purchases[asset_ticker] = (needed_amount / total_capital_needed_for_target) * new_capital
        surplus_capital = 0.0
    actual_purchases = {k: v for k, v in actual_purchases.items() if v > 0.01}
    return actual_purchases, surplus_capital

# --- Main execution block for testing ---
if __name__ == '__main__':
    print("Running financial_analyzer_enhanced.py tests...")
    test_ativos = ['AAPL', 'MSFT', 'GOOGL']
    start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    end_date = datetime.today().strftime('%Y-%m-%d')
    print("\n--- Testing Historical Data ---")
    df_retornos = obter_dados_historicos_yf(test_ativos, start_date_str=start_date, end_date_str=end_date)
    if not df_retornos.empty:
        print(f"Successfully fetched historical returns for {len(df_retornos.columns)} assets. Shape: {df_retornos.shape}")
        print(df_retornos.head())
    else:
        print("Failed to fetch historical returns.")
    print("\n--- Testing Fundamental Data ---")
    df_fund_data = obter_dados_fundamentalistas_detalhados_br(test_ativos)
    if not df_fund_data.empty:
        df_fund_data['Piotroski_F_Score'], df_fund_data['Piotroski_F_Detalhes'] = zip(*df_fund_data.apply(lambda row: calcular_piotroski_f_score_br(row, verbose=True), axis=1))
        df_fund_data['Altman_Z_Score'] = df_fund_data.apply(calcular_altman_z_score, axis=1)
        df_fund_data['Beneish_M_Score'] = df_fund_data.apply(calcular_beneish_m_score, axis=1)
        # E assim por diante...
        vc_metrics_test = {
            'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better', 'dividendYield': 'higher_is_better',
            'returnOnEquity': 'higher_is_better', 'netMargin': 'higher_is_better'
        }
        df_fund_data['Quant_Value_Score'] = calcular_value_composite_score(df_fund_data, vc_metrics_test)
        print(df_fund_data[['ticker', 'Piotroski_F_Score', 'Quant_Value_Score']].head())
    else:
        print("Failed to fetch fundamental data.")
    print("\n--- Testing Fama-French Factors ---")
    ff_factors = get_fama_french_factors(start_date, end_date)
    if not ff_factors.empty:
        print(f"Successfully fetched Fama-French factor proxies. Shape: {ff_factors.shape}")
        print(ff_factors.head())
    else:
        print("Failed to fetch Fama-French factors.")
    print("\n--- Testing Advanced Portfolio Optimization (SciPy) ---")
    if not df_retornos.empty and not df_fund_data.empty and not ff_factors.empty:
        df_fund_data.set_index('ticker', inplace=True, drop=False)
        optimized_portfolio_advanced, _ = otimizar_portfolio_scipy(
            ativos=test_ativos,
            df_retornos_historicos=df_retornos,
            df_fundamental_completo=df_fund_data,
            fama_french_factors=ff_factors,
            taxa_livre_risco=RISK_FREE_RATE_DEFAULT,
            objetivo='max_sharpe'
        )
        if optimized_portfolio_advanced:
            print("Advanced Optimized Portfolio (Max Sharpe):")
            for k, v in optimized_portfolio_advanced.items():
                if k == 'pesos': print(f"  {k}: { {tk: f'{p*100:.2f}%' for tk,p in v.items()} }")
                elif isinstance(v, dict): print(f"  {k}: Present")
                elif isinstance(v, float): print(f"  {k}: {v:.4f}")
                else: print(f"  {k}: {v}")
        else:
            print("Advanced portfolio optimization failed.")
    else:
        print("Skipping advanced optimization test due to missing data (returns, fundamentals, or factors).")
    print("\n--- Testing Original Markowitz Optimization (Monte Carlo) ---")
    if not df_retornos.empty:
        portfolio_sharpe_mc, _ = otimizar_portfolio_markowitz_mc(test_ativos, df_retornos, taxa_livre_risco=RISK_FREE_RATE_DEFAULT)
        if portfolio_sharpe_mc:
            print("MC Optimized Portfolio (Max Sharpe):")
            print(f"  Pesos: { {tk: f'{p*100:.2f}%' for tk,p in portfolio_sharpe_mc['pesos'].items()} }")
            print(f"  Retorno Esperado: {portfolio_sharpe_mc['retorno_esperado']:.4f}")
            print(f"  Volatilidade: {portfolio_sharpe_mc['volatilidade']:.4f}")
            print(f"  Sharpe Ratio: {portfolio_sharpe_mc['sharpe_ratio']:.4f}")
        else:
            print("MC portfolio optimization failed.")
    else:
        print("Skipping MC optimization test due to missing return data.")
    print("\nfinancial_analyzer_enhanced.py tests completed.")
# Compatibilidade: garantir nome antigo para import do Streamlit
calcular_piotroski_f_score = calcular_piotroski_f_score_br
beneish_m_score = calcular_beneish_m_score
obter_dados_fundamentalistas_detalhados = obter_dados_fundamentalistas_detalhados_br
