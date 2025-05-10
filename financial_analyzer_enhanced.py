#!/usr/bin/python3.11
# financial_analyzer_enhanced.py

import pandas as pd
import numpy as np
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
def obter_dados_fundamentalistas_detalhados(ativos):
    """Obtém dados fundamentalistas detalhados para cálculo de métricas como Piotroski F-Score."""
    dados_fund = {}
    for ativo in ativos:
        print(f"Fetching fundamental data for {ativo}...")
        insights = load_insights_data_from_json(ativo)
        info, yf_ticker_obj = get_yfinance_ticker_info(ativo)
        
        temp_data = {'ticker': ativo}
        
        # From insights if available
        if insights:
            if insights.get('summaryProfile'):
                temp_data['sector'] = insights['summaryProfile'].get('sector')
                temp_data['industry'] = insights['summaryProfile'].get('industry')
            if insights.get('financialData'):
                temp_data['returnOnEquity'] = insights['financialData'].get('returnOnEquity')
                temp_data['currentPrice'] = insights['financialData'].get('currentPrice')
                temp_data['targetMeanPrice'] = insights['financialData'].get('targetMeanPrice')
                temp_data['numberOfAnalystOpinions'] = insights['financialData'].get('numberOfAnalystOpinions')
            if insights.get('defaultKeyStatistics'):
                temp_data['enterpriseToEbitda'] = insights['defaultKeyStatistics'].get('enterpriseToEbitda')
                temp_data['trailingPE'] = insights['defaultKeyStatistics'].get('trailingPE') # P/L
                temp_data['priceToBook'] = insights['defaultKeyStatistics'].get('priceToBook') # P/VP
                temp_data['forwardPE'] = insights['defaultKeyStatistics'].get('forwardPE')
                temp_data['sharesOutstanding'] = insights['defaultKeyStatistics'].get('sharesOutstanding')
            if insights.get('summaryDetail'):
                temp_data['dividendYield'] = insights['summaryDetail'].get('dividendYield')
                temp_data['marketCap'] = insights['summaryDetail'].get('marketCap')
                temp_data['volume'] = insights['summaryDetail'].get('volume')

        # Supplement/overwrite with yfinance.Ticker.info for more direct access or missing fields
        if info:
            temp_data['returnOnEquity'] = info.get('returnOnEquity', temp_data.get('returnOnEquity'))
            temp_data['enterpriseToEbitda'] = info.get('enterpriseToEbitda', temp_data.get('enterpriseToEbitda'))
            temp_data['trailingPE'] = info.get('trailingPE', temp_data.get('trailingPE'))
            temp_data['priceToBook'] = info.get('priceToBook', temp_data.get('priceToBook'))
            temp_data['dividendYield'] = info.get('dividendYield', temp_data.get('dividendYield'))
            temp_data['netIncomeToCommon'] = info.get('netIncomeToCommon') # Net Income
            temp_data['totalAssets'] = info.get('totalAssets') # For ROA
            temp_data['operatingCashflow'] = info.get('operatingCashflow')
            temp_data['totalDebt'] = info.get('totalDebt')
            temp_data['currentRatio'] = info.get('currentRatio')
            temp_data['sharesOutstanding'] = info.get('sharesOutstanding', temp_data.get('sharesOutstanding'))
            temp_data['grossMargins'] = info.get('grossMargins')
            temp_data['totalRevenue'] = info.get('totalRevenue')
            temp_data['earningsGrowth'] = info.get('earningsGrowth')
            temp_data['revenueGrowth'] = info.get('revenueGrowth')
            temp_data['netMargin'] = info.get('profitMargins') # yfinance calls it profitMargins

        # For Piotroski, we need historical data (financials, balance sheet, cashflow)
        # This part can be very extensive. We'll try to get what's available for current and prior year.
        if yf_ticker_obj:
            try:
                financials = yf_ticker_obj.financials
                balance_sheet = yf_ticker_obj.balance_sheet
                cashflow = yf_ticker_obj.cashflow

                if not financials.empty and len(financials.columns) >= 1:
                    latest_year_col = financials.columns[0]
                    temp_data['NI_curr'] = financials.loc['Net Income', latest_year_col] if 'Net Income' in financials.index else None
                    temp_data['Revenue_curr'] = financials.loc['Total Revenue', latest_year_col] if 'Total Revenue' in financials.index else None
                    temp_data['GrossProfit_curr'] = financials.loc['Gross Profit', latest_year_col] if 'Gross Profit' in financials.index else None
                
                if not balance_sheet.empty and len(balance_sheet.columns) >= 1:
                    latest_year_bs_col = balance_sheet.columns[0]
                    temp_data['TotalAssets_curr'] = balance_sheet.loc['Total Assets', latest_year_bs_col] if 'Total Assets' in balance_sheet.index else None
                    temp_data['TotalLiab_curr'] = balance_sheet.loc['Total Liab', latest_year_bs_col] if 'Total Liab' in balance_sheet.index else None
                    temp_data['CurrentAssets_curr'] = balance_sheet.loc['Total Current Assets', latest_year_bs_col] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['CurrentLiab_curr'] = balance_sheet.loc['Total Current Liabilities', latest_year_bs_col] if 'Total Current Liabilities' in balance_sheet.index else None
                    temp_data['LongTermDebt_curr'] = balance_sheet.loc['Long Term Debt', latest_year_bs_col] if 'Long Term Debt' in balance_sheet.index else None
                
                if not cashflow.empty and len(cashflow.columns) >= 1:
                    latest_year_cf_col = cashflow.columns[0]
                    temp_data['CFO_curr'] = cashflow.loc['Total Cash From Operating Activities', latest_year_cf_col] if 'Total Cash From Operating Activities' in cashflow.index else None

                # Prior year data (if available)
                if len(financials.columns) >= 2:
                    prior_year_col = financials.columns[1]
                    temp_data['NI_prev'] = financials.loc['Net Income', prior_year_col] if 'Net Income' in financials.index else None
                    temp_data['Revenue_prev'] = financials.loc['Total Revenue', prior_year_col] if 'Total Revenue' in financials.index else None
                    temp_data['GrossProfit_prev'] = financials.loc['Gross Profit', prior_year_col] if 'Gross Profit' in financials.index else None
                if len(balance_sheet.columns) >= 2:
                    prior_year_bs_col = balance_sheet.columns[1]
                    temp_data['TotalAssets_prev'] = balance_sheet.loc['Total Assets', prior_year_bs_col] if 'Total Assets' in balance_sheet.index else None
                    temp_data['TotalLiab_prev'] = balance_sheet.loc['Total Liab', prior_year_bs_col] if 'Total Liab' in balance_sheet.index else None
                    temp_data['CurrentAssets_prev'] = balance_sheet.loc['Total Current Assets', prior_year_bs_col] if 'Total Current Assets' in balance_sheet.index else None
                    temp_data['CurrentLiab_prev'] = balance_sheet.loc['Total Current Liabilities', prior_year_bs_col] if 'Total Current Liabilities' in balance_sheet.index else None
                    temp_data['LongTermDebt_prev'] = balance_sheet.loc['Long Term Debt', prior_year_bs_col] if 'Long Term Debt' in balance_sheet.index else None
                # Shares outstanding for prior period might need to be inferred or is harder to get consistently for the exact prior period of financials.
                # For simplicity, we might use current shares outstanding if historical isn't easily available for the check.

            except Exception as e:
                print(f"Error fetching detailed financials for {ativo} via yfinance object: {e}")        
        
        dados_fund[ativo] = temp_data
        
    df_fund = pd.DataFrame.from_dict(dados_fund, orient='index')
    # Convert relevant columns to numeric, coercing errors
    cols_to_numeric = ['returnOnEquity', 'enterpriseToEbitda', 'trailingPE', 'priceToBook', 'dividendYield',
                       'netIncomeToCommon', 'totalAssets', 'operatingCashflow', 'totalDebt', 'currentRatio',
                       'sharesOutstanding', 'grossMargins', 'totalRevenue', 'earningsGrowth', 'revenueGrowth', 'netMargin',
                       'NI_curr', 'Revenue_curr', 'GrossProfit_curr', 'TotalAssets_curr', 'TotalLiab_curr',
                       'CurrentAssets_curr', 'CurrentLiab_curr', 'LongTermDebt_curr', 'CFO_curr',
                       'NI_prev', 'Revenue_prev', 'GrossProfit_prev', 'TotalAssets_prev', 'TotalLiab_prev',
                       'CurrentAssets_prev', 'CurrentLiab_prev', 'LongTermDebt_prev']
    for col in cols_to_numeric:
        if col in df_fund.columns:
            df_fund[col] = pd.to_numeric(df_fund[col], errors='coerce')
    return df_fund

def calcular_piotroski_f_score(df_fund_row):
    """Calculates Piotroski F-Score for a single asset's fundamental data row."""
    score = 0
    try:
        # Profitability
        ni_curr = df_fund_row.get('NI_curr')
        assets_curr = df_fund_row.get('TotalAssets_curr')
        assets_prev = df_fund_row.get('TotalAssets_prev')
        cfo_curr = df_fund_row.get('CFO_curr')
        ni_prev = df_fund_row.get('NI_prev')

        if pd.notna(ni_curr) and ni_curr > 0: score += 1 # 1. Net Income
        if pd.notna(cfo_curr) and cfo_curr > 0: score += 1 # 2. Operating Cash Flow
        
        if pd.notna(ni_curr) and pd.notna(assets_curr) and assets_curr > 0 and \
           pd.notna(ni_prev) and pd.notna(assets_prev) and assets_prev > 0:
            roa_curr = ni_curr / assets_curr
            roa_prev = ni_prev / assets_prev
            if roa_curr > roa_prev: score += 1 # 3. ROA increasing
        
        if pd.notna(cfo_curr) and pd.notna(ni_curr) and cfo_curr > ni_curr: score += 1 # 4. CFO > NI (Quality of Earnings)

        # Leverage, Liquidity, Source of Funds
        lt_debt_curr = df_fund_row.get('LongTermDebt_curr')
        lt_debt_prev = df_fund_row.get('LongTermDebt_prev')
        # Using debt-to-assets ratio change
        if pd.notna(lt_debt_curr) and pd.notna(assets_curr) and assets_curr > 0 and \
           pd.notna(lt_debt_prev) and pd.notna(assets_prev) and assets_prev > 0:
            leverage_curr = lt_debt_curr / assets_curr
            leverage_prev = lt_debt_prev / assets_prev
            if leverage_curr < leverage_prev: score += 1 # 5. Leverage decreasing
        
        curr_assets_curr = df_fund_row.get('CurrentAssets_curr')
        curr_liab_curr = df_fund_row.get('CurrentLiab_curr')
        curr_assets_prev = df_fund_row.get('CurrentAssets_prev')
        curr_liab_prev = df_fund_row.get('CurrentLiab_prev')
        if pd.notna(curr_assets_curr) and pd.notna(curr_liab_curr) and curr_liab_curr > 0 and \
           pd.notna(curr_assets_prev) and pd.notna(curr_liab_prev) and curr_liab_prev > 0:
            current_ratio_curr = curr_assets_curr / curr_liab_curr
            current_ratio_prev = curr_assets_prev / curr_liab_prev
            if current_ratio_curr > current_ratio_prev: score += 1 # 6. Current Ratio increasing
        
        # Shares outstanding check is simplified; yfinance doesn't easily give historical daily shares for exact comparison period.
        # Assuming 'sharesOutstanding' from info is recent. If it's stable or decreasing (e.g. buybacks), it's good.
        # This criterion is often hard to implement perfectly without point-in-time historical share data.
        # For now, we'll skip or simplify: if no info on share issuance, assume neutral.
        # Placeholder: if shares_outstanding_curr <= shares_outstanding_prev: score +=1 (needs shares_outstanding_prev)
        # For now, let's assume this is neutral (no score added or subtracted) due to data limitations.

        # Operating Efficiency
        gp_curr = df_fund_row.get('GrossProfit_curr')
        rev_curr = df_fund_row.get('Revenue_curr')
        gp_prev = df_fund_row.get('GrossProfit_prev')
        rev_prev = df_fund_row.get('Revenue_prev')
        if pd.notna(gp_curr) and pd.notna(rev_curr) and rev_curr > 0 and \
           pd.notna(gp_prev) and pd.notna(rev_prev) and rev_prev > 0:
            gross_margin_curr = gp_curr / rev_curr
            gross_margin_prev = gp_prev / rev_prev
            if gross_margin_curr > gross_margin_prev: score += 1 # 8. Gross Margin increasing

        if pd.notna(rev_curr) and pd.notna(assets_curr) and assets_curr > 0 and \
           pd.notna(rev_prev) and pd.notna(assets_prev) and assets_prev > 0:
            asset_turnover_curr = rev_curr / assets_curr
            asset_turnover_prev = rev_prev / assets_prev
            if asset_turnover_curr > asset_turnover_prev: score += 1 # 9. Asset Turnover increasing
            
    except Exception as e:
        print(f"Error calculating Piotroski F-Score for row: {e}")
        return np.nan
    return score

def calcular_value_composite_score(df_fundamental, metrics_config):
    """
    Calculates a composite value score based on specified metrics and their ranking direction.
    metrics_config: dict, e.g., {'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 'dividendYield': 'higher_is_better'}
    """
    if df_fundamental.empty:
        return pd.Series(dtype=float)
    
    ranked_df = pd.DataFrame(index=df_fundamental.index)
    for metric, ranking_type in metrics_config.items():
        if metric in df_fundamental.columns:
            # Ensure metric is numeric and handle NaNs by not including them in rank or assigning worst rank
            numeric_metric = pd.to_numeric(df_fundamental[metric], errors='coerce')
            if ranking_type == 'lower_is_better':
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True) 
            elif ranking_type == 'higher_is_better':
                ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=False)
            else: # Default to ascending if not specified (lower is better)
                 ranked_df[metric + '_rank'] = numeric_metric.rank(pct=True, ascending=True)
        else:
            print(f"Warning: Metric {metric} not found in fundamental data for Value Composite Score.")

    # Calculate composite score (average of ranks)
    rank_cols = [col for col in ranked_df.columns if '_rank' in col]
    if not rank_cols:
        return pd.Series(dtype=float, index=df_fundamental.index)
        
    composite_score = ranked_df[rank_cols].mean(axis=1)
    return composite_score

# --- Econometric Models ---
def calcular_volatilidade_garch(returns_series, p=1, q=1):
    """Calculates forecasted GARCH volatility."""
    if returns_series.empty or len(returns_series) < (p + q + 20): # Need enough data points
        print(f"Not enough data points for GARCH model on series: {returns_series.name}")
        return np.nan
    try:
        # Scale returns for GARCH model (common practice)
        scaled_returns = returns_series * 100
        model = arch_model(scaled_returns, vol='Garch', p=p, q=q, rescale=False)
        res = model.fit(disp='off', show_warning=False)
        forecast = res.forecast(horizon=1)
        # Annualize daily forecasted variance. Variance is h.1, take sqrt for vol, then scale back from %
        # The forecast variance is already for the scaled returns.
        forecasted_std_dev_daily_scaled = np.sqrt(forecast.variance.iloc[-1,0])
        forecasted_std_dev_daily = forecasted_std_dev_daily_scaled / 100.0
        # Annualize: multiply daily std dev by sqrt(252)
        return forecasted_std_dev_daily * np.sqrt(252)
    except Exception as e:
        print(f"Error fitting GARCH for {returns_series.name}: {e}")
        return np.nan # Return historical if GARCH fails

def get_fama_french_factors(start_date, end_date, risk_free_rate_series=None):
    """ 
    Downloads Fama-French 3 factors (Mkt-RF, SMB, HML) and Momentum (WML) as proxies using yfinance.
    This is a simplified proxy approach.
    Returns a DataFrame with factor returns.
    """
    print("Fetching Fama-French factor proxies...")
    # Proxies (US-centric, adjust for Brazil if primary market)
    # Market: S&P 500 (^GSPC)
    # SMB: Russell 2000 (^RUT) - S&P 500 (^GSPC)
    # HML: Russell 1000 Value (^RLV) - Russell 1000 Growth (^RLG) (Example)
    # WML: Momentum ETF (MTUM) (Example)
    # Risk-Free: Use a constant or fetch e.g. ^IRX (13 Week Treasury Bill)
    
    factor_tickers = {
        'MKT_PROXY': '^GSPC', # Market proxy
        'SMB_SMALL': '^RUT',  # Small cap proxy for SMB
        'SMB_LARGE': '^GSPC', # Large cap proxy for SMB
        'HML_VALUE': 'IVE',   # Example: iShares S&P 500 Value ETF for HML
        'HML_GROWTH':'IVW',  # Example: iShares S&P 500 Growth ETF for HML
        'WML_MOM': 'MTUM'     # Example: iShares MSCI USA Momentum Factor ETF for WML
    }
    # For Brazilian market, one might use: IBOV for MKT, SMLL for SMB_SMALL, IVVB11 or similar for large cap comparison,
    # specific value/growth ETFs if available, or a momentum index/ETF.

    factor_data_all_cols = yf.download(list(factor_tickers.values()), start=start_date, end=end_date, progress=False)
    if factor_data_all_cols.empty:
        print("Could not download factor proxy data (empty main DataFrame).")
        return pd.DataFrame()
    try:
        # With auto_adjust=True (default in yfinance), 'Close' column is adjusted.
        # For multiple tickers, yf.download returns a multi-index DataFrame.
        # Accessing ['Close'] should give a DataFrame of Close prices with tickers as columns.
        factor_data_raw = factor_data_all_cols['Close']
    except KeyError:
        try:
            # Fallback if 'Close' isn't there but 'Adj Close' is (less likely with modern yfinance defaults)
            factor_data_raw = factor_data_all_cols['Adj Close']
        except KeyError:
            print("Error: Neither 'Close' nor 'Adj Close' found as a primary column key for factor data.")
            available_keys = factor_data_all_cols.columns.levels[0] if isinstance(factor_data_all_cols.columns, pd.MultiIndex) else factor_data_all_cols.columns
            print(f"Available primary column keys: {available_keys}")
            return pd.DataFrame()

    if factor_data_raw.empty:
        print("Extracted factor price data ('Close' or 'Adj Close') is empty.")
        return pd.DataFrame()
    if factor_data_raw.empty:
        print("Could not download factor proxy data.")
        return pd.DataFrame()

    factor_returns = factor_data_raw.pct_change().dropna()

    factors_df = pd.DataFrame(index=factor_returns.index)

    # Risk-Free Rate
    if risk_free_rate_series is None:
        # Try to fetch ^IRX (13 Week Treasury Bill) as daily risk-free proxy
        rf_data_raw = yf.download('^IRX', start=start_date, end=end_date, progress=False)
        if not rf_data_raw.empty:
            try:
                # Try to access 'Close' first, as it's often the adjusted price with yfinance defaults
                rf_data_series = rf_data_raw['Close']
            except KeyError:
                try:
                    # Fallback to 'Adj Close' if 'Close' is not found
                    rf_data_series = rf_data_raw['Adj Close']
                except KeyError:
                    print("Warning: Neither 'Close' nor 'Adj Close' found for ^IRX. RF rate might be inaccurate.")
                    rf_data_series = pd.Series(dtype=float) # Empty series
            
            if not rf_data_series.empty and pd.api.types.is_numeric_dtype(rf_data_series):
                # ^IRX is an annualized yield. So, daily_rf = (yield / 100) / 252.
                daily_rf = (rf_data_series / 100) / 252 
                factors_df['RF'] = daily_rf.reindex(factors_df.index, method='ffill') # Align and ffill
            else:
                print("Warning: ^IRX data for risk-free rate is empty or not numeric after extraction. Using default constant / 252.")
                factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
        else:
            print("Warning: Could not fetch ^IRX for risk-free rate (empty DataFrame). Using default constant / 252.")
            factors_df['RF'] = (RISK_FREE_RATE_DEFAULT / 252)
    else:
        factors_df['RF'] = risk_free_rate_series
    
    factors_df.fillna(method='ffill', inplace=True) # Fill NaNs in RF if any
    factors_df.dropna(inplace=True) # Drop any remaining NaNs after join

    # Mkt-RF
    if factor_tickers['MKT_PROXY'] in factor_returns.columns and 'RF' in factors_df.columns:
        mkt_rf_series = factor_returns[factor_tickers['MKT_PROXY']] - factors_df['RF']
        factors_df['Mkt-RF'] = mkt_rf_series.loc[factors_df.index] # Align indices

    # SMB
    if factor_tickers['SMB_SMALL'] in factor_returns.columns and factor_tickers['SMB_LARGE'] in factor_returns.columns:
        smb_series = factor_returns[factor_tickers['SMB_SMALL']] - factor_returns[factor_tickers['SMB_LARGE']]
        factors_df['SMB'] = smb_series.loc[factors_df.index]

    # HML
    if factor_tickers['HML_VALUE'] in factor_returns.columns and factor_tickers['HML_GROWTH'] in factor_returns.columns:
        hml_series = factor_returns[factor_tickers['HML_VALUE']] - factor_returns[factor_tickers['HML_GROWTH']]
        factors_df['HML'] = hml_series.loc[factors_df.index]

    # WML (Momentum)
    if factor_tickers['WML_MOM'] in factor_returns.columns and 'RF' in factors_df.columns:
        # Momentum factor is often Mkt_Mom - RF or just Mkt_Mom returns
        wml_series = factor_returns[factor_tickers['WML_MOM']] - factors_df['RF'] # Example: Mom-RF
        factors_df['WML'] = wml_series.loc[factors_df.index]
    
    factors_df.dropna(inplace=True)
    return factors_df[['Mkt-RF', 'SMB', 'HML', 'WML', 'RF']].copy()

def estimar_fatores_alpha_beta(asset_returns_series, factor_df):
    """Estimates alpha and betas using OLS regression against Fama-French factors."""
    if asset_returns_series.empty or factor_df.empty:
        return np.nan, {} # Return NaN for alpha, empty dict for betas
    
    # Align data
    common_index = asset_returns_series.index.intersection(factor_df.index)
    if len(common_index) < 20: # Need enough observations for regression
        print(f"Not enough common data points for factor regression on {asset_returns_series.name}")
        return np.nan, {}
        
    y = asset_returns_series.loc[common_index] - factor_df.loc[common_index, 'RF'] # Excess returns
    X = factor_df.loc[common_index, ['Mkt-RF', 'SMB', 'HML', 'WML']]
    X = sm.add_constant(X) # Add intercept term for alpha
    X.dropna(inplace=True) # Drop rows with NaNs in factors
    y = y.loc[X.index] # Align y with X after dropping NaNs

    if len(y) < 20:
        print(f"Not enough data points after alignment for factor regression on {asset_returns_series.name}")
        return np.nan, {}
        
    try:
        model = sm.OLS(y, X, missing='drop').fit()
        alpha = model.params.get('const', np.nan) * 252 # Annualize alpha
        betas = model.params.drop('const', errors='ignore').to_dict()
        return alpha, betas
    except Exception as e:
        asset_name_for_log = asset_returns_series.name if isinstance(asset_returns_series, pd.Series) else str(asset_returns_series.columns[0]) if isinstance(asset_returns_series, pd.DataFrame) and not asset_returns_series.empty else "Unknown Asset"
        print(f"Error in OLS regression for {asset_name_for_log}: {e}")
        return np.nan, {}

def prever_retornos_arima(returns_series, order=(5,1,0)):
    """Predicts next period return using ARIMA model."""
    if returns_series.empty or len(returns_series) < (sum(order) + 20):
        print(f"Not enough data points for ARIMA model on series: {returns_series.name}")
        return np.nan
    try:
        model = ARIMA(returns_series, order=order)
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=1)
        return forecast.iloc[0]
    except Exception as e:
        # Common errors: non-stationarity, convergence issues
        print(f"Error fitting ARIMA for {returns_series.name}: {e}. Trying auto_arima or simpler model.")
        # Fallback to simpler model or auto_arima if pmdarima was installed
        try:
            model = ARIMA(returns_series, order=(1,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=1)
            return forecast.iloc[0]
        except Exception as e2:
            print(f"Fallback ARIMA failed for {returns_series.name}: {e2}")
            return np.nan

# --- Portfolio Optimization ---
def calcular_metricas_portfolio(pesos, retornos_medios_anuais, matriz_covariancia_anual, taxa_livre_risco):
    retorno_portfolio = np.sum(retornos_medios_anuais * pesos)
    volatilidade_portfolio = np.sqrt(np.dot(pesos.T, np.dot(matriz_covariancia_anual, pesos)))
    sharpe_ratio = (retorno_portfolio - taxa_livre_risco) / volatilidade_portfolio if volatilidade_portfolio != 0 else -np.inf
    return retorno_portfolio, volatilidade_portfolio, sharpe_ratio

def ajustar_retornos_esperados(base_retornos_medios_anuais, df_fundamental, alphas, arima_forecasts, quant_value_scores, piotroski_scores):
    """Adjusts expected returns based on fundamental scores, alpha, and ARIMA forecasts."""
    adjusted_retornos = base_retornos_medios_anuais.copy()
    for ativo in adjusted_retornos.index:
        adjustment_factor = 1.0
        # Piotroski Score: Higher is better. Scale score (0-9) to an adjustment factor.
        if piotroski_scores is not None and ativo in piotroski_scores and pd.notna(piotroski_scores[ativo]):
            # Example: score of 0-2: 0.8, 3-4: 0.9, 5-6: 1.0, 7-9: 1.1-1.2
            pscore = piotroski_scores[ativo]
            if pscore <= 2: adjustment_factor *= 0.9 # Penalize weak fundamentals
            elif pscore >=7: adjustment_factor *= 1.1 # Reward strong fundamentals
        
        # Quant Value Score: Higher is better (assuming it's 0-1, higher means more undervalued)
        if quant_value_scores is not None and ativo in quant_value_scores and pd.notna(quant_value_scores[ativo]):
            # Example: score > 0.7 adds 5%, score < 0.3 subtracts 5%
            qscore = quant_value_scores[ativo]
            adjustment_factor *= (1 + (qscore - 0.5) * 0.2) # e.g. score 0 -> 0.9, 0.5 -> 1.0, 1.0 -> 1.1

        adjusted_retornos[ativo] *= adjustment_factor

        # Add Alpha (already annualized)
        if alphas is not None and ativo in alphas and pd.notna(alphas[ativo]):
            adjusted_retornos[ativo] += alphas[ativo]
        
        # Add ARIMA forecast (this is tricky, as ARIMA forecasts 1-step ahead daily return)
        # For simplicity, if ARIMA predicts a positive daily return, add a small kicker, if negative, a small drag.
        # A more robust approach would be to use ARIMA to forecast for a longer horizon or incorporate its confidence.
        if arima_forecasts is not None and ativo in arima_forecasts and pd.notna(arima_forecasts[ativo]):
            # Example: annualize the daily forecast and add a fraction of it
            annualized_arima_forecast = arima_forecasts[ativo] * 252
            adjusted_retornos[ativo] = (adjusted_retornos[ativo] * 0.8) + (annualized_arima_forecast * 0.2) # Weighted average
            
    return adjusted_retornos

def otimizar_portfolio_scipy(
    ativos,
    df_retornos_historicos,
    df_fundamental_completo=None, # Contains all fundamental data including Piotroski, Quant Value
    fama_french_factors=None,
    taxa_livre_risco=RISK_FREE_RATE_DEFAULT,
    pesos_atuais=None, # Dict: {'ATIVO': peso_decimal}
    restricoes_pesos_min_max=None, # Dict: {'ATIVO': (min_pct, max_pct)} or global (min_pct, max_pct)
    objetivo='max_sharpe' # 'max_sharpe', 'min_volatility', 'target_return'
    # target_return_value=None # if objetivo is 'target_return'
):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    
    retornos_considerados = df_retornos_historicos[ativos].copy()
    if retornos_considerados.shape[1] != len(ativos) or retornos_considerados.isnull().values.any():
        print("Warning: Missing return data for some assets or all assets. Dropping NaNs.")
        retornos_considerados.dropna(axis=1, how='all', inplace=True) # Drop columns that are all NaN
        retornos_considerados.dropna(axis=0, how='any', inplace=True) # Drop rows with any NaN
        ativos = list(retornos_considerados.columns)
        if not ativos:
            print("Error: No valid asset data left after cleaning for optimization.")
            return None, None, []

    retornos_medios_diarios = retornos_considerados.mean()
    base_retornos_medios_anuais = retornos_medios_diarios * 252
    matriz_covariancia_diaria = retornos_considerados.cov()
    matriz_covariancia_anual_historica = matriz_covariancia_diaria * 252
    num_ativos = len(ativos)

    # --- Advanced Adjustments ---
    adj_retornos_esperados = base_retornos_medios_anuais.copy()
    adj_matriz_covariancia = matriz_covariancia_anual_historica.copy()

    alphas = {}
    arima_forecasts = {}
    garch_volatilities = {}

    if df_fundamental_completo is not None and not df_fundamental_completo.empty:
        # Calculate Piotroski and Quant Value scores if not already present
        if 'Piotroski_F_Score' not in df_fundamental_completo.columns:
            df_fundamental_completo['Piotroski_F_Score'] = df_fundamental_completo.apply(calcular_piotroski_f_score, axis=1)
        
        # Define metrics for Quant Value Score (example)
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

        # Filter assets based on fundamental scores (example: Piotroski >= 3)
        # assets_to_keep_indices = piotroski_scores_series[piotroski_scores_series >= 3].index
        # if len(assets_to_keep_indices) < num_ativos and len(assets_to_keep_indices) > 0:
        #     print(f"Filtering assets based on Piotroski score. Kept: {list(assets_to_keep_indices)}")
        #     ativos = list(assets_to_keep_indices)
        #     retornos_considerados = retornos_considerados[ativos]
        #     base_retornos_medios_anuais = base_retornos_medios_anuais.loc[ativos]
        #     adj_matriz_covariancia = adj_matriz_covariancia.loc[ativos, ativos]
        #     num_ativos = len(ativos)
        # else:
        #     print("Piotroski filter not applied or resulted in no assets.")

        # Econometric adjustments
        for ativo in ativos:
            asset_returns = retornos_considerados[ativo].dropna()
            if fama_french_factors is not None and not fama_french_factors.empty:
                alpha, _ = estimar_fatores_alpha_beta(asset_returns, fama_french_factors)
                alphas[ativo] = alpha
            
            arima_forecasts[ativo] = prever_retornos_arima(asset_returns)
            garch_vol = calcular_volatilidade_garch(asset_returns)
            if pd.notna(garch_vol) and garch_vol > 0:
                # Replace diagonal of covariance matrix with GARCH forecasted vol (squared for variance)
                # This is a simplification; a full GARCH-DCC model would be more robust for covariance.
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

    # --- Optimization using SciPy Minimize ---
    def portfolio_volatility(weights, cov_matrix):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    def portfolio_return(weights, expected_returns):
        return np.sum(expected_returns * weights)

    def neg_sharpe_ratio(weights, expected_returns, cov_matrix, risk_free_rate):
        p_return = portfolio_return(weights, expected_returns)
        p_volatility = portfolio_volatility(weights, cov_matrix)
        if p_volatility == 0: return -np.inf # Should be large positive due to negation
        return -(p_return - risk_free_rate) / p_volatility

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}) # Sum of weights = 1
    
    # Bounds for each weight (0 to 1 by default)
    # Incorporate user-defined min/max per asset or global
    bounds = []
    default_min_b, default_max_b = 0.0, 1.0
    if isinstance(restricoes_pesos_min_max, tuple) and len(restricoes_pesos_min_max) == 2:
        default_min_b, default_max_b = restricoes_pesos_min_max[0], restricoes_pesos_min_max[1]

    for ativo in ativos:
        min_b, max_b = default_min_b, default_max_b
        if isinstance(restricoes_pesos_min_max, dict) and ativo in restricoes_pesos_min_max:
            min_b, max_b = restricoes_pesos_min_max[ativo]
        elif pesos_atuais and ativo in pesos_atuais: # Use current weights to inform bounds if specific restrictions not given
            # Example: allow deviation of +/- X% from current weight, capped by 0-1
            # This needs careful definition based on user's intent for "ponto de partida"
            pass # For now, simple 0-1 bounds unless specified
        bounds.append((min_b, max_b))
    
    # Initial guess for weights
    # If pesos_atuais provided and sum to 1, use them. Otherwise, equal weights.
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
        opt_func = lambda w, cov: portfolio_volatility(w, cov) # Directly minimize volatility
    # elif objetivo == 'target_return':
        # constraints = (
        #     {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1},
        #     {'type': 'eq', 'fun': lambda weights: portfolio_return(weights, adj_retornos_esperados.values) - target_return_value}
        # )
        # opt_args = (adj_matriz_covariancia.values,)
        # opt_func = lambda w, cov: portfolio_volatility(w, cov)
    else:
        raise ValueError("Invalid objective function specified.")

    optimized_result = minimize(opt_func, initial_weights, args=opt_args,
                                method='SLSQP', bounds=bounds, constraints=constraints)

    if not optimized_result.success:
        print(f"Optimization failed: {optimized_result.message}")
        # Fallback to simpler simulation or return None
        # For now, we'll return the (potentially non-optimal) weights if any
        # return None, None, [] 

    optimal_weights = optimized_result.x
    # Clean very small weights to zero
    optimal_weights[optimal_weights < 1e-4] = 0
    optimal_weights = optimal_weights / np.sum(optimal_weights) # Re-normalize

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

    # For plotting the frontier, we'd typically vary target return and minimize volatility.
    # The Monte Carlo simulation from the original code can still be used for visualization if needed.
    # Here, we are returning a single optimized portfolio based on the objective.
    # To generate a frontier with scipy, one would iterate `minimize` for different `target_return_value`s.
    
    # Placeholder for frontier points if needed by Streamlit app (can be generated via Monte Carlo or iterative optimization)
    fronteira_pontos_simulados = [] 
    # Example: Run a simple Monte Carlo for visualization if needed
    # num_portfolios_simulados = 5000
    # for _ in range(num_portfolios_simulados):
    #     sim_weights = np.random.random(num_ativos)
    #     sim_weights /= np.sum(sim_weights)
    #     r, v, s = calcular_metricas_portfolio(sim_weights, adj_retornos_esperados, adj_matriz_covariancia, taxa_livre_risco)
    #     fronteira_pontos_simulados.append({'retorno': r, 'volatilidade': v, 'sharpe': s, 'pesos': dict(zip(ativos, sim_weights))})

    return portfolio_otimizado, fronteira_pontos_simulados # Returning single optimized portfolio and empty frontier for now

# --- Original Markowitz (Monte Carlo) for comparison or baseline ---
def otimizar_portfolio_markowitz_mc(ativos, df_retornos_historicos, taxa_livre_risco=RISK_FREE_RATE_DEFAULT, num_portfolios_simulados=10000):
    if df_retornos_historicos.empty or len(ativos) == 0:
        return None, None, []
    
    retornos_considerados = df_retornos_historicos[ativos].copy()
    # Ensure no NaN values in the columns being used for calculation
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

# --- Allocation Suggestion (from original, can be kept or adapted) ---
def sugerir_alocacao_novo_aporte(
    current_portfolio_composition_values: dict, 
    new_capital: float,
    target_portfolio_weights_decimal: dict,
):
    # (Implementation from original script - can be used as is or adapted)
    # ... (ensure this function is robust)
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
    else: # total_capital_needed_for_target > new_capital
        for asset_ticker, needed_amount in purchases_to_reach_target.items():
            actual_purchases[asset_ticker] = (needed_amount / total_capital_needed_for_target) * new_capital
        surplus_capital = 0.0
    actual_purchases = {k: v for k, v in actual_purchases.items() if v > 0.01}
    return actual_purchases, surplus_capital

# --- Main execution block for testing ---
if __name__ == '__main__':
    print("Running financial_analyzer_enhanced.py tests...")
    test_ativos = ['AAPL', 'MSFT', 'GOOGL'] # Use US tickers for broader data availability with yfinance
    # test_ativos_br = ['PETR4.SA', 'VALE3.SA', 'ITUB4.SA'] # For Brazilian assets
    
    start_date = (datetime.today() - timedelta(days=10*365)).strftime('%Y-%m-%d') # 10 years data
    end_date = datetime.today().strftime('%Y-%m-%d')

    # 1. Test Historical Data Fetching
    print("\n--- Testing Historical Data ---")
    df_retornos = obter_dados_historicos_yf(test_ativos, start_date_str=start_date, end_date_str=end_date)
    if not df_retornos.empty:
        print(f"Successfully fetched historical returns for {len(df_retornos.columns)} assets. Shape: {df_retornos.shape}")
        print(df_retornos.head())
    else:
        print("Failed to fetch historical returns.")

    # 2. Test Fundamental Data Fetching
    print("\n--- Testing Fundamental Data ---")
    df_fund_data = obter_dados_fundamentalistas_detalhados(test_ativos)
    if not df_fund_data.empty:
        print(f"Successfully fetched fundamental data. Shape: {df_fund_data.shape}")
        # Calculate Piotroski F-Score
        df_fund_data['Piotroski_F_Score'] = df_fund_data.apply(calcular_piotroski_f_score, axis=1)
        # Calculate Value Composite Score
        vc_metrics_test = {
            'trailingPE': 'lower_is_better', 'priceToBook': 'lower_is_better', 
            'enterpriseToEbitda': 'lower_is_better', 'dividendYield': 'higher_is_better',
            'returnOnEquity': 'higher_is_better', 'netMargin': 'higher_is_better'
        }
        df_fund_data['Quant_Value_Score'] = calcular_value_composite_score(df_fund_data, vc_metrics_test)
        print(df_fund_data[['ticker', 'Piotroski_F_Score', 'Quant_Value_Score']].head())
    else:
        print("Failed to fetch fundamental data.")

    # 3. Test Fama-French Factor Proxies
    print("\n--- Testing Fama-French Factors ---")
    ff_factors = get_fama_french_factors(start_date, end_date)
    if not ff_factors.empty:
        print(f"Successfully fetched Fama-French factor proxies. Shape: {ff_factors.shape}")
        print(ff_factors.head())
    else:
        print("Failed to fetch Fama-French factors.")

    # 4. Test Advanced Optimization (SciPy based)
    print("\n--- Testing Advanced Portfolio Optimization (SciPy) ---")
    if not df_retornos.empty and not df_fund_data.empty and not ff_factors.empty:
        # Ensure df_fund_data index is the ticker symbol for reindexing later
        df_fund_data.set_index('ticker', inplace=True, drop=False) # Keep ticker column if needed elsewhere
        
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
                elif isinstance(v, dict): print(f"  {k}: Present") # For dicts like garch_volatilities
                elif isinstance(v, float): print(f"  {k}: {v:.4f}")
                else: print(f"  {k}: {v}")
        else:
            print("Advanced portfolio optimization failed.")
    else:
        print("Skipping advanced optimization test due to missing data (returns, fundamentals, or factors).")

    # 5. Test Original Markowitz (Monte Carlo)
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


