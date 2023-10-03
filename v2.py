#!/usr/bin/env python
# coding: utf-8

# In[73]:


import pandas as pd 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
pd.set_option('display.max_rows', None)


# In[250]:


def fetch_financial_data(ticker):
    global wherefile
    balance_sheet_file = f"data/{ticker}_balance_sheet.csv"
    income_statement_file = f"data/{ticker}_income_statement.csv"
    cashflow_file = f"data/{ticker}_cashflow.csv"
    info_file = f"data/{ticker}_info.csv"
    
    # Check if files already exist
    if os.path.exists(balance_sheet_file) and os.path.exists(income_statement_file) and os.path.exists(cashflow_file) and os.path.exists(info_file):
        balance_sheet = pd.read_csv(balance_sheet_file, index_col=0)
        income_statement = pd.read_csv(income_statement_file, index_col=0)
        cashflow = pd.read_csv(cashflow_file, index_col=0)
        c_info = pd.read_csv(info_file, index_col=0)
        wherefile = "csv"
    else:
        wherefile = "yfinance"
        stock = yf.Ticker(ticker)
        balance_sheet = stock.balance_sheet
        income_statement = stock.financials
        cashflow = stock.cashflow
        c_info = stock.info
        
        # Create empty dictionary to store the data
        c_info = {
            'Ticker': [],
            'Trailing P/E': [],
            'Value to Book': []
        }

        balance_sheet = balance_sheet.T
        balance_sheet.index.name = 'year'
        balance_sheet = balance_sheet.sort_index()
        balance_sheet = balance_sheet['2020':'2022']
        balance_sheet.index = balance_sheet.index.year

        income_statement = income_statement.T
        income_statement.index.name = 'year'
        income_statement = income_statement.sort_index()
        income_statement = income_statement['2020':'2022']
        income_statement.index = income_statement.index.year

        cashflow = cashflow.T
        cashflow.index.name = 'year'
        cashflow = cashflow.sort_index()
        cashflow = cashflow['2020':'2022']
        cashflow.index = cashflow.index.year
        
        # Short-term Solvency
        balance_sheet['Current Ratio'] = balance_sheet["Current Assets"] / balance_sheet["Current Liabilities"]
        balance_sheet['Quick Ratio'] = (balance_sheet["Current Assets"] - balance_sheet["Inventory"]) / balance_sheet["Current Liabilities"]
        balance_sheet['Cash Ratio'] = balance_sheet["Cash Cash Equivalents And Short Term Investments"] / balance_sheet["Current Liabilities"]
        
        # Long-term Solvency
        balance_sheet['Debt Ratio'] = balance_sheet["Total Liabilities Net Minority Interest"] / balance_sheet["Total Assets"]
        balance_sheet['Debt to Equity Ratio'] = balance_sheet["Total Liabilities Net Minority Interest"] / balance_sheet["Stockholders Equity"]
        balance_sheet['Equity Multiplier'] = balance_sheet["Total Assets"] / balance_sheet["Stockholders Equity"]
        income_statement['Times Interest Earned'] = income_statement["EBIT"] / income_statement["Interest Expense"]
        
        # Asset Utilization
        income_statement['Inventory Turnover'] = income_statement["Cost Of Revenue"] / balance_sheet["Inventory"]   
        income_statement['Total Asset Turnover'] = income_statement["Total Revenue"] / balance_sheet["Total Assets"]
        
        # Profitability
        income_statement['Profit Margin'] = income_statement["Net Income"] / income_statement["Total Revenue"]
        income_statement['Return on Assets'] = income_statement["Net Income"] / balance_sheet["Total Assets"]

        # Market value
        c_info['Ticker'].append(ticker)
        c_info['Trailing P/E'].append(stock.info["trailingPE"])
        c_info['Value to Book'].append(stock.info["priceToBook"])
        
        c_info = pd.DataFrame(c_info)
        c_info.set_index('Ticker', inplace=True)
        
        # Save the dataframes to CSV files
        balance_sheet.to_csv(balance_sheet_file)
        income_statement.to_csv(income_statement_file)
        cashflow.to_csv(cashflow_file)
        c_info.to_csv(info_file)
    
    return balance_sheet, income_statement, c_info

st.sidebar.header('Compare financial ratio between 2 companies')
ticker1 = st.sidebar.selectbox('Company 1',('L.TO','MSFT','WMT'))
ticker2 = st.sidebar.selectbox('Company 2',('MRU.TO','MSFT','WMT'))

st.header(f"{ticker1} vs {ticker2}", divider='rainbow')
tickers = [ticker1,ticker2]
start = '2020-01-01'
year = 2022

balance_sheets = {}
income_statements = {}
c_infos = {}
price_df = pd.DataFrame()

for ticker in tickers:
    price_file = f'data/{ticker}_price.csv'
    balance_sheet, income_statement, c_info = fetch_financial_data(ticker)
    
    balance_sheets[ticker] = balance_sheet
    income_statements[ticker] = income_statement
    c_infos[ticker] = c_info
    
    # Check if CSV file for the ticker exists
    if os.path.exists(price_file):
        # Load data from CSV file
        price_df[ticker] = pd.read_csv(price_file, index_col=0, parse_dates=True)["Close"]
    else:
        # Fetch data using yf.download()
        data = yf.download(ticker, start=start, progress=False)["Close"]
        price_df[ticker] = data
        # Save to individual CSV file
        data.to_csv(price_file)


# Market value
mv_ratios={}
for ticker in tickers:
    pe = c_infos[ticker]['Trailing P/E'].values[0]
    bv = c_infos[ticker]['Value to Book'].values[0]
    mv_ratios[ticker] = [pe, bv]
mv_ratios = pd.DataFrame(mv_ratios, index=['P/E', 'BV'])

# In[236]:

# Creating a dictionary to store the ratio dataframes
long_ratio = {}
short_ratio = {}
asset_utilization_ratio = {}
profitability_ratio = {}

long_ratio["Debt Ratio"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Debt Ratio'] for ticker in tickers
})

long_ratio["Debt to Equity Ratio"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Debt to Equity Ratio'] for ticker in tickers
})
long_ratio["Equity Multiplier"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Equity Multiplier'] for ticker in tickers
})

long_ratio["Times Interest Earned"] = pd.DataFrame({
    ticker: income_statements[ticker]['Times Interest Earned'] for ticker in tickers
})

# Short
short_ratio["Current Ratio"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Current Ratio'] for ticker in tickers
})
short_ratio["Quick Ratio"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Quick Ratio'] for ticker in tickers
})
short_ratio["Cash Ratio"] = pd.DataFrame({
    ticker: balance_sheets[ticker]['Cash Ratio'] for ticker in tickers
})

# Asset Utilization
asset_utilization_ratio['Inventory Turnover'] = pd.DataFrame({
    ticker: income_statements[ticker]['Inventory Turnover'] for ticker in tickers
})
asset_utilization_ratio['Total Asset Turnover'] = pd.DataFrame({
    ticker: income_statements[ticker]['Total Asset Turnover'] for ticker in tickers
})

# Profitability
profitability_ratio['Profit Margin'] = pd.DataFrame({
    ticker: income_statements[ticker]['Total Asset Turnover'] for ticker in tickers
})
profitability_ratio['Return on Assets'] = pd.DataFrame({
    ticker: income_statements[ticker]['Return on Assets'] for ticker in tickers
})

def plotLineChart(ratio_df, ratio_name):

    latest_year = ratio_df.index[-1]  # Get the latest year from the index
    print(f"{ratio_name} for the year {latest_year}:")
    st.markdown(f"##### {ratio_name} for the year {latest_year}:")
    
    # Convert the series to a dictionary and print
    ratio_values = ratio_df.loc[latest_year].to_dict()
    for ticker, value in ratio_values.items():
        print(f"{ticker}: {value}")
        st.text(f"{ticker}: {value:.4f}")
    
    print("\n")

    plt.figure(figsize=(12, 6))
    sns.lineplot(data=ratio_df,linewidth=2)

    # Ensure each year in the dataframe index gets a corresponding tick on the x-axis
    plt.xticks(ticks=ratio_df.index, labels=ratio_df.index)

    plt.title(f'{ratio_name} Over Years', fontsize=20)
    #plt.ylabel(ratio_name)
    plt.ylabel(None)
    plt.xlabel('Year', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.3)
    plt.legend(title='Company',loc='center right', fontsize=18)
    plt.tight_layout()
    #st.pyplot(plt)
    plt.show()
    st.pyplot(plt)


st.markdown(f"### :blue[Long term Solvency Ratio]")
current_long_ratios = pd.DataFrame(columns=tickers)
for ratio_name, ratio_df in long_ratio.items():
    
    for ticker in tickers:
        current_long_ratios.loc[ratio_name, ticker] = ratio_df[ticker].loc[year]
    plotLineChart(ratio_df, ratio_name)


st.markdown(f"### :blue[Short term Solvency Ratio]")
current_short_ratios = pd.DataFrame(columns=tickers)
for ratio_name, ratio_df in short_ratio.items():
    
    for ticker in tickers:
        current_short_ratios.loc[ratio_name, ticker] = ratio_df[ticker].loc[year]
    
    plotLineChart(ratio_df, ratio_name)

st.markdown(f"### :blue[Asset Utilization Ratio]")
current_assetutil_ratios = pd.DataFrame(columns=tickers)
for ratio_name, ratio_df in asset_utilization_ratio.items():
    
    for ticker in tickers:
        current_assetutil_ratios.loc[ratio_name, ticker] = ratio_df[ticker].loc[year]
    
    plotLineChart(ratio_df, ratio_name)

st.markdown(f"### :blue[Profitability Ratio]")
current_profit_ratios = pd.DataFrame(columns=tickers)
for ratio_name, ratio_df in profitability_ratio.items():
    
    for ticker in tickers:
        current_profit_ratios.loc[ratio_name, ticker] = ratio_df[ticker].loc[year]
    
    plotLineChart(ratio_df, ratio_name)

def ratio_comparison(compare_ratio, compare_name):
    global year
    # Visualization
    sns.set(style="whitegrid")
    compare_ratio.plot(kind="bar", figsize=(10, 6))
    plt.title(f"{compare_name} Comparison year {year}", fontsize=20)
    #plt.xlabel("Ratios", fontsize=16)
    #plt.ylabel("Value", fontsize=16)
    plt.xlabel(None)
    plt.ylabel(None)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, which='both', linestyle='--', linewidth=0.3)
    plt.tight_layout()
    plt.legend(title="Companies")
    plt.show()
    st.pyplot(plt)

st.markdown(f"### :blue[financials Ratio Comparison]")

ratio_comparison(current_long_ratios, "Long-term Solvency Ratios")
ratio_comparison(current_short_ratios, "Short-term Solvency Ratios")
ratio_comparison(current_assetutil_ratios, "Asset Utilization Ratios")
ratio_comparison(current_profit_ratios, "Profitability Ratios")


st.markdown(f"### :blue[Market Value]")
st.table(mv_ratios)

# Resample
price_df = price_df.resample('W').last()

# In[297]:

st.markdown(f"### Portfolio return")
df_norm = (price_df/price_df.iloc[0]) * 100
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12.0, 8.0)

df_norm.plot()
plt.title(f"portfolio return")
#plt.show()
st.pyplot(plt)


# In[298]:


df_pct = price_df.pct_change()
df_pct.dropna(inplace=True)


# In[299]:


correlation_matrix = df_pct.corr()
correlation_coefficient = correlation_matrix.iloc[0, 1]

st.markdown(f"### Correlation coefficient between {tickers[0]} and {tickers[1]} : {correlation_coefficient:.4f}")
print(f"Correlation coefficient between {tickers[0]} and {tickers[1]} : {correlation_coefficient:.4f}")


# In[307]:


sns.jointplot(x=price_df[tickers[0]], y=price_df[tickers[1]], kind='reg', height=10);
print(f'Correlation: {correlation_coefficient:.4f}')
#plt.show()
st.pyplot(plt)

st.text(f"data fetched from {wherefile}")
