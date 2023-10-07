import pandas as pd 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import os
import json
pd.set_option('display.max_rows', None)
palette=['#4c72b0', '#dd8452','#939393']
sns.set_palette(palette)

#tickers = ["AAPL", "MSFT", "GOOGL"]
tickers = ["L.TO","MRU.TO"]
index_benchmark = "^GSPTSE"
short_ratios_col = ['Current Ratio','Quick Ratio','Cash Ratio']
long_ratios_col = ['Debt to Equity Ratio','Debt Ratio','Equity Multiplier']
asset_ratios_col = ['Inventory Turnover','Total Asset Turnover','Receivables Turnover']
profit_ratios_col = ['Net Profit Margin','ROA','ROE']

start_date = "2020-01-01"
end_date = "2022-12-31"
ticker_to_name = {}
price_df = pd.DataFrame()

def calculate_ratios(balance_sheet, income_statement,cashflow):
        # Liquidity Ratios
        current_ratio = balance_sheet.loc['Current Assets'] / balance_sheet.loc['Current Liabilities']
        quick_ratio = (balance_sheet.loc['Current Assets'] - balance_sheet.loc['Inventory']) / balance_sheet.loc['Current Liabilities']
        cash_ratio = balance_sheet.loc['Cash Cash Equivalents And Short Term Investments']/balance_sheet.loc['Current Liabilities']
        

        # Long-term Solvency Ratios
        debt_to_equity = balance_sheet.loc['Total Liabilities Net Minority Interest'] / balance_sheet.loc['Stockholders Equity']
        debt_ratio = balance_sheet.loc['Total Liabilities Net Minority Interest'] / balance_sheet.loc['Total Assets']
        equity_multiplier = balance_sheet.loc["Total Assets"] / balance_sheet.loc["Stockholders Equity"]

        # Asset Utilization Ratios (assuming COGS and Net Sales are in the income statement)
        avg_inventory = balance_sheet.loc['Inventory']  # A simplification. Ideally, you'd want the average of start & end period inventory.
        inventory_turnover = income_statement.loc['Cost Of Revenue'] / avg_inventory
        avg_total_assets = balance_sheet.loc['Total Assets']  # Similarly, this is a simplification.
        total_asset_turnover = income_statement.loc['Total Revenue'] / avg_total_assets
        receivables_turnover = income_statement.loc['Total Revenue']/balance_sheet.loc['Accounts Receivable']

        # Profitability Ratios
        net_profit_margin = income_statement.loc['Net Income'] / income_statement.loc['Total Revenue']
        roa = income_statement.loc['Net Income'] / avg_total_assets
        avg_equity = balance_sheet.loc['Stockholders Equity']  # Again, this is a simplification.
        roe = income_statement.loc['Net Income'] / avg_equity

     # Extracting years from date columns
        #years = [date.year for date in balance_sheet.columns]
        years = [pd.to_datetime(date).year for date in balance_sheet.columns]

        # Consolidating the ratios into a DataFrame
        df_ratios = pd.DataFrame({
                'Year': years,
                'Date': balance_sheet.columns,
                'Current Ratio': current_ratio.values,
                'Quick Ratio': quick_ratio.values,
                'Cash Ratio': cash_ratio.values,
                'Debt to Equity Ratio': debt_to_equity.values,
                'Debt Ratio': debt_ratio.values,
                'Equity Multiplier': equity_multiplier.values,
                'Inventory Turnover': inventory_turnover.values,
                'Total Asset Turnover': total_asset_turnover.values,
                'Receivables Turnover': receivables_turnover.values,
                'Net Profit Margin': net_profit_margin.values,
                'ROA': roa.values,
                'ROE': roe.values
        })

        # Set 'Year' as the index of the DataFrame for a clear structure
        df_ratios.set_index('Year', inplace=True)

        return df_ratios

def compare_tickers(tickers):
        global start_date, end_date, ticker_to_name, price_df
        ratio_dfs = []
        for ticker in tickers:
            historical_data, income_statement, balance_sheet, cashflow, info_data = fetch_or_load_data(ticker,start_date,end_date)
            price_df[ticker]=historical_data['Close']
            ratios = calculate_ratios(balance_sheet, income_statement,cashflow)
            ratios['PE']=info_data["trailingPE"]
            ratios['Price to Book']=info_data["priceToBook"]
            ratio_dfs.append(ratios)

        # Combine all DataFrames
        
        # Update tickers list using the mapping
        combined_df = pd.concat(ratio_dfs, axis=1, keys=tickers)
        combined_df = combined_df.sort_index()
        combined_df = combined_df.loc[pd.to_datetime(start_date).year:pd.to_datetime(end_date).year]
        return combined_df

def plot_comparison(comparison_df, tickers, columns):
        # Update tickers list using the mapping
        global ticker_to_name, end_date
        tickers_updated = [ticker_to_name[ticker] if ticker in ticker_to_name else ticker for ticker in tickers]
        sns.set(style="whitegrid")
        for column in columns:
            st.markdown(f"#### {column}")
            plt.figure(figsize=(12, 8))

            for ticker in tickers:
                ticker_label = ticker_to_name[ticker]
                plt.plot(comparison_df.index, comparison_df[(ticker, column)], label=ticker_label, marker='o')
            plt.title(column + ' over Time', fontsize=24)
            plt.xlabel('Year', fontsize=18)
            plt.ylabel(column, fontsize=18)
            plt.xticks(ticks=comparison_df.index, labels=comparison_df.index)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=16)
            plt.grid(True, which='both', linestyle='--', linewidth=0.3)
            plt.tight_layout()
            #plt.show()
            st.pyplot(plt)

            current_ratios = pd.DataFrame(columns=tickers)

            year = pd.to_datetime(end_date).year
            for ratio_name in short_ratios_col:

                    for ticker in tickers:
                            current_ratios.loc[ratio_name, ticker] = comparison_df[ticker,ratio_name].loc[year]
            
            current_ratios=current_ratios.rename(columns=ticker_to_name)
            
            st.table(comparison_df.xs(column, level=1, axis=1).rename(columns=ticker_to_name))

        # Visualization
        current_ratios.plot(kind="bar", figsize=(12, 8))
        plt.title(f"Ratios Comparison year {year}", fontsize=24)
        plt.ylabel("Ratio", fontsize=18)
        plt.xlabel(None)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
        plt.grid(True, which='both', linestyle='--', linewidth=0.3)
        plt.tight_layout()
        plt.legend(fontsize=16)
        #plt.show()
        st.pyplot(plt)
                

def fetch_or_load_data(ticker_symbol, start_date, end_date):
        
        # Filenames with path to yfinance subfolder
        historical_data_file = os.path.join("yfinance", f"{ticker_symbol}_historical_data.csv")
        income_statement_file = os.path.join("yfinance", f"{ticker_symbol}_income_statement.csv")
        balance_sheet_file = os.path.join("yfinance", f"{ticker_symbol}_balance_sheet.csv")
        cashflow_file = os.path.join("yfinance", f"{ticker_symbol}_cashflow.csv")
        info_file = os.path.join("yfinance", f"{ticker_symbol}_info.json")
        
        
        # Check if historical data file exists
        if os.path.exists(historical_data_file):
                historical_data = pd.read_csv(historical_data_file, index_col=0)
                historical_data.index = pd.to_datetime(historical_data.index)
                print(f"Loaded historical data for {ticker_symbol} from {historical_data_file}")
        else:
                historical_data = yf.download(ticker_symbol, start=start_date)
                #historical_data.to_csv(historical_data_file)
                print(f"Fetched and saved historical data for {ticker_symbol} to {historical_data_file}")
        
        # Create a Ticker object
        ticker = yf.Ticker(ticker_symbol)
        
        # Check if income statement file exists
        if os.path.exists(income_statement_file):
                income_statement = pd.read_csv(income_statement_file, index_col=0)
                print(f"Loaded income statement for {ticker_symbol} from {income_statement_file}")
        else:
                income_statement = ticker.financials
                #income_statement.to_csv(income_statement_file)
                print(f"Fetched and saved income statement for {ticker_symbol} to {income_statement_file}")
        
        # Check if balance sheet file exists
        if os.path.exists(balance_sheet_file):
                balance_sheet = pd.read_csv(balance_sheet_file, index_col=0)
                print(f"Loaded balance sheet for {ticker_symbol} from {balance_sheet_file}")
        else:
                balance_sheet = ticker.balance_sheet
                #balance_sheet.to_csv(balance_sheet_file)
                print(f"Fetched and saved balance sheet for {ticker_symbol} to {balance_sheet_file}")
        
        # Check if cashflow file exists
        if os.path.exists(cashflow_file):
                cashflow = pd.read_csv(cashflow_file, index_col=0)
                print(f"Loaded cash flow for {ticker_symbol} from {cashflow_file}")
        else:
                cashflow = ticker.cashflow
                #cashflow.to_csv(cashflow_file)
                print(f"Fetched and saved cash flow for {ticker_symbol} to {cashflow_file}")

        # Check if info file exists
        if os.path.exists(info_file):
                with open(info_file, 'r') as file:
                        info_data = json.load(file)
                print(f"Loaded info for {ticker_symbol} from {info_file}")
        else:
                info_data = ticker.info
                with open(info_file, 'w') as file:
                        json.dump(info_data, file)
                print(f"Fetched and saved info for {ticker_symbol} to {info_file}")

        ticker_to_name[ticker_symbol] = info_data['shortName']
                
        return historical_data, income_statement, balance_sheet, cashflow, info_data





comparison_df = compare_tickers(tickers)

historical_data, income_statement, balance_sheet, cashflow, info_data = fetch_or_load_data(index_benchmark,start_date,end_date)
price_df[index_benchmark]=historical_data['Close']

st.header(f"{ticker_to_name[tickers[0]]} vs {ticker_to_name[tickers[1]]}", divider='rainbow')

st.markdown(f"### :blue[Short term Solvency Ratio]")
plot_comparison(comparison_df, tickers, short_ratios_col)

st.markdown(f"### :blue[Long term Solvency Ratio]")
plot_comparison(comparison_df, tickers, long_ratios_col)

st.markdown(f"### :blue[Asset Ratio]")
plot_comparison(comparison_df, tickers, asset_ratios_col)

st.markdown(f"### :blue[Profitability Ratio]")
plot_comparison(comparison_df, tickers, profit_ratios_col)

# Index to Datetime, Sample data to week, drop na
price_df.index = pd.to_datetime(price_df.index)
price_df = price_df.resample('W').last()
price_df.dropna(inplace=True)

# Rename columns of price_df using the mapping
price_df = price_df.rename(columns=ticker_to_name)

#Market Value
st.markdown(f"### :blue[Market Value]")

pe = pd.DataFrame(comparison_df.xs("PE", level=1, axis=1).rename(columns=ticker_to_name).iloc[-1].round(2))
pe = pe.rename(columns={pe.columns[0]: 'PE'})
bv = pd.DataFrame(comparison_df.xs("Price to Book", level=1, axis=1).rename(columns=ticker_to_name).iloc[-1].round(2))
bv = bv.rename(columns={bv.columns[0]: 'Price to Book'})

mv = pd.concat([pe,bv], axis=1)

st.table(mv)

# Normalize prices
df_norm = (price_df/price_df.iloc[0]) * 100

st.markdown(f"### :blue[Normalized Return]")

#df_norm.plot()
plt.figure(figsize=(12, 8))
#sns.lineplot(data=df_norm,dashes=False, palette=palette)
linewidths = [2, 2, 1]
# Plot each line separately with its own linewidth
for col, color, lw in zip(df_norm.columns, palette, linewidths):
    sns.lineplot(data=df_norm, x=df_norm.index, y=col, color=color, lw=lw, label=col)


#plt.title(f"Normalized Return", fontsize=24)
plt.title(None)
plt.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.75)
#plt.xlabel("Date", fontsize=16)
plt.xlabel(None)
plt.ylabel("Return", fontsize=16)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend(fontsize=16)
#plt.show()
st.pyplot(plt)


df_pct = price_df.pct_change()
df_pct.dropna(inplace=True)
correlation_matrix = df_pct.corr()
correlation_coefficient = correlation_matrix.iloc[0, 1]

print(f"Correlation coefficient between {ticker_to_name[tickers[0]]} and {ticker_to_name[tickers[1]]}: {correlation_coefficient:.2f}")
sns.jointplot(x=price_df[ticker_to_name[tickers[0]]], y=price_df[ticker_to_name[tickers[1]]], kind='reg', height=10);
plt.grid(True, which='both', linestyle='--', linewidth=0.3, alpha=0.75)
#print(f'Correlation: {correlation_coefficient:.4f}')
st.markdown(f"### :blue[Correlation coefficient between {ticker_to_name[tickers[0]]} and {ticker_to_name[tickers[1]]} is {correlation_coefficient:.2f}]")
#plt.show()
st.pyplot(plt)
