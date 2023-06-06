

from tabulate import tabulate
import pandas as pd
import numpy as np
import datetime , pytz
import matplotlib.pyplot as plt
import os
import math 
import glob

import re

import quantstats as qs

import sys


from PIL import Image, ImageDraw, ImageFilter


qs.extend_pandas()


starting_capital = 30000000
pd.options.display.float_format = '{:.2f}'.format

#=== Code to get all performance stats from trades file
#===== Currently thinking about
#--- avg profit/loss(%) , Win/Loss ration

#--- avg winner(%) , avg loser(%)
#-- The formula for calculating the Expectancy Ratio is:

#------------Expectancy Ratio = (Total Profit / Number of Winning Trades) / (Total Loss / Number of Losing Trades)
#---Profit factor
#--------Average Holding Period
#--- Maximum Adverse Excursion , MFE

#--- trade frequency

#-- max consecutive wins/losses



#----xlxs or csv file  file format for input


#---File format ----

#---'Symbol' optional 

#--- trade is direction(long/short ,  Buy/Sell)

#--- entry date , entryprice and  exitdate and exit price and qty required


    # 	Symbol	Trade	Entry_Date	Entry_Price	Exit_ date	Exit_ Price	Qty
    # 0	NIFTY14JAN2114250PE.NFO	Short	2021-01-08 14:55:00	83.74200	2021-01-08 15:25:00	74.10	1234
    # 1	NIFTY14JAN2114300PE.NFO	Short	2021-01-08 14:55:00	101.97300	2021-01-08 15:25:00	90.15	1012

loc = r"C:\Users\aakas\Desktop\Data\Strategy_EC's_and_timeseries\Less_opt_Call_sell_nifty\Live\March_2023.xlsx"


def read_trades(loc):
    
    
    
    """
    read the trade log and clear all column names etc
    
    
    Parameters:
    df (pandas.Series): A pandas dataframe of trades
    
    
    Returns:
    plotly image of the EC
    
    """
    
    import os
    
    if os.path.exists(loc):
    
        #--- get the file extension
        file_ext = os.path.splitext(loc)[1]
        
        # Read the file if it's in CSV format
        if file_ext == '.csv':
            df = pd.read_csv(loc)

        # Read the file if it's in Excel format
        elif file_ext in ['.xls', '.xlsx']:
            df = pd.read_excel(loc)

        # Raise an error if the file format is not supported
        else:
            raise ValueError('File format not supported')
            return None
        #-- if file read is not empty or None
        
        if len(df.columns)<=5:
            
            print("\n \n Incorrect file please check")
            print("\n \n File should have only 7 columns: ----- Symbol , Trade , entry price , entry date , exit price , exit date\n \n ")
            raise ValueError('\n File format not correct. Check number of columns')
            return None  
        
        
        if df is not None and not df.empty :
            df.columns = df.columns.str.lower()
            return df
          
        


def convert_datetime(df, column_name , is_index=0):
    
    # convert index to datetime
    formats = [ '%Y%m%d %H%M%S',"%m/%d/%Y %I:%M:%S %p" , "%m/%d/%Y %H:%M:%S %p"  ,  '%d-%m-%Y', '%Y-%d-%m %H:%M:%S', '%Y-%m-%d %H:%M:%S', '%Y/%m/%d %H:%M:%S' , '%m-%d-%Y %H:%M:%S' , '%Y-%m-%d', '%m%d%Y %H:%M:%S', '%Y/%m/%d' , "%Y-%d-%m" , '%Y-%m-%dT%H:%M:%S.%f']
    for fmt in formats:
        try:
            if is_index ==0:
                df[column_name] = pd.to_datetime(df[column_name], format=fmt, errors='raise')
                return df
            if is_index==1:
                
                df.index = pd.to_datetime(df.index ,format=fmt, errors='raise' )
                return df
                       
             
            break
                    
        except ValueError:
            pass


        
def preprocessing(df):

    #-- mappings for different kinds of name found in the trade files
    entry_date_map = {'e.date': 'entry_date', 'entrydate': 'entry_date' , 'date':'entry_date' , 'Entry_date':'entry_date' }
    exit_date_map = {'ex. date': 'exit_date', 'exit_ date': 'exit_date' , 'exitdate': 'exit_date' , 'Exit_date':'exit_date'}
    exit_price_map = {'exit_ price': 'exit_price', 'ex.price': 'exit_price' ,'exit':'exit_price' , 'exitprice':'exit_price' ,'ex. price': 'exit_price' , 'Ex. Price':'exit_price' }
    entry_price_map = {'price': 'entry_price', 'entry': 'entry_price' ,'entryprice':'entry_price', 'entry price': 'entry_price' }
    qty_map = {'contract':'qty' , 'contracts':'qty' , 'shares':'qty' , 'lots':'qty' , 'quantity':'qty' , 'Shares':'qty' }
    
    trade_map = {'side':'trade' , 'position':'trade' , 'Trade':'trade' }
    

    # Rename the column names using the dictionaries
    
    df.rename(columns=entry_date_map, inplace=True)
    df.rename(columns=exit_date_map, inplace=True)
    df.rename(columns=exit_price_map, inplace=True)
    df.rename(columns=entry_price_map, inplace=True)
    df.rename(columns=qty_map, inplace=True)
    df.rename(columns=trade_map, inplace=True)
    
    
    if 'entry_date' in df.columns:
        convert_datetime(df ,'entry_date'  , 0 )      

    if 'exit_date' in df.columns:

        convert_datetime(df ,'exit_date'  , 0 )


    if 'entry_price' in df.columns:

        df['entry_price'] = df['entry_price'].astype('float')

    if 'exit_price' in df.columns:

        df['exit_price'] = df['exit_price'].astype('float') 


    if 'qty' in df.columns:

        df['qty'] = df['qty'].astype('int')
    
    return df
            
#====================================================================


def trade_count(df):
    
    return len(df)
    
    
    


def entry_exposure(df):
    
    if 'qty' in df.columns:
        
        df['qty'] = df['qty'].astype('int')
        
        return df['qty']*df['entry_price'] 
    
        
def exit_exposure(df):
    
    if 'qty' in df.columns:
        
        df['qty'] = df['qty'].astype('int')
        
        return df['qty']*df['exit_price'] 

#================================================================

def abs_trade_pnl(df):
    
    """
    Finds the absolute of each trade from the dataframe of trades 
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: list of pnl of each trade
    
    """
    
    long_trade = np.array(['BUY', 'buy', 'Long', '1', 'LONG' , 'Long' , 'Buy'])
    short_trade = np.array(['SELL', 'sell', 'SHORT', '-1', 'Short' , 'short' , 'Sell'])
    allowed_values = np.concatenate([long_trade, short_trade])  # Combine allowed values into one array
    
    trade_values = df['trade'].unique()  # Get unique values of 'trade' column
    
    if not np.in1d(trade_values, allowed_values).all():
        raise ValueError('\n \n Invalid trade(position) value found in Trade File. All signals should be long/short or Buy/Sell \n ')
    long_mask = np.isin(df['trade'], long_trade)
    short_mask = np.isin(df['trade'], short_trade)

    #print("LM is : \n" , long_mask , "\n--------------")
    
    #print("SM is : \n", short_mask , "\n --------------------")
    
    
    if long_mask.any():
        long_val = (exit_exposure(df)-entry_exposure(df)) 
    else:
        long_val = 0  # or None, or any default value

# If short_mask is not empty, calculate the value, else set a default value (like 0 or None)
    if short_mask.any():
        short_val = (entry_exposure(df)-exit_exposure(df))
    else:
        short_val = 0  # or None, or any default value

    return np.where(long_mask, long_val, np.where(short_mask, short_val, None))
        
        
    






     
def trade_pnl(df):
    
    """
    Finds the %pnl of each trade from the dataframe of trades 
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: list of pnl of each trade
    
    """
    long_trade = np.array(['BUY', 'buy', 'Long', '1', 'LONG' , 'Long' , 'Buy'])
    short_trade = np.array(['SELL', 'sell', 'SHORT', '-1', 'Short' , 'short' , 'Sell'])
    allowed_values = np.concatenate([long_trade, short_trade])  # Combine allowed values into one array
    
    trade_values = df['trade'].unique()  # Get unique values of 'trade' column
    
    if not np.in1d(trade_values, allowed_values).all():
        raise ValueError('\n \n Invalid trade(position) value found in Trade File. All signals should be long/short or Buy/Sell \n ')
    long_mask = np.isin(df['trade'], long_trade)
    short_mask = np.isin(df['trade'], short_trade)

    #print("LM is : \n" , long_mask , "\n--------------")
    
    #print("SM is : \n", short_mask , "\n --------------------")
    
    
    if long_mask.any():
        long_val = (exit_exposure(df)/entry_exposure(df)) - 1
    else:
        long_val = 0  # or None, or any default value

# If short_mask is not empty, calculate the value, else set a default value (like 0 or None)
    if short_mask.any():
        short_val = (entry_exposure(df)-exit_exposure(df))/entry_exposure(df)
    else:
        short_val = 0  # or None, or any default value

    return np.where(long_mask, long_val, np.where(short_mask, short_val, None))
        
        
    


def max_consecutive_win_loss(df):

    pnl = trade_pnl(df)

    win_streak = 0
    loss_streak = 0
    max_win_streak = 0
    max_loss_streak = 0

    for i in range(len(pnl)):
        if pnl[i] is not None:
            if pnl[i] > 0:
                win_streak += 1
                loss_streak = 0
                if win_streak > max_win_streak:
                    max_win_streak = win_streak
            else:
                loss_streak += 1
                win_streak = 0
                if loss_streak > max_loss_streak:
                    max_loss_streak = loss_streak

    return max_win_streak , max_loss_streak

#===================================================

def win_percent(df):
        
    pnl = trade_pnl(df)
    return np.round(np.count_nonzero(pnl > 0)/len(pnl) , 2)

def loss_percent(df):
    
    pnl = trade_pnl(df)
    return np.round(np.count_nonzero(pnl<=0)/len(pnl) , 2)    

def avg_profit_perc(df):
    
    return np.round(np.mean(trade_pnl(df)) , 3) 


def avg_profit_abs(df):
    
    return np.round(np.mean(abs_trade_pnl(df)) , 2)
    


def rolling_avg_profit_perc(df , window=20):
    
    pnl = trade_pnl(df)
    
    rolling_mean = pd.Series(pnl).rolling(window=window).mean()
    
    ema_pnl = pd.Series(pnl).ewm(span=window, adjust=False).mean()
                                                                                                                
    return ema_pnl


def monthly_pnl(df):
    """
    Aggregates all the trades for the month and finds the perc PnL  
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: of monthly cumulative percentage pnl for each month 
    
    """
    
    import copy
    df_copy = copy.deepcopy(df)
    pnl = trade_pnl(df_copy)
    df_copy['daily_pnl'] = pnl
    
    monthly_pnl = df_copy.resample('M', on='entry_date')['daily_pnl'].sum()
    
    return monthly_pnl


def monthly_pnl_abs(df):
    
    """
    Aggregates all the trades for the month and finds the absolute PnL  
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: of monthly cumulative absolute pnl for each month 
    
    """
    
    
    import copy
    df_copy = copy.deepcopy(df)
    pnl = abs_trade_pnl(df_copy)
    df_copy['daily_pnl'] = pnl
    
    monthly_pnl_abs = df_copy.resample('M', on='entry_date')['daily_pnl'].sum()
    
    return monthly_pnl_abs.astype('int')
    
    
def monthly_return_on_exposure(df , starting_cap = 10000000):
    
    
    """
    Aggregates absolute pnl of each month as percentage of capital  
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: of monthly cumulative absolute pnl for each month 
    
    """
    
    
    
    month_abs_ret = monthly_pnl_abs(df)
    
    return month_abs_ret/starting_cap*100
    
    
def avg_month_ret_on_exposure(df , starting_capital):
    
    
    month_ret = monthly_return_on_exposure(df , starting_cap=starting_capital)
    
    return month_ret.mean()
    
    
        
     
 
    
    
    



def plot_monthly_pnl(df):
    
    
    mon_pnl = monthly_pnl(df)
    
    mon_pnl = mon_pnl.to_frame()     
    
    
      # Print DataFrame index
    print(mon_pnl.index)
    
    mon_pnl['Year'] = mon_pnl.index.year
    mon_pnl['Month'] = mon_pnl.index.month_name()
    
    #mon_pnl.set_index(['Year', 'Month'], inplace=True)

    
    return qs.stats.monthly_returns(mon_pnl)

    
    


def average_monthy_return(df):
    
    monthly_pnlser = monthly_pnl(df)
    
    return np.round(monthly_pnlser.mean() , 2)

def average_monthly_drawdown(df):
    
    """Calculate the average monthly drawdown percentage for the given DataFrame."""
    # Calculate the monthly returns series
    
    monthly_pnlser = monthly_pnl(df)
    monthly_cumulative_pnl = monthly_pnlser.cumsum()
    
    # Calculate the monthly drawdown for the portfolio
    monthly_drawdown = monthly_cumulative_pnl - monthly_cumulative_pnl.cummax()
    
    # Calculate the percentage drawdown for the portfolio
    monthly_percentage_drawdown = monthly_drawdown / monthly_cumulative_pnl.cummax()
    #print(monthly_percentage_drawdown)
    # Calculate the average monthly drawdown percentage
    return monthly_percentage_drawdown.mean()

    

#  Expectancy Ratio = (Win Rate x Average Win) / (Loss Rate x Average Loss)
#  >1.25 is a good sign
def expectancy_ration(df):
    
    pnl = trade_pnl(df)
    positive_pnl = pnl[pnl > 0]
   
    mean_positive_pnl = np.mean(positive_pnl)   # calculate the mean of positive values
    
    pos_exp = mean_positive_pnl*win_percent(df)
   
    #=== for losers now
    
    negative_pnl = pnl[pnl<=0]
    mean_negative_pnl = np.mean(negative_pnl)
    
    neg_exp = (abs(mean_negative_pnl))*loss_percent(df)
    
    
    return pos_exp/neg_exp



#--- worst trades based on a quantile
def worst_trades(df , quan=0.05):
    
    """
    Finds the worst(losers) of all the trades for the month based on a quantile.
    eg. if you want top 10% of losers set quan=0.1
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: of monthly cumulative pnl for each month 
    
    """
    import copy
    df_copy = copy.deepcopy(df)
    
    df_copy['pnl'] = trade_pnl(df_copy)
    
    sorted_trades = df_copy.sort_values(by=['pnl'] , ascending=True)
    
    #--- find the quantile threshold required
    
    threshold  = sorted_trades['pnl'].quantile(quan)
    worst_trades = sorted_trades[sorted_trades['pnl'] < threshold]
    
    return worst_trades
    

def avg_holding_period(df ,trading_hours_per_day=6.5):
    
    """
    
    Finds the average holding period of all trades(in minutes) from entry and exit dates
    
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    
    
    Returns:
    dataframe: of monthly cumulative pnl for each month 
    
    """
    
    import copy
    df_copy = copy.deepcopy(df)
    
    
    df_copy['business_days'] = df_copy.apply(lambda row: pd.bdate_range(start=row['entry_date'], end=row['exit_date'], freq='B').size-1 if row['entry_date'].normalize() != row['exit_date'].normalize() else 0, axis=1)
    
    hold_min = (trading_hours_per_day * df_copy['business_days']*60)  + (df_copy['exit_date'].dt.hour*60 + df_copy['exit_date'].dt.minute) - (df_copy['entry_date'].dt.hour*60 + df_copy['entry_date'].dt.minute)     
    
    
    return np.mean(hold_min)    


#--- average number of trades per day

def trade_freq(df):
    
    return df.groupby(by =df['entry_date'].dt.date)['trade'].count().mean()
    
    
def exposure(df):
    
    
    """
    Finds the exposure as in numbers of dates traded in a year as a percentage
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
        
    Returns:
    percentage of days traded 
    
    """
    
    
    trading_days = df['entry_date'].dt.date.nunique()
    
    return np.round(trading_days/252 , 2) 


def cvar(df , alpha=0.95):
    
    """
    Finds the Cvar of all the trades based on the confidence interval as input
    
    eg. if alpha = 0.95 than Cvar tells that worst 5% of cases we have an average loss = Cvar
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    alpha = 0.95 if you want 5% worst cases
    
    Returns:
    dataframe: of monthly cumulative pnl for each month 
    
    """
    import copy
    df_copy = copy.deepcopy(df)
    
    
    # Calculate the returns for each trade
    df_copy['Returns'] = trade_pnl(df)

    # Calculate the portfolio returns
    portfolio_returns = df_copy['Returns'].sum()

    # Calculate the portfolio VaR at the desired confidence level
    portfolio_var = np.percentile(df_copy['Returns'], 100*(1-alpha))

    # Calculate the portfolio returns that fall below the VaR
    portfolio_losses = df_copy['Returns'][df_copy['Returns'] < portfolio_var]

    # Calculate the CVaR as the average of the losses that fall below the VaR
    portfolio_cvar = np.mean(portfolio_losses)

    #print(f"The CVaR of the trades at {alpha*100}% confidence is {portfolio_cvar:.4f}.")
    
    return portfolio_cvar*100


def monthly_drawdowns(df , starting_cap=1000000):

    """
    Finds the monthly dradwons of the strategy for based on a starting capital
    
    Parameters:
    df (pandas.Series): A pandas df of trades 
    starting_cap = initial(required) starting capital for each strategy
    
    Returns:
    dataframe: of monthly drawdowns(percent)  for each month 
    
    """
    
    
    import pandas as pd
    import copy


    # Create a copy of the DataFrame
    df_copy = copy.deepcopy(df)

    # Calculate absolute trade pnl
    pnl = abs_trade_pnl(df_copy)
    df_copy['daily_pnl'] = pnl

    # Convert 'entry_date' column to datetime format
    df_copy = convert_datetime(df_copy, 'entry_date', is_index=False)

    # Set 'entry_date' as the DataFrame index
    df_copy.set_index('entry_date', inplace=True)

    # Convert 'daily_pnl' column to numeric dtype
    df_copy['daily_pnl'] = pd.to_numeric(df_copy['daily_pnl'], errors='coerce')

    # Add starting capital of 100,000 to the beginning of each group
    
    #df_copy['cumulative_pnl'] = df_copy['daily_pnl'].cumsum() + starting_capital

    # Define function to calculate maximum drawdown
    def calculate_max_drawdown(series):
        cum_max = series.cummax()
        drawdown = (series - cum_max) / cum_max*100
        return drawdown.min()


    #-- we add the strategy capital to each month to calculate each month drawdown correct
    
    def add_value(group , value_to_add = starting_cap):
        
        group['cum_pnl'] = group['daily_pnl'].cumsum() + value_to_add
        return group



    modified_groups = df_copy.resample('M').apply(add_value)


    # Calculate maximum drawdown for each month
    monthly_drawdown = modified_groups['cum_pnl'].resample('M').apply(calculate_max_drawdown)

    # Display the monthly drawdown
    return monthly_drawdown


def average_month_dd_on_exp(df , starting_cap = 30000000):
    
    mdd = monthly_drawdowns(df , starting_cap=starting_cap)
    

    return mdd.mean()




dd_path = r"C:\Users\aakas\Desktop\Aargo\test\month_dd.png"


def plot_monthly_drawdowns(df , starting_cap = 10000000 , save_path=None):
    import io
    import base64
    
    month_dd =  qs.stats.monthly_returns(monthly_drawdowns(df , starting_cap=starting_cap))
        
    if 'EOY' in month_dd.columns:
        month_dd = month_dd.drop(columns=['EOY'])
    plt.figure(figsize=(12, 6))
    sns.heatmap(month_dd, annot=True, cmap='RdBu', fmt=".2f" , cbar=False , annot_kws={"size": 16})
    plt.title('Monthly Drawdowns'  , size=20)
    #plt.xlabel('Month', size=15)
    #plt.ylabel('Year', size=15)
    #plt.show()
    
    if save_path is not None:
        plt.savefig(save_path, format='png')
        plt.close()
        return save_path
    
    print("Monthly_dd file saved")
    
    
    
    
    
    
    
        
    


loc1 = r"C:\Users\aakas\Downloads\Test1.xlsx"

df = pd.read_excel(loc1)

df = preprocessing(df)

print("\n " , df)



month_dd = monthly_drawdowns(df , starting_cap=10000000)

import seaborn as sns
# create a DataFrame
df1 = pd.DataFrame(list(month_dd.items()), columns=['Date', 'Value'])

# extract year and month
df1['Year'] = month_dd.index.year
df1['Month'] = month_dd.index.month_name()


# Get total number of bars to plot
num_bars = df1['Year'].nunique() * df1['Month'].nunique()

# Estimate figure size based on number of bars
fig_width = num_bars * 0.6
fig_height = 10  # set a constant figure height

# Create the figure and the axes
fig, ax = plt.subplots(figsize=(fig_width, fig_height))

# Plot the data
sns.barplot(x='Month', y='Value', hue='Year', data=df1, palette='viridis', ax=ax)

# Set x, y labels and title
ax.set_xlabel('Month', fontsize=20)
ax.set_ylabel('Drawdown', fontsize=20)
ax.set_title('Monthly Drawdowns(%) ', fontsize=16)

# Increase size of y-axis labels
ax.tick_params(axis='y', labelsize=14)

# Adjust the location and orientation of the legend
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., title="Year")

plt.show()





import io
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def create_trade_report(loc):
    df = read_trades(loc)
    df = preprocessing(df)
   
    
    trades = trade_count(df)
    avgpnl = avg_profit_perc(df)
    avg_abs = avg_profit_abs(df)
    winp = win_percent(df)
    losp = loss_percent(df)
    monret = average_monthy_return(df)
    mon_dd = average_month_dd_on_exp(df , starting_cap = starting_exposure)
    mon_ret_exp = avg_month_ret_on_exposure(df , starting_capital=starting_exposure)
    tradef = trade_freq(df)
    holp = avg_holding_period(df , trading_hours_per_day=6.5)
    expec = expectancy_ration(df)
    cvarv = cvar(df , alpha=0.95)
    #month_dd = monthly_drawdowns(df , starting_cap=starting_exposure)
        #==== Metrics to display... 

    vars_dict = {'Total Trades': trades, 
                 'Average Profit Perc per trade': format(avgpnl*100, '.2f') + '%',
                 'Average Profit(Absolute) per trade': avg_abs,
                 'Win Percentage' : format(winp*100, '.2f') + '%',
                 'Loss Percentage': format(losp*100, '.2f') + '%',
                 'Avg Holding period(min)' : format(holp, '.2f'), 
                 'Avg Monthly Return': format(monret, '.2f') ,
                 'Avg Monthly Return on Exposure': format(mon_ret_exp , '.2f') , 
                 'Avg Monthly Drawdown on Exposure': format(mon_dd, '.2f') + '%',
                 'Avg Trades per Day': format(trade_freq(df), '.2f'), 
                 '% days traded in a year(days)': format(exposure(df)*100, '.2f') + '%' 
                 }          

    print(vars_dict)
    
    
    

    vars_df = pd.DataFrame(vars_dict.items(), columns=['Variable', 'Value'])
    vars_df['Value'] = vars_df['Value'].apply(lambda x: f'<b>{x}</b>')
    vars_html = vars_df.to_html(index=False, header=None, border=2, justify='center', escape=False)

    # Generate the plot and convert it to a base64 string
    plot_monthly_drawdowns(df , starting_cap=starting_exposure , save_path=dd_path)
    
    
 # Add title and plot to the HTML
    html = f'''
    <html>
        <head>
            <title style="text-align:center;">Strategy Trade Metrics</title>
            <style>
                .container {{
                    width: 80%;
                    margin: 0 auto;
                    padding: 20px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    padding: 10px;
                    text-align: left;
                    border: 1px solid #ddd;
                    font-size: 16px;
                }}
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                .table-striped tbody tr:nth-of-type(odd) {{
                    background-color: #f1f1f1;
                }}
                .table-hover tbody tr:hover {{
                    background-color: #f5f5f5;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1 style="text-align: center;">Strategy Trade Metrics</h1>
                {vars_html}
                
                <img src="{dd_path}" alt="Monthly Drawdowns" style="display: block; margin-left: auto; margin-right: auto; width: auto;">
            </div>
        </body>
    </html>
    '''

    # Save the HTML report to a file
    with open(r"C:\Users\aakas\Desktop\Aargo\test\tpplots.html", 'w', encoding='utf-8') as f:
        f.write(html)


starting_exposure = 30000000

create_trade_report(loc=loc1)













