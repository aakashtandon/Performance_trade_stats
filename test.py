

import pandas as pd
import quantstats as qs
import os
import glob
qs.extend_pandas()
import matplotlib.pyplot as plt

#######
file_path =r"C:\Users\aakas\Desktop\Aargo_docs\Team_performance_docs_Format_for_evaluation\Aakash_new.xlsx"
out_folder = r"C:\Users\aakas\Desktop\Aargo_docs\Quanstats_tearsheets_all_traders"

#=== output path for excel 

fname = os.path.splitext(os.path.basename(file_path))[0]

print("\n \n Processing file: " , fname)

out_exc_path = os.path.join(out_folder, fname + '.xlsx')

plot_path = os.path.splitext(out_folder)[0]




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
    
    print("\n Dataframe is : \n" , df)
    
    if df is not None and not df.empty:
        if df.shape[1] > 3:
            print('\n The DataFrame has more than two columns \n ')

            return 

        if 'Date/ Time' in df.columns:
            df.rename(columns={'Date/ Time':'Date/Time'} , inplace=True)


        if 'Date/Time' in df.columns:

            convert_datetime(df , 'Date/Time' ,0)
            df.set_index('Date/Time' , inplace=True)

        if 'Date' in df.columns:

            convert_datetime(df , 'Date' ,0)
            df.set_index('Date' , inplace=True)

            #== after conversion to datetime set it as index and convert strategy returns to daily incase they arent


        return df


if os.path.exists(file_path):

    #--- get the file extension
    file_ext = os.path.splitext(file_path)[1]

    # Read the file if it's in CSV format
    if file_ext == '.csv':
        df = pd.read_csv(file_path)

    # Read the file if it's in Excel format
    elif file_ext in ['.xls', '.xlsx']:
        df = pd.read_excel(file_path)

    # Raise an error if the file format is not supported
    else:
        raise ValueError('File format not supported')
        
df = preprocessing(df)


print(df.index)
# df1 = df.pct_change().dropna()

df = df.groupby(df.index.date).last()
#df.set_index('timestamp1' , inplace=True)
df.index = pd.to_datetime(df.index)

print(df)

qs.reports.html(df.iloc[:, 0], output=r"Henry.html", title="quantstats-tearsheet" , compounded=True)

print("done")

# qs.stats.max_drawdown(df1)

# dd_info.columns = dd_info.columns.droplevel()



# Generate the report summary
stats = qs.reports.metrics(df)

import io
import re
from contextlib import redirect_stdout

#==== Put the metrics in a df
# Calculate the performance statistics for the strategy
buffer = io.StringIO()
with redirect_stdout(buffer):
    qs.reports.metrics(df)
metrics_output = buffer.getvalue()

# Use regular expressions to extract metrics and their values
pattern = re.compile(r'^(.+?)\s+([\d\.\-%]+)$', re.MULTILINE)
metrics_list = [match.groups() for match in pattern.finditer(metrics_output)]

# Convert the list of tuples to a DataFrame
metrics_df = pd.DataFrame(metrics_list, columns=['Metric', 'Value'])

print(metrics_df)



#--- edit the metrics
rows_to_remove = ['Risk-Free Rate', 'Common Sense Ratio', 'CPC Index', 'Prob. Sharpe Ratio', 'Tail Ratio', 'Outlier Win Ratio', 'Outlier Loss Ratio', 'Ulcer Index', 'Serenity Index','Gain/Pain (1M)']


# Find the indices of the rows to remove
indices_to_remove = metrics_df[metrics_df['Metric'].isin(rows_to_remove)].index

# Remove the rows
metrics_filtered = metrics_df.drop(indices_to_remove)


metrics_filtered.to_excel(out_exc_path)







# # Save the main plots as images
# plots = [
#      qs.plots.snapshot(returns, title='Performance Snapshot'),
#      qs.plots.monthly_heatmap(returns, title='Monthly Heatmap')]
# #     qs.plots.drawdown(returns, title='Drawdown'),
#     qs.plots.rolling_volatility(returns, title='Rolling Volatility'),
#     qs.plots.rolling_sharpe(returns, title='Rolling Sharpe Ratio'),
#     qs.plots.returns(returns, title='Returns'),
#     qs.plots.cumulative_returns(returns, title='Cumulative Returns'),
#     qs.plots.rolling_beta(returns, title='Rolling Beta'),
#     qs.plots.rolling_correlation(returns, title='Rolling Correlation')
# ]

# for i, plot in enumerate(plots, start=1):
#     fig = plot.get_figure()
#     fig.savefig(f'plot_{i}.png')
#     plt.close(fig)



def save_plots(file_path, returns):
    plots = [
        qs.plots.returns(returns=df, ylabel='Returns' ,show=False ),
        qs.plots.daily_returns(df , show=False),
        qs.plots.monthly_heatmap(returns=df , show=False),
        qs.plots.snapshot(returns=df.iloc[:, 0] , show=False)
    ]

    for i, plot in enumerate(plots, start=1):
        
        print("\n Processing plots")
        if plot is not None:
            fig = plot.get_figure()
            fig.savefig(f'{file_path}/plot_{i}.png')
            plt.close(fig)
        else:
            
            print("Plot is None")
                
                
       
        
#save_plots(file_path=plot_path , returns=df)        




#==== code to combine multiple EC's and show combined stats

# all_returns_path = r"C:\Users\aakas\Desktop\Aargo_docs\Final_EC_values_for_evaluation"

# com_df = pd.DataFrame()

# if(os.path.exists(all_returns_path)):
    
#     if os.path.isdir(all_returns_path):
#         print('The path is a directory')    
#         print("\n MTM Folder found")
#         for fl in glob.glob(all_returns_path+ "/*.xlsx"):

                    
#                     fname = os.path.basename(fl).split('.')[0]

#                     #sym_name = fname.split('.')[0].split('_')[1]
#                     print("\n === Processing === \n" , fname)

#                     df = pd.read_excel(fl)
#                     df = preprocessing(df)
                                       
                    
#                     df.index = df.index.date
#                     df = df.loc[~df.index.duplicated(keep='first')]
#                     df.columns = [fname]
                    
#                     com_df = pd.concat([com_df, df] , join='outer' , axis=1)
                    
                    
                    
# print("\n" , com_df)





# all_returns_path = r'C:\Users\aakas\Desktop\Aargo_docs\Final_EC_values_for_evaluation'
# dataframes = []

# if os.path.exists(all_returns_path):
    
#     if os.path.isdir(all_returns_path):
#         print('The path is a directory')    
#         print("\n MTM Folder found")
        
#         for fl in glob.glob(all_returns_path+ "/*.xlsx"):
#             fname = os.path.basename(fl).split('.')[0]
#             print("\n === Processing === \n" , fname)

#             df = pd.read_excel(fl)
#             df = preprocessing(df)
#             df.index = df.index.date
#             df = df.loc[~df.index.duplicated(keep='first')]#
#             df.columns = [fname]

#             dataframes.append(df)
#             print( "\n ", dataframes)
# # Find the DataFrame with the largest index
# largest_df = max(dataframes, key=lambda x: len(x.index))

# # Merge the other DataFrames with the largest DataFrame
# com_df = largest_df.copy()
# for df in dataframes:
#     if df is not largest_df:
#         com_df = com_df.merge(df, left_index=True, right_index=True, how='left')

# print(com_df)


# com_df = com_df.fillna(method='ffill')

# com_df.to_excel(r"C:\Users\aakas\Desktop\Aargo_docs\Final_EC_values_for_evaluation\combined.xlsx")