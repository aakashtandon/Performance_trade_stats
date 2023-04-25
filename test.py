

import pandas as pd
import quantstats as qs
import os
qs.extend_pandas()


#######
file_path = r"C:\Users\aakas\Downloads\ret.csv"



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
    
    if df is not None and not df.empty:
        if df.shape[1] > 2:
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

qs.reports.html(df.iloc[:, 0], output=r"Xiao.html", title="quantstats-tearsheet" , compounded=False)

print("done")

# qs.stats.max_drawdown(df1)

# dd_info.columns = dd_info.columns.droplevel()

