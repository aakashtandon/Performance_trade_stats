

import pandas as pd
import quantstats as qs

qs.extend_pandas()


#######
file_path = r"C:\Users\aggar\Downloads\2020.csv"

df = pd.read_csv(file_path, index_col=0)
df.index = pd.to_datetime(df.index, format="%d-%m-%Y %H:%M")

# df1 = df.pct_change().dropna()

df = df.groupby(df.index.date).last()
df.index = pd.to_datetime(df.index)

qs.reports.html(df.iloc[:, 0], output=r"2020.html", title="2020")

print("done")

# qs.stats.max_drawdown(df1)

# dd_info.columns = dd_info.columns.droplevel()

