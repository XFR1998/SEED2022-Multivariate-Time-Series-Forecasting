#%%

import pandas as pd

#%%

res_list = ['./lgb+xgb+cab.csv', 'lstm.csv','informer.csv']

df0 = pd.read_csv(res_list[0])
df1 = pd.read_csv(res_list[1])
df2 = pd.read_csv(res_list[2])


#%%

sub = pd.DataFrame()
sub['ID'] = df0['ID']
sub['Time'] = df0['Time']
sub['Aps'] = df0['Aps']
# sub['Ai'] = (df0['Ai']+df1['Ai']) / 2
sub['Ai'] = (df0['Ai']+df1['Ai']+df2['Ai']) / 3

sub.to_csv('./all_results.csv', index=False)