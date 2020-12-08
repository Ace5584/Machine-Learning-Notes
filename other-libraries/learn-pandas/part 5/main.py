#------------------------------#
# Conditional changes and      #
# Aggregate Statistics(groupby)#
# And working with large       #
# amounts of data.             #
#------------------------------#

import pandas as pd

df = pd.read_csv('C:/src/learn-pandas/pandas code/modified.csv')

#df.loc[df['Type 1'] == 'Fire', 'Type 1'] = 'Flamer'
#df.loc[df['Type 1'] == 'Fire', 'Legendary'] = True
#df.loc[df['Total'] > 500, ['Generation', 'Legendary']] = ['Test 1', 'Test 2']
# Changing Specific word in the csv

df['count'] = 1

print(df.groupby(['Type 1']).sum())
print(df.groupby(['Type 1']).mean().sort_values('Defense', ascending=False))
print(df.groupby(['Type 1']).count()['count'])
print(df.groupby(['Type 1', 'Type 2']).count()['count'])


