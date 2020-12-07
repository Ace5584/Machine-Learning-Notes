#-----------------------------#
# Filtering data based on     #
# conditions                  #
#-----------------------------#

import pandas as pd
import re

df = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.csv')

# Filtering data
print(df.loc[df['Type 1'] == 'Grass'])
print(df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison')])
print(df.loc[(df['Type 1'] == 'Grass') | (df['Type 2'] == 'Poison')])
# Removing Mega pokemon from the lists
print(df.loc[~df['Name'].str.contains('Mega')])
# Creating new_df after the with the filter
new_df = df.loc[(df['Type 1'] == 'Grass') & (df['Type 2'] == 'Poison') & (df['HP'] > 60)]

new_df.reset_index(drop=True, inplace=True) # Resetting the index of the filtered data 
# drop=True means dropping the old index and inplace meaning replaced the current list
print(new_df)

print(df.loc[df['Type 1'].str.contains('Fire|Grass', regex=True)])
print(df.loc[df['Type 1'].str.contains('fire|grass', flags=re.I, regex=True)]) 
# flags=re.I make it ignore capital letters

print(df.loc[df['Name'].str.contains('^ch[a-z]*', flags=re.I, regex=True)]) 
# * means one or more

