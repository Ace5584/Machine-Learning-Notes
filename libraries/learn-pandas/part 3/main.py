#-----------------------------#
# Making Changes to data      #
# Adding rows, removing rows, #
# And changing row position.  #
# Also saving new csv file    #
#-----------------------------#
import pandas as pd

df = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.csv')

# Creating new row that adds up all the total values
print(df.head(5))
df['Total'] = df['HP'] + df['Attack'] + df['Defense'] + df['Sp. Atk'] + df['Sp. Def'] + df['Speed'] #Craeting total row
print(df)
df = df.drop(columns=['Total']) # Removing total row
df['Total'] = df.iloc[:, 4:10].sum(axis=1) # Creating total row with iloc
print(df)

# Rearranging the data set
cols = list(df.columns.values)
df = df[cols[0:4] + [cols[-1]] + cols[4:12]]
print(df)

# Exoporting data as csv file
df.to_csv('modified.csv', index=False) # Exported csv
#index = false means it doesn't save the index

# Exporting data as excel file
df.to_excel('modified.xlsx', index=False) # Export xlsx
# index = false means it doesn't save the index


