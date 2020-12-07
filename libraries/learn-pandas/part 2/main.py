#-----------------------------#
# Reading data, getting rows, #
# columns, cells, headers,    #
# etc... And sorting          #
# /discribing data. And High  #
# Level description on data   #
#-----------------------------#
import pandas as pd

df = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.csv')

# Reading headers 
print(df.columns)

# Reading each column
print(df['Name'][0:5]) 
# [0:5] stands for how many data you want to print with index form
print(df.Name[0:5]) # Only words for one word

# Reading each row
print(df.iloc[1]) # iloc -> integer location
print(df.iloc[1:4]) # Getting multiple rows with index numbers

# Reading a specific location
print(df.iloc[2, 1])

# Looping through rows and columns

for index, row in df.iterrows(): # Iterate through each row
    #print(index, row)
    print(index, row['Name']) # With only names 

# Locate specific rows with specific data
print(df.loc[df['Type 1'] == "Grass"])

# High level disription on data
print(df.describe())
# Produces count, mean, std, min, 25%, 50%, 75%, and max for each row of data

# Sorting data
print(df.sort_values('Name', ascending=False))
print(df.sort_values(['Type 1', 'HP'], ascending=[0,1])) # Numbers represends true/false


