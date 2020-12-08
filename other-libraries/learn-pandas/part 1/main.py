#-----------------------------#
# Loading data into pandas    #
# with files like csv, txt    #
# Excels, etc...              #
#-----------------------------#

import pandas as pd

df = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.csv') #import csv with read_csv
df_xlsx = pd.read_excel('C:/src/learn-pandas/pandas code/pokemon_data.xlsx') #improt xlsx format file 
df_txt = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.txt', delimiter='\t') 
#import txt file using delimiter which is what seperates the data 

# Limit amount of data you can see: 
print(df.head(3)) # Read Top 3 rows
print(df.tail(3)) # Read bottom 3 rows

print(df_xlsx) # Printing df_xlsx

print(df_txt) # Printing df_txt



