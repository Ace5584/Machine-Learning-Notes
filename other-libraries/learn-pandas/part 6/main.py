#------------------------------#
# working with large amounts   #
# of data.                     #
#------------------------------#

import pandas as pd

df = pd.read_csv('C:/src/learn-pandas/pandas code/pokemon_data.csv')

new_df = pd.DataFrame(columns=df.columns)

for x in pd.read_csv('C:/src/learn-pandas/pandas code/modified.csv', chunksize=5):
    results = df.groupby(['Type 1']).count()
    new_df = pd.concat([new_df, results])

print(new_df)


