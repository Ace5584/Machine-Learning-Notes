import pandas as pd
from matplotlib import pyplot as plt

# Plotting a 2D graph
x = [1, 2, 3]
y = [2, 4, 6]
z = [4, 8, 12]

plt.plot(x, y) # Plot the data
plt.plot(x, z) # Plot the second line in the graph
plt.title("Test Plot") # Giving the linear graph a title
plt.xlabel("x") # Labels at x
plt.ylabel("y and z") # Labels at y
plt.legend(['line y', 'line z'])

plt.show() # Show the plotted data

# Plotting Graph with sample data
plt.clf()
sample_data = pd.read_csv('C:/src/learn-matplotlib/code/sample_data.csv')
print(sample_data)
plt.plot(sample_data.column_a, sample_data.column_b, 'o')
plt.plot(sample_data.column_a, sample_data.column_c)
plt.show()

# Plotting Graph with real data
plt.clf()
countries = pd.read_csv('C:/src/learn-matplotlib/code/countries.csv')
print(countries)
# Compare the population growth between UK and US
UK = countries.loc[countries['country'] == 'United Kingdom']
US = countries.loc[countries['country'] == 'United States']
plt.plot(UK.year, UK.population/UK.population.iloc[0] * 100)
plt.plot(US.year, US.population/US.population.iloc[0] * 100)
plt.xlabel("Year")
plt.ylabel("Population in Millions")
plt.title("Population Grown of United States and United Kingdom")
plt.legend(['United Kingdom', 'United States'])
plt.show()





