#------------------------------------------#
# Maths with numpy:                        #
# Simple functions like +, -, /, *         #
# Linear algebra                           #
# Statistics                               #
#------------------------------------------#

import numpy as np

a = np.array([1,2,3,4])

# +,-,*,/ from each element
a += 2
print(a)

b = np.array([1, 0, 1, 2]) 

c = a + b
print(c)

a = a**2
print(a)

#sin, cos, tan
d = np.cos(a)
print(d)

# Linear Algebra

x_1 = np.ones((2,3))
x_2 = np.zeros((3,2))
print(x_1)
print(x_2)
print(np.matmul(x_1, x_2))
# Find the determinate
y = np.identity(3)
y = np.linalg.det(y)
print(y)
# And there is alot more you could do with np.linalg.

# Statistics

stats = np.array([[1,2,3],[4,5,6]])
print(np.min(stats)) # Minimum Value
print(np.max(stats)) # Maximum Value
print(np.sum(stats)) # Sum of all values


