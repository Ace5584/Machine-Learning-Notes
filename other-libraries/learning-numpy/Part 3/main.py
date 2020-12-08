#------------------------------------------#
# Using numpy arrays, here are some ways   #
# to inizialize different types of array   #
# in different ways                        #
#------------------------------------------#

import numpy as np

# All 0 matrix
a = np.zeros((5, 5))
print(a)

# All 1 matrix
b = np.ones((2,4, 3), dtype="int16")
print(b)

# Any other number or symbol
c = np.full((2, 4), 9)
print(c)

# Create an array from the syntax of another array
d = np.full_like(c, 3)
print(d)

# Random decimal numbers 
e = np.random.rand(2, 4, 3)
print(e)
#With shape from another array
f = np.random.random_sample(d.shape)
print(f)

# Random Integer values 
g = np.random.randint(4, size=(3, 4))
print(g)

# Identity matrix (Square matrix)
h = np.identity(5)
print(h)

# Repeat an array
i = np.array([[1,2,3]])
i = np.repeat(i, 3, axis=0)
print(i)


