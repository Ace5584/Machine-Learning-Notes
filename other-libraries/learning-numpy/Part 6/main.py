#------------------------------------------#
# Reorganizing Numpy arrays                #
# Miscellaneous, load data from file       #
# Indexing and boolean masking             #
#------------------------------------------#

import numpy as np

before = np.array([[1,2,3],[4,5,6]])
print(before.shape)

after = before.reshape((3,2))
print(after)

# Vertically stacking vectors
a = np.array([1,2,3,4])
b = np.array([5,6,7,8])
print(np.stack([a,b,a,b]))

# Horizontal stack
c = np.ones((2,4))
d = np.zeros((2,2))

print(np.hstack([c,d]))

data = np.genfromtxt('C:/Users/Alex Lai.DESKTOP-AJOHRHM/Desktop/numpy code/Part 6/data.txt', delimiter=",")
data = data.astype('int32')
print(data)

# Advanced indexing and boolean masking

# Check weather the value in the array is > 50
# Or any combinations
print(data > 50)
print(data[data > 50])
print(np.any(data > 50, axis=0))
print((~(data > 50) & (data < 100)))

# Indexing with numpy
x = np.array([1, 2, 3, 4, 5, 6, 7, 8])
print(x[[1, 3, 4]])


