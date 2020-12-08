#------------------------------------------#
# This part of np is about inizilizing and #
# understanding and seeing types of data   #
# sets and sizes                           #
#------------------------------------------#

import numpy as np

#init with dtype specifies the data type
# dtype='int16'
# dtype='int32' 
# etc...

a = np.array([1, 2, 3]) 
print(a)

b = np.array([[10.2, 3, 34, 2], [2, 38, 20.0, 3]])
print(b)

#get dimentions
print(a.ndim)
print(b.ndim)

#Get shape
print(a.shape)
print(b.shape)

#Get type
print(a.dtype)
print(b.dtype)

#Get size
print(a.itemsize) #itemsize is the size one item in the array
print(b.itemsize)
print(a.size) #size counts how many items are in the list
print(b.size)
#The total size in memory would be:
print(a.size * a.itemsize) #This is the hard way by calculating it 
print(b.size * b.itemsize)
#Easier way:
print(a.nbytes)
print(b.nbytes)



