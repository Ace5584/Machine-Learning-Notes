#------------------------------------------#
# This part of the np is about acccessing  #
# specific parts of the np array and       #
# Changing each the value of each array    #
#------------------------------------------#

import numpy as np

a = np.array([[1, 2, 3, 4, 5, 6, 7], [8, 9, 10, 11, 12, 13, 14]]) #2D Array
b = np.array([[[1,2],[3, 4]],[[5,6],[7,8]]]) #3D array
print(a)
print(b)
print(a.shape)
print(b.shape)


#####2D Array#######

#Get a spcific element [r, c]
print(a[1][5])
print(a[1][-2])

#Get a specific row
print(a[0])

#Get a specific column
print(a[:,2])

#[start_index:end_index:step_size]
print(a[0, 1:-1:2])

#Changing numbers/values in a np array
a[1, 5] = 20
print(a[1, 5])
#Changing row/column
a[:, 2] = 5
print(a)
a[0, :] = 10
print(a)

######3D Array######

#Get specific Elements
print(b[0][1][1])
print(b[0][-1][-1])

#Replace Items in the array
b[:,1,:] = [[6, 6],[9, 9]]  

