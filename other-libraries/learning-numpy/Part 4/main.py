#------------------------------------------#
# Try to inizialize this array:            #
# 1 1 1 1 1                                #
# 1 0 0 0 1                                #
# 1 0 9 0 1                                #
# 1 0 0 0 1                                #
# 1 1 1 1 1                                #
#------------------------------------------#

import numpy as np

matrix = np.ones((5, 5), dtype="int16")
matrix[1:4, 1:4] = 0
matrix[2, 2] = 9
print(matrix)